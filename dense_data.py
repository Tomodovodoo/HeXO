"""Dense self-play data: shard format, KataGo-style replay window and batch rendering.

A shard is an immutable directory `<run>/shards/NNNNNN/` holding
  episodes.json  [{moves, winner, reason, opening_plies, actor, root_values, full_search}]
  rows.json      [{game, ply, player, remaining, target, weight, legal_sha256}]
  targets.npz    offsets [rows+1], probabilities: row i's improved policy over its native
                 `Game.legal_moves()` order is probabilities[offsets[i]:offsets[i+1]] (empty slice: no policy target)
  manifest.json  schema, created_at, identity, actor, files (sha256), counts
`row.game` indexes the shard's episode list; `row.target` is p(win) for the side to move or null,
and `row.weight` 0 masks the value loss.
"""
from collections import namedtuple
import hashlib
import json
import queue
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

from klent import digest
from train import write_json

SCHEMA = 'hexo-dense-shard-v1'
FILES = ('episodes.json', 'rows.json', 'targets.npz')
Ref = namedtuple('Ref', 'shard index row episode')


def value_targets(players, root_values, winner, lam=.9):
    """Return (targets, weights) per ply: p(win) for the side to move at each ply.

    Terminal games (winner 0/1): target 1.0 at plies where `players[t] == winner`, else 0.0, weight 1.
    Capped games (winner -1): root values v_t in [-1, 1] from the side to move at ply t are first put in
    player 0's frame, u_t = v_t if players[t] == 0 else -v_t, then the TD(lambda) return with zero reward,
    gamma 1 and a bootstrap from the final root value is
        G_{T-1} = u_{T-1},   G_t = (1 - lam) * u_{t+1} + lam * G_{t+1}
    i.e. G_t = (1-lam) * sum_{k=1}^{T-2-t} lam^(k-1) u_{t+k} + lam^(T-2-t) u_{T-1}. Plies with a null root
    value are skipped (G_t = G_{t+1}; a trailing null bootstraps from the last known value). The target is
    (1 + s_t G_t) / 2 with s_t = +1 for player 0, -1 for player 1. A capped game with no root values at all
    gets target None and weight 0 everywhere.
    """
    T = len(players)
    if winner >= 0:
        return [float(p == winner) for p in players], [1.]*T
    if root_values is None or all(v is None for v in root_values):
        return [None]*T, [0.]*T
    if len(root_values) != T:
        raise ValueError('Root values must cover every ply')
    u = [None if v is None else (v if p == 0 else -v) for p, v in zip(players, root_values)]
    G = [0.]*T
    g = next(x for x in reversed(u) if x is not None)
    G[T-1] = g
    for t in range(T-2, -1, -1):
        if u[t+1] is not None:
            g = (1-lam)*u[t+1] + lam*g
        G[t] = g
    return [(1 + (g if p == 0 else -g))/2 for p, g in zip(players, G)], [1.]*T


def write_shard(path, identity, episodes, rows):
    """Atomically publish a shard. `identity` must carry `actor_sha256`; each row carries `policy`
    (float array over the legal moves, or None/empty for no policy target) besides the stored fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f'Shard already exists: {path}')
    policies = [np.zeros(0, np.float32) if r.get('policy') is None else np.asarray(r['policy'], np.float32) for r in rows]
    for r, p in zip(rows, policies):
        if not 0 <= r['game'] < len(episodes) or (r['target'] is not None and not 0 <= r['target'] <= 1):
            raise ValueError('Malformed shard row')
        if len(p) and (not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-4)):
            raise ValueError('Invalid policy target')
    keys = ('game', 'ply', 'player', 'remaining', 'target', 'weight', 'legal_sha256')
    counts = dict(games=len(episodes), rows=len(rows), policy_rows=sum(len(p) > 0 for p in policies),
                  terminal_games=sum(e['winner'] >= 0 for e in episodes), capped_games=sum(e['winner'] < 0 for e in episodes))
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='pending-') as temporary:
        stage = Path(temporary)/'shard'
        stage.mkdir()
        write_json(stage/'episodes.json', episodes)
        write_json(stage/'rows.json', [{k: r[k] for k in keys} for r in rows])
        np.savez_compressed(stage/'targets.npz', offsets=np.cumsum([0]+[len(p) for p in policies]).astype(np.int64),
                            probabilities=np.concatenate(policies+[np.zeros(0, np.float32)]))
        manifest = dict(schema=SCHEMA, created_at=time.time(), identity=identity, actor=identity['actor_sha256'],
                        files={name: digest(stage/name) for name in FILES}, counts=counts)
        write_json(stage/'manifest.json', manifest)
        stage.rename(path)
    return manifest


def manifest(path):
    data = json.loads((Path(path)/'manifest.json').read_text())
    if data['schema'] != SCHEMA:
        raise ValueError(f'Expected a dense shard: {path}')
    return data


def verify(path):
    """Return the manifest after checking every file hash."""
    data = manifest(path)
    for name in FILES:
        if digest(Path(path)/name) != data['files'][name]:
            raise ValueError(f'Shard changed: {Path(path)/name}')
    return data


def load_policies(path, rows):
    """Return (offsets, probabilities) after checking they partition the flat vector one slice per row."""
    with np.load(Path(path)/'targets.npz', allow_pickle=False) as data:
        offsets = data['offsets']; probabilities = data['probabilities']
    if len(offsets) != rows+1 or offsets[0] != 0 or offsets[-1] != len(probabilities) or np.any(np.diff(offsets) < 0):
        raise ValueError(f'Malformed policy offsets: {path}')
    return offsets, probabilities


def read_shard(path, policies=True):
    """Verify and read a shard; with `policies` each row gets its `policy` slice (possibly empty)."""
    data = verify(path)
    episodes = json.loads((Path(path)/'episodes.json').read_text()); rows = json.loads((Path(path)/'rows.json').read_text())
    if len(episodes) != data['counts']['games'] or len(rows) != data['counts']['rows']:
        raise ValueError(f'Shard counts disagree with manifest: {path}')
    if policies:
        offsets, probabilities = load_policies(path, len(rows))
        for i, row in enumerate(rows):
            row['policy'] = probabilities[offsets[i]:offsets[i+1]]
    return episodes, rows


def shard_dirs(run_dir):
    root = Path(run_dir)/'shards'
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()) if root.exists() else []


def stats(run_dir):
    """Totals over every published shard (manifests only) plus the newest shard and its actor."""
    paths = shard_dirs(run_dir); items = [manifest(p) for p in paths]
    total = {k: sum(m['counts'][k] for m in items) for k in ('games', 'rows', 'policy_rows', 'terminal_games', 'capped_games')}
    return dict(total, shards=len(items), newest_shard=paths[-1].name if items else None,
                newest_actor=items[-1]['actor'] if items else None)


def window_size(total, min_rows=20000, expand_per_row=.4, taper_exponent=.65):
    """KataGo shuffle window: every row up to min_rows, then min_rows*(1 + e*((N/min_rows)^t - 1)/t)."""
    if total <= min_rows:
        return total
    return int(min_rows*(1 + expand_per_row*((total/min_rows)**taper_exponent - 1)/taper_exponent))


class ReplayWindow:
    """The newest min(capacity_rows, window_size(N_total)) rows, taken shard by shard from the newest shard.

    Shards are ordered by directory name. The oldest admitted shard contributes only its last rows.
    Episodes and rows of admitted shards stay in memory; policy vectors load per shard on first use and
    are dropped with the shard. A lock serialises methods so a background rendering thread can sample
    while the learner calls `refresh()`.
    """

    def __init__(self, run_dir, capacity_rows, min_rows=20000, expand_per_row=.4, taper_exponent=.65):
        self.run_dir = Path(run_dir); self.capacity_rows = capacity_rows
        self.shape = dict(min_rows=min_rows, expand_per_row=expand_per_row, taper_exponent=taper_exponent)
        self.manifests = {}; self.shards = {}; self.policies = {}
        self.lock = threading.Lock()
        self.refresh()

    def refresh(self):
        """Rescan manifests, recompute the window and load newly admitted shards; returns window rows."""
        with self.lock:
            for path in shard_dirs(self.run_dir):
                if path.name not in self.manifests:
                    self.manifests[path.name] = manifest(path)
            names = sorted(self.manifests)
            self.total_rows = sum(self.manifests[n]['counts']['rows'] for n in names)
            want = min(self.capacity_rows, window_size(self.total_rows, **self.shape))
            admitted = []; have = 0
            for name in reversed(names):
                if have >= want:
                    break
                take = min(self.manifests[name]['counts']['rows'], want-have)
                admitted.append((name, take)); have += take
            for name in set(self.shards) - {n for n, _ in admitted}:
                del self.shards[name]; self.policies.pop(name, None)
            for name, _ in admitted:
                if name not in self.shards:
                    self.shards[name] = read_shard(self.run_dir/'shards'/name, policies=False)
            self.admitted = admitted[::-1]; self.rows = have
            # Flat (shard, row) index, oldest first, for uniform or recency-weighted sampling.
            self.index = [(name, i) for name, take in self.admitted
                          for i in range(self.manifests[name]['counts']['rows']-take, self.manifests[name]['counts']['rows'])]
            return have

    def sample(self, rng, n, recency=0.):
        """n Refs drawn with replacement; the k-th oldest of W window rows has weight ((k+1)/W)^recency."""
        with self.lock:
            W = len(self.index)
            if not W:
                raise ValueError('Replay window is empty')
            if recency:
                w = (np.arange(1, W+1)/W)**recency
                picks = rng.choice(W, n, p=w/w.sum())
            else:
                picks = rng.integers(W, size=n)
            out = []
            for k in picks:
                name, i = self.index[k]; episodes, rows = self.shards[name]
                out.append(Ref(name, i, rows[i], episodes[rows[i]['game']]))
            return out

    def policy(self, ref):
        """The row's policy vector (empty when the ply had no full search)."""
        with self.lock:
            if ref.shard not in self.policies:
                self.policies[ref.shard] = load_policies(self.run_dir/'shards'/ref.shard, self.manifests[ref.shard]['counts']['rows'])
            offsets, probabilities = self.policies[ref.shard]
            return probabilities[offsets[ref.index]:offsets[ref.index+1]]


def render(refs, rng, hexcrop):
    """Encode each ref's position (history = episode moves before `row.ply`) under a random symmetry
    (hexcrop picks uniformly among symmetries fitting the smallest bucket). The replayed side to move
    and the hash of the native legal list must match the row."""
    samples = []
    for ref in refs:
        s = hexcrop.encode(ref.episode['moves'][:ref.row['ply']], rng=rng)
        if (s.player, s.remaining) != (ref.row['player'], ref.row['remaining']) \
                or hashlib.sha256(s.actions.astype(np.int64).tobytes()).hexdigest() != ref.row['legal_sha256']:
            raise ValueError(f'Replayed position disagrees with row: {ref.shard}/{ref.index}')
        samples.append(s)
    return samples


def targets(window, refs):
    """Per-ref (policy, policy_weight, value, value_weight); an empty policy slice gets policy_weight 0
    and a null value target gets value 0.5 with value_weight 0."""
    out = []
    for ref in refs:
        p = window.policy(ref); v = ref.row['target']
        out.append((p, float(len(p) > 0), .5 if v is None else v, 0. if v is None else ref.row['weight']))
    return out


def collate(samples, targets):
    """Group by crop size into {S: batch} of torch tensors:
      planes uint8 [B,8,S,S]; cells int64 [B,N] (flat crop index, -1 for far cells and padding);
      mask bool [B,N] (True for the first counts[b] entries, far cells included); counts int64 [B];
      policy float32 [sum counts] in the order of cells[mask], row b at policy[offsets[b]:offsets[b+1]]
        (zeros where policy_weight is 0; far cells keep their target mass); offsets int64 [B+1];
      policy_weight, value (p(win) of the side to move), value_weight float32 [B]; player, remaining int64 [B].
    """
    groups = {}
    for s, t in zip(samples, targets):
        groups.setdefault(s.size, []).append((s, t))
    out = {}
    for size, items in sorted(groups.items()):
        counts = np.array([len(s.cells) for s, _ in items], np.int64)
        cells = np.full((len(items), counts.max()), -1, np.int64)
        policy = []
        for b, (s, (p, pw, _, _)) in enumerate(items):
            cells[b, :counts[b]] = s.cells
            if pw and len(p) != counts[b]:
                raise ValueError('Policy target length disagrees with legal moves')
            policy.append(p if pw else np.zeros(counts[b], np.float32))
        out[size] = dict(
            planes=torch.from_numpy(np.stack([s.planes for s, _ in items])),
            cells=torch.from_numpy(cells), mask=torch.from_numpy(np.arange(cells.shape[1]) < counts[:, None]),
            counts=torch.from_numpy(counts), offsets=torch.from_numpy(np.concatenate([[0], np.cumsum(counts)])),
            policy=torch.from_numpy(np.concatenate(policy).astype(np.float32)),
            policy_weight=torch.tensor([t[1] for _, t in items], dtype=torch.float32),
            value=torch.tensor([t[2] for _, t in items], dtype=torch.float32),
            value_weight=torch.tensor([t[3] for _, t in items], dtype=torch.float32),
            player=torch.tensor([int(s.player) for s, _ in items]), remaining=torch.tensor([int(s.remaining) for s, _ in items]))
    return out


def batches(window, rng, batch_size, hexcrop, recency=0.):
    """Endless generator of collated {S: batch} dicts covering `batch_size` positions in total."""
    while True:
        refs = window.sample(rng, batch_size, recency)
        yield collate(render(refs, rng, hexcrop), targets(window, refs))


class Prefetch:
    """Drain an iterator in a daemon thread, keeping up to `depth` items ready; iterate to consume.
    An exception in the producer is re-raised in the consumer."""

    def __init__(self, iterator, depth=4):
        self.queue = queue.Queue(depth)
        self.thread = threading.Thread(target=self.run, args=(iterator,), daemon=True)
        self.thread.start()

    def run(self, iterator):
        try:
            for item in iterator:
                self.queue.put(item)
        except BaseException as error:
            self.queue.put(error)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item
