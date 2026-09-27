"""Dense self-play data: shard format, KataGo-style replay window and batch rendering.

A shard is an immutable directory `<run>/shards/NNNNNN/` holding
  episodes.json  [{moves, winner, reason, opening_plies, actor, root_values, full_search}]
  rows.json      [{game, ply, player, remaining, target, weight, legal_sha256}]
  targets.npz    offsets [rows+1], probabilities: row i's improved policy over its native
                 `Game.legal_moves()` order is probabilities[offsets[i]:offsets[i+1]] (empty slice: no policy target)
  manifest.json  schema, created_at, identity, actor, files (sha256), counts
`row.game` indexes the shard's episode list. `episode.winner` is 0/1 for finished games and -1 for capped
games; `episode.root_values` is null or one entry per ply (searched root value in [-1, 1] for the side to
move at that ply, or null). The learner derives every target from the episode (`examples`); the stored
`row.target`/`row.weight` (p(win) or null, weight) are informational and optional.
"""
from collections import namedtuple
import hashlib
import json
import multiprocessing
import queue
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import hexcrop
from hexo import Game
from klent import digest
from train import write_json

SCHEMA = 'hexo-dense-shard-v1'
FILES = ('episodes.json', 'rows.json', 'targets.npz')
Ref = namedtuple('Ref', 'shard index row episode')
Shard = namedtuple('Shard', 'episodes rows full held following')
FUTURE = (6, 20)


def player_at(ply):
    """Side to move before placement `ply`: ply 0 is player 0's single stone, then two-stone turns."""
    return ((ply+1)//2) % 2


def holdout(episode, fraction):
    """Stable validation membership: the game's moves hash below `fraction` of the 64-bit range."""
    h = hashlib.sha256(json.dumps(episode['moves']).encode()).digest()
    return int.from_bytes(h[:8], 'little') < fraction*2**64


def value_targets(players, root_values, winner, lam=.9, full=None):
    """Return (targets, weights) per ply: p(win) for the side to move at each ply.

    Terminal games (winner 0/1): target 1.0 at plies where `players[t] == winner`, else 0.0, weight 1.
    Capped games (winner -1): root values v_t in [-1, 1] from the side to move at ply t are first put in
    player 0's frame, u_t = v_t if players[t] == 0 else -v_t, then the TD(lambda) return with zero reward,
    gamma 1 and a bootstrap from the final root value is
        G_{T-1} = u_{T-1},   G_t = (1 - lam) * u_{t+1} + lam * G_{t+1}
    i.e. G_t = (1-lam) * sum_{k=1}^{T-2-t} lam^(k-1) u_{t+k} + lam^(T-2-t) u_{T-1}. Plies with a null root
    value are skipped (G_t = G_{t+1}; a trailing null bootstraps from the last known value). The target is
    (1 + s_t G_t) / 2 with s_t = +1 for player 0, -1 for player 1. A capped game with no root values at all
    gets target None and weight 0 everywhere. With `full` (per-ply bools, e.g. episode.full_search), root values
    of plies where full[t] is False are treated as null, so the chain runs and bootstraps over full searches only.
    """
    T = len(players)
    if full is not None and root_values is not None:
        root_values = [v if f else None for v, f in zip(root_values, full, strict=True)]
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
    (float array over the legal moves, or None/empty for no policy target) besides the stored fields;
    `target`/`weight` default to null/0."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f'Shard already exists: {path}')
    policies = [np.zeros(0, np.float32) if r.get('policy') is None else np.asarray(r['policy'], np.float32) for r in rows]
    for r, p in zip(rows, policies):
        if not 0 <= r['game'] < len(episodes) or (r.get('target') is not None and not 0 <= r['target'] <= 1):
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
        write_json(stage/'rows.json', [{**dict(target=None, weight=0.), **{k: r[k] for k in keys if k in r}} for r in rows])
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
    """The rows of the newest shards holding min(capacity_rows, window_size(N_full)) full-search rows.

    Only full-search rows (rows with a policy) count toward N_full and the window size; cheap-search rows ride
    along with their shard. Shards are ordered by directory name; the oldest admitted shard contributes its rows
    from its take-th last full-search row onward. `total_rows`/`rows` count every row (all shards / window),
    `total_full_rows`/`full_rows` only full-search rows. Rows of games selected by
    `holdout(episode, validation_fraction)` form the validation index and are never drawn for training.
    Episodes and rows of admitted shards stay in memory; policy vectors load per shard on first use and are
    dropped with the shard. A lock serialises methods so a rendering thread can sample while the learner
    calls `refresh()`.
    """

    def __init__(self, run_dir, capacity_rows, min_rows=100000, expand_per_row=.4, taper_exponent=.65, validation_fraction=0.):
        self.run_dir = Path(run_dir); self.capacity_rows = capacity_rows; self.validation_fraction = validation_fraction
        self.shape = dict(min_rows=min_rows, expand_per_row=expand_per_row, taper_exponent=taper_exponent)
        self.manifests = {}; self.shards = {}; self.policies = {}; self.values = {}
        self.lock = threading.RLock()
        self.refresh()

    def load(self, name):
        path = self.run_dir/'shards'/name
        episodes, rows = read_shard(path, policies=False)
        with np.load(path/'targets.npz', allow_pickle=False) as data:
            full = np.diff(data['offsets']) > 0
        where = {(r['game'], r['ply']): i for i, r in enumerate(rows)}
        return Shard(episodes, rows, full, [holdout(e, self.validation_fraction) for e in episodes],
                     [where.get((r['game'], r['ply']+1), -1) for r in rows])

    def refresh(self):
        """Rescan manifests, recompute the window and load newly admitted shards; returns window rows."""
        with self.lock:
            for path in shard_dirs(self.run_dir):
                if path.name not in self.manifests:
                    self.manifests[path.name] = manifest(path)
            names = sorted(self.manifests)
            self.total_rows = sum(self.manifests[n]['counts']['rows'] for n in names)
            self.total_full_rows = sum(self.manifests[n]['counts']['policy_rows'] for n in names)
            want = min(self.capacity_rows, window_size(self.total_full_rows, **self.shape))
            admitted = []; have = 0
            for name in reversed(names):
                if have >= want:
                    break
                take = min(self.manifests[name]['counts']['policy_rows'], want-have)
                admitted.append((name, take)); have += take
            for name in set(self.shards) - {n for n, _ in admitted}:
                del self.shards[name]; self.policies.pop(name, None)
                self.values = {k: v for k, v in self.values.items() if k[0] != name}
            for name, _ in admitted:
                if name not in self.shards:
                    self.shards[name] = self.load(name)
            self.admitted = admitted[::-1]; self.full_rows = have
            # Flat (shard, row) indices, oldest first, for uniform or recency-weighted sampling.
            self.index = []; self.validation = []
            for name, take in self.admitted:
                shard = self.shards[name]; positions = np.flatnonzero(shard.full)
                start = 0 if take >= len(positions) else int(positions[-take]) if take else len(shard.rows)
                for i in range(start, len(shard.rows)):
                    (self.validation if shard.held[shard.rows[i]['game']] else self.index).append((name, i))
            self.rows = len(self.index)+len(self.validation)
            return self.rows

    def ref(self, name, i):
        shard = self.shards[name]
        return Ref(name, i, shard.rows[i], shard.episodes[shard.rows[i]['game']])

    def sample(self, rng, n, recency=0., validation=False):
        """n Refs drawn with replacement from the training (or validation) index; the k-th oldest of
        W rows has weight ((k+1)/W)^recency."""
        with self.lock:
            index = self.validation if validation else self.index
            W = len(index)
            if not W:
                raise ValueError('Replay window is empty')
            if recency:
                w = (np.arange(1, W+1)/W)**recency
                picks = rng.choice(W, n, p=w/w.sum())
            else:
                picks = rng.integers(W, size=n)
            return [self.ref(*index[k]) for k in picks]

    def following(self, ref):
        """Ref of the same game's row at ply+1 in the same shard, or None."""
        with self.lock:
            i = self.shards[ref.shard].following[ref.index]
            return None if i < 0 else self.ref(ref.shard, i)

    def policy(self, ref):
        """The row's policy vector (empty when the ply had no full search)."""
        with self.lock:
            if ref.shard not in self.policies:
                self.policies[ref.shard] = load_policies(self.run_dir/'shards'/ref.shard, self.manifests[ref.shard]['counts']['rows'])
            offsets, probabilities = self.policies[ref.shard]
            return probabilities[offsets[ref.index]:offsets[ref.index+1]]

    def value_targets(self, ref, lam, full_only):
        """value_targets of the ref's episode (full-search root values only with `full_only`), cached while
        its shard stays admitted."""
        key = (ref.shard, ref.row['game'], lam, full_only)
        with self.lock:
            if key not in self.values:
                e = ref.episode
                self.values[key] = value_targets([player_at(t) for t in range(len(e['moves']))], e['root_values'], e['winner'],
                                                 lam, e['full_search'] if full_only else None)
            return self.values[key]


def _check(ref, s):
    if (s.player, s.remaining) != (ref.row['player'], ref.row['remaining']) \
            or hashlib.sha256(s.actions.astype(np.int64).tobytes()).hexdigest() != ref.row['legal_sha256']:
        raise ValueError(f'Replayed position disagrees with row: {ref.shard}/{ref.index}')


def render(refs, rng, hexcrop):
    """Encode each ref's position (history = episode moves before `row.ply`) under a random symmetry
    (hexcrop picks uniformly among symmetries fitting the smallest bucket). The replayed side to move
    and the hash of the native legal list must match the row."""
    samples = []
    for ref in refs:
        s = hexcrop.encode(ref.episode['moves'][:ref.row['ply']], rng=rng)
        _check(ref, s)
        samples.append(s)
    return samples


def crop_index(s, points):
    """Flat crop indices of original points [N, 2] under the sample's symmetry; -1 off the crop plane."""
    qmin, rmin, ox, oy = s.offset
    xy = points @ hexcrop.SYMMETRIES[s.symmetry] + (ox-qmin, oy-rmin)
    x, y = xy[:, 0], xy[:, 1]
    inside = (x >= 0) & (x < s.size) & (y >= 0) & (y < s.size)
    inside[inside] = s.planes[3, y[inside], x[inside]] > 0
    return np.where(inside, y*s.size+x, -1)


def target_options(settings):
    """examples() keyword arguments from a LearnerSettings."""
    return dict(lam=settings.td_lambda, bootstrap_weight=settings.bootstrap_weight, horizon=settings.short_value_horizon,
                cheap_value_weight=settings.cheap_value_weight, full_only=settings.bootstrap_full_only)


def examples(window, refs, rng, lam=.9, bootstrap_weight=1., horizon=16, cheap_value_weight=.25, full_only=True):
    """Render refs under random symmetries and derive every learner target from the episodes.

    Returns (samples, targets); each target is a dict of
      policy, policy_weight: the row's improved policy (weight 0 when empty, i.e. a cheap-search row);
      value, value_weight: value_targets(..., lam, full_search if full_only) at the ply; weight 1 for finished
        games, `bootstrap_weight` for capped games with root values, 0 otherwise, times `cheap_value_weight`
        for cheap-search rows;
      short_value, short_weight: p(win) of the side to move from the root value `horizon` plies later
        (negated when that ply's mover is the opponent); the outcome when a finished game ends within the
        horizon; weight 0 when that root value is null or a capped game ends first;
      future uint8 [2, S, S]: crop-plane cells occupied after the next 6 / 20 placements (stones already on
        the board included; truncated at the game end);
      next_cells int64 [M], next_policy float32 [M], next_weight: the next ply's policy (the ply+1 row's
        improved policy over the ply+1 native legal list: the opponent's reply after the second stone of a
        turn, the same player's second stone after the first), each cell mapped into this crop (-1 off the
        crop plane, mass dropped, the rest renormalised); weight 0 when ply+1 has no row with a policy or no
        mass lands in the crop. The network's opponent_policy head is trained on it.
    """
    samples, out = [], []
    for ref in refs:
        e, t = ref.episode, ref.row['ply']
        moves = np.asarray(e['moves'], np.int64).reshape(-1, 2); T = len(moves); me = player_at(t)
        game = Game(e['moves'][:t])
        s = hexcrop.encode_game(game, moves[:t], rng=rng)
        _check(ref, s)
        policy = window.policy(ref)
        values, weights = window.value_targets(ref, lam, full_only)
        value_weight = weights[t]*(1. if e['winner'] >= 0 else bootstrap_weight)*(1. if len(policy) else cheap_value_weight)
        u = t+horizon; roots = e['root_values']
        if u >= T:
            short = (float(me == e['winner']), 1.) if e['winner'] >= 0 else (.5, 0.)
        elif roots is not None and roots[u] is not None:
            v = roots[u] if player_at(u) == me else -roots[u]
            short = ((1+v)/2, 1.)
        else:
            short = (.5, 0.)
        future = np.zeros((2, s.size, s.size), np.uint8)
        occupied = crop_index(s, moves[:min(T, t+max(FUTURE))])
        for k, h in enumerate(FUTURE):
            cells = occupied[:min(T, t+h)]
            future[k].reshape(-1)[cells[cells >= 0]] = 1
        nref = window.following(ref); following = (np.zeros(0, np.int64), np.zeros(0, np.float32), 0.)
        if nref is not None and len(p := window.policy(nref)):
            game.play(*e['moves'][t])
            actions = hexcrop.legal_array(game, moves[:t+1])
            if len(actions) != len(p) or hashlib.sha256(actions.tobytes()).hexdigest() != nref.row['legal_sha256']:
                raise ValueError(f'Next-ply legal list disagrees with row: {nref.shard}/{nref.index}')
            cells = crop_index(s, actions); p = np.where(cells >= 0, p, 0).astype(np.float32)
            if p.sum() > 0:
                following = (cells, p/p.sum(), 1.)
        game.close()
        samples.append(s)
        out.append(dict(policy=policy, policy_weight=float(len(policy) > 0),
                        value=.5 if values[t] is None else values[t], value_weight=value_weight,
                        short_value=short[0], short_weight=short[1], future=future,
                        next_cells=following[0], next_policy=following[1], next_weight=following[2]))
    return samples, out


def collate(samples, targets):
    """Group by crop size into {S: batch} of torch tensors:
      planes uint8 [B,8,S,S]; cells int64 [B,N] (flat crop index, -1 for far cells and padding);
      mask bool [B,N] (True for the first counts[b] entries, far cells included); counts int64 [B];
      policy float32 [sum counts] in the order of cells[mask], row b at policy[offsets[b]:offsets[b+1]]
        (zeros where policy_weight is 0; far cells keep their target mass); offsets int64 [B+1];
      future uint8 [B,2,S,S]; next_cells int64 [B,M] (-1 off the crop and on padding),
        next_counts int64 [B], next_policy float32 [B,M] (zero beyond counts);
      policy_weight, value, value_weight, short_value, short_weight, next_weight, future_weight (ones)
        float32 [B]; player, remaining int64 [B].
    """
    groups = {}
    for s, t in zip(samples, targets):
        groups.setdefault(s.size, []).append((s, t))
    out = {}
    for size, items in sorted(groups.items()):
        counts = np.array([len(s.cells) for s, _ in items], np.int64)
        cells = np.full((len(items), counts.max()), -1, np.int64)
        policy = []
        for b, (s, t) in enumerate(items):
            cells[b, :counts[b]] = s.cells
            if t['policy_weight'] and len(t['policy']) != counts[b]:
                raise ValueError('Policy target length disagrees with legal moves')
            policy.append(t['policy'] if t['policy_weight'] else np.zeros(counts[b], np.float32))
        next_counts = np.array([len(t['next_cells']) for _, t in items], np.int64)
        next_cells = np.full((len(items), max(1, next_counts.max())), -1, np.int64)
        next_policy = np.zeros(next_cells.shape, np.float32)
        for b, (_, t) in enumerate(items):
            next_cells[b, :next_counts[b]] = t['next_cells']
            next_policy[b, :next_counts[b]] = t['next_policy']
        column = lambda key: torch.tensor([t[key] for _, t in items], dtype=torch.float32)
        out[size] = dict(
            planes=torch.from_numpy(np.stack([s.planes for s, _ in items])),
            cells=torch.from_numpy(cells), mask=torch.from_numpy(np.arange(cells.shape[1]) < counts[:, None]),
            counts=torch.from_numpy(counts), offsets=torch.from_numpy(np.concatenate([[0], np.cumsum(counts)])),
            policy=torch.from_numpy(np.concatenate(policy).astype(np.float32)),
            future=torch.from_numpy(np.stack([t['future'] for _, t in items])),
            next_cells=torch.from_numpy(next_cells), next_counts=torch.from_numpy(next_counts),
            next_policy=torch.from_numpy(next_policy),
            **{k: column(k) for k in ('policy_weight', 'value', 'value_weight', 'short_value', 'short_weight', 'next_weight')},
            future_weight=torch.ones(len(items)),
            player=torch.tensor([int(s.player) for s, _ in items]), remaining=torch.tensor([int(s.remaining) for s, _ in items]))
    return out


def batches(window, rng, batch_size, settings, validation=False):
    """Endless generator of collated {S: batch} dicts covering `batch_size` positions in total.
    `settings()` returns the current LearnerSettings, read per batch (recency and target_options)."""
    while True:
        s = settings()
        refs = window.sample(rng, batch_size, s.recency, validation)
        yield collate(*examples(window, refs, rng, **target_options(s)))


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


def _render_worker(run, settings, seed, output):
    """Worker process body: put numpy-packed `batches` from a private ReplayWindow (refreshed every 30 s)."""
    try:
        window = ReplayWindow(run, settings.window_capacity, settings.window_min_rows, settings.window_expand_per_row,
                              settings.window_taper, settings.validation_fraction)
        rng = np.random.default_rng(seed); refreshed = time.time()
        for batch in batches(window, rng, settings.batch, lambda: settings):
            output.put({size: {k: v.numpy() for k, v in b.items()} for size, b in batch.items()})
            if time.time()-refreshed > 30:
                window.refresh(); refreshed = time.time()
    except BaseException:
        output.put(RuntimeError('Render worker failed:\n'+traceback.format_exc()))


class Renderers:
    """`batches` of the run rendered by `workers` spawned processes, each with its own ReplayWindow, so
    rendering never holds the trainer's GIL. Settings (a LearnerSettings) are fixed per pool: close() it and
    start another to change them. Iterate to consume {S: batch of torch tensors}; a worker's exception or
    death is raised in the consumer."""

    def __init__(self, run, settings, seed, workers=2, depth=3):
        context = multiprocessing.get_context('spawn')
        self.queue = context.Queue(depth*workers)
        self.processes = [context.Process(target=_render_worker, args=(str(run), settings, [*seed, i], self.queue), daemon=True)
                          for i in range(workers)]
        for process in self.processes:
            process.start()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self.queue.get(timeout=5)
            except queue.Empty:
                if not all(p.is_alive() for p in self.processes):
                    raise RuntimeError('A render worker exited')
                continue
            if isinstance(item, BaseException):
                raise item
            return {size: {k: torch.from_numpy(v) for k, v in b.items()} for size, b in item.items()}

    def close(self):
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            process.join()
        self.queue.cancel_join_thread()
