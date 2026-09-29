"""Dense self-play data: shard format, KataGo-style replay window and batch rendering (run layout: dense_config).

A shard is an immutable directory `<run>/shards/<name>/` (all digits; converted corpora use six, actors use
millisecond time plus pid) holding
  episodes.json  [{moves, winner, reason, opening_plies, actor, root_values, full_search, actors?, opponent?,
                  trained_side?, origin?, restart?}]
  rows.json      [{game, ply, player, remaining, target, weight, legal_sha256, proven?, proof_turns?, solver_nodes?,
                  solver_budget?, proof_action?}]
  targets.npz    offsets [rows+1], probabilities: row i's improved policy over its native
                 `Game.legal_moves()` order is probabilities[offsets[i]:offsets[i+1]] (empty slice: no policy target)
  manifest.json  schema, created_at, origin, identity, actor, files (sha256), counts (opponent_rows,
                 restart_games and forced_plies may be absent: 0)
  proofs.jsonl   optional sidecar written later by the proof pass (dense_solve; not in `files`): one line per proven
                 window {game, mover, plies, ...}; rows of the listed plies are proven wins for their side to move.
                 {kind: 'deblunder', game, first_ply, owner} records soften earlier losing-owner targets only
                 with --deblunder-weight > 0; they never add exact labels.
`origin` is 'converted' (dense_bootstrap) or 'actor' (dense_selfplay); `origin()` infers it for older manifests.
`row.game` indexes the shard's episode list. `episode.winner` is 0/1 for finished games ('six-in-a-row') and -1
for capped games ('cap' at the ply limit, 'span' when a searched position does not fit the largest crop);
`episode.root_values` is null or one entry per ply (searched root value in [-1, 1] for the side to
move at that ply, or null). The learner derives every target from the episode (`examples`); the stored
`row.target`/`row.weight` (p(win) or null, weight) are informational and optional. Rows of games played with the
solver (episode `solver`, dense_solver.record) carry `proven` (+1 / -1: the side to move wins / loses by a verified
proof, 0: unproven), `proof_turns` (attacker turns of that proof, 0 when unproven), `solver_nodes` (solver work
spent on the ply's search) and `solver_budget` (the node budgets granted to its queries); absent means 0. The
manifest counts the proven rows as `proven_rows`. Readers (ReplayWindow, ValidationSets) also set proven = +1 on the
rows a sidecar lists (proof_labels) once it appears. A game ended at a proof (actor adjudicate_proven) has reason
'proven', its winner and `adjudicated` {ply, winner, line_plies: placements of the certificate's forced line from
there}; rows of that line played without search carry `line` True and no policy. The manifest counts
`proven_games`, `line_rows` and `adjudicated_plies`.
Winning rows may carry `proof_action`, the certificate's remaining placements [[q,r], ...] at that row.
Sidecars carry the same field as a mapping from ply strings to placements, applied by both readers. Missing
actions preserve legacy training. `proof_policy_weight` mixes these witnesses into the search policy at learning
time; the stored search distribution is unchanged.
Actor episodes record `origin` ('selfplay' or 'restart'; absent: 'selfplay'). A restart game (dense_selfplay.Restarts)
starts from a buffer position: its first `restart.ply` moves are the source game's, replayed without search, so it
has rows only from that ply on (null root values and full_search False before it); `restart` names the source
{shard, game, ply, kind, regret, plies_to_proof}. The manifest counts them as `restart_games` and their
replayed plies without rows as `forced_plies`.
`episode.actor` is the sha256 of the evaluator being trained. Actor shards also record `actors` {"0": sha, "1": sha}
per colour, `opponent` (null for self-play, else the checkpoint id of a frozen historical opponent) and
`trained_side` (null for self-play, else the colour the trained evaluator played). Every ply (from the restart ply
of a restart game) keeps a row so ply indexing stays contiguous, but a ply of the opponent's colour (`trained`
False) has no policy, a null root value and full_search False; it never enters the replay window and does not count
toward `total_rows`.
"""
from collections import Counter, OrderedDict, namedtuple
import hashlib
import json
import math
import zlib
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import tempfile
import time
import traceback
import types

import numpy as np

import hexcrop
from train import digest, write_json

SCHEMA = 'hexo-dense-shard-v1'
FILES = ('episodes.json', 'rows.json', 'targets.npz')
SIDECAR = 'proofs.jsonl'
Ref = namedtuple('Ref', 'shard index row episode')
Shard = namedtuple('Shard', 'game ply player remaining proven proof_action legal offsets following start moves roots searched has_roots '
                             'has_search winner side held')
FUTURE = (6, 20)
ORIGINS = ('converted', 'actor')
SOURCES = ('converted', 'fresh', 'newest')
# Calibration map basis (calibration_features): hat functions of log2(plies remaining) centred on H_KNOTS, and the
# root value's logit with |v| clipped to V_CLIP.
H_KNOTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
V_CLIP = .99
CALIBRATION_FEATURES = 2*len(H_KNOTS)
CALIBRATION_MIN_GAMES = 200
CALIBRATION_RIDGE = 1.
CALIBRATION_ITERATIONS = 25
POLICY_ATTEMPTS = 20
STALE_STAGING_SECONDS = 600.  # a policy staging file this old belongs to a writer that died


def legal_digest(actions):
    """`legal_sha256` of a native legal list: sha256 of its int64 [N, 2] bytes."""
    return hashlib.sha256(np.ascontiguousarray(actions, np.int64).tobytes()).hexdigest()


def player_at(ply):
    """Side to move before placement `ply`: ply 0 is player 0's single stone, then two-stone turns."""
    return ((ply+1)//2) % 2


def holdout(episode, fraction):
    """Stable validation membership: the game's moves hash below `fraction` of the 64-bit range."""
    h = hashlib.sha256(json.dumps(episode['moves']).encode()).digest()
    return int.from_bytes(h[:8], 'little') < fraction*2**64


def trained(episode, ply):
    """Whether `ply` was played by the evaluator being trained: every ply unless the episode records a `trained_side`."""
    side = episode.get('trained_side')
    return side is None or player_at(ply) == side


def calibration_features(v, h):
    """Tensor-product basis [N, CALIBRATION_FEATURES] = [B_k(h), s * B_k(h)] for the degree-1 B-splines (hats) B_k
    of log2(h) centred on log2(H_KNOTS[k]) (h clamped to the knot range; the hats sum to 1) and s = log((1+v)/(1-v)),
    the logit of (1+v)/2 with |v| clipped to V_CLIP. The logit of the map is a(h) + b(h) s, with a and b piecewise
    linear in log2(h)."""
    x = np.clip(np.log2(np.maximum(np.asarray(h, np.float64), 1)), 0, len(H_KNOTS)-1)
    hats = np.maximum(0, 1-np.abs(x[:, None]-np.arange(len(H_KNOTS))))
    v = np.clip(np.asarray(v, np.float64), -V_CLIP, V_CLIP)
    return np.concatenate([hats, np.log((1+v)/(1-v))[:, None]*hats], 1)


class Calibration(namedtuple('Calibration', 'coef base')):
    """A fitted map P(side to move wins | v, h): sigmoid(calibration_features(v, h) @ coef), and `base` (the
    fitted games' win rate of the side to move) where v is unknown (NaN). Hashable: coef is a tuple."""
    __slots__ = ()

    def predict(self, v, h):
        v, h = np.asarray(v, np.float64), np.asarray(h, np.float64)
        p, known = np.full(v.shape, self.base), np.isfinite(v)
        if known.any():
            p[known] = 1/(1+np.exp(-calibration_features(v[known], h[known]) @ np.asarray(self.coef)))
        return p


def pack_calibration(calibration):
    """A Calibration (or None) as CALIBRATION_FEATURES+1 floats [base, *coef]; None packs as NaNs."""
    return [math.nan]*(CALIBRATION_FEATURES+1) if calibration is None else [calibration.base, *calibration.coef]


def unpack_calibration(values):
    values = list(values)
    return None if math.isnan(values[0]) else Calibration(tuple(values[1:]), values[0])


def carried_values(roots, full=None):
    """Per ply, the side to move's value from the latest known root value at or before that ply (negated when
    that ply's mover differs), NaN before the first. `roots` holds side-to-move root values with None or NaN
    for null; with `full`, plies where full[t] is False count as null."""
    u = np.array([math.nan if v is None else v for v in roots], np.float64)
    if full is not None:
        u[~np.asarray(full, bool)] = math.nan
    sign = 1-2*((np.arange(len(u))+1)//2 % 2)
    u *= sign
    last = np.maximum.accumulate(np.where(np.isfinite(u), np.arange(len(u)), -1))
    return np.where(last >= 0, u[np.maximum(last, 0)], math.nan)*sign


def fit_calibration_rows(v, h, z, base, ridge=CALIBRATION_RIDGE, iterations=CALIBRATION_ITERATIONS):
    """The Calibration with `base` fitted to rows (v, h, z): a logistic regression of z on calibration_features(v, h)
    over the rows with a finite v, by at most `iterations` Newton steps (stopping once no coefficient moves by 1e-8)
    from, and with an L2 penalty `ridge` toward, the map that carries no information: a(h) = logit(base) and
    b(h) = 0. Without rows with a finite v the map is that prior. Each Newton system carries an extra 1e-9 on its
    diagonal, so ridge = 0 stays solvable when the rows leave the basis rank-deficient. Deterministic."""
    v, h, z = (np.asarray(x, np.float64) for x in (v, h, z))
    known = np.isfinite(v)
    X, y = calibration_features(v[known], h[known]), z[known]
    prior = np.zeros(CALIBRATION_FEATURES); prior[:len(H_KNOTS)] = math.log(max(base, 1e-6)/max(1-base, 1e-6))
    coef = prior.copy()
    for _ in range(iterations):
        p = 1/(1+np.exp(-X @ coef))
        step = np.linalg.solve((X*(p*(1-p))[:, None]).T @ X + (ridge+1e-9)*np.eye(CALIBRATION_FEATURES), X.T @ (p-y) + ridge*(coef-prior))
        coef -= step
        if np.abs(step).max() < 1e-8:
            break
    return Calibration(tuple(coef.tolist()), base)


def fit_calibration(games, min_games=CALIBRATION_MIN_GAMES, ridge=CALIBRATION_RIDGE, iterations=CALIBRATION_ITERATIONS):
    """fit_calibration_rows (with `ridge` and `iterations`) on finished games [(roots, full or None, winner)]
    (roots as for carried_values), or None for fewer than `min_games` games. Every ply t of a game of T plies is a
    row (v_t = carried value, NaN before the first, h_t = T - t, z_t = 1 if the side to move won); base is the mean
    z_t over every ply."""
    if len(games) < min_games:
        return None
    v, h, z = [], [], []
    for roots, full, winner in games:
        T = len(roots)
        v.append(carried_values(roots, full)); h.append(T-np.arange(T)); z.append((((np.arange(T)+1)//2 % 2) == winner).astype(np.float64))
    z = np.concatenate(z)
    return fit_calibration_rows(np.concatenate(v), np.concatenate(h), z, float(z.mean()), ridge, iterations)


def value_targets(players, root_values, winner, lam=.9, full=None, outcome_lam=1., calibration=None):
    """Return (targets, weights) per ply: p(win) for the side to move at each ply.

    Root values v_t in [-1, 1] from the side to move at ply t are put in player 0's frame, u_t = v_t if
    players[t] == 0 else -v_t, and a TD(lambda) return with zero reward and gamma 1 runs backwards from the
    last ply T-1:
        G_t = (1 - l) * u_{t+1} + l * G_{t+1}
    skipping plies with a null root value (G_t = G_{t+1} when u_{t+1} is null). The target is (1 + s_t G_t) / 2
    with s_t = +1 for player 0, -1 for player 1. Every weight is 1 unless stated otherwise.
    Terminal games (winner 0/1) with a `calibration`: calibration.predict(v_t, T - t), v_t = carried_values of the
    (full-filtered) root values, the base rate where v_t is unknown. Otherwise, with outcome_lam >= 1 the target
    is 1.0 at plies where `players[t] == winner`, else 0.0. With outcome_lam < 1, l = outcome_lam and the chain
    starts from the outcome z = +1 if winner == 0 else -1, G_{T-1} = z, so
    G_t = (1-l) * sum_{k=1}^{T-1-t} l^(k-1) u_{t+k} + l^(T-1-t) z when no root value is null: the last ply's
    target is exactly the outcome, and earlier ones blend in the searched values.
    Capped games (winner -1): l = lam and the chain bootstraps from the last known root value, G_{T-1} = u_{T-1}
    (a trailing null bootstraps from the last known value); with no root values at all, target None and
    weight 0 everywhere.
    With `full` (per-ply bools, e.g. episode.full_search), root values of plies where full[t] is False are
    treated as null, so the chain runs and bootstraps over full searches only.
    """
    T = len(players)
    if full is not None and root_values is not None:
        root_values = [v if f else None for v, f in zip(root_values, full, strict=True)]
    if winner >= 0 and calibration is not None:
        return calibration.predict(carried_values(root_values or [None]*T), T-np.arange(T)).tolist(), [1.]*T
    if winner >= 0 and (outcome_lam >= 1 or root_values is None):
        return [float(p == winner) for p in players], [1.]*T
    if winner < 0 and (root_values is None or all(v is None for v in root_values)):
        return [None]*T, [0.]*T
    if len(root_values) != T:
        raise ValueError('Root values must cover every ply')
    u = [None if v is None else (v if p == 0 else -v) for p, v in zip(players, root_values)]
    if winner >= 0:
        lam, g = outcome_lam, 1. if winner == 0 else -1.
    else:
        g = next(x for x in reversed(u) if x is not None)
    G = [0.]*T
    G[T-1] = g
    for t in range(T-2, -1, -1):
        if u[t+1] is not None:
            g = (1-lam)*u[t+1] + lam*g
        G[t] = g
    return [(1 + (g if p == 0 else -g))/2 for p, g in zip(players, G)], [1.]*T


def episode_value_targets(e, lam, full_only, outcome_lam=1., calibration=None):
    """value_targets of episode `e` (full-search root values only with `full_only`)."""
    return value_targets([player_at(t) for t in range(len(e['moves']))], e['root_values'], e['winner'],
                         lam, e['full_search'] if full_only else None, outcome_lam, calibration)


def write_shard(path, identity, episodes, rows, origin='actor'):
    """Atomically publish a shard of `origin` (one of ORIGINS). `identity` must carry `actor_sha256`; each row
    carries `policy` (float array over the legal moves, or None/empty for no policy target) besides the stored
    fields; `target`/`weight` default to null/0."""
    if origin not in ORIGINS:
        raise ValueError(f'Unknown shard origin {origin!r}')
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
    keys = ('game', 'ply', 'player', 'remaining', 'target', 'weight', 'legal_sha256', 'proven', 'proof_turns', 'solver_nodes',
            'solver_budget', 'line', 'proof_action')
    counts = dict(games=len(episodes), rows=len(rows), policy_rows=sum(len(p) > 0 for p in policies),
                  opponent_rows=sum(not trained(episodes[r['game']], r['ply']) for r in rows),
                  terminal_games=sum(e['winner'] >= 0 for e in episodes), capped_games=sum(e['winner'] < 0 for e in episodes),
                  proven_rows=sum(bool(r.get('proven')) for r in rows),
                  proven_games=sum(e.get('reason') == 'proven' for e in episodes), line_rows=sum(bool(r.get('line')) for r in rows),
                  adjudicated_plies=sum(e['adjudicated']['line_plies'] for e in episodes if e.get('adjudicated')),
                  restart_games=sum(e.get('origin') == 'restart' for e in episodes),
                  forced_plies=sum(e['restart']['ply'] for e in episodes if e.get('origin') == 'restart'))
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='pending-') as temporary:
        stage = Path(temporary)/'shard'
        stage.mkdir()
        write_json(stage/'episodes.json', episodes)
        write_json(stage/'rows.json', [{**dict(target=None, weight=0.), **{k: r[k] for k in keys if k in r}} for r in rows])
        np.savez_compressed(stage/'targets.npz', offsets=np.cumsum([0]+[len(p) for p in policies]).astype(np.int64),
                            probabilities=np.concatenate(policies+[np.zeros(0, np.float32)]))
        manifest = dict(schema=SCHEMA, created_at=time.time(), origin=origin, identity=identity, actor=identity['actor_sha256'],
                        files={name: digest(stage/name) for name in FILES}, counts=counts)
        write_json(stage/'manifest.json', manifest)
        stage.rename(path)
    return manifest


def manifest(path):
    data = json.loads((Path(path)/'manifest.json').read_text())
    if data['schema'] != SCHEMA:
        raise ValueError(f'Expected a dense shard: {path}')
    return data


def origin(manifest):
    """The shard's origin: its manifest `origin`, else (manifests written before the field existed) 'converted'
    when the identity names a `source` corpus (only dense_bootstrap writes one), else 'actor'."""
    return manifest.get('origin') or ('converted' if 'source' in manifest['identity'] else 'actor')


def verify(path):
    """Return the manifest after checking every file hash."""
    data = manifest(path)
    for name in FILES:
        if digest(Path(path)/name) != data['files'][name]:
            raise ValueError(f'Shard changed: {Path(path)/name}')
    return data


def load_offsets(path, rows):
    """The shard's policy offsets after checking they are rows+1 nondecreasing values from 0."""
    with np.load(Path(path)/'targets.npz', allow_pickle=False) as data:
        offsets = data['offsets']
    if len(offsets) != rows+1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(f'Malformed policy offsets: {path}')
    return offsets


def load_policies(path, rows):
    """Return (offsets, probabilities) after checking they partition the flat vector one slice per row."""
    offsets = load_offsets(path, rows)
    with np.load(Path(path)/'targets.npz', allow_pickle=False) as data:
        probabilities = data['probabilities']
    if offsets[-1] != len(probabilities):
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


def proof_records(path):
    """The records of the shard's proof sidecar, or None while it has none."""
    try:
        text = (Path(path)/SIDECAR).read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    return [json.loads(line) for line in text.splitlines() if line]


def proof_windows(path):
    """Window records only, keeping readers of older sidecars compatible."""
    records = proof_records(path)
    return None if records is None else [r for r in records if r.get('kind') != 'deblunder']


def proof_annotations(path):
    """Exact labels and deblunder intervals by game. Each interval starts after the previous proof window."""
    records = proof_records(path)
    if records is None:
        return None, {}
    windows = [r for r in records if r.get('kind') != 'deblunder']
    labels = {(w['game'], t): w.get('proof_action', {}).get(str(t)) for w in windows for t in w['plies']}
    ranges = {}
    for r in records:
        if r.get('kind') == 'deblunder':
            first = r['first_ply']
            start = max((max(w['plies'])+1 for w in windows
                         if w['game'] == r['game'] and min(w['plies']) < first), default=0)
            ranges.setdefault(r['game'], []).append((start, first, r['owner']))
    return labels, ranges


def deblunder_row(row, winner, ranges):
    """Only the losing owner's rows strictly before its window are eligible."""
    return any(start <= row['ply'] < stop and row['player'] == owner and winner == 1-owner
               for start, stop, owner in ranges.get(row['game'], ()))


def proof_labels(path):
    """{(game, ply)} of sidecar wins; None while the shard has no sidecar."""
    labels = proof_annotations(path)[0]
    return None if labels is None else set(labels)


def label(shard, labels):
    """Set proven = +1 on the rows of `shard` (a Shard) at the (game, ply) in `labels` that record no proof."""
    if labels:
        where = {key: i for i, key in enumerate(zip(shard.game.tolist(), shard.ply.tolist()))}
        for key in labels.keys() & where.keys():
            i = where[key]
            if not shard.proven[i]:
                shard.proven[i] = 1
            if shard.proven[i] > 0 and labels[key]:
                shard.proof_action.setdefault(i, labels[key])


def shard_dirs(run_dir):
    root = Path(run_dir)/'shards'
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()) if root.exists() else []


def window_size(total, min_rows=20000, expand_per_row=.4, taper_exponent=.65):
    """KataGo shuffle window: every row up to min_rows, then min_rows*(1 + e*((N/min_rows)^t - 1)/t)."""
    if total <= min_rows:
        return total
    return int(min_rows*(1 + expand_per_row*((total/min_rows)**taper_exponent - 1)/taper_exponent))


class Rows:
    """An ordered list of (shard name, row index) pairs stored as two int32 arrays over a list of shard names."""

    def __init__(self, names, shard, row):
        self.names, self.shard, self.row = names, shard, row

    def __len__(self):
        return len(self.row)

    def __getitem__(self, k):
        return self.names[self.shard[k]], int(self.row[k])

    def __iter__(self):
        return (self[k] for k in range(len(self)))


class ReplayWindow:
    """The rows of the newest shards holding min(capacity_rows, window_size(N_full)) full-search rows.

    Only full-search rows (rows with a policy) count toward N_full and the window size; cheap-search rows ride
    along with their shard. Shards are ordered by directory name; the oldest admitted shard contributes its rows
    from its take-th last full-search row onward. `total_rows`/`rows` count every trained row (all shards /
    window; see `trained`), `total_full_rows`/`full_rows` only full-search rows, `proven_rows` the window's rows
    with a nonzero `proven` (sidecar labels included). Rows of games selected by
    `holdout(episode, validation_fraction)` form the validation index and are never drawn for training.
    `index`/`validation` are Rows, oldest first.

    Memory: each admitted shard is held as numpy arrays (Shard): per row its game, ply, player, remaining, proven,
    raw legal digest, next-ply row and policy offset (proven includes the shard's proof_labels, applied at
    load or at the first refresh after its sidecar appears); per game its ply offset, winner, trained side and
    held-out flag; per ply its move, root value (NaN for null) and full_search flag. A Ref's row dict and its
    episode dict {moves, winner, root_values, full_search, trained_side} are rebuilt on demand. Value targets are
    cached for the VALUE_CACHE most recently used (episode, target options) keys.

    Policies: on first use a shard's policy vector is written uncompressed as raw float32 to
    `policy_dir`/<shard>.f32 (default <run>/cache/policies; written under a temporary name and hard-linked into
    place, never over an existing file, so a present file is complete), and each row's slice is read from it, so
    windows of several processes on one directory share one copy in the OS page cache and hold no file open between
    reads. Every refresh deletes the directory's files of shards outside this window (a file another process is
    reading stays until a later refresh; a deleted file is rewritten on its next use), so windows sharing a directory
    must admit the same shards: one directory per learner (dense_learn.policy_dir).
    """

    VALUE_CACHE = 4096

    def __init__(self, run_dir, capacity_rows, min_rows=100000, expand_per_row=.4, taper_exponent=.65, validation_fraction=0.,
                 policy_dir=None):
        self.run_dir = Path(run_dir); self.capacity_rows = capacity_rows; self.validation_fraction = validation_fraction
        self.shape = dict(min_rows=min_rows, expand_per_row=expand_per_row, taper_exponent=taper_exponent)
        self.policy_dir = self.run_dir/'cache'/'policies' if policy_dir is None else Path(policy_dir)
        self.manifests = {}; self.shards = {}; self.values = OrderedDict()
        self.unlabelled = set(); self.deblunders = {}
        self.regret_mtime = None; self.regret_entries = {}
        self.refresh()
        self.refresh_regret()

    def load(self, name):
        path = self.run_dir/'shards'/name
        episodes, rows = read_shard(path, policies=False)
        offsets = load_offsets(path, len(rows))
        labels, self.deblunders[name] = proof_annotations(path)
        if labels is None:
            self.unlabelled.add(name)
        game = np.array([r['game'] for r in rows], np.int32); ply = np.array([r['ply'] for r in rows], np.int32)
        where = {(g, t): i for i, (g, t) in enumerate(zip(game.tolist(), ply.tolist()))}
        per_ply = lambda key: [v for e in episodes for v in (e.get(key) or [None]*len(e['moves']))]
        shard = Shard(
            game=game, ply=ply.astype(np.int16), player=np.array([r['player'] for r in rows], np.int8),
            remaining=np.array([r['remaining'] for r in rows], np.int8),
            proven=np.array([r.get('proven', 0) for r in rows], np.int8),
            proof_action={i: r['proof_action'] for i, r in enumerate(rows) if r.get('proof_action')},
            legal=np.frombuffer(bytes.fromhex(''.join(r['legal_sha256'] for r in rows)), np.uint8).reshape(-1, 32),
            offsets=offsets, following=np.array([where.get((g, t+1), -1) for g, t in zip(game.tolist(), ply.tolist())], np.int32),
            start=np.cumsum([0]+[len(e['moves']) for e in episodes]).astype(np.int32),
            moves=np.array([m for e in episodes for m in e['moves']], np.int32).reshape(-1, 2),
            roots=np.array([np.nan if v is None else v for v in per_ply('root_values')], np.float64),
            searched=np.array([bool(f) for f in per_ply('full_search')], bool),
            has_roots=np.array([e['root_values'] is not None for e in episodes], bool),
            has_search=np.array([e.get('full_search') is not None for e in episodes], bool),
            winner=np.array([e['winner'] for e in episodes], np.int8),
            side=np.array([-1 if e.get('trained_side') is None else e['trained_side'] for e in episodes], np.int8),
            held=np.array([holdout(e, self.validation_fraction) for e in episodes], bool))
        label(shard, labels)
        return shard

    def refresh(self):
        """Rescan manifests, recompute the window and load newly admitted shards; returns window rows."""
        for path in shard_dirs(self.run_dir):
            if path.name not in self.manifests:
                self.manifests[path.name] = manifest(path)
        names = sorted(self.manifests)
        self.total_rows = sum(self.manifests[n]['counts']['rows']-self.manifests[n]['counts'].get('opponent_rows', 0) for n in names)
        self.total_full_rows = sum(self.manifests[n]['counts']['policy_rows'] for n in names)
        want = min(self.capacity_rows, window_size(self.total_full_rows, **self.shape))
        admitted = []; have = 0
        for name in reversed(names):
            if have >= want:
                break
            take = min(self.manifests[name]['counts']['policy_rows'], want-have)
            admitted.append((name, take)); have += take
        for name in set(self.shards) - {n for n, _ in admitted}:
            del self.shards[name]; self.unlabelled.discard(name)
            del self.deblunders[name]
            for key in [k for k in self.values if k[0] == name]:
                del self.values[key]
        self.prune({n for n, _ in admitted})
        for name, _ in admitted:
            if name not in self.shards:
                self.shards[name] = self.load(name)
            elif name in self.unlabelled:
                labels, self.deblunders[name] = proof_annotations(self.run_dir/'shards'/name)
                if labels is not None:
                    label(self.shards[name], labels); self.unlabelled.discard(name)
        self.admitted = admitted[::-1]; self.full_rows = have; self.proven_rows = 0
        parts = ([], []), ([], [])    # (shard ids, rows) of the training and validation index
        self.starts = {}
        for k, (name, take) in enumerate(self.admitted):
            s = self.shards[name]; positions = np.flatnonzero(np.diff(s.offsets) > 0)
            start = self.starts[name] = 0 if take >= len(positions) else int(positions[-take]) if take else len(s.game)
            i = np.arange(start, len(s.game), dtype=np.int32)
            side = s.side[s.game[i]]
            i = i[(side < 0) | ((s.ply[i].astype(np.int32)+1)//2 % 2 == side)]
            self.proven_rows += int(np.count_nonzero(s.proven[i]))
            held = s.held[s.game[i]]
            for split, (ids, rows) in enumerate(parts):
                ids.append(np.full(int((held == split).sum()), k, np.int32)); rows.append(i[held == split])
        names = [name for name, _ in self.admitted]
        flat = lambda arrays: np.concatenate(arrays) if arrays else np.zeros(0, np.int32)
        self.index, self.validation = (Rows(names, flat(ids), flat(rows)) for ids, rows in parts)
        self.rows = len(self.index)+len(self.validation)
        self.regret_positions = np.array([k for k, (name, i) in enumerate(self.index)
                                          if (name, int(self.shards[name].game[i]), int(self.shards[name].ply[i]))
                                          in self.regret_entries], np.int32) if self.regret_entries else np.zeros(0, np.int32)
        self.regret_weights = np.array([self.regret_entries[(name, int(self.shards[name].game[i]),
                                                             int(self.shards[name].ply[i]))]
                                        for name, i in (self.index[k] for k in self.regret_positions)], np.float64)
        self.regret_rows = len(self.regret_positions)
        self.regret_probability_cache = {}
        self.regret_distribution_cache = {}
        return self.rows

    def refresh_regret(self):
        """Read the proof buffer only when its mtime changes, then match its entries to training rows."""
        path = self.run_dir/'restarts.json'
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = None
        except OSError:
            return
        if mtime != self.regret_mtime:
            entries = {}
            try:
                if mtime is not None:
                    for entry in json.loads(path.read_text(encoding='utf-8'))['entries']:
                        key = (str(entry['shard']), int(entry['game']), int(entry['ply']))
                        regret = float(entry['regret'])
                        if not math.isfinite(regret):
                            raise ValueError('Nonfinite regret')
                        if regret > 0:
                            entries[key] = max(entries.get(key, 0.), regret)
            except (OSError, ValueError, OverflowError, KeyError, TypeError):
                return
            self.regret_mtime = mtime
            self.set_regret(entries)

    def set_regret(self, entries):
        """Use one captured buffer snapshot for this window's priority rows."""
        if entries != self.regret_entries:
            self.regret_entries = entries
            self.refresh()

    def regret_distribution(self, recency):
        """The existing recency distribution, capped at fourfold uniform when priority sampling is enabled."""
        if recency not in self.regret_distribution_cache:
            W = len(self.index)
            if recency:
                weights = (np.arange(1, W+1)/W)**recency
                base = weights/weights.sum()
            else:
                base = np.full(W, 1/W)
            cap = 4/W
            if base.max() > cap:
                if np.count_nonzero(base)*cap <= 1:
                    order = np.arange(W-1, -1, -1) if recency > 0 else np.arange(W)
                    full = int(1/cap)
                    base = np.zeros(W)
                    base[order[:full]] = cap
                    base[order[full]] = 1-full*cap
                else:
                    low, high = 0., 1.
                    while np.minimum(base*high, cap).sum() < 1:
                        high *= 2
                    for _ in range(50):
                        mid = (low+high)/2
                        if np.minimum(base*mid, cap).sum() < 1:
                            low = mid
                        else:
                            high = mid
                    base = np.minimum(base*high, cap)
                    base /= base.sum()
            self.regret_distribution_cache[recency] = base
        return self.regret_distribution_cache[recency]

    def regret_baseline(self, recency):
        return self.regret_distribution(recency)[self.regret_positions]

    def regret_count(self, batch_size, fraction, recency=0.):
        """Number of priority draws allowed by the fourfold per-row probability cap."""
        if not fraction or not self.regret_rows:
            return 0
        W, K = len(self.index), self.regret_rows
        baseline = self.regret_baseline(recency).sum()
        limit = 1. if K == W or baseline == 1 else max(0., min(1., (4*K/W-baseline)/(1-baseline)))
        return min(batch_size, int(batch_size*min(fraction, limit)+1e-12))

    def regret_share(self, batch_size, fraction, recency=0.):
        return self.regret_count(batch_size, fraction, recency)/batch_size

    def regret_probabilities(self, share, recency=0.):
        """Regret weights normalized with the cap on total uniform-plus-priority probability."""
        key = (share, recency)
        if key in self.regret_probability_cache:
            return self.regret_probability_cache[key]
        cap = (4/len(self.index)-(1-share)*self.regret_baseline(recency))/share
        weights = self.regret_weights
        if cap.sum() <= 1:
            self.regret_probability_cache[key] = cap/cap.sum()
            return self.regret_probability_cache[key]
        low, high = 0., 1./min(weights)
        while np.minimum(weights*high, cap).sum() < 1:
            high *= 2
        for _ in range(50):
            mid = (low+high)/2
            if np.minimum(weights*mid, cap).sum() < 1:
                low = mid
            else:
                high = mid
        probabilities = np.minimum(weights*high, cap)
        self.regret_probability_cache[key] = probabilities/probabilities.sum()
        return self.regret_probability_cache[key]

    def ref(self, name, i, episode=None):
        """Ref of row i of admitted shard `name`; `episode`, when given, must be the episode dict of the row's game."""
        s = self.shards[name]; g = int(s.game[i]); a, b = int(s.start[g]), int(s.start[g+1])
        row = dict(game=g, ply=int(s.ply[i]), player=int(s.player[i]), remaining=int(s.remaining[i]),
                   proven=int(s.proven[i]), legal_sha256=s.legal[i].tobytes().hex())
        if i in s.proof_action:
            row['proof_action'] = s.proof_action[i]
        if deblunder_row(row, int(s.winner[g]), self.deblunders[name]):
            row['deblunder'] = True
        episode = episode or dict(moves=s.moves[a:b].tolist(), winner=int(s.winner[g]), trained_side=None if s.side[g] < 0 else int(s.side[g]),
                       root_values=[None if v != v else v for v in s.roots[a:b].tolist()] if s.has_roots[g] else None,
                       full_search=s.searched[a:b].tolist() if s.has_search[g] else None)
        return Ref(name, i, row, episode)

    def finished_games(self, n):
        """The newest `n` finished games outside the validation split with a row inside the window (at or after
        its shard's cutoff row), newest first, as (root values with NaN for null, full_search flags or None when the
        episode records none, winner)."""
        out = []
        for name, _ in reversed(self.admitted):
            s = self.shards[name]
            inside = np.zeros(len(s.winner), bool); inside[s.game[self.starts[name]:]] = True
            for g in reversed(np.flatnonzero(inside & (s.winner >= 0) & ~s.held).tolist()):
                if len(out) >= n:
                    return out
                a, b = int(s.start[g]), int(s.start[g+1])
                out.append((s.roots[a:b], s.searched[a:b] if s.has_search[g] else None, int(s.winner[g])))
        return out

    def sample(self, rng, n, recency=0., validation=False, regret_fraction=0.):
        """n Refs drawn with replacement from the training (or validation) index; the k-th oldest of
        W rows has weight ((k+1)/W)^recency."""
        index = self.validation if validation else self.index
        W = len(index)
        if not W:
            raise ValueError('Replay window is empty')
        if recency:
            w = (np.arange(1, W+1)/W)**recency
            p = self.regret_distribution(recency) if regret_fraction and self.regret_rows and not validation else w/w.sum()
            picks = rng.choice(W, n, p=p)
        else:
            picks = rng.integers(W, size=n)
        priority = 0 if validation else self.regret_count(n, regret_fraction, recency)
        if priority:
            share = priority/n
            picks[:priority] = rng.choice(self.regret_positions, priority, p=self.regret_probabilities(share, recency))
        return [self.ref(*index[k]) for k in picks]

    def following(self, ref):
        """Ref of the same game's row at ply+1 in the same shard, or None."""
        i = int(self.shards[ref.shard].following[ref.index])
        return None if i < 0 else self.ref(ref.shard, i, ref.episode)

    def policy(self, ref):
        """The row's policy vector read from its shard's policy file (empty float32 when the ply had no full
        search), publishing the file first when it is missing; a file another process is deleting is retried for
        up to POLICY_ATTEMPTS reads 50 ms apart. Raises ValueError when the file's size disagrees with the shard."""
        s = self.shards[ref.shard]
        a, b = int(s.offsets[ref.index]), int(s.offsets[ref.index+1])
        if b == a:
            return np.zeros(0, np.float32)
        path = self.policy_dir/f'{ref.shard}.f32'
        for attempt in range(POLICY_ATTEMPTS):
            try:
                with open(path, 'rb') as stream:
                    if os.fstat(stream.fileno()).st_size != 4*int(s.offsets[-1]):
                        raise ValueError(f'Policy file disagrees with its shard: {path}')
                    stream.seek(4*a)
                    out = np.empty(b-a, np.float32)
                    stream.readinto(out)
                    return out
            except FileNotFoundError:
                self.publish(ref.shard, path)
            except PermissionError:    # Windows: another process is deleting the file
                if attempt == POLICY_ATTEMPTS-1:
                    raise
                time.sleep(.05)
        raise FileNotFoundError(f'Policy file keeps disappearing: {path}')

    def publish(self, name, path):
        """Write the shard's policy probabilities as raw float32 under a temporary name and link it to `path`
        unless another process has published it first; an existing file is never replaced."""
        _, probabilities = load_policies(self.run_dir/'shards'/name, len(self.shards[name].game))
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        staged = self.policy_dir/f'.{name}.{os.getpid()}.tmp'
        try:
            probabilities.astype(np.float32).tofile(staged)
            os.link(staged, path)
        except FileExistsError:
            pass
        finally:
            staged.unlink(missing_ok=True)

    def prune(self, keep):
        """Delete the policy_dir files of shards not in `keep` and staging files older than STALE_STAGING_SECONDS,
        skipping files that cannot be deleted now."""
        if not self.policy_dir.exists():
            return
        stale = time.time()-STALE_STAGING_SECONDS
        for path in self.policy_dir.iterdir():
            try:
                if path.suffix == '.f32' and path.stem not in keep or path.suffix == '.tmp' and path.stat().st_mtime < stale:
                    path.unlink()
            except OSError:
                pass

    def value_targets(self, ref, lam, full_only, outcome_lam=1., calibration=None):
        """episode_value_targets of the ref's episode, cached."""
        key = (ref.shard, ref.row['game'], lam, full_only, outcome_lam, calibration)
        if key in self.values:
            self.values.move_to_end(key)
        else:
            self.values[key] = episode_value_targets(ref.episode, lam, full_only, outcome_lam, calibration)
            if len(self.values) > self.VALUE_CACHE:
                self.values.popitem(last=False)
        return self.values[key]


class ValidationSets:
    """Fixed per-source row subsets for validation, independent of the replay window.

    Sources (SOURCES): 'converted' draws from converted shards, 'fresh' from actor shards and 'newest' from the
    actor-shard episodes played by the newest actor: among the episode actors with full-search rows in any actor
    shard, the one whose first shard as manifest `actor` comes latest in name order (never-published actors rank
    oldest; ties go to the most rows). It only moves to a higher-ranked actor, so a lagging worker's shard of an
    older model cannot move it back, and a shard written right after a checkpoint switch that holds only the
    previous model's games does not leave the subset empty. `newest_checkpoint` is the identity `checkpoint`
    of the first shard published by the newest actor (None when it never published one). Each source has a 'held'
    subset of full-search rows of games selected by holdout(episode, fraction) and a 'train' subset of full-search
    rows of the other games. A subset walks
    its source's shards in name order and takes from each at most `quota` rows, in the order of a permutation
    seeded by (seed, crc32(shard name)), until it holds `limit` rows. Shards are immutable and named in
    creation order, so a subset only grows, by rows of newer shards, until it is full, and a restart rebuilds
    it exactly; the 'newest' subsets start over when the newest actor changes. refresh() updates them.
    `subsets[source, split]` lists Refs; policy, value_targets and following serve examples() like
    ReplayWindow. Retained between refreshes: the chosen rows with their next-ply rows and episodes, the names
    of the shards each subset has consumed, and per scanned shard its full-search row count per episode actor
    (`actors`); a shard's row candidates live for one refresh, so a new newest actor rescans the shards it
    played. Each refresh sets proven = +1 on chosen rows (and successors) that their shard's proof sidecar lists
    (proof_labels), keeping the labels of the shards the chosen rows come from.
    """

    def __init__(self, run_dir, fraction, seed, limit, quota):
        self.run_dir, self.fraction, self.seed, self.limit, self.quota = Path(run_dir), fraction, seed, limit, quota
        self.manifests = {}; self.actors = {}; self.entries = {}; self.following_index = {}; self.labels = {}
        self.deblunders = {}
        self.subsets = {(source, split): [] for source in SOURCES for split in ('held', 'train')}
        self.picks = {key: [] for key in self.subsets}; self.walked = {key: set() for key in self.subsets}
        self.newest = self.newest_checkpoint = None

    def scan(self, name, scanned):
        """A shard's full-search rows as permuted (index, held, actor) candidates, kept in `scanned` (one
        refresh); records the shard's full-search rows per episode actor in `actors`."""
        if name not in scanned:
            episodes, rows = read_shard(self.run_dir/'shards'/name)
            full = [i for i, r in enumerate(rows) if len(r['policy'])]
            held = [holdout(e, self.fraction) for e in episodes]
            order = np.random.default_rng([self.seed, zlib.crc32(name.encode())]).permutation(len(full))
            scanned[name] = [(i, held[rows[i]['game']], episodes[rows[i]['game']]['actor']) for i in (full[k] for k in order)]
            self.actors[name] = Counter(a for _, _, a in scanned[name])
        return scanned[name]

    def refresh(self):
        """Rescan shard manifests and extend (for a new newest actor, rebuild) every subset."""
        for path in shard_dirs(self.run_dir):
            if path.name not in self.manifests:
                self.manifests[path.name] = manifest(path)
        names = sorted(self.manifests)
        actors = [n for n in names if origin(self.manifests[n]) == 'actor']
        published = {}
        for n in actors:
            published.setdefault(self.manifests[n]['actor'], (len(published), self.manifests[n]['identity'].get('checkpoint')))
        scanned, previous = {}, self.newest
        rows = Counter()
        for name in actors:
            if name not in self.actors:
                self.scan(name, scanned)
            rows += self.actors[name]
        rank = lambda a: (published.get(a, (-1,))[0], rows[a])
        if rows and (self.newest is None or rank(max(rows, key=rank))[0] > rank(self.newest)[0]):
            self.newest = max(rows, key=rank)
            self.newest_checkpoint = published.get(self.newest, (None, None))[1]
        if self.newest != previous:
            for split in ('held', 'train'):
                self.picks['newest', split], self.walked['newest', split] = [], set()
        played = lambda n: self.actors[n][self.newest] > 0 if n in self.actors else \
            self.newest in self.manifests[n]['identity'].get('actors', [self.manifests[n]['actor']])
        shards = dict(converted=[n for n in names if origin(self.manifests[n]) == 'converted'], fresh=actors,
                      newest=[n for n in actors if played(n)])
        for (source, split), chosen in self.picks.items():
            walked = self.walked[source, split]
            for name in shards[source]:
                if len(chosen) >= self.limit:
                    break
                if name in walked:
                    continue
                rows = [i for i, h, a in self.scan(name, scanned) if h == (split == 'held') and (source != 'newest' or a == self.newest)]
                chosen += [(name, i) for i in rows[:min(self.quota, self.limit-len(chosen))]]
                walked.add(name)
        entries, following, missing = {}, {}, {}
        for name, i in {key for chosen in self.picks.values() for key in chosen}:
            if (name, i) in self.following_index:    # cached as a chosen row, not only as a successor
                following[name, i] = j = self.following_index[name, i]
                entries.update({(name, k): self.entries[name, k] for k in (i, j) if k is not None})
            else:
                missing.setdefault(name, []).append(i)
        for name, indices in missing.items():
            episodes, rows = read_shard(self.run_dir/'shards'/name)
            where = {(r['game'], r['ply']): k for k, r in enumerate(rows)}
            for i in indices:
                j = where.get((rows[i]['game'], rows[i]['ply']+1))
                following[name, i] = j if j is not None and len(rows[j]['policy']) else None
                for k in (i, following[name, i]):
                    if k is not None:    # copies, so the shard's policy array can be freed
                        row = {key: v for key, v in rows[k].items() if key != 'policy'}
                        entries[name, k] = (row, episodes[row['game']], rows[k]['policy'].copy())
        self.entries, self.following_index = entries, following
        names = {n for n, _ in entries}
        for name in names:
            if self.labels.get(name) is None:
                self.labels[name], self.deblunders[name] = proof_annotations(self.run_dir/'shards'/name)
        self.labels = {n: self.labels[n] for n in names}
        self.deblunders = {n: self.deblunders[n] for n in names}
        for (name, _), (row, episode, _) in entries.items():
            labels = self.labels[name] or {}
            key = row['game'], row['ply']
            if key in labels:
                if not row.get('proven'):
                    row['proven'] = 1
                if row['proven'] > 0 and labels[key]:
                    row.setdefault('proof_action', labels[key])
            if deblunder_row(row, episode['winner'], self.deblunders[name]):
                row['deblunder'] = True
        self.subsets = {key: [self.ref(*k) for k in chosen] for key, chosen in self.picks.items()}

    def ref(self, name, i):
        row, episode, _ = self.entries[name, i]
        return Ref(name, i, row, episode)

    def policy(self, ref):
        return self.entries[ref.shard, ref.index][2]

    def following(self, ref):
        """Ref of the same game's next-ply row when it has a policy, else None."""
        j = self.following_index[ref.shard, ref.index]
        return None if j is None else self.ref(ref.shard, j)

    def value_targets(self, ref, lam, full_only, outcome_lam=1., calibration=None):
        return episode_value_targets(ref.episode, lam, full_only, outcome_lam, calibration)


def crop_index(s, points):
    """Flat crop indices of original points [N, 2] under the sample's symmetry; -1 off the crop plane."""
    qmin, rmin, ox, oy = s.offset
    xy = points @ hexcrop.SYMMETRIES[s.symmetry] + (ox-qmin, oy-rmin)
    x, y = xy[:, 0], xy[:, 1]
    inside = (x >= 0) & (x < s.size) & (y >= 0) & (y < s.size)
    inside[inside] = s.planes[3, y[inside], x[inside]] > 0
    return np.where(inside, y*s.size+x, -1)


def target_options(settings, calibration=None):
    """examples() keyword arguments from a LearnerSettings and the current Calibration: finished games get
    outcome_lambda only with value_target 'td' and `calibration` only with 'calibrated' (hard outcomes while
    it is None)."""
    return dict(lam=settings.td_lambda, bootstrap_weight=settings.bootstrap_weight, horizon=settings.short_value_horizon,
                cheap_value_weight=settings.cheap_value_weight, full_only=settings.bootstrap_full_only,
                outcome_lam=settings.outcome_lambda if settings.value_target == 'td' else 1.,
                calibration=calibration if settings.value_target == 'calibrated' else None,
                proven_weight=settings.proven_value_weight, proof_policy_weight=settings.proof_policy_weight,
                deblunder_weight=settings.deblunder_weight, future_target=settings.future_target)


def examples(window, refs, rng, lam=.9, bootstrap_weight=1., horizon=16, cheap_value_weight=.25, full_only=False,
             outcome_lam=1., calibration=None, proven_weight=2., deblunder_weight=0., proof_policy_weight=0., future_target='legacy'):
    """Render refs under random symmetries and derive every learner target from the episodes.

    Positions are encoded from the move prefix without replaying it (hexcrop.Position); the side to move and the
    legal list must match each row. Returns (samples, targets); each target is a
    dict of
      policy, policy_weight: the row's improved policy (weight 0 when empty). With proof_policy_weight > 0,
        a proven win carrying proof_action mixes (search + weight * proof)/(1 + weight), where proof is uniform
        over the certificate's remaining placements. Without search, use proof with loss weight equal to
        proof_policy_weight. Losing rows and rows without a witness keep their original policy;
      value, value_weight: value_targets(..., lam, full_search if full_only, outcome_lam, calibration) at the ply;
        weight 1 for finished games, `bootstrap_weight` for capped games with root values, 0 otherwise, times
        `cheap_value_weight` for cheap-search rows; a row with a nonzero `proven` instead gets the proven value
        (1. for +1, 0. for -1) with weight `proven_weight`;
      outcome, outcome_weight: 1. when the side to move won a finished game, else 0.; weight value_weight for
        finished games, 0 for capped games (outcome .5) and for rows with an exact label (a nonzero `proven`,
        forced-line rows included), whose value target is the proven result;
      exact: 1. for a row with an exact label, else 0.;
      With deblunder_weight > 0, eligible losing-owner rows use w*1 + (1-w)*outcome for value and the
        extra outcome loss, overriding calibration/TD at those rows. Exact labels always take precedence.
        outcome stays original; outcome_target and deblundered are added only when enabled.
      short_value, short_weight: p(win) of the side to move from the root value `horizon` plies later
        (negated when that ply's mover is the opponent); the outcome when a finished game ends within the
        horizon; weight 0 when that root value is null or a capped game ends first;
      future uint8 [2, S, S]: crop-plane cells occupied after the next 6 / 20 placements (stones already on
        the board included; truncated at the game end). With future_target='masked', uint8 [S,S] classes
        0 empty, 1 own, 2 opponent after 20 placements, relative to this row's mover. Only future placements
        are rendered; the loss excludes cells occupied now. A capped game needs the full horizon;
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
        s = hexcrop.encode_game(hexcrop.Position(moves[:t]), moves[:t], rng=rng)
        if (s.player, s.remaining) != (ref.row['player'], ref.row['remaining']) or legal_digest(s.actions) != ref.row['legal_sha256']:
            raise ValueError(f'Replayed position disagrees with row: {ref.shard}/{ref.index}')
        policy = window.policy(ref)
        values, weights = window.value_targets(ref, lam, full_only, outcome_lam, calibration)
        value_weight = weights[t]*(1. if e['winner'] >= 0 else bootstrap_weight)*(1. if len(policy) else cheap_value_weight)
        proven = ref.row.get('proven', 0)
        policy_weight = float(len(policy) > 0)
        if proof_policy_weight > 0 and proven > 0 and ref.row.get('proof_action'):
            action = np.asarray(ref.row['proof_action'], np.int64).reshape(-1, 2)
            matches = (s.actions[:, None, :] == action[None, :, :]).all(2)
            if not matches.any(0).all():
                raise ValueError(f'Proof action is not legal: {ref.shard}/{ref.index}')
            proof_policy = matches.any(1).astype(np.float32)
            proof_policy /= proof_policy.sum()
            if len(policy):
                policy = (policy + proof_policy_weight*proof_policy)/(1+proof_policy_weight)
            else:
                policy, policy_weight = proof_policy, proof_policy_weight
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
        known = np.zeros(2, np.float32)
        for k, h in enumerate(FUTURE):
            cells = occupied[:min(T, t+h)]
            future[k].reshape(-1)[cells[cells >= 0]] = 1
            known[k] = e['winner'] >= 0 or t+h <= T
        if future_target == 'masked':
            future = np.zeros((s.size, s.size), np.uint8)
            for u in range(t, min(T, t+20)):
                cell = occupied[u]
                if cell >= 0:
                    future.reshape(-1)[cell] = 1 if player_at(u) == me else 2
            known = known[1:]
        nref = window.following(ref); following = (np.zeros(0, np.int64), np.zeros(0, np.float32), 0.)
        if nref is not None and len(p := window.policy(nref)):
            actions = hexcrop.legal_array(hexcrop.Position(moves[:t+1]), moves[:t+1])
            if len(actions) != len(p) or legal_digest(actions) != nref.row['legal_sha256']:
                raise ValueError(f'Next-ply legal list disagrees with row: {nref.shard}/{nref.index}')
            cells = crop_index(s, actions); p = np.where(cells >= 0, p, 0).astype(np.float32)
            if p.sum() > 0:
                following = (cells, p/p.sum(), 1.)
        samples.append(s)
        out.append(dict(policy=policy, policy_weight=policy_weight,
                        value=float(proven > 0) if proven else .5 if values[t] is None else values[t],
                        value_weight=proven_weight if proven else value_weight,
                        outcome=float(me == e['winner']) if e['winner'] >= 0 else .5,
                        outcome_weight=value_weight if e['winner'] >= 0 and not proven else 0., exact=float(proven != 0),
                        short_value=short[0], short_weight=short[1], future=future, future_weight=known,
                        next_cells=following[0], next_policy=following[1], next_weight=following[2]))
        if deblunder_weight:
            target = out[-1]
            changed = bool(ref.row.get('deblunder') and not proven)
            target['deblundered'] = float(changed)
            target['outcome_target'] = target['outcome']
            if changed:
                target['value'] = target['outcome_target'] = deblunder_weight + (1-deblunder_weight)*target['outcome']
    return samples, out


def collate_arrays(samples, targets):
    """Group by crop size into {S: batch} of numpy arrays:
      planes uint8 [B,8,S,S]; cells int64 [B,N] (flat crop index, -1 for far cells and padding);
      mask bool [B,N] (True for the first counts[b] entries, far cells included); counts int64 [B];
      policy float32 [sum counts] in the order of cells[mask], row b at policy[offsets[b]:offsets[b+1]]
        (zeros where policy_weight is 0; far cells keep their target mass); offsets int64 [B+1];
      future uint8 [B,2,S,S] (legacy) or [B,S,S] (masked); next_cells int64 [B,M] (-1 off the crop and on padding),
        next_counts int64 [B], next_policy float32 [B,M] (zero beyond counts);
      policy_weight, value, value_weight, outcome, outcome_weight, exact, short_value, short_weight, next_weight;
      future_weight [B,2] per horizon (legacy) or [B,1] (masked)
      (0 where a capped game ends before the horizon)
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
        column = lambda key: np.array([t[key] for _, t in items], np.float32)
        out[size] = dict(
            planes=np.stack([s.planes for s, _ in items]),
            cells=cells, mask=np.arange(cells.shape[1]) < counts[:, None],
            counts=counts, offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            policy=np.concatenate(policy).astype(np.float32),
            future=np.stack([t['future'] for _, t in items]),
            next_cells=next_cells, next_counts=next_counts, next_policy=next_policy,
            **{k: column(k) for k in ('policy_weight', 'value', 'value_weight', 'outcome', 'outcome_weight', 'exact',
                                      'short_value', 'short_weight', 'next_weight')},
            future_weight=np.stack([t['future_weight'] for _, t in items]),
            player=np.array([int(s.player) for s, _ in items], np.int64),
            remaining=np.array([int(s.remaining) for s, _ in items], np.int64))
        if 'outcome_target' in items[0][1]:
            out[size].update({k: column(k) for k in ('outcome_target', 'deblundered')})
    return out


def tensors(batch):
    """A collate_arrays batch as torch tensors sharing its memory. Torch is imported here only, so render workers
    (which import this module) never load it."""
    import torch
    return {size: {k: torch.from_numpy(v) for k, v in b.items()} for size, b in batch.items()}


def collate(samples, targets):
    """collate_arrays as torch tensors."""
    return tensors(collate_arrays(samples, targets))


def batches(window, rng, batch_size, settings, validation=False, calibration=lambda: None):
    """Endless generator of collate_arrays {S: batch} dicts covering `batch_size` positions in total.
    `settings()` and `calibration()` return the current LearnerSettings and Calibration, read per batch
    (recency and target_options)."""
    while True:
        s = settings()
        refs = window.sample(rng, batch_size, s.recency, validation, s.regret_fraction)
        yield collate_arrays(*examples(window, refs, rng, **target_options(s, calibration())))


def start_hidden(processes):
    """Start spawn-context `processes` with the parent's __main__ hidden, so each child imports only the modules
    its target and arguments need instead of re-importing the parent's main script (and torch with it)."""
    main = sys.modules['__main__']
    sys.modules['__main__'] = types.ModuleType('__main__')
    try:
        for process in processes:
            process.start()
    finally:
        sys.modules['__main__'] = main


def _render_worker(run, settings, seed, output, calibration, policy_dir, regret_updates, initial_regret):
    """Worker process body: put `batches` from a private ReplayWindow on `policy_dir` (refreshed every 30 s), with
    the Calibration packed in the shared array `calibration`."""
    try:
        window = ReplayWindow(run, settings.window_capacity, settings.window_min_rows, settings.window_expand_per_row,
                              settings.window_taper, settings.validation_fraction, policy_dir)
        if initial_regret is not None:
            window.set_regret(initial_regret)
        rng = np.random.default_rng(seed); refreshed = time.time()
        while not window.index:
            time.sleep(5); window.refresh(); refreshed = time.time()
        for batch in batches(window, rng, settings.batch, lambda: settings, calibration=lambda: unpack_calibration(calibration[:])):
            output.put(batch)
            latest_regret = None
            try:
                while True:
                    latest_regret = regret_updates.get_nowait()
            except queue.Empty:
                pass
            if latest_regret is not None:
                window.set_regret(latest_regret)
            if time.time()-refreshed > 30:
                window.refresh(); refreshed = time.time()
    except BaseException:
        output.put(RuntimeError('Render worker failed:\n'+traceback.format_exc()))


class Renderers:
    """`batches` of the run rendered by `workers` spawned processes (started by start_hidden, so they never load
    torch), each with its own ReplayWindow on `policy_dir` (None: the run's default), so rendering never holds the
    trainer's GIL; at most depth * workers rendered batches wait in the queue. Worker i draws from its own generator
    seeded [*seed, i]. Settings (a LearnerSettings) are fixed per pool: close() it and start another to change them;
    set_calibration() replaces the Calibration of batches rendered from then on. Iterate to consume {S: batch of
    torch tensors}; a worker's exception or death is raised in the consumer."""

    def __init__(self, run, settings, seed, workers=2, depth=3, calibration=None, policy_dir=None, regret_entries=None):
        context = multiprocessing.get_context('spawn')
        self.queue = context.Queue(depth*workers)
        self.calibration = context.Array('d', CALIBRATION_FEATURES+1)
        self.regret_updates = [context.Queue() for _ in range(workers)]
        self.set_calibration(calibration)
        self.processes = [context.Process(target=_render_worker, daemon=True,
                                          args=(str(run), settings, [*seed, i], self.queue, self.calibration, policy_dir,
                                                self.regret_updates[i], regret_entries))
                          for i in range(workers)]
        start_hidden(self.processes)

    def set_calibration(self, calibration):
        self.calibration[:] = pack_calibration(calibration)

    def refresh_regret(self, entries):
        for updates in self.regret_updates:
            updates.put(entries)

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
            return tensors(item)

    def close(self):
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            process.join()
        self.queue.cancel_join_thread()
        for updates in self.regret_updates:
            updates.cancel_join_thread()
