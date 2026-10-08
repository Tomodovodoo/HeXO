"""Local browser game: people and engines on either side, analysis and review from saved evaluations.

Run `python python/play.py --dense-run runs/bubble --device cpu`, then open http://127.0.0.1:8765. Engine work runs
on one background worker thread as jobs with ids; HTTP requests only read or change the session, so the page never
waits on an engine. See docs/play.md.
"""
import argparse
import contextlib
import functools
import hashlib
import heapq
import itertools
import json
import math
import os
import queue
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import formats
from engine_setup import Setups
from hexo import Game
from notation import NotationConflict, dumps, loads
from process_tree import TreeProcess
from time_control import Clock, TimeControl, duration, milliseconds
from timed_engine import saved_ids

ROOT = Path(__file__).resolve().parents[1]
# Bubble's levels are work per stone (`simulations`, the hybrid owner's completed simulations) and root query nodes,
# chosen for wall time per turn on main/185000 (docs/play.md): on a GPU from under half a second to about four.
PRESETS = dict(
    bubble=dict(lightning=dict(simulations=16, solver_nodes=1024), quick=dict(simulations=64, solver_nodes=2048),
                standard=dict(simulations=256, solver_nodes=4096), strong=dict(simulations=512, solver_nodes=8192),
                deep=dict(simulations=1024, solver_nodes=16384), dangerous=dict(simulations=65536, solver_nodes=4_000_000)),
    drip=dict(lightning=dict(ms=100), quick=dict(ms=250), standard=dict(ms=1000), strong=dict(ms=3000),
              deep=dict(ms=10000), dangerous=dict(ms=60000)),
    seal=dict(lightning=dict(ms=100), quick=dict(ms=250), standard=dict(ms=1000), strong=dict(ms=3000), deep=dict(ms=10000),
              dangerous=dict(ms=60000)),
    six=dict(lightning=dict(nodes=120), quick=dict(nodes=240), standard=dict(nodes=480), strong=dict(nodes=960),
             deep=dict(nodes=1920), dangerous=dict(nodes=2_000_000)),
    strix=dict(lightning=dict(simulations=2), quick=dict(simulations=8), standard=dict(simulations=64),
               strong=dict(simulations=128), deep=dict(simulations=512), dangerous=dict(simulations=4096)))
# Bubble on a CPU server (two torch threads, about 26 simulations a second): the same span of wall time.
CPU_PRESETS = dict(lightning=dict(simulations=4, solver_nodes=512), quick=dict(simulations=8, solver_nodes=1024),
                   standard=dict(simulations=16, solver_nodes=2048), strong=dict(simulations=32, solver_nodes=4096),
                   deep=dict(simulations=64, solver_nodes=8192), dangerous=dict(simulations=1024, solver_nodes=131072))
PRESET_NAMES = list(PRESETS['bubble'])
# Milliseconds per two-stone Bubble turn at each level of the GPU and the CPU ladder, measured on 2026-10-08 with
# network 205000 (RTX 3070 Ti, Ryzen 9 5900X, the training run sharing both): the strength slider's estimate until
# this server has timed moves of its own (`Session.note_pace`).
TURN_MS = dict(GPU=dict(lightning=250, quick=650, standard=1400, strong=2000, deep=4100),
               CPU=dict(lightning=750, quick=850, standard=1700, strong=3100, deep=6600))
HYBRID = dict(quantum=64, views=8, depth=8)  # the graph owner of every Bubble search (see `search`, docs/play.md)
PROOF_BUDGET = 1.  # share of the owner's time its proof frontier may take (ProofLoop owner_budget)
SEEDS = itertools.count(1740)  # owner seeds: repeated searches of a position draw new root samples
REFRESH_SHARE = .25  # share of a saved analysis's simulations its refresh searches again (Session.refresh)
REFRESH_PLIES = 4  # earlier placements whose saved analysis a finished analysis searches again (Session.refresh)
REFRESH_ROUNDS = 3  # further refreshes of one position while each refresh still moves its result
REFRESH_MOVE = .05  # a refresh moved its result when its value changed by more than this, or its stones changed
REVIEW_SOLVERS = 4                      # tactical workers a review queries at once
REVIEW_CHUNK = 24                       # positions per pooled review step; urgent analysis waits at most one step
SOLVER = dict(simulations=128, solver_nodes=32768, solver_ms=120_000)  # the analysis solver preset (see `prove`)
SOLVER_WORKERS = max(2, min(12, (os.cpu_count() or 4) - 4))  # its native proof workers
LIMITS = dict(simulations=0, solver_nodes=0, ms=10, nodes=1)
# The custom budgets whose two amounts exclude each other: work (Bubble's Nodes, Six's Positions) or Time in ms.
EXCLUSIVE = dict(bubble=('simulations', 'ms'), six=('nodes', 'ms'))
# The engines take budgets as 32-bit signed integers.
MAX_BUDGET = 2 ** 31 - 1
KIND_LIMITS = dict(strix=dict(simulations=1))
HEXO_SITES = {'hexo.did.science': 'https://hexo.did.science/api',
              'hexo.mineking.dev': 'https://hexo.mineking.dev/proxy/api'}
SHOWN = ('id', 'name', 'label', 'kind', 'badge', 'presets', 'checkpoints')
# Each library Six's backends need, as its Windows and its POSIX file name
SIX_LIBRARIES = dict(cuda=('cudart64_12.dll', 'libcudart.so.12'), cudnn=('cudnn64_9.dll', 'libcudnn.so.9'),
                     tensorrt=('nvinfer_10.dll', 'libnvinfer.so.10'), directml=('DirectML.dll', None),
                     cuda_build=('onnxruntime_providers_cuda.dll', 'libonnxruntime_providers_cuda.so'),
                     tensorrt_build=('onnxruntime_providers_tensorrt.dll', 'libonnxruntime_providers_tensorrt.so'))


def player_at(ply):
    """Side placing stone number `ply` (0-based): X opens with one stone, then two per turn."""
    return 0 if ply == 0 else ((ply - 1) // 2 + 1) % 2


def moved(found, before):
    """True when the evaluation `found` differs from the saved evaluation `before` in its stones or by more than
    REFRESH_MOVE in value."""
    stones = lambda record: sorted(tuple(map(int, p)) for p in record.get('moves') or [])
    return stones(found) != stones(before) or abs(float(found['value']) - float(before['value'])) > REFRESH_MOVE


def turn_starts(length):
    """Plies at which a turn starts, for a history of `length` stones."""
    return [0, *range(1, length, 2)]


def review_plies(history):
    """Positions a review evaluates: every position before a stone, plus the final position unless the game is over."""
    game = replay(history)
    try:
        return list(range(len(history) + (game.winner < 0)))
    finally:
        game.close()


def replay(history):
    """A native game with `history` played; raises ValueError on an illegal stone."""
    game = Game()
    try:
        for q, r in history:
            game.play(int(q), int(r))
    except Exception:
        game.close()
        raise
    return game


def checked_turn(history, moves):
    """`moves` cut to one legal turn from `history`: stops at a win, raises ValueError when incomplete or illegal."""
    game = replay(history)
    try:
        side, played = game.player, []
        for q, r in moves:
            game.play(int(q), int(r))
            played.append([int(q), int(r)])
            if game.winner >= 0 or game.player != side:
                return played
        raise ValueError('Engine returned an incomplete turn')
    finally:
        game.close()


def book_openings(path, mode='wide', count=None, seed=0):
    """Freeze a read-only selection of the book's canonical openings, without replacement."""
    if mode not in ('narrow', 'wide', 'all'):
        raise ValueError('Opening range must be narrow, wide or all')
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError('unique_openings must be positive')
    if type(seed) is not int:
        raise ValueError('Opening seed must be an integer')
    path = Path(path)
    raw = path.read_bytes()
    book = json.loads(raw)
    if book['schema'] != 'hexo-opening-book-v2':
        raise ValueError('Choose a v2 opening book')
    # Known-result tactical cases stay outside the ordinary comparison pool.
    nodes = {n['key']: n for n in book['nodes'] if n['status'] == 'opening'
             and (mode == 'all' or not n.get('off_policy'))}
    pool = sorted(nodes.values(), key=lambda n: n['key'])
    count = min(8, len(pool)) if count is None and mode == 'narrow' else len(pool) if count is None else count
    if not pool or count > len(pool):
        raise ValueError(f'{mode} has {len(pool)} unique openings; requested {count}')
    rng = random.Random(seed)
    if mode == 'narrow':
        if any(n.get('champion_probability') is None for n in pool):
            raise ValueError('Narrow selection needs recorded champion policy probabilities for this book')
        pool.sort(key=lambda n: (-n['champion_probability'], n['key']))
        selected = pool[:count]
        cutoff = selected[-1]['champion_probability']
    else:
        selected = rng.sample(pool, count)
        cutoff = None
    rng.shuffle(selected)
    from hexcrop import SYMMETRIES
    nodes = []
    for node in selected:
        orientation = rng.randrange(len(SYMMETRIES))
        transform = SYMMETRIES[orientation]
        moves = [[int(q*transform[0, 0]+r*transform[1, 0]), int(q*transform[0, 1]+r*transform[1, 1])]
                 for q, r in node['moves']]
        nodes.append({k: node.get(k) for k in ('key', 'champion_probability', 'scored_by', 'off_policy')} |
                     dict(moves=moves, symmetry=orientation))
    return dict(book=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), range=mode, seed=seed,
                eligible=len(pool), unique_openings=count, policy_cutoff=cutoff, refreshed_by=book.get('refreshed_by'),
                nodes=nodes)


BOOKS = ('narrow', 'wide', 'all')


def pick_opening(nodes, played, rng):
    """One of the book selection `nodes` for a game against a person, preferring lines not yet played: walking
    the tree of canonical keys (turns split at `|`) from the root, each step picks uniformly among the branches
    that still hold an opening never played (`played(key)` 0), so a branch counts as chosen only once all of its
    openings have been; when every opening has been played, one of the least played is picked uniformly."""
    tree = {}
    for node in nodes:
        branch = tree
        for turn in node['key'].split('|'):
            branch = branch.setdefault(turn, {})
        branch[None] = node

    def fresh(branch):
        return any(fresh(child) if turn is not None else not played(child['key']) for turn, child in branch.items())

    branch = tree
    while fresh(branch):
        options = [(turn, child) for turn, child in sorted(branch.items(), key=lambda item: str(item[0]))
                   if (not played(child['key']) if turn is None else fresh(child))]
        turn, child = rng.choice(options)
        if turn is None:
            return child
        branch = child
    least = min(played(node['key']) for node in nodes)
    return rng.choice([node for node in nodes if played(node['key']) == least])


class Coverage:
    """Openings started against a person, per side the person played: an append-only JSON-lines file of
    {book, key, side, at}, counted in memory."""

    def __init__(self, path=None):
        self.path, self.counts = Path(path) if path else None, {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                with contextlib.suppress(ValueError, KeyError, TypeError):
                    row = json.loads(line)
                    self.count(row['book'], row['key'], row['side'], 1)

    def count(self, book, key, side, add=0):
        found = self.counts[(book, key, side)] = self.counts.get((book, key, side), 0) + add
        return found

    def add(self, book, key, side):
        self.count(book, key, side, 1)
        if self.path:
            with open(self.path, 'a', encoding='utf-8') as out:
                out.write(json.dumps(dict(book=book, key=key, side=side,
                                          at=datetime.now(timezone.utc).isoformat(timespec='seconds'))) + '\n')


def lock_match(directory):
    """Hold an OS lock for the batch; a process exit releases it, including after a crash."""
    handle = (Path(directory) / 'match.lock').open('a+b')
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise ValueError('This batch is already owned by another player') from error
    return handle


# Engines


def run_checkpoints(run):
    """Checkpoint ids of a run directory, champion first, then newest first."""
    exports = sorted((p for p in Path(run).glob('checkpoints/*/*/ema.pt') if p.parent.name.isdigit()),
                     key=lambda p: (int(p.parent.name), p.parent.parent.name), reverse=True)
    ids = [p.parent.relative_to(Path(run) / 'checkpoints').as_posix() for p in exports]
    try:
        champion = json.loads((Path(run) / 'champion.json').read_text(encoding='utf-8')).get('checkpoint')
    except (OSError, ValueError, AttributeError):
        champion = None
    return ([champion] if champion in ids else []) + [c for c in ids if c != champion]


def six_backend(folder):
    """The fastest backend `sixengine` in `folder` can run here, as (name, flags, PATH directories to add):
    TensorRT, then CUDA, when the build ships ONNX Runtime's provider for it and its libraries are found, then
    DirectML for the DirectML build (it ships DirectML.dll), then the CPU. Libraries are looked for in the folder,
    PyTorch's and TensorRT's Python packages, PATH and LD_LIBRARY_PATH."""
    import importlib.util

    def package(name, *parts):
        spec = importlib.util.find_spec(name)
        return Path(spec.submodule_search_locations[0], *parts) if spec and spec.submodule_search_locations else None

    extra = [d for d in (Path(folder), package('torch', 'lib'), package('tensorrt_libs')) if d and d.is_dir()]
    listed = os.pathsep.join(os.environ.get(name, '') for name in ('PATH', 'LD_LIBRARY_PATH'))
    dirs = [*extra, *map(Path, filter(None, listed.split(os.pathsep)))]

    def found(kind, where=dirs):
        name = SIX_LIBRARIES[kind][os.name != 'nt']
        return name is not None and any((d / name).exists() for d in where)

    if found('cuda_build', [Path(folder)]) and found('cuda') and found('cudnn'):
        if found('tensorrt_build', [Path(folder)]) and found('tensorrt'):
            return 'TensorRT', ['--trt'], extra
        return 'CUDA', [], extra
    if found('directml', [Path(folder)]):
        return 'DirectML', [], extra
    return 'CPU', ['--cpu'], []


def presets_of(kind, spec, device=None):
    """The presets of an engine entry: the kind's defaults (CPU_PRESETS for a Bubble on a 'cpu' server `device`),
    overridden per preset by `spec`. Six-protocol presets may add `args`, extra command-line arguments for an engine
    whose strength is set at launch, like Shrimp."""
    defaults = CPU_PRESETS if kind == 'bubble' and device == 'cpu' else PRESETS[kind]
    presets = {name: dict(budget) for name, budget in defaults.items()}
    if not isinstance(spec or {}, dict):
        raise ValueError('presets must be an object')
    for name, budget in (spec or {}).items():
        if name not in presets or not isinstance(budget, dict):
            raise ValueError(f'unknown preset {name}')
        for key, value in budget.items():
            if key == 'args' and kind == 'six' and isinstance(value, list) and all(isinstance(v, str) for v in value):
                continue
            limits = LIMITS | KIND_LIMITS.get(kind, {})
            if (key not in PRESETS[kind]['standard'] | KIND_LIMITS.get(kind, {}) or type(value) is not int
                    or not limits[key] <= value <= MAX_BUDGET):
                raise ValueError(f'bad {key} in preset {name}')
        presets[name] = presets[name] | budget
    return presets


def scan(models=None, runs=None, extra_runs=(), seal=None, device=None):
    """Every engine on offer, by id.

    Bubble runs come from `extra_runs`, the directories in `runs`, and `models`; single `.pt` exports come from
    `models`, except inside folders an engine setup made (they hold `setup.json`, see engine_setup). A directory
    in `models` holding `sixengine` and `gen-*.onnx` networks offers Six once, on the backend `six_backend` finds,
    labelled with that backend; its networks are its `checkpoints`, newest first. `models/<name>.json` adds one
    entry: {"name", "kind": "bubble", "path", optional "q_range_floor" (0 to 2, the searches' neural_search floor)},
    {"name", "kind": "six", "command", "mirrored", "presets", optional "files" a match also hashes and
    "badge"; a command starting with "python" runs on this server's Python},
    {"name", "kind": "strix", "model" or "networks" ({id: path}, the default first), optional "engine"} or
    {"name", "kind": "seal", "library"}, paths relative to the file. `seal`, the library built with -DHEXO_SEAL_SOURCE, adds Seal when it exists. Entries carry `id`,
    `name`, `kind` (how the server runs it), `badge` (which bot it is, as the page shows it: the kind unless a Six
    protocol entry names another bot, like Shrimp), `presets`, `label` (the name the page shows; the first run in `extra_runs` is labelled
    Bubble), `checkpoints` (Bubble checkpoints or Six networks), and the server-only `path` (Bubble), `command`,
    `cwd`, `mirrored`, `libraries`, `files` and, for a Six folder, `networks` ({name: path}) and `backend` (Six protocol,
    see `command_of`), `model` or `networks` and `engine` (Strix; its networks are its `checkpoints`) or `library`
    (Seal). An id is `kind:name`;
    entries sharing one get a suffix from `engine_identity`, so an id never moves to another engine. `device`, the
    server's ('cpu' or 'cuda'), picks Bubble's preset ladder (see `presets_of`)."""
    found, seen = [], set()
    models = models and Path(models).resolve()

    def add(kind, name, presets=None, label=None, badge=None, **fields):
        found.append(dict(name=name, label=label or name, kind=kind, badge=badge or kind, presets=presets_of(kind, presets, device),
                          **fields))

    def bubble(path, name=None, label=None, q_range_floor=0.):
        path = Path(path).resolve()
        if type(q_range_floor) not in (int, float) or not 0 <= q_range_floor <= 2:
            raise ValueError('q_range_floor must lie in [0, 2]')
        if (path, q_range_floor) in seen:
            return
        seen.add((path, q_range_floor))
        floor = dict(q_range_floor=float(q_range_floor)) if q_range_floor else {}
        if path.is_dir():
            if checkpoints := run_checkpoints(path):
                add('bubble', name or path.name, label=label, checkpoints=checkpoints, path=path, **floor)
        elif path.suffix == '.pt' and path.exists():
            add('bubble', name or (path.parent.name if path.stem == 'ema' else path.stem), label=label,
                checkpoints=[''], path=path, **floor)

    for index, run in enumerate(extra_runs):
        bubble(run, label=None if index else 'Bubble')
    for folder in (runs, models):
        if folder and Path(folder).is_dir():
            for child in sorted(Path(folder).iterdir()):
                if child.is_dir():
                    bubble(child)
    if models and Path(models).is_dir():
        installed = {folder for folder in Path(models).iterdir() if (folder / 'setup.json').is_file()} | {models / '.setup'}
        for path in sorted(Path(models).rglob('*.pt')):
            if 'checkpoints' not in path.relative_to(models).parts and not installed & set(path.parents):
                bubble(path)
        for folder in sorted(p for p in Path(models).iterdir() if p.is_dir()):
            binary = folder / ('sixengine.exe' if os.name == 'nt' else 'sixengine')
            if not binary.exists():
                continue
            backend, flags, libraries = six_backend(folder)
            networks = {n.stem: n for n in sorted(folder.glob('gen-*.onnx'), reverse=True)}
            if networks:
                add('six', f'Six · {backend}', label=backend, checkpoints=list(networks), networks=networks,
                    backend=backend, command=[str(binary), *flags], cwd=folder, mirrored=True, libraries=libraries)
        for path in sorted(Path(models).glob('*.json')):
            try:
                spec = json.loads(path.read_text(encoding='utf-8'))
                name, kind = spec.get('name') or path.stem, spec.get('kind')
                if kind == 'bubble':
                    bubble(path.parent / spec['path'], name, q_range_floor=spec.get('q_range_floor', 0.))
                elif kind == 'six':
                    command = spec['command']
                    if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
                        raise ValueError('command must be a list of arguments')
                    first = path.parent / command[0]
                    command[0] = sys.executable if command[0] == 'python' else str(first) if first.exists() else command[0]
                    badge = spec.get('badge', 'six')
                    if not isinstance(badge, str) or not re.fullmatch(r'[a-z][a-z0-9-]*', badge):
                        raise ValueError('badge must be a lowercase word')
                    add('six', name, spec.get('presets'), badge=badge, command=command, cwd=path.parent,
                        mirrored=spec.get('mirrored') is True, libraries=[],
                        files=[(path.parent / file).resolve() for file in spec.get('files', [])])
                elif kind == 'strix':
                    engine = {'engine': (path.parent / spec['engine']).resolve()} if 'engine' in spec else {}
                    if 'networks' in spec:
                        networks = {str(k): (path.parent / v).resolve() for k, v in spec['networks'].items()}
                        add('strix', name, spec.get('presets'), checkpoints=list(networks), networks=networks, **engine)
                    else:
                        add('strix', name, spec.get('presets'), model=(path.parent / spec['model']).resolve(), **engine)
                elif kind == 'seal' and (path.parent / spec['library']).is_file():
                    add('seal', name, spec.get('presets'), library=(path.parent / spec['library']).resolve())
            except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
                continue
    add('drip', 'Drip')
    if seal is not None and Path(seal).exists():
        add('seal', 'Seal', library=Path(seal))
    bases = [f"{e['kind']}:{e['name']}" for e in found]
    entries = OrderedDict()
    for base, entry in zip(bases, found):
        suffix = hashlib.blake2b(engine_identity(entry).encode(), digest_size=3).hexdigest()
        key = base if bases.count(base) == 1 else f'{base}~{suffix}'
        entries[key] = dict(id=key, **entry)
    return entries


def engine_identity(entry):
    """What makes an entry its engine: its path (and a Bubble's Q range floor), command or library, or its model and
    executable (Strix)."""
    identity = str(entry.get('path') or entry.get('command') or entry.get('library') or
                   (entry.get('model') or entry.get('networks'), entry.get('engine')))
    return identity + (f"@q{entry['q_range_floor']!r}" if entry.get('q_range_floor') else '')


def command_of(entry, checkpoint):
    """The command line of a Six-protocol entry at `checkpoint`: a Six folder's engine with `--net` and that
    network, any other entry's own command."""
    if not entry.get('networks'):
        return list(entry['command'])
    return [entry['command'][0], '--net', str(entry['networks'][checkpoint]), *entry['command'][1:]]


def export_path(entry, checkpoint):
    """The weights file of a Bubble entry at `checkpoint`."""
    return entry['path'] / 'checkpoints' / checkpoint / 'ema.pt' if checkpoint else entry['path']


def file_identity(path):
    """Path, size and modification, change and inode stamps: a new value whenever the file is replaced."""
    stat = Path(path).stat()
    return str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


@functools.lru_cache(maxsize=256)
def file_digest(identity):
    """SHA-256 of the file `identity` (see `file_identity`) names."""
    from legacy.train import digest
    return digest(identity[0])


@functools.lru_cache(maxsize=8)
def file_build(package, record, binary):
    """The solver build of `package`, or 'none' when its record is unreadable."""
    import tactical_proof
    try:
        return tactical_proof.build_hash(package)[:8]
    except (OSError, ValueError, KeyError, TypeError):
        return 'none'


def model_key(path):
    """Evaluations are keyed by the weights they came from: the first 16 hex digits of the file's SHA-256."""
    return file_digest(file_identity(path))[:16]


def search_key(weights, entry, budget=None):
    """The weights with the entry's Q floor, the solver preset's clock and a Bubble budget's Time, so saved searches
    cannot mix."""
    key = weights + (f"~q{entry['q_range_floor']!r}" if entry.get('q_range_floor') else '')
    if budget and budget.get('solver_ms'):
        key += f"~solver{budget['solver_ms']}"
    if budget and entry.get('kind') == 'bubble':
        key += f"~ms{budget['ms']}" if budget.get('ms') else ''
    return key


class Cancelled(Exception):
    """The job was cancelled while the engine was working."""


class Yielded(Exception):
    """A review stepped aside for more urgent jobs; it goes back in the queue."""


class Watched:
    """A network evaluator that reports each batch's size to `watch`, which may raise Cancelled."""

    def __init__(self, inner, watch):
        self.inner, self.watch = inner, watch

    def evaluate(self, histories):
        self.watch(len(histories))
        return self.inner.evaluate(histories)


class Bubble:
    """One HexNet export on `device` with its evaluation cache: `evaluator` for single positions, `scheduler()` for the
    hybrid scheduler's packed batches."""

    def __init__(self, path, device):
        import hexnet
        from legacy.train import digest
        from neural_search import EvaluationCache
        self.sha256 = digest(path)
        self.evaluator = hexnet.DenseEvaluator(hexnet.load_model(path), device, self.sha256, max_batch=16)
        self.cache = EvaluationCache(4096)
        self.native = None

    def scheduler(self):
        """The same weights for the hybrid scheduler's packed batches of up to 128 rows (see `search`), made on first use."""
        import hexnet
        if self.native is None:
            self.native = hexnet.DenseEvaluator(self.evaluator.model, self.evaluator.device, self.sha256, max_batch=128)
            self.native.free = []  # Packed forwards own their staging until their completion fence.
        return self.native


def verified(result):
    return result.get('status') == 'PROVEN_WIN' and result.get('native_verified')


def searched(result):
    """True when a solver query ran to a verdict (a proof, or a reason in `dense_solver.VERDICTS`); False when it
    failed to run (worker starting or restarting, deadline, crash)."""
    from dense_solver import VERDICTS
    return (result.get('status') in ('PROVEN_WIN', 'PROVEN_LOSS') and result.get('native_verified')) or result.get('reason') in VERDICTS


def principal_variation(history, certificate, *, attacker=None, known=()):
    """The principal variation of a verified certificate for the side to move at `history`, as (stones, plies).

    `stones` are [q, r, player, ply] up to the winning stone, `ply` counting placements from 1: at each attacker
    turn the certificate's primary choice (its fewest worst-case attacker turns), at each defender turn the covered
    reply whose subtree lasts the most attacker turns, among equals the one whose stones lie nearest the attacker's
    last stone (least summed hex distance), then the first listed. At an unstoppable fork every reply loses at once,
    so the defender's two stones are left out, their plies skipped, and the attacker completes the shortest of the
    certificate's threats. `plies` counts the placements up to the winning stone, those two left-out stones
    included, so it is the last stone's `ply`."""
    from dense_solver import Proof
    proof = Proof([tuple(map(int, p)) for p in history], certificate)
    local, stones, plies, index = replay(history), [], 0, proof.root
    try:
        attacker = local.player if attacker is None else attacker
        for i, node in enumerate(proof.nodes):
            if node['kind'] == 'exact':
                fact = known[node['fact']]
                n = len(fact['history'])
                proof.depth[i] = proof_turns(fact['plies'], 2 if n % 2 else 1, player_at(n) == attacker)
        near = lambda reply: sum(hex_distance(cell, stones[-1] if stones else history[-1]) for cell in reply['action'])
        while local.winner < 0:
            node = proof.nodes[index]
            if node['kind'] == 'link':
                index = node['child']
                continue
            if node['kind'] == 'exact':
                fact = known[node['fact']]
                after, line = node.get('after', []), fact.get('pv', [])
                if [p[:2] for p in line[:len(after)]] == after:
                    stones += [[*p[:3], p[3] + plies - len(after)] for p in line[len(after):]]
                plies += fact['plies'] - len(after)
                break
            if node['kind'] == 'unstoppable':
                if node.get('threats'):
                    threat = min(node['threats'], key=len)
                    stones += [[int(q), int(r), attacker, plies + local.remaining + 1 + i] for i, (q, r) in enumerate(threat)]
                    plies += local.remaining + len(threat)
                break
            if node['kind'] in ('defender_replies', 'zone_replies'):
                responses = node['responses'] if node['kind'] == 'defender_replies' else [
                    r for r in node['responses'] if local.legal(*r['action'][0])]
                if not responses:
                    break
                reply = min(responses, key=lambda r: (-proof.turns(r['child']), near(r)))
                action, index = reply['action'], reply['child']
            else:
                action, index = node['action'], node.get('child')
            for q, r in action:
                if local.winner < 0:
                    plies += 1
                    stones.append([int(q), int(r), local.player, plies])
                    local.play(int(q), int(r))
            if index is None:
                break
        return stones, plies
    finally:
        local.close()


def hex_distance(a, b):
    """Steps between hex cells `a` and `b` ([q, r] axial)."""
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def proof_turns(plies, remaining, mover_wins):
    """The winner's turns in a proof `plies` placements long, from a position where the side to move has
    `remaining` stones left in its turn and every later turn is two stones."""
    if mover_wins:
        return 1 if plies <= remaining else 1 + -(-(plies - remaining) // 4)
    return -(-(plies - remaining) // 4)


def interruptible(call, watch, abort):
    """`call(stop)` on its own thread, polling `watch(0)` meanwhile; when `watch` raises, the event `stop` is set
    before `abort()` ends the call, so a call that has not started its work yet returns at once too, and the
    exception propagates."""
    result, stop = {}, threading.Event()
    def run():
        try:
            result['value'] = call(stop)
        except Exception as error:
            result['error'] = error
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(.05)
        try:
            watch(0)
        except Cancelled:
            stop.set()
            abort()
            raise
    if 'error' in result:
        raise result['error']
    return result['value']


def move_row(action, probability, value=None):
    """A `top` row: [q, r, probability] (from a search, the stone's share of the improved policy), then, from a
    search, the stone's completed Q as the mover's win probability and 1 or -1 when the search proved that stone wins
    or loses (an exact child has value exactly +-1), else 0."""
    row = [*map(int, action), round(float(probability), 4)]
    if value is None:
        return row
    return row + [round((float(value) + 1) / 2, 4), 1 if value >= 1 else -1 if value <= -1 else 0]


def top_rows(actions, policy, values=None, count=5, lead=None, won=False):
    """The `count` `move_row`s to show: `lead` (the stone played) first, then stones the search proved to win,
    the others by policy and stones it proved to lose last. A stone below a policy share of 0.00005 is left out
    unless the search proved it wins: its value is only the search's fill-in for a stone it never tried. With
    `won` the solver proved `lead` wins, and its row says so whatever the search found."""
    import numpy as np
    policy = np.asarray(policy, dtype=float)
    rank = np.zeros(len(policy)) if values is None else np.where(values >= 1, 0, np.where(values <= -1, 2, 1))
    shown = (policy >= .00005) | (rank == 0) if values is not None else policy >= .00005
    first = [i for i, a in enumerate(actions) if lead is not None and list(map(int, a)) == list(map(int, lead))]
    order = first + [i for i in np.lexsort((-policy, rank)) if shown[i] and i not in first]
    rows = [move_row(actions[i], policy[i], None if values is None else values[i]) for i in order[:count]]
    if won:
        row = rows[0] if first else move_row(lead, 0.)
        rows = [row[:3] + [1., 1]] + (rows[1:] if first else rows[:count - 1])
    return rows


def glimpse(frame, player):
    """A search under way at its root, from a hybrid progress `frame` (InferenceService.progress) for `player` to
    move, or None before the root has statistics: `top` (five first stones as in `evaluate`), `value` (win
    probability of the side to move under the current improved policy, the root's own value when every shown stone
    is proven lost) and `completed` (the owner's completed simulations)."""
    import numpy as np
    edges = np.asarray(frame['edges'], np.float64)
    actions, policy = edges[:, :2].astype(np.int64), edges[:, 5]
    if not len(actions) or not policy.sum() > 0:
        return None
    policy = policy / policy.sum()
    top = top_rows(actions, policy, edges[:, 3])
    winner = frame['exact_winner']
    refuted = bool(top) and all(row[4] < 0 for row in top) and winner < 0
    value = ((1. if winner == player else -1.) if winner >= 0 else frame['node_value'] if refuted
             else float(policy @ edges[:, 4]))
    return dict(top=top, value=round((value + 1) / 2, 4), completed=int(frame['completed']),
                refuted=len(top) if refuted else 0)


def proof_evidence(result, certificate):
    """Save only used premises, with dense indices so the saved certificate can
    be checked against its own dependency list after the query is gone."""
    dependencies = result.get('dependencies', [])
    indices = {d['fact']: i for i, d in enumerate(dependencies)}
    certificate = dict(certificate, nodes=[dict(n, fact=indices[n['fact']]) if n['kind'] == 'exact' else n
                                         for n in certificate['nodes']])
    return dict(certificate=certificate, dependencies=[dict(fact=i, outcome=d['outcome']) for i, d in enumerate(dependencies)],
                solver_build=result.get('build_hash'))


def winning_line(history, result, known=()):
    """The continuation and bound of an already verified mover certificate."""
    cert = result.get('certificate') or json.loads(result['certificate_json'])
    pv, plies = principal_variation(history, cert, known=known)
    reused = any(n['kind'] in ('stamp', 'stamp_link', 'zone_replies') for n in cert['nodes'])
    if reused:
        plies = (2 if len(history) % 2 else 1) + 4*(result['proof_turns']-1)
    evidence = proof_evidence(result, cert) if result.get('dependencies') or reused else {}
    return dict(moves=[list(m) for m in result['moves']], pv=pv,
                proof=dict(winner=player_at(len(history)), turns=result['proof_turns'], plies=plies, **evidence))


def solve(prover, history, solver_nodes, watch=lambda n: None, known=(), limit_ms=None):
    """What the solver knows of `history` for `evaluate`: `moves` and `pv` (see `principal_variation`) of a proven
    win for the side to move, its certificate tightened to the fewest attacker turns the budget allows, `proof`
    ({winner, turns, plies} or None), `threat` (the stones of the opponent's forced win if it moved now), `solved`
    (False when a query failed to run) and `used` (nodes spent). `solver_nodes` 0 or no `prover` asks nothing.
    `limit_ms` caps each query's deadline, for a budget in time."""
    found = dict(moves=[], pv=[], proof=None, threat=[], solved=True, used=0)
    if prover is None or not solver_nodes:
        return found
    history = [tuple(map(int, p)) for p in history]
    game = replay(history)
    player, remaining = game.player, game.remaining
    game.close()
    longest, end = min(60_000, max(10_000, solver_nodes // 8)), time.monotonic() + limit_ms / 1000 if limit_ms else None

    def deadline():
        """Each query's ms: its own deadline, within what is left of `limit_ms` for all of them."""
        return longest if end is None else max(1, min(longest, int((end - time.monotonic()) * 1000)))

    limited, cells = [], 0
    for fact in known:
        if len(limited) >= 4096 or cells + len(fact['history']) > 200_000:
            break
        limited.append(fact)
        cells += len(fact['history'])
    known = limited
    premises = [{k: fact[k] for k in ('history', 'winner', 'plies')} for fact in known]
    options = dict(known=premises) if premises else {}
    # A winner alone does not supply the winning turn. Keep searching for its
    # witness, while still reusing every interior fact.
    mine_known = [f for f in known if len(f['history']) != len(history) or f['winner'] != player]
    mine_options = dict(known=[{k: f[k] for k in ('history', 'winner', 'plies')} for f in mine_known]) if mine_known else {}
    mine = interruptible(lambda stop: prover.history(history, attacker='mover', nodes=solver_nodes, ms=deadline(),
                                                     shortest=True, cancel_event=stop, **mine_options),
                         watch, prover.abort)
    found.update(solved=searched(mine), used=mine.get('nodes_used', 0))
    if verified(mine) and mine['moves']:
        found.update(winning_line(history, mine, mine_known))
        return found
    theirs = interruptible(lambda stop: prover.history(history, attacker='opponent', nodes=solver_nodes, ms=deadline(),
                                                       cancel_event=stop, **options), watch, prover.abort)
    found.update(solved=found['solved'] and searched(theirs), used=found['used'] + theirs.get('nodes_used', 0))
    if verified(theirs):
        found['threat'] = [list(m) for m in theirs['moves']]
    if not verified(theirs) and not known:
        return found
    # The real defender root can use graph facts even when the flipped-turn
    # proposal search did not find a threat within its budget.
    defended = interruptible(lambda stop: prover.history(history, attacker='defender', nodes=solver_nodes, ms=deadline(),
                                                        cancel_event=stop, **options), watch, prover.abort)
    found.update(solved=found['solved'] and searched(defended), used=found['used'] + defended.get('nodes_used', 0))
    if defended.get('status') == 'PROVEN_LOSS' and defended.get('native_verified'):
        cert = defended.get('certificate') or json.loads(defended['certificate_json'])
        pv, _ = principal_variation(history, cert, attacker=1-player, known=known)
        # The line can stop at an exact premise or an unstoppable fork without
        # a displayed witness. Keep the verified worst-case bound, not its prefix.
        plies = remaining + 2 + 4 * (defended['proof_turns'] - 1)
        found.update(pv=pv, proof=dict(winner=1-player, turns=defended['proof_turns'], plies=plies,
                                     **proof_evidence(defended, cert)))
    return found


def prove(bubble, prover, proofs, history, ms, watch=lambda n: None, live=None, known=(), stamps=False):
    """The solver preset's view of `history`, in `solve`'s shape, found by proof work alone for up to `ms`.

    Two provers run until one returns a verified proof for either side or the clock ends: the tactical solver
    `prover` asks the root for a mover win with 32,768 nodes, then four times as many each round, and a hybrid
    scheduler search of the position feeds the shared ProofWorkers `proofs` the positions its neural search reaches,
    with the whole owner budget and proof stamps when `stamps`. `live(seen)` receives {solver: {elapsed_ms, root
    ('asking', or 'no forcing win' once the solver rules out a mover win before spending its nodes), root_nodes,
    root_budget (the current query's nodes), frontier (queued and running proof jobs), busy, workers, frontier_nodes,
    checked (frontier jobs answered), proof}} about twice a second, read from the running loops. `found['solver']`
    reports the totals, among them `native_nodes` and `certificates` of the native frontier."""
    from neural_search import GameGraph
    import tactical_proof
    from hybrid_scheduler import SearchPool, InferenceService
    history = [tuple(map(int, p)) for p in history]
    game = replay(history)
    player, remaining = game.player, game.remaining
    game.close()
    start, workers = time.monotonic(), proofs.stats()['workers']
    end = start + ms / 1000
    found = dict(moves=[], pv=[], proof=None, threat=[], solved=True, used=0)
    premises, cells = [], 0
    for f in known:   # the tactical solver takes at most 4,096 premises of 200,000 cells in all (see `solve`)
        if len(premises) >= 4096 or cells + len(f['history']) > 200_000:
            break
        if len(f['history']) != len(history) or f['winner'] != player:
            premises.append({k: f[k] for k in ('history', 'winner', 'plies')})
            cells += len(f['history'])
    root, stop = dict(nodes=0, budget=32768, result=None, ruled=False), threading.Event()

    def ask():
        nodes = 32768
        while not stop.is_set() and time.monotonic() < end:
            root['budget'] = nodes
            left = int((end - time.monotonic()) * 1000)
            result = prover.history(history, attacker='mover', nodes=nodes, ms=max(1, min(left, 60_000, max(10_000, nodes // 8))),
                                    shortest=True, cancel_event=stop, **(dict(known=premises) if premises else {}))
            root['nodes'] += result.get('nodes_used', 0)
            if verified(result) and result['moves']:
                root['result'] = result
                return
            if not searched(result):   # the worker was starting or restarting: ask again
                stop.wait(.05)
                continue
            if result.get('nodes_used', 0) < nodes:   # the solver ruled the root out before spending its nodes
                root['ruled'] = True
                return
            nodes = min(4 * nodes, tactical_proof.MAX_NODES)
    asking = threading.Thread(target=ask, daemon=True)
    evaluator = bubble.scheduler()
    graph = GameGraph(evaluator, bubble.sha256, history, seed=1740, cache=bubble.cache, tactics=True, round_barrier=True)
    pool = SearchPool([graph], quantum=64, views=8, depth=8, work=1, seed=1740)
    service, frame, event, shown = None, None, None, 0.
    try:
        pool.enable_proofs(shared=proofs, slice_ms=8, stamps=stamps, owner_budget=1.)
        service = InferenceService([pool], [evaluator], batch_size=128, quantum=64, pending=2, flights=2,
                                   interleave_feedback=True, progress=True)
        service.start(continuous=True)
        service.launch()
        service.retarget(0, 0, [list(p) for p in history], expected=0, work=0, ms=max(1, int(ms)), samples=16, views=8)
        asking.start()
        sequence = 0
        while root['result'] is None and time.monotonic() < end:
            watch(0)
            if (event := service.event()) is not None:
                break
            if (latest := service.progress(0, 0, token=1, after=sequence)) is not None:
                frame, sequence = latest, latest['snapshot_sequence']
                if frame['exact_winner'] >= 0:
                    break
            if live and time.monotonic() >= shown:
                busy, loop = proofs.stats(), pool.proofs.progress()
                live(dict(solver=dict(elapsed_ms=round((time.monotonic() - start) * 1000),
                                      root='no forcing win' if root['ruled'] else 'asking', root_nodes=root['nodes'],
                                      root_budget=root['budget'], frontier=busy['queued'] + busy['live'],
                                      busy=busy['active'], workers=workers, frontier_nodes=loop['fresh_nodes'],
                                      checked=loop['finished'], proof=None)))
                shown = time.monotonic() + .5
            service.wait(50)
    finally:
        stop.set()
        if asking.is_alive():
            prover.abort()
            asking.join()
        if service is not None:
            service.cancel()
            service.close()
        stats = pool.proofs.stats() if pool.proofs is not None else {}
        records = pool.proofs.records() if pool.proofs is not None else []
        pool.close()
        graph.close()
    native = event if event is not None and event.get('exact_winner', -1) >= 0 else frame
    found['solver'] = dict(elapsed_ms=round((time.monotonic() - start) * 1000), root_nodes=root['nodes'],
                           root='no forcing win' if root['ruled'] else 'asking',
                           native_nodes=stats.get('fresh_nodes', 0), certificates=stats.get('records', 0), workers=workers)
    found['used'] = root['nodes'] + stats.get('fresh_nodes', 0)
    facts = frontier_facts(records)
    if facts:
        found['proofs'] = facts   # the frontier's verified answers join the game's proof table (Session.save)
    if root['result'] is not None:
        found.update(winning_line(history, root['result'], [f for f in known if len(f['history']) != len(history)
                                                               or f['winner'] != player]))
    elif native is not None and native.get('exact_winner', -1) >= 0:
        winner, plies = native['exact_winner'], int(native['proof_plies'])
        if winner == player and native['winning_turn']:
            found['moves'] = [list(p) for p in native['winning_turn']]
            found['pv'] = [[*p, player, i + 1] for i, p in enumerate(found['moves'])]
        found['proof'] = dict(winner=winner, turns=proof_turns(plies, remaining, winner == player), plies=plies)
    found['solver']['proof'] = found['proof']
    return found


def proof_key(history):
    """`history`'s position whatever the order of its stones: each side's cells, sorted. The stone count fixes the
    side to move and its remaining stones, so equal keys have equal game values."""
    sides = ([], [])
    for i, (q, r) in enumerate(history):
        sides[player_at(i)].append((int(q), int(r)))
    return ' / '.join(' '.join(f'{q},{r}' for q, r in sorted(side)) for side in sides)


def proof_plies(proof, history):
    """Placements to the winning stone of a saved `proof` ({winner, turns, plies?}) at `history`: its `plies`, else
    the bound its `turns` give (docs/search-outcomes.md)."""
    if proof.get('plies'):
        return int(proof['plies'])
    game = replay(history)
    try:
        return game.remaining + (0 if proof['winner'] == game.player else 2) + 4 * (int(proof['turns']) - 1)
    finally:
        game.close()


def proof_line_length(history, winner, pv, offset=0):
    """Length of a drawn line ending in six; an incomplete line cannot win a distance tie."""
    if not pv or any(len(p) != 4 for p in pv):
        return math.inf
    own = {(q, r) for i, (q, r) in enumerate(history) if player_at(i) == winner}
    own.update((q, r) for q, r, side, _ in pv if side == winner)
    q, r, side, ply = pv[-1]
    if side != winner:
        return math.inf
    for dq, dr in ((1, 0), (0, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            for i in range(1, 6):
                if (q + sign*i*dq, r + sign*i*dr) not in own:
                    break
                count += 1
        if count >= 6:
            return ply - offset
    return math.inf


class Proofs:
    """The proven positions of a game, keyed by `proof_key`: {winner, plies, pv}, where the winner completes six
    within `plies` placements against any defence and `pv` ([q, r, player, ply], ply from 1) is the known line from
    that position. `add` indexes a saved evaluation that holds a proof together with the positions along its line;
    `edges` and `known` read what a position's continuations prove. Safe to use from several threads."""

    def __init__(self):
        self.entries, self.sizes, self.seen, self.lock = {}, {}, set(), threading.RLock()
        self.known_cache = {}

    def add(self, history, record, line=None):
        """Index the root proof and any proven continuations retained in `record['proofs']`.
        `line`, the record's saved text, skips a record already indexed."""
        if line in self.seen:
            return
        for fact in record.get('proofs', []):
            self.add(fact['history'], dict(proof=dict(winner=fact['winner'], plies=fact['plies']), pv=fact['pv']))
        proof = record.get('proof')
        if not proof:
            if line is not None:
                self.seen.add(line)
            return
        current = [tuple(map(int, p)) for p in history]
        plies, pv = proof_plies(proof, current), [list(p) for p in record.get('pv') or []]
        # A turn bound can outlast this line; do not export its finished board as a solver premise.
        end = min(plies, proof_line_length(current, proof['winner'], pv))
        with self.lock:
            if line is not None:
                self.seen.add(line)
            self.put(current, proof['winner'], plies, pv)
            for i, stone in enumerate(pv):
                if len(stone) != 4 or stone[3] != i + 1 or stone[2] != player_at(len(current)) or stone[3] >= end:
                    break
                q, r, _, ply = stone
                current.append((q, r))
                self.put(current, proof['winner'], plies - ply, [[*p[:3], p[3] - ply] for p in pv[i + 1:]])

    def put(self, history, winner, plies, pv):
        key = proof_key(history)
        old = self.entries.get(key)
        witnessed = lambda line: len(line) if all(len(stone) == 4 for stone in line) else 0
        length = proof_line_length(history, winner, pv)
        if old is None or old['winner'] == winner and (plies, length, -witnessed(pv)) < (old['plies'], old['length'], -witnessed(old['pv'])):
            self.known_cache.clear()
            stones = frozenset((q, r, player_at(i)) for i, (q, r) in enumerate(history))
            self.entries[key] = dict(history=[list(p) for p in history], winner=int(winner), plies=int(plies), pv=pv, stones=stones, length=length)
            self.sizes.setdefault(len(history), set()).add(key)

    def line(self, history, outcome):
        """Follow the shortest known attack and longest covered defence within this proof's bound.
        A partial set of defensive replies never tightens the position's universal bound. Omitted defender stones
        end the walk because the position after them is unknown."""
        current, pv, i = list(history), outcome.get('pv') or [], 0
        with self.lock:
            while i <= len(pv):
                length = proof_line_length(current, outcome['winner'], pv[i:], i)
                ready = self.known_cache.get(proof_key(current)) if i else None
                if (ready is not None and ready['winner'] == outcome['winner'] and ready['plies'] + i <= outcome['plies']
                        and math.isfinite(proof_line_length(current, ready['winner'], ready['pv']))
                        and (ready['plies'] + i < outcome['plies'] or proof_line_length(current, ready['winner'], ready['pv']) <= length)):
                    return pv[:i] + [[*p[:3], p[3]+i] for p in ready['pv']]
                entry = self.choice(current)
                replacement = proof_line_length(current, entry['winner'], entry['pv']) if entry is not None else math.inf
                if (entry is not None and entry['winner'] == outcome['winner'] and entry['plies'] + i <= outcome['plies']
                        and (entry['plies'] + i < outcome['plies'] or replacement < length
                             or replacement == length and len(entry['pv']) > len(pv) - i)
                        and all(len(p) == 4 for p in entry['pv'])):
                    pv = pv[:i] + [[*p[:3], p[3] + i] for p in entry['pv']]
                if player_at(len(current)) != outcome['winner']:
                    replies = []
                    for action, (winner, _, child) in self.edges(current).items():
                        if winner != outcome['winner']:
                            continue
                        child = self.known(current + [action]) or child
                        if child['plies'] + 1 + i <= outcome['plies'] and child['pv']:
                            last = child['pv'][-1]
                            replies.append(((last[3] if len(last) == 4 else len(child['pv'])) + 1, action, child))
                    if replies:
                        distance, action, reply = min(replies, key=lambda e: (-e[0], -e[2]['plies'], e[1]))
                        # Retain the certificate's equal-length reply and its tie-break.
                        first = tuple(pv[i][:2]) if i < len(pv) else None
                        if first != action and not any(a == first and d == distance and o['plies'] == reply['plies'] for d, a, o in replies):
                            pv = pv[:i] + [[*action, player_at(len(current)), i + 1]] + [
                                [*p[:3], p[3] + i + 1] for p in reply['pv']]
                if i == len(pv) or len(pv[i]) != 4 or pv[i][3] != i + 1 or pv[i][2] != player_at(len(current)):
                    break
                current.append(tuple(pv[i][:2]))
                i += 1
        return pv

    def facts(self, history):
        """Exact outcomes reachable from this board, nearest positions first."""
        base = frozenset((int(q), int(r), player_at(i)) for i, (q, r) in enumerate(history))
        with self.lock:
            entries = sorted((e for e in self.entries.values() if base <= e['stones']), key=lambda e: len(e['history']))
            found = {proof_key(e['history']): {k:e[k] for k in ('history', 'winner', 'plies', 'pv')} for e in entries[:2048]}
        if history and len(history) % 2 == 0:
            for action, (winner, _, outcome) in self.edges(history).items():
                if winner == player_at(len(history)):
                    continue
                child = [list(p) for p in history] + [list(action)]
                key, old = proof_key(child), found.get(proof_key(child))
                if old is None or outcome['plies'] < old['plies']:
                    found[key] = dict(history=child, **outcome)
        return sorted(found.values(), key=lambda e: len(e['history']))[:2048]

    def edges(self, history):
        """{(q, r): (winner, distance, outcome)} for each stone from `history` whose position is proven: in the
        table, or because one more stone by that position's mover reaches a position the mover wins. `distance`
        counts the stone itself, as hxg_mark_exact takes it; `outcome` is the position's {winner, plies, pv}."""
        size = len(history)
        base = frozenset((int(q), int(r), player_at(i)) for i, (q, r) in enumerate(history))
        found = {}
        with self.lock:
            # A lost half-turn after A covers every legal A,B. On the alternate
            # half-turn after B, A reaches the same board. Keep only the verdict:
            # the saved defender's reply/PV need not be B.
            if size and size % 2 == 0:
                mover = player_at(size)
                for key in self.sizes.get(size, ()):
                    entry = self.entries[key]
                    if entry['winner'] == mover or entry['plies'] < 2:
                        continue
                    missing, replaced = entry['stones'] - base, base - entry['stones']
                    if len(missing) != 1 or len(replaced) != 1:
                        continue
                    first, prior = next(iter(missing)), next(iter(replaced))
                    if first[2] != mover or prior[2] != mover:
                        continue
                    outcome = dict(winner=entry['winner'], plies=entry['plies'] - 1, pv=[])
                    found[first[:2]] = (entry['winner'], entry['plies'], outcome)
            for extra in (1, 2):
                for key in self.sizes.get(size + extra, ()):
                    entry = self.entries[key]
                    if not base <= entry['stones']:
                        continue
                    stones = list(entry['stones'] - base)
                    for first, second in ([(stones[0], None)] if extra == 1 else [stones, stones[::-1]]):
                        if first[2] != player_at(size):
                            continue
                        if second is None:
                            outcome = {k: entry[k] for k in ('winner', 'plies', 'pv')}
                        elif second[2] == player_at(size + 1) == entry['winner']:
                            outcome = dict(winner=entry['winner'], plies=entry['plies'] + 1,
                                           pv=[[*second[:2], entry['winner'], 1]] + [[*p[:3], p[3] + 1] for p in entry['pv']])
                        else:
                            continue
                        old = found.get(first[:2])
                        if old is None or old[2]['plies'] > outcome['plies']:
                            found[first[:2]] = (outcome['winner'], outcome['plies'] + 1, outcome)
        return found

    def choice(self, history):
        """The tightest stored guarantee, also considering shorter winning continuations."""
        with self.lock:
            entry = self.entries.get(proof_key(history))
            if entry is None and len(history) > 1 and len(history) % 2 == 1:
                # A losing half-turn covers adding any legal stone of its mover,
                # even when that stone was earlier in this reordered history.
                # A scalar win does not supply the next winning move.
                base = frozenset((int(q), int(r), player_at(i)) for i, (q, r) in enumerate(history))
                mover = player_at(len(history))
                candidates = []
                for key in self.sizes.get(len(history) - 1, ()):
                    loss = self.entries[key]
                    if loss['winner'] != mover or loss['plies'] < 2 or not loss['stones'] < base:
                        continue
                    missing = next(iter(base - loss['stones']))
                    if missing[2] != 1-mover:
                        continue
                    candidates.append(dict(winner=mover, plies=loss['plies'] - 1, pv=[]))
                if candidates:
                    entry = min(candidates, key=lambda e: e['plies'])
        mover = player_at(len(history))
        if entry is not None and entry['winner'] != mover:
            return entry
        wins = []
        for action, (winner, _, child) in self.edges(history).items():
            if winner != mover:
                continue
            child = self.known(list(history) + [action]) or child
            pv = [[*action, mover, 1]] + [[*p[:3], p[3]+1] for p in child['pv']]
            wins.append((child['plies']+1, proof_line_length(history, mover, pv), action, pv))
        wins.sort()
        if not wins or entry is not None and (entry['plies'] < wins[0][0] or entry['plies'] == wins[0][0]
                                              and entry['pv'] and proof_line_length(history, mover, entry['pv']) <= wins[0][1]):
            return entry
        distance, _, _, pv = wins[0]
        return dict(winner=mover, plies=distance, pv=pv)

    def known(self, history):
        """The best known guarantee and its updated continuation, or None when nothing is proven."""
        at = proof_key(history)
        with self.lock:
            if at in self.known_cache:
                return self.known_cache[at]
            found = self.choice(history)
            outcome = None if found is None else dict(winner=found['winner'], plies=found['plies'], pv=self.line(history, found))
            self.known_cache[at] = outcome
            if len(self.known_cache) > 4096:
                del self.known_cache[next(iter(self.known_cache))]
            return outcome

    def extend(self, store, history):
        """Index the saved evaluations holding a proof of every position of `history` (see `Evaluations.proven`)."""
        for ply in range(len(history) + 1):
            for line in store.proven(history[:ply]):
                if line not in self.seen:
                    self.add(history[:ply], json.loads(line), line)


def answered(history, known):
    """The evaluation of `history` that the proof table `known` gives without a search, when it proves a win for
    the side to move whose line holds the rest of the turn: `moves` (those stones), `value` 1, the winning stone as
    the only top row, `proof` and `pv`; else None."""
    outcome = known.known(history) if known is not None else None
    if outcome is None:
        return None
    game = replay(history)
    try:
        mover, remaining = game.player, game.remaining
        if outcome['winner'] != mover:
            return None
        moves = []
        for stone in outcome['pv'][:remaining]:
            if len(stone) != 4 or stone[2] != mover or stone[3] != len(moves) + 1:
                break
            moves.append(stone[:2])
            game.play(*stone[:2])
        if not moves or len(moves) < remaining and game.winner != mover:
            return None
    finally:
        game.close()
    return dict(moves=moves, value=1., top=[[*moves[0], 1., 1., 1]],
                proof=dict(winner=mover, turns=proof_turns(outcome['plies'], remaining, True), plies=outcome['plies']),
                pv=outcome['pv'], threat=[], solved=True, ms=0, actual_completed=0, actual_solver_nodes=0, later=[])


def frontier_facts(records):
    """The verified answers among a proof loop's `records` (ProofLoop.records) as proof-table facts
    {history, winner, plies, pv}."""
    facts = []
    for record in records:
        request, answer = record['request'], record['result']
        if answer.get('status') not in ('PROVEN_WIN', 'PROVEN_LOSS') or not answer.get('native_verified'):
            continue
        turns, cert = answer['proof_turns'], answer.get('certificate') or json.loads(answer['certificate_json'])
        plies = len(answer['moves']) + 4 * (turns - 1) if answer['status'] == 'PROVEN_WIN' else 4 * turns + 2
        pv, _ = principal_variation(request['history'], cert, attacker=answer['winner'], known=request.get('known', ()))
        facts.append(dict(history=request['history'], winner=answer['winner'], plies=plies, pv=pv))
    return facts


def search(bubble, roots, proofs=None, watch=lambda n: None, live=None, stamps=False, ms=None):
    """Hybrid scheduler work on the game graphs of `roots` ([(graph, simulations)], each graph at its root), in one
    SearchPool whose games share `bubble`'s packed network batches. Each graph's native owner completes
    `simulations` simulations over up to HYBRID['views'] views: its root and the positions below it, at most
    HYBRID['depth'] placements down, in quanta of at most HYBRID['quantum'], with 16 root samples and the round
    barrier. With `proofs` (shared hybrid_scheduler.ProofWorkers) each owner also runs a proof frontier on those
    workers, at most PROOF_BUDGET of the owner's time. `watch(n)` receives the network rows launched since its last
    call and may raise Cancelled, which ends every search; while it returns True the network work waits and the
    proofs go on. `live(frame)` receives the first root's progress frames
    (InferenceService.progress) at most every 0.3 s. Returns per root its owner's final root record in
    GameGraph.result's shape (`actions`, improved `policy`, `completed_q`, `values`, `visits`, `node_value`,
    `exact_winner`, `proven`, `proof_plies`), its `action` the stone of the highest improved policy, or the owner's
    own choice at an exact or policy-free root; `completed`, the simulations the owner completed; `proofs`, the
    verified positions its frontier proved (frontier_facts); and `frontier_nodes`, the solver nodes that frontier
    spent. Each graph is left at its root. With `ms` each owner also stops at that clock, its `simulations` a
    ceiling."""
    import numpy as np
    from hybrid_scheduler import InferenceService, SearchPool
    evaluator = bubble.scheduler()
    histories = [[list(p) for p in graph.history] for graph, _ in roots]
    quantum = max(4, min(HYBRID['quantum'], max(simulations for _, simulations in roots)))
    pool = SearchPool([graph for graph, _ in roots], quantum=quantum, views=HYBRID['views'], depth=HYBRID['depth'],
                      work=1, seed=next(SEEDS))
    service, events, records, effort = None, {}, [], {}
    try:
        if proofs is not None:
            pool.enable_proofs(shared=proofs, slice_ms=8, stamps=stamps, owner_budget=PROOF_BUDGET)
        service = InferenceService([pool], [evaluator], batch_size=evaluator.max_batch, quantum=64, pending=2,
                                   flights=2, interleave_feedback=True, progress=live is not None)
        service.start(continuous=True)
        service.launch()
        for game, (history, (_, simulations)) in enumerate(zip(histories, roots)):
            service.retarget(0, game, history, expected=0, work=max(1, simulations), ms=max(1, int(ms)) if ms else 0,
                             samples=16, views=HYBRID['views'])
        launched, sequence, shown, held = 0, 0, 0., False
        while len(events) < len(roots):
            rows = service.stats()['launched_rows']
            if bool(watch(rows - launched)) != held:
                held = not held
                if held:
                    service.pause()
                else:
                    service.resume()
            launched = rows
            if live is not None and 0 not in events and time.monotonic() >= shown:
                if (frame := service.progress(0, 0, token=1, after=sequence)) is not None:
                    sequence, shown = frame['snapshot_sequence'], time.monotonic() + .3
                    live(frame)
            while (event := service.event()) is not None:
                events[event['game']] = event
            if len(events) < len(roots):
                service.wait(50)
    finally:
        try:
            if service is not None:
                service.close()
        finally:
            if pool.proofs is not None:
                records = pool.proofs.records()
                effort = {game: sum(e['fresh_nodes'] for e in pool.proofs.effort(game).values())
                          for game in range(len(roots))}
            pool.close()
    facts, results = frontier_facts(records), []
    for game, ((graph, _), history) in enumerate(zip(roots, histories)):
        if 'error' in events[game]:
            raise ValueError(f"Hybrid search failed: {events[game]['error']}")
        event, edges = events[game], np.asarray(events[game]['edges'], np.float64)
        graph.at(history)
        winner, mover = event['exact_winner'], player_at(len(history))
        policy = edges[:, 5]
        result = dict(actions=edges[:, :2].astype(np.int64), policy=policy, completed_q=edges[:, 3], values=edges[:, 4],
                      visits=edges[:, 6].astype(np.int64), node_value=float(event['node_value']), exact_winner=winner,
                      proven=0 if winner < 0 else 1 if winner == mover else -1,
                      proof_plies=int(event['proof_plies']) if winner >= 0 else 0,
                      action=(list(event['action']) if winner >= 0 or not policy.sum() > 0
                              else edges[np.argmax(policy), :2].astype(np.int64).tolist()),
                      completed=int(event['completed']), frontier_nodes=effort.get(game, 0),
                      proofs=[f for f in facts if f['history'][:len(history)] == history])
        results.append(result)
    return results


class TurnSearch:
    """One evaluation under way (see `evaluate`): the solver's findings, then a search per stone of the turn until
    the turn is complete. When the solver already gave the turn, its stones are played and each position of the turn
    is still searched, for its rows. The searches of later stones are kept in `later` as evaluations of the positions
    inside the turn, without a separate root query; on a solver-proven turn they carry the rest of its `pv`.
    Without a solver proof, a position the search proves has the turn's own stones as its `pv`. `request()` names
    the game graph and simulations of the next search, None when the turn is complete, (None, 0) for the raw
    policy; `take(result)` applies that search's result (see `search`), None for the raw policy.
    `trees(history, simulations, network)` gives the graph and the simulations to run for a stone; by default one
    fresh GameGraph is advanced through the turn, each stone searched with `simulations`. A seat's move passes its
    game's graph (`Engines.game_graph`). With a proof table `known` (`Proofs`) a position it proves lost for the side
    to move gets that proof and line unless the solver proved one (a proven win needs its turn, see `answered`), each
    search starts with the root's proven edges settled (`Proofs.edges`; the mover's losses first, then its wins from
    the shortest, since the first win settles the root), and a proof whose turn reaches a proven position continues
    into that position's line. The positions a search's proof frontier proves join `known` and the result's
    `proofs`."""

    def __init__(self, bubble, network, history, simulations, solved, trees=None, q_range_floor=0., known=None):
        self.bubble, self.network, self.simulations, self.q_range_floor = bubble, network, simulations, q_range_floor
        self.history = [tuple(map(int, p)) for p in history]
        self.local = replay(self.history)
        if self.local.winner >= 0:
            self.local.close()
            raise ValueError('The game has finished')
        self.player, self.start = self.local.player, time.perf_counter()
        self.moves, self.pv, self.proof = list(solved['moves']), solved['pv'], solved['proof']
        self.threat, self.solved, self.solver_used = solved['threat'], solved['solved'], solved['used']
        self.given, self.top, self.value, self.completed, self.tree = bool(self.moves), [], None, 0, None
        self.node_value = None
        self.later, self.played, self.facts = [], 0, {}
        self.trees = trees or self.advanced
        self.known = known if known is not None else Proofs()
        outcome = known.known(self.history) if known is not None and self.proof is None else None
        if outcome is not None and outcome['winner'] != self.player:
            self.proof = dict(winner=outcome['winner'], plies=outcome['plies'],
                              turns=proof_turns(outcome['plies'], self.local.remaining, False))
            self.pv = outcome['pv']

    def advanced(self, history, simulations, network):
        from neural_search import GameGraph
        if self.tree is None:
            self.tree = GameGraph(network, self.bubble.sha256, history, seed=1740, cache=self.bubble.cache,
                                  tactics=True, q_range_floor=self.q_range_floor, round_barrier=True)
        else:
            self.tree.at(history)
        return self.tree, simulations

    def finished(self):
        """True once the turn is complete."""
        return self.local.player != self.player or self.local.winner >= 0 or self.given and self.played >= len(self.moves)

    def request(self):
        if self.finished():
            return None
        if not self.simulations:
            return None, 0
        cells = [tuple(cell[:2]) for cell in self.local.cells]
        tree, simulations = self.trees(cells, self.simulations, self.network)
        if self.proof is not None and self.proof['winner'] != self.player:
            tree.prove_loss(self.proof['winner'], max(1, self.proof['plies'] - self.played))
        edges = self.known.edges(cells) if self.known is not None else {}
        if edges:
            tree.expand()
            mover = self.local.player
            for action, (winner, distance, _) in sorted(edges.items(), key=lambda e: (e[1][0] == mover, e[1][1], e[0])):
                with contextlib.suppress(ValueError):
                    tree.mark(action, winner, distance)
        return tree, simulations

    def take(self, result):
        """Apply a search result, or with simulations 0 evaluate the raw policy."""
        import numpy as np
        from dense_selfplay import root_value
        exact = None
        if result is not None:
            self.completed += result.get('completed', 0)
            self.solver_used += result.get('frontier_nodes', 0)
            for fact in result.get('proofs', []):
                key, old = proof_key(fact['history']), self.facts.get(proof_key(fact['history']))
                if old is None or (fact['plies'], -len(fact['pv'])) < (old['plies'], -len(old['pv'])):
                    self.facts[key] = fact
            if result.get('proofs'):
                self.known.add(self.history, dict(proofs=result['proofs']))
            proven = result.get('proven') or 0
            exact = dict(winner=self.player if proven > 0 else 1 - self.player,
                         turns=proof_turns(result['proof_plies'], self.local.remaining, proven > 0),
                         plies=int(result['proof_plies'])) if proven else None
            if self.proof is None and (proven > 0 or proven < 0 and not self.moves):
                self.proof = dict(exact, plies=exact['plies'] + self.played)
        # Installing a root loss can settle a fresh graph before its first neural
        # expansion. The proof is complete, but playing still needs a legal move.
        if result is None or result['action'] is None:
            current = [tuple(cell[:2]) for cell in self.local.cells]
            found = self.network.evaluate([current])[0]
            actions, values = found['actions'], None
            policy = np.exp(found['logits'] - found['logits'].max())
            policy /= policy.sum()
            action, stone_value = actions[policy.argmax()].tolist(), float(found['q'][0])
            if exact is not None:
                stone_value = 1. if exact['winner'] == self.player else -1.
        else:
            action, policy, actions, values = result['action'], result['policy'], result['actions'], result['completed_q']
            stone_value = root_value(result, self.local.player)
        if not self.given:
            self.moves.append([int(action[0]), int(action[1])])
        stone = self.moves[self.played]
        rows = top_rows(actions, policy, values, lead=stone, won=self.given)
        if not self.played:
            self.top, self.value = rows, (stone_value + 1) / 2
            self.node_value = (result.get('node_value', stone_value) + 1) / 2 if result is not None else self.value
        else:
            if self.given:
                exact = dict(self.proof, plies=self.proof['plies'] - self.played)
            value = (1. if exact['winner'] == self.player else 0.) if exact else (stone_value + 1) / 2
            pv = ([[q, r, side, ply - self.played] for q, r, side, ply in self.pv[self.played:]] if self.given else
                  [[*stone, self.player, 1]] if exact else [])
            self.later.append(dict(history=[tuple(cell[:2]) for cell in self.local.cells], moves=[stone],
                                   value=round(value, 4), top=rows, proof=exact, pv=pv, threat=[]))
        self.local.play(*self.moves[self.played])
        self.played += 1

    def record(self):
        if self.tree is not None and self.played > 1 and not self.given:
            # The later stones' searches changed the graph under the first root: read that root again.
            from dense_selfplay import root_value
            self.tree.at(self.history)
            stats = self.tree.result(0, 0, 0, 0, choice='policy')
            self.top = top_rows(stats['actions'], stats['policy'], stats['completed_q'], lead=self.moves[0])
            self.value = (root_value(stats, self.player) + 1) / 2
            self.node_value = (stats['node_value'] + 1) / 2
        value = (1. if self.proof['winner'] == self.player else 0.) if self.proof else self.value
        if self.top and all(len(r) > 4 and r[4] < 0 for r in self.top) and not self.proof:
            value = self.node_value
        pv = self.pv or ([[*m, self.player, i + 1] for i, m in enumerate(self.moves)] if self.proof else [])
        extended = self.known is not None and pv and not self.pv
        after = self.known.known([*self.history, *map(tuple, self.moves)]) if extended else None
        if after is not None and after['winner'] == self.proof['winner']:
            pv = pv + [[*p[:3], p[3] + len(self.moves)] for p in after['pv']]
        found = dict(moves=self.moves, value=round(value, 4), node_value=self.node_value, top=self.top, proof=self.proof, pv=pv,
                    threat=self.threat, solved=self.solved, ms=round((time.perf_counter() - self.start) * 1000),
                    actual_completed=self.completed, actual_solver_nodes=self.solver_used, later=self.later)
        if self.facts:
            found['proofs'] = list(self.facts.values())
        given = answered(self.history, self.known)
        if given is not None and (not self.proof or given['proof']['plies'] <= proof_plies(self.proof, self.history)):
            found.update({k: given[k] for k in ('moves', 'value', 'top', 'proof', 'pv')})
        elif self.proof:
            found['pv'] = self.known.line(self.history, dict(winner=self.proof['winner'],
                plies=proof_plies(self.proof, self.history), pv=pv))
        return found

    def close(self):
        if self.trees == self.advanced and self.tree is not None:
            self.tree.close()
        self.local.close()


def evaluate(bubble, prover, history, simulations, solver_nodes, watch=lambda n: None, live=None, trees=None,
             solved=None, q_range_floor=0., known=None, solver_ms=0, package=None, proofs=None, stamps=False, ms=0,
             placed=None):
    """Bubble's turn from `history` and what it thinks of the position.

    Returns `moves` (the turn it plays), `value` (win probability of the side to move), `top` (five best first
    stones as `move_row`s, the turn's first stone first), `proof` (None or {winner, turns, plies}: the solver proved
    a win for the side to move, or the search proved the position exact; `plies` placements to the winning stone),
    `pv` (the principal variation as [q, r, player, ply]: the solver's, see `principal_variation`, else on a search
    proof the turn's own stones; [] when unproven) and
    `threat` (the stones of a forced win the opponent would have if it moved now). `solved` is False when a solver
    initial root query failed to run (worker restarting, deadline), so the result must not count as solver-checked.
    Each stone is searched on the hybrid scheduler (see `search`) with `simulations` of work; `simulations` 0 plays
    the raw policy. `solver_nodes` 0 skips root queries and the searches' proof frontier, and no `prover` skips all
    solver work; otherwise the frontier runs on `proofs` (shared ProofWorkers), or on SOLVER_WORKERS of this call's
    own on the tactical `package`, with proof stamps when `stamps`. `watch(n)` receives the network rows launched
    and may raise Cancelled; `live(glimpse)` receives the search of the first stone as it goes, a few times a
    second. `trees` is `TurnSearch`'s graph source; `solved`, when given, is the solver's view (see `solve`) and no
    initial root query is made. `q_range_floor` is the fresh graphs' neural_search floor.
    `known`, a proof table (`Proofs`), answers a position it proves won for the side to move without solver or
    search (see `answered`) and otherwise informs the search (see `TurnSearch`).
    `solver_ms` (the solver preset) replaces the root queries with `prove` for up to that long; the search then
    plays the proven turn or, without a proof, searches as usual, and the result carries `prove`'s `solver`
    totals. `ms` (a budget in time) is the turn's clock: the root queries get at most a quarter of it, the first of
    two stones 60% of what is left, the last stone the rest, with `simulations` a ceiling per stone. `placed(moves)`
    receives the turn's stones so far as each searched stone before the last is decided."""
    from hybrid_scheduler import ProofWorkers
    end = time.monotonic() + ms / 1000 if ms else None
    game = replay(history)
    finished = game.winner >= 0
    game.close()
    if finished:
        raise ValueError('The game has finished')
    if (found := answered(history, known)) is not None:
        return found
    network = Watched(bubble.evaluator, watch)
    facts = known.facts(history) if known is not None else []
    if trees is not None:
        tree, _ = trees(history, simulations, network)
        # Native facts and persisted proof-table facts use the same rule identity.
        merged = {proof_key(f['history']): f for f in tree.facts()}
        for fact in facts:
            key = proof_key(fact['history'])
            old = merged.get(key)
            if old is not None and old['winner'] != fact['winner']:
                raise ValueError('Contradictory graph and stored proofs')
            if old is None or fact['plies'] <= old['plies']:
                merged[key] = fact
        facts = sorted(merged.values(), key=lambda f: len(f['history']))[:2048]
    own = (ProofWorkers(package, workers=SOLVER_WORKERS)
           if proofs is None and prover is not None and (solver_nodes or solver_ms) else None)
    proofs = proofs if own is None else own
    turn = None
    try:
        if solved is None and solver_ms and prover is not None:
            solved = prove(bubble, prover, proofs, history, solver_ms, watch, live, facts, stamps)
            if solved.get('proofs'):
                # The frontier's verified positions settle the follow-up search's edges.
                known = known if known is not None else Proofs()
                known.add(history, dict(proofs=solved['proofs']))
        turn = TurnSearch(bubble, network, history, simulations,
                          solved or solve(prover, history, solver_nodes, watch, facts, ms / 4 if ms else None),
                          trees, q_range_floor, known)
        frontier = proofs if prover is not None and solver_nodes else None
        while (asked := turn.request()) is not None:
            tree, count = asked
            if tree is None:
                turn.take(None)
                if placed and not turn.finished():
                    placed([list(m) for m in turn.moves])
                continue
            turn.tree = tree
            show = None
            if live and not turn.played:
                def show(frame):
                    if (seen := glimpse(frame, turn.player)) is not None:
                        live(seen)
            clock = None
            if end is not None:
                clock = max(1., (end - time.monotonic()) * 1000)
                clock *= .6 if turn.local.remaining == 2 else 1
            [result] = search(bubble, [(tree, max(1, count))], frontier, watch, show, stamps, clock)
            turn.take(result)
            if placed and not turn.finished():
                placed([list(m) for m in turn.moves])
        found = turn.record()
        if solved and 'solver' in solved:
            found['solver'] = solved['solver']
            if solved.get('proofs'):
                found['proofs'] = found.get('proofs', []) + solved['proofs']
        return found
    finally:
        try:
            if turn is not None:
                turn.close()
        finally:
            if own is not None:
                own.close()


def evaluate_many(bubble, provers, histories, simulations, solver_nodes, watch=lambda n: None, q_range_floor=0.,
                  known=None, proofs=None, stamps=False, ms=0):
    """`evaluate` of every position in `histories`, as one pooled job: the solver queries run concurrently, one
    position per prover in `provers` at a time, each distinct position solved once, their proofs added to the
    proof table `known`; then fresh game graphs, one per position, search together with that table in one
    SearchPool (see `search`), stone by stone, sharing network batches, with their proof frontiers on `proofs`
    when `solver_nodes` and `provers` are given. Each result is what `evaluate` would give at that budget with the
    table; with `ms` each turn has that clock as in `evaluate` (root queries at most a quarter, the first of two
    stones 60% of what is left, the last stone the rest), the pooled owners sharing it. `watch` may raise Cancelled."""
    from concurrent.futures import ThreadPoolExecutor
    given = [answered(h, known) for h in histories]
    keys = [position_text(h) for h in histories]
    end = time.monotonic() + ms / 1000 if ms else None
    solved = {}
    if provers and solver_nodes:
        free = queue.Queue()
        for prover in provers:
            free.put(prover)

        def ask(history):
            prover = free.get()
            try:
                # Every wave of root queries ends by the first quarter of the shared clock.
                return solve(prover, history, solver_nodes, watch, known.facts(history) if known is not None else (),
                             max(1., (end - time.monotonic()) * 1000 - .75 * ms) if ms else None)
            finally:
                free.put(prover)

        with ThreadPoolExecutor(len(provers)) as pool:
            unique = {k: h for k, h, a in zip(keys, histories, given) if a is None}
            for k, found in zip(unique, pool.map(ask, unique.values())):
                solved[k] = found
                if known is not None:
                    known.add(unique[k], found)
    network = Watched(bubble.evaluator, watch)
    frontier = proofs if provers and solver_nodes else None
    turns = []
    try:
        for k, history, a in zip(keys, histories, given):
            if a is None:
                turns.append(TurnSearch(bubble, network, history, simulations, solved.get(k) or solve(None, history, 0),
                                        q_range_floor=q_range_floor, known=known))
        while live := [turn for turn in turns if not turn.finished()]:
            if end is not None and any(turn.local.remaining == 2 for turn in live):
                # Under a clock one-stone turns wait for the second phase, so no owner holds the first one past 60%.
                live = [turn for turn in live if turn.local.remaining == 2]
            asked = [(turn, turn.request()) for turn in live]
            searching = []
            for turn, (tree, count) in asked:
                if tree is None:
                    turn.take(None)
                else:
                    turn.tree = tree
                    searching.append((turn, (tree, max(1, count))))
            if searching:
                clock = None
                if end is not None:   # a round is all first stones of two, or all last stones
                    clock = max(1., (end - time.monotonic()) * 1000)
                    clock *= .6 if any(turn.local.remaining == 2 for turn, _ in searching) else 1
                for (turn, _), result in zip(searching, search(bubble, [root for _, root in searching], frontier, watch,
                                                               stamps=stamps, ms=clock)):
                    turn.take(result)
        searched = iter(turns)
        return [a if a is not None else next(searched).record() for a in given]
    finally:
        for turn in turns:
            turn.close()


class SearchChild:
    """A search process and its output lines. It is a TreeProcess, so ending it also ends what it started, like
    Strix's native search; its temporary files go in a private folder, removed when it ends."""

    def __init__(self, command):
        self.folder = tempfile.TemporaryDirectory(prefix='hexo-play-', ignore_cleanup_errors=True)
        env = os.environ | dict.fromkeys(('TMPDIR', 'TEMP', 'TMP'), self.folder.name)
        self.process = TreeProcess(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                   encoding='utf-8', bufsize=1)
        self.lines = queue.Queue()
        self.pump = threading.Thread(target=self.forward, daemon=True)
        self.pump.start()

    def forward(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def end(self):
        """Kill the process and everything it started, reap it and remove its temporary folder."""
        self.process.kill()
        self.process.wait()
        self.pump.join()
        with contextlib.suppress(OSError):
            self.process.stdin.close()
        self.process.stdout.close()
        self.folder.cleanup()


class Engines:
    """Loaded engines, used only from the worker thread. Keeps the three most recent Bubble exports."""

    def __init__(self, device, tactical_package=None, seal=None, *, proof_stamps=True):
        self.proof_stamps = proof_stamps
        self.device, self.tactical_package, self.seal_path = device, tactical_package, seal
        self.bubbles, self.prover, self.prover_build = OrderedDict(), None, None
        self.helpers, self.kept, self.graphs, self.graph_ids = [], None, OrderedDict(), itertools.count(1)
        self.workers = None
        self.external = {}
        self.children = {}
        self.last_turn = {}

    def bubble(self, path, device=None):
        """The loaded export at `path`, reloaded when the file changes."""
        device = device or self.device
        key = (device, *file_identity(path))
        if key not in self.bubbles:
            self.bubbles[key] = Bubble(path, device)
            while len(self.bubbles) > 3:
                self.bubbles.popitem(last=False)
        self.bubbles.move_to_end(key)
        return self.bubbles[key]

    def solver(self):
        """The tactical solver in its own process, so a cancelled query can be ended, and its build; (None, 'none')
        when not built. A rebuilt library replaces the running solver."""
        import tactical_proof
        build = self.solver_build()
        if build == 'none':
            return None, build
        if self.prover is None or self.prover_build != build:
            if self.prover is not None:
                old = self.prover
                old.abort()
                threading.Thread(target=old.close, daemon=True).start()
            package = self.tactical_package or tactical_proof.PACKAGE
            self.prover, self.prover_build = tactical_proof.IsolatedTactics(package, priority='below_normal',
                                                                           **(dict(stamps=True) if self.proof_stamps else {})), build
        return self.prover, build

    def evaluate(self, entry, checkpoint, budget, history, watch, device=None, live=None, keep=False, line=None,
                 known=None, game=None, refresh=None, used=None, placed=None):
        """`evaluate` with the entry's export and the proof table `known`; returns the evaluation, the budget it
        really had (no solver nodes when the solver is not built) and the key of the weights it used (see
        `model_key`). `line` (a seat's game, see `Session.lines`) or `game` (the analysis board's game, see
        `Session.analysis_line`) searches that game's graph (see `game_graph`); without either the turn searches a
        fresh graph. Searches with solver nodes run their proof frontier on this lane's `proof_workers`. With `keep`
        (a deepening tier) the search runs only the simulations the root lacks, and a proof the solver found for the
        position is reused. `refresh`, a saved evaluation of `history`, searches the game graph again with the
        REFRESH_SHARE of that evaluation's simulations a stone and its solver findings; the graph is the one `budget` (the seat's) selects. A seat's
        evaluation or a tier's has a key ending in `:kept`, so continued evaluations are never mistaken for fresh
        ones. `used`, a list, receives the number `game_graph` gave each graph searched, before the search, so a
        caller sees it even when the search is cancelled."""
        bubble = self.bubble(export_path(entry, checkpoint), device)
        solver, build = self.solver() if budget['solver_nodes'] and refresh is None else (None, 'none')
        spent = budget if solver or refresh is not None else budget | dict(solver_nodes=0)
        trees = solved = None
        floor = entry.get('q_range_floor', 0.)
        if refresh is not None:
            build = self.solver_build() if budget['solver_nodes'] else 'none'
            share = max(1, round(REFRESH_SHARE * refresh['simulations']))
            if spent.get('ms'):   # a refresh in time gets the same share of the clock
                spent = spent | dict(ms=max(LIMITS['ms'], round(REFRESH_SHARE * spent['ms'])))
            trees = self.game_graph(bubble, game, build, floor, share, used=used)
            solved = dict(moves=[], pv=[], proof=None, threat=refresh.get('threat') or [], solved=True, used=0)
        elif line is not None or game is not None:
            trees = self.game_graph(bubble, game if line is None else ('seat', line), build, floor, keep=keep, used=used)
        found = evaluate(bubble, solver, history, spent['simulations'], spent['solver_nodes'], watch, live, trees,
                         solved, floor, known, solver_ms=spent.get('solver_ms', 0) if solver else 0,
                         package=self.tactical_package, proofs=self.proof_workers() if solver else None,
                         stamps=self.proof_stamps, ms=spent.get('ms', 0), placed=placed)
        # A budget in time ends its root queries at its clock on purpose: that still counts as spending them.
        if not found.pop('solved') and not spent.get('ms'):
            spent = spent | dict(solver_nodes=0)
        weights = search_key(bubble.sha256[:16], entry, spent)
        kept = ':kept' if keep or line is not None else ''
        return found, spent, f"{weights}:{build if spent['solver_nodes'] else 'none'}{kept}"

    def game_graph(self, bubble, game, build='none', q_range_floor=0., simulations=None, keep=False, used=None):
        """A `TurnSearch` tree source over the GameGraph of `game`: one graph per game, model, solver `build` and
        `q_range_floor`, moved to each stone's position and searched there with the full simulations, or with
        `simulations` when given, or with `keep` only those its root lacks (at least one). The three most recently
        used games (two seats and the analysis board) keep their graphs; a game whose key changed starts a new one.
        `used`, when given, receives the number of the graph searched; each graph built gets a new number."""
        from neural_search import GameGraph
        key = (bubble.sha256, build, q_range_floor)

        def trees(cells, count, network):
            kept = self.graphs.pop(game, None)
            if kept and kept[0] != key:
                kept[1].close()
                kept = None
            if kept:
                graph, ident = kept[1:]
            else:
                graph = GameGraph(network, bubble.sha256, cells, seed=1740, cache=bubble.cache, tactics=True,
                                  q_range_floor=q_range_floor, round_barrier=True)
                ident = next(self.graph_ids)
            self.graphs[game] = key, graph, ident
            if used is not None:
                used.append(ident)
            while len(self.graphs) > 3:
                self.graphs.popitem(last=False)[1][1].close()
            graph.at(cells)
            graph.evaluator = network
            if simulations is not None:
                return graph, simulations
            return graph, max(1, count - int(graph.result(0, 0, 0, 0)['visits'].sum())) if keep else count
        return trees

    def solvers(self, count):
        """Up to `count` tactical workers of one build, and that build: the solver and helpers kept for pooled
        work; ([], 'none') when the solver is not built."""
        import tactical_proof
        prover, build = self.solver()
        if prover is None:
            return [], 'none'
        if any(helper_build != build for _, helper_build in self.helpers):
            for helper, _ in self.helpers:
                helper.abort()
                threading.Thread(target=helper.close, daemon=True).start()
            self.helpers = []
        package = self.tactical_package or tactical_proof.PACKAGE
        while len(self.helpers) < count - 1:
            self.helpers.append((tactical_proof.IsolatedTactics(package, priority='below_normal',
                                                               **(dict(stamps=True) if self.proof_stamps else {})), build))
        return [prover, *(helper for helper, _ in self.helpers[:count - 1])], build

    def evaluate_many(self, entry, checkpoint, budget, histories, watch, device=None, known=None):
        """`evaluate_many` of `histories` with the entry's export and the proof table `known`, its solver queries
        spread over REVIEW_SOLVERS workers and its proof frontiers on `proof_workers`; returns (evaluation, budget it really had, key of the weights) per
        position as `evaluate` does."""
        bubble = self.bubble(export_path(entry, checkpoint), device)
        provers, build = self.solvers(REVIEW_SOLVERS) if budget['solver_nodes'] else ([], 'none')
        spent = budget if provers else budget | dict(solver_nodes=0)
        found = evaluate_many(bubble, provers, histories, spent['simulations'], spent['solver_nodes'], watch,
                              entry.get('q_range_floor', 0.), known, proofs=self.proof_workers() if provers else None,
                              stamps=self.proof_stamps, ms=spent.get('ms', 0))
        out = []
        for record in found:
            used = spent if record.pop('solved') or spent.get('ms') else spent | dict(solver_nodes=0)
            weights = search_key(bubble.sha256[:16], entry, used)
            out.append((record, used, f"{weights}:{build if used['solver_nodes'] else 'none'}"))
        return out

    def effective(self, budget):
        """`budget` as it can run here: no solver work when the tactical library is not built."""
        return budget if self.solver_build() != 'none' else budget | {k: 0 for k in ('solver_nodes', 'solver_ms') if k in budget}

    def proof_workers(self):
        """This lane's SOLVER_WORKERS native proof workers (hybrid_scheduler.ProofWorkers), which every search with
        solver nodes shares, on the tactical build; made on first use and again when the build changes, None when
        the solver is not built."""
        from hybrid_scheduler import ProofWorkers
        build = self.solver_build()
        if build == 'none':
            return None
        if self.workers is None or self.workers[1] != build:
            if self.workers is not None:
                self.workers[0].close()
            self.workers = ProofWorkers(self.tactical_package, workers=SOLVER_WORKERS), build
        return self.workers[0]

    def solver_build(self):
        """The first 8 hex digits of the tactical library's recorded SHA-256, or 'none' when the library or its
        record is missing."""
        import tactical_proof
        package = self.tactical_package or tactical_proof.PACKAGE
        binary = tactical_proof.library(package)
        record = binary.with_name(binary.name + '.json')
        try:
            build = file_build(str(package), file_identity(record), file_identity(binary))
            return build + (':stamps' if self.proof_stamps and build != 'none' else '')
        except OSError:
            return 'none'

    def turn(self, entry, budget, history, stop=lambda: False, checkpoint=None):
        """A turn from a non-Bubble engine, a Six folder's or a Strix entry's at network `checkpoint`. A Six-protocol engine's search
        is stopped when `stop()` turns true.
        Drip, Seal and Strix searches cannot be interrupted in process, so each kind searches in a SearchChild;
        when `stop()` turns true the child and everything it started are killed and a fresh one starts on the next
        turn. Both raise Cancelled."""
        self.last_turn = {}
        kind = entry['kind']
        if kind == 'six':
            return self.protocol(entry, budget, history, stop, checkpoint)
        if kind == 'strix' and not history:
            return [[0, 0]]   # the only legal first stone; Strix searches only once it is on the board
        if kind in self.children and self.children[kind].process.poll() is not None:
            self.children.pop(kind).end()
        if kind not in self.children:
            self.children[kind] = SearchChild([sys.executable, str(Path(__file__).resolve()), 'search', kind])
        child = self.children[kind]
        model = entry['networks'][checkpoint] if entry.get('networks') else entry.get('model')
        request = dict(budget, history=[list(p) for p in history], model=str(model),
                       **{key: str(entry[key]) for key in ('engine', 'library') if entry.get(key)})
        child.process.stdin.write(json.dumps(request) + '\n')
        child.process.stdin.flush()
        while True:
            try:
                line = child.lines.get(timeout=.05)
                break
            except queue.Empty:
                if stop():
                    self.children.pop(kind).end()
                    raise Cancelled() from None
        if line is None:
            raise RuntimeError(f'{kind} search process exited')
        answer = json.loads(line)
        if 'error' in answer:
            raise RuntimeError(answer['error'])
        self.last_turn = answer.get('measurements', {})
        return answer['moves']

    def protocol(self, entry, budget, history, stop, checkpoint=None):
        """A turn from a Six-protocol engine, kept running between turns, one process per command line: a new
        preset reuses the process, a new network or launch `args` starts another. For a Six folder the two most
        recently used networks stay running, so two seats on different networks do not restart each turn; older
        ones are closed."""
        from six_engine import ProtocolError, SixEngine
        command = [*command_of(entry, checkpoint), *budget.get('args', ())]
        key = (*command, entry['mirrored'])
        if key in self.external:
            self.external[key] = self.external.pop(key)
        cancel = self.external[key].cancel if key in self.external else threading.Event()
        cancel.clear()
        done = threading.Event()

        def watch():
            while not done.wait(.05):
                if stop():
                    cancel.set()
                    return

        threading.Thread(target=watch, daemon=True).start()
        game = replay(history)
        try:
            if key not in self.external:
                print(f"{entry['name']}: {shlex.join(command)}", flush=True)
                for old in [k for k in self.external if entry.get('networks') and k[0] == command[0]][:-1]:
                    self.external.pop(old).close()
                self.external[key] = SixEngine(command, mirrored=entry['mirrored'], cwd=entry['cwd'],
                                               path=entry['libraries'], cancel=cancel, log=True,
                                               startup=900)
            moves = self.external[key](game, budget.get('ms'), nodes=budget.get('nodes'))
            self.last_turn = dict(nodes=self.external[key].info.get('nodes'))
            return moves
        except ProtocolError:
            if cancel.is_set():
                raise Cancelled() from None
            raise
        finally:
            done.set()
            game.close()

    def close(self):
        """Release the models and end every child process."""
        self.kept = None
        while self.graphs:
            self.graphs.popitem()[1][1].close()
        for helper, _ in self.helpers:
            helper.close()
        self.helpers = []
        if self.workers is not None:
            self.workers[0].close()
            self.workers = None
        self.bubbles.clear()
        for child in self.children.values():
            child.end()
        self.children.clear()
        if self.prover is not None:
            self.prover.close()
            self.prover = None
        for engine in self.external.values():
            engine.close()
        self.external.clear()


# Saved evaluations


def position_text(history):
    return ' '.join(f'{int(q)},{int(r)}' for q, r in history)


@functools.lru_cache(maxsize=4096)
def position_key(stones):
    """The index key of the position `stones`, a tuple of (q, r) tuples."""
    return hashlib.blake2b(position_text(stones).encode(), digest_size=16).digest()


def clock_spec(clock):
    """A clock request checked and normalized: {"mode": "fixed"}, {"mode": "move", "ms": N} (a complete turn in N ms,
    no banking) or {"mode": "game", "tc": "180+2"} (Absolute, or Fischer with an increment), which carries base_ms and
    increment_ms."""
    clock = clock or dict(mode='fixed')
    mode = clock.get('mode') if isinstance(clock, dict) else None
    if mode == 'fixed':
        return dict(mode='fixed')
    if mode == 'move':
        return dict(mode='move', ms=milliseconds(clock.get('ms'), 'move time', positive=True))
    if mode == 'game':
        return dict(mode='game', **TimeControl.parse(clock.get('tc', clock)).json())
    raise ValueError('Clock mode must be fixed, move or game')


def device_of(entry, device):
    """'GPU' or 'CPU': where the server runs `entry`'s network, a Bubble on the player's `device` and Six on the backend
    its catalogue entry names; engines without a network (Drip, Seal) and other bots' drivers run on the CPU."""
    if entry['kind'] == 'bubble':
        return 'GPU' if str(device).startswith('cuda') else 'CPU'
    if entry['kind'] == 'six' and entry.get('badge', 'six') == 'six':
        return 'CPU' if entry.get('backend', 'CPU') == 'CPU' else 'GPU'
    return 'CPU'


def keeps_clock(entry):
    """Whether an engine plays to a clock through `timed_engine`: Bubble, Drip, Seal and Six itself. Strix's adapter
    and the Six-protocol drivers of other bots (Shrimp) play a fixed budget."""
    return entry['kind'] in ('bubble', 'drip', 'seal') or entry['kind'] == 'six' and entry.get('badge', 'six') == 'six'


def well_formed(record):
    """True for a dict with the fields of a saved evaluation, each of the right shape."""
    def number(v):
        return type(v) in (int, float) and math.isfinite(v)

    def cells(v, size, third=lambda x: type(x) is int):
        """A list of [q, r], [q, r, x] or [q, r, x, n] items with integer coordinates, a third value passing `third`
        and an integer fourth."""
        return isinstance(v, list) and all(isinstance(c, list) and len(c) == size and type(c[0]) is int
                                           and type(c[1]) is int and (size == 2 or third(c[2]))
                                           and (size < 4 or type(c[3]) is int) for c in v)

    if not isinstance(record, dict):
        return False
    proof = record.get('proof')
    return (isinstance(record.get('position'), str) and isinstance(record.get('engine'), str)
            and all(type(record.get(k)) is int for k in ('simulations', 'solver_nodes'))
            and number(record.get('value')) and cells(record.get('moves'), 2)
            and isinstance(record.get('top'), list)
            and all(isinstance(c, list) and 3 <= len(c) <= 5 and cells([c[:3]], 3, number) and all(map(number, c[3:]))
                    for c in record['top'])
            and (cells(record.get('pv', []), 3) or cells(record.get('pv', []), 4)) and cells(record.get('threat', []), 2)
            and isinstance(record.get('proofs', []), list)
            and all(isinstance(f, dict) and cells(f.get('history'), 2) and f.get('winner') in (0, 1)
                    and type(f.get('plies')) is int and f['plies'] > 0 and cells(f.get('pv'), 4)
                    for f in record.get('proofs', []))
            and (proof is None or isinstance(proof, dict) and proof.get('winner') in (0, 1)
                 and type(proof.get('turns')) is int))


def valued(record):
    """True when every top move of a saved evaluation carries its value, or the evaluation is the raw policy,
    which has no value per move."""
    return record['simulations'] == 0 or all(len(move) > 3 for move in record['top'])


class Evaluations:
    """Append-only JSON lines, one evaluation per line, indexed in memory.

    Each line holds `position` (the ordered stones), `engine` (see `model_key`), `simulations`, `solver_nodes`,
    the evaluation fields, `model` (a readable name) and `at`. The index keeps the newest `limit` (position, engine, budget)
    entries; `best` picks the one to show for a position and engine, `proven` lists those holding a proof. The file
    is only appended to; on start a dated copy is kept next to it, the newest `backups` of them."""

    def __init__(self, path=None, limit=200_000, backups=3):
        self.path, self.limit = Path(path) if path else None, limit
        self.lock = threading.Lock()
        self.order, self.by_position, self.proofs = OrderedDict(), {}, {}
        if self.path and self.path.exists():
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            shutil.copyfile(self.path, self.path.with_name(f'{self.path.name}.{stamp}.bak'))
            for old in sorted(self.path.parent.glob(f'{self.path.name}.*.bak'))[:-backups]:
                old.unlink()
            with open(self.path, encoding='utf-8') as lines:
                for line in lines:
                    try:
                        self.index(json.loads(line), line.strip())
                    except (ValueError, KeyError, TypeError):
                        continue
            with open(self.path, 'rb+') as data:
                size = data.seek(0, 2)
                if size:
                    data.seek(size - 1)
                    if data.read(1) != b'\n':
                        data.write(b'\n')

    @staticmethod
    def key(history):
        return position_key(tuple(map(tuple, history)))

    def index(self, record, line):
        """Index one record; ValueError unless `well_formed(record)`."""
        if not well_formed(record):
            raise ValueError('Not an evaluation record')
        position = hashlib.blake2b(record['position'].encode(), digest_size=16).digest()
        budget = (record['simulations'], record['solver_nodes'])
        full = (position, record['engine'], budget)
        self.order.pop(full, None)
        self.order[full] = line
        self.by_position.setdefault((position, record['engine']), set()).add(budget)
        if record.get('proof') or record.get('proofs'):
            self.proofs.setdefault(position, set()).add(full)
        else:
            self.proofs.get(position, set()).discard(full)
        while len(self.order) > self.limit:
            (old, engine, spent), _ = self.order.popitem(last=False)
            budgets = self.by_position[(old, engine)]
            budgets.discard(spent)
            if not budgets:
                del self.by_position[(old, engine)]
            self.proofs.get(old, set()).discard((old, engine, spent))

    def add(self, history, engine, budget, evaluation):
        record = dict(position=position_text(history), engine=engine, simulations=budget['simulations'],
                      solver_nodes=budget['solver_nodes'], **evaluation,
                      at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
        record.update({k: budget[k] for k in ('solver_ms', 'ms') if k in budget})
        with self.lock:
            saved = self.order.get((self.key(history), engine, (budget['simulations'], budget['solver_nodes'])))
            saved, facts = json.loads(saved) if saved else {}, {}
            for fact in [*saved.get('proofs', []), *record.get('proofs', [])]:
                key = proof_key(fact['history'])
                old = facts.get(key)
                if old is None or (fact['plies'], -len(fact['pv'])) < (old['plies'], -len(old['pv'])):
                    facts[key] = fact
            if facts:
                record['proofs'] = list(facts.values())
            line = json.dumps(record, separators=(',', ':'))
            if self.path:
                with open(self.path, 'a', encoding='utf-8') as out:
                    out.write(line + '\n')
            self.index(record, line)
        return record

    def export(self):
        """The whole file, read while no record is being appended."""
        with self.lock:
            return self.path.read_bytes() if self.path and self.path.exists() else b''

    def get(self, history, engine, budget):
        """The saved evaluation of `history` by `engine` at exactly `budget`, or None."""
        with self.lock:
            line = self.order.get((self.key(history), engine, (budget['simulations'], budget['solver_nodes'])))
        return json.loads(line) if line else None

    def covering(self, history, engine, budget):
        """The deepest saved evaluation of `history` by `engine` whose simulations and solver nodes both reach
        `budget`'s and whose top moves carry their values, or None."""
        position, need = self.key(history), (budget['simulations'], budget['solver_nodes'])
        with self.lock:
            enough = sorted((b for b in self.by_position.get((position, engine), ())
                             if b[0] >= need[0] and b[1] >= need[1]), reverse=True)
            lines = [self.order[(position, engine, b)] for b in enough]
        return next((r for r in map(json.loads, lines) if valued(r)), None)

    def proven(self, history):
        """The saved lines of every evaluation of `history` that holds a proof, whatever its engine and budget."""
        with self.lock:
            return [self.order[full] for full in self.proofs.get(self.key(history), ())]

    def best(self, history, engine):
        """The saved evaluation of `history` by `engine` to show, or None: one holding a proof first, since a proof
        is exact, then one whose top moves carry their values, then the most simulations, then the most solver
        nodes."""
        position = self.key(history)
        with self.lock:
            lines = [self.order[(position, engine, b)] for b in self.by_position.get((position, engine), ())]
        found = [json.loads(line) for line in lines]
        return max(found, key=lambda e: (e.get('proof') is not None, valued(e), e['simulations'], e['solver_nodes']),
                   default=None)


# Review


UNGRADED = dict(label=None, before=None, after=None, better=None, line=None)


def review(history, lookup, winner=-1):
    """Label every turn of `history`, and each of its stones, from saved evaluations.

    `lookup(prefix)` returns the evaluation of a position (fields of `evaluate`) or None. A turn is judged on the
    mover's win probability before and after it; the first label that applies wins: win (made six), lost (the
    opponent already had a proven win), kept or missed (the mover had one and kept it, or played the engine's
    stones, or lost it), allowed (handed the opponent one), found (proved one), best (the engine's own stones), then
    the loss bands good (< 0.05), inaccuracy (< 0.10), mistake (< 0.20) and blunder. For inaccuracy and worse,
    missed and allowed, `better` is the engine's stones, as many as were played, and `line` its continuation, unless
    the stones played were the engine's.
    Each turn also has `grades`, one per stone with the same fields judged on that stone alone: the second stone
    from the position after the first, where the engine's stone is its best second stone given the first. A turn
    whose second stone is still to come is listed with label None and its first stone graded. Labels lacking an
    evaluation are None."""
    seen = {}

    def look(prefix):
        key = tuple(map(tuple, prefix))
        if key not in seen:
            seen[key] = lookup(prefix)
        return seen[key]

    def judge(s, e, me):
        stones = [list(p) for p in history[s:e]]
        grade = dict(UNGRADED)
        if e == len(history) and winner == me:
            grade['label'] = 'win'
            return grade
        before, after = look(history[:s]), look(history[:e])
        if before is None or after is None:
            return grade
        grade['before'] = before['value']
        grade['after'] = after['value'] if player_at(e) == me else 1 - after['value']
        had = (before.get('proof') or {}).get('winner')
        has = (after.get('proof') or {}).get('winner')
        loss = grade['before'] - grade['after']
        engine = [tuple(m) for m in before['moves']]
        played_best = bool(engine) and all(tuple(p) in engine for p in stones)
        if had == 1 - me:
            label = 'lost'
        elif had == me:
            label = 'kept' if has == me or played_best else 'missed'
        elif has == 1 - me:
            label = 'allowed'
        elif has == me:
            label = 'found'
        elif played_best:
            label = 'best'
        else:
            label = 'good' if loss < .05 else 'inaccuracy' if loss < .1 else 'mistake' if loss < .2 else 'blunder'
        grade['label'] = label
        if label in ('inaccuracy', 'mistake', 'blunder', 'missed', 'allowed') and engine and not played_best:
            grade['better'] = before['moves'][:len(stones)]
            if before.get('pv'):
                grade['line'] = before['pv']
            else:
                reply = look([*history[:s], *engine])
                grade['line'] = [[*p, me] for p in before['moves']] + \
                                [[*p, 1 - me] for p in (reply or {}).get('moves', [])]
        return grade

    turns = []
    starts = turn_starts(len(history))
    for s, e in zip(starts, [*starts[1:], len(history)]):
        if s == e:
            break
        me = player_at(s)
        complete = e - s == (1 if s == 0 else 2) or e == len(history) and winner == me
        turns.append(dict(ply=s, player=me, stones=[list(p) for p in history[s:e]],
                          **(judge(s, e, me) if complete else UNGRADED), grades=[judge(p, p + 1, me) for p in range(s, e)]))
    return turns


# Session and jobs


def pair_elo(results):
    """A's Elo over B from the colour-swapped pairs among `results` (game, winner 0 for A, 1 for B, None for a
    capped game): {a_minus_b, interval (95%), pairs}, or None before the first complete pair."""
    pairs = [0] * 5
    ordered = sorted(results, key=lambda r: r['game'])
    for first, second in zip(ordered[::2], ordered[1::2]):
        points = sum(1 if r['winner'] == 0 else .5 if r['winner'] is None else 0 for r in (first, second))
        pairs[round(points * 2)] += 1
    if not sum(pairs):
        return None
    from dense_posterior import Posterior
    elo, sd = Posterior(['A', 'B'], 'B', [('A', 'B', pairs)], matchup_prior=0).difference('A', 'B', matchup=False)
    return dict(a_minus_b=elo, interval=[elo - 1.96 * sd, elo + 1.96 * sd], pairs=sum(pairs))


def lane(job):
    """The worker that runs `job`: moves on one, analysis and review on the other."""
    return 'move' if job.kind == 'move' else 'analysis'


class Job:
    """One unit of engine work. `kind` is move, analyse or review; progress is `done` of `total`. A move's `placed`
    holds the stones of its turn already on the board (see `Session.place`)."""
    ids = itertools.count(1)

    def __init__(self, kind, priority, history, **fields):
        self.id, self.kind, self.priority, self.history = next(Job.ids), kind, priority, tuple(history)
        self.status, self.done, self.total, self.error, self.cancelled, self.ended = 'queued', 0, 1, None, False, None
        self.placed = ()
        self.__dict__.update(fields)

    def reached(self):
        """The position the job's board stands at: its history and the stones it has placed."""
        return self.history + self.placed

    def summary(self):
        return dict(id=self.id, kind=self.kind, status=self.status, done=self.done, total=self.total,
                    error=self.error, ply=len(self.history), side=getattr(self, 'side', None),
                    live=getattr(self, 'live', None))

    def is_set(self):
        return self.cancelled


def custom_form(kind, standard, custom):
    """The custom budget form of a Bubble or a Six (EXCLUSIVE kinds): both amounts, the work (Bubble's `simulations`,
    shown as Nodes; Six's `nodes`, shown as Positions) and Time `ms`, with `active` naming the one that applies.
    Missing values come from the `standard` preset and 1000 ms. Without `active`, Time applies when only it is given
    and the work otherwise, so a custom budget saved before the form keeps its work; a Bubble's old `solver_nodes`
    and `views` are dropped (every Bubble search has HYBRID['views'] views)."""
    work, _ = EXCLUSIVE[kind]
    if custom is not None and not isinstance(custom, dict):
        raise ValueError('A custom budget is an object of numbers')
    custom = {k: v for k, v in (custom or {}).items() if not (kind == 'bubble' and k in ('solver_nodes', 'views'))}
    form = {work: standard[work], 'ms': 1000, 'active': 'ms' if 'ms' in custom and work not in custom else work}
    for key, value in custom.items():
        if key == 'args':
            continue
        if key == 'active':
            if value not in EXCLUSIVE[kind]:
                raise ValueError(f'active must be one of {", ".join(EXCLUSIVE[kind])}')
        elif key not in form:
            raise ValueError(f'{key} is not a budget of this engine')
        elif type(value) is not int or not LIMITS[key] <= value <= MAX_BUDGET:
            raise ValueError(f"{key} must be an integer from {LIMITS[key]} to {MAX_BUDGET}")
        form[key] = value
    return form


def form_kind(entry):
    """'bubble' or 'six' for an engine whose custom budget is the exclusive form (see `custom_form`): Bubble, and Six
    itself but not another bot behind Six's protocol, like Shrimp; None otherwise."""
    six = entry['kind'] == 'six' and entry.get('badge', 'six') == 'six'
    return entry['kind'] if entry['kind'] == 'bubble' or six else None


def custom_budget(kind, form, standard=None):
    """The budget a custom `form` (see `custom_form`) runs: its active amount, and for Bubble its root query
    nodes, sixteen per simulation or four per ms between 1,024 and 4,000,000. A Time budget keeps a work
    ceiling: Bubble's `simulations` of 65,536 per stone, Six's `nodes` of MAX_BUDGET. A Six keeps the `standard`
    preset's launch `args`."""
    active = form['active']
    if kind == 'six':
        launch = {'args': standard['args']} if standard and 'args' in standard else {}
        return (dict(ms=form['ms'], nodes=MAX_BUDGET) if active == 'ms' else dict(nodes=form['nodes'])) | launch
    if active == 'ms':
        return dict(simulations=65536, ms=form['ms'], solver_nodes=min(4_000_000, max(1024, 4 * form['ms'])))
    return dict(simulations=form['simulations'],
                solver_nodes=min(4_000_000, max(1024, 16 * form['simulations'])) if form['simulations'] else 0)


def budget_of(presets, preset, custom=None, kind=None, form=None):
    """The budget of `preset` from an entry's `presets`, or the standard budget with `custom` values checked
    against the smallest values in LIMITS and the `kind`'s own (larger budgets only take longer) (a preset's launch `args` are not custom).
    With `form` (by default for EXCLUSIVE kinds; see `form_kind`) the custom budget is the one its form selects (see
    `custom_form`). A Bubble's 'solver' is the analysis solver preset SOLVER."""
    limits = LIMITS | KIND_LIMITS.get(kind, {})
    if preset == 'solver' and kind == 'bubble':
        return dict(SOLVER)
    if preset != 'custom':
        if preset not in presets:
            raise ValueError('Unknown preset')
        return dict(presets[preset])
    if form if form is not None else kind in EXCLUSIVE:
        return custom_budget(kind, custom_form(kind, presets['standard'], custom), presets['standard'])
    if custom is not None and not isinstance(custom, dict):
        raise ValueError('A custom budget is an object of numbers')
    budget = dict(presets['standard'])
    for key, value in (custom or {}).items():
        if key == 'args':
            continue
        if (key not in budget and key not in KIND_LIMITS.get(kind, {})) or key not in limits:
            raise ValueError(f'{key} is not a budget of this engine')
        if type(value) is not int or not limits[key] <= value <= MAX_BUDGET:
            raise ValueError(f'{key} must be an integer from {limits[key]} to {MAX_BUDGET}')
        budget[key] = value
    return budget


class Session:
    """The game, the seats, the analysis settings and the job queues. HTTP threads call the public methods; two
    worker threads run the jobs: engine moves through `engines`, analysis and review through `analysis_engines`
    (`engines` when not given), so analysis keeps up while engines play. `revision` grows with every change the
    page must redraw. `proofs` (`Proofs`) indexes the proven positions of the game's saved evaluations and of their
    lines; analysis and review search with it, and the page is shown what it proves (see `proven`)."""

    def __init__(self, entries, engines, store, rescan=lambda: None, book=None, archive=None, study_store=None,
                 analysis_engines=None, save_initial=True):
        self.entries, self.engines, self.store, self.rescan_entries = entries, engines, store, rescan
        self.analysis_engines = analysis_engines or engines
        self.book = book
        self.archive = Path(archive) if archive else None
        self.study_store = study_store
        self.saved_matches, self.study, self.saved_game = {}, None, None
        self.freeplay_directory, self.freeplay_signature = None, None
        self.freeplay_created = None
        self.freeplay_records = {}
        self.models_folder, self.opening_book, self.book_mode, self.opening = None, bool(book), 'narrow', None
        self.coverage = Coverage(Path(store.path).with_name('play-openings.jsonl') if store.path else None)
        self.lock, self.rescanning = threading.Condition(), threading.Lock()
        self.history, self.revision, self.paused = [], 0, False
        self.proofs = Proofs()
        self.line_ids = itertools.count()
        self.lines = [next(self.line_ids), next(self.line_ids)]
        self.analysis_line, self.analysis_graph, self.graph_plies = next(self.line_ids), None, {}
        self.instance, self.closing = os.urandom(4).hex(), False
        self.match, self.match_worker = None, None
        self.game_clock, self.timed_engines = None, []
        self.clock_spec, self.seat_engines, self.clock_preparing = dict(mode='fixed'), [None, None], None
        self.preparing_sides = set()
        self.outcome, self.clock_turns, self.notice, self.clock_partial_ms = None, [], None, 0
        self.match_file = None
        self.retries = {}
        self.paces = {}   # engine id -> {preset: ms per two-stone turn} measured from its moves (see `note_pace`)
        # The position and analysis settings whose analysis was cancelled (see `dismissal`), or None.
        self.dismissed = None
        self.jobs, self.queues, self.order = OrderedDict(), dict(move=[], analysis=[]), itertools.count()
        bubble = next((e for e in entries.values() if e['kind'] == 'bubble'), None)
        opponent = bubble or entries['drip:Drip']
        self.seats = [dict(engine='human'), self.seat(opponent['id'], None, 'standard')]
        self.analysis = self.seat(bubble['id'], None, 'standard') | dict(auto=True) if bubble else None
        if save_initial:
            self.save_freeplay()
        self.workers = [threading.Thread(target=self.work, args=(lane,), daemon=True) for lane in self.queues]
        for worker in self.workers:
            worker.start()

    def seat(self, engine, checkpoint, preset, custom=None):
        if engine == 'human':
            return dict(engine='human')
        entry = self.entries.get(engine)
        if entry is None:
            raise ValueError('Unknown engine')
        if entry.get('checkpoints'):
            checkpoint = entry['checkpoints'][0] if checkpoint is None else checkpoint
            if checkpoint not in entry['checkpoints']:
                raise ValueError('Unknown checkpoint')
        else:
            checkpoint = None
        budget = budget_of(entry['presets'], preset, custom, entry['kind'], form_kind(entry) is not None)
        if preset == 'custom' and form_kind(entry):
            form = custom_form(entry['kind'], entry['presets']['standard'], custom)
            return dict(engine=engine, checkpoint=checkpoint, preset=preset, budget=budget, custom=form)
        return dict(engine=engine, checkpoint=checkpoint, preset=preset, budget=budget)

    def new_lines(self, *sides):
        """Give `sides` (both when none) a new line: the game graph their Bubble seats search (see
        `Engines.game_graph`). Undo, a new or loaded game and a seat change start one; giving both sides one also
        starts a new `analysis_line`, the graph analysis searches."""
        for side in sides or (0, 1):
            self.lines[side] = next(self.line_ids)
        if not sides:
            self.analysis_line, self.analysis_graph = next(self.line_ids), None

    def engine_key(self, seat):
        """Evaluations are keyed by the weights and the solver build that produced them ('none' for a budget
        without solver nodes); None when the weights file is gone."""
        try:
            entry = self.entries[seat['engine']]
            budget = self.engines.effective(seat['budget'])
            weights = search_key(model_key(export_path(entry, seat['checkpoint'])), entry, budget)
        except (OSError, KeyError):
            return None
        return f"{weights}:{self.engines.solver_build() if budget['solver_nodes'] else 'none'}"

    # Reading

    def models(self):
        with self.lock:
            return [{k: e[k] for k in SHOWN if k in e} | dict(clocks=keeps_clock(e)) for e in self.entries.values()]

    def analysis_keys(self):
        """The store keys of the analysis model's fresh and kept-tree evaluations, with and without solver checks."""
        key = self.engine_key(self.analysis) if self.analysis else None
        if key is None:
            return ()
        bare = key.split(':')[0] + ':none'
        return tuple(dict.fromkeys((key, key + ':kept', bare, bare + ':kept')))

    def lookup(self, history, keys=None):
        """The evaluation of `history` to show: the best of those under `keys` (`analysis_keys()` when None), see
        `Evaluations.best`."""
        found = [e for k in (self.analysis_keys() if keys is None else keys) if (e := self.store.best(history, k))]
        return max(found, key=lambda e: (e.get('proof') is not None, valued(e), e['simulations'], e['solver_nodes']),
                   default=None)

    def review_seat(self):
        """The analysis engine, checkpoint and strength (its preset, or its custom budget): a review uses it for every
        position, so each verdict compares evaluations made with one budget. The solver preset reviews at Standard:
        a review is a pooled search, not a proof hunt per position."""
        if not self.analysis:
            return None
        a = self.analysis
        preset = 'standard' if a['preset'] == 'solver' else a['preset']
        return self.seat(a['engine'], a['checkpoint'], preset, a.get('custom', a['budget']) if preset == 'custom' else None)

    def review_target(self):
        """(store key, budget) of the review evaluations, the key None without an analysis model."""
        seat = self.review_seat()
        return (self.engine_key(seat), self.engines.effective(seat['budget'])) if seat else (None, None)

    def review_lookup(self, history, target=None):
        """The evaluation of `history` at exactly the review budget (`target` as `review_target()`), or None."""
        key, budget = self.review_target() if target is None else target
        return self.store.get(history, key, budget) if key else None

    def proven(self, history, found, played=None):
        """`found` (an evaluation, or None) with what the game's proof table proves of `history`: when `found` has
        no proof, the position's own proof, its value (1 or 0 for the side to move) and its line; an existing proof's
        line extended from stored children; and each stone
        to a proven position as a top row marked won or lost, proven wins first, the shortest leading and among
        equals the stone `played` next in the game, losses last. A position the table proves without an evaluation
        gets one with no simulations; None when there is neither."""
        outcome, edges = self.proofs.known(history), self.proofs.edges(history)
        if found is None and outcome is None:
            return None
        mover = player_at(len(history))
        shown = dict(found or dict(moves=[], top=[], threat=[], simulations=0, solver_nodes=0))
        if outcome is not None and (not shown.get('proof') or
                outcome['winner'] == shown['proof']['winner'] and outcome['plies'] < proof_plies(shown['proof'], history)):
            game = replay(history)
            try:
                remaining = game.remaining
            finally:
                game.close()
            won = outcome['winner'] == mover
            shown.update(value=1. if won else 0., pv=outcome['pv'], proof=dict(
                winner=outcome['winner'], plies=outcome['plies'], turns=proof_turns(outcome['plies'], remaining, won)))
            if won:
                moves = [p[:2] for i, p in enumerate(outcome['pv'][:remaining]) if p[2] == mover and p[3] == i + 1]
                shown['moves'] = moves if len(moves) == remaining or outcome['plies'] <= len(moves) else []
        if shown.get('proof'):
            shown['pv'] = self.proofs.line(history, dict(winner=shown['proof']['winner'],
                                                        plies=proof_plies(shown['proof'], history), pv=shown.get('pv') or []))
        lost = shown.get('proof') and shown['proof']['winner'] != mover
        remaining = 2 if len(history) % 2 else 1
        turn = [p[:2] for i, p in enumerate((shown.get('pv') or [])[:remaining])
                if len(p) == 4 and p[2] == mover and p[3] == i + 1] if shown.get('proof') else []
        if len(turn) == remaining:
            shown['moves'] = turn
        defence = turn if lost else []
        rows = [list(row) for row in shown.get('top') or []]
        for action, (winner, distance, _) in edges.items():
            row = next((row for row in rows if tuple(row[:2]) == action), None)
            if row is None and winner == mover:
                rows.append(row := [*action, 0.])
            if row is not None:
                row[3:] = [1., 1] if winner == mover else [0., -1]
        if lost:
            for row in rows:
                row[3:] = [0., -1]
            if defence:
                row = next((r for r in rows if r[:2] == defence[0]), None)
                if row is not None:
                    rows.remove(row)
                rows.insert(0, row if row is not None else [*defence[0], 0., 0., -1])
        flag = lambda row: row[4] if len(row) > 4 else 0
        rows.sort(key=lambda row: (1 - flag(row), edges.get(tuple(row[:2]), (0, math.inf))[1] if flag(row) > 0 else 0,
                                   flag(row) > 0 and played is not None and tuple(row[:2]) != tuple(played)))
        shown['top'] = rows[:5]
        shown['refuted'] = len(shown['top']) if shown['top'] and all(flag(r) < 0 for r in shown['top']) and not shown.get('proof') else 0
        if shown['refuted'] and shown.get('node_value') is not None:
            shown['value'] = shown['node_value']
        return shown

    def state(self):
        with self.lock:
            self.check_time()
            history, game = list(self.history), replay(self.history)
            try:
                board = dict(player=game.player, remaining=game.remaining, winner=game.winner)
            finally:
                game.close()
            if board['winner'] < 0 and self.outcome:
                board['winner'] = self.outcome['winner']
            evaluations, stale, keys, target = {}, [], self.analysis_keys(), self.review_target()
            for ply in range(len(history) + 1):
                played = history[ply] if ply < len(history) else None
                if (found := self.proven(history[:ply], self.lookup(history[:ply], keys), played)) is not None:
                    evaluations[ply] = {k: found.get(k) for k in
                                        ('value', 'node_value', 'moves', 'top', 'proof', 'pv', 'threat', 'simulations', 'solver_nodes', 'solver_ms', 'ms', 'refuted', 'solver')}
                    if self.stale(found, ply):
                        stale.append(ply)
            device = getattr(self.engines, 'device', 'cpu')
            entries = [{k: e[k] for k in SHOWN if k in e} | dict(clocks=keeps_clock(e), device='Server ' + device_of(e, device))
                       | (dict(pace=TURN_MS[device_of(e, device)] | self.paces.get(e['id'], {})) if e['kind'] == 'bubble' else {})
                       for e in self.entries.values()]
            return dict(instance=self.instance, revision=self.revision, history=[list(p) for p in history], **board,
                        paused=self.paused, seats=self.seats, analysis=self.analysis, engines=entries,
                        match={k: v for k, v in self.match.items() if k not in ('results', 'openings', 'opening_selection')}
                        if self.match else None,
                        clock=self.game_clock.json() if self.game_clock else None,
                        clock_spec=self.match['clock'] if self.match else self.clock_spec,
                        clock_preparing=self.clock_preparing is not None, outcome=self.outcome, notice=self.notice,
                        saved_game=self.saved_game, models_folder=self.models_folder,
                        book=dict(available=bool(self.book), enabled=self.opening_book, mode=self.book_mode,
                                  opening=self.opening),
                        evaluations=evaluations, stale=stale,
                        review=review(history, lambda h: self.proven(h, self.review_lookup(h, target)), board['winner']),
                        review_preset=('standard' if self.analysis['preset'] == 'solver' else self.analysis['preset']) if self.analysis else None,
                        jobs=self.job_list())

    def job_list(self):
        """Queued and running jobs, then jobs that failed in the last ten seconds with their `error`; a job's live
        search is shown only while its position is still on the board. Analysis failures belong to their settings
        and position, so a late failure cannot stop analysis of a newly selected configuration."""
        now, current = time.time(), tuple(self.history)
        shown = []
        for job in self.jobs.values():
            same_position = job.history == current[:len(job.history)]
            if job.status == 'failed':
                if now - job.ended >= 10:
                    continue
                if job.kind == 'analyse' and (not same_position or any(
                        job.seat.get(key) != (self.analysis or {}).get(key)
                        for key in ('engine', 'checkpoint', 'preset', 'budget', 'device'))):
                    continue
            elif job.status not in ('queued', 'running'):
                continue
            shown.append(job.summary() | ({} if same_position else dict(live=None)))
        return shown

    def poll(self, since):
        with self.lock:
            self.check_time()
            if since == self.revision:
                return dict(instance=self.instance, revision=self.revision, jobs=self.job_list(),
                            **(dict(clock=self.game_clock.json()) if self.game_clock else {}))
        return self.state()

    # Changing

    def changed(self):
        """Call with the lock held after any change: bumps the revision and queues the jobs the change calls for."""
        self.revision += 1
        self.proofs.extend(self.store, self.history)
        game = replay(self.history)
        try:
            player, winner, remaining = game.player, game.winner, game.remaining
        finally:
            game.close()
        if winner < 0 and self.outcome and not self.match:
            winner = self.outcome['winner']
        seat = self.seats[player]
        busy = any(j.kind == 'move' and j.status in ('queued', 'running') and not j.cancelled
                   and j.reached() == tuple(self.history) for j in self.jobs.values())
        waiting = self.match and self.match['active'] and (self.match['between'] or self.match['preparing'] or
                                                          self.match.get('outcome') or
                                                          self.match['max_placements'] and len(self.history) >= self.match['max_placements'])
        freeplay = self.game_clock is not None and not self.match and not self.saved_game
        waiting = waiting or freeplay and (self.clock_preparing is not None or self.outcome is not None)
        if freeplay and winner < 0 and not self.paused and not waiting and self.game_clock.running is None:
            self.game_clock.start(player)
        if winner < 0 and not self.paused and seat['engine'] != 'human' and not busy and not waiting:
            if self.game_clock and self.match and self.match['active'] and self.game_clock.running is None:
                self.game_clock.start(player)
            self.submit(Job('move', 1, self.history, side=player, seat=dict(seat), line=self.lines[player]))
        for job in self.jobs.values():
            stale = job.history != tuple(self.history[:len(job.history)])
            if job.kind in ('analyse', 'review') and job.status in ('queued', 'running') and stale:
                job.cancelled = True
                if job.status == 'queued':
                    job.status = 'cancelled'
        opening = not self.history or remaining == 2
        if self.dismissed is not None and self.dismissed != self.dismissal(self.history):
            self.dismissed = None
        if (self.analysis and self.analysis['auto'] and winner < 0 and opening and not self.deepening(winner)
                and self.dismissed is None):
            self.request_analysis(self.history, 1)
        self.deepen(winner)
        self.save_freeplay()
        self.lock.notify_all()

    def deepening(self, winner):
        """True while the current position deepens (see `deepen`): Auto is on at a preset (not custom, not the
        solver), an engine seat plays, the game is neither paused, finished nor a batch. The configured analysis
        then comes from the deepening, so the faster presets land first."""
        return bool(self.analysis and self.analysis['auto'] and self.analysis['preset'] in PRESET_NAMES
                    and not self.paused and winner < 0
                    and any(seat['engine'] != 'human' for seat in self.seats)
                    and not (self.match and self.match['active']))

    def deepen(self, winner):
        """While an engine seat plays, the game is not paused and Auto is on, evaluate the current position with the analysis
        model at each preset in turn, fastest first, up to the analysis preset: the first preset not yet saved is
        queued at the lowest priority, and `changed` queues the next when it lands. Deepening of other positions, and all of it once
        deepening stops, is dropped; a preset that failed for this position, or whose solver gave no verdict, is
        skipped."""
        current, active = tuple(self.history), self.deepening(winner)
        for job in self.jobs.values():
            if hasattr(job, 'tier') and job.status in ('queued', 'running') and (job.history != current or not active):
                job.cancelled = True
                if job.status == 'queued':
                    job.status = 'cancelled'
        busy = any(hasattr(job, 'tier') and job.history == current and job.status in ('queued', 'running')
                   for job in self.jobs.values())
        if not active or busy or self.dismissed == self.dismissal(current):
            return
        for tier in PRESET_NAMES[:PRESET_NAMES.index(self.analysis['preset']) + 1]:
            failed = any(getattr(job, 'tier', None) == tier and job.history == current
                         and (job.status == 'failed' or getattr(job, 'incomplete', False))
                         for job in self.jobs.values())
            seat = self.seat(self.analysis['engine'], self.analysis['checkpoint'], tier)
            key, budget = self.engine_key(seat), self.engines.effective(seat['budget'])
            if key and not failed and not self.store.covering(current, key + ':kept', budget):
                self.submit(Job('analyse', 3, current, seat=seat, force=False, tier=tier, game=self.analysis_line))
                return

    def dismissal(self, history):
        """The key a cancelled analysis of `history` leaves in `dismissed`: the position and the analysis settings
        without Auto. While it matches, Auto and deepening do not queue that position again; any move, settings
        change or analysis request clears it."""
        settings = {k: v for k, v in (self.analysis or {}).items() if k != 'auto'}
        return tuple(map(tuple, history)), json.dumps(settings, sort_keys=True)

    def submit(self, job):
        self.jobs[job.id] = job
        self.revision += 1
        heapq.heappush(self.queues[lane(job)], (job.priority, next(self.order), job))
        while len(self.jobs) > 200:
            oldest = next(iter(self.jobs))
            if self.jobs[oldest].status in ('queued', 'running'):
                break
            del self.jobs[oldest]
        self.lock.notify_all()
        return job

    def request_analysis(self, history, priority, force=False):
        """Queue an evaluation of `history` by the analysis engine unless it is saved or already queued at the same
        or a more urgent `priority` (lower runs first)."""
        settings = dict(self.analysis)
        game = replay(history)
        try:
            if game.winner >= 0:
                return None
        finally:
            game.close()
        for job in self.jobs.values():
            if job.kind == 'analyse' and job.history == tuple(history) and job.seat == settings:
                if job.status == 'queued' and job.priority > priority:
                    job.cancelled, job.status = True, 'cancelled'
                elif job.status in ('queued', 'running'):
                    return job
                if job.status in ('failed', 'done') and not force and time.time() - job.ended < 30:
                    return None
        key, budget = self.engine_key(settings), self.engines.effective(settings['budget'])
        if key is None or not force and self.store.covering(history, key, budget):
            return None
        return self.submit(Job('analyse', priority, history, seat=settings, force=force, game=self.analysis_line))

    def play(self, q, r):
        """Place a person's stone; placing one resumes a paused game, so the engine seat answers it."""
        with self.lock:
            self.match_editable()
            self.check_time()
            game = replay(self.history)
            try:
                side = game.player
                if game.winner >= 0 or self.outcome and not self.match:
                    raise ValueError('The game has finished')
                if self.seats[side]['engine'] != 'human':
                    raise ValueError('It is not your turn')
                game.play(q, r)
                done = game.winner >= 0 or game.player != side
            finally:
                game.close()
            if done and self.game_clock is not None and not self.match and self.game_clock.running == side                     and self.game_clock.expired():
                self.check_time()
                raise ValueError('The game has finished')
            self.fork_freeplay()
            self.history.append((q, r))
            if done and self.game_clock is not None and not self.match and self.game_clock.running == side:
                self.clock_turn(side)
            self.paused = False
            self.changed()

    def people(self, people=None):
        """The sides people play: `people` (a list of sides) when given, for a page that plays a seat itself (the
        browser engines), otherwise the human seats."""
        if people is None:
            return [i for i, seat in enumerate(self.seats) if seat['engine'] == 'human']
        if not isinstance(people, list) or any(side not in (0, 1) or type(side) is not int for side in people):
            raise ValueError('People must be a list of sides')
        return people

    def undo(self, people=None):
        """Take back stones to the start of the latest turn a person played (`people` as for `people`), or one
        stone without people."""
        with self.lock:
            people = self.people(people)
            self.match_editable()
            if not self.history:
                return
            self.fork_freeplay()
            self.history.pop()
            while people and self.history and not (player_at(len(self.history)) in people
                                                   and len(self.history) in turn_starts(len(self.history) + 1)):
                self.history.pop()
            self.stop_moves()
            self.reset_clock()
            self.new_lines()
            self.changed()

    def new_game(self, people=None):
        """Start again; with the book on, from one of its openings (`book_mode`: narrow, wide or all, as for
        batches) in a random orientation, shown as played stones. Against a person (`people` as for `people`) the
        opening comes from `pick_opening` over what that person has played on that side (`Coverage`), otherwise
        uniformly."""
        history, opening, played = [], None, None
        people = self.people(people)
        if self.opening_book and self.book:
            rng = random.Random()
            selection = book_openings(self.book, self.book_mode, None, rng.randrange(1 << 30))
            if len(people) == 1:
                book = selection['sha256'][:16]
                node = pick_opening(selection['nodes'], lambda key: self.coverage.count(book, key, people[0]), rng)
                played = (book, node['key'], people[0])
            else:
                node = rng.choice(selection['nodes'])
            history, opening = node['moves'], dict(mode=self.book_mode, key=node['key'])
        with self.lock:
            self.load(history, False)
            self.opening = opening
            self.save_freeplay()
            if played:
                self.coverage.add(*played)

    def use_book(self, enabled, mode=None, people=None):
        """Turn book openings on or off and choose the book (`BOOKS`); an empty board starts from one at once
        (`new_game` with `people`)."""
        if type(enabled) is not bool or mode not in (None, *BOOKS):
            raise ValueError('enabled must be true or false and the book narrow, wide or all')
        if enabled and not self.book:
            raise ValueError('No opening book; start the player with --dense-run or --book')
        with self.lock:
            self.opening_book, self.book_mode = enabled, mode or self.book_mode
            self.revision += 1
            if enabled and not self.history:
                self.new_game(people)

    def load(self, history, paused, saved_game=None):
        """Replace the game with `history` (validated)."""
        replay(history).close()
        with self.lock:
            self.match_editable()
            if saved_game is None or self.freeplay_directory is not None:
                self.save_freeplay()
            self.freeplay_directory, self.freeplay_signature = None, None
            self.freeplay_records = {}
            self.history, self.paused, self.opening = [tuple(map(int, p)) for p in history], paused, None
            leaving, self.match = self.match is not None, None
            self.proofs = Proofs()
            self.saved_game = saved_game
            self.reset_clock()
            if leaving:
                self.prepare_timed()
            self.stop_moves()
            self.new_lines()
            self.changed()

    def stop_moves(self, side=None):
        """Cancel queued and running engine moves, of one `side` or of both."""
        for job in self.jobs.values():
            if job.kind == 'move' and job.status in ('queued', 'running') and side in (None, job.side):
                job.cancelled = True
                if job.status == 'queued':
                    job.status = 'cancelled'

    def configure_seat(self, side, engine, checkpoint=None, preset='standard', custom=None):
        with self.lock:
            self.match_editable()
            if preset == 'solver':
                raise ValueError('The solver preset is for analysis')
            seat = self.seat(engine, checkpoint, preset, custom)
            if self.clock_spec['mode'] != 'fixed' and not self.match and seat['engine'] != 'human':
                self.timed_config(seat)
            self.seats[side] = seat
            self.stop_moves(side)
            self.new_lines(side)
            if not self.match:
                self.prepare_timed([side])
            self.changed()

    def configure_analysis(self, engine, checkpoint=None, preset='standard', custom=None, auto=True):
        with self.lock:
            seat = self.seat(engine, checkpoint, preset, custom)
            if self.entries[engine]['kind'] != 'bubble' or type(auto) is not bool:
                raise ValueError('Analysis needs a Bubble model')
            self.analysis = seat | dict(auto=auto)
            self.stop_analysis()
            self.changed()

    def stop_analysis(self):
        """Cancel analysis jobs made for other analysis settings than the current ones, and review and deepening
        jobs made for another analysis model."""
        current = {k: v for k, v in (self.analysis or {}).items() if k != 'auto'}
        for job in self.jobs.values():
            settings = {k: v for k, v in getattr(job, 'seat', {}).items() if k != 'auto'}
            if job.kind == 'review' or hasattr(job, 'tier'):
                stale = (settings.get('engine'), settings.get('checkpoint')) != (current.get('engine'), current.get('checkpoint'))
            else:
                stale = settings != current
            if job.kind in ('analyse', 'review') and job.status in ('queued', 'running') and stale:
                job.cancelled = True
                if job.status == 'queued':
                    job.status = 'cancelled'

    def analyse(self, ply, force=False):
        with self.lock:
            if not self.analysis or not 0 <= ply <= len(self.history):
                raise ValueError('Nothing to analyse')
            self.dismissed = None
            for job in self.jobs.values():
                stale = job.kind == 'analyse' and job.priority == 0 and job.history != tuple(self.history[:ply])
                if stale and job.status in ('queued', 'running'):
                    job.cancelled = True
                    if job.status == 'queued':
                        job.status = 'cancelled'
            game = replay(self.history)
            try:
                winner = game.winner
                deepening = ply == len(self.history) and self.deepening(winner)
            finally:
                game.close()
            if deepening:
                self.deepen(winner)
            history = self.history[:ply]
            saved = self.lookup(history)
            if not force and self.stale(saved, ply) and not saved.get('proof'):
                # A position viewed again: search the analysis graph there again, from what it holds now.
                job = self.submit(Job('analyse', 0, history, seat=dict(self.analysis), force=True,
                                      game=self.analysis_line, refresh=saved))
            else:
                job = None if deepening and not force else self.request_analysis(history, 0, force)
            self.lock.notify_all()
            return job.id if job else None

    def review_game(self, history=None, tries=0):
        """Queue a review of `history` (the game by default); a review whose solver checks failed queues itself
        again half a minute later, up to three times."""
        with self.lock:
            history = self.history if history is None else history
            if not self.analysis or tuple(history) != tuple(self.history[:len(history)]):
                raise ValueError('Review needs a Bubble model and a position of this game')
            seat = self.review_seat()
            for job in self.jobs.values():
                if job.kind == 'review' and job.status in ('queued', 'running') and job.history == tuple(history) \
                        and job.seat == seat:
                    return job.id
            job = self.submit(Job('review', 2, history, seat=seat, tries=tries))
            job.total = len(review_plies(history))
            return job.id

    def review_again(self, history, seat, tries):
        """Queue the review of `history` again when the review settings are still `seat`."""
        with self.lock:
            if self.review_seat() == seat:
                with contextlib.suppress(ValueError):
                    self.review_game(history, tries)

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job.status in ('queued', 'running'):
                job.cancelled = True
                if job.kind == 'analyse' and tuple(map(tuple, job.history)) == tuple(map(tuple, self.history)):
                    self.dismissed = self.dismissal(job.history)
                if job.kind == 'move':
                    self.paused = True
                    if self.game_clock:
                        self.pause_clock()
                        self.save_match_position()
                if job.status == 'queued':
                    job.status = 'cancelled'
                self.revision += 1

    def pause(self, paused):
        with self.lock:
            self.paused = bool(paused)
            if self.paused:
                self.stop_moves()
                if self.game_clock:
                    self.pause_clock()
            if self.match:
                self.save_match_position()
            self.changed()

    # Bot matches use the same seats, jobs and board as interactive play.

    def remember_match(self, directory):
        directory = Path(directory).resolve()
        ident = hashlib.sha256(str(directory).encode()).hexdigest()[:16]
        self.saved_matches[ident] = directory
        if self.archive:
            self.archive.mkdir(parents=True, exist_ok=True)
            path = self.archive / f'{ident}.json'
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(str(directory)), encoding='utf-8')
            temporary.replace(path)
        return ident

    def fork_freeplay(self):
        if self.saved_game or self.match:
            leaving = self.match is not None
            self.saved_game, self.match = None, None
            self.reset_clock()
            if leaving:
                self.prepare_timed()
            self.freeplay_directory, self.freeplay_signature = None, None
            self.freeplay_records = {}

    def save_freeplay(self):
        """Update the current freeplay archive. Call with the session lock held after a change."""
        if not self.archive or self.match or self.saved_game:
            return
        if self.freeplay_directory is None:
            name = datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-freeplay-' + os.urandom(3).hex()
            self.freeplay_directory = self.archive / name
            self.freeplay_directory.mkdir(parents=True, exist_ok=False)
            self.freeplay_created = datetime.now().astimezone().isoformat()
            self.remember_match(self.freeplay_directory)
        players = [seat | dict(name=self.entries.get(seat['engine'], {}).get('name', 'Human')) for seat in self.seats]
        game = replay(self.history)
        try:
            winner = game.winner if game.winner >= 0 else None
        finally:
            game.close()
        keys = set()
        for seat in [*self.seats, self.analysis]:
            key = self.engine_key(seat) if seat and seat['engine'] != 'human' else None
            if key:
                bare = key.split(':')[0] + ':none'
                keys.update((key, key + ':kept', bare, bare + ':kept'))
        with self.store.lock:
            for ply in range(len(self.history) + 1):
                position = self.store.key(self.history[:ply])
                for key in sorted(keys):
                    for budget in sorted(self.store.by_position.get((position, key), ())):
                        self.freeplay_records[(position, key, budget)] = self.store.order[(position, key, budget)]
        records = list(self.freeplay_records.values())
        reason = 'six' if winner is not None else 'saved'
        if winner is None and self.outcome:
            winner, reason = self.outcome['winner'], self.outcome['reason']
        record = dict(format='bubble-replay', version=1, game=1, history=list(self.history), players=players,
                      winner=winner, reason=reason, opening=self.opening)
        if self.clock_spec['mode'] != 'fixed':
            record.update(clock=self.clock_spec, turns=self.clock_turns)
        signature = json.dumps(record) + '\n'.join(records)
        if signature == self.freeplay_signature:
            return
        summary = dict(output=str(self.freeplay_directory), single=True, kind='freeplay', games=1, completed=1,
                       created_at=self.freeplay_created,
                       players=players, wins=[int(winner == 0), int(winner == 1)], capped=0,
                       results=[dict(game=1, winner=winner, reason=record['reason'], placements=len(self.history))])
        self.write_match(summary, 'game-0001.json', record)
        path = self.freeplay_directory / 'evaluations.jsonl'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(line + '\n' for line in records), encoding='utf-8')
        temporary.replace(path)
        self.write_match(summary)
        self.freeplay_signature = signature

    def match_catalogue(self):
        if self.archive:
            for path in self.archive.glob('*.json'):
                self.saved_matches[path.stem] = Path(json.loads(path.read_text(encoding='utf-8')))
        rows = []
        for ident, directory in list(self.saved_matches.items()):
            try:
                match = saved_ids(json.loads((directory / 'summary.json').read_text(encoding='utf-8')))
            except FileNotFoundError:
                continue
            # A replay is committed before the summary. Recover only that trailing suffix,
            # so ordinary catalogue reads do not reopen every saved game.
            for number in range(len(match['results'])+1, match['games']+1):
                path = directory / f'game-{number:04d}.json'
                if not path.exists():
                    break
                game = json.loads(path.read_text(encoding='utf-8'))
                winner = game['winner']
                player = (winner if number % 2 else 1-winner) if winner is not None else None
                if player is None:
                    match['capped'] += 1
                else:
                    match['wins'][player] += 1
                match['results'].append(dict(game=number, winner=player, reason=game['reason'],
                    placements=len(game['history']), opening=((number-1)//2) % len(match['openings'])))
            match['completed'] = len(match['results'])
            rows.append(dict(id=ident, name=directory.name, **{key: match[key] for key in
                ('games', 'completed', 'wins', 'capped', 'results')}, players=[p['name'] for p in match['players']],
                elo=pair_elo(match['results']), clock=match.get('clock'), opening_range=match.get('opening_range'),
                single=match.get('single', False), kind=match.get('kind', 'match'),
                player_specs=match['players'], created_at=match.get('created_at')))
        return sorted(rows, key=lambda row: row['name'], reverse=True)

    def saved_replay(self, ident, number):
        self.match_catalogue()
        if ident not in self.saved_matches or type(number) is not int or number < 1:
            raise ValueError('No such saved game')
        directory = self.saved_matches[ident]
        game = saved_ids(json.loads((directory / f'game-{number:04d}.json').read_text(encoding='utf-8')))
        return directory, game

    def open_saved_game(self, ident, number):
        directory, game = self.saved_replay(ident, number)
        with self.lock:
            if self.study is None:
                # Analysis has its own queue and CPU model; it cannot spend a live game's clock.
                package = getattr(self.engines, 'tactical_package', None)
                options = dict(proof_stamps=getattr(self.engines, 'proof_stamps', False))
                self.study = Session(dict(self.entries), Engines('cpu', package, **options), Evaluations(self.study_store),
                                     self.rescan_entries, archive=self.archive, analysis_engines=Engines('cpu', package, **options),
                                     save_initial=False)
            study = self.study
        with study.lock:
            for job in study.jobs.values():
                if job.status in ('queued', 'running'):
                    job.cancelled = True
            study.entries.update(self.entries)
            study.seats = [dict(engine='human'), dict(engine='human')]
            if study.analysis:
                study.analysis['auto'] = False
            # Reuse the game's evaluations without editing its files.
            path, proven = directory / 'evaluations.jsonl', []
            if path.exists():
                with study.store.lock, path.open(encoding='utf-8') as lines:
                    for line in lines:
                        try:
                            record = json.loads(line)
                            key = (hashlib.blake2b(record['position'].encode(), digest_size=16).digest(), record['engine'],
                                   (record['simulations'], record['solver_nodes']))
                            if key not in study.store.order:
                                study.store.index(record, line.strip())
                            if record.get('proof') or record.get('proofs'):
                                proven.append((record, line.strip()))
                        except (ValueError, KeyError, TypeError):
                            continue
            study.load(game['history'], True, saved_game=dict(batch=ident, name=directory.name, game=number,
                       players=[p['name'] for p in game['players']], winner=game['winner'], reason=game['reason']))
            study.outcome = dict(winner=game['winner'], reason='time') if game['reason'] == 'time' else None
            study.clock_spec = clock_spec(game.get('clock'))
            study.clock_turns = list(game.get('turns') or [])
            study.game_clock = study.new_game_clock()
            last = study.clock_turns[-1] if study.clock_turns else {}
            if study.game_clock and last.get('cross_ms') is not None and last.get('circle_ms') is not None:
                study.game_clock.balances = [int(last['cross_ms'] * 1e6), int(last['circle_ms'] * 1e6)]
            # The game's file also keeps evaluations of positions it left by undo; their proofs belong to its table.
            for record, line in proven:
                study.proofs.add([tuple(map(int, cell.split(','))) for cell in record['position'].split()], record, line)
            study.changed()
        return study

    def match_editable(self):
        if self.match and self.match['active'] or self.match_worker and self.match_worker.is_alive():
            raise ValueError('Stop the match before changing its players or position')

    def match_seat(self, specification, preset):
        specification = dict(engine=specification) if isinstance(specification, str) else dict(specification)
        selector = specification['engine']
        if selector.endswith('}') and '{' in selector:
            selector, custom = selector[:-1].rsplit('{', 1)
            specification.update(preset='custom', custom={k.strip(): v.strip() if k.strip() == 'active' else int(v.strip())
                                 for k, v in (item.split('=', 1) for item in custom.split(','))})
        if '@' in selector:
            selector, suffix = selector.rsplit('@', 1)
            specification['preset' if suffix in PRESET_NAMES else 'checkpoint'] = suffix
            if '@' in selector:
                selector, specification['checkpoint'] = selector.rsplit('@', 1)
        if Path(selector).is_file() and Path(selector).suffix == '.pt':
            found = scan(extra_runs=[Path(selector)], device=getattr(self.engines, 'device', None))
            entry = next(e for e in found.values() if e['kind'] == 'bubble')
            same = next((e for e in self.entries.values() if e['kind'] == 'bubble' and e['path'] == entry['path']
                         and not e.get('q_range_floor')), None)
            if same:
                entry = same
            elif entry['id'] in self.entries:
                entry['id'] += '~' + hashlib.sha256(str(entry['path']).encode()).hexdigest()[:8]
            self.entries[entry['id']] = entry
            selector = entry['id']
        name = selector.casefold()
        matches = [e for e in self.entries.values()
                   if name in (e['id'].casefold(), e['name'].casefold(), e.get('label', '').casefold())]
        if not matches:
            matches = [e for e in self.entries.values() if name in (e['kind'], e.get('badge'))]
        if not matches and name.startswith('bubble:') and name[7:].isdigit():
            specification['checkpoint'] = name[7:]
            matches = [e for e in self.entries.values() if e['kind'] == 'bubble' and
                       any(c.rsplit('/', 1)[-1].isdigit() and int(c.rsplit('/', 1)[-1]) == int(name[7:])
                           for c in e['checkpoints'])]
        if len(matches) != 1:
            choices = ', '.join(e['id'] for e in matches or self.entries.values())
            raise ValueError(f'Choose an unambiguous engine for {specification["engine"]!r}: {choices}')
        entry = matches[0]
        checkpoint = specification.get('checkpoint')
        if entry['kind'] == 'bubble' and checkpoint:
            if checkpoint == 'champion':
                path = Path(entry['path'])
                checkpoint = json.loads((path / 'champion.json').read_text(encoding='utf-8'))['checkpoint']
            elif checkpoint.isdigit():
                found = [c for c in entry['checkpoints'] if c.rsplit('/', 1)[-1].isdigit() and
                         int(c.rsplit('/', 1)[-1]) == int(checkpoint)]
                if len(found) != 1:
                    raise ValueError('Choose a full, unambiguous checkpoint id')
                checkpoint = found[0]
        if entry['kind'] == 'bubble' and 'solver_nodes' in (specification.get('custom') or {}):
            raise ValueError('solver_nodes is not a custom Bubble budget: root query nodes follow its simulations or ms')
        seat = self.seat(entry['id'], checkpoint, specification.get('preset', preset),
                         specification.get('custom'))
        source = {k: str(v) if isinstance(v, Path) else v for k, v in entry.items()
                  if k in ('kind', 'badge', 'name', 'path', 'cwd', 'model', 'engine', 'library', 'mirrored', 'q_range_floor')}
        if entry['kind'] == 'six':
            source['command'] = command_of(entry, seat['checkpoint'])
        if entry['kind'] == 'strix' and entry.get('networks'):
            source['model'] = str(entry['networks'][seat['checkpoint']])
        if 'libraries' in entry:
            source['libraries'] = list(map(str, entry['libraries']))
        if entry['kind'] == 'bubble':
            source['weights'] = str(export_path(entry, seat['checkpoint']).resolve())
            source['weights_sha256'] = file_digest(file_identity(source['weights']))
            seat['device'] = specification.get('device', getattr(self.engines, 'device', 'cpu'))
            if seat['device'] not in ('cpu', 'cuda'):
                raise ValueError('Bubble device must be cpu or cuda')
            if seat['preset'] in PRESET_NAMES and seat['device'] != getattr(self.engines, 'device', 'cpu'):
                # A named level follows the seat's own device's ladder, not the server's.
                seat['budget'] = dict((CPU_PRESETS if seat['device'] == 'cpu' else PRESETS['bubble'])[seat['preset']])
        elif 'device' in specification:
            raise ValueError('Choose the external engine backend from its catalogue entry')
        source['device'] = seat.get('device', entry.get('backend') or entry['name'].rsplit(' · ', 1)[-1]
                                    if entry['kind'] == 'six' else 'cpu')
        from hexo import library
        files = [library]
        if entry['kind'] == 'bubble':
            files += [library.with_name(library.name.replace('hexo', 'hexo_gumbel'))]
            source['solver_build'] = self.engines.solver_build() if seat['budget']['solver_nodes'] else 'none'
        elif entry['kind'] == 'six':
            command_files = [Path(entry.get('cwd') or os.getcwd()) / arg for arg in source['command']]
            files += [path for path in command_files if path.is_file()] + list(entry.get('files', []))
        elif entry['kind'] == 'strix':
            files += [Path(source['model']), *([Path(entry['engine'])] if entry.get('engine') else [])]
        elif entry['kind'] == 'seal':
            files += [Path(entry['library'])]
        source['files'] = {str(path.resolve()): file_digest(file_identity(path)) for path in files}
        return seat | dict(name=entry['name'] + (f"/{seat['checkpoint']}" if seat['checkpoint'] else ''), source=source)

    def start_match(self, players, games=None, preset='standard', output=None, openings=None, max_placements=512,
                    book=None, opening_range=None, unique_openings=None, seed=0, evaluations=None, clock=None,
                    replace=False):
        """Start a batch on this board; an unfinished game against a person is refused unless `replace`."""
        with self.lock:
            self.match_editable()
            board = replay(self.history)
            try:
                if self.history and board.winner < 0 and any(s['engine'] == 'human' for s in self.seats) and not replace:
                    raise ValueError('A human game is on this board; use a new player port or finish/reset that game')
            finally:
                board.close()
            if unique_openings is not None and (type(unique_openings) is not int or unique_openings < 1):
                raise ValueError('unique_openings must be positive')
            if games is None:
                games = 2 * (len(openings) if openings is not None else unique_openings or 1)
            if len(players) != 2 or type(games) is not int or games < 1:
                raise ValueError('A match needs two engines and a positive number of games')
            if type(max_placements) is not int or max_placements < 0 or max_placements == 1:
                raise ValueError('max_placements must be at least 2, or 0 for uncapped')
            seats = [self.match_seat(p, preset) for p in players]
            clock = clock_spec(clock)
            mode = clock['mode']
            if mode != 'fixed':
                for seat in seats:
                    self.timed_config(seat)  # Refuse unsupported adapters before starting or writing results.
            selection = None
            book = book or self.book
            if openings is None and (book or opening_range or unique_openings):
                if not book:
                    raise ValueError('Select an opening book with --book or --dense-run')
                count = unique_openings or (games + 1) // 2
                if games % 2:
                    raise ValueError('Book matches need an even game count for colour-swapped opening pairs')
                if unique_openings and games < 2 * unique_openings:
                    raise ValueError('Each unique book opening needs two colour-swapped games')
                selection = book_openings(book, opening_range or 'wide', count, seed)
                openings = [n['moves'] for n in selection['nodes']]
            elif openings is not None and (opening_range or unique_openings):
                raise ValueError('Choose explicit openings or a book selection, not both')
            openings = [[[0, 0]]] if openings is None else openings
            if not openings:
                raise ValueError('Provide at least one opening')
            openings = [import_history(json.dumps(h)) for h in openings]
            for history in openings:
                game = replay(history)
                try:
                    if game.winner >= 0 or max_placements and len(history) >= max_placements:
                        raise ValueError('An opening must be unfinished and shorter than max_placements')
                    dumps([tuple(p) for p in history])
                finally:
                    game.close()
            name = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + os.urandom(3).hex()
            directory = Path(output) if output else ROOT / 'artifacts' / 'play' / name
            directory.mkdir(parents=True, exist_ok=False)
            match = dict(schema='bubble-match-v1', active=True, between=False, games=games, completed=0, current=1, wins=[0, 0], capped=0,
                         players=seats, results=[], openings=openings, max_placements=max_placements,
                         output=str(directory.resolve()), error=None, opening_selection=selection,
                         unique_openings=len(openings), opening_range=selection['range'] if selection else 'explicit',
                         clock=clock, preparing=mode != 'fixed', turns=[], outcome=None, pentanomial=[0]*5, seed=seed,
                         partial_spent_ms=0)
            self.write_match(match)
            self.remember_match(directory)
            evaluation_path = Path(evaluations) if evaluations else directory / 'evaluations.jsonl'
            evaluation_path.parent.mkdir(parents=True, exist_ok=True)
            if self.freeplay_directory or self.history:
                self.save_freeplay()
            self.store = Evaluations(evaluation_path)
            self.match = match
            self.match_file = lock_match(directory)
            self.game_clock = self.new_game_clock()
            for job in self.jobs.values():
                if job.status in ('queued', 'running'):
                    job.cancelled = True
            if self.analysis:
                self.analysis = self.analysis | dict(auto=False)
            self.history = [tuple(p) for p in openings[0]]
            self.seats = [{k: v for k, v in s.items() if k not in ('name', 'source')} for s in seats]
            self.new_lines()
            self.paused = False
            self.save_match_position()
            self.changed()
            self.match_worker = threading.Thread(target=self.run_match, args=(match,), daemon=True)
            self.match_worker.start()

    def write_match(self, match, name='summary.json', data=None):
        path = Path(match['output']) / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(match if data is None else data, indent=2), encoding='utf-8')
        temporary.replace(path)

    def new_game_clock(self):
        """Full balances for the match's clock, or the freeplay clock (`clock_spec`) outside a match; None for fixed."""
        specification = self.match['clock'] if self.match else self.clock_spec
        if specification['mode'] == 'fixed':
            return None
        return Clock(dict(base_ms=specification['ms']) if specification['mode'] == 'move' else specification)

    def save_match_position(self):
        if self.match:
            self.write_match(self.match, 'current.json', dict(game=self.match['current'], history=list(self.history),
                             seats=self.seats, turns=self.match['turns'],
                             partial_spent_ms=self.match.get('partial_spent_ms', 0),
                             balances=self.game_clock.remaining() if self.game_clock else None))

    def reset_clock(self):
        """Full balances for a freeplay game on `clock_spec`, with no outcome and an empty turn log. Call with the lock
        held; a match or a saved game keeps its own clock."""
        if self.match or self.saved_game:
            return
        self.game_clock, self.outcome, self.clock_turns, self.clock_partial_ms = self.new_game_clock(), None, [], 0

    def set_clock(self, clock):
        """Put the freeplay game on `clock` (see `clock_spec`) from full balances at the current position. Engine seats
        then play through timed engines (`timed_config`), started before the clock runs; a seat whose engine cannot keep
        a clock is refused."""
        spec = clock_spec(clock)
        with self.lock:
            self.match_editable()
            if self.saved_game:
                raise ValueError('A saved game has no clock')
            if spec['mode'] != 'fixed':
                for seat in self.seats:
                    if seat['engine'] != 'human':
                        self.timed_config(seat)
            self.match, self.clock_spec, self.notice = None, spec, None
            self.stop_moves()
            self.reset_clock()
            self.prepare_timed()
            self.changed()

    def prepare_timed(self, sides=(0, 1)):
        """Replace the timed engines of the freeplay `sides` (and of any still being prepared): for a clocked game, one per
        engine seat, started in the background while `clock_preparing` holds new moves, and the clock too when it runs
        for one of those sides (stopped, its time so far kept for the turn). The other side's engine keeps playing. Call
        with the lock held."""
        sides = set(sides) | self.preparing_sides
        for side in sides:
            if self.seat_engines[side]:
                self.seat_engines[side].close()
                self.seat_engines[side] = None
        seats = {side: dict(self.seats[side]) for side in sides if self.seats[side]['engine'] != 'human'}
        if self.match or self.clock_spec['mode'] == 'fixed' or not seats:
            self.clock_preparing, self.preparing_sides = None, set()
            return
        if self.game_clock and self.game_clock.running in sides:
            self.pause_clock()
        generation = self.clock_preparing = object()
        self.preparing_sides = set(sides)

        def prepare():
            from timed_engine import TimedEngine
            engines, failure = {}, None
            try:
                for side, seat in seats.items():
                    engines[side] = TimedEngine(self.timed_config(seat))
            except Exception as error:
                failure = error
            with self.lock:
                if self.clock_preparing is not generation or self.closing or failure is not None:
                    for engine in engines.values():
                        engine.close()
                    if self.clock_preparing is not generation or self.closing:
                        return
                self.clock_preparing, self.preparing_sides = None, set()
                if failure is None:
                    for side, engine in engines.items():
                        self.seat_engines[side] = engine
                else:
                    for engine in self.seat_engines:
                        if engine:
                            engine.close()
                    self.seat_engines = [None, None]
                    self.clock_spec, self.notice = dict(mode='fixed'), f'The clock is off: {failure}'
                    self.reset_clock()
                self.changed()
        threading.Thread(target=prepare, daemon=True).start()

    def clock_turn(self, side, at=None):
        """Charge `side`'s completed turn to the freeplay clock (lock held): finished after its time ran out it loses on
        time, otherwise it gets its increment, or a full allowance again on a per-turn clock. Logs the balances in
        `clock_turns`, the turn's time including what it spent before a pause; True when the turn stands."""
        clock = self.game_clock
        expired = clock.expired(at)
        spent = clock.stop(completed=not expired, at=at) / 1e6 + self.clock_partial_ms
        self.clock_partial_ms = 0
        if expired:
            self.outcome = dict(winner=1 - side, reason='time')
        elif self.clock_spec['mode'] == 'move':
            clock.balances = [int(self.clock_spec['ms'] * 1e6)] * 2
        balances = clock.json()
        self.clock_turns.append(dict(ply=len(self.history), side=side, spent_ms=round(spent),
                                     cross_ms=round(balances['cross_ms']), circle_ms=round(balances['circle_ms'])))
        return not expired

    def balances(self, flagged=None):
        """Both clock balances in whole ms for a turn record, the side that ran out of time (`flagged`) at zero."""
        clock = self.game_clock.json()
        return dict(cross_ms=0 if flagged == 0 else round(clock['cross_ms']),
                    circle_ms=0 if flagged == 1 else round(clock['circle_ms']))

    def check_time(self):
        """A freeplay side whose clock ran out while it was to move loses on time (lock held)."""
        clock = self.game_clock
        if clock is None or self.match or self.outcome or self.paused or clock.running is None or not clock.expired():
            return
        side = clock.running
        if any(j.kind == 'move' and j.status == 'running' and getattr(j, 'received', None) is not None
               and not clock.expired(j.received) for j in self.jobs.values()):
            return
        spent = clock.stop() / 1e6 + self.clock_partial_ms
        self.clock_partial_ms = 0
        self.outcome = dict(winner=1 - side, reason='time')
        balances = clock.json()
        self.clock_turns.append(dict(ply=len(self.history), side=side, spent_ms=round(spent),
                                     cross_ms=0 if side == 0 else round(balances['cross_ms']),
                                     circle_ms=0 if side == 1 else round(balances['circle_ms'])))
        self.stop_moves()
        self.changed()

    def pause_clock(self):
        """Stop the running clock without an increment, charging the time spent so far."""
        if self.game_clock:
            elapsed = self.game_clock.stop()/1e6
            if self.match:
                self.match['partial_spent_ms'] = self.match.get('partial_spent_ms', 0) + elapsed
            else:
                self.clock_partial_ms += elapsed

    def resume_match(self, directory):
        with self.lock:
            self.match_editable()
            if self.history and any(s['engine'] == 'human' for s in self.seats):
                raise ValueError('Use an empty player board to resume a saved batch')
            self.match_catalogue()
            directory = Path(self.saved_matches.get(str(directory), directory)).resolve()
            handle = lock_match(directory)
            try:
                match = saved_ids(json.loads((directory / 'summary.json').read_text(encoding='utf-8')))
                if match['schema'] != 'bubble-match-v1':
                    raise ValueError('Not a saved player batch')
                # Completed replay files are the commits; summary/current may lag one file after a crash.
                results, wins, capped, pairs = [], [0, 0], 0, [0]*5
                for number in range(1, match['games']+1):
                    path = directory / f'game-{number:04d}.json'
                    if not path.exists():
                        break
                    game = json.loads(path.read_text(encoding='utf-8'))
                    notation = path.with_suffix('.htttx')
                    temporary = notation.with_suffix('.htttx.tmp')
                    temporary.write_text(dumps([tuple(p) for p in game['history']]), encoding='utf-8')
                    temporary.replace(notation)
                    winner = game['winner']
                    player = (winner if number % 2 else 1-winner) if winner is not None else None
                    if player is None:
                        capped += 1
                    else:
                        wins[player] += 1
                    results.append(dict(game=number, winner=player, reason=game['reason'], placements=len(game['history']),
                                        opening=((number-1)//2) % len(match['openings'])))
                    if number % 2 == 0:
                        points = sum(1 if r['winner'] == 0 else .5 if r['winner'] is None else 0 for r in results[-2:])
                        pairs[round(points*2)] += 1
                if len(results) >= match['games']:
                    raise ValueError('This batch is already complete')
                registry = {}
                for seat in match['players']:
                    source = seat['source']
                    if seat['budget'].get('solver_nodes') and source['solver_build'] != self.engines.solver_build():
                        raise ValueError('Tactical solver build changed since the batch started')
                    if source.get('weights') and file_digest(file_identity(source['weights'])) != source['weights_sha256']:
                        raise ValueError('Checkpoint weights changed since the batch started')
                    for path, digest in source['files'].items():
                        if file_digest(file_identity(path)) != digest:
                            raise ValueError(f'Engine file changed since the batch started: {path}')
                    entry = {k: Path(v) if k in ('path', 'model', 'engine', 'library') else v for k, v in source.items()
                             if k in ('kind', 'path', 'command', 'cwd', 'model', 'engine', 'library', 'mirrored',
                                      'libraries')}
                    entry.update(id=seat['engine'], name=source.get('name', seat['name']), presets=PRESETS[source['kind']],
                                 badge=source.get('badge', source['kind']))
                    if seat['checkpoint'] is not None:
                        entry['checkpoints'] = [seat['checkpoint']]
                    registry[seat['engine']] = entry
                number = len(results)+1
                current_path = directory / 'current.json'
                saved = json.loads(current_path.read_text(encoding='utf-8')) if current_path.exists() else {}
                continuing = saved.get('game') == number
                history = saved['history'] if continuing else match['openings'][((number-1)//2) % len(match['openings'])]
                replay(history).close()
                match.update(active=True, completed=len(results), current=number, results=results, wins=wins, capped=capped,
                             pentanomial=pairs, output=str(directory), between=False, preparing=match['clock']['mode'] != 'fixed',
                             outcome=None, error=None, turns=saved.get('turns', []) if continuing else [])
                match['partial_spent_ms'] = saved.get('partial_spent_ms', 0) if continuing else 0
                self.entries.update(registry)
                self.match, self.match_file = match, handle
                self.game_clock = self.new_game_clock()
                if self.game_clock and continuing and saved.get('balances'):
                    self.game_clock.balances = saved['balances']
                self.store = Evaluations(directory / 'evaluations.jsonl')
                self.history = list(map(tuple, history))
                order = [0, 1] if number % 2 else [1, 0]
                self.seats = [{k: v for k, v in match['players'][i].items() if k not in ('name', 'source')} for i in order]
                self.stop_moves()
                self.new_lines()
                if self.analysis:
                    self.analysis['auto'] = False
                self.paused = False
                self.write_match(match)
                self.remember_match(directory)
                self.changed()
                self.match_worker = threading.Thread(target=self.run_match, args=(match,), daemon=True)
                self.match_worker.start()
            except BaseException:
                handle.close()
                raise

    def timed_config(self, seat):
        """The `timed_engine` configuration of `seat`, its simulations as a ceiling and a nonzero solver_nodes turning
        on the turn's time-sliced proofs; ValueError for an engine that cannot keep a clock (`keeps_clock`)."""
        entry, budget = self.entries[seat['engine']], seat['budget']
        kind = entry['kind']
        if not keeps_clock(entry):
            raise ValueError(f"{entry['name']} plays a fixed budget; its adapter cannot keep a clock")
        if kind == 'bubble':
            return dict(kind=kind, model=str(export_path(entry, seat['checkpoint']).resolve()),
                        tactical_package=str(self.engines.tactical_package) if getattr(self.engines, 'tactical_package', None) else None,
                        device=seat.get('device', getattr(self.engines, 'device', 'cpu')), search=dict(enabled=budget['simulations'] > 0,
                        max_simulations=max(1, budget['simulations']), q_range_floor=entry.get('q_range_floor', 0.)),
                        solver=dict(enabled=budget['solver_nodes'] > 0, stamps=getattr(self.engines, 'proof_stamps', False)))
        if kind == 'six':
            return dict(kind=kind, command=command_of(entry, seat['checkpoint']) + budget.get('args', []),
                        cwd=str(entry.get('cwd') or ROOT), path=list(map(str, entry.get('libraries', []))),
                        mirrored=entry.get('mirrored', False), nodes=budget['nodes'])
        return dict(kind=kind, max_ms=budget['ms'], **({'library': str(entry['library'])} if entry.get('library') else {}))

    def run_match(self, match):
        try:
            if match['clock']['mode'] != 'fixed':
                from timed_engine import TimedEngine
                for seat in match['players']:
                    if not match['active'] or self.closing:
                        return
                    engine = TimedEngine(self.timed_config(seat))
                    self.timed_engines.append(engine)
                    seat['source']['timed_identity'] = engine.identity
                with self.lock:
                    match['preparing'] = False
                    self.write_match(match)
                    self.changed()
            with self.lock:
                while match['active'] and not self.closing:
                    game = replay(self.history)
                    try:
                        winner = game.winner
                    finally:
                        game.close()
                    if self.game_clock and self.game_clock.expired():
                        received = any(j.status == 'running' and getattr(j, 'received', None) is not None
                                       and not self.game_clock.expired(j.received) for j in self.jobs.values())
                        if not received:
                            side = self.game_clock.running
                            match['outcome'] = dict(winner=1-side, reason='time')
                            self.pause_clock()
                            match['turns'].append(dict(ply=len(self.history), side=side, stop_reason='deadline',
                                                      clock_spent_ms=match['partial_spent_ms'], completed=None, nodes=None,
                                                      **self.balances(side)))
                            self.stop_moves()
                    if match['outcome']:
                        winner = match['outcome']['winner']
                    if winner < 0 and (not match['max_placements'] or len(self.history) < match['max_placements']):
                        self.lock.wait(timeout=.1 if self.game_clock else None)
                        continue
                    number = match['current']
                    order = [0, 1] if number % 2 else [1, 0]
                    result = dict(format='bubble-replay', version=1, game=number,
                                  players=[match['players'][i] for i in order], history=list(self.history),
                                  winner=winner if winner >= 0 else None, reason='win' if winner >= 0 else 'capped',
                                  clock=match['clock'], turns=match['turns'])
                    if match['outcome']:
                        result.update(match['outcome'])
                    opening_index = ((number - 1) // 2) % len(match['openings'])
                    result['opening'] = (match['opening_selection']['nodes'][opening_index] if match['opening_selection']
                                         else dict(moves=match['openings'][opening_index]))
                    self.write_match(match, f'game-{number:04d}.json', result)
                    (Path(match['output']) / f'game-{number:04d}.htttx').write_text(dumps(self.history), encoding='utf-8')
                    if winner >= 0:
                        match['wins'][order[winner]] += 1
                    else:
                        match['capped'] += 1
                    match['results'].append(dict(game=number, winner=order[winner] if winner >= 0 else None,
                                                 reason=result['reason'], placements=len(self.history), opening=opening_index))
                    match['completed'] = number
                    match['between'] = True
                    if number % 2 == 0:
                        points = sum(1 if r['winner'] == 0 else .5 if r['winner'] is None else 0
                                     for r in match['results'][-2:])
                        match['pentanomial'][round(points*2)] += 1
                        from dense_posterior import Posterior
                        posterior = Posterior(['A', 'B'], 'B', [('A', 'B', match['pentanomial'])], matchup_prior=0)
                        elo, sd = posterior.difference('A', 'B', matchup=False)
                        match['elo'] = dict(a_minus_b=elo, interval=[elo-1.96*sd, elo+1.96*sd],
                                            pairs=sum(match['pentanomial']))
                    self.write_match(match)
                    print(f"Game {number}/{match['games']}: {result['reason']}; wins {match['wins']}, "
                          f"capped {match['capped']}", flush=True)
                    self.changed()
                    if number == match['games']:
                        break
                    # Leave the finished board visible briefly; Pause also holds the next game.
                    self.lock.wait_for(lambda: not match['active'] or self.closing, timeout=2)
                    self.lock.wait_for(lambda: not self.paused or not match['active'] or self.closing)
                    if not match['active'] or self.closing:
                        break
                    match['current'] += 1
                    match['between'] = False
                    match['outcome'], match['turns'] = None, []
                    match['partial_spent_ms'] = 0
                    self.game_clock = self.new_game_clock()
                    self.history = [tuple(p) for p in match['openings'][(number // 2) % len(match['openings'])]]
                    order = [0, 1] if match['current'] % 2 else [1, 0]
                    self.seats = [{k: v for k, v in match['players'][i].items() if k not in ('name', 'source')}
                                  for i in order]
                    self.new_lines()
                    self.save_match_position()
                    self.changed()
        except Exception as error:
            match['error'] = str(error)
            print(f'Match stopped: {error}', file=sys.stderr, flush=True)
        finally:
            with self.lock:
                match['active'] = False
                self.paused = True
                self.stop_moves()
                if self.game_clock:
                    self.pause_clock()
                try:
                    self.save_match_position()
                    self.write_match(match)
                except OSError as error:
                    match['error'] = str(error)
                    print(f'Cannot save match: {error}', file=sys.stderr, flush=True)
                self.changed()
            for engine in self.timed_engines:
                engine.close()
            self.timed_engines.clear()
            if self.match_file:
                self.match_file.close()
                self.match_file = None

    def stop_match(self):
        with self.lock:
            if self.match:
                self.match['active'] = False
            self.paused = True
            self.stop_moves()
            if self.game_clock:
                self.pause_clock()
            self.save_match_position()
            self.changed()

    def rescan(self):
        """Replace the engine list. A seat follows its engine, matched by kind and path, to its id in the new list;
        a seat whose engine or checkpoint is gone becomes a person, and analysis falls back to the first Bubble
        model. Rescans run one at a time, so an older scan never replaces a newer one."""
        with self.rescanning:
            entries = self.rescan_entries()
            identity = lambda e: (e['kind'], engine_identity(e))
            def follow(seat):
                old = self.entries.get(seat['engine'])
                moved = next((e['id'] for e in entries.values() if old and identity(e) == identity(old)), None)
                return seat | dict(engine=moved)
            def valid(seat):
                entry = entries.get(seat['engine']) if seat['engine'] else None
                return entry is not None and (not entry.get('checkpoints') or seat['checkpoint'] in entry['checkpoints'])
            with self.lock:
                self.match_editable()
                followed = [seat if seat['engine'] == 'human' else follow(seat) for seat in self.seats]
                analysis = follow(self.analysis) if self.analysis else None
                previous, self.entries = self.entries, entries
                try:
                    seats = [seat if seat['engine'] == 'human' or valid(seat) else dict(engine='human') for seat in followed]
                    if analysis is None or not valid(analysis):
                        bubble = next((e for e in entries.values() if e['kind'] == 'bubble'), None)
                        analysis = self.seat(bubble['id'], None, 'standard') | dict(auto=True) if bubble else None
                except Exception:
                    self.entries = previous
                    raise
                self.seats, self.analysis = seats, analysis
                self.stop_moves()
                self.stop_analysis()
                if not self.match:
                    self.prepare_timed()
                self.changed()

    # Working

    def close(self, timeout=30):
        """Cancel every job, wait for the worker to stop, then close the engines."""
        with self.lock:
            self.save_freeplay()
            self.closing = True
            for job in self.jobs.values():
                if job.status in ('queued', 'running'):
                    job.cancelled = True
            self.lock.notify_all()
        for worker in self.workers:
            worker.join(timeout)
        if self.match_worker:
            self.match_worker.join(timeout)
        for engine in self.seat_engines:
            if engine:
                engine.close()
        self.engines.close()
        if self.analysis_engines is not self.engines:
            self.analysis_engines.close()
        if self.study:
            self.study.close(timeout)

    def lane_engines(self, job):
        return self.engines if lane(job) == 'move' else self.analysis_engines

    def work(self, name):
        """Run the jobs of one lane, most urgent first."""
        queue = self.queues[name]
        while True:
            with self.lock:
                while not queue and not self.closing:
                    self.lock.wait()
                if self.closing:
                    return
                job = heapq.heappop(queue)[2]
                if job.cancelled:
                    job.status = 'cancelled'
                    continue
                job.status = 'running'
                self.revision += 1
            result, failure = None, None
            try:
                result = self.run(job)
                job.received = time.monotonic_ns()
            except Yielded:
                with self.lock:
                    job.status = 'queued'
                    heapq.heappush(queue, (job.priority, next(self.order), job))
                continue
            except Cancelled:
                job.cancelled = True
            except Exception as error:
                failure = error
            with self.lock:
                job.ended = time.time()
                if failure is not None:
                    job.status, job.error = 'failed', str(failure)
                    self.paused = self.paused or job.kind == 'move'
                    if job.kind == 'move' and not self.match:
                        self.pause_clock()
                    if self.match and self.match['active'] and job.kind == 'move':
                        self.match['error'] = str(failure)
                        self.pause_clock()
                        self.save_match_position()
                else:
                    job.status = 'cancelled' if job.cancelled else 'done'
                    turn = [tuple(p) for p in result or ()] if job.kind == 'move' else []
                    shown = list(job.placed)
                    # A turn that disagrees with the stones already shown leaves them; `changed` then queues a move
                    # for the rest of the turn.
                    if job.kind == 'move' and not job.cancelled and list(job.reached()) == self.history \
                            and turn[:len(shown)] == shown:
                        expired = self.game_clock and self.game_clock.expired(job.received)
                        if self.match and self.match['active']:
                            if expired:
                                self.match['outcome'] = dict(winner=1-job.side, reason='time')
                            elapsed = self.game_clock.stop(completed=not expired, at=job.received) if self.game_clock else None
                            self.match['turns'].append(dict(ply=len(job.history), side=job.side,
                                **getattr(job, 'measurements', {}),
                                clock_spent_ms=elapsed/1e6+self.match['partial_spent_ms'] if elapsed is not None else None))
                            self.match['partial_spent_ms'] = 0
                            if self.game_clock and not expired and self.match['clock']['mode'] == 'move':
                                self.game_clock.balances = [int(self.match['clock']['ms']*1e6)]*2
                            if self.game_clock:
                                self.match['turns'][-1].update(self.balances(job.side if expired else None))
                        freeplay = self.game_clock is not None and not self.match and self.game_clock.running == job.side
                        if expired and freeplay:
                            self.clock_turn(job.side, job.received)
                        if not expired:
                            if not (self.match and self.match['active']):
                                self.fork_freeplay()
                            self.history.extend(turn[len(shown):])
                            if freeplay:
                                self.clock_turn(job.side, job.received)
                        if self.match and self.match['active']:
                            self.match['error'] = None
                            self.save_match_position()
                try:
                    self.changed()
                except Exception as error:
                    self.revision += 1
                    job.status, job.error = 'failed', f'{failure or ""} {error}'.strip()

    def watcher(self, job, count=True):
        """A network-row callback (see `search`) that stops a cancelled job and, when `count`, adds row counts to its
        progress. A deepening job steps aside for more urgent analysis; deepening and review return True, which holds
        their network work, while an engine seat searches."""
        def watch(n):
            busy = False
            if hasattr(job, 'tier') or job.kind == 'review':
                with self.lock:
                    if hasattr(job, 'tier') and self.queues['analysis'] and self.queues['analysis'][0][0] < job.priority:
                        job.cancelled = True
                    busy = any(j.kind == 'move' and j.status == 'running' for j in self.jobs.values())
            if job.cancelled:
                raise Cancelled()
            if count:
                job.done += n
            return busy
        return watch

    def evaluation(self, job, seat, history, force=False):
        """The evaluation of `history` for `seat`, saved. Unless forced, a saved one at least as deep is reused.
        Analysis searches the analysis board's game graph and a move its seat's (see `Engines.evaluate`); a move
        never reuses a saved evaluation, and moves and deepening tiers are saved under the kept key. A finished
        analysis refreshes the earlier positions of its game (see `refresh`); a refresh job (`refresh`, the saved
        evaluation it replaces) searches the graph at its position again and replaces that evaluation under its own
        engine key and budget."""
        key, budget, keep = self.engine_key(seat), self.engines.effective(seat['budget']), hasattr(job, 'tier')
        line, game, refresh = getattr(job, 'line', None), getattr(job, 'game', None), getattr(job, 'refresh', None)
        if key is None:
            raise ValueError('The model file is gone; rescan the engines')
        key += ':kept' if keep else ''
        saved = None if force or line is not None else self.store.covering(history, key, budget)
        if saved:
            return saved
        live = (lambda seen: setattr(job, 'live', seen)) if job.kind != 'review' else None
        used = []
        try:
            found, spent, weights = self.lane_engines(job).evaluate(
                self.entries[seat['engine']], seat['checkpoint'], budget, history, self.watcher(job, job.kind != 'review'),
                live=live, keep=keep, line=line, known=self.proofs if job.kind != 'move' else None,
                **({'device': seat['device']} if 'device' in seat else {}), **({'game': game} if game is not None else {}),
                **({'refresh': refresh} if refresh is not None else {}), used=used,
                **({'placed': lambda moves: self.place(job, moves)} if job.kind == 'move' else {}))
            if job.cancelled:
                raise Cancelled()
        finally:
            # Any primary search, even a cancelled one, changed the graph for the positions before it.
            stamp = self.searched(used[-1], len(history), refresh is None) if job.kind == 'analyse' and used else None
        if stamp is not None:
            found['graph'] = stamp
            for step in found.get('later', []):
                step['graph'] = stamp
        entry = self.entries[seat['engine']]
        model = f"{entry['name']}/{seat['checkpoint']}" if seat['checkpoint'] else entry['name']
        job.incomplete = spent['solver_nodes'] < budget['solver_nodes']
        if job.incomplete and job.kind == 'analyse':
            with self.lock:
                tried = (tuple(history), key, budget['simulations'], budget['solver_nodes'])
                tries = self.retries[tried] = self.retries.get(tried, 0) + 1
            if tries <= 3:
                timer = threading.Timer(31, self.retry, args=(list(history),))
                timer.daemon = True
                timer.start()
        if refresh is not None:
            weights, spent = refresh['engine'], dict(simulations=refresh['simulations'], solver_nodes=refresh['solver_nodes'])
            spent.update({k: refresh[k] for k in ('solver_ms', 'ms') if k in refresh})
            if refresh.get('solver'):
                found['solver'] = refresh['solver']   # a refresh rereads the graph; the proof work it replaces stays reported
        saved = self.save(history, weights, spent, found, model)
        if job.kind == 'analyse' and game is not None:
            if refresh is None:
                self.refresh(history, seat, game)
            elif used and moved(found, refresh):
                # The refresh changed this position's result: that is new evidence for the positions before it, and
                # worth another round here while the result keeps moving.
                self.searched(used[-1], len(history))
                self.refresh(history, seat, game)
                rounds = getattr(job, 'rounds', 0) + 1
                if rounds < REFRESH_ROUNDS:
                    with self.lock:
                        self.submit(Job('analyse', 2, history, seat=dict(seat), force=True, game=game, refresh=saved, rounds=rounds))
        return saved

    def searched(self, graph, ply, count=True):
        """Note a search at `ply` on analysis graph number `graph`, which becomes the graph analysis searched last, and
        return the stamp an evaluation it produced is saved with: [instance, graph, searches so far]. A primary
        analysis counts (`count`) and records its ply; a refresh re-reads evidence the graph already holds, so it
        takes the current count without adding to it and never stales another position."""
        with self.lock:
            self.analysis_graph = graph
            plies = self.graph_plies.setdefault(graph, [])
            if count:
                plies.append(ply)
            return [self.instance, graph, len(plies)]

    def stale(self, record, ply):
        """True when `record`, the saved evaluation of the position at `ply`, predates a primary analysis of a deeper
        position on the graph analysis searched last (the graph's number from `Engines.game_graph`, in this
        session): values under that position flowed up to this one. A record the graph stamped is older than the
        searches after its stamp; one without a stamp of this graph (a review's, or an earlier graph's) is older
        than every search on it. A refresh never stales a record, and a proven record is never stale."""
        if not record or record.get('proof') or self.analysis_graph is None:
            return False
        stamp = record.get('graph')
        since = stamp[2] if stamp and stamp[:2] == [self.instance, self.analysis_graph] else 0
        return any(searched > ply for searched in self.graph_plies.get(self.analysis_graph, [])[since:])

    def refresh(self, history, seat, game):
        """Queue a refresh of each position up to REFRESH_PLIES placements before `history` whose shown evaluation
        (`lookup`, a fresh analysis or a deepening tier) is stale (see `stale`) and holds no proof, nearest first, with
        `seat`'s network: the search on the game graph `game` moved the values those positions reach (see
        `Engines.evaluate`)."""
        with self.lock:
            for ply in range(len(history) - 1, max(-1, len(history) - REFRESH_PLIES - 1), -1):
                saved = self.lookup(history[:ply])
                if self.stale(saved, ply) and not saved.get('proof') and not any(
                        getattr(j, 'refresh', None) is not None and j.history == tuple(history[:ply])
                        and j.status == 'queued' for j in self.jobs.values()):
                    self.submit(Job('analyse', 2, history[:ply], seat=dict(seat), force=True, game=game, refresh=saved))

    def note_pace(self, seat, ms, stones):
        """Record that Bubble `seat` searched a turn of `stones` stones at its preset in `ms`: the page shows the entry's
        `pace` (ms per two-stone turn, a one-stone turn counted twice) by the strength slider, and each move moves it
        halfway to its own time. A custom budget records nothing."""
        if seat['preset'] not in PRESET_NAMES or not stones:
            return
        with self.lock:
            paces, turn = self.paces.setdefault(seat['engine'], {}), ms * 2 / stones
            old = paces.get(seat['preset'])
            paces[seat['preset']] = round(turn if old is None else (old + turn) / 2)

    def place(self, job, moves):
        """Put the stones of move `job`'s turn decided so far (`moves`, from `evaluate`) on the board before the turn
        ends, while the board still stands where the job left it and the game is not paused: the next stone's search
        goes on, and the finished move adds only the stones not shown (`Job.placed`)."""
        with self.lock:
            stones = tuple(tuple(p) for p in moves)
            shown, at = job.placed, list(job.reached())
            if job.cancelled or self.paused or len(stones) <= len(shown) or stones[:len(shown)] != shown or self.history != at:
                return
            game = replay(at + list(stones[len(shown):]))
            try:
                ongoing = game.winner < 0 and game.player == job.side
            finally:
                game.close()
            if not ongoing:
                return
            if not (self.match and self.match['active']):
                self.fork_freeplay()
            self.history.extend(stones[len(shown):])
            job.placed = stones
            self.changed()

    def save_timed(self, seat, history, moves, found):
        """Save a timed Bubble move's search as a kept-tree evaluation of `history` at the work it completed: shown in
        the game and its replay like any evaluation, never reused for a budget."""
        entry, value = self.entries[seat['engine']], found.get('win_probability')
        key = self.engine_key(seat) if entry['kind'] == 'bubble' and value is not None else None
        if key is None:
            return
        spent = dict(simulations=int(found.get('completed') or 0), solver_nodes=int(found.get('solver_nodes') or 0))
        model = f"{entry['name']}/{seat['checkpoint']}" if seat['checkpoint'] else entry['name']
        self.store.add(history, key + ':kept', spent, dict(value=round(float(value), 4), moves=moves, top=[[*moves[0], 1.0, float(value)]],
                                                           pv=[], threat=[], proof=None, model=model))

    def save(self, history, weights, spent, found, model):
        """Save an evaluation, and the searches of the later stones of its turn (see `TurnSearch`) as evaluations
        of those positions without solver checks; the proof table takes the proofs among them. A saved evaluation
        holding a proof is kept over an unproven one of the same position, engine and budget."""
        later = found.pop('later', [])
        saved = self.store.get(history, weights, spent)
        if saved and saved.get('proof') and not found.get('proof'):
            if found.get('proofs'):
                record = self.store.add(history, weights, spent, found | dict(model=model) |
                                        {k: saved[k] for k in ('moves', 'value', 'top', 'proof', 'pv')})
            else:
                record = saved
        else:
            record = self.store.add(history, weights, spent, found | dict(model=model))
        self.proofs.add(history, record)
        bare = weights.split(':')[0] + ':none' + (':kept' if weights.endswith(':kept') else '')
        for step in later:
            position = step.pop('history')
            self.proofs.add(position, self.store.add(position, bare, spent | dict(solver_nodes=0), step | dict(model=model)))
        return record

    def evaluations(self, job, seat, histories):
        """Fresh evaluations of `histories` for `seat` at its budget, pooled (see `Engines.evaluate_many`), saved."""
        entry, budget = self.entries[seat['engine']], self.engines.effective(seat['budget'])
        found = self.lane_engines(job).evaluate_many(entry, seat['checkpoint'], budget, histories, self.watcher(job, False),
                                                     known=self.proofs,
                                                     **({'device': seat['device']} if 'device' in seat else {}))
        if job.cancelled:
            raise Cancelled()
        model = f"{entry['name']}/{seat['checkpoint']}" if seat['checkpoint'] else entry['name']
        return [self.save(history, weights, spent, record, model)
                for history, (record, spent, weights) in zip(histories, found)]

    def retry(self, history):
        """Analyse `history` again after a solver failure, when automatic analysis is on; at most three times for
        one position, model, solver build and budget, unless that analysis was cancelled since (see `dismissal`)."""
        with self.lock:
            if (self.analysis and self.analysis['auto'] and tuple(history) == tuple(self.history[:len(history)])
                    and self.dismissed != self.dismissal(history)):
                self.request_analysis(history, 1)

    def run(self, job):
        seat, history = job.seat, list(job.history)
        entry = self.entries[seat['engine']]
        if job.kind == 'move':
            started = time.monotonic()
            if self.match and self.match['active']:
                source = self.match['players'][job.side if self.match['current'] % 2 else 1-job.side]['source']
                if seat['budget'].get('solver_nodes') and source['solver_build'] != self.engines.solver_build():
                    raise ValueError('Tactical solver build changed during this batch')
                files = source['files'] | ({source['weights']: source['weights_sha256']} if source.get('weights') else {})
                if any(file_digest(file_identity(path)) != digest for path, digest in files.items()):
                    raise ValueError('An engine or checkpoint file changed during this batch')
            if self.game_clock and (not self.match or self.match['active']):
                with self.lock:
                    if self.match:
                        engine = self.timed_engines[job.side if self.match['current'] % 2 else 1-job.side]
                    else:
                        engine = self.seat_engines[job.side]
                    specification = self.match['clock'] if self.match else self.clock_spec
                    clock = self.game_clock.json() if specification['mode'] == 'game' else None
                    move_ms = self.game_clock.json()['cross_ms' if job.side == 0 else 'circle_ms'] if clock is None else None
                if engine is None:
                    raise ValueError('The timed engine for this seat is not running')
                game = replay(history)
                try:
                    def publish(found):
                        job.done = found.get('completed') or found.get('nodes') or 0
                    try:
                        found = engine.turn(game, move_ms, clock=clock, cancel=job, publish=publish)
                    except TimeoutError:
                        # External adapters stop before the response reserve. Let the host clock
                        # finish that allowance, then score the timeout instead of pausing the batch.
                        with self.lock:
                            self.lock.wait_for(lambda: job.cancelled or self.game_clock.expired(),
                                timeout=max(0, self.game_clock.remaining()[job.side]/1e9))
                        if job.cancelled:
                            raise Cancelled()
                        job.measurements = dict(stop_reason='deadline')
                        return []
                    if job.cancelled:
                        raise Cancelled()
                    job.measurements = {k: found.get(k) for k in ('elapsed_ms', 'completed', 'evaluated', 'solver_nodes',
                                                                  'nodes', 'stop_reason', 'allowance')}
                    moves = checked_turn(history, found['moves'])
                    if not self.match:
                        self.save_timed(seat, history, moves, found)
                    return moves
                finally:
                    game.close()
            if entry['kind'] == 'bubble':
                job.total = max(1, seat['budget']['simulations']) * 2
                found = self.evaluation(job, seat, history)
                moves = found['moves']
                if found.get('actual_completed'):
                    self.note_pace(seat, (time.monotonic() - started) * 1000, len(moves))
                counts = dict(completed=found.get('actual_completed'), solver_nodes=found.get('actual_solver_nodes'))
            else:
                moves = self.engines.turn(entry, seat['budget'], history, lambda: job.cancelled, seat['checkpoint'])
                counts = getattr(self.engines, 'last_turn', {})
            job.measurements = dict(elapsed_ms=(time.monotonic()-started)*1000, completed=None, nodes=None,
                                    evaluated=job.done if entry['kind'] == 'bubble' else None, stop_reason='budget') | counts
            return checked_turn(history, moves)
        if job.kind == 'analyse':
            job.total = max(1, seat['budget']['simulations']) * 2
            return self.evaluation(job, seat, history, job.force)
        identity, budget = self.engine_key(seat), self.engines.effective(seat['budget'])
        plies = review_plies(history)
        # From the last position backwards: what a later position proves is known when an earlier one is searched.
        missing = [p for p in sorted(set(plies), reverse=True) if not self.store.get(history[:p], identity, budget)]
        job.done = len(plies) - len(missing)
        for start in range(0, len(missing), REVIEW_CHUNK):
            if job.cancelled:
                raise Cancelled()
            with self.lock:
                if self.queues['analysis'] and self.queues['analysis'][0][0] < job.priority:
                    raise Yielded()
            self.evaluations(job, seat, [history[:p] for p in missing[start:start + REVIEW_CHUNK]])
            job.done += len(missing[start:start + REVIEW_CHUNK])
            with self.lock:
                self.revision += 1
        incomplete = any(self.store.get(history[:p], identity, budget) is None for p in plies)
        changed = self.engine_key(seat) != identity
        if changed or incomplete and job.tries < 3:
            timer = threading.Timer(1 if changed else 31, self.review_again,
                                    args=(list(history), dict(seat), job.tries + (not changed)))
            timer.daemon = True
            timer.start()
        return None


# HTTP


def import_history(text):
    """A history from HTTTX notation, a replay file ({"history": [...]}) or a JSON list of [q, r]."""
    stripped = text.strip()
    if not stripped:
        raise ValueError('Nothing to import')
    if stripped.startswith(('{', '[')):
        data = json.loads(stripped)
        history = data['history'] if isinstance(data, dict) else data
        if not isinstance(history, list) or any(not isinstance(p, list) or len(p) != 2 or
                                                 any(type(v) is not int for v in p) for p in history):
            raise ValueError('A replay holds a list of [q, r] stones')
        return history
    return [list(p) for p in loads(text).history]


def fetch_json(url, body=None):
    """The JSON answer of `url`, at most 4 MB, to a GET, or to a POST of `body` as JSON; ValueError when it cannot
    be had."""
    request = urllib.request.Request(url, None if body is None else json.dumps(body).encode(),
                                     {'User-Agent': 'bubble-player', 'Accept': 'application/json',
                                      'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read(4 << 20))
    except urllib.error.HTTPError as error:
        raise ValueError('No game or position at that link' if error.code == 404
                         else f'{urlparse(url).hostname} answered {error.code}') from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ValueError(f'Cannot reach {urlparse(url).hostname}') from error


def linked_history(text, fetch=None):
    """The stones behind a link, or None when `text` is not a link of these sites:

    - hexo.did.science and hexo.mineking.dev: /games/<id>, /account/games/<id> and /sandbox/<id>, read from the API
      the site's own page reads (HEXO_SITES). A site's (x, y) is HTTTX's (x + y, -y); the first stone is moved to
      the origin.
    - hexo.tyto.cc: an analysis link `#c=<code>` (decoded here) or a game link `#g=<id>`, whose HTTTX the site
      answers to a POST of {"game_id"} to /game_htttx.

    `fetch(url, body=None)` returns the JSON answer, `fetch_json` by default. Raises ValueError for other pages of
    these sites, stones out of turn order and illegal stones."""
    url = urlparse(text.strip())
    if url.scheme not in ('http', 'https') or url.hostname not in (*HEXO_SITES, 'hexo.tyto.cc'):
        return None
    fetch = fetch or fetch_json
    if url.hostname == 'hexo.tyto.cc':
        if url.fragment.startswith('c='):
            return formats.tyto_loads(url.fragment[2:])
        if re.fullmatch(r'g=[A-Za-z0-9-]{1,64}', url.fragment):
            return import_history(fetch('https://hexo.tyto.cc/game_htttx', dict(game_id=url.fragment[2:]))['htttx'])
        raise ValueError('Paste a Tyto analysis (#c=) or game (#g=) link')
    found = re.fullmatch(r'/(?:account/)?(games|sandbox)/([A-Za-z0-9-]{1,64})/?', url.path)
    if not found:
        raise ValueError('Paste the link of a finished game or a saved sandbox position')
    page, ident = found.groups()
    if page == 'games':
        data = fetch(f'{HEXO_SITES[url.hostname]}/finished-games/{ident}')
        stones = sorted(data['moves'], key=lambda m: m['moveNumber'])
        placed = [(m['x'], m['y'], m['playerId']) for m in stones]
    else:
        data = fetch(f'{HEXO_SITES[url.hostname]}/sandbox-positions/{ident.lower()}')
        stones = sorted(data['gamePosition']['cells'], key=lambda c: c['moveId'])
        placed = [(c['x'], c['y'], c['player']) for c in stones]
    if not placed or any(type(x) is not int or type(y) is not int for x, y, _ in placed):
        raise ValueError('That link holds no stones')
    sides = {}
    for ply, (_, _, owner) in enumerate(placed):
        if sides.setdefault(owner, player_at(ply)) != player_at(ply):
            raise ValueError(f'Stone {ply + 1} breaks the turn order of one stone, then two each')
    origin = placed[0][0] + placed[0][1], -placed[0][1]
    history = [[x + y - origin[0], -y - origin[1]] for x, y, _ in placed]
    replay(history).close()
    return history


def read_game(text, fetch=None):
    """The history in pasted `text`, whichever it is: a link (`linked_history`), HTTTX, a replay file, a JSON list
    of [q, r], or Rectilinear notation. A text that is none of them raises the HTTTX error when it looks like
    HTTTX, else the Rectilinear one."""
    linked = linked_history(text, fetch)
    if linked is not None:
        return linked
    try:
        return import_history(text)
    except ValueError as error:
        try:
            return formats.rectilinear_loads(text)
        except ValueError as other:
            raise (error if '[' in text and ';' in text else other) from None


def export(history, kind):
    """`history` written as `kind` (htttx, rectilinear or tyto): {text, spans}, where each span [start, end, q, r]
    marks the token of one stone."""
    history = [tuple(p) for p in history]
    if kind == 'htttx':
        text = dumps(history)
        tokens = re.finditer(r'\[-?\d+,-?\d+\]', text[text.index(';'):])
        offset = text.index(';')
        spans = [(m.start() + offset, m.end() + offset) for m in tokens]
        return dict(text=text, spans=[[*span, *p] for span, p in zip(spans, history[1:])])
    if kind == 'rectilinear':
        text, spans = formats.rectilinear_dumps(history)
        return dict(text=text, spans=[[*span, *p] for span, p in zip(spans, history)])
    if kind == 'tyto':
        return dict(text=formats.tyto_dumps(history), spans=[])
    raise ValueError('Format must be htttx, rectilinear or tyto')


STATIC_TYPES = {'.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json',
                '.onnx': 'application/octet-stream', '.safetensors': 'application/octet-stream',
                '.html': 'text/html; charset=utf-8'}
# Cross-origin isolation (SharedArrayBuffer for the browser engine's threads) without blocking credentialless subresources
ISOLATION = (('Cross-Origin-Opener-Policy', 'same-origin'), ('Cross-Origin-Embedder-Policy', 'credentialless'))


class Handler(BaseHTTPRequestHandler):
    session = None
    setups = None
    page = ROOT / 'web' / 'index.html'

    def log_message(self, *args):
        pass

    def respond(self, status, data, content_type='application/json', headers=()):
        payload = data if isinstance(data, bytes) else data.encode() if isinstance(data, str) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def static(self, path):
        """A file of the browser engine bundle (web/engine) or web/coi-sw.js, with the cross-origin isolation headers
        that let its WebAssembly use threads."""
        web = self.page.parent.resolve()
        target = (web / path.lstrip('/')).resolve()
        inside = target == web / 'coi-sw.js' or (web / 'engine') in target.parents
        if not inside or target.suffix not in STATIC_TYPES or not target.is_file():
            return self.respond(404, dict(error='Not found'))
        return self.respond(200, target.read_bytes(), STATIC_TYPES[target.suffix], ISOLATION)

    def local(self):
        """True for requests addressed to this server by its loopback name, which keeps pages on other sites out
        even when their hostname resolves to 127.0.0.1; POSTs must also come from such a page."""
        port = self.server.server_address[1]
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        origin = self.headers.get('Origin')
        return self.headers.get('Host') in hosts and (origin is None or origin in {f'http://{h}' for h in hosts})

    def do_GET(self):
        if not self.local():
            return self.respond(403, dict(error='Host rejected'))
        url, session = urlparse(self.path), self.session
        if url.path == '/':
            return self.respond(200, self.page.read_bytes(), 'text/html; charset=utf-8', ISOLATION)
        if url.path.startswith('/engine/') or url.path == '/coi-sw.js':
            return self.static(url.path)
        if url.path == '/setup' and self.setups:
            return self.respond(200, dict(engines=self.setups.catalogue()))
        if url.path.startswith('/study/'):
            session = session.study
            if session is None:
                return self.respond(404, dict(error='Choose a saved game to analyse'))
            url = url._replace(path=url.path[len('/study'):])
        if url.path in ('/matches', '/matches/game'):
            try:
                if url.path == '/matches':
                    return self.respond(200, dict(matches=session.match_catalogue()))
                query = parse_qs(url.query)
                _, game = session.saved_replay(query['batch'][0], int(query['game'][0]))
                if query.get('format') == ['htttx']:
                    return self.respond(200, dumps([tuple(p) for p in game['history']]), 'text/plain; charset=utf-8',
                                        [('Content-Disposition', f'attachment; filename="game-{game["game"]:04d}.htttx"')])
                return self.respond(200, game)
            except (ValueError, KeyError, TypeError, OSError) as error:
                return self.respond(400, dict(error=str(error)))
        if url.path == '/state':
            since = parse_qs(url.query).get('since', [''])[0]
            return self.respond(200, session.poll(int(since)) if since.isdigit() else session.state())
        if url.path == '/models':
            return self.respond(200, dict(api='bubble-player-v1', models=session.models(),
                                          book=str(session.book) if session.book else None))
        if url.path == '/match':
            with session.lock:
                return self.respond(200, dict(match=session.match, paused=session.paused))
        if url.path == '/openings':
            try:
                if not session.book:
                    raise ValueError('No opening book configured; start the player with --book or --dense-run')
                query = parse_qs(url.query)
                count = query.get('count', [None])[0]
                return self.respond(200, book_openings(session.book, query.get('range', ['wide'])[0],
                                                       int(count) if count is not None else None,
                                                       int(query.get('seed', ['0'])[0])))
            except (ValueError, KeyError, TypeError, OSError) as error:
                return self.respond(400, dict(error=str(error)))
        if url.path == '/htttx':
            try:
                return self.respond(200, dumps(list(session.history)), 'text/plain; charset=utf-8')
            except NotationConflict as error:
                return self.respond(409, dict(error=str(error)))
        if url.path == '/export':
            try:
                query = parse_qs(url.query)
                history = list(session.history)
                ply = int(query.get('ply', [len(history)])[0])
                return self.respond(200, export(history[:max(0, ply)], query.get('format', ['htttx'])[0]))
            except ValueError as error:
                return self.respond(409 if isinstance(error, NotationConflict) else 400, dict(error=str(error)))
        if url.path == '/replay':
            names = [seat['engine'] for seat in session.seats]
            body = dict(format='bubble-replay', version=1, players=names, history=[list(p) for p in session.history])
            return self.respond(200, json.dumps(body), headers=[('Content-Disposition', 'attachment; filename="game.json"')])
        if url.path == '/evaluations':
            return self.respond(200, session.store.export(), 'application/x-ndjson',
                                [('Content-Disposition', 'attachment; filename="evaluations.jsonl"')])
        if url.path == '/match/results':
            with session.lock:
                if session.match:
                    return self.respond(200, json.dumps(session.match), headers=[
                        ('Content-Disposition', 'attachment; filename="match.json"')])
            return self.respond(404, dict(error='No match'))
        if url.path == '/match/replay':
            try:
                number = int(parse_qs(url.query).get('game', ['0'])[0])
                with session.lock:
                    if not session.match or not 1 <= number <= session.match['completed']:
                        raise ValueError('No such completed game')
                    path = Path(session.match['output']) / f'game-{number:04d}.htttx'
                    return self.respond(200, path.read_bytes(), 'text/plain; charset=utf-8',
                                        [('Content-Disposition', f'attachment; filename="game-{number:04d}.htttx"')])
            except (ValueError, OSError) as error:
                return self.respond(400, dict(error=str(error)))
        self.respond(404, dict(error='Not found'))

    def do_POST(self):
        if not self.local():
            return self.respond(403, dict(error='Origin rejected'))
        session = self.session
        if self.path.startswith('/study/'):
            session = session.study
            if session is None:
                return self.respond(404, dict(error='Choose a tournament game to analyse'))
            self.path = self.path[len('/study'):]
            if self.path.startswith('/match'):
                return self.respond(400, dict(error='Start tournaments on the live board'))
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 <= length <= 1 << 20:
                raise ValueError('Request too large')
            args = json.loads(self.rfile.read(length) or '{}')
            reply = {}
            if self.path == '/matches/open':
                session.open_saved_game(args['batch'], args['game'])
                return self.respond(200, dict(url='/?study=1'))
            elif self.path == '/play':
                session.play(args['q'], args['r'])
            elif self.path == '/undo':
                session.undo(args.get('people'))
            elif self.path == '/new':
                session.new_game(args.get('people'))
            elif self.path == '/book':
                session.use_book(args['enabled'], args.get('mode'), args.get('people'))
            elif self.path == '/retry':
                ply = args['ply']
                if type(ply) is not int or not 0 <= ply <= len(session.history):
                    raise ValueError('No such position')
                session.load(session.history[:ply], False)
            elif self.path == '/import':
                session.load(read_game(args['text']), True)
            elif self.path == '/seat':
                if args.get('side') not in (0, 1):
                    raise ValueError('Side must be 0 or 1')
                session.configure_seat(args['side'], args['engine'], args.get('checkpoint'),
                                       args.get('preset', 'standard'), args.get('custom'))
            elif self.path == '/analysis':
                session.configure_analysis(args['engine'], args.get('checkpoint'), args.get('preset', 'standard'),
                                           args.get('custom'), args.get('auto', True))
            elif self.path == '/analyse':
                if type(args.get('ply')) is not int:
                    raise ValueError('Choose a position')
                reply = dict(job=session.analyse(args['ply'], args.get('force') is True))
            elif self.path == '/review':
                reply = dict(job=session.review_game())
            elif self.path == '/cancel':
                session.cancel(args['id'])
            elif self.path == '/clock':
                session.set_clock(args)
            elif self.path == '/pause':
                session.pause(args['paused'])
            elif self.path == '/rescan':
                session.rescan()
            elif self.path == '/setup' and self.setups:
                self.setups.start(args['engine'])
                return self.respond(200, dict(engines=self.setups.catalogue()))
            elif self.path == '/match':
                action = args.get('action', 'start')
                if action == 'start':
                    openings = ([import_history(text) for text in args['opening_texts']] if args.get('opening_texts')
                                else args.get('openings'))
                    session.start_match(args['players'], args.get('games'), args.get('preset', 'standard'),
                                        args.get('output'), openings, args.get('max_placements', 512),
                                        args.get('book'), args.get('opening_range'), args.get('unique_openings'),
                                        args.get('seed', 0), clock=args.get('clock'),
                                        replace=args.get('replace') is True)
                elif action == 'resume' and args.get('batch'):
                    session.resume_match(args['batch'])
                elif action in ('pause', 'resume'):
                    if not session.match or not session.match['active']:
                        raise ValueError('No active batch; use resume with a saved batch directory to continue one')
                    session.pause(action == 'pause')
                elif action == 'stop':
                    session.stop_match()
                else:
                    raise ValueError('Match action must be start, pause, resume or stop')
            elif self.path == '/match/stop':
                session.stop_match()
            else:
                return self.respond(404, dict(error='Not found'))
            self.respond(200, session.state() | reply)
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
            self.respond(400, dict(error=str(error)))


def search_child(kind):
    """The child process of `Engines.turn`: one JSON line {history, the budget, model} in, one line {moves} or
    {error} out. A Strix child keeps the two most recent settings loaded, one per seat."""
    engines = OrderedDict()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            measurements = {}
            game = replay(request['history'])
            try:
                if kind == 'drip':
                    found = game.search(request['ms'])
                    moves, measurements = found['moves'], dict(nodes=found.get('nodes'))
                elif kind == 'seal':
                    if request.get('library') not in engines:
                        from legacy.arena import Seal
                        engines[request.get('library')] = Seal(request.get('library'))
                    moves = engines[request.get('library')](game, request['ms'])
                else:
                    key = (request['model'], request['simulations'], request.get('engine'))
                    if key not in engines:
                        if str(ROOT) not in sys.path:
                            sys.path.insert(0, str(ROOT))
                        from tools.strix_learned_adapter import StrixLearned
                        engines[key] = StrixLearned(request['model'], simulations=request['simulations'],
                                                    timeout_ms=min(600_000, max(5_000, 250 * request['simulations'])),
                                                    executable=request.get('engine'))
                        while len(engines) > 2:
                            engines.popitem(last=False)[1].close()
                    engines.move_to_end(key)
                    moves = engines[key](game, 0)
            finally:
                game.close()
            answer = dict(moves=[list(map(int, m)) for m in moves], measurements=measurements)
        except Exception as error:
            answer = dict(error=str(error))
        print(json.dumps(answer), flush=True)


def main():
    if sys.argv[1:2] == ['search']:
        return search_child(sys.argv[2])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--dense-run', type=Path, help='a Bubble run; its champion is the default opponent')
    parser.add_argument('--dense-model', type=Path, help='an ema.pt file; the default opponent, ahead of --dense-run')
    parser.add_argument('--models', type=Path, default=ROOT / 'models', help='folder scanned for models and engines')
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs', help='folder whose runs are offered as models')
    parser.add_argument('--evaluations', type=Path,
                        help='saved evaluations file; default play-evaluations.jsonl in the run or models folder')
    parser.add_argument('--tactical-package', type=Path, help='directory with the built tactical solver')
    parser.add_argument('--proof-stamps', action=argparse.BooleanOptionalAction, default=True,
                        help='reuse checked local strategies and test quiet defender turns (default: enabled)')
    parser.add_argument('--device', default='auto', help='cuda, cpu, or auto: cuda when a GPU is available')
    parser.add_argument('--idle', action='store_true', help='start paused, without automatic analysis, for API clients')
    parser.add_argument('--list-engines', action='store_true', help='list engine ids, names and checkpoints, then exit')
    parser.add_argument('--match', nargs=2, metavar=('A', 'B'), help='play a batch using engine names, ids or unique kinds')
    parser.add_argument('--games', type=int, help='games in --match; defaults to 2 per requested unique opening, else 2')
    parser.add_argument('--preset', choices=PRESET_NAMES, default='standard')
    clocks = parser.add_mutually_exclusive_group()
    clocks.add_argument('--tc', help='shared game clock, seconds+increment, e.g. 180+2')
    clocks.add_argument('--move', type=duration, help='shared time per complete turn, e.g. 5s')
    for side in ('a', 'b'):
        parser.add_argument(f'--{side}-checkpoint', help=f'checkpoint of engine {side.upper()} in --match')
        parser.add_argument(f'--{side}-preset', choices=PRESET_NAMES)
    parser.add_argument('--opening', type=Path, action='append', help='HTTTX or replay opening; repeat for paired openings')
    parser.add_argument('--book', type=Path, help='read-only v2 opening book; defaults to openings.json in --dense-run')
    parser.add_argument('--openings', choices=['narrow', 'wide', 'all'], help='book selection, default wide')
    parser.add_argument('--unique-openings', type=int, help='distinct book openings; each is played with colours swapped')
    parser.add_argument('--seed', type=int, default=0, help='repeatable book selection and ordering')
    parser.add_argument('--max-placements', type=int, default=512, help='cap each match game after a complete turn')
    parser.add_argument('--out', type=Path, help='new match output directory; default artifacts/play/<unique match>')
    args = parser.parse_args()
    from hexo import library
    seal = library.with_name(library.name.replace('hexo', 'hexo_seal'))
    runs = [path for path in (args.dense_model, args.dense_run) if path]
    try:
        import torch
        torch.set_num_threads(2)
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if args.device == 'auto':
        args.device = 'cuda' if cuda else 'cpu'
    find = lambda: scan(args.models, args.runs, runs, seal, args.device)
    entries = find()
    book = args.book or (args.dense_run / 'openings.json' if args.dense_run and
                        (args.dense_run / 'openings.json').exists() else None)
    if args.list_engines:
        for entry in entries.values():
            print(entry['id'])
            if entry.get('checkpoints'):
                print('  checkpoints: ' + ', '.join(entry['checkpoints']))
        return
    if args.match and args.out is None:
        name = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + os.urandom(3).hex()
        args.out = ROOT / 'artifacts' / 'play' / name
    store_path = args.evaluations or ((args.out / 'evaluations.jsonl') if args.match else
                                     (args.dense_run or args.models) / 'play-evaluations.jsonl')
    if not args.match:
        store_path.parent.mkdir(parents=True, exist_ok=True)
    # Bind before starting any engine work: a busy port must not leave a hidden match running.
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        Handler.session = Session(entries, Engines(args.device, args.tactical_package, proof_stamps=args.proof_stamps),
                                  Evaluations(None if args.match else store_path), find, book,
                                  archive=ROOT / 'artifacts' / 'play' / 'matches',
                                  study_store=ROOT / 'artifacts' / 'play' / f'study-{args.port}.jsonl',
                                  analysis_engines=Engines(args.device, args.tactical_package, proof_stamps=args.proof_stamps), save_initial=not args.match)
        Handler.session.models_folder = str(args.models.resolve())
        Handler.setups = Setups(args.models, Handler.session.rescan, lambda: Handler.session.entries)
        try:
            if args.match:
                players = [dict(engine=name, checkpoint=getattr(args, f'{side}_checkpoint'),
                                preset=getattr(args, f'{side}_preset') or args.preset)
                           for side, name in zip(('a', 'b'), args.match)]
                openings = [import_history(path.read_text(encoding='utf-8')) for path in args.opening] if args.opening else None
                Handler.session.start_match(players, args.games, args.preset, args.out, openings, args.max_placements,
                                            opening_range=args.openings, unique_openings=args.unique_openings, seed=args.seed,
                                            evaluations=args.evaluations, clock=dict(mode='game', tc=args.tc) if args.tc else
                                            dict(mode='move', ms=args.move) if args.move else None)
                print(f'Match results: {args.out.resolve()}', flush=True)
            else:
                with Handler.session.lock:
                    if args.idle:
                        Handler.session.paused = True
                        if Handler.session.analysis:
                            Handler.session.analysis['auto'] = False
                    Handler.session.changed()
            print(f'Bubble is ready at http://127.0.0.1:{server.server_port}', flush=True)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            Handler.session.close()


if __name__ == '__main__':
    main()
