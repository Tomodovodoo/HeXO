"""Native independently checked tactical strategies, separate from estimated values.

Certificates cover every legal defense, including a free second placement.
An optional root candidate expands those obligations explicitly; quiet defender
nodes remain UNKNOWN. A single native worker limits caller wait; reconstruction
may finish in the background. Late results are not exposed as exact values.

Budgets: `nodes` bounds the total native search work (IDTT nodes plus PDS-PN
level-1 nodes and level-2 expansions; `idtt_nodes` of it go to the optional IDTT
probe), so a verdict and its certificate depend only on (position, attacker,
nodes, idtt_nodes, build). The native checker permits at most
min(200000, max(50000, 8*nodes)) certificate nodes and visits. This limit is
derived from the granted node budget, including any gate adjustment. `ms` is a
safety cap: a query that reaches it returns UNKNOWN with reason 'deadline'.

`gate` = dict(weight, floor, cap_low, cap_high) sizes the budget by the attacker's
forcing material (forcing_material.gate_level g of the queried position): `floor`
nodes when g is None, else min(cap_low + (cap_high-cap_low)*g, round(nodes*(1+weight*g))).
The score is computed where the query runs (in the worker process for IsolatedTactics),
so the verdict stays a function of (position, attacker, nodes, gate, build). Results
carry the granted `budget` and `gate_score` (None without a gate).

`table_mb` > 0 keeps the worker thread's transposition table and proven-node set (that
many megabytes per attacker colour) across queries; a verdict then also depends on the
earlier queries of the same worker, never on anything unverified.

`attacker='mover'` asks whether the side to move has a forced win.
`attacker='opponent'` asks whether its opponent, moving now with a fresh
two-placement turn on the current stones, has one; `threat_cells` of that
certificate names the threatening first turn.
"""
import contextlib
import ctypes as C
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

PROVEN_WIN, UNKNOWN = 'PROVEN_WIN', 'UNKNOWN'
PACKAGE = Path(__file__).resolve().parents[1]/'tools/tactical'
MAX_NODES = 10000000
MAX_TABLE_MB = 256  # resident table per attacker colour; two of them stay well inside a worker's 1536 MB cap
DEFAULT_NODES, DEFAULT_MS = 2500, 1000
REQUEST_LIMIT = 64*1024*1024
# Room for a certificate at the 200,000-node native cap.
RESPONSE_LIMIT = 64*1024*1024
# IsolatedTactics worker priorities: (Windows priority class, POSIX nice increment); None inherits.
PRIORITIES = {None: None, 'below_normal': (0x4000, 5), 'idle': (0x40, 19)}


def check_budgets(ms, nodes, idtt_nodes, depth, attacker, gate=None, table_mb=0):
    counts = [nodes] if gate is None else [nodes, gate.get('floor'), gate.get('cap_low'), gate.get('cap_high')]
    if (type(ms) is not int or not 1 <= ms <= 60000 or any(type(n) is not int or not 1 <= n <= MAX_NODES for n in counts)
            or type(idtt_nodes) is not int or not 0 <= idtt_nodes < min(counts)
            or type(depth) is not int or not 1 <= depth <= 64 or attacker not in ('mover', 'opponent')
            or type(table_mb) is not int or not 0 <= table_mb <= MAX_TABLE_MB
            or (gate is not None and (set(gate) != {'weight', 'floor', 'cap_low', 'cap_high'}
                                      or not 0 <= gate['weight'] <= 100 or gate['cap_low'] > gate['cap_high']))):
        raise ValueError('Invalid tactical budgets')


def gated_nodes(history, attacker, nodes, gate):
    """(node budget, forcing-material score) of a query under `gate` (module contract)."""
    if gate is None:
        return nodes, None
    from forcing_material import forcing_material, gate_level
    from hexo import Game
    game = Game([tuple(p) for p in history])
    try:
        score = forcing_material(game, game.player if attacker == 'mover' else 1-game.player)
    finally:
        game.close()
    g = gate_level(score)
    if g is None:
        return gate['floor'], score
    return min(round(gate['cap_low']+(gate['cap_high']-gate['cap_low'])*g), round(nodes*(1+gate['weight']*g))), score


def unknown_result(reason, start, attacker, build_hash=None):
    """An UNKNOWN result with every documented field; `build_hash` is None when no library answered."""
    return dict(status=UNKNOWN, native_verified=False, moves=[], certificate=None, proof_turns=None,
                nodes_used=0, attacker=attacker, build_hash=build_hash, reason=reason,
                elapsed_ms=(time.perf_counter()-start)*1000)


def library(package=PACKAGE):
    """Path of the built native library of `package`; its identity record is the same path plus '.json'."""
    name = 'hexo_tactical.dll' if sys.platform == 'win32' else ('libhexo_tactical.dylib' if sys.platform == 'darwin' else 'libhexo_tactical.so')
    return Path(package)/'target/release'/name


def build_hash(package=PACKAGE):
    """The recorded SHA-256 of `package`'s built library: the `build_hash` its results carry."""
    binary = library(package)
    return json.loads(binary.with_suffix(binary.suffix+'.json').read_text(encoding='utf-8'))['binary_sha256']


class NativeTactics:
    """In-process native solver; one query at a time.

    Every result carries `status`, `moves` (the verified first turn), `certificate`,
    `nodes_used` (search work charged against the budget), `budget` and `gate_score`
    (module contract), `proof_turns` (most attacker turns on any certificate path, the
    completing turn included; None unless PROVEN_WIN), `attacker` and `build_hash`
    (SHA-256 of the loaded library). Verification accepts at most
    min(200000, max(50000, 8*budget)) certificate nodes and visits.
    """

    def __init__(self, package=PACKAGE):
        package = Path(package)
        binary = library(package)
        self.metadata = json.loads(binary.with_suffix(binary.suffix+'.json').read_text(encoding='utf-8'))
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        if (digest(binary) != self.metadata['binary_sha256'] or
                any(digest(package/p) != value for p, value in self.metadata['sources'].items())):
            raise ValueError('Tactical build identity changed; rebuild the native library')
        self.lib = C.CDLL(str(binary))
        self.lib.hexo_tactical_query.argtypes = [C.c_char_p]
        self.lib.hexo_tactical_query.restype = C.c_void_p
        self.lib.hexo_tactical_free.argtypes = [C.c_void_p]
        self.lib.hexo_tactical_free.restype = None
        self.lock = threading.Lock()

    def solve(self, game, **budgets):
        return self.history([cell[:2] for cell in game.cells], **budgets)

    def history(self, history, *, nodes=DEFAULT_NODES, ms=DEFAULT_MS, idtt_nodes=0, depth=8, attacker='mover',
                certificate=None, root_moves=None, gate=None, table_mb=0):
        check_budgets(ms, nodes, idtt_nodes, depth, attacker, gate, table_mb)
        start = time.perf_counter()
        nodes, score = gated_nodes(history, attacker, nodes, gate)
        unknown = lambda reason: dict(unknown_result(reason, start, attacker, self.metadata['binary_sha256']),
                                      budget=nodes, gate_score=score)
        if not self.lock.acquire(timeout=ms/1000):
            return unknown('lock deadline')
        try:
            remaining = math.floor(ms-(time.perf_counter()-start)*1000)
            if remaining < 1:
                return unknown('deadline')
            request = dict(history=history, ms=remaining, nodes=nodes, idtt_nodes=idtt_nodes, depth=depth,
                           attacker=attacker, table_mb=table_mb)
            if certificate is not None:
                request['certificate'] = certificate
            if root_moves is not None:
                request['root_moves'] = root_moves
            payload = json.dumps(request, separators=(',', ':')).encode()
            if len(payload) > REQUEST_LIMIT:
                return unknown('request size limit')
            output = self.lib.hexo_tactical_query(payload)
            if not output:
                return unknown('null native response')
            try:
                result = unknown('native error') | json.loads(C.string_at(output))
            finally:
                self.lib.hexo_tactical_free(output)
            if time.perf_counter()-start >= ms/1000:
                result.update(unknown('deadline'), nodes_used=result['nodes_used'])
            result.update(elapsed_ms=(time.perf_counter()-start)*1000, budget=nodes, gate_score=score)
            return result
        finally:
            self.lock.release()


class IsolatedTactics:
    """A tactical engine in a disposable child process with a hard deadline and a memory cap.

    `history` returns within `ms + grace_ms`. A child that has not answered by then, or that
    reports abandoned native work still running, is killed at once; reaping it and starting
    its replacement happen off the caller's clock, and the next query waits for the
    replacement only within its own budget. No query competes with an earlier one. The
    child's private memory is capped at `memory_mb` before it loads the engine; exceeding it
    ends the child and the query returns UNKNOWN. A child not ready `startup_ms` after it
    was started is replaced. `engine` names the `module:Class` constructed in the child with
    `package`. `priority` ('below_normal', 'idle' or None: inherited) is the child's CPU
    scheduling class, set by the child before it loads the engine.

    `history` takes the same budgets, `attacker`, `gate` and `table_mb` as `NativeTactics.history`, and its
    results carry the same `nodes_used`, `budget`, `gate_score`, `proof_turns`, `attacker` and
    `build_hash` (None when no worker answered; `budget` and `gate_score` None too). A verdict depends on the node budget only; a kill at the hard deadline is a
    failure to investigate, not a verdict.
    Results carry `certificate=None` and the strategy as undecoded JSON text in
    `certificate_json`; decoding it is left to the caller, outside the deadline.
    """

    def __init__(self, package=PACKAGE, *, grace_ms=100, memory_mb=1536, startup_ms=10000,
                 engine='tactical_proof:NativeTactics', priority=None):
        if priority not in PRIORITIES:
            raise ValueError(f'Unknown worker priority {priority!r}')
        self.command = [sys.executable, str(Path(__file__).resolve()), 'serve', engine, str(Path(package).resolve()),
                        str(memory_mb), str(priority)]
        self.grace_ms, self.startup_ms = grace_ms, startup_ms
        self.stats = dict(queries=0, spawns=0, kills=0, exits=0)
        self.lock = threading.Lock()
        self.job = _memory_job(memory_mb) if sys.platform == 'win32' else None
        self.replacement = None
        self._spawn()

    def _spawn(self):
        process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, encoding='utf-8', bufsize=1)
        try:
            if self.job:
                _assign(self.job, process.pid)
            # The child loads the engine only after this line, so the cap already applies.
            process.stdin.write('go\n')
            process.stdin.flush()
        except OSError:
            process.kill()
            raise
        lines = queue.Queue()
        self.pump = threading.Thread(target=_pump, args=(process.stdout, lines), daemon=True)
        self.pump.start()
        self.process, self.lines, self.ready = process, lines, False
        self.started = time.perf_counter()
        self.stats['spawns'] += 1

    def _retire(self, killed):
        """Kill the child now; reap it and start its replacement in the background."""
        self.stats['kills' if killed else 'exits'] += 1
        self.process.kill()
        self.ready = False
        self.replacement = threading.Thread(target=self._replace, args=(self.process, self.pump), daemon=True)
        self.replacement.start()

    def _replace(self, process, pump):
        _reap(process, pump)
        with contextlib.suppress(OSError):  # the next query's write fails and retires it again
            self._spawn()

    def _line(self, deadline):
        try:
            line = self.lines.get(timeout=max(0.0, deadline-time.perf_counter()))
        except queue.Empty:
            return 'timeout'
        return 'exit' if line is None else line

    def solve(self, game, **budgets):
        return self.history([cell[:2] for cell in game.cells], **budgets)

    def history(self, history, *, nodes=DEFAULT_NODES, ms=DEFAULT_MS, idtt_nodes=0, depth=8, attacker='mover',
                certificate=None, root_moves=None, gate=None, table_mb=0):
        check_budgets(ms, nodes, idtt_nodes, depth, attacker, gate, table_mb)
        start = time.perf_counter()
        hard = start+(ms+self.grace_ms)/1000
        unknown = lambda reason, build_hash=None: dict(unknown_result(reason, start, attacker, build_hash), budget=None,
                                                       gate_score=None)
        if not self.lock.acquire(timeout=ms/1000):
            return unknown('lock deadline')
        try:
            self.stats['queries'] += 1
            if self.replacement:
                self.replacement.join(timeout=max(0.0, start+ms/1000-time.perf_counter()))
                if self.replacement.is_alive():
                    return unknown('tactical worker restarting')
                self.replacement = None
            if not self.ready:
                line = self._line(min(start+ms/1000, self.started+self.startup_ms/1000))
                if line == 'timeout':
                    if time.perf_counter() >= self.started+self.startup_ms/1000:
                        self._retire(killed=True)
                        return unknown('tactical worker not ready; replaced')
                    return unknown('tactical worker starting')
                if line in ('exit', 'oversize') or 'error' in line:
                    self._retire(killed=False)
                    reason = line if isinstance(line, str) else line['error']
                    return unknown(f'tactical worker failed to start: {reason}')
                self.ready = True
            remaining = math.floor(ms-(time.perf_counter()-start)*1000)
            if remaining < 1:
                return unknown('deadline')
            request = dict(history=history, ms=remaining, nodes=nodes, idtt_nodes=idtt_nodes, depth=depth,
                           attacker=attacker, certificate=certificate, root_moves=root_moves, gate=gate,
                           table_mb=table_mb)
            payload = json.dumps(request, separators=(',', ':'))
            if len(payload) > REQUEST_LIMIT:
                return unknown('request size limit')
            self.process.stdin.write(payload+'\n')
            self.process.stdin.flush()
            result = self._line(hard)
            if result == 'timeout':
                self._retire(killed=True)
                return unknown('hard deadline; tactical worker killed')
            if result == 'exit':
                self._retire(killed=False)
                return unknown('tactical worker exited (memory cap or crash)')
            if result == 'oversize':
                self._retire(killed=True)
                return unknown('response size limit')
            if result.get('background_worker_busy'):
                self._retire(killed=True)
            if time.perf_counter()-start >= ms/1000:
                result.update(unknown('deadline', result.get('build_hash')), nodes_used=result.get('nodes_used', 0),
                              budget=result.get('budget'), gate_score=result.get('gate_score'))
            result['elapsed_ms'] = (time.perf_counter()-start)*1000
            return result
        except OSError:
            self._retire(killed=False)
            return unknown('tactical worker pipe closed')
        finally:
            self.lock.release()

    def close(self):
        with self.lock:
            if self.replacement:
                self.replacement.join()
                self.replacement = None
            if self.process:
                self.process.kill()
                _reap(self.process, self.pump)
                self.process = None
            if self.job:
                C.windll.kernel32.CloseHandle(self.job)
                self.job = None


def _reap(process, pump):
    process.wait()
    pump.join()
    with contextlib.suppress(OSError):
        process.stdin.close()
    process.stdout.close()


class _BasicLimits(C.Structure):
    _fields_ = [('PerProcessUserTimeLimit', C.c_int64), ('PerJobUserTimeLimit', C.c_int64), ('LimitFlags', C.c_uint32),
                ('MinimumWorkingSetSize', C.c_size_t), ('MaximumWorkingSetSize', C.c_size_t),
                ('ActiveProcessLimit', C.c_uint32), ('Affinity', C.c_size_t), ('PriorityClass', C.c_uint32),
                ('SchedulingClass', C.c_uint32)]


class _ExtendedLimits(C.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimits), ('IoInfo', C.c_uint64*6), ('ProcessMemoryLimit', C.c_size_t),
                ('JobMemoryLimit', C.c_size_t), ('PeakProcessMemoryUsed', C.c_size_t), ('PeakJobMemoryUsed', C.c_size_t)]


def _memory_job(memory_mb):
    """Windows job object: per-process committed-memory cap; members die when the owner closes it."""
    kernel32 = C.windll.kernel32
    kernel32.CreateJobObjectW.restype = C.c_void_p
    kernel32.SetInformationJobObject.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_uint32]
    kernel32.AssignProcessToJobObject.argtypes = [C.c_void_p, C.c_void_p]
    kernel32.CloseHandle.argtypes = [C.c_void_p]
    job = kernel32.CreateJobObjectW(None, None)
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x100 | 0x2000  # PROCESS_MEMORY | KILL_ON_JOB_CLOSE
    limits.ProcessMemoryLimit = memory_mb*2**20
    if not job or not kernel32.SetInformationJobObject(job, 9, C.byref(limits), C.sizeof(limits)):
        raise OSError('Unable to create the tactical worker job object')
    return job


def _assign(job, pid):
    kernel32 = C.windll.kernel32
    kernel32.OpenProcess.restype = C.c_void_p
    handle = kernel32.OpenProcess(0x0101, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    try:
        if not handle or not kernel32.AssignProcessToJobObject(job, handle):
            raise OSError('Unable to apply the tactical worker memory cap')
    finally:
        if handle:
            kernel32.CloseHandle(handle)


def _pump(stream, lines):
    """Forward worker results: a small decoded header plus, when present, the raw certificate line.

    A line over RESPONSE_LIMIT becomes 'oversize' and ends the stream.
    """
    def read():
        line = stream.readline(RESPONSE_LIMIT+1)
        return line if line.endswith('\n') else ('oversize' if line else None)
    while (line := read()) not in (None, 'oversize'):
        result = json.loads(line)
        if result.pop('has_certificate', False):
            line = read()
            if line in (None, 'oversize'):
                break
            result['certificate_json'] = line[:-1]
        lines.put(result)
    lines.put(line)


def _serve(engine, package, memory_mb, priority='None'):
    """Child side of IsolatedTactics: one JSON request per stdin line, one JSON result per stdout line.

    On POSIX the child caps its own address space and lowers its priority before loading
    the engine; the parent never runs code between fork and exec.
    """
    sys.stdin.readline()  # the parent has applied the job-object cap (Windows)
    level = PRIORITIES[None if priority == 'None' else priority]
    if sys.platform == 'win32':
        if level:
            C.windll.kernel32.SetPriorityClass(C.c_void_p(-1), level[0])
    else:
        import resource
        cap = int(memory_mb)*2**20
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        if level:
            os.nice(level[1])
    module, name = engine.split(':')
    try:
        tactics = getattr(importlib.import_module(module), name)(package)
    except Exception as error:
        print(json.dumps(dict(error=f'{type(error).__name__}: {error}')), flush=True)
        return
    print(json.dumps(dict(ready=True)), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        result = tactics.history(request.pop('history'), **request)
        certificate = result.pop('certificate', None)
        result.update(certificate=None, has_certificate=certificate is not None)
        print(json.dumps(result, separators=(',', ':')), flush=True)
        if certificate is not None:
            print(json.dumps(certificate, separators=(',', ':')), flush=True)


def threat_cells(certificate):
    """The cells of a certificate's first attacking turn, as (q, r) tuples.

    For a verified attacker='opponent' certificate these are the placements of the
    opponent's forced win if it moved now; they equal the result's `moves`.
    """
    node = certificate['nodes'][certificate['root']]
    if node['kind'] not in ('immediate_win', 'attacker_move'):
        raise ValueError('Certificate root is not an attacking turn')
    return [tuple(cell) for cell in node['action']]


def independent_verify(certificate, history, attacker='mover', deadline_seconds=10.):
    """Second checker via proof.py, independent of both native search and verifier.

    `attacker` is the query's attacker: 'opponent' checks the certificate on the
    flipped-turn position. Returns PROVEN_WIN, raises ValueError for an invalid
    certificate and proof.VerificationTimeout when the check takes longer than
    `deadline_seconds`. This checker allows up to 200,000 visits, the native
    ceiling; native acceptance also depends on the query's node budget.
    """
    from proof import verify
    work = 0
    def expand(index, stack):
        nonlocal work
        work += 1
        if (type(index) is not int or not 0 <= index < len(certificate['nodes']) or
                index in stack or len(stack) >= 128 or work > 200000):
            raise ValueError('Invalid certificate edge, cycle, depth or work limit')
        node = certificate['nodes'][index]
        stack = stack | {index}
        if node['kind'] == 'immediate_win':
            return dict(kind='move', moves=node['action'], child=dict(kind='terminal'))
        if node['kind'] == 'attacker_move':
            return dict(kind='move', moves=node['action'], child=expand(node['child'], stack))
        if node['kind'] == 'unstoppable':
            return dict(kind='uncovered')
        if node['kind'] == 'defender_replies':
            return dict(kind='defenses_all', branches=[dict(moves=r['action'], child=expand(r['child'], stack))
                                                   for r in node['responses']])
        raise ValueError('Unknown certificate node')
    if certificate['version'] != 1 or certificate['width'] != 'wide':
        raise ValueError('Unsupported certificate schema')
    if attacker not in ('mover', 'opponent'):
        raise ValueError('Unknown attacker')
    n = len(history)
    flipped = attacker == 'opponent'
    start = n+1+n % 2 if flipped else n
    converted = dict(version=1, history=[list(p) for p in history], attacker=((start+1)//2) % 2 if start else 0,
                     flipped=flipped, tree=expand(certificate['root'], set()))
    return verify(converted, history, deadline=time.perf_counter()+deadline_seconds)


if __name__ == '__main__' and sys.argv[1:2] == ['serve']:
    _serve(*sys.argv[2:6])
