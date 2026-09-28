"""Native independently checked tactical strategies, separate from estimated values.

Certificates cover every legal defense, including a free second placement.
An optional root candidate expands those obligations explicitly; quiet defender
nodes remain UNKNOWN. A single native worker limits caller wait; reconstruction
may finish in the background. Late results are not exposed as exact values.
"""
import contextlib
import ctypes as C
import hashlib
import importlib
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

PROVEN_WIN, UNKNOWN = 'PROVEN_WIN', 'UNKNOWN'
PACKAGE = Path(__file__).resolve().parent/'tools/tactical'


def check_budgets(ms, idtt_ms, nodes, depth):
    if (type(ms) is not int or not 1 <= ms <= 60000 or type(idtt_ms) is not int
            or not 0 <= idtt_ms < ms or type(nodes) is not int or not 1 <= nodes <= 10000000
            or type(depth) is not int or not 1 <= depth <= 64):
        raise ValueError('Invalid tactical budgets')


def unknown_result(reason, start):
    return dict(status=UNKNOWN, native_verified=False, moves=[], certificate=None,
                reason=reason, elapsed_ms=(time.perf_counter()-start)*1000)


class NativeTactics:
    def __init__(self, package=PACKAGE):
        package = Path(package)
        name = 'hexo_tactical.dll' if sys.platform == 'win32' else ('libhexo_tactical.dylib' if sys.platform == 'darwin' else 'libhexo_tactical.so')
        binary = package/'target/release'/name
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

    def solve(self, game, *, ms=100, idtt_ms=20, nodes=100000, depth=8, certificate=None, root_moves=None):
        return self.history([cell[:2] for cell in game.cells], ms=ms, idtt_ms=idtt_ms,
                            nodes=nodes, depth=depth, certificate=certificate, root_moves=root_moves)

    def history(self, history, *, ms=100, idtt_ms=20, nodes=100000, depth=8, certificate=None, root_moves=None):
        check_budgets(ms, idtt_ms, nodes, depth)
        start = time.perf_counter()
        unknown = lambda reason: unknown_result(reason, start)
        if not self.lock.acquire(timeout=ms/1000):
            return unknown('lock deadline')
        try:
            remaining = math.floor(ms-(time.perf_counter()-start)*1000)
            if remaining < 1:
                return unknown('deadline')
            request = dict(history=history, ms=remaining, idtt_ms=min(idtt_ms, remaining-1),
                           nodes=nodes, depth=depth)
            if certificate is not None:
                request['certificate'] = certificate
            if root_moves is not None:
                request['root_moves'] = root_moves
            payload = json.dumps(request, separators=(',', ':')).encode()
            if len(payload) > 8*1024*1024:
                return unknown('request size limit')
            output = self.lib.hexo_tactical_query(payload)
            if not output:
                return unknown('null native response')
            try:
                result = json.loads(C.string_at(output))
            finally:
                self.lib.hexo_tactical_free(output)
            if time.perf_counter()-start >= ms/1000:
                result.update(unknown('deadline'))
            result['elapsed_ms'] = (time.perf_counter()-start)*1000
            result['build'] = self.metadata
            return result
        finally:
            self.lock.release()


class IsolatedTactics:
    """A tactical engine in a disposable child process with a hard deadline and a memory cap.

    `history` returns within `ms + grace_ms`. A child that has not answered by then, or that
    reports abandoned native work still running, is killed and replaced by a fresh one, so no
    query ever waits for or competes with an earlier one. The child's private memory is capped
    at `memory_mb`; exceeding it ends the child and the query returns UNKNOWN. `engine` names
    the `module:Class` constructed in the child with `package`.
    """

    def __init__(self, package=PACKAGE, *, grace_ms=100, memory_mb=1536, engine='tactical_proof:NativeTactics'):
        self.command = [sys.executable, str(Path(__file__).resolve()), 'serve', engine, str(Path(package).resolve())]
        self.grace_ms, self.memory_mb = grace_ms, memory_mb
        self.stats = dict(queries=0, spawns=0, kills=0, exits=0)
        self.lock = threading.Lock()
        self.job = _memory_job(memory_mb) if sys.platform == 'win32' else None
        self.process = None
        self._spawn()

    def _spawn(self):
        limit = None
        if sys.platform != 'win32':
            import resource
            cap = self.memory_mb*2**20
            limit = lambda: resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, encoding='utf-8', bufsize=1, preexec_fn=limit)
        if self.job:
            try:
                _assign(self.job, process.pid)
            except OSError:
                process.kill()
                raise
        lines = queue.Queue()
        self.pump = threading.Thread(target=_pump, args=(process.stdout, lines), daemon=True)
        self.pump.start()
        self.process, self.lines, self.ready = process, lines, False
        self.stats['spawns'] += 1

    def _stop(self):
        self.process.kill()
        self.process.wait()
        self.pump.join()
        with contextlib.suppress(OSError):
            self.process.stdin.close()
        self.process.stdout.close()

    def _restart(self, killed):
        self.stats['kills' if killed else 'exits'] += 1
        self._stop()
        self._spawn()

    def _line(self, deadline):
        try:
            line = self.lines.get(timeout=max(0.0, deadline-time.perf_counter()))
        except queue.Empty:
            return 'timeout'
        return 'exit' if line is None else json.loads(line)

    def solve(self, game, **budgets):
        return self.history([cell[:2] for cell in game.cells], **budgets)

    def history(self, history, *, ms=100, idtt_ms=20, nodes=100000, depth=8, certificate=None, root_moves=None):
        check_budgets(ms, idtt_ms, nodes, depth)
        start = time.perf_counter()
        hard = start+(ms+self.grace_ms)/1000
        if not self.lock.acquire(timeout=ms/1000):
            return unknown_result('lock deadline', start)
        try:
            self.stats['queries'] += 1
            if not self.ready:
                line = self._line(start+ms/1000)
                if line == 'timeout':
                    return unknown_result('tactical worker starting', start)
                if line == 'exit' or 'error' in line:
                    self._restart(killed=False)
                    return unknown_result(f"tactical worker failed to start: {line if line == 'exit' else line['error']}", start)
                self.ready = True
            remaining = math.floor(ms-(time.perf_counter()-start)*1000)
            if remaining < 1:
                return unknown_result('deadline', start)
            request = dict(history=history, ms=remaining, idtt_ms=min(idtt_ms, remaining-1), nodes=nodes, depth=depth,
                           certificate=certificate, root_moves=root_moves)
            self.process.stdin.write(json.dumps(request, separators=(',', ':'))+'\n')
            self.process.stdin.flush()
            result = self._line(hard)
            if result == 'timeout':
                self._restart(killed=True)
                return unknown_result('hard deadline; tactical worker killed', start)
            if result == 'exit':
                self._restart(killed=False)
                return unknown_result('tactical worker exited (memory cap or crash)', start)
            if result.get('background_worker_busy'):
                self._restart(killed=True)
            if time.perf_counter()-start >= ms/1000:
                result.update(unknown_result('deadline', start))
            result['elapsed_ms'] = (time.perf_counter()-start)*1000
            return result
        except OSError:
            self._restart(killed=False)
            return unknown_result('tactical worker pipe closed', start)
        finally:
            self.lock.release()

    def close(self):
        if self.process:
            self._stop()
            self.process = None
        if self.job:
            C.windll.kernel32.CloseHandle(self.job)
            self.job = None


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
    for line in stream:
        lines.put(line)
    lines.put(None)


def _serve(engine, package):
    """Child side of IsolatedTactics: one JSON request per stdin line, one JSON result per stdout line."""
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
        result.pop('build', None)
        print(json.dumps(result, separators=(',', ':')), flush=True)


def independent_verify(certificate, history):
    """Second checker via proof.py, independent of both native search and verifier."""
    from proof import verify
    work = 0
    def expand(index, stack):
        nonlocal work
        work += 1
        if (type(index) is not int or not 0 <= index < len(certificate['nodes']) or
                index in stack or len(stack) >= 128 or work > 50000):
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
    n = len(history)
    attacker = ((n+1)//2) % 2 if n else 0
    converted = dict(version=1, history=[list(p) for p in history], attacker=attacker, tree=expand(certificate['root'], set()))
    return verify(converted, history, deadline=time.perf_counter()+10)


if __name__ == '__main__' and sys.argv[1:2] == ['serve']:
    _serve(*sys.argv[2:4])
