"""Native independently checked tactical strategies, separate from estimated values.

Certificates cover every legal defense, including a free second placement.
An optional root candidate expands those obligations explicitly; quiet defender
nodes remain UNKNOWN. A single native worker limits caller wait; reconstruction
may finish in the background. Late results are not exposed as exact values.
"""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

PROVEN_WIN, UNKNOWN = 'PROVEN_WIN', 'UNKNOWN'
PACKAGE = Path(__file__).resolve().parent/'tools/tactical'


class NativeTactics:
    def __init__(self):
        name = 'hexo_tactical.dll' if sys.platform == 'win32' else ('libhexo_tactical.dylib' if sys.platform == 'darwin' else 'libhexo_tactical.so')
        binary = PACKAGE/'target/release'/name
        self.metadata = json.loads(binary.with_suffix(binary.suffix+'.json').read_text(encoding='utf-8'))
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        if (digest(binary) != self.metadata['binary_sha256'] or
                any(digest(PACKAGE/p) != value for p, value in self.metadata['sources'].items())):
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
        if (type(ms) is not int or not 1 <= ms <= 60000 or type(idtt_ms) is not int
                or not 0 <= idtt_ms < ms or type(nodes) is not int or not 1 <= nodes <= 10000000
                or type(depth) is not int or not 1 <= depth <= 64):
            raise ValueError('Invalid tactical budgets')
        start = time.perf_counter()
        unknown = lambda reason: dict(status=UNKNOWN, native_verified=False, moves=[], certificate=None,
                                     reason=reason, elapsed_ms=(time.perf_counter()-start)*1000)
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
