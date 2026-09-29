"""Read Seal loss positions and measure certificate-breaking turns without loading a network.

python tools/measure_defence.py --report <main-065000-vs-seal/report.json> --package <matching tactical package>
Writes JSON to stdout only. One BelowNormal process, two OpenMP threads, 1900 MiB Windows job limit.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_solver import defence_hit, defence_turns
from tactical_proof import NativeTactics, PACKAGE, PROVEN_WIN, _assign, _memory_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    parser.add_argument('--nodes', type=int, default=27000)
    parser.add_argument('--candidates', type=int, default=8)
    args = parser.parse_args()
    resources = dict(omp_threads=2)
    if sys.platform == 'win32':
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError()
        job = _memory_job(1900)
        _assign(job, os.getpid())
        kernel.GetPriorityClass.argtypes = (ctypes.c_void_p,)
        resources.update(priority_class=kernel.GetPriorityClass(kernel.GetCurrentProcess()), memory_limit_mib=1900)
    solver = NativeTactics(args.package)
    games = json.loads(args.report.read_text())['games']
    rows, start = [], time.perf_counter()
    for pair in (15, 28, 26):
        game = next(g for g in games if g['pair'] == pair and g['winner'] != g['challenger_color'])
        for ply in range(7 if pair != 26 else 5, 32, 4):
            history = game['moves'][:ply]
            began = time.perf_counter()
            threat = solver.history(history, attacker='opponent', nodes=args.nodes, ms=60000, table_mb=0)
            row = dict(pair=pair, ply=ply, threat_status=threat['status'], threat_nodes=threat['nodes_used'],
                       candidates=[])
            rows.append(row)
            if threat['status'] == PROVEN_WIN and threat['native_verified']:
                for turn in defence_turns(history, threat['certificate'], args.candidates):
                    result = solver.history(history+[list(c) for c in turn], nodes=args.nodes, ms=60000, table_mb=0)
                    row['candidates'].append(dict(turn=turn, survives=defence_hit(result), status=result['status'],
                                                  reason=result['reason'], nodes=result['nodes_used']))
            row['elapsed_ms'] = (time.perf_counter()-began)*1000
            row['defence_nodes'] = sum(c['nodes'] for c in row['candidates'])
            row['survivors'] = sum(c['survives'] for c in row['candidates'])
            print(f"pair {pair} ply {ply}: {row['survivors']}/{len(row['candidates'])}, "
                  f"{row['defence_nodes']} defence nodes", file=sys.stderr, flush=True)
    if sys.platform == 'win32':
        class Memory(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_uint32), ('faults', ctypes.c_uint32)] + [
                (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged', 'paged', 'peak_nonpaged',
                                                    'nonpaged', 'commit', 'peak_commit', 'private')]
        memory = Memory()
        memory.cb = ctypes.sizeof(memory)
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = (ctypes.c_void_p, ctypes.POINTER(Memory), ctypes.c_uint32)
        if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb):
            raise ctypes.WinError()
        resources.update(peak_rss_bytes=memory.peak_rss, peak_commit_bytes=memory.peak_commit)
    print(json.dumps(dict(build_hash=solver.metadata['binary_sha256'], nodes=args.nodes, candidates=args.candidates,
                          seconds=time.perf_counter()-start, resources=resources, rows=rows), indent=2))


if __name__ == '__main__':
    main()
