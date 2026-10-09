"""Local correctness checks and reproducible engine measurements."""
import os
import unittest

# The slow tier: real games, real solves, exports, spawned engines and load reproduction. It runs only with
# HEXO_SLOW=1 (locally, and in a manual dispatch of the CI workflows); everything else is the fast tier.
SLOW = os.environ.get('HEXO_SLOW') == '1'
slow = unittest.skipUnless(SLOW, 'slow tier (set HEXO_SLOW=1)')
# Seconds a test waits for a condition (a batch, an event, a finished job) before it fails. Waits end as soon as the
# condition holds, so only a failing test spends this long; a loaded machine never decides the outcome.
PATIENCE = 60.
# Solver time cap (the most a query accepts) for queries whose node budget must decide the answer.
QUERY_MS = 60_000


def resident_bytes():
    """This process's resident memory: its working set on Windows, its resident pages elsewhere."""
    if os.name == 'nt':
        import ctypes as C
        class Counters(C.Structure):
            _fields_ = [('cb', C.c_uint32), ('faults', C.c_uint32)] + [(name, C.c_size_t) for name in (
                'peak_resident', 'resident', 'peak_paged_pool', 'paged_pool', 'peak_nonpaged_pool', 'nonpaged_pool',
                'pagefile', 'peak_pagefile')]
        counters = Counters(cb=C.sizeof(Counters))
        current, read = C.windll.kernel32.GetCurrentProcess, C.windll.psapi.GetProcessMemoryInfo
        current.restype = C.c_void_p
        read.argtypes = (C.c_void_p, C.POINTER(Counters), C.c_uint32)
        if not read(current(), C.byref(counters), counters.cb):
            raise C.WinError()
        return counters.resident
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1])*os.sysconf('SC_PAGE_SIZE')
