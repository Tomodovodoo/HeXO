"""Local correctness checks and reproducible engine measurements."""
import os
import unittest

# Classes too slow for the per-PR CI job, which sets HEXO_SKIP_SLOW; local runs and the nightly job run them.
slow = unittest.skipIf(os.environ.get('HEXO_SKIP_SLOW'), 'slow (HEXO_SKIP_SLOW is set)')
# Seconds a test waits for a condition (a batch, an event, a finished job) before it fails. Waits end as soon as the
# condition holds, so only a failing test spends this long; a loaded machine never decides the outcome.
PATIENCE = 60.
# Solver time cap (the most a query accepts) for queries whose node budget must decide the answer.
QUERY_MS = 60_000
