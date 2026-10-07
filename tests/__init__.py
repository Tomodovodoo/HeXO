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
