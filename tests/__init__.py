"""Local correctness checks and reproducible engine measurements."""
import os
import unittest

# Classes too slow for the per-PR CI job, which sets HEXO_SKIP_SLOW; local runs and the nightly job run them.
slow = unittest.skipIf(os.environ.get('HEXO_SKIP_SLOW'), 'slow (HEXO_SKIP_SLOW is set)')
