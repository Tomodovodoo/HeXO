"""Run the test modules in parallel processes: `python -m tests [-j N] [PATTERN ...]`.

Without HEXO_SLOW=1 this is the fast tier; with it, every test. A PATTERN is a glob over module names
(`test_web_*`); one starting with `!` excludes. Each module runs in its own `python -m unittest` process, largest
file first; a failing module's output is printed in full and the exit status is 1.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path

from tests import SLOW

HERE = Path(__file__).resolve().parent


def modules(patterns):
    keep = [p for p in patterns if not p.startswith('!')]
    drop = [p[1:] for p in patterns if p.startswith('!')]
    found = [p for p in HERE.glob('test_*.py')
             if (not keep or any(fnmatch(p.stem, k) for k in keep)) and not any(fnmatch(p.stem, d) for d in drop)]
    return [p.stem for p in sorted(found, key=lambda p: -p.stat().st_size)]


def run(name):
    start = time.monotonic()
    done = subprocess.run([sys.executable, '-m', 'unittest', f'tests.{name}'], cwd=HERE.parent, capture_output=True,
                          text=True, encoding='utf-8', errors='replace')
    return name, done, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(prog='python -m tests', description=__doc__.split('\n')[0])
    parser.add_argument('-j', '--jobs', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('patterns', nargs='*')
    args = parser.parse_args()
    names = modules(args.patterns)
    if not names:
        sys.exit(f'no test module matches {args.patterns}')
    start, failed = time.monotonic(), []
    with ThreadPoolExecutor(args.jobs) as pool:
        for future in as_completed([pool.submit(run, name) for name in names]):
            name, done, seconds = future.result()
            lines = done.stderr.strip().splitlines() or ['no output']
            summary = ' '.join([line for line in lines if line.startswith('Ran ')][-1:] + lines[-1:])
            print(f'{"ok  " if done.returncode == 0 else "FAIL"} {name:28} {seconds:6.1f}s  {summary}', flush=True)
            if done.returncode:
                failed.append(name)
                print(done.stdout + done.stderr, flush=True)
    tier = 'all tests' if SLOW else 'fast tier'
    print(f'{tier}: {len(names) - len(failed)}/{len(names)} modules passed in {time.monotonic() - start:.0f}s'
          + (f'; failed: {" ".join(failed)}' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
