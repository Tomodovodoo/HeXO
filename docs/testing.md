# Testing

Tests check what the system answers or can do from the outside: legal turns, verified proofs, rows the learner can
read, a clock that ends a turn. They do not pin implementation details such as node counts, call orders, hashes of
outputs or bit-identical results against an older build. A test that fails because a legitimate change moved such a
detail gets rewritten as a capability check or deleted.

## Two tiers

The fast tier holds every contract and invariant test. It runs on every pull request and before every push.

The slow tier holds tests whose cost is the point: real games, real solver searches, ONNX exports, Seal matches,
spawned worker processes and load reproduction. A slow test carries the `@slow` decorator from `tests/__init__.py`
on its class or method and runs only with `HEXO_SLOW=1`. When a capability can only be shown by a real search, the
search goes in the slow tier and a cheap fast-tier test checks the surrounding contract.

## Commands

Build the native libraries in the checkout first (`cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release`,
`cmake --build build -j 4`, `python tools/build_tactical.py`). From the checkout root, with `python/` and the root on
`PYTHONPATH` (or the package installed with `pip install -e .`):

```sh
python -m tests                       # the fast tier; run this before every push
HEXO_SLOW=1 python -m tests           # every test
python -m tests 'test_play' 'test_dense*'   # some modules; a pattern starting with ! excludes
python -m unittest tests.test_play.Matches -v   # one class while working on it
```

`python -m tests` runs each module in its own process, four at a time by default (`-j N`), largest first, and
prints one line per module. It exits with status 1 and prints the failing modules' output when anything fails.
Without node, onnxruntime, Seal or the optional engines the tests that need them skip.

The native tests outside Python:

```sh
c++ -std=c++20 -O2 tests/tt_injection.cpp -o build/tt_injection && build/tt_injection
c++ -std=c++20 -O2 tests/gumbel.cpp -o build/gumbel_test && build/gumbel_test
cargo test --release --manifest-path tools/tactical/Cargo.toml
```

## CI

| Workflow | Job | Pull requests | Manual dispatch |
| --- | --- | --- | --- |
| Native rules (`native.yml`) | `rules` | `tt_injection.cpp`, `gumbel.cpp`, the Rust solver tests, and `NativeRules` plus `test_leaf_tactics` against the profile-guided build | same |
| Contracts (`contracts.yml`) | `contracts` | fast tier of every Python module except `test_web_*` | every test of those modules |
| Browser engine (`web.yml`) | `parity` | fast tier of `test_web_*`, with Seal built | every test of those modules |

No Python test runs in two jobs. The `rules` job runs two modules again, against a different build of the library.
