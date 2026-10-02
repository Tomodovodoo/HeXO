"""Build the engines release files for this system, or pin a published release's hashes into tools/engines.json.

```sh
python tools/build_engines.py build --out dist                 # Strix, and Shrimp's wheels for this Python
python tools/build_engines.py build --out dist --all-pythons   # Shrimp's wheels for every Python maturin finds
python tools/build_engines.py pin engines-v1                   # the release's SHA256SUMS into tools/engines.json
```

Building needs Rust (cargo) and git. Seal has no licence, so it is never built for the release.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import engine_setup  # noqa: E402


def manifest():
    return json.loads(engine_setup.MANIFEST.read_text(encoding='utf-8'))


def get(url):
    with urlopen(Request(url, headers={'User-Agent': 'bubble'}), timeout=60) as response:
        return response.read()


def build(out, all_pythons):
    """Build Strix's executable and Shrimp's wheels into `out`, named as the setup looks for them; prints each
    file's SHA-256."""
    spec, toolchain = manifest(), engine_setup.cargo()
    if toolchain is None:
        raise SystemExit('cargo not found')
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hexo-engines-') as work:
        work = Path(work)
        executable = engine_setup.build_strix(work, toolchain)
        shutil.copy2(executable, out / f"{spec['strix']['executable']}-{engine_setup.system()}{executable.suffix}")
        archive = work / 'source.zip'
        archive.write_bytes(get(engine_setup.source_url(spec['shrimp'])))
        source = engine_setup.source_tree(archive, work / 'src', spec['shrimp']['sources'])
        engine_setup.build_wheels(source, out, spec['shrimp']['crates'], None if all_pythons else [sys.executable])
    for path in sorted(out.iterdir()):
        print(hashlib.sha256(path.read_bytes()).hexdigest(), path.name)


def pin(tag):
    """Write the file hashes of release `tag`, from its SHA256SUMS, into tools/engines.json."""
    spec = manifest()
    release = spec['release']
    sums = get(f"https://github.com/{release['repository']}/releases/download/{tag}/SHA256SUMS").decode()
    files = {name.lstrip('*'): digest for digest, name in (line.split(maxsplit=1) for line in sums.splitlines() if line)}
    spec['release'] = release | dict(tag=tag, files=dict(sorted(files.items())))
    engine_setup.MANIFEST.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    print(f'{len(files)} files of {tag} pinned in {engine_setup.MANIFEST}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    building = commands.add_parser('build', help="build this system's release files")
    building.add_argument('--out', type=Path, required=True)
    building.add_argument('--all-pythons', action='store_true', help="Shrimp's wheels for every Python found")
    pinning = commands.add_parser('pin', help="pin a published release's hashes")
    pinning.add_argument('tag')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.out, args.all_pythons)
    else:
        pin(args.tag)


if __name__ == '__main__':
    main()
