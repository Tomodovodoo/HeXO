"""Build the optional native tactical library (vendored hexo-strix solver) and record compiled identities."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent / 'tactical'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cargo', default='cargo')
    args = parser.parse_args()
    vendored = sorted(path for path in (PACKAGE/'vendor').rglob('*') if path.suffix in ('.rs', '.toml'))
    paths = ['Cargo.toml', 'Cargo.lock', 'stamps.json'] + [path.relative_to(PACKAGE).as_posix()
                                         for path in sorted((PACKAGE/'src').glob('*.rs')) + vendored]
    before = {p: digest(PACKAGE/p) for p in paths}
    command = [args.cargo, 'build', '--release', '--locked', '--manifest-path', str(PACKAGE/'Cargo.toml')]
    subprocess.run(command, check=True)
    if before != {p: digest(PACKAGE/p) for p in paths}:
        raise ValueError('Tactical sources changed during build')
    name = 'hexo_tactical.dll' if sys.platform == 'win32' else ('libhexo_tactical.dylib' if sys.platform == 'darwin' else 'libhexo_tactical.so')
    binary = PACKAGE/'target/release'/name
    metadata = dict(binary_sha256=digest(binary), sources=before, command=command,
                    cargo=subprocess.check_output([args.cargo, '--version'], text=True).strip(),
                    upstream_revision='5a771e572553a8bd8e010112b2ce65f16e5afa1b')
    binary.with_suffix(binary.suffix+'.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(binary)


if __name__ == '__main__':
    main()
