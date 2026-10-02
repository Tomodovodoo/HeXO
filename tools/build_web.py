"""Build the browser engine bundle in web/engine.

wasm   src/gumbel.cpp -> gumbel.mjs + gumbel.wasm (em++ on PATH, or --emxx), tools/tactical -> tactical.wasm
       (cargo with the wasm32-wasip1 target); build.json binds them to their sources (committed).
ort    onnxruntime-web from the npm registry, checked against its published integrity, into ort/ (ignored).
model  --checkpoint ema.pt, or --release TAG (or 'latest') from the GitHub releases, exported by export_web into
       model/ (ignored).
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'web'/'engine'
TACTICAL = ROOT/'tools'/'tactical'
ORT_VERSION = '1.30.0'
ORT_INTEGRITY = 'sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ=='
ORT_FILES = ('ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm',
             'ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm')
GUMBEL_EXPORTS = ('malloc', 'free', 'hxg_new', 'hxg_free', 'hxg_error', 'hxg_begin', 'hxg_next', 'hxg_history',
                  'hxg_legal', 'hxg_fulfill', 'hxg_cancel', 'hxg_advance', 'hxg_stats', 'hxg_policy', 'hxg_completed',
                  'hxg_done', 'hxg_tactics', 'hxg_graph', 'hxg_exact', 'hxg_distance', 'hxg_census',
                  'hx_new', 'hx_free', 'hx_play', 'hx_winner', 'hx_player', 'hx_remaining', 'hx_moves')
GUMBEL_FLAGS = ['-std=c++20', '-O3', '-fwasm-exceptions', '-msimd128', '-sMODULARIZE', '-sEXPORT_ES6',
                '-sENVIRONMENT=web,worker,node', '-sALLOW_MEMORY_GROWTH', '-sMAXIMUM_MEMORY=4GB', '-sFILESYSTEM=0',
                '-sEXPORTED_RUNTIME_METHODS=HEAP32,HEAPF64,HEAPU8,UTF8ToString']


def digest(path):
    """SHA-256 of a file with CRLF read as LF, so checkouts with either line ending agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def sources():
    """{relative path: sha256} of every source the wasm artefacts are built from."""
    paths = [ROOT/'src'/name for name in ('gumbel.cpp', 'hexo.cpp', 'hexo.hpp', 'nnue.hpp')]
    paths += sorted(p for p in TACTICAL.rglob('*') if p.suffix in ('.rs', '.toml', '.lock') and 'target' not in p.parts)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def build_wasm(emxx, cargo):
    before = sources()
    exports = ','.join('_'+name for name in GUMBEL_EXPORTS)
    gumbel = [emxx, str(ROOT/'src'/'gumbel.cpp'), '-I', str(ROOT/'src'), *GUMBEL_FLAGS,
              f'-sEXPORTED_FUNCTIONS={exports}', '-o', str(ENGINE/'gumbel.mjs')]
    subprocess.run(gumbel, check=True)
    with tempfile.TemporaryDirectory() as target:
        tactical = [cargo, 'build', '--release', '--locked', '--lib', '--target', 'wasm32-wasip1',
                    '--manifest-path', str(TACTICAL/'Cargo.toml'), '--target-dir', target]
        subprocess.run(tactical, check=True)
        shutil.copyfile(Path(target)/'wasm32-wasip1'/'release'/'hexo_tactical.wasm', ENGINE/'tactical.wasm')
    if sources() != before:
        raise ValueError('Sources changed during the build')
    tools = dict(emxx=subprocess.check_output([emxx, '--version'], text=True).splitlines()[0],
                 cargo=subprocess.check_output([cargo, '--version'], text=True).strip())
    record = dict(sources=before, artefacts={name: digest(ENGINE/name) for name in ('gumbel.mjs', 'gumbel.wasm', 'tactical.wasm')},
                  tools=tools)
    (ENGINE/'build.json').write_text(json.dumps(record, indent=1)+'\n', encoding='utf-8')


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'hexo-build-web'})) as response:
        return response.read()


def build_ort():
    data = fetch(f'https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz')
    if 'sha512-'+base64.b64encode(hashlib.sha512(data).digest()).decode() != ORT_INTEGRITY:
        raise ValueError('onnxruntime-web archive does not match its published integrity')
    out = ENGINE/'ort'
    out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for name in ORT_FILES:
            (out/name).write_bytes(archive.extractfile(f'package/dist/{name}').read())
    (out/'version.json').write_text(json.dumps(dict(version=ORT_VERSION))+'\n', encoding='utf-8')


def build_model(checkpoint, release):
    sys.path.insert(0, str(ROOT/'python'))
    import export_web
    with tempfile.TemporaryDirectory() as folder:
        if checkpoint is None:
            base = 'https://github.com/Tomodovodoo/HeXO/releases/'
            tag = release if release != 'latest' else urlopen(Request(base+'latest', headers={'User-Agent': 'hexo-build-web'})).geturl().rsplit('/', 1)[1]
            checkpoint = Path(folder)/'ema.pt'
            checkpoint.write_bytes(fetch(f'{base}download/{tag}/ema.pt'))
        export_web.export(checkpoint, ENGINE/'model')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('parts', nargs='+', choices=('wasm', 'ort', 'model'))
    parser.add_argument('--emxx', default=shutil.which('em++') or 'em++')
    parser.add_argument('--cargo', default='cargo')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--release', default='latest')
    args = parser.parse_args()
    if 'wasm' in args.parts:
        build_wasm(args.emxx, args.cargo)
    if 'ort' in args.parts:
        build_ort()
    if 'model' in args.parts:
        build_model(args.checkpoint, args.release)


if __name__ == '__main__':
    main()
