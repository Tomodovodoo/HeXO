"""Build the browser engine bundle in web/engine.

wasm   src/gumbel.cpp -> gumbel.mjs + gumbel.wasm and src/hexo.cpp -> native/native.mjs + native/native.wasm (em++ on
       PATH, or --emxx), tools/tactical -> tactical.wasm and tools/shrimp_web -> shrimp/shrimp.wasm (cargo with the
       wasm32-wasip1 target); build.json binds them to their sources (committed).
ort    onnxruntime-web from the npm registry, checked against its published integrity, into ort/ (ignored).
model  --checkpoint ema.pt, or --release TAG (or 'latest') from the GitHub releases, exported by export_web into
       model/ (ignored).
shrimp Shrimp's main_7 weights, --shrimp-weights or downloaded from hexo-bot at their pinned SHA-256, exported by
       tools/shrimp_web/export.py into shrimp/model/ (ignored).
seal   Seal's headers at the revision pinned in tools/engines.json, each checked against its SHA-256, with
       tools/seal_adapter.cpp -> seal/engine.mjs + engine.wasm and seal/manifest.json (ignored).
"""
import argparse
import base64
import hashlib
import importlib.util
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
SHRIMP = ROOT/'tools'/'shrimp_web'
SEAL = ENGINE/'seal'
ORT_VERSION = '1.30.0'
ORT_INTEGRITY = 'sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ=='
ORT_FILES = ('ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm',
             'ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm')
GUMBEL_EXPORTS = ('malloc', 'free', 'hxg_new', 'hxg_free', 'hxg_error', 'hxg_begin', 'hxg_next', 'hxg_history',
                  'hxg_legal', 'hxg_fulfill', 'hxg_cancel', 'hxg_advance', 'hxg_stats', 'hxg_policy', 'hxg_completed',
                  'hxg_done', 'hxg_tactics', 'hxg_graph', 'hxg_q_range_floor', 'hxg_exact', 'hxg_distance', 'hxg_census',
                  'hx_new', 'hx_free', 'hx_play', 'hx_winner', 'hx_player', 'hx_remaining', 'hx_moves')
NATIVE_EXPORTS = ('malloc', 'free', 'hx_new', 'hx_free', 'hx_play', 'hx_winner', 'hx_player', 'hx_remaining', 'hx_search')
WASM_FLAGS = ['-std=c++20', '-O3', '-fwasm-exceptions', '-msimd128', '-sMODULARIZE', '-sEXPORT_ES6',
                '-sENVIRONMENT=web,worker,node', '-sALLOW_MEMORY_GROWTH', '-sMAXIMUM_MEMORY=4GB', '-sFILESYSTEM=0',
                '-sEXPORTED_RUNTIME_METHODS=HEAP32,HEAPF64,HEAPU8,UTF8ToString']
ARTEFACTS = ('gumbel.mjs', 'gumbel.wasm', 'tactical.wasm', 'native/native.mjs', 'native/native.wasm', 'shrimp/shrimp.wasm')


def digest(path):
    """SHA-256 of a file with CRLF read as LF, so checkouts with either line ending agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def sources():
    """{relative path: sha256} of every source the wasm artefacts are built from."""
    paths = [ROOT/'src'/name for name in ('gumbel.cpp', 'hexo.cpp', 'hexo.hpp', 'nnue.hpp')]
    paths += sorted(p for p in TACTICAL.rglob('*') if p.suffix in ('.rs', '.toml', '.lock') and 'target' not in p.parts)
    paths += sorted(p for p in SHRIMP.rglob('*') if (p.suffix in ('.rs', '.lock') or p.name == 'Cargo.toml')
                    and 'target' not in p.parts)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def build_wasm(emxx, cargo):
    before = sources()
    for source, exports, out in (('gumbel.cpp', GUMBEL_EXPORTS, 'gumbel.mjs'), ('hexo.cpp', NATIVE_EXPORTS, 'native/native.mjs')):
        subprocess.run([emxx, str(ROOT/'src'/source), '-I', str(ROOT/'src'), *WASM_FLAGS,
                        f"-sEXPORTED_FUNCTIONS={','.join('_'+name for name in exports)}", '-o', str(ENGINE/out)], check=True)
    with tempfile.TemporaryDirectory() as target:
        tactical = [cargo, 'build', '--release', '--locked', '--lib', '--target', 'wasm32-wasip1',
                    '--manifest-path', str(TACTICAL/'Cargo.toml'), '--target-dir', target]
        subprocess.run(tactical, check=True)
        shutil.copyfile(Path(target)/'wasm32-wasip1'/'release'/'hexo_tactical.wasm', ENGINE/'tactical.wasm')
        shrimp = [cargo, 'build', '--release', '--locked', '--lib', '--target', 'wasm32-wasip1',
                  '--manifest-path', str(SHRIMP/'Cargo.toml'), '--target-dir', target]
        subprocess.run(shrimp, check=True)
        shutil.copyfile(Path(target)/'wasm32-wasip1'/'release'/'shrimp_web.wasm', ENGINE/'shrimp'/'shrimp.wasm')
    if sources() != before:
        raise ValueError('Sources changed during the build')
    tools = dict(emxx=subprocess.check_output([emxx, '--version'], text=True).splitlines()[0],
                 cargo=subprocess.check_output([cargo, '--version'], text=True).strip())
    record = dict(sources=before, artefacts={name: digest(ENGINE/name) for name in ARTEFACTS}, tools=tools)
    (ENGINE/'build.json').write_text(json.dumps(record, indent=1)+'\n', encoding='utf-8')


def build_seal(emxx):
    """Compile tools/seal_adapter.cpp against Seal's pinned headers into seal/; seal/manifest.json names Seal's
    revision and the wasm's SHA-256, which the browser caches it under."""
    spec = json.loads((ROOT/'tools'/'engines.json').read_text(encoding='utf-8'))['seal']
    with tempfile.TemporaryDirectory() as folder:
        for file in spec['files']:
            data = fetch(file['url'])
            if hashlib.sha256(data).hexdigest() != file['sha256']:
                raise ValueError(f"Seal's {file['path']} does not match its pinned SHA-256")
            target = Path(folder)/file['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        SEAL.mkdir(exist_ok=True)
        subprocess.run([emxx, str(ROOT/'tools'/'seal_adapter.cpp'), '-I', str(Path(folder)/'cpp'), *WASM_FLAGS,
                        '-sEXPORTED_FUNCTIONS=_malloc,_free,_seal_move', '-o', str(SEAL/'engine.mjs')], check=True)
    manifest = dict(revision=spec['revision'], sha256=hashlib.sha256((SEAL/'engine.wasm').read_bytes()).hexdigest())
    (SEAL/'manifest.json').write_text(json.dumps(manifest)+'\n', encoding='utf-8')


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


def build_shrimp(weights):
    """Export Shrimp's network into shrimp/model/ from `weights`, or from the pinned download when None."""
    spec = importlib.util.spec_from_file_location('shrimp_export', SHRIMP/'export.py')
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    with tempfile.TemporaryDirectory() as folder:
        if weights is None:
            weights = Path(folder)/'shrimp_main7_infer.pt'
            weights.write_bytes(fetch(export.WEIGHTS['url']))
        export.export(weights, ENGINE/'shrimp'/'model')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('parts', nargs='+', choices=('wasm', 'ort', 'model', 'shrimp', 'seal'))
    parser.add_argument('--emxx', default=shutil.which('em++') or 'em++')
    parser.add_argument('--cargo', default='cargo')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--release', default='latest')
    parser.add_argument('--shrimp-weights', type=Path)
    args = parser.parse_args()
    if 'wasm' in args.parts:
        build_wasm(args.emxx, args.cargo)
    if 'ort' in args.parts:
        build_ort()
    if 'model' in args.parts:
        build_model(args.checkpoint, args.release)
    if 'shrimp' in args.parts:
        build_shrimp(args.shrimp_weights)
    if 'seal' in args.parts:
        build_seal(args.emxx)


if __name__ == '__main__':
    main()
