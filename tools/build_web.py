"""Build the browser engine bundle in web/engine.

wasm   src/gumbel.cpp -> gumbel.mjs + gumbel.wasm and src/hexo.cpp -> native/native.mjs + native/native.wasm (em++ on
       PATH, or --emxx), tools/tactical -> tactical.wasm and tools/shrimp_web -> shrimp/shrimp.wasm (cargo with the
       wasm32-wasip1 target), tools/six -> six/six.mjs + six/six.wasm (Six's network search, built as Six builds it
       for its site); build.json binds them to their sources (committed).
ort    onnxruntime-web from the npm registry, checked against its published integrity, into ort/ (ignored).
model  --checkpoint ema.pt, or --release TAG (or 'latest') from the GitHub releases, exported by export_web into
       model/ (ignored).
shrimp Shrimp's main_7 weights, --shrimp-weights or downloaded from hexo-bot at their pinned SHA-256, exported by
       tools/shrimp_web/export.py into shrimp/model/ (ignored).
six    Six's networks (CixMango/Six, MIT): the pinned release's from its archive (or --six-archive, that file saved
       locally), older generations from its 'networks' release, each checked against its pinned SHA-256 and rewritten
       for WebGPU, into six/networks/ (ignored).
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
import zipfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'web'/'engine'
TACTICAL = ROOT/'tools'/'tactical'
SHRIMP = ROOT/'tools'/'shrimp_web'
SIX = ROOT/'tools'/'six'
ORT_VERSION = '1.30.0'
ORT_INTEGRITY = 'sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ=='
ORT_FILES = ('ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm',
             'ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm')
GUMBEL_EXPORTS = ('malloc', 'free', 'hxg_new', 'hxg_free', 'hxg_error', 'hxg_begin', 'hxg_next', 'hxg_history',
                  'hxg_legal', 'hxg_fulfill', 'hxg_cancel', 'hxg_advance', 'hxg_stats', 'hxg_policy', 'hxg_completed',
                  'hxg_done', 'hxg_tactics', 'hxg_graph', 'hxg_exact', 'hxg_distance', 'hxg_census',
                  'hx_new', 'hx_free', 'hx_play', 'hx_winner', 'hx_player', 'hx_remaining', 'hx_moves')
NATIVE_EXPORTS = ('malloc', 'free', 'hx_new', 'hx_free', 'hx_play', 'hx_winner', 'hx_player', 'hx_remaining', 'hx_search')
WASM_FLAGS = ['-std=c++20', '-O3', '-fwasm-exceptions', '-msimd128', '-sMODULARIZE', '-sEXPORT_ES6',
                '-sENVIRONMENT=web,worker,node', '-sALLOW_MEMORY_GROWTH', '-sMAXIMUM_MEMORY=4GB', '-sFILESYSTEM=0',
                '-sEXPORTED_RUNTIME_METHODS=HEAP32,HEAPF64,HEAPU8,UTF8ToString']
ARTEFACTS = ('gumbel.mjs', 'gumbel.wasm', 'tactical.wasm', 'native/native.mjs', 'native/native.wasm', 'shrimp/shrimp.wasm',
             'six/six.mjs', 'six/six.wasm')
SIX_RELEASE = 'v1.3.3'
SIX_ARCHIVE = ('Six-1.3.3-macos-arm64.zip', '167b4ce038844377bb704c24b9470abe4c4016389e0d15290b975574114d4f69')
SIX_NETWORKS = {'gen-0455': 'a934a8b171cd9a715fcd54ffc3e192c24901f0a7d40caa4216ea9fbc0b074687',
                'gen-0400': '812110537328f638b916640599360f1055eab98db564d35ce4655cd3acb3615d',
                'gen-0300': 'd9cc22c415059742f1b499eda2f53e9691e783de24f410da55d8bafe95f4788e',
                'gen-0200': '17328c43677eb40c3ad72c7d0984c9cfa3c962f905b20d2c2967f48ba4b25256',
                'gen-0100': 'a666b400170354d98e507cf3471d655531cf6b20a87c50e475e8b07e5982903a'}
SIX_CROP_CELLS = 25*25
SIX_SOURCES = ('board.cpp', 'tactics.cpp', 'search.cpp', 'threats.cpp', 'planes.cpp', 'mcts.cpp')
SIX_EXPORTS = ('six_turn', 'six_stop', 'six_score', 'six_nodes', 'six_new_game', 'six_set_option', 'six_crop_cells', 'six_plane_count',
               'six_crop', 'malloc', 'free')
SIX_FLAGS = ['-std=c++20', '-O3', '-msimd128', '-fexceptions', '-sASYNCIFY', '-sASYNCIFY_STACK_SIZE=65536',
             '-sSTACK_SIZE=1048576', '-sMODULARIZE', '-sEXPORT_ES6', '-sENVIRONMENT=web,worker,node', '-sALLOW_MEMORY_GROWTH',
             '-sINITIAL_MEMORY=67108864', '-sEXPORTED_RUNTIME_METHODS=ccall,UTF8ToString,HEAPF32']


def digest(path):
    """SHA-256 of a file with CRLF read as LF, so checkouts with either line ending agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def sources():
    """{relative path: sha256} of every source the wasm artefacts are built from."""
    paths = [ROOT/'src'/name for name in ('gumbel.cpp', 'hexo.cpp', 'hexo.hpp', 'nnue.hpp')]
    paths += sorted(p for p in TACTICAL.rglob('*') if p.suffix in ('.rs', '.toml', '.lock') and 'target' not in p.parts)
    paths += sorted(p for p in SHRIMP.rglob('*') if (p.suffix in ('.rs', '.lock') or p.name == 'Cargo.toml')
                    and 'target' not in p.parts)
    paths += sorted(p for p in SIX.rglob('*') if p.suffix in ('.cpp', '.hpp'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def build_wasm(emxx, cargo):
    before = sources()
    for source, exports, out in (('gumbel.cpp', GUMBEL_EXPORTS, 'gumbel.mjs'), ('hexo.cpp', NATIVE_EXPORTS, 'native/native.mjs')):
        subprocess.run([emxx, str(ROOT/'src'/source), '-I', str(ROOT/'src'), *WASM_FLAGS,
                        f"-sEXPORTED_FUNCTIONS={','.join('_'+name for name in exports)}", '-o', str(ENGINE/out)], check=True)
    (ENGINE/'six').mkdir(exist_ok=True)
    six = [emxx, '-I', str(SIX/'src'), *(str(SIX/'src'/name) for name in SIX_SOURCES), str(SIX/'web_bot.cpp'), *SIX_FLAGS,
           '-sEXPORTED_FUNCTIONS='+','.join('_'+name for name in SIX_EXPORTS), '-o', str(ENGINE/'six'/'six.mjs')]
    subprocess.run(six, check=True)
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


def pinned(data, sha256, name):
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f'{name} does not match its pinned SHA-256')
    return data


def webgpu_graph(model):
    """Six's network rewritten so ONNX Runtime Web keeps all of it on WebGPU, as Six's engine/web/prepare_net.py does
    for its site: comparisons are cast to numbers at once and And becomes Mul, the policy reshape gets a constant
    shape, and the unused opponent policy is dropped."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, utils
    graph = model.graph
    nodes = list(graph.node)
    producer = {out: n for n in nodes for out in n.output}
    casts = [n for n in nodes if n.op_type == 'Cast' and n.input[0] in producer
             and producer[n.input[0]].op_type in ('Greater', 'Less', 'And')]
    targets = {helper.get_attribute_value(next(a for a in n.attribute if a.name == 'to')) for n in casts}
    if len(targets) != 1:
        raise ValueError(f'Six network casts its masks to {targets}')
    to = targets.pop()
    out = []
    for n in nodes:
        if n.op_type in ('Greater', 'Less'):
            out.append(helper.make_node(n.op_type, list(n.input), [n.output[0]+'_bool'], name=n.name))
            out.append(helper.make_node('Cast', [n.output[0]+'_bool'], [n.output[0]], name=n.name+'_num', to=to))
        elif n.op_type == 'And':
            out.append(helper.make_node('Mul', list(n.input), list(n.output), name=n.name))
        elif n in casts:
            out.append(helper.make_node('Identity', list(n.input), list(n.output), name=n.name))
        elif n.op_type == 'Reshape' and n.input[1] in producer and producer[n.input[1]].op_type == 'Concat':
            heads = next(t for t in graph.initializer if t.name == producer[n.input[0]].input[1]).dims[0]
            graph.initializer.append(numpy_helper.from_array(np.array([-1, heads, SIX_CROP_CELLS], np.int64), n.name+'_shape'))
            out.append(helper.make_node('Reshape', [n.input[0], n.name+'_shape'], list(n.output), name=n.name))
        else:
            out.append(n)
    del graph.node[:]
    graph.node.extend(out)
    kept = [o.name for o in graph.output if o.name != 'opponent']
    rewritten = utils.Extractor(model).extract_model([i.name for i in graph.input], kept)
    onnx.checker.check_model(rewritten)
    return rewritten


def same_outputs(original, rewritten):
    """Raises unless both serialized graphs give the same policy, value and score for a batch of random planes."""
    import numpy as np
    import onnxruntime
    planes = (np.random.default_rng(0).random((8, 8, 25, 25)) < .1).astype(np.float32)
    runs = [onnxruntime.InferenceSession(graph, providers=['CPUExecutionProvider']).run(['policy', 'value', 'score'], {'planes': planes})
            for graph in (original, rewritten)]
    for name, a, b in zip(('policy', 'value', 'score'), *runs):
        if not np.allclose(a, b, rtol=0, atol=1e-5):
            raise ValueError(f'The rewritten Six network changes its {name}')


def build_six(archive=None):
    """Six's networks as six/networks/<name>.onnx and manifest.json, newest first, each from its pinned source."""
    import onnx
    base = 'https://github.com/CixMango/Six/releases/download/'
    name, sha256 = SIX_ARCHIVE
    data = pinned(Path(archive).read_bytes() if archive else fetch(f'{base}{SIX_RELEASE}/{name}'), sha256, name)
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        released = {n: bundle.read(f'Six/runs/rl/{n}/net.onnx') for n in SIX_NETWORKS
                    if f'Six/runs/rl/{n}/net.onnx' in bundle.namelist()}
    out = ENGINE/'six'/'networks'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    networks = []
    for network, sha256 in sorted(SIX_NETWORKS.items(), reverse=True):
        source = pinned(released[network] if network in released else fetch(f'{base}networks/{network}.onnx'), sha256, network)
        graph = webgpu_graph(onnx.load_from_string(source)).SerializeToString()
        same_outputs(source, graph)
        (out/f'{network}.onnx').write_bytes(graph)
        networks.append(dict(name=network, file=f'{network}.onnx', sha256=hashlib.sha256(graph).hexdigest(),
                             source_sha256=sha256, bytes=len(graph)))
    manifest = dict(schema='six-web-v1', release=SIX_RELEASE, networks=networks)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1)+'\n', encoding='utf-8')


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
    parser.add_argument('parts', nargs='+', choices=('wasm', 'ort', 'model', 'shrimp', 'six'))
    parser.add_argument('--emxx', default=shutil.which('em++') or 'em++')
    parser.add_argument('--cargo', default='cargo')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--release', default='latest')
    parser.add_argument('--shrimp-weights', type=Path)
    parser.add_argument('--six-archive', type=Path)
    args = parser.parse_args()
    if 'wasm' in args.parts:
        build_wasm(args.emxx, args.cargo)
    if 'ort' in args.parts:
        build_ort()
    if 'model' in args.parts:
        build_model(args.checkpoint, args.release)
    if 'shrimp' in args.parts:
        build_shrimp(args.shrimp_weights)
    if 'six' in args.parts:
        build_six(args.six_archive)


if __name__ == '__main__':
    main()
