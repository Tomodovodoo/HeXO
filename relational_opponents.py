"""Immutable current Seal and public learned Strix inputs for relational matches.

freeze_opponent writes a new opponent directory. Its returned manifest belongs in
the enclosing evaluation provenance. load_opponent verifies that manifest and
runs private copies; it never builds, downloads, or substitutes another engine.
Windows system DLLs remain host dependencies, with their exact bytes verified.
"""
import builtins
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _dependencies(binary, runtime_dirs, objdump):
    """Resolve the transitive PE imports; keep OS DLLs explicitly host-bound."""
    if os.name != 'nt':
        raise RuntimeError('Opponent freezing currently supports Windows PE builds')
    system = Path(os.environ['SystemRoot'])/'System32'
    compiler = shutil.which('g++')
    directories = [binary.parent, *map(Path, runtime_dirs)]
    if compiler:
        directories.append(Path(compiler).parent)
    pending, visited, copies, host, imports = [binary], set(), {}, {}, {}
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        output = subprocess.check_output([objdump, '-p', str(current)], text=True)
        names = re.findall(r'DLL Name:\s*(\S+)', output)
        if not names and current == binary:
            raise ValueError(f'No PE dependency information for {current}')
        imports[current.name] = names
        for name in names:
            # Windows resolves API-set contracts itself, not from app files.
            if name.lower().startswith(('api-ms-win-', 'ext-ms-win-')):
                continue
            os_file = system/name
            if os_file.is_file():
                host[str(os_file.resolve())] = _sha(os_file.read_bytes())
                pending.append(os_file.resolve())
                continue
            source = next((p/name for p in directories if (p/name).is_file()), None)
            if source is None:
                raise FileNotFoundError(f'Unresolved native opponent dependency: {name}')
            data = source.read_bytes()
            if name.lower() in copies and copies[name.lower()][1] != data:
                raise ValueError(f'Conflicting native dependency: {name}')
            copies[name.lower()] = (source.name, data)
            pending.append(source.resolve())
    return copies, host, imports


def freeze_opponent(kind, destination, config):
    """config: binary, source_root; Strix also model, simulations/actions/timeout_ms/seed.

    source_root defaults to this checkout. Build manifests must match its exact
    adapter sources. runtime_dirs and objdump optionally locate PE dependencies.
    """
    if kind not in ('seal-current', 'strix'):
        raise ValueError('Opponent must be seal-current or strix')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Opponent destination must be a new directory')
    source = Path(config.get('source_root', ROOT)).resolve()
    binary = Path(config['binary']).resolve()
    image = binary.read_bytes()
    files = {}
    settings = {}
    if kind == 'seal-current':
        from tools.seal_current import REVISION, WEIGHTS_SHA256
        manifest_path = binary.with_suffix(binary.suffix+'.json')
        build_bytes = manifest_path.read_bytes()
        build = json.loads(build_bytes)
        adapter = (source/'tools/seal_current_adapter.cpp').read_bytes()
        if (build['revision'] != REVISION or build['weights_sha256'] != WEIGHTS_SHA256
                or build['binary_sha256'] != _sha(image)
                or build['adapter_source_sha256'] != _sha(adapter)):
            raise ValueError('Current Seal build/source identity mismatch')
        entry = 'build/hexo_seal_current.dll'
        files.update({'tools/seal_current.py': (source/'tools/seal_current.py').read_bytes(),
                      'tools/seal_current_adapter.cpp': adapter, entry+'.json': build_bytes})
        budget = dict(unit='milliseconds per complete turn', deadline='upstream best-effort',
                      equal_compute=False, randomness='upstream random_device; arena seed does not seed Seal')
    else:
        from strix_reference import REVISION
        from tools.strix_learned_adapter import MODEL_SHA256
        build_bytes = binary.with_name('build-provenance.json').read_bytes()
        build = json.loads(build_bytes)
        model = Path(config['model']).read_bytes()
        if (build['revision'] != REVISION or build['executable_sha256'] != _sha(image)
                or _sha(model) != MODEL_SHA256):
            raise ValueError('Learned Strix build/model identity mismatch')
        for name, digest in build['wrapper_sha256'].items():
            data = (source/'tools/strix_learned'/name).read_bytes()
            if _sha(data) != digest:
                raise ValueError(f'Learned Strix wrapper source changed: {name}')
            files['tools/strix_learned/'+name] = data
        setup = (source/'tools/build_strix_learned.py').read_bytes()
        if _sha(setup) != build['setup_sha256']:
            raise ValueError('Learned Strix build script changed')
        files['tools/build_strix_learned.py'] = setup
        for name in ('strix_reference.py', 'tools/strix_learned_adapter.py'):
            files[name] = (source/name).read_bytes()
        entry = 'build/hexo-strix-learned.exe'
        files['build/build-provenance.json'] = build_bytes
        files['models/model.safetensors'] = model
        settings = {k: config.get(k, v) for k,v in
                    dict(simulations=8, actions=4, timeout_ms=5000, seed=0).items()}
        if (any(type(settings[k]) is not int for k in settings)
                or not 1 <= settings['simulations'] <= 100000
                or not 1 <= settings['actions'] <= 1024
                or not 1 <= settings['timeout_ms'] <= 600000
                or not 0 <= settings['seed'] < 1 << 64):
            raise ValueError('Invalid learned Strix budgets')
        budget = dict(unit='simulations per placement', **settings,
                      deadline='wall timeout per complete turn; process killed on timeout',
                      equal_compute=False, caller_milliseconds_ignored=True,
                      root_forcing=dict(generator='wide', depth=6, nodes=2000),
                      leaf_forcing=False, independent_proof=False)
    copies, host, imports = _dependencies(binary, config.get('runtime_dirs', []), config.get('objdump', 'objdump'))
    # StrixReference starts a private executable copy. The pinned public build
    # is self-contained apart from OS DLLs; reject builds needing app-local DLLs.
    if kind == 'strix' and copies:
        raise ValueError('Learned Strix executable must have only Windows system DLL imports')
    for name, data in copies.values():
        files['build/'+name] = data
    if binary.read_bytes() != image:
        raise ValueError('Opponent binary changed while resolving dependencies')
    files[entry] = image
    metadata = dict(schema='hexo-relational-opponent-v1', kind=kind, revision=REVISION,
        binary=entry, binary_sha256=_sha(image), settings=settings, budget=budget,
        files_sha256={name:_sha(data) for name,data in files.items()},
        host_system_files_sha256=host, imports=imports,
        runtime_scope='Exact private copies of app inputs and non-system DLLs; verified host Windows DLLs and API-set contracts',
        build=build)
    if kind == 'strix':
        metadata.update(model='models/model.safetensors', model_sha256=MODEL_SHA256)
    destination.mkdir(parents=True)
    for name, data in files.items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (destination/'opponent.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return metadata


def _module(path, reference=None):
    spec = importlib.util.spec_from_file_location('_frozen_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    if reference is not None:
        def import_reference(name, *args, **kwargs):
            return reference if name == 'strix_reference' else builtins.__import__(name, *args, **kwargs)
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=import_reference)
    spec.loader.exec_module(module)
    return module


class FrozenOpponent:
    def __init__(self, metadata, frozen_dir):
        self.metadata = json.loads(json.dumps(metadata))
        if metadata['files_sha256'].get(metadata['binary']) != metadata['binary_sha256']:
            raise ValueError('Opponent binary is not bound to its file manifest')
        self.directory = tempfile.TemporaryDirectory(prefix='hexo-opponent-')
        self.backend = None
        self.dependencies = []
        self.dll_directory = None
        self.private_runtime_paths = set()
        self.closed = False
        try:
            private = Path(self.directory.name)
            for name, digest in metadata['files_sha256'].items():
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Opponent manifest paths must stay inside the snapshot')
                data = (Path(frozen_dir)/relative).read_bytes()
                if _sha(data) != digest:
                    raise ValueError(f'Frozen opponent input changed: {name}')
                target = private/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if target.suffix.lower() in ('.dll', '.exe'):
                    self.private_runtime_paths.add(target.resolve())
            for name, digest in metadata['host_system_files_sha256'].items():
                if _sha(Path(name).read_bytes()) != digest:
                    raise ValueError(f'Opponent system dependency changed: {name}')
            self.dll_directory = os.add_dll_directory(str(private/'build'))
            if metadata['kind'] == 'seal-current':
                entry = private/metadata['binary']
                for path in (private/'build').glob('*.dll'):
                    if path != entry:
                        self.dependencies.append(C.WinDLL(str(path)))
                self.backend = _module(private/'tools/seal_current.py').SealCurrent()
                # Windows can reuse an already-loaded DLL with the same name.
                # Require its bytes to match our frozen dependency as well.
                kernel = C.WinDLL('kernel32', use_last_error=True)
                kernel.GetModuleHandleW.argtypes = [C.c_wchar_p]
                kernel.GetModuleHandleW.restype = C.c_void_p
                kernel.GetModuleFileNameW.argtypes = [C.c_void_p, C.c_wchar_p, C.c_ulong]
                for path in (private/'build').glob('*.dll'):
                    handle = kernel.GetModuleHandleW(path.name)
                    buffer = C.create_unicode_buffer(32768)
                    if not handle or not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
                        raise C.WinError(C.get_last_error())
                    if _sha(Path(buffer.value).read_bytes()) != _sha(path.read_bytes()):
                        raise ValueError(f'Loaded opponent DLL differs from frozen bytes: {path.name}')
            elif metadata['kind'] == 'strix':
                reference = _module(private/'strix_reference.py')
                adapter = _module(private/'tools/strix_learned_adapter.py', reference)
                self.backend = adapter.StrixLearned(private/'models/model.safetensors',
                    executable=private/metadata['binary'], **metadata['settings'])
                self.backend.warm_up()
            else:
                raise ValueError('Unknown frozen opponent kind')
            self.metadata['loaded'] = self.backend.metadata
        except Exception:
            self.close()
            raise

    def reset(self):
        if self.closed:
            raise RuntimeError('Opponent is closed')
        if self.metadata['kind'] == 'seal-current':
            self.backend.reset()
        else:
            self.backend.calls = 0
            self.backend.last_result = None

    def set_history(self, history=()):
        # Both engines receive the complete colored board on every query.
        self.reset()

    def turn(self, game, milliseconds):
        if self.closed:
            raise RuntimeError('Opponent is closed')
        start = time.perf_counter()
        moves = self.backend(game, milliseconds)
        result = dict(moves=[list(p) for p in moves], backend=self.metadata['kind'],
                      elapsed_ms=(time.perf_counter()-start)*1000, budget=self.metadata['budget'])
        if self.metadata['kind'] == 'strix':
            result['search'] = self.backend.last_result
        else:
            result['requested_ms'] = milliseconds
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.backend is not None:
            if self.metadata['kind'] == 'strix':
                self.backend.close()
            else:
                import _ctypes
                _ctypes.FreeLibrary(self.backend.lib._handle)
            self.backend = None
        if self.dependencies:
            import _ctypes
            for dependency in reversed(self.dependencies):
                _ctypes.FreeLibrary(dependency._handle)
            self.dependencies.clear()
        if self.dll_directory is not None:
            self.dll_directory.close()
        self.directory.cleanup()


def load_opponent(metadata, frozen_dir):
    if metadata.get('schema') != 'hexo-relational-opponent-v1':
        raise ValueError('Unsupported frozen opponent schema')
    return FrozenOpponent(metadata, frozen_dir)
