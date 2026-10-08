"""One-click engine setup with recorded downloads: hashes, layouts, registry pickup and the HTTP surface."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
import unittest.mock
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import engine_setup
from engine_setup import MANIFEST, SetupError, Setups, interpreter_tags, six_member, system, unpack, wheel_fits
from play import Evaluations, Handler, Session, scan
from tests.test_play import FakeEngines, entries

WINDOWS = os.name == 'nt'
SIX_BINARY = 'sixengine.exe' if WINDOWS else 'sixengine'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive(name, members):
    """A zip or tar.gz, by `name`'s suffix, holding `members` ({path: bytes})."""
    buffer = io.BytesIO()
    if name.endswith('.zip'):
        with zipfile.ZipFile(buffer, 'w') as files:
            for path, data in members.items():
                files.writestr(path, data)
    else:
        with tarfile.open(fileobj=buffer, mode='w:gz') as files:
            for path, data in members.items():
                info = tarfile.TarInfo(path)
                info.size, info.mode = len(data), 0o755
                files.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class Web:
    """Answers requests from `files` ({url: bytes}) the way urlopen does and records them; others fail."""

    def __init__(self, files):
        self.files, self.asked = files, []

    def __call__(self, request, timeout=None):
        self.asked.append(request.full_url)
        if request.full_url not in self.files:
            raise URLError(f'{request.full_url} not found')
        response = io.BytesIO(self.files[request.full_url])
        response.headers = {'Content-Length': str(len(self.files[request.full_url]))}
        return response


class Recipes(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.models = Path(self.folder.name) / 'models'
        self.models.mkdir()
        self.manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.rescans = 0

    def tearDown(self):
        self.folder.cleanup()

    def setups(self, files, which=lambda name: None, cargo=lambda: None):
        path = Path(self.folder.name) / 'engines.json'
        path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.web = Web(files)

        def rescan():
            self.rescans += 1
        return Setups(self.models, rescan, lambda: scan(self.models), path, self.web, which, cargo)

    def strix_networks(self, model):
        """Pins two Strix networks, `model` and a second one, and returns their downloads by URL."""
        files = {f'https://example.test/{n}': data for n, data in (('a', model), ('b', model + b' b'))}
        self.manifest['strix']['networks'] = [dict(id=url.rsplit('/', 1)[1], url=url, sha256=sha(data), size=len(data))
                                              for url, data in files.items()]
        return files

    def finish(self, setups, engine):
        setups.start(engine)
        return setups.wait(engine, 30)

    def six_release(self, digest=None):
        suffix = self.manifest['six']['assets'][system()]
        name = f'Six-9.9.9{suffix}'
        data = archive(name, {f'Six/engine/{SIX_BINARY}': b'engine', 'Six/engine/onnxruntime.dll': b'runtime',
                              'Six/runs/rl/gen-0455/net.onnx': b'network', 'Six/node/node.exe': b'node',
                              'Six/README.txt': b'readme'})
        assets = [dict(name=name, size=len(data), digest='sha256:' + (digest or sha(data)),
                       browser_download_url='https://example.test/six-archive'),
                  dict(name='Six-9.9.9-Setup-windows.exe', size=1, digest=None,
                       browser_download_url='https://example.test/installer')]
        api = json.dumps(dict(tag_name='v9.9.9', assets=assets)).encode()
        return {'https://api.github.com/repos/CixMango/Six/releases/latest': api, 'https://example.test/six-archive': data}

    @unittest.skipUnless(system() in ('windows-x64', 'linux-x64', 'macos-arm64'), 'Six has no build here')
    def test_six_flattens_the_release_into_one_folder_the_registry_offers(self):
        setups = self.setups(self.six_release())
        self.assertFalse(next(e for e in setups.catalogue() if e['engine'] == 'six')['installed'])
        job = self.finish(setups, 'six')
        self.assertEqual(job.json(), dict(state='done', progress=1.0, busy=False, error=None))
        folder = self.models / 'six'
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         sorted([SIX_BINARY, 'gen-0455.onnx', 'onnxruntime.dll', 'setup.json']))
        self.assertEqual((folder / 'gen-0455.onnx').read_bytes(), b'network')
        self.assertFalse((self.models / '.setup' / 'six').exists())
        self.assertEqual(self.rescans, 1)
        six = [e for e in scan(self.models).values() if e['kind'] == 'six']
        self.assertEqual([e['checkpoints'] for e in six], [['gen-0455']])
        self.assertTrue(next(e for e in setups.catalogue() if e['engine'] == 'six')['installed'])
        self.assertNotIn('https://example.test/installer', self.web.asked)

    @unittest.skipUnless(system() in ('windows-x64', 'linux-x64', 'macos-arm64'), 'Six has no build here')
    def test_a_download_that_fails_its_hash_installs_nothing(self):
        job = self.finish(self.setups(self.six_release(digest='0' * 64)), 'six')
        self.assertEqual(job.state, 'failed')
        self.assertIn('SHA-256', job.error)
        self.assertEqual(list(self.models.iterdir()), [self.models / '.setup'])
        self.assertEqual(list((self.models / '.setup').iterdir()), [])
        self.assertEqual(self.rescans, 0)
        release = self.six_release()
        api = 'https://api.github.com/repos/CixMango/Six/releases/latest'
        listed = json.loads(release[api])
        listed['assets'][0]['digest'] = None
        job = self.finish(self.setups(release | {api: json.dumps(listed).encode()}), 'six')
        self.assertIn('no SHA-256', job.error)
        self.assertNotIn('https://example.test/six-archive', self.web.asked)

    def test_strix_comes_from_the_pinned_release_without_a_toolchain(self):
        executable, model = b'strix executable', b'strix model'
        name = f"hexo-strix-learned-{system()}{'.exe' if WINDOWS else ''}"
        self.manifest['release']['files'] = {name: sha(executable)}
        networks = self.strix_networks(model)
        setups = self.setups({f'https://github.com/Tomodovodoo/HeXO/releases/download/engines-v1/{name}': executable,
                              **networks})
        job = self.finish(setups, 'strix')
        self.assertEqual(job.state, 'done', job.error)
        local = 'hexo-strix-learned.exe' if WINDOWS else 'hexo-strix-learned'
        self.assertEqual((self.models / 'strix' / local).read_bytes(), executable)
        entry = json.loads((self.models / 'strix.json').read_text(encoding='utf-8'))
        self.assertEqual(entry, dict(name='Strix', kind='strix', engine=f'strix/{local}', setup='strix',
                                     networks=dict(a='strix/a.safetensors', b='strix/b.safetensors')))
        strix = next(e for e in scan(self.models).values() if e['kind'] == 'strix')
        self.assertEqual(strix['checkpoints'], ['a', 'b'])
        self.assertEqual([strix['networks'][n].read_bytes() for n in 'ab'], [model, model + b' b'])
        self.assertEqual(strix['engine'].read_bytes(), executable)

    def test_an_unpinned_release_is_checked_against_its_own_sums(self):
        executable, model = b'strix executable', b'strix model'
        name = f"hexo-strix-learned-{system()}{'.exe' if WINDOWS else ''}"
        networks = self.strix_networks(model)
        release = 'https://github.com/Tomodovodoo/HeXO/releases/download/engines-v1/'
        sums = f'{sha(executable)}  {name}\n{"0" * 64}  other-file\n'.encode()
        job = self.finish(self.setups({release + 'SHA256SUMS': sums, release + name: executable,
                                       **networks}), 'strix')
        self.assertEqual(job.state, 'done', job.error)
        job = self.finish(self.setups({release + 'SHA256SUMS': sums, release + name: b'tampered',
                                       **networks}), 'strix')
        self.assertIn('SHA-256', job.error)

    def test_without_a_toolchain_or_a_published_build_the_setup_says_so(self):
        job = self.finish(self.setups({}), 'strix')
        self.assertEqual(job.state, 'failed')
        self.assertIn('No published build', job.error)
        self.assertFalse((self.models / 'strix.json').exists())
        with unittest.mock.patch('engine_setup.build_strix', side_effect=OSError('linker missing')):
            job = self.finish(self.setups({}, which=lambda name: name, cargo=lambda: 'cargo'), 'strix')
        self.assertEqual(job.error, 'linker missing')

    def test_a_failed_local_build_falls_back_to_the_published_one(self):
        executable, model = b'published', b'model'
        name = f"hexo-strix-learned-{system()}{'.exe' if WINDOWS else ''}"
        self.manifest['release']['files'] = {name: sha(executable)}
        networks = self.strix_networks(model)
        setups = self.setups({f'https://github.com/Tomodovodoo/HeXO/releases/download/engines-v1/{name}': executable,
                              **networks}, which=lambda name: name, cargo=lambda: 'cargo')
        with unittest.mock.patch('engine_setup.build_strix', side_effect=SetupError('linker missing')):
            job = self.finish(setups, 'strix')
        self.assertEqual(job.state, 'done', job.error)
        self.assertEqual(next(e for e in scan(self.models).values() if e['kind'] == 'strix')['engine'].read_bytes(),
                         executable)

    def test_shrimp_gets_the_driver_weights_and_wheels_and_runs_on_this_python(self):
        files, web = [], {}
        for number, file in enumerate(self.manifest['shrimp']['files']):
            data = f'file {number}'.encode()
            web[f'https://example.test/{number}'] = data
            files.append(file | dict(url=f'https://example.test/{number}', sha256=sha(data)))
        self.manifest['shrimp']['files'] = files
        release = {}
        for crate in self.manifest['shrimp']['crates']:
            name = f'{crate}-0.1.0-{interpreter_tags()[0]}.whl'
            web[f'https://github.com/Tomodovodoo/HeXO/releases/download/engines-v1/{name}'] = data = \
                archive('x.zip', {f'{crate}/__init__.py': b'', f'{crate}-0.1.0.dist-info/WHEEL': b''})
            release[name] = sha(data)
        release['shrimp-0.1.0-cp30-cp30-other.whl'] = '0' * 64
        self.manifest['release']['files'] = release
        with unittest.mock.patch('engine_setup.importlib.util.find_spec', return_value=object()):
            job = self.finish(self.setups(web), 'shrimp')
        self.assertEqual(job.state, 'done', job.error)
        folder = self.models / 'shrimp'
        self.assertTrue((folder / 'site' / 'shrimp' / '__init__.py').is_file())
        self.assertTrue((folder / 'site' / 'hexo_engine' / '__init__.py').is_file())
        self.assertEqual((folder / 'rivals/shrimp/models/shrimp_main7_infer.pt').read_bytes(), b'file 3')
        self.assertIn('shrimp_driver.py', (folder / 'launch.py').read_text(encoding='utf-8'))
        entries = scan(self.models)
        shrimp = entries['six:Shrimp']
        self.assertEqual(shrimp['command'], [sys.executable, 'shrimp/launch.py', '--threads', '2'])
        self.assertEqual(json.loads((self.models / 'shrimp.json').read_text(encoding='utf-8'))['command'][0], 'python')
        self.assertTrue(shrimp['mirrored'])
        self.assertEqual(shrimp['presets']['strong'], dict(nodes=1, args=['--visits', '512']))
        self.assertIn((folder / 'rivals/shrimp/models/shrimp_main7_infer.pt').resolve(), shrimp['files'])
        self.assertEqual(len(shrimp['files']), len(self.manifest['shrimp']['files']))
        self.assertFalse(any(e['kind'] == 'bubble' for e in entries.values()))
        unfinished = self.models / '.setup' / 'shrimp' / 'shrimp' / 'rivals' / 'shrimp' / 'models' / 'weights.pt'
        unfinished.parent.mkdir(parents=True)
        unfinished.write_bytes(b'weights')
        self.assertFalse(any(e['kind'] == 'bubble' for e in scan(self.models).values()))

    def test_the_shrimp_launcher_runs_the_driver_as_a_script_with_its_modules(self):
        root = Path(self.folder.name) / 'shrimp'
        (root / 'arena' / 'drivers').mkdir(parents=True)
        (root / 'site').mkdir()
        (root / 'launch.py').write_text(engine_setup.SHRIMP_LAUNCHER, encoding='utf-8')
        (root / 'arena' / 'drivers' / 'sixdriver.py').write_text('NAME = "six"\n', encoding='utf-8')
        (root / 'site' / 'shrimp.py').write_text('NAME = "shrimp"\n', encoding='utf-8')
        (root / 'arena' / 'drivers' / 'shrimp_driver.py').write_text(
            'import sys\nimport shrimp, sixdriver\nif __name__ == "__main__":\n'
            '    print(shrimp.NAME, sixdriver.NAME, *sys.argv[1:])\n', encoding='utf-8')
        done = subprocess.run([sys.executable, '-S', 'launch.py', '--visits', '8'], cwd=root, capture_output=True,
                              text=True, timeout=30)
        self.assertEqual((done.stdout.split(), done.stderr), (['shrimp', 'six', '--visits', '8'], ''))

    def test_shrimp_refuses_a_python_without_pytorch(self):
        with unittest.mock.patch('engine_setup.importlib.util.find_spec', return_value=None):
            job = self.finish(self.setups({}), 'shrimp')
        self.assertIn('PyTorch', job.error)

    def test_seal_is_compiled_here_from_pinned_headers(self):
        self.assertIn('C++ compiler', self.finish(self.setups({}), 'seal').error)
        files, web = [], {}
        for number, file in enumerate(self.manifest['seal']['files']):
            web[f'https://example.test/{number}'] = f'header {number}'.encode()
            files.append(file | dict(url=f'https://example.test/{number}', sha256=sha(web[f'https://example.test/{number}'])))
        self.manifest['seal']['files'] = files
        commands = []

        def compile(command, cwd=None, env=None):
            commands.append(command)
            Path(command[command.index('-o') + 1]).write_bytes(b'library')
        with unittest.mock.patch('engine_setup.run', side_effect=compile):
            job = self.finish(self.setups(web, which=lambda name: f'/bin/{name}' if name == 'clang++' else None), 'seal')
        self.assertEqual(job.state, 'done', job.error)
        self.assertEqual(commands[0][0], '/bin/clang++')
        self.assertEqual(Path(commands[0][commands[0].index('-I') + 1]).name, 'cpp')
        seal = [e for e in scan(self.models).values() if e['kind'] == 'seal']
        self.assertEqual(len(seal), 1)
        self.assertEqual(seal[0]['library'].read_bytes(), b'library')
        self.assertTrue(next(e for e in self.setups({}).catalogue() if e['engine'] == 'seal')['installed'])

    def test_a_setup_never_replaces_files_it_did_not_make(self):
        (self.models / 'strix').mkdir()
        (self.models / 'strix' / 'mine.txt').write_text('keep')
        staged = Path(self.folder.name) / 'staged'
        staged.mkdir()
        with self.assertRaisesRegex(SetupError, 'already exists'):
            self.setups({}).place(staged, 'strix')
        (self.models / 'seal.json').write_text(json.dumps(dict(name='Seal', kind='seal', library='x')))
        with self.assertRaisesRegex(SetupError, 'already exists'):
            self.setups({}).place(staged, 'seal', dict(name='Seal'))
        self.assertEqual((self.models / 'strix' / 'mine.txt').read_text(), 'keep')

    def test_a_setup_waits_for_the_match_to_end_before_it_is_done(self):
        refusals = [ValueError('Stop the match before changing its players or position')] * 2

        def rescan():
            if refusals:
                raise refusals.pop()
            self.rescans += 1
        path = Path(self.folder.name) / 'engines.json'
        path.write_text(json.dumps(self.manifest), encoding='utf-8')
        setups = Setups(self.models, rescan, dict, path, Web({}), lambda name: None, lambda: None)
        setups.retry = .05
        with unittest.mock.patch.object(Setups, 'seal', lambda self, job, work: None):
            job = setups.start('seal')
            self.assertEqual(setups.wait('seal', 5).state, 'done')
        self.assertEqual((refusals, self.rescans), ([], 1))
        self.assertFalse(job.json()['busy'])
        refusals.append(PermissionError('models folder locked'))
        with unittest.mock.patch.object(Setups, 'seal', lambda self, job, work: None):
            setups.start('seal')
            self.assertEqual(setups.wait('seal', 5).json()['error'], 'models folder locked')

    def test_rescans_run_one_at_a_time(self):
        inside, overlaps = [], []

        def scan_slowly():
            overlaps.append(bool(inside))
            inside.append(1)
            threading.Event().wait(.2)
            inside.pop()
            return entries()
        session = Session(entries(), FakeEngines(), Evaluations(), scan_slowly)
        try:
            threads = [threading.Thread(target=session.rescan) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        finally:
            session.close()
        self.assertEqual(overlaps, [False, False])

    def test_a_clocked_seal_seat_names_its_library(self):
        library = self.models / 'seal' / 'hexo_seal.dll'
        library.parent.mkdir()
        library.write_bytes(b'library')
        (self.models / 'seal.json').write_text(json.dumps(dict(name='Seal', kind='seal', library='seal/hexo_seal.dll')))
        session = Session(scan(self.models), FakeEngines(), Evaluations())
        try:
            config = session.timed_config(dict(engine='seal:Seal', budget=dict(ms=100)))
        finally:
            session.close()
        self.assertEqual(config, dict(kind='seal', max_ms=100, library=str(library.resolve())))

    def test_strix_entries_differ_by_executable(self):
        for name in ('a', 'b'):
            (self.models / f'{name}.json').write_text(json.dumps(dict(name='Strix', kind='strix', model='m.safetensors',
                                                                      engine=f'{name}.exe')))
        strix = [e for e in scan(self.models).values() if e['kind'] == 'strix']
        self.assertEqual(len({e['id'] for e in strix}), 2)
        self.assertEqual({e['engine'].name for e in strix}, {'a.exe', 'b.exe'})

    def test_six_is_offered_only_where_it_publishes_a_build(self):
        setups = self.setups({})
        with unittest.mock.patch('engine_setup.system', return_value='linux-arm64'):
            self.assertEqual([e['engine'] for e in setups.catalogue()], ['strix', 'shrimp', 'seal'])
        with unittest.mock.patch('engine_setup.system', return_value='macos-arm64'):
            self.assertEqual([e['engine'] for e in setups.catalogue()], ['six', 'strix', 'shrimp', 'seal'])

    def test_read_only_leftovers_of_an_earlier_setup_are_cleared(self):
        leftover = self.models / '.setup' / 'seal' / 'clone' / '.git' / 'pack'
        leftover.parent.mkdir(parents=True)
        leftover.write_bytes(b'pack')
        leftover.chmod(0o444)
        with unittest.mock.patch.object(Setups, 'seal', lambda self, job, work: None):
            setups = self.setups({})
            setups.start('seal')
            self.assertEqual(setups.wait('seal', 5).json()['error'], None)
        self.assertFalse((self.models / '.setup' / 'seal').exists())

    def test_a_second_start_joins_the_running_setup(self):
        gate = threading.Event()
        setups = self.setups({})
        with unittest.mock.patch.object(Setups, 'seal', lambda self, job, work: gate.wait(5)):
            first = setups.start('seal')
            self.assertIs(setups.start('seal'), first)
            self.assertEqual(next(e for e in setups.catalogue() if e['engine'] == 'seal')['job']['state'], 'running')
            gate.set()
            self.assertEqual(setups.wait('seal', 5).state, 'done')
        with self.assertRaises(ValueError):
            setups.start('kraken')


class Pieces(unittest.TestCase):
    def test_six_members_and_unsafe_paths(self):
        self.assertEqual(six_member('Six/runs/rl/gen-0455/net.onnx'), 'gen-0455.onnx')
        self.assertEqual(six_member('Six/engine/sixengine.exe'), 'sixengine.exe')
        self.assertIsNone(six_member('Six/node/node.exe'))
        self.assertIsNone(six_member('Six/runs/rl/gen-0455/other.onnx'))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.zip'
            path.write_bytes(archive('bad.zip', {'Six/engine/../../escape': b'x'}))
            with self.assertRaisesRegex(SetupError, 'leaves'):
                unpack(path, Path(folder) / 'six', six_member)

    def test_source_trees_keep_only_the_listed_paths_without_the_top_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.zip'
            path.write_bytes(archive('source.zip', {'repo-abc/Cargo.toml': b'w', 'repo-abc/packages/shrimp/lib.rs': b's',
                                                    'repo-abc/packages/other/deep/file.py': b'o', 'repo-abc/README.md': b'r'}))
            root = engine_setup.source_tree(path, Path(folder) / 'src', ['Cargo.toml', 'packages/shrimp/'])
            self.assertEqual(sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()),
                             ['Cargo.toml', 'packages/shrimp/lib.rs'])
            self.assertFalse(path.exists())

    def test_wheels_match_the_interpreters_tags(self):
        windows = ['cp314-cp314-win_amd64']
        linux = ['cp312-cp312-manylinux_2_31_x86_64', 'cp312-cp312-manylinux_2_17_x86_64', 'cp312-abi3-linux_x86_64']
        self.assertTrue(wheel_fits('shrimp-0.1.0-cp314-cp314-win_amd64.whl', 'shrimp', windows))
        self.assertFalse(wheel_fits('hexo_engine-0.1.0-cp314-cp314-win_amd64.whl', 'shrimp', windows))
        self.assertFalse(wheel_fits('shrimp-0.1.0-cp313-cp313-win_amd64.whl', 'shrimp', windows))
        self.assertTrue(wheel_fits('shrimp-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', 'shrimp',
                                   linux))
        self.assertFalse(wheel_fits('shrimp-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl', 'shrimp', linux))
        self.assertFalse(wheel_fits('hexo-strix-learned-linux-x64', 'shrimp', linux))
        self.assertTrue(wheel_fits(f'shrimp-0.1.0-{interpreter_tags()[0]}.whl', 'shrimp'))

    def test_job_progress_counts_steps_and_bytes(self):
        job = engine_setup.Job(4)
        job.advance()
        job.fraction = .5
        self.assertEqual(job.json()['progress'], .375)
        job.state = 'done'
        self.assertEqual(job.json()['progress'], 1.0)


class Http(unittest.TestCase):
    def test_the_page_lists_and_starts_setups(self):
        started = []

        class Fake:
            def catalogue(self):
                return [dict(engine='six', name='Six', kind='six', installed=False, job=None)]

            def start(self, engine):
                if engine != 'six':
                    raise ValueError(f'Unknown engine {engine}')
                started.append(engine)
        server = ThreadingHTTPServer(('127.0.0.1', 0), type('H', (Handler,), dict(setups=Fake())))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        root = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(root + '/setup', timeout=5) as response:
                self.assertEqual(json.load(response)['engines'][0]['engine'], 'six')
            request = Request(root + '/setup', json.dumps(dict(engine='six')).encode(), {'Content-Type': 'application/json'})
            with urlopen(request, timeout=5) as response:
                self.assertFalse(json.load(response)['engines'][0]['installed'])
            self.assertEqual(started, ['six'])
            request = Request(root + '/setup', json.dumps(dict(engine='kraken')).encode(), {'Content-Type': 'application/json'})
            with self.assertRaises(HTTPError) as refused:
                urlopen(request, timeout=5)
            self.assertEqual(refused.exception.code, 400)
            refused.exception.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
