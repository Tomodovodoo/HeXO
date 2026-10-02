"""One-click setup of the external engines named in tools/engines.json.

A setup downloads an engine, checks every file against its SHA-256, lays it out in the models folder the way
`play.scan` reads it and rescans. Work happens in `<models>/.setup/<engine>`; the finished folder moves to
`<models>/<engine>` with a `setup.json` marker, and engines that need a JSON entry get `<models>/<engine>.json`.
Engines that need compiling are built here when the toolchain is present and otherwise come from the release named
in the manifest, whose file hashes are checked in. Nothing here needs the training modules.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
TOOLS = next((d for d in (HERE.parent / 'tools', HERE / 'tools') if (d / 'engines.json').is_file()),
             HERE.parent / 'tools')
MANIFEST = TOOLS / 'engines.json'
CHUNK = 1 << 20
WINDOWS = os.name == 'nt'
SHRIMP_LAUNCHER = '''import runpy
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
drivers = here / 'arena' / 'drivers'
sys.path[:0] = [str(drivers), str(here / 'site')]
sys.argv[0] = str(drivers / 'shrimp_driver.py')
runpy.run_path(sys.argv[0], run_name='__main__')
'''
INSTALLED = dict(six=lambda e: e['kind'] == 'six' and bool(e.get('networks')),
                 strix=lambda e: e['kind'] == 'strix',
                 shrimp=lambda e: e['kind'] == 'six' and e['badge'] == 'shrimp',
                 seal=lambda e: e['kind'] == 'seal')


class SetupError(RuntimeError):
    pass


def system():
    """This machine as a release platform name: windows-x64, linux-x64, macos-arm64 and so on."""
    name = {'win32': 'windows', 'linux': 'linux', 'darwin': 'macos'}.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    return f"{name}-{dict(amd64='x64', x86_64='x64', aarch64='arm64').get(machine, machine)}"


def interpreter_tags():
    """The wheel tags this interpreter accepts, as 'python-abi-platform' strings, most specific first."""
    from packaging.tags import sys_tags
    return [str(tag) for tag in sys_tags()]


def wheel_fits(name, crate, tags=None):
    """True when the wheel file `name` is `crate`'s and one of its tags is in `tags` (this interpreter's by
    default); compressed tag sets like manylinux_2_17_x86_64.manylinux2014_x86_64 count each member."""
    parts = name.removesuffix('.whl').split('-')
    if not name.endswith('.whl') or len(parts) < 5 or parts[0] != crate:
        return False
    accepted = set(interpreter_tags() if tags is None else tags)
    return any(f'{python}-{abi}-{where}' in accepted for python in parts[-3].split('.')
               for abi in parts[-2].split('.') for where in parts[-1].split('.'))


def cargo():
    """The cargo executable on PATH or in ~/.cargo/bin, else None."""
    home = Path.home() / '.cargo' / 'bin' / ('cargo.exe' if WINDOWS else 'cargo')
    return shutil.which('cargo') or (str(home) if home.is_file() else None)


def compiler(which=shutil.which):
    """A GCC-compatible C++ compiler on PATH, else None."""
    return next(filter(None, map(which, ('g++', 'clang++', 'c++'))), None)


def run(command, cwd=None, env=None):
    """Run a build command at low priority with two build jobs; a failure prints its output to stderr and raises
    SetupError with its last line."""
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW if WINDOWS else 0
    environment = os.environ | dict(CARGO_BUILD_JOBS='2') | (env or {})
    done = subprocess.run(list(map(str, command)), cwd=cwd, env=environment, capture_output=True, text=True,
                          errors='replace', creationflags=flags)
    if done.returncode:
        print(done.stdout, done.stderr, sep='\n', file=sys.stderr, flush=True)
        lines = (done.stderr or done.stdout).strip().splitlines()
        raise SetupError(lines[-1] if lines else f'{Path(str(command[0])).name} failed')
    return done.stdout


def remove(path):
    """Delete the folder `path` if it exists, read-only files too, such as a git clone's objects."""
    def writable(function, name, _):
        os.chmod(name, stat.S_IWRITE)
        function(name)
    if Path(path).exists():
        shutil.rmtree(path, **{'onexc' if sys.version_info >= (3, 12) else 'onerror': writable})


def with_cargo(cargo_path):
    """The environment change that puts `cargo_path`'s folder first on PATH."""
    return dict(PATH=str(Path(cargo_path).parent) + os.pathsep + os.environ.get('PATH', ''))


def build_strix(work, cargo_path):
    """Build the pinned Strix wrapper with tools/build_strix_learned.py, both copied into `work` so nothing is
    written beside the installed tools, from a clone in `work`; returns the executable, with its
    build-provenance.json beside it."""
    tools = Path(work) / 'tools'
    shutil.copytree(TOOLS / 'strix_learned', tools / 'strix_learned',
                    ignore=shutil.ignore_patterns('target', 'build-local.toml'))
    shutil.copy2(TOOLS / 'build_strix_learned.py', tools)
    run([sys.executable, tools / 'build_strix_learned.py', Path(work) / 'hexo-strix', '--cargo', cargo_path],
        env=with_cargo(cargo_path))
    return tools / 'strix_learned' / 'target' / 'release' / ('hexo-strix-learned.exe' if WINDOWS else
                                                             'hexo-strix-learned')


def build_wheels(source, out, crates, cargo_path, interpreters=None):
    """Build `crates` of the hexo-bot checkout `source` into wheels in `out` with maturin and `cargo_path`, for
    `interpreters` (paths), or every interpreter maturin finds when None; returns the wheel paths."""
    tools = Path(out) / '.maturin'
    run([sys.executable, '-m', 'pip', 'install', '--quiet', '--target', tools, 'maturin>=1.7,<2'])
    maturin = next((path for folder in ('bin', 'Scripts') for path in sorted((tools / folder).glob('maturin*'))), None)
    if maturin is None:
        raise SetupError('maturin did not install')
    chosen = ['--find-interpreter'] if interpreters is None else [x for i in interpreters for x in ('-i', i)]
    for crate in crates:
        run([maturin, 'build', '--release', *chosen, '-m', Path(source) / 'packages' / crate / 'Cargo.toml',
             '--out', out], env=with_cargo(cargo_path) | dict(CARGO_TARGET_DIR=str(Path(out) / '.target')))
    remove(tools)
    remove(Path(out) / '.target')
    return sorted(Path(out).glob('*.whl'))


class Job:
    """Progress of one setup: `state` running, done or failed; `progress` from 0 to 1 over its steps; `busy`
    while a step has no byte count; `error` when it failed."""

    def __init__(self, steps):
        self.state, self.error, self.steps, self.step, self.fraction, self.busy = 'running', None, steps, 0, 0.0, False

    def advance(self, busy=False):
        self.step, self.fraction, self.busy = min(self.step + 1, self.steps), 0.0, busy

    def json(self):
        progress = 1.0 if self.state == 'done' else min(1.0, (self.step + self.fraction) / max(1, self.steps))
        return dict(state=self.state, progress=round(progress, 4), busy=self.busy and self.state == 'running',
                    error=self.error)


class Setups:
    """Setup jobs for the engines in `manifest`, one thread each. `rescan` refreshes the registry after an install;
    while it refuses with ValueError, as during a match, the job keeps running and tries again every `retry`
    seconds. `entries` returns the registry's current entries, which decide what counts as installed."""

    def __init__(self, models, rescan=lambda: None, entries=lambda: {}, manifest=MANIFEST, opener=urlopen,
                 which=shutil.which, cargo=cargo):
        self.models, self.rescan, self.entries = Path(models).resolve(), rescan, entries
        self.manifest = json.loads(Path(manifest).read_text(encoding='utf-8'))
        self.opener, self.which, self.cargo = opener, which, cargo
        self.jobs, self.threads, self.lock, self.retry = {}, {}, threading.Lock(), 5

    def catalogue(self):
        """Every engine that can be set up on this system as {engine, name, kind, badge, installed, job}: `installed` is
        the id of the registry entry that provides it, else None, and `job` the last setup's progress. A setup
        turns done once the registry has been rescanned. Six is left out where it publishes no build."""
        entries = list(self.entries().values())
        offered = [key for key in INSTALLED if key != 'six' or system() in self.manifest['six']['assets']]
        with self.lock:
            return [dict(engine=key, name=self.manifest[key]['name'], kind=self.manifest[key]['kind'],
                         badge=self.manifest[key].get('badge', self.manifest[key]['kind']),
                         installed=next((e['id'] for e in entries if INSTALLED[key](e)), None),
                         job=self.jobs[key].json() if key in self.jobs else None) for key in offered]

    def start(self, engine):
        """Start setting up `engine` unless its setup is already running."""
        if engine not in INSTALLED:
            raise ValueError(f'Unknown engine {engine}')
        with self.lock:
            if engine in self.jobs and self.jobs[engine].state == 'running':
                return self.jobs[engine]
            steps = dict(six=3, strix=3, shrimp=len(self.manifest['shrimp']['files']) + 3,
                         seal=len(self.manifest['seal']['files']) + 2)[engine]
            job = self.jobs[engine] = Job(steps)
            self.threads[engine] = threading.Thread(target=self.work, args=(engine, job), daemon=True)
            self.threads[engine].start()
        return job

    def wait(self, engine, timeout=None):
        self.threads[engine].join(timeout)
        return self.jobs[engine]

    def work(self, engine, job):
        work = self.models / '.setup' / engine
        try:
            remove(work)
            work.mkdir(parents=True)
            getattr(self, engine)(job, work)
        except Exception as error:
            job.state, job.error = 'failed', str(error) or type(error).__name__
            return
        finally:
            with contextlib.suppress(OSError):
                remove(work)
        job.busy = True
        while True:
            try:
                self.rescan()
                break
            except ValueError:
                time.sleep(self.retry)
            except Exception as error:
                job.state, job.error = 'failed', str(error) or type(error).__name__
                return
        job.state = 'done'

    # Steps shared by the recipes

    def download(self, job, url, path, sha256=None, size=None):
        """Stream `url` into `path`, counting bytes into `job`; a SHA-256 mismatch removes the file and fails."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        digest, done = hashlib.sha256(), 0
        with self.opener(Request(url, headers={'User-Agent': 'bubble'}), timeout=60) as response, \
                open(path, 'wb') as out:
            total = size or int(response.headers.get('Content-Length') or 0)
            while chunk := response.read(CHUNK):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                job.fraction = min(done / total, 1.0) if total else 0.0
        if sha256 and digest.hexdigest() != sha256:
            path.unlink()
            raise SetupError(f'{path.name} does not match its SHA-256')
        job.advance()
        return path

    def fetch_json(self, url):
        with self.opener(Request(url, headers={'User-Agent': 'bubble', 'Accept': 'application/vnd.github+json'}),
                         timeout=30) as response:
            return json.loads(response.read())

    def release_files(self):
        """The engines release's files and their SHA-256s: the ones pinned in the manifest, or, while none are
        pinned, the release's own SHA256SUMS; empty when neither is there."""
        release = self.manifest['release']
        if release['files']:
            return release['files']
        url = f"https://github.com/{release['repository']}/releases/download/{release['tag']}/SHA256SUMS"
        try:
            with self.opener(Request(url, headers={'User-Agent': 'bubble'}), timeout=30) as response:
                return parse_sums(response.read().decode())
        except (OSError, ValueError):
            return {}

    def published(self, job, fits, path, failure=None):
        """Download the one file of the engines release that `fits(name)` accepts to `path`, checked against its
        hash from `release_files`. Without one, raise `failure`, the local build's error, if there was one."""
        release, files = self.manifest['release'], self.release_files()
        names = [name for name in files if fits(name)]
        if not names:
            raise failure or SetupError(f'No published build for {system()}; install the build tools to build it here')
        url = f"https://github.com/{release['repository']}/releases/download/{release['tag']}/{names[0]}"
        return self.download(job, url, path, files[names[0]])

    def place(self, staged, name, entry=None):
        """Move the folder `staged` to `<models>/<name>`, replacing an earlier setup there but never other
        files, then write the JSON entry `entry` as `<models>/<name>.json`."""
        target, entry_path = self.models / name, self.models / f'{name}.json'
        if target.exists() and not (target / 'setup.json').is_file():
            raise SetupError(f'{target} already exists')
        if entry and entry_path.exists() and 'setup' not in json.loads(entry_path.read_text(encoding='utf-8')):
            raise SetupError(f'{entry_path} already exists')
        (Path(staged) / 'setup.json').write_text(json.dumps(dict(engine=name)), encoding='utf-8')
        remove(target)
        os.replace(staged, target)
        if entry:
            temporary = entry_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(entry | dict(setup=name), indent=1), encoding='utf-8')
            os.replace(temporary, entry_path)

    # Recipes, one per engine

    def six(self, job, work):
        """The newest Six release for this system: its engine folder and network, flattened into models/six."""
        spec = self.manifest['six']
        if system() not in spec['assets']:
            raise SetupError(f'Six has no build for {system()}')
        release = self.fetch_json(f"https://api.github.com/repos/{spec['repository']}/releases/latest")
        job.advance()
        asset = next((a for a in release['assets'] if a['name'].endswith(spec['assets'][system()])), None)
        if asset is None:
            raise SetupError(f"Six {release['tag_name']} has no {spec['assets'][system()]}")
        digest = (asset.get('digest') or '').removeprefix('sha256:')
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise SetupError(f"Six {release['tag_name']} publishes no SHA-256 for {asset['name']}")
        archive = self.download(job, asset['browser_download_url'], work / asset['name'], digest, asset['size'])
        job.busy = True
        staged = work / 'six'
        unpack(archive, staged, six_member)
        archive.unlink()
        binary = staged / ('sixengine.exe' if WINDOWS else 'sixengine')
        if not binary.is_file() or not any(staged.glob('gen-*.onnx')):
            raise SetupError(f"Six {release['tag_name']} has no engine or network")
        binary.chmod(binary.stat().st_mode | 0o111)
        self.place(staged, 'six')
        job.advance()

    def strix(self, job, work):
        """The pinned Strix wrapper, built here or from the release, and the public model from its site."""
        spec = self.manifest['strix']
        staged = work / 'strix'
        staged.mkdir()
        name = spec['executable'] + ('.exe' if WINDOWS else '')
        built = failure = None
        if (toolchain := self.cargo()) and self.which('git'):
            job.busy = True
            try:
                built = build_strix(work, toolchain)
            except Exception as error:
                failure = error
        if built:
            shutil.copy2(built, staged / name)
            if (built.parent / 'build-provenance.json').is_file():
                shutil.copy2(built.parent / 'build-provenance.json', staged / 'build-provenance.json')
            job.advance()
        else:
            self.published(job, lambda n: n == f"{spec['executable']}-{system()}{'.exe' if WINDOWS else ''}",
                           staged / name, failure)
            (staged / name).chmod((staged / name).stat().st_mode | 0o111)
        model = spec['model']
        self.download(job, model['url'], staged / 'model.safetensors', model['sha256'], model['size'])
        self.place(staged, 'strix', dict(name=spec['name'], kind='strix', model='strix/model.safetensors',
                                         engine=f'strix/{name}'))
        job.advance()

    def shrimp(self, job, work):
        """Six's Shrimp driver, Shrimp's weights and search profile, and its two native modules, built here or
        from the release, run by this Python with its own PyTorch."""
        spec = self.manifest['shrimp']
        if sys.version_info < (3, 11) or not all(map(importlib.util.find_spec, ('torch', 'numpy'))):
            raise SetupError('Shrimp needs Python 3.11 or newer with PyTorch and NumPy')
        staged = work / 'shrimp'
        for file in spec['files']:
            self.download(job, file['url'], staged / file['path'], file['sha256'])
        wheels, failure = [], None
        if toolchain := self.cargo():
            try:
                source = self.source(job, spec, work)
                job.busy = True
                wheels = build_wheels(source, work / 'built', spec['crates'], toolchain, [sys.executable])
                job.advance()
            except Exception as error:
                wheels, failure = [], error
        if not wheels:
            wheels = [self.published(job, lambda n, c=crate: wheel_fits(n, c), work / 'wheels' / f'{crate}.whl',
                                     failure) for crate in spec['crates']]
        job.busy = True
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(staged / 'site')
        (staged / 'launch.py').write_text(SHRIMP_LAUNCHER, encoding='utf-8')
        payload = [f"shrimp/{file['path']}" for file in spec['files']] + sorted(
            f'shrimp/{path.relative_to(staged).as_posix()}' for path in (staged / 'site').rglob('*')
            if path.suffix in ('.pyd', '.so'))
        self.place(staged, 'shrimp', dict(name=spec['name'], kind='six', badge=spec['badge'], mirrored=True, presets=spec['presets'],
                                          command=['python', 'shrimp/launch.py', '--threads', '2'],
                                          files=payload))
        job.advance()

    def source(self, job, spec, work):
        """The source tree of `spec`'s repository at its pinned revision, from GitHub's archive."""
        return source_tree(self.download(job, source_url(spec), work / 'source.zip'), work / 'src', spec['sources'])

    def seal(self, job, work):
        """Seal's pinned headers compiled with tools/seal_adapter.cpp. Seal has no licence, so it is never
        republished and needs a C++ compiler here."""
        spec = self.manifest['seal']
        found = compiler(self.which)
        if found is None:
            raise SetupError('Seal needs a C++ compiler (g++ or clang++)')
        for file in spec['files']:
            self.download(job, file['url'], work / 'source' / file['path'], file['sha256'])
        job.busy = True
        staged = work / 'seal'
        staged.mkdir()
        library = 'hexo_seal.dll' if WINDOWS else 'libhexo_seal.so'
        run([found, '-std=c++20', '-O3', '-shared', '-I', work / 'source' / 'cpp', TOOLS / 'seal_adapter.cpp',
             '-o', staged / library, *(['-static-libgcc', '-static-libstdc++'] if WINDOWS else ['-fPIC'])])
        job.advance()
        self.place(staged, 'seal', dict(name=spec['name'], kind='seal', library=f'seal/{library}'))
        job.advance()


def parse_sums(text):
    """{file name: SHA-256} from `sha256sum` output."""
    return {name.lstrip('*'): digest for digest, name in (line.split(maxsplit=1) for line in text.splitlines()
                                                          if line.strip())}


def source_url(spec):
    """GitHub's zip archive of `spec`'s repository at its pinned revision."""
    return f"https://codeload.github.com/{spec['repository']}/zip/{spec['revision']}"


def source_tree(archive, target, keep=('',)):
    """Unpack the files of the GitHub archive `archive` whose paths start with one of `keep` into `target`,
    without the archive's top folder, so paths stay short; removes the archive and returns `target`."""
    def place(name):
        relative = name.partition('/')[2]
        return relative if relative and not relative.endswith('/') and relative.startswith(tuple(keep)) else None
    unpack(archive, target, place)
    Path(archive).unlink()
    return Path(target)


def six_member(name):
    """Where a member of a Six release archive goes in models/six: the engine folder's files keep their place,
    `runs/rl/gen-NNNN/net.onnx` becomes `gen-NNNN.onnx`, everything else is skipped."""
    if found := re.fullmatch(r'Six/runs/rl/(gen-\d+)/net\.onnx', name):
        return f'{found.group(1)}.onnx'
    return name.removeprefix('Six/engine/') if name.startswith('Six/engine/') and not name.endswith('/') else None


def unpack(archive, target, place):
    """Extract the members of the zip or tar.gz `archive` that `place(name)` maps to a relative path, into
    `target`; paths leaving `target` are refused."""
    target = Path(target).resolve()

    def destination(name):
        relative = place(name)
        if relative is None:
            return None
        path = (target / relative).resolve()
        if not path.is_relative_to(target):
            raise SetupError(f'{name} leaves the engine folder')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if str(archive).endswith('.zip'):
        with zipfile.ZipFile(archive) as files:
            for info in files.infolist():
                if path := destination(info.filename):
                    with files.open(info) as source, open(path, 'wb') as out:
                        shutil.copyfileobj(source, out, CHUNK)
    else:
        with tarfile.open(archive) as files:
            for member in files:
                if member.isfile() and (path := destination(member.name)):
                    with files.extractfile(member) as source, open(path, 'wb') as out:
                        shutil.copyfileobj(source, out, CHUNK)
                    path.chmod(member.mode | 0o600)
