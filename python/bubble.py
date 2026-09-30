"""Start, stop and inspect a Bubble training run, or play against it.

Service identity is checked through the process command line, which needs Windows or a /proc file system.

`train` creates the run directory and its first checkpoint when they are missing, then starts the learner,
the actors, the evaluator, the proof pass and the dashboard as detached processes. Their process ids go to
`<run>/processes.json`, their output to `<run>/logs/`. `stop` ends those processes, `status` reports them,
and `play` serves the browser game against the run's champion, a given weights file, or the newest
released Bubble, which it downloads when the run has no checkpoints.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'python'
SERVICES = ('learner', 'actors', 'evaluator', 'proof', 'dashboard')
RELEASES = 'https://github.com/Tomodovodoo/HeXO/releases/'


def resolve_device(device):
    """'auto' picks cuda when torch can see a GPU; other values pass through."""
    if device != 'auto':
        return device
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def commands(run, actors=4, dashboard_port=8766, seal=False, kernels='reference'):
    """The command line of every service for `run`, as argument lists starting with the interpreter."""
    python = [sys.executable, '-u']
    evaluator = [*python, str(PYTHON / 'dense_eval.py'), 'loop', '--run', str(run), '--net-kernels', kernels]
    if not seal:
        evaluator += ['--eval-anchor-games', '0', '--no-eval-anchor-on-promotion', '--eval-anchor-target-halfwidth', '0']
    return dict(
        learner=[*python, str(PYTHON / 'dense_learn.py'), '--run', str(run), '--net-kernels', kernels],
        actors=[*python, str(PYTHON / 'dense_selfplay.py'), '--run', str(run), '--processes', str(actors),
                '--net-kernels', kernels],
        evaluator=evaluator,
        proof=[*python, str(PYTHON / 'dense_solve.py'), '--run', str(run)],
        dashboard=[*python, str(PYTHON / 'dashboard.py'), '--run', str(run), '--port', str(dashboard_port)])


def spawn(command, log):
    """Start `command` detached from this terminal, writing its output to `log`.out.log and `log`.err.log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    out = open(f'{log}.out.log', 'ab')
    err = open(f'{log}.err.log', 'ab')
    env = {**os.environ, 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '2')}
    flags = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == 'nt' \
        else dict(start_new_session=True)
    return subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, **flags).pid


def arguments(pid):
    """The argument list of the process `pid`, or [] when no such process exists."""
    if os.name == 'nt':
        query = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"
        line = subprocess.run(['powershell', '-NoProfile', '-Command', query], capture_output=True, text=True).stdout.strip()
        return [part.strip('"') for part in shlex.split(line, posix=False)] if line else []
    try:
        return [part.decode(errors='replace') for part in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if part]
    except OSError:
        return []


def matches(parts, script, run):
    """True when argument list `parts` runs `script` (by file name) on exactly the run directory `run`."""
    return any(Path(part).name == script for part in parts) and run in parts


def alive(pid, script, run, arguments=arguments):
    """True when `pid` is this run's `script`, so a reused pid, or the same service of another run, never passes."""
    return matches(arguments(pid), script, run)


def kill(pid):
    """End the process and, on Windows, everything it started."""
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass


class Launcher:
    """Service bookkeeping for one run directory; `spawn`, `alive` and `kill` are injectable for tests."""

    def __init__(self, run, spawn=spawn, arguments=arguments, kill=kill):
        self.run, self.spawn, self.arguments, self.kill = Path(run).resolve(), spawn, arguments, kill
        self.state_file = self.run / 'processes.json'

    def state(self):
        return json.loads(self.state_file.read_text(encoding='utf-8')) if self.state_file.exists() else {}

    def running(self):
        return {name: entry for name, entry in self.state().items() if self.alive(entry)}

    def alive(self, entry):
        return alive(entry['pid'], entry['script'], str(self.run), self.arguments)

    def prepare(self, device, run_steps):
        """Create the run configuration and the first checkpoint when either is missing; `run_steps` runs a command."""
        if not (self.run / 'config.json').exists():
            run_steps([sys.executable, str(PYTHON / 'dense_config.py'), '--run', str(self.run), '--device', device])
        exports = (self.run / 'checkpoints' / 'main').glob('*/ema.pt')
        if not any(path.with_name('manifest.json').exists() for path in exports):
            run_steps([sys.executable, str(PYTHON / 'dense_learn.py'), '--run', str(self.run), '--steps', '0'])

    def lock(self):
        """Create `processes.lock` exclusively, holding this process id. A lock whose owner is no longer a running
        launcher was abandoned by a crash and is taken over."""
        path = self.run / 'processes.lock'
        for attempt in range(2):
            try:
                handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                try:
                    owner = int(path.read_text(encoding='utf-8').strip() or 0)
                except (OSError, ValueError):
                    owner = 0
                if attempt or any(Path(part).name == 'bubble.py' for part in self.arguments(owner)):
                    raise RuntimeError(f'another start is in progress for {self.run} ({path} held by pid {owner})') from None
                path.unlink(missing_ok=True)
                continue
            with os.fdopen(handle, 'w', encoding='utf-8') as stream:
                stream.write(str(os.getpid()))
            return path

    def start(self, plan, prepare=lambda: None):
        """Under the run lock: `prepare` the run, check nothing is running, spawn every service, publish the state."""
        lock = self.lock()
        state = {}
        try:
            prepare()
            running = self.running()
            if running:
                raise RuntimeError(f"{', '.join(sorted(running))} already running for {self.run}; stop first")
            for name in SERVICES:
                script = Path(next(part for part in plan[name] if part.endswith('.py'))).name
                pid = self.spawn(plan[name], self.run / 'logs' / name)
                state[name] = dict(pid=pid, script=script, command=plan[name], started_at=time.time())
            partial = self.state_file.with_suffix('.json.tmp')
            partial.write_text(json.dumps(state, indent=2), encoding='utf-8')
            os.replace(partial, self.state_file)
        except BaseException:
            for entry in state.values():
                self.kill(entry['pid'])
            raise
        finally:
            lock.unlink(missing_ok=True)
        return state

    def stop(self):
        state = self.state()
        for name in reversed(SERVICES):
            if name in state and self.alive(state[name]):
                self.kill(state[name]['pid'])
        if self.state_file.exists():
            self.state_file.unlink()
        return sorted(state)

    def status(self):
        """One line per service: alive or gone, plus what its status file says."""
        state, lines = self.state(), []
        files = dict(learner='learner-status.json', actors='actor-status.json', evaluator='evaluator-status.json',
                     proof='solver-status.json')
        for name in SERVICES:
            entry = state.get(name)
            if entry is None:
                lines.append(f'{name:<10} not started')
                continue
            life = 'alive' if self.alive(entry) else 'gone'
            detail = ''
            path = self.run / files.get(name, '')
            if name in files and path.exists():
                status = json.loads(path.read_text(encoding='utf-8'))
                age = int(time.time() - status.get('updated_at', 0))
                detail = ' '.join(str(status[k]) for k in ('stage', 'step', 'checkpoint') if status.get(k) is not None)
                detail = f'{detail} ({age}s ago)'
            lines.append(f'{name:<10} {life:<5} pid {entry["pid"]} {detail}'.rstrip())
        champion = self.run / 'champion.json'
        if champion.exists():
            lines.append(f"champion   {json.loads(champion.read_text(encoding='utf-8'))['checkpoint']}")
        return lines


def fetch(url):
    """GET `url`; returns the body and the final URL after redirects."""
    from urllib.request import Request, urlopen
    with urlopen(Request(url, headers={'User-Agent': 'bubble'})) as response:
        return response.read(), response.geturl()


def exports(run):
    return sorted(Path(run).glob('checkpoints/*/*/ema.pt'))


def install(run, source, step, variant='main'):
    """Copy the weights file `source` into `run` as checkpoint `variant/step` and make it the champion. Files land
    under temporary names and are renamed, so a reader never sees a half-written checkpoint."""
    checkpoint = f'{variant}/{step:06d}'
    target = Path(run) / 'checkpoints' / checkpoint
    target.mkdir(parents=True, exist_ok=True)
    manifest = Path(source).with_name('manifest.json')
    for name, path in (('ema.pt', Path(source)), ('manifest.json', manifest)):
        if path.exists():
            shutil.copyfile(path, target / (name + '.tmp'))
            os.replace(target / (name + '.tmp'), target / name)
    champion = Path(run) / 'champion.json'
    champion.with_suffix('.json.tmp').write_text(json.dumps(dict(checkpoint=checkpoint)), encoding='utf-8')
    os.replace(champion.with_suffix('.json.tmp'), champion)
    return target


def download(run, fetch=fetch):
    """Fetch the newest released Bubble into `run`. `releases/latest` redirects to the release's tag page, whose
    number is the checkpoint step; the assets are then read from that release."""
    from urllib.error import HTTPError
    _, tag_page = fetch(RELEASES + 'latest')
    tag = tag_page.rstrip('/').rsplit('/', 1)[-1]
    step = int(re.search(r'(\d+)', tag).group(1))
    staging = Path(run) / 'checkpoints' / 'download'
    staging.mkdir(parents=True, exist_ok=True)
    (staging / 'ema.pt').write_bytes(fetch(f'{RELEASES}download/{tag}/ema.pt')[0])
    try:
        (staging / 'manifest.json').write_bytes(fetch(f'{RELEASES}download/{tag}/manifest.json')[0])
    except HTTPError:
        pass
    target = install(run, staging / 'ema.pt', step)
    shutil.rmtree(staging)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'stop', 'status', 'play'):
        p = sub.add_parser(name)
        p.add_argument('--run', default='runs/bubble')
    train = sub.choices['train']
    train.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    train.add_argument('--actors', type=int, default=4, help='self-play processes')
    train.add_argument('--dashboard-port', type=int, default=8766)
    train.add_argument('--seal', action='store_true', help='rate champions against Seal (needs the Seal build)')
    train.add_argument('--net-kernels', choices=['reference', 'fused'], default='reference')
    play = sub.choices['play']
    play.add_argument('--port', type=int, default=8765)
    play.add_argument('--model', type=Path, help="an ema.pt file to play instead of the run's checkpoints")
    play.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    launcher = Launcher(args.run)
    if args.command == 'train':
        plan = commands(launcher.run, args.actors, args.dashboard_port, args.seal, args.net_kernels)
        prepare = lambda: launcher.prepare(resolve_device(args.device),
                                           lambda command: subprocess.run(command, cwd=ROOT, check=True))
        for name, entry in launcher.start(plan, prepare).items():
            print(f'{name:<10} pid {entry["pid"]}')
        print(f'dashboard  http://127.0.0.1:{args.dashboard_port}  logs {launcher.run / "logs"}')
    elif args.command == 'stop':
        stopped = launcher.stop()
        print('stopped ' + ', '.join(stopped) if stopped else 'nothing running')
    elif args.command == 'status':
        print('\n'.join(launcher.status()))
    else:
        run = launcher.run
        if args.model:
            digits = re.findall(r'\d+', str(args.model.resolve().parent.name))
            run = ROOT / 'runs' / 'play'
            install(run, args.model, int(digits[-1]) if digits else 0)
        elif not exports(run):
            print(f'no checkpoints under {run}; downloading the latest released Bubble')
            print(f'installed {download(run)}')
        sys.exit(subprocess.call([sys.executable, str(PYTHON / 'play.py'), '--dense-run', str(run),
                                  '--port', str(args.port), '--device', args.device], cwd=ROOT))


if __name__ == '__main__':
    main()
