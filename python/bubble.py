"""Start, stop and inspect a Bubble training run, or play against it.

Services are identified through their command lines, which needs Windows or a /proc file system; on other
systems `train`, `stop` and `status` refuse to run.

`train` creates the run directory and its first checkpoint when they are missing, then starts the learner,
the actors, the evaluator, the proof pass and the dashboard as detached processes. Their process ids go to
`<run>/processes.json`, their output to `<run>/logs/`. `stop` ends those processes, `status` reports them,
and `play` serves the browser game against a run's champion, a given weights file, or the newest released
Bubble, which it downloads into `runs/play` when that run has no checkpoints.
"""
import argparse
from contextlib import contextmanager
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


def commands(run, actors=4, dashboard_port=8766, seal=False, kernels='reference', proof=True):
    """The command line of every service for `run`, as argument lists starting with the interpreter; the proof pass
    only when `proof`, since it needs the tactical solver build."""
    python = [sys.executable, '-u']
    evaluator = [*python, str(PYTHON / 'dense_eval.py'), 'loop', '--run', str(run), '--net-kernels', kernels]
    if not seal:
        evaluator += ['--eval-anchor-games', '0', '--no-eval-anchor-on-promotion', '--eval-anchor-target-halfwidth', '0']
    plan = dict(
        learner=[*python, str(PYTHON / 'dense_learn.py'), '--run', str(run), '--net-kernels', kernels],
        actors=[*python, str(PYTHON / 'dense_selfplay.py'), '--run', str(run), '--processes', str(actors),
                '--net-kernels', kernels],
        evaluator=evaluator,
        proof=[*python, str(PYTHON / 'dense_solve.py'), '--run', str(run)],
        dashboard=[*python, str(PYTHON / 'dashboard.py'), '--run', str(run), '--port', str(dashboard_port)])
    if not proof:
        del plan['proof']
    return plan


def tactical_built():
    """True when the tactical solver loads with a matching build identity."""
    try:
        from tactical_proof import NativeTactics
        NativeTactics()
    except (OSError, ValueError, KeyError):
        return False
    return True


def spawn(command, log):
    """Start `command` detached from this terminal, writing its output to `log`.out.log and `log`.err.log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    out = open(f'{log}.out.log', 'ab')
    err = open(f'{log}.err.log', 'ab')
    env = {**os.environ, 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '2')}
    flags = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == 'nt' \
        else dict(start_new_session=True)
    return subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, **flags).pid


def arguments_of(line):
    """The arguments of a Windows command line, quotes removed; a line shlex cannot parse is split on spaces."""
    if not line:
        return []
    try:
        return [part.strip('"') for part in shlex.split(line, posix=False)]
    except ValueError:
        return line.split()


def arguments(pid):
    """The argument list of the process `pid`, or [] when no such process exists."""
    if os.name == 'nt':
        query = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"
        line = subprocess.run(['powershell', '-NoProfile', '-Command', query], capture_output=True, text=True).stdout.strip()
        return arguments_of(line)
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


def members(candidates, run, root, parents=None):
    """The pids among `candidates` ({pid: argument list}) that belong to the service `root` of `run`: they name
    the run directory, their parent (`parents`, {pid: ppid}, or a Windows `parent_pid=` argument) is an accepted
    member, or they are multiprocessing workers of the service's own session (`parents` maps them to `root`)."""
    parents = parents or {}
    accepted = {root} | {pid for pid, parts in candidates.items() if run in parts}
    while True:
        added = {pid for pid, parts in candidates.items() if pid not in accepted and
                 (parents.get(pid) in accepted or
                  any(f'parent_pid={parent}' in ' '.join(parts) for parent in accepted))}
        if not added:
            return accepted - {root}
        accepted |= added


def windows_tree(pid, run):
    """Windows: the pids of every live descendant of `pid` that belongs to `run`."""
    query = ("Get-CimInstance Win32_Process | ForEach-Object { "
             "\"$($_.ProcessId)|$($_.ParentProcessId)|$($_.CommandLine)\" }")
    out = subprocess.run(['powershell', '-NoProfile', '-Command', query], capture_output=True, text=True).stdout
    children, lines = {}, {}
    for line in out.splitlines():
        parts = line.split('|', 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
            lines[int(parts[0])] = parts[2]
    descendants, queue = {}, [pid]
    while queue:
        for child in children.get(queue.pop(), []):
            descendants[child] = arguments_of(lines.get(child, ''))
            queue.append(child)
    return sorted(members(descendants, run, pid))


def group_members(pid, run):
    """The pids of processes started under the service `pid` that still belong to it: on Windows its descendants
    by parent id, on POSIX the members of its process group (each service starts its own session). A POSIX group
    whose leader is alive but runs another run's service is not ours; workers of the service's own session count
    even after the leader exited, since they can only have been created inside it."""
    if os.name == 'nt':
        return windows_tree(pid, run)
    if not Path('/proc').is_dir():
        return []
    leader = arguments(pid)
    if leader and run not in leader:
        return []
    group, parents = {}, {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            member, ppid, pgrp, session = int(entry.name), int(fields[1]), int(fields[2]), int(fields[3])
            if pgrp == pid:
                group[member] = arguments(member)
                parents[member] = pid if session == pid and any('multiprocessing' in part for part in group[member]) else ppid
        except (OSError, ValueError, IndexError):
            continue
    return sorted(members(group, run, pid, parents))


def kill(pid, force=False):
    """End the process group; on Windows everything the process started, on POSIX with SIGTERM or, when `force`,
    SIGKILL."""
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
    else:
        try:
            os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
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
        """The service leader still runs this run's script, or workers started under it still exist."""
        return alive(entry['pid'], entry['script'], str(self.run), self.arguments) or bool(self.group(entry))

    def group(self, entry):
        return group_members(entry['pid'], str(self.run))

    def variant(self):
        """The learner's checkpoint variant from the run configuration, `main` for a run without one."""
        try:
            return (self.config() or {})['learner']['variant']
        except (KeyError, TypeError):
            return 'main'

    def config(self):
        try:
            return json.loads((self.run / 'config.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None

    def prepare(self, device, run_steps, actors=None):
        """Create the run configuration (on `device`, `auto` resolved then) and the first checkpoint when either is
        missing; `run_steps` runs a command. An existing run must agree with an explicit `device`, and its phase
        schedule may not wait for more workers than `actors` starts."""
        config = self.config()
        if config is None:
            run_steps([sys.executable, str(PYTHON / 'dense_config.py'), '--run', str(self.run), '--device',
                       resolve_device(device)])
        else:
            saved = config.get('device', device)
            if device != 'auto' and device != saved:
                raise RuntimeError(f'{self.run} is configured for {saved}; pass --device {saved} or start a new run')
            phase_actors = (config.get('learner') or {}).get('phase_actors', 0)
            if actors is not None and phase_actors > actors:
                raise RuntimeError(f'{self.run} phases wait for {phase_actors} actors; pass --actors {phase_actors} or more')
            if phase_actors > 0 and not (config.get('actor') or {}).get('phase_follow', False):
                raise RuntimeError(f'{self.run} phases wait for actors that never acknowledge them; '
                                   'set actor.phase_follow to true in its config.json')
        # Play-only installs (variant `play`, weights without optimizer state) never count as a learner checkpoint.
        exports = (self.run / 'checkpoints' / self.variant()).glob('*/ema.pt')
        learner_files = ('model.pt', 'optimizer.pt', 'manifest.json')
        if not any(all(path.with_name(name).exists() for name in learner_files) for path in exports):
            run_steps([sys.executable, str(PYTHON / 'dense_learn.py'), '--run', str(self.run), '--steps', '0'])

    @contextmanager
    def locked(self):
        """Hold the run's OS lock (`processes.lock`) so starts and stops of one run never overlap. The operating
        system releases it when the holder exits, so a crash leaves nothing to recover."""
        self.run.mkdir(parents=True, exist_ok=True)
        handle = open(self.run / 'processes.lock', 'a+')
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError(f'another start or stop is in progress for {self.run}') from None
        try:
            yield
        finally:
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def start(self, plan, prepare=lambda: None, timeout=30.):
        """Under the run lock: `prepare` the run, check nothing is running, spawn every service, publish the state."""
        with self.locked():
            prepare()
            running = self.running()
            if running:
                raise RuntimeError(f"{', '.join(sorted(running))} already running for {self.run}; stop first")
            state = {}
            try:
                for name in [name for name in SERVICES if name in plan]:
                    script = Path(next(part for part in plan[name] if part.endswith('.py'))).name
                    pid = self.spawn(plan[name], self.run / 'logs' / name)
                    state[name] = dict(pid=pid, script=script, command=plan[name], started_at=time.time())
                self.publish(state)
            except BaseException:
                survivors = {name: entry for name, entry in state.items() if not self.terminate(entry, timeout)}
                if survivors:
                    self.publish(survivors)
                raise
        return state

    def publish(self, state):
        partial = self.state_file.with_suffix('.json.tmp')
        partial.write_text(json.dumps(state, indent=2), encoding='utf-8')
        os.replace(partial, self.state_file)

    def stop(self, timeout=30.):
        """End every recorded service and wait until each has exited, forcing it after `timeout` seconds on POSIX;
        the records are removed only once all are gone, so a following start never overlaps a stopping service."""
        with self.locked():
            state = self.state()
            for name in reversed(SERVICES):
                entry = state.get(name)
                if entry is None or not self.alive(entry):
                    continue
                if not self.terminate(entry, timeout):
                    raise RuntimeError(f"{name} (pid {entry['pid']}) did not exit; its record is kept")
            self.state_file.unlink(missing_ok=True)
        return sorted(state)

    def terminate(self, entry, timeout):
        """Kill the service and its surviving workers and wait until all are gone, forcing once after `timeout`."""
        for force in (False, True):
            self.kill(entry['pid'], force)
            if os.name == 'nt':   # taskkill's tree option reaches nothing once the leader is gone; POSIX signals the group
                for pid in self.group(entry):
                    self.kill(pid, force)
            if self.gone(entry, timeout):
                return True
        return False

    def gone(self, entry, timeout):
        """True once neither the service nor, on POSIX, any process left in its group exists."""
        deadline = time.time() + timeout
        while self.alive(entry):
            if time.time() > deadline:
                return False
            time.sleep(0.2)
        return True

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


def install(run, source, step, variant='play'):
    """Copy the weights file `source` into `run` as checkpoint `variant/step` and make it the champion, under the
    run's lock so installs never interleave. The `play` variant keeps such installs apart from the learner's own
    exports. Files land under temporary names and are renamed, so a reader never sees a half-written checkpoint."""
    if not Path(source).is_file():
        raise FileNotFoundError(f'no weights file at {source}')
    with Launcher(run).locked():
        return _install(run, source, step, variant)


def _install(run, source, step, variant):
    checkpoint = f'{variant}/{step:06d}'
    target = Path(run) / 'checkpoints' / checkpoint
    target.mkdir(parents=True, exist_ok=True)
    manifest = Path(source).with_name('manifest.json')
    stamp = f'.{os.getpid()}.tmp'
    for name, path in (('ema.pt', Path(source)), ('manifest.json', manifest)):
        if path.exists():
            shutil.copyfile(path, target / (name + stamp))
            os.replace(target / (name + stamp), target / name)
    champion = Path(run) / 'champion.json'
    champion.with_suffix(f'.json{stamp}').write_text(json.dumps(dict(checkpoint=checkpoint)), encoding='utf-8')
    os.replace(champion.with_suffix(f'.json{stamp}'), champion)
    return target


def download(run, fetch=fetch):
    """Fetch the newest released Bubble into `run`. `releases/latest` redirects to the release's tag page, whose
    number is the checkpoint step; the assets are then read from that release. Staging and temporary names are
    unique per process, so concurrent installs never touch each other's files."""
    from urllib.error import HTTPError
    _, tag_page = fetch(RELEASES + 'latest')
    tag = tag_page.rstrip('/').rsplit('/', 1)[-1]
    step = int(re.search(r'(\d+)', tag).group(1))
    import tempfile
    (Path(run) / 'checkpoints').mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='download.', dir=Path(run) / 'checkpoints'))
    (staging / 'ema.pt').write_bytes(fetch(f'{RELEASES}download/{tag}/ema.pt')[0])
    try:
        (staging / 'manifest.json').write_bytes(fetch(f'{RELEASES}download/{tag}/manifest.json')[0])
    except HTTPError:
        pass
    target = install(run, staging / 'ema.pt', step)
    shutil.rmtree(staging)
    return target


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'stop', 'status', 'play'):
        p = sub.add_parser(name)
        p.add_argument('--run', default='runs/play' if name == 'play' else 'runs/bubble',
                       help='the run directory; play defaults to runs/play, where released weights are installed')
    train = sub.choices['train']
    train.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    train.add_argument('--actors', type=positive, default=4, help='self-play processes')
    train.add_argument('--dashboard-port', type=int, default=8766)
    train.add_argument('--seal', action='store_true', help='rate champions against Seal (needs the Seal build)')
    train.add_argument('--net-kernels', choices=['reference', 'fused'], default='reference')
    play = sub.choices['play']
    play.add_argument('--port', type=int, default=8765)
    play.add_argument('--model', type=Path, help="an ema.pt file to play, passed straight to the server")
    play.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.command in ('train', 'stop', 'status') and os.name != 'nt' and not Path('/proc').is_dir():
        sys.exit('bubble.py manages services on Windows and Linux only: it identifies them through /proc')
    launcher = Launcher(args.run)
    if args.command == 'train':
        proof = tactical_built()
        if not proof:
            print('proof pass skipped: build the tactical solver first (python tools/build_tactical.py)')
        plan = commands(launcher.run, args.actors, args.dashboard_port, args.seal, args.net_kernels, proof)
        prepare = lambda: launcher.prepare(args.device, lambda command: subprocess.run(command, cwd=ROOT, check=True),
                                           args.actors)
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
            if not args.model.is_file():
                sys.exit(f'no weights file at {args.model}')
        elif not exports(run):
            if run != (ROOT / 'runs' / 'play').resolve():
                sys.exit(f'no checkpoints under {run}; pass --model, or use the default run to download a release')
            print(f'no checkpoints under {run}; downloading the latest released Bubble')
            print(f'installed {download(run)}')
        model = ['--dense-model', str(args.model.resolve())] if args.model else []
        sys.exit(subprocess.call([sys.executable, str(PYTHON / 'play.py'), '--dense-run', str(run), *model,
                                  '--port', str(args.port), '--device', args.device], cwd=ROOT))


if __name__ == '__main__':
    main()
