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


def commands(run, actors=4, dashboard_port=8766, seal=False, kernels=None, proof=True):
    """The command line of every service for `run`, as argument lists starting with the interpreter; the proof pass
    only when `proof`, since it needs the tactical solver build. `kernels` overrides the run's configured network
    kernels only when given."""
    python = [sys.executable, '-u']
    net = ['--net-kernels', kernels] if kernels else []
    graphs = ['--no-cuda-graphs'] if kernels == 'reference' else []   # CUDA graphs exist only for fused kernels
    evaluator = [*python, str(PYTHON / 'dense_eval.py'), 'loop', '--run', str(run), *net]
    if not seal:
        evaluator += ['--eval-anchor-games', '0', '--no-eval-anchor-on-promotion', '--eval-anchor-target-halfwidth', '0']
    plan = dict(
        learner=[*python, str(PYTHON / 'dense_learn.py'), '--run', str(run), *net],
        actors=[*python, str(PYTHON / 'dense_selfplay.py'), '--run', str(run), '--processes', str(actors), *net, *graphs],
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

    def prepare(self, device, run_steps, actors=None, tactical=True):
        """Create the run configuration (on `device`, `auto` resolved then) and the first checkpoint when either is
        missing; `run_steps` runs a command. An existing run must agree with an explicit `device`, its phase
        schedule may not wait for more workers than `actors` starts, and solver budgets in its configuration or
        its registered evaluation variants need the tactical build (`tactical`)."""
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
            queries = ('solver_root_nodes', 'solver_finalist_nodes', 'solver_threat_nodes', 'solver_leaf_nodes',
                       'solver_deep_nodes')   # the budgets that enable queries; caps and floors only bound them
            sections = [config.get(section) or {} for section in ('actor', 'evaluation')]
            try:
                league = json.loads((self.run / 'league.json').read_text(encoding='utf-8'))
                sections += [variant.get('settings') or {} for variant in league.get('variants', [])]
            except (OSError, ValueError, AttributeError):
                pass
            budgets = [section.get(key, 0) for section in sections for key in queries]
            if not tactical and any(budgets):
                raise RuntimeError(f'{self.run} uses solver budgets but the tactical solver is not built; '
                                   'run python tools/build_tactical.py or set the solver_*_nodes settings to 0')
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


def player_request(port, path, body=None):
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    request = Request(f'http://127.0.0.1:{port}{path}', None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=15) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            message = json.load(error).get('error', str(error))
        except (ValueError, AttributeError):
            message = str(error)
        raise RuntimeError(message) from error


def ensure_player(args, start=True):
    """Reuse the selected player, or start an idle server without touching training services."""
    from urllib.error import URLError
    try:
        catalogue = player_request(args.port, '/models')
    except RuntimeError as error:
        raise RuntimeError(f'Port {args.port} does not offer the current player API; choose another port') from error
    except URLError as error:
        if not start or not isinstance(error.reason, ConnectionRefusedError):
            raise
        folder = ROOT / 'artifacts' / 'play'
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / f'server-{args.port}.log'
        command = [sys.executable, '-u', str(PYTHON / 'play.py'), '--idle', '--port', str(args.port),
                   '--device', args.device, '--evaluations', str(folder / f'server-{args.port}-evaluations.jsonl')]
        for option, value in (('--dense-run', args.run), ('--models', args.models), ('--book', args.book)):
            if value is not None:
                command += [option, str(value.resolve())]
        options = (dict(creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS |
                       subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                   if os.name == 'nt' else dict(start_new_session=True))
        with log.open('ab') as output:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                       env=os.environ | dict(OMP_NUM_THREADS='2'), **options)
        (folder / f'server-{args.port}.json').write_text(json.dumps(dict(pid=process.pid, command=command)), encoding='utf-8')
        deadline = time.monotonic() + 30
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Player exited; see {log}')
            try:
                catalogue = player_request(args.port, '/models')
                break
            except URLError:
                if time.monotonic() >= deadline:
                    process.terminate()
                    process.wait(timeout=5)
                    raise RuntimeError(f'Player did not become ready; see {log}')
                time.sleep(.1)
    if catalogue.get('api') != 'bubble-player-v1':
        raise RuntimeError(f'Port {args.port} belongs to another service')
    return catalogue


def player_client(args):
    action = args.players[0] if args.command == 'match' and len(args.players) == 1 else None
    controls = ('status', 'pause', 'resume', 'stop')
    resume = args.command == 'match' and args.resume
    if args.command == 'match' and not resume and action not in controls and len(args.players) != 2:
        raise ValueError('Choose two players, or status, pause, resume or stop')
    catalogue = ensure_player(args, start=bool(resume) or action not in controls)
    if args.command == 'models':
        for model in catalogue['models']:
            print(f"{model['id']}  [{model['kind']}; {'clocks supported' if model['clocks'] else 'fixed budget only'}]")
            if model.get('checkpoints'):
                print('  ' + ', '.join(model['checkpoints']))
        return
    if resume:
        if args.players:
            raise ValueError('--resume loads the saved players; omit player names')
        state = player_request(args.port, '/match', dict(action='resume', batch=str(args.resume.resolve())))
    elif action == 'status':
        state = player_request(args.port, '/match')
    elif action in controls:
        state = player_request(args.port, '/match', dict(action=action))
    else:
        players = [dict(engine=engine, **({'device': device} if device else {}))
                   for engine, device in zip(args.players, (args.a_device, args.b_device))]
        body = dict(players=players, games=args.games, preset=args.preset, opening_range=args.openings,
                    unique_openings=args.unique_openings, seed=args.seed, max_placements=args.max_placements)
        if args.opening:
            body['opening_texts'] = [path.read_text(encoding='utf-8') for path in args.opening]
        if args.tc:
            body['clock'] = dict(mode='game', tc=args.tc)
        elif args.move:
            body['clock'] = dict(mode='move', ms=args.move)
        if args.book:
            body['book'] = str(args.book.resolve())
        if args.out:
            body['output'] = str(args.out.resolve())
        state = player_request(args.port, '/match', body)
    match = state.get('match')
    print(f'Watch: http://127.0.0.1:{args.port}')
    if match:
        print(f"{match['completed']}/{match['games']} games; wins {match['wins']}; capped {match['capped']}")
        print(f"Results: {match['output']}")
        if match.get('error'):
            print(match['error'])
    else:
        print('No batch on this player')


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
    train.add_argument('--net-kernels', choices=['reference', 'fused'], help="override the run's configured kernels")
    play = sub.choices['play']
    play.add_argument('--port', type=int, default=8765)
    play.add_argument('--model', type=Path, help="an ema.pt file to play, passed straight to the server")
    play.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    for command in ('models', 'match'):
        client = sub.add_parser(command, help='inspect or control the live browser player')
        client.add_argument('--port', type=int, default=8765)
        client.add_argument('--run', type=Path, help='run offered by an automatically started player')
        client.add_argument('--models', type=Path, help='engine catalogue directory for an automatically started player')
        client.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
        client.add_argument('--book', type=Path, help='opening book; read only')
    match = sub.choices['match']
    match.add_argument('players', nargs='*', help='two catalogue names, or status/pause/resume/stop')
    match.add_argument('--games', type=positive)
    match.add_argument('--unique-openings', type=positive)
    match.add_argument('--openings', choices=['narrow', 'wide', 'all'])
    match.add_argument('--opening', type=Path, action='append', help='custom HTTTX or replay opening, repeatable')
    match.add_argument('--seed', type=int, default=0)
    match.add_argument('--preset', choices=['lightning', 'quick', 'standard', 'strong', 'deep', 'dangerous'], default='standard')
    from time_control import duration
    clocks = match.add_mutually_exclusive_group()
    clocks.add_argument('--tc', help='shared game clock in seconds+increment, e.g. 180+2')
    clocks.add_argument('--move', type=duration, help='shared time per complete turn, e.g. 5s')
    match.add_argument('--a-device', choices=['cpu', 'cuda'])
    match.add_argument('--b-device', choices=['cpu', 'cuda'])
    match.add_argument('--out', type=Path)
    match.add_argument('--resume', type=Path, help='resume a saved batch directory from its last completed turn')
    match.add_argument('--max-placements', type=positive, default=512)
    args = parser.parse_args()
    if args.command in ('models', 'match'):
        try:
            return player_client(args)
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f'{error}\n')
    if args.command in ('train', 'stop', 'status') and os.name != 'nt' and not Path('/proc').is_dir():
        sys.exit('bubble.py manages services on Windows and Linux only: it identifies them through /proc')
    launcher = Launcher(args.run)
    if args.command == 'train' and launcher.run == (ROOT / 'runs' / 'play').resolve():
        sys.exit('runs/play holds downloaded weights for playing; train in another run directory')
    if args.command == 'train':
        proof = tactical_built()
        if not proof:
            print('proof pass skipped: build the tactical solver first (python tools/build_tactical.py)')
        plan = commands(launcher.run, args.actors, args.dashboard_port, args.seal, args.net_kernels, proof)
        prepare = lambda: launcher.prepare(args.device, lambda command: subprocess.run(command, cwd=ROOT, check=True),
                                           args.actors, proof)
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
