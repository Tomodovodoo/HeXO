"""Start, stop and inspect a Bubble training run, or play against it.

`train` creates the run directory and its first checkpoint when they are missing, then starts the learner,
the actors, the evaluator, the proof pass and the dashboard as detached processes. Their process ids go to
`<run>/processes.json`, their output to `<run>/logs/`. `stop` ends those processes, `status` reports them,
and `play` serves the browser game against the run's champion.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'python'
SERVICES = ('learner', 'actors', 'evaluator', 'proof', 'dashboard')


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


def alive(pid):
    if os.name == 'nt':
        out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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

    def __init__(self, run, spawn=spawn, alive=alive, kill=kill):
        self.run, self.spawn, self.alive, self.kill = Path(run), spawn, alive, kill
        self.state_file = self.run / 'processes.json'

    def state(self):
        return json.loads(self.state_file.read_text(encoding='utf-8')) if self.state_file.exists() else {}

    def running(self):
        return {name: entry for name, entry in self.state().items() if self.alive(entry['pid'])}

    def prepare(self, device, run_steps):
        """Create the run configuration and the first checkpoint when either is missing; `run_steps` runs a command."""
        if not (self.run / 'config.json').exists():
            run_steps([sys.executable, str(PYTHON / 'dense_config.py'), '--run', str(self.run), '--device', device])
        if not (self.run / 'checkpoints' / 'main').exists():
            run_steps([sys.executable, str(PYTHON / 'dense_learn.py'), '--run', str(self.run), '--steps', '0'])

    def start(self, plan):
        running = self.running()
        if running:
            raise RuntimeError(f"{', '.join(sorted(running))} already running for {self.run}; stop first")
        state = {}
        for name in SERVICES:
            pid = self.spawn(plan[name], self.run / 'logs' / name)
            state[name] = dict(pid=pid, command=plan[name], started_at=time.time())
        self.state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
        return state

    def stop(self):
        state = self.state()
        for name in reversed(SERVICES):
            if name in state and self.alive(state[name]['pid']):
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
            life = 'alive' if self.alive(entry['pid']) else 'gone'
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
    args = parser.parse_args()
    launcher = Launcher(args.run)
    if args.command == 'train':
        launcher.prepare(resolve_device(args.device), lambda command: subprocess.run(command, cwd=ROOT, check=True))
        plan = commands(launcher.run, args.actors, args.dashboard_port, args.seal, args.net_kernels)
        for name, entry in launcher.start(plan).items():
            print(f'{name:<10} pid {entry["pid"]}')
        print(f'dashboard  http://127.0.0.1:{args.dashboard_port}  logs {launcher.run / "logs"}')
    elif args.command == 'stop':
        stopped = launcher.stop()
        print('stopped ' + ', '.join(stopped) if stopped else 'nothing running')
    elif args.command == 'status':
        print('\n'.join(launcher.status()))
    else:
        sys.exit(subprocess.call([sys.executable, str(PYTHON / 'play.py'), '--dense-run', str(launcher.run),
                                  '--port', str(args.port)], cwd=ROOT))


if __name__ == '__main__':
    main()
