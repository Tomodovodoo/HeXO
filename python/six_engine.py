"""Bubble on the Six engine protocol, and a client for Six-compatible engines."""
import argparse
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time

from hexo import Game


class ProtocolError(RuntimeError):
    pass


class ProtocolTimeout(ProtocolError, TimeoutError):
    pass


class IllegalReply(ProtocolError, ValueError):
    pass


def serve(player, source=sys.stdin, out=sys.stdout):
    """Read commands while thinking; stop returns the latest complete legal turn."""
    from timed_engine import TimedEngine, legal_turn
    from time_control import allowance
    lines, replies = queue.Queue(), queue.Queue()
    def read():
        for raw in source:
            lines.put(raw)
        lines.put(None)
    threading.Thread(target=read, daemon=True).start()
    game = Game()
    active = None
    generation = 0
    def finish():
        nonlocal active
        if active:
            active['cancel'].set()
            print('bestmove ' + ' '.join(f'{q} {r}' for q, r in active['best']), file=out, flush=True)
            active = None
    def calculate(job, milliseconds, clock):
        local = Game(job['history'])
        try:
            if isinstance(player, TimedEngine):
                result = player.turn(local, milliseconds, clock=clock, cancel=job['cancel'],
                                     publish=lambda result: job.update(best=result['moves']))
            else:
                result = player.turn(local, milliseconds=milliseconds)
            played = []
            side = local.player
            for q, r in result['moves'][:local.remaining]:
                local.play(int(q), int(r))
                played.append([int(q), int(r)])
                if local.winner >= 0:
                    break
            if local.winner < 0 and local.player == side:
                raise ValueError('player did not complete its turn')
            replies.put((job['id'], played, None))
        except Exception as error:
            replies.put((job['id'], None, str(error)))
        finally:
            local.close()
            job['done'].set()
    try:
        while True:
            while not replies.empty():
                ident, moves, error = replies.get()
                if active and ident == active['id']:
                    if error:
                        print(f'error {error}', file=out, flush=True)
                        active = None
                    else:
                        active['best'] = moves
                        finish()
            if active and time.monotonic() >= active['deadline']:
                finish()
            try:
                raw = lines.get(timeout=.005)
            except queue.Empty:
                continue
            if raw is None:
                if active:
                    active['done'].wait(min(.005, max(0, active['deadline']-time.monotonic())))
                    while not replies.empty():
                        ident, moves, error = replies.get()
                        if active and ident == active['id'] and not error:
                            active['best'] = moves
                    finish()
                return
            words = raw.strip().split()
            if not words:
                continue
            command = words[0]
            try:
                if command == 'six':
                    print(f'id name Bubble {player.checkpoint}', file=out)
                    print(f'id version {player.model_sha256[:12]}', file=out)
                    print('sixok', file=out, flush=True)
                elif command == 'isready':
                    print('readyok', file=out, flush=True)
                elif command == 'newgame':
                    finish()
                    player.set_history()
                    game.close()
                    game = Game()
                elif command == 'position':
                    finish()
                    if len(words) < 3 or words[1] != 'radius' or int(words[2]) != 8:
                        raise ValueError('Bubble supports radius 8')
                    if len(words) > 3 and words[3] != 'moves':
                        raise ValueError('setup and tomove are unsupported')
                    numbers = [int(v) for v in words[4:]]
                    if len(numbers) % 2:
                        raise ValueError('moves need coordinate pairs')
                    next_game = Game()
                    try:
                        for i in range(0, len(numbers), 2):
                            next_game.play(numbers[i], numbers[i + 1])
                    except Exception:
                        next_game.close()
                        raise
                    game.close()
                    game = next_game
                elif command == 'go':
                    if active:
                        raise ValueError('Search already running')
                    if game.winner >= 0:
                        raise ValueError('game has finished')
                    options = words[1:]
                    allowed = ('movetime', 'nodes', 'depth', 'xtime', 'otime', 'xinc', 'oinc',
                               'wtime', 'btime', 'winc', 'binc')
                    if len(options) % 2 or any(options[i] not in allowed or
                                                   int(options[i+1]) < 0 for i in range(0, len(options), 2)):
                        raise ValueError('bad go options')
                    values = dict(zip(options[::2], map(int, options[1::2])))
                    milliseconds = values.get('movetime')
                    cross, circle = values.get('xtime', values.get('wtime')), values.get('otime', values.get('btime'))
                    clock = None
                    if cross is not None or circle is not None:
                        if cross is None or circle is None:
                            raise ValueError('Both clocks are required')
                        increment = values.get('xinc', values.get('winc', 0)) if game.player == 0 else values.get('oinc', values.get('binc', 0))
                        clock = dict(cross_ms=cross, circle_ms=circle, increment_ms=increment)
                    generation += 1
                    history = [list(c[:2]) for c in game.cells]
                    budget = allowance(clock, game.player, milliseconds)['hard_ms']
                    active = dict(id=generation, history=history, best=legal_turn(history),
                                  cancel=threading.Event(), done=threading.Event(),
                                  deadline=time.monotonic()+budget/1000)
                    threading.Thread(target=calculate, args=(active, milliseconds, clock), daemon=True).start()
                    active['done'].wait(.001)
                elif command == 'stop':
                    finish()
                elif command == 'quit':
                    finish()
                    return
                else:
                    raise ValueError(f'unknown command {command}')
            except Exception as error:
                print(f'error {error}', file=out, flush=True)
    finally:
        if active:
            active['cancel'].set()
        game.close()


def mirror(q, r):
    """Six's and Strix's axial frame from ours and back: HTTTX (q, r) is their (q + r, -r); its own inverse."""
    return q + r, -r


class SixEngine:
    """An external Six-protocol opponent with the call shape of legacy.arena.Seal.

    A `mirrored` engine uses Six's frame, so positions and moves pass through `mirror`. `path` directories go in
    front of PATH for the engine process, for the libraries of its GPU backend; `cwd` is its working folder."""

    def __init__(self, command, timeout=30., *, cancel=None, mirrored=False, cwd=None, path=()):
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.timeout = timeout
        self.cancel = cancel
        self.mirrored, self.cwd = mirrored, cwd
        self.env = {**os.environ, 'PATH': os.pathsep.join([*map(str, path), os.environ.get('PATH', '')])} \
            if path else None
        self.game = None
        self.proc = None
        self._start()

    def _start(self):
        self.lines = queue.Queue()
        self.proc = subprocess.Popen(self.command, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                                     bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        threading.Thread(target=self._read, args=(self.proc, self.lines), daemon=True).start()
        try:
            self._send('six')
            self._expect('sixok', max(5., self.timeout))
            self._send('isready')
            self._expect('readyok', max(5., self.timeout))
        except Exception:
            self._stop()
            raise

    @staticmethod
    def _read(proc, lines):
        for line in proc.stdout:
            lines.put(line.rstrip('\r\n'))
        lines.put(None)

    def _send(self, command):
        try:
            self.proc.stdin.write(command + '\n')
            self.proc.stdin.flush()
        except (OSError, ValueError) as error:
            raise ProtocolError(f'engine stopped accepting commands: {error}') from error

    def _expect(self, prefix, timeout):
        import time
        deadline = time.monotonic() + timeout
        while True:
            if self.cancel is not None and self.cancel.is_set():
                raise ProtocolError('engine request cancelled')
            try:
                wait = max(0, deadline-time.monotonic())
                line = self.lines.get(timeout=min(.01, wait) if self.cancel is not None else wait)
            except queue.Empty as error:
                if time.monotonic() < deadline:
                    continue
                raise ProtocolTimeout(f'engine timed out waiting for {prefix}') from error
            if line is None:
                raise ProtocolError('engine exited')
            if line.startswith('error'):
                raise ProtocolError(line)
            if line.startswith(prefix):
                return line

    def __call__(self, game, ms=None, nodes=None):
        """The engine's turn for `game`, searched for `nodes` nodes when given, else for `ms` milliseconds."""
        frame = mirror if self.mirrored else (lambda q, r: (q, r))
        try:
            if self.proc is None:
                self._start()
            if game is not self.game:
                self._send('newgame')
                self._send('isready')
                self._expect('readyok', self.timeout)
                self.game = game
            moves = ' '.join('%d %d' % frame(q, r) for q, r, _ in game.cells)
            self._send('position radius 8' + (f' moves {moves}' if moves else ''))
            self._send(f'go nodes {nodes}' if nodes else f'go movetime {ms}')
            line = self._expect('bestmove', self.timeout + (nodes / 1000 if nodes else 3*ms/1000))
            parts = line.split()[1:]
            if len(parts) not in (2, 4):
                raise IllegalReply(f'unreadable bestmove: {line}')
            numbers = [int(v) for v in parts]
            turn = [frame(q, r) for q, r in zip(numbers[::2], numbers[1::2])]
            probe = Game([tuple(c[:2]) for c in game.cells])
            try:
                played = []
                for q, r in turn:
                    probe.play(q, r)
                    played.append((q, r))
                    if probe.winner >= 0:
                        break
                if len(played) != len(turn):
                    raise IllegalReply('engine sent a move after the winning stone')
                if probe.winner < 0 and probe.player == game.player:
                    raise IllegalReply('engine did not complete its turn')
            finally:
                probe.close()
            return turn
        except (ProtocolError, ValueError) as error:
            self._stop()
            self.game = None
            if self.cancel is None or not self.cancel.is_set():
                self._start()
            error_class = IllegalReply if isinstance(error, ValueError) else type(error)
            raise error_class(str(error)) from error

    def _stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.proc = None

    def close(self):
        if self.proc is not None:
            try:
                self._send('quit')
                self.proc.wait(timeout=1)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['serve'])
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument('--run', type=Path)
    selected.add_argument('--model', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--simulations', type=int, help='Optional simulation cap for each timed turn')
    parser.add_argument('--solver-nodes', type=int, default=32768)
    parser.add_argument('--net-kernels', choices=['fused', 'reference'], default='fused')
    args = parser.parse_args()
    if args.simulations is not None and args.simulations <= 0:
        parser.error('--simulations must be positive')
    import torch
    from timed_engine import TimedEngine
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    config = dict(kind='bubble', run=str((args.run or Path('.')).resolve()), device=device,
                  search=dict(max_simulations=args.simulations), net_kernels=args.net_kernels,
                  solver=dict(enabled=args.solver_nodes > 0, nodes=max(1, args.solver_nodes)))
    if args.model:
        config['model'] = str(args.model.resolve())
    player = TimedEngine(config)
    try:
        serve(player)
    finally:
        player.close()


if __name__ == '__main__':
    main()
