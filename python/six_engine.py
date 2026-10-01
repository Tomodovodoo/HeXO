"""Bubble on the Six engine protocol, and a client for Six-compatible engines."""
import argparse
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading

from hexo import Game


class ProtocolError(RuntimeError):
    pass


def serve(player, source=sys.stdin, out=sys.stdout):
    """Run one synchronous protocol session. Search time is advisory."""
    game = Game()
    try:
        for raw in source:
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
                    player.set_history()
                    game.close()
                    game = Game()
                elif command == 'position':
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
                    if game.winner >= 0:
                        raise ValueError('game has finished')
                    options = words[1:]
                    if len(options) % 2 or any(options[i] not in ('movetime', 'nodes', 'depth') or
                                                   int(options[i+1]) < 1 for i in range(0, len(options), 2)):
                        raise ValueError('bad go options')
                    turn = player.turn(game, milliseconds=int(options[options.index('movetime')+1])
                                       if 'movetime' in options else None)['moves']
                    probe = Game([tuple(c[:2]) for c in game.cells])
                    try:
                        played = []
                        for q, r in turn[:probe.remaining]:
                            probe.play(int(q), int(r))
                            played.append((q, r))
                            if probe.winner >= 0:
                                break
                        if probe.winner < 0 and probe.player == game.player:
                            raise ValueError('player did not complete its turn')
                    finally:
                        probe.close()
                    print('bestmove ' + ' '.join(f'{q} {r}' for q, r in played), file=out, flush=True)
                elif command == 'stop':
                    pass  # Search runs synchronously and is already finished before the next line is read.
                elif command == 'quit':
                    return
                else:
                    raise ValueError(f'unknown command {command}')
            except Exception as error:
                print(f'error {error}', file=out, flush=True)
    finally:
        game.close()


class SixEngine:
    """An external Six-protocol opponent with the call shape of legacy.arena.Seal."""

    def __init__(self, command, timeout=30.):
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.timeout = timeout
        self.game = None
        self.proc = None
        self._start()

    def _start(self):
        self.lines = queue.Queue()
        self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        threading.Thread(target=self._read, args=(self.proc, self.lines), daemon=True).start()
        try:
            self._send('six')
            self._expect('sixok', self.timeout)
            self._send('isready')
            self._expect('readyok', self.timeout)
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
            try:
                line = self.lines.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty as error:
                raise ProtocolError(f'engine timed out waiting for {prefix}') from error
            if line is None:
                raise ProtocolError('engine exited')
            if line.startswith('error'):
                raise ProtocolError(line)
            if line.startswith(prefix):
                return line

    def __call__(self, game, ms):
        try:
            if game is not self.game:
                self._send('newgame')
                self._send('isready')
                self._expect('readyok', self.timeout)
                self.game = game
            moves = ' '.join(f'{q} {r}' for q, r, _ in game.cells)
            self._send('position radius 8' + (f' moves {moves}' if moves else ''))
            self._send(f'go movetime {ms}')
            line = self._expect('bestmove', self.timeout + 3*ms/1000)
            parts = line.split()[1:]
            if len(parts) not in (2, 4):
                raise ProtocolError(f'unreadable bestmove: {line}')
            numbers = [int(v) for v in parts]
            turn = list(zip(numbers[::2], numbers[1::2]))
            probe = Game([tuple(c[:2]) for c in game.cells])
            try:
                played = []
                for q, r in turn:
                    probe.play(q, r)
                    played.append((q, r))
                    if probe.winner >= 0:
                        break
                if len(played) != len(turn):
                    raise ProtocolError('engine sent a move after the winning stone')
                if probe.winner < 0 and probe.player == game.player:
                    raise ProtocolError('engine did not complete its turn')
            finally:
                probe.close()
            return turn
        except (ProtocolError, ValueError) as error:
            self._stop()
            self.game = None
            self._start()
            raise ProtocolError(str(error)) from error

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


class BubblePlayer:
    """Bubble behind `serve`: one export searched with a fixed budget."""

    def __init__(self, path, device, simulations, solver_nodes):
        from play import Bubble, Engines
        path = Path(path).resolve()
        self.bubble, self.simulations, self.solver_nodes = Bubble(path, device), simulations, solver_nodes
        self.prover = Engines(device).solver()[0] if solver_nodes else None
        self.model_sha256 = self.bubble.sha256
        in_run = path.name == 'ema.pt' and path.parent.parent.parent.name == 'checkpoints'
        self.checkpoint = f'{path.parent.parent.name}/{path.parent.name}' if in_run else path.stem

    def set_history(self):
        self.bubble.cache.entries.clear()

    def turn(self, game, milliseconds=None):
        from play import evaluate
        return evaluate(self.bubble, self.prover, [c[:2] for c in game.cells], self.simulations, self.solver_nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['serve'])
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument('--run', type=Path, help="serve the run's champion")
    selected.add_argument('--model', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--simulations', type=int, default=128)
    parser.add_argument('--solver-nodes', type=int, default=32768)
    args = parser.parse_args()
    import torch
    from play import run_checkpoints
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = args.model or args.run / 'checkpoints' / run_checkpoints(args.run)[0] / 'ema.pt'
    serve(BubblePlayer(model, device, args.simulations, args.solver_nodes))


if __name__ == '__main__':
    main()
