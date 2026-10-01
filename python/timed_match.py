"""Clocked games and a match runner, independent of the fixed-simulation league."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import uuid

from hexo import Game
from notation import Record, dumps, loads
from time_control import Clock, TimeControl, milliseconds
from timed_engine import TimedEngine, legal_turn


class Match:
    def __init__(self, specification, *, directory=None, now=time.monotonic_ns, match_id=None):
        self.id = match_id or uuid.uuid4().hex
        self.specification = specification
        control = TimeControl.parse(specification['time_control'])
        if specification.get('turn_cap_ms') is not None:
            milliseconds(specification['turn_cap_ms'], 'turn_cap_ms', positive=True)
        self.history = [list(p) for p in specification.get('history', [[0, 0]])]
        self.game = Game(self.history)
        if self.game.winner >= 0:
            self.game.close()
            raise ValueError('Opening is already terminal')
        self.clock = Clock(control.json(), now)
        self.state, self.result = 'ready', None
        self.turn_id = self.revision = self.sequence = 0
        self.turn_spent_ns = 0
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thinking = None
        self.created = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        self.directory = Path(directory)/self.id if directory else None
        try:
            self.notation()
        except BaseException:
            self.game.close()
            raise
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._write('spec.json', specification)
        self.record('ready')

    def _write(self, name, body):
        path = self.directory/name
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(body, allow_nan=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    def snapshot(self, at=None):
        with self.lock:
            at = self.clock.now() if at is None else at
            cap = self.specification.get('turn_cap_ms')
            spent = self.turn_spent_ns + (at-self.clock.started if self.clock.running is not None else 0)
            return dict(match_id=self.id, sequence=self.sequence, turn_id=self.turn_id,
                        revision=self.revision, state=self.state, history=self.history.copy(),
                        side=('x', 'o')[self.game.player], remaining=self.game.remaining,
                        **self.clock.json(at), result=self.result, thinking=self.thinking,
                        turn_cap_remaining_ms=None if cap is None else max(0, cap-spent/1_000_000))

    def record(self, event, **fields):
        self.sequence += 1
        at = self.clock.now()
        spent = self.turn_spent_ns + (at-self.clock.started if self.clock.running is not None else 0)
        row = dict(type=event, **self.snapshot(at), balances_ns=self.clock.remaining(at),
                   turn_spent_ns=spent, **fields)
        if self.directory:
            with (self.directory/'events.jsonl').open('a', encoding='utf-8') as output:
                output.write(json.dumps(row, allow_nan=False)+'\n')
            self._write('state.json', row)
        return row

    @classmethod
    def restore(cls, path, *, now=time.monotonic_ns):
        """Reload an unfinished match paused, without starting an engine or a clock."""
        path = Path(path)
        saved = json.loads((path/'state.json').read_text(encoding='utf-8'))
        match = cls.__new__(cls)
        match.id = saved['match_id']
        match.specification = json.loads((path/'spec.json').read_text(encoding='utf-8'))
        match.history = saved['history']
        match.game = Game(match.history)
        match.clock = Clock(match.specification['time_control'], now)
        match.clock.balances = saved['balances_ns']
        match.state = 'finished' if saved['state'] == 'finished' else 'paused'
        match.result = saved['result']
        match.turn_id, match.revision = saved['turn_id']+1, saved['revision']
        match.turn_spent_ns = saved.get('turn_spent_ns', 0)
        match.sequence = saved['sequence']
        match.created = match.specification.get('utcdatetime', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
        match.directory = path
        match.lock, match.cancel = threading.RLock(), threading.Event()
        match.thinking = None
        return match

    def start(self):
        with self.lock:
            if self.state != 'ready':
                raise ValueError('Match is not ready')
            self.state = 'playing'
            self.cancel = threading.Event()
            self.clock.start(self.game.player)
            self.record('start')

    def finish(self, winner, reason):
        self.clock.stop()
        self.cancel.set()
        self.state = 'finished'
        self.result = dict(winner=None if winner is None else ('x', 'o')[winner], reason=reason)
        self.record('finish')
        if self.directory:
            self._write('game.json', dict(specification=self.specification, **self.snapshot()))
            (self.directory/'game.htttx').write_text(self.notation(), encoding='utf-8')

    def tick(self):
        with self.lock:
            if self.state == 'playing' and self.expired():
                self.finish(1-self.game.player, 'time')
            return self.snapshot()

    def expired(self, at=None):
        if self.clock.expired(at):
            return True
        cap = self.specification.get('turn_cap_ms')
        if cap is None or self.clock.running is None:
            return False
        elapsed = (self.clock.now() if at is None else at)-self.clock.started
        return self.turn_spent_ns+elapsed >= cap*1_000_000

    def submit(self, pieces, turn_id, revision, *, partial=False, received=None):
        with self.lock:
            received = self.clock.now() if received is None else received
            if self.state != 'playing' or turn_id != self.turn_id or revision != self.revision:
                raise ValueError('Request no longer matches the active position')
            if self.expired(received):
                self.finish(1-self.game.player, 'time')
                return self.snapshot()
            if partial:
                if len(pieces) != 1:
                    raise ValueError('place accepts exactly one placement')
                if len(pieces[0]) != 2 or any(type(v) is not int for v in pieces[0]):
                    raise ValueError('Placements require two integer coordinates')
            else:
                legal_turn(self.history, pieces)
            side = self.game.player
            before = self.clock.json(received)
            probe = Game(self.history)
            try:
                for q, r in pieces:
                    probe.play(q, r)
            finally:
                probe.close()
            for q, r in pieces:
                self.game.play(q, r)
                self.history.append([q, r])
                self.revision += 1
            completed = self.game.player != side or self.game.winner >= 0
            elapsed = self.clock.stop(completed=True, at=received) if completed else None
            self.record('turn' if completed else 'placement', pieces=pieces, elapsed_ns=elapsed,
                        clock_before=before)
            if self.game.winner >= 0:
                self.finish(self.game.winner, 'win')
            elif completed:
                self.turn_spent_ns = 0
                self.turn_id += 1
                self.cancel = threading.Event()
                self.thinking = None
                self.clock.start(self.game.player)
                self.record('clock')
            return self.snapshot()

    def pause(self):
        with self.lock:
            self.tick()
            if self.state != 'playing':
                raise ValueError('Match is not playing')
            elapsed = self.clock.stop()
            self.turn_spent_ns += elapsed
            self.cancel.set()
            self.turn_id += 1
            self.state = 'paused'
            self.record('pause', elapsed_ns=elapsed)
            return self.snapshot()

    def resume(self):
        with self.lock:
            if self.state != 'paused':
                raise ValueError('Match is not paused')
            self.cancel = threading.Event()
            self.turn_id += 1
            self.state = 'playing'
            self.clock.start(self.game.player)
            self.record('resume')
            return self.snapshot()

    def resign(self, side):
        with self.lock:
            self.tick()
            if self.state not in ('playing', 'paused'):
                raise ValueError('Match cannot be resigned')
            self.finish(1-side, 'resign')
            return self.snapshot()

    def notation(self):
        metadata = dict(version='1', name=self.specification.get('name', 'Bubble timed match'),
                        platform='Bubble', utcdatetime=self.created,
                        timecontrol=self.clock.control.text())
        for key, side in [('playercross', 'cross'), ('playercircle', 'circle')]:
            player = self.specification['players'][side]
            metadata[key] = str(player.get('checkpoint', player['kind']))
        if self.result:
            if self.result['reason'] in ('win', 'time', 'resign', 'draw'):
                metadata['endreason'] = self.result['reason']
            if self.result['winner']:
                metadata['winner'] = 'cross' if self.result['winner'] == 'x' else 'circle'
        return dumps(Record(list(map(tuple, self.history)), metadata, None))

    def close(self):
        self.cancel.set()
        self.game.close()


def play_turn(match, engine):
    state = match.tick()
    if state['state'] != 'playing':
        return state
    game = Game(state['history'])
    cancellation = match.cancel
    try:
        def publish(result):
            with match.lock:
                if match.turn_id == state['turn_id'] and match.state == 'playing' and not cancellation.is_set():
                    match.thinking = result
        result = engine.turn(game, clock=state, cancel=cancellation, publish=publish,
                             milliseconds=state['turn_cap_remaining_ms'])
        received = match.clock.now()
        with match.lock:
            if cancellation.is_set() or match.state != 'playing':
                return match.snapshot()
            try:
                return match.submit(result['moves'], state['turn_id'], state['revision'], received=received)
            except ValueError:
                match.finish(1-game.player, 'illegal')
                return match.snapshot()
    except Exception as error:
        with match.lock:
            if match.state == 'playing' and not cancellation.is_set():
                match.tick()
                if match.state == 'playing':
                    match.finish(1-game.player, 'crash')
                    match.record('engine_error', error=str(error))
            return match.snapshot()
    finally:
        game.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/dense-v1'))
    parser.add_argument('--a', required=True)
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument('--b', help='Checkpoint id, or native')
    opponent.add_argument('--b-command', help='External Six-protocol command')
    opponent.add_argument('--b-url', help='External HTTTX HTTP API root')
    parser.add_argument('--tc', default='180+2')
    parser.add_argument('--pairs', type=int, default=1)
    parser.add_argument('--opening', type=Path, help='HTTTX initial position')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--net-kernels', choices=['fused', 'reference'], default='fused')
    parser.add_argument('--solver-nodes', type=int, default=32768)
    parser.add_argument('--concurrency', type=int, choices=[1], default=1)
    parser.add_argument('--max-placements', type=int, default=512)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('--pairs must be positive')
    control = TimeControl.parse(args.tc)
    history = loads(args.opening.read_text(encoding='utf-8')).history if args.opening else [[0, 0]]
    def config(checkpoint):
        if checkpoint == 'native':
            return dict(kind='native')
        return dict(kind='bubble', run=str(args.run.resolve()), checkpoint=checkpoint,
                    device=args.device, net_kernels=args.net_kernels,
                    solver=dict(enabled=args.solver_nodes > 0, nodes=max(1, args.solver_nodes)))
    opponent = (dict(kind='six', command=args.b_command) if args.b_command else
                dict(kind='htttx', url=args.b_url) if args.b_url else config(args.b))
    specs = [config(args.a), opponent]
    results = []
    with TimedEngine(specs[0]) as a, TimedEngine(specs[1]) as b:
        for pair in range(args.pairs):
            for swapped in (False, True):
                engines = [b, a] if swapped else [a, b]
                players = specs[::-1] if swapped else specs
                match = Match(dict(players=dict(cross=players[0], circle=players[1]), history=history,
                                   time_control=control.json(), pair=pair,
                                   identities=[engine.identity for engine in engines]), directory=args.out)
                try:
                    match.start()
                    while match.state == 'playing':
                        play_turn(match, engines[match.game.player])
                        if match.state == 'playing' and len(match.history) >= args.max_placements:
                            match.finish(None, 'capped')
                    result = dict(match_id=match.id, swapped=swapped, **match.result)
                    results.append(result)
                    print(json.dumps(result), flush=True)
                finally:
                    match.close()
    (args.out/'summary.json').write_text(json.dumps(dict(time_control=control.json(), games=results), indent=2),
                                        encoding='utf-8')


if __name__ == '__main__':
    main()
