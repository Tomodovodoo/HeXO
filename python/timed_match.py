"""Clocked games and a match runner, independent of the fixed-simulation league."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
        self.receipt = None
        self.thinking = None
        self.created = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        self.specification.setdefault('utcdatetime', self.created)
        self.created = self.specification['utcdatetime']
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
        match.receipt = None
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
        self.turn_spent_ns += self.clock.stop()
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
                receipt = self.receipt
                # An on-time reply may be waiting to acquire this lock for submission.
                if not (receipt and receipt[:2] == (self.turn_id, self.revision)
                        and receipt[2][0] is not None and not self.expired(receipt[2][0])):
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
            if completed:
                self.turn_spent_ns += elapsed
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
                        platform='Bubble', utcdatetime=self.created)
        control = self.clock.control
        if control.base_ms % 1000 == 0 and control.increment_ms % 1000 == 0:
            metadata['timecontrol'] = control.text()
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


def play_turn(match, engine, *, wait_worker=False, record_engine=False):
    with match.lock:
        state = match.tick()
        if state['state'] != 'playing':
            return state
        cancellation = match.cancel
        receipt = (state['turn_id'], state['revision'], [None])
        match.receipt = receipt
    game = Game(state['history'])
    worker_error = False
    try:
        def publish(result):
            with match.lock:
                if match.turn_id == state['turn_id'] and match.state == 'playing' and not cancellation.is_set():
                    match.thinking = result
        started = match.clock.now()
        budget = match.snapshot(started)
        result = engine.turn(game, clock=None if match.specification.get('fixed_turn_time') else budget,
                             cancel=cancellation, publish=publish,
                             milliseconds=budget['turn_cap_remaining_ms'])
        received = receipt[2][0] = match.clock.now()
        try:
            worker_wait_ms = engine.wait_idle() if wait_worker else 0
        except Exception as error:
            worker_error = True
            raise RuntimeError('Worker remained active after its allowance') from error
        with match.lock:
            if cancellation.is_set() or match.state != 'playing':
                return match.snapshot()
            if record_engine:
                match.record('engine_reply', received_ns=received, controller_ns=received-started,
                             worker_wait_ms=worker_wait_ms, engine=result)
            try:
                return match.submit(result['moves'], state['turn_id'], state['revision'], received=received)
            except ValueError:
                match.finish(1-game.player, 'illegal')
                return match.snapshot()
    except Exception as error:
        with match.lock:
            if match.state == 'playing' and not cancellation.is_set():
                if not worker_error:
                    match.tick()
                if match.state == 'playing':
                    reason = 'engine_timeout' if isinstance(error, TimeoutError) else 'illegal' if isinstance(error, ValueError) else 'crash'
                    match.finish(1-game.player, reason)
                    match.record('engine_error', error=str(error))
            return match.snapshot()
    finally:
        with match.lock:
            if match.receipt is receipt:
                match.receipt = None
        game.close()


def side_settings(path):
    if path is None:
        return {}
    settings = json.loads(path.read_text(encoding='utf-8'))
    allowed = dict(search={'enabled', 'simulations', 'max_simulations', 'root_samples', 'q_range_floor'},
                   solver={'enabled', 'nodes', 'leaf'})
    if not isinstance(settings, dict) or settings.keys()-allowed.keys():
        raise ValueError('Side settings contain only search and solver objects')
    for group, options in settings.items():
        if not isinstance(options, dict) or options.keys()-allowed[group]:
            raise ValueError(f'Unsupported {group} settings: {options}')
    return settings


def paired_openings(run, pairs, seed, *, suite=None, opening=None):
    """Weighted frozen-book draws without replacement; each draw is used for both colors."""
    if suite is None:
        history = loads(opening.read_text(encoding='utf-8')).history if opening else [[0, 0]]
        return [dict(seed=seed+i, history=[list(p) for p in history]) for i in range(pairs)]
    import numpy as np
    from dense_openings import Book, LIVE
    from hexcrop import SYMMETRIES
    if suite == LIVE:
        raise ValueError('Clocked comparisons need a frozen opening suite')
    book = Book(run, suite=suite)
    nodes = [node for node in book.openings() if node['weight'] > 0]
    if pairs > len(nodes):
        raise ValueError(f'{suite} has {len(nodes)} distinct positive-weight openings, fewer than {pairs} pairs')
    weights = np.array([n['weight'] for n in nodes], float)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(nodes), size=pairs, replace=False, p=weights/weights.sum())
    digest = hashlib.sha256(json.dumps(book.data, sort_keys=True).encode()).hexdigest()
    out = []
    for i, index in enumerate(selected):
        node = nodes[index]
        history = np.asarray(node['moves'], np.int64) @ SYMMETRIES[rng.integers(len(SYMMETRIES))]
        out.append(dict(seed=seed+i, key=node['key'], history=history.tolist(), suite=suite, book_sha256=digest))
    return out


def comparison_summary(results, use_sprt=False):
    from dense_stats import tally, sprt
    valid = all(g['reason'] in ('win', 'time', 'resign', 'draw', 'capped', 'engine_timeout') for g in results)
    records = [dict(seed=g['seed'], challenger_color=int(g['swapped']),
                    winner=-1 if g['winner'] is None else int(g['winner'] == 'o')) for g in results]
    stats = tally(records, (lambda rows: sprt(rows, 0, 30, .05, .05)) if use_sprt and valid else None)
    stats['valid'] = valid
    stats['decision'] = None
    if stats['pair_interval']:
        stats['elo_interval'] = [None if p in (0, 1) else 400*math.log10(p/(1-p))
                                 for p in stats['pair_interval']]
    if not valid:
        for key in ('pair_score', 'pair_interval', 'elo_delta', 'elo_interval', 'llr', 'bound_lower', 'bound_upper'):
            stats[key] = None
    elif use_sprt and stats['pairs']*2 == len(results) and results:
        stats['decision'] = sprt(records, 0, 30, .05, .05)['decision']
    return stats


def timing_summary(replies):
    import numpy as np
    out = []
    for side in replies:
        times = [r['controller_ns']/1_000_000 for r in side]
        out.append(dict(turns=len(times), mean_ms=float(np.mean(times)) if times else None,
                        p50_ms=float(np.median(times)) if times else None,
                        p95_ms=float(np.percentile(times, 95)) if times else None,
                        worker_wait_ms=sum(r['worker_wait_ms'] for r in side),
                        reported_evaluations=sum(r['engine'].get('evaluated', 0) for r in side),
                        reported_completed_visits=sum(r['engine'].get('completed', 0) for r in side)))
    return dict(a=out[0], b=out[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/dense-v1'))
    parser.add_argument('--a', required=True)
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument('--b', help='Checkpoint id, or native')
    opponent.add_argument('--b-command', help='External Six-protocol command')
    opponent.add_argument('--b-url', help='External HTTTX HTTP API root')
    parser.add_argument('--tc', default='180+2')
    parser.add_argument('--turn-ms', type=float, help='Equal complete-turn allowance, including both stones')
    parser.add_argument('--pairs', type=int, default=1)
    openings = parser.add_mutually_exclusive_group()
    openings.add_argument('--opening', type=Path, help='HTTTX initial position')
    openings.add_argument('--suite', help='Unique frozen-book starts, such as standard-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--a-settings', type=Path, help='JSON search/solver overrides for Bubble A')
    parser.add_argument('--b-settings', type=Path, help='JSON search/solver overrides for Bubble B')
    parser.add_argument('--sprt', action='store_true', help='Stop at a completed pair on SPRT 0 vs +30 Elo')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--net-kernels', choices=['fused', 'reference'], default='fused')
    parser.add_argument('--solver-nodes', type=int, default=32768)
    parser.add_argument('--concurrency', type=int, choices=[1], default=1)
    parser.add_argument('--max-placements', type=int, default=512)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.pairs < 1:
        parser.error('--pairs must be positive')
    if args.max_placements < 2:
        parser.error('--max-placements must be at least two')
    if args.turn_ms is not None:
        milliseconds(args.turn_ms, '--turn-ms', positive=True)
    control = (TimeControl(args.turn_ms*(args.max_placements+1)) if args.turn_ms is not None else
               TimeControl.parse(args.tc))
    starts = paired_openings(args.run, args.pairs, args.seed, suite=args.suite, opening=args.opening)
    overrides = [side_settings(args.a_settings), side_settings(args.b_settings)]
    checkpoints = {}
    def config(checkpoint):
        if checkpoint == 'native':
            return dict(kind='native')
        if checkpoint not in checkpoints:
            if checkpoint == 'champion':
                resolved = json.loads((args.run/'champion.json').read_text(encoding='utf-8'))['checkpoint']
            elif checkpoint == 'newest':
                resolved = max((p.parent for p in (args.run/'checkpoints').glob('*/*/ema.pt')),
                               key=lambda p: int(p.name)).relative_to(args.run/'checkpoints').as_posix()
            else:
                resolved = checkpoint
            checkpoints[checkpoint] = resolved
        checkpoint = checkpoints[checkpoint]
        return dict(kind='bubble', run=str(args.run.resolve()), checkpoint=checkpoint,
                    model=str((args.run/'checkpoints'/checkpoint/'ema.pt').resolve()),
                    device=args.device, net_kernels=args.net_kernels,
                    solver=dict(enabled=args.solver_nodes > 0, nodes=max(1, args.solver_nodes)))
    opponent = (dict(kind='six', command=args.b_command) if args.b_command else
                dict(kind='htttx', url=args.b_url) if args.b_url else config(args.b))
    specs = [config(args.a), opponent]
    for specification, settings in zip(specs, overrides):
        if settings and specification['kind'] != 'bubble':
            parser.error('Side settings apply to Bubble checkpoints')
        for group, options in settings.items():
            specification[group] = specification.get(group, {}) | options
    results = []
    replies = [[], []]
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'summary.json').exists():
        parser.error('Output already holds a match summary; choose a new directory')
    from legacy.train import write_json
    with TimedEngine(specs[0]) as a, TimedEngine(specs[1]) as b:
        write_json(args.out/'summary.json', dict(time_control=control.json(), turn_ms=args.turn_ms,
                   identities=[a.identity, b.identity], starts=starts, games=[],
                   comparison=comparison_summary([], args.sprt), timing=timing_summary(replies)))
        for pair, opening in enumerate(starts):
            for swapped in (False, True):
                engines = [b, a] if swapped else [a, b]
                players = specs[::-1] if swapped else specs
                for engine in engines:
                    engine.reset(opening['history'])
                match = Match(dict(players=dict(cross=players[0], circle=players[1]), history=opening['history'],
                                   time_control=control.json(), turn_cap_ms=args.turn_ms,
                                   fixed_turn_time=args.turn_ms is not None, pair=pair, opening=opening,
                                   identities=[engine.identity for engine in engines]), directory=args.out)
                try:
                    match.start()
                    while match.state == 'playing':
                        play_turn(match, engines[match.game.player], wait_worker=True, record_engine=True)
                        if match.state == 'playing' and len(match.history) >= args.max_placements:
                            match.finish(None, 'capped')
                    result = dict(match_id=match.id, swapped=swapped, seed=opening['seed'], pair=pair,
                                  opening=opening, **match.result)
                    results.append(result)
                    for line in (match.directory/'events.jsonl').read_text(encoding='utf-8').splitlines():
                        event = json.loads(line)
                        if event['type'] == 'engine_reply':
                            index = int(event['side'] == 'o') ^ int(swapped)
                            replies[index].append(event)
                    stats = comparison_summary(results, args.sprt)
                    write_json(args.out/'summary.json', dict(time_control=control.json(), turn_ms=args.turn_ms,
                               identities=[a.identity, b.identity], starts=starts, games=results,
                               comparison=stats, timing=timing_summary(replies)))
                    print(json.dumps(result), flush=True)
                finally:
                    match.close()
                if not stats['valid']:
                    raise RuntimeError('Comparison contains an engine failure; inspect the saved game')
            if stats['decision']:
                break


if __name__ == '__main__':
    main()
