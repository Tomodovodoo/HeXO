"""Persistent engine worker and a responsive controller for complete timed turns."""
import multiprocessing as mp
from pathlib import Path
import threading
import time
import json
import hashlib
import subprocess
from urllib.request import Request, urlopen
from urllib.error import URLError

from hexo import Game, library
from time_control import allowance


class HTTTXEngine:
    """An HTTP opponent using the published per-turn allowance, in seconds."""
    def __init__(self, url):
        self.url = url.rstrip('/')
        with urlopen(self.url+'/capabilities.json', timeout=30) as response:
            self.capabilities = json.load(response)
        versions = self.capabilities.get('stateless', {}).get('versions', {})
        if 'v1-alpha' not in versions:
            raise ValueError('Opponent does not offer the stateless HTTTX API')
        self.root = versions['v1-alpha'].get('api_root', 'stateless/v1-alpha')
        self.request_id = 0

    def turn(self, game, ms):
        from bot_api import _coord
        self.request_id += 1
        body = dict(board=dict(to_move=('x', 'o')[game.player],
                               cells=[dict(q=q, r=r, p=('x', 'o')[side]) for q, r, side in game.cells]),
                    time_limit=ms/1000, request_id=self.request_id)
        request = Request(self.url+'/'+self.root.strip('/')+'/turn', data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=max(.001, ms/1000)) as response:
                result = json.loads(response.read(1_048_577))
        except URLError as error:
            if isinstance(error.reason, TimeoutError) or getattr(error, 'code', None) == 408:
                raise TimeoutError(str(error)) from error
            raise
        if self.capabilities['stateless']['versions']['v1-alpha'].get('request_id') and result.get('request_id') != self.request_id:
            raise ValueError('Opponent returned a different request_id')
        return [list(_coord(cell)) for cell in result['move']['pieces']]

    def close(self):
        pass


def legal_turn(history, moves=None):
    """Validate a complete turn, or construct one without inference."""
    game = Game(history)
    side, chosen = game.player, []
    try:
        if game.winner >= 0:
            raise ValueError('Game has finished')
        if moves is None:
            while game.player == side and game.winner < 0:
                action = game.legal_moves()[0]
                game.play(*action)
                chosen.append(list(action))
        else:
            for action in moves:
                if len(action) != 2 or any(type(v) is not int for v in action):
                    raise ValueError('Placements require two integer coordinates')
                if game.player != side or game.winner >= 0:
                    raise ValueError('Placement after turn completion')
                game.play(*action)
                chosen.append(list(action))
            if game.winner < 0 and game.player == side:
                raise ValueError('Incomplete turn')
        return chosen
    finally:
        game.close()


def complete_candidate(history, prefix):
    game = Game(history)
    side, moves = game.player, []
    try:
        for action in prefix:
            if game.winner >= 0 or game.player != side:
                break
            game.play(*action)
            moves.append(list(action))
        while game.winner < 0 and game.player == side:
            action = game.legal_moves()[0]
            game.play(*action)
            moves.append(list(action))
        return moves
    finally:
        game.close()


def dense_turn(player, history, limits, cancel, publish=lambda result: None, analyze=False):
    """One allowance for proofs and both stones; publish only complete legal turns."""
    if getattr(player, 'native_scheduler', False) and player.options['search']:
        return native_turn(player, history, limits, cancel, publish, analyze)
    import numpy as np
    from neural_search import NeuralSearch
    from dense_selfplay import root_value
    started = time.monotonic()
    hard = started + max(0, limits['hard_ms']-limits['reserve_ms'])/1000
    normal = min(hard, started+limits['normal_ms']/1000)
    game = Game(history)
    side, remaining = game.player, game.remaining
    result = dict(moves=legal_turn(history), backend='dense', checkpoint=player.checkpoint,
                  player=side, win_probability=None, suggestions=[], winning_line=[], threat=None,
                  proof_status='UNKNOWN', solver_status='off',
                  settings=dict(player.options) | dict(simulations=limits.get('simulations')),
                  completed=0, evaluated=0, solver_nodes=0)
    tree = None
    solver_left = [limits['normal_ms']*.25]
    class ProofBudget:
        def history(self, position, ms, **kwargs):
            budget = max(0, min(ms, solver_left[0], (hard-time.monotonic())*1000))
            if budget < 1 or cancel.is_set():
                return dict(status='UNKNOWN')
            t0 = time.monotonic()
            answer = player.prover.history(position, ms=int(budget), nodes=player.options['solver_nodes'], **kwargs)
            solver_left[0] -= (time.monotonic()-t0)*1000
            return answer
    proof_budget = ProofBudget()
    def stopped():
        return cancel.is_set() or time.monotonic() >= hard
    def emit(moves, **updates):
        result.update(updates, moves=legal_turn(history, moves), elapsed_ms=(time.monotonic()-started)*1000)
        publish(dict(result))
    try:
        emit(result['moves'])
        if player.options['solver'] and not stopped():
            ms = max(0, min(solver_left[0], (hard-time.monotonic())*1000))
            if ms >= 1:
                proof = proof_budget.history(history, ms=int(ms))
                result['solver_status'] = proof['status']
                result['solver_nodes'] = proof.get('nodes_used', 0)
                if not stopped() and proof['status'] == 'PROVEN_WIN' and proof.get('native_verified'):
                    moves = complete_candidate(history, proof['moves'])
                    emit(moves, proof_status='PROVEN_WIN', win_probability=1.,
                         winning_line=[[q, r, side] for q, r in moves])
                    return result
        selected = []
        tree = getattr(player, '_timed_tree', None)
        points = list(map(tuple, history))
        if tree and list(map(tuple, tree.history)) != points[:len(tree.history)]:
            tree.close()
            tree = None
        if tree:
            for action in points[len(tree.history):]:
                tree.advance(action)
        else:
            tree = NeuralSearch(player.evaluator, player.model_sha256, history, seed=1740,
                                cache=player.cache, tactics=True,
                                proof_solver=proof_budget if limits.get('leaf_solver') else None,
                                q_range_floor=limits.get('q_range_floor', 0.))
        player._timed_tree = tree
        tree.proof_solver = proof_budget if limits.get('leaf_solver') else None
        while game.player == side and game.winner < 0 and not stopped():
            first = remaining == 2 and not selected
            second_share = limits['normal_ms']*.4/1000
            stage_end = (min(hard, started + limits['normal_ms']*.6/1000) if first else
                         min(hard, max(normal, time.monotonic()+second_share)))
            if first:
                stage_end = min(stage_end, hard-max(.001, getattr(player, 'batch_seconds', .01)))
            available = stage_end-time.monotonic()
            simulation_cap = limits.get('simulations')
            if first and simulation_cap is not None and simulation_cap > 1:
                simulation_cap = max(1, int(simulation_cap*.6))
            if available <= 0:
                break
            current = [cell[:2] for cell in game.cells]
            if player.options['search']:
                rate = getattr(player, 'simulations_per_second', 200.)
                sims = max(2, min(16384, int(rate*available)))
                if limits.get('simulations') is not None:
                    sims = min(sims, max(0, simulation_cap-result['completed']))
                    if sims < 1:
                        break
                t0 = time.monotonic()
                searched = tree.search(sims, root_samples=min(limits.get('root_samples', 16), sims), batch_size=min(16, sims),
                                       milliseconds=available*1000, stop=cancel.is_set, anytime=True,
                                       batch_seconds=getattr(player, 'batch_seconds', 0.))
                duration = max(.000001, time.monotonic()-t0)
                if searched['completed']:
                    player.simulations_per_second = .7*rate + .3*searched['completed']/duration
                player.batch_seconds = max(.001, searched.get('batch_seconds', .001))
                result['completed'] += searched['completed']
                result['evaluated'] += searched['evaluated']
                extension_end = hard-max(second_share, player.batch_seconds) if first else hard
                if (searched['action'] is not None and not searched['stable_choice'] and not searched['proven'] and
                        searched['completed'] >= sims and not cancel.is_set() and
                        extension_end-time.monotonic() > player.batch_seconds):
                    extra = max(1, min(16384, int(player.simulations_per_second*(extension_end-time.monotonic()))))
                    if limits.get('simulations') is not None:
                        extra = min(extra, max(0, simulation_cap-result['completed']))
                    order = np.argsort(-searched['scores'])
                    finalists = searched['actions'][[i for i in order if np.isfinite(searched['scores'][i])][:2]]
                    if extra and len(finalists):
                        extended = tree.search(extra, root_samples=min(2, extra), batch_size=min(16, extra),
                            milliseconds=max(.001, (extension_end-time.monotonic())*1000), stop=cancel.is_set,
                            anytime=True, batch_seconds=player.batch_seconds, priority=finalists)
                        result['completed'] += extended['completed']
                        result['evaluated'] += extended['evaluated']
                        if extended['action'] is not None:
                            searched = extended
                        elif tuple(searched['action']) not in {tuple(a) for a, p in
                                zip(extended['actions'], extended['policy']) if p > 0}:
                            extended['action'] = extended['actions'][extended['policy'].argmax()].tolist()
                            searched = extended
                if searched['proven']:
                    result['proof_status'] = searched['proof_status']
                action = searched['action']
                if action is None:
                    break
                value = root_value(searched, game.player)
                policy, actions = searched['policy'], searched['actions']
            else:
                if available < getattr(player, 'batch_seconds', 0.):
                    break
                t0 = time.monotonic()
                prediction = player.evaluator.evaluate([current])[0]
                player.batch_seconds = time.monotonic()-t0
                if stopped():
                    break
                actions = prediction['actions']
                policy = np.exp(prediction['logits']-max(prediction['logits']))
                policy /= policy.sum()
                action, value = actions[policy.argmax()].tolist(), float(prediction['q'][0])
                result['evaluated'] += 1
            if not selected:
                result['suggestions'] = [dict(move=actions[i].tolist(), probability=float(policy[i]))
                                         for i in np.argsort(-policy)[:5]]
                result['win_probability'] = (value+1)/2
            game.play(*action)
            selected.append(list(action))
            tree.advance(action)
            emit(complete_candidate(history, selected))
        result['stop_reason'] = 'stop' if cancel.is_set() else 'deadline' if stopped() else 'budget'
        return result
    finally:
        game.close()


def native_turn(player, history, limits, cancel, publish=lambda result: None, analyze=False):
    """Clocked native graph owner, batched inference and concurrent proof slices."""
    started = time.monotonic()
    if player.options['solver'] and player.solver_nodes_explicit:
        raise ValueError('Native timed solving uses time slices; omit solver_nodes or disable native_scheduler')
    if limits.get('leaf_solver'):
        raise ValueError('Native timed solving uses a proof frontier; disable leaf solver queries')
    import numpy as np
    from neural_search import GameGraph
    from native_scheduler import SearchPool, InferenceService
    hard = limits.get('response_deadline', started + max(0, limits['hard_ms'])/1000)
    normal = min(hard-limits['reserve_ms']/1000,
                 limits.get('search_deadline', started + limits['normal_ms']/1000))
    game = Game(history)
    side, remaining = game.player, game.remaining
    result = dict(moves=legal_turn(history), backend='dense', checkpoint=player.checkpoint,
                  model_sha256=player.model_sha256, player=side, win_probability=None,
                  suggestions=[], winning_line=[], threat=None,
                  proof_status='UNKNOWN', solver_status='concurrent' if player.options['solver'] else 'off',
                  settings=dict(player.options) | dict(native_scheduler=True,
                      simulations=limits.get('simulations'), solver_nodes=None, solver_slice_ms=8),
                  completed=0, scheduler_completed=0, evaluated=0, solver_nodes=0, stones=[])
    service = None
    def stopped():
        return cancel.is_set() or time.monotonic() >= normal
    def emit(moves):
        if service is not None:
            result['evaluated'] = service.stats()['launched_rows']
        result.update(moves=legal_turn(history, moves), elapsed_ms=(time.monotonic()-started)*1000)
        publish(dict(result, stones=list(result['stones'])))
    try:
        emit(result['moves'])
        if stopped():
            result['stop_reason'] = 'stop' if cancel.is_set() else 'deadline'
            return result
        signature = (player.model_sha256, player.options['solver'],
                     bool(getattr(player.prover, 'stamps', False)), limits.get('q_range_floor', 0.))
        kept = getattr(player, '_timed_native', None)
        if kept is not None and kept[2] != signature:
            player.set_history(history)
            kept = None
        if kept is None:
            graph = GameGraph(player.evaluator, player.model_sha256, history, seed=1740,
                              cache=player.cache, tactics=True, q_range_floor=signature[3], round_barrier=True)
            pool = None
            try:
                pool = SearchPool([graph], quantum=64, views=8, depth=8, work=1, seed=1740)
                if player.options['solver']:
                    pool.enable_proofs(player.tactical_package, workers=2, queue=16,
                                       slice_ms=8, stamps=signature[2])
            except BaseException:
                if pool is not None:
                    pool.close()
                graph.close()
                raise
            kept = player._timed_native = graph, pool, signature
        _, pool, _ = kept
        # The preceding service stopped the pool. Re-arm its retained graph only
        # while detached, before a new service takes exclusive ownership.
        pool.retarget(0, history, work=1)
        before = pool.proofs.stats() if pool.proofs is not None else None
        service = InferenceService([pool], [player.evaluator], batch_size=128, quantum=64,
                                   pending=2, flights=2, interleave_feedback=True)
        service.start(continuous=True)
        # Inference collection may wait for a whole CPU/GPU batch. Keep it off
        # the turn thread so a ready root choice can be published immediately.
        service.launch()
        selected, token = [], 0
        while game.player == side and game.winner < 0 and not stopped():
            first = remaining == 2 and not selected
            end = min(normal, started + limits['normal_ms']*.6/1000) if first else normal
            ms = max(0., (min(hard, end)-time.monotonic())*1000)
            cap = limits.get('simulations')
            work = 0 if cap is None else max(0, cap-result['scheduler_completed'])
            if first and work:
                work = max(1, int(work*.6))
            if ms <= 0 or cap is not None and not work:
                break
            current = [list(cell[:2]) for cell in game.cells]
            service.retarget(0, 0, current, expected=token, work=work, ms=ms,
                             samples=limits.get('root_samples', 16), views=8)
            found = None
            while True:
                found = service.event()
                if found is not None:
                    break
                if stopped():
                    # Stop admission, then receive the writer's final choice.
                    # Response time is separate from search time; no graph is
                    # inspected here and no new forward is launched.
                    service.cancel()
                    while time.monotonic() < hard:
                        found = service.event()
                        if found is not None:
                            break
                        time.sleep(.0001)
                    if found is None:
                        finals = service.close(completions=True)
                        if len(finals) > 1:
                            raise ValueError('Multiple final completions for one native turn root')
                        found = finals[0] if finals else None
                    break
                service.wait(max(0., min(1., (normal-time.monotonic())*1000)))
            if found is None:
                break
            token += 1
            if (found['producer'], found['game'], found['model'], found['token'], found['history']) != (0, 0, 0, token, current):
                raise ValueError('Native turn completion does not match the current position')
            if 'error' in found:
                if found['error'] in ('deadline', 'cancelled'):
                    break
                raise ValueError(f"Native turn search failed: {found['error']}")
            edges = np.asarray(found['edges'], np.float64)
            winner = found['exact_winner']
            value = (1. if winner == side else -1.) if winner >= 0 else float(edges[:, 5] @ edges[:, 4])
            probability = (value+1)/2
            result['completed'] += found['root_completed']
            result['scheduler_completed'] += found['completed']
            if not selected:
                result['win_probability'] = probability
                result['proof_status'] = ('PROVEN_WIN' if winner == side else 'PROVEN_LOSS') if winner >= 0 else 'UNKNOWN'
                result['suggestions'] = [dict(move=edges[i, :2].astype(np.int64).tolist(), probability=float(edges[i, 5]))
                                         for i in np.argsort(-edges[:, 5])[:5]]
            elif winner == side:
                # One winning same-player continuation proves the original root;
                # a losing chosen continuation does not cover its alternatives.
                result.update(win_probability=1., proof_status='PROVEN_WIN')
            action = found['action']
            if winner < 0 and edges[:, 5].sum() > 0:
                action = edges[np.argmax(edges[:, 5]), :2].astype(np.int64).tolist()
            witness = found.get('winning_turn', [])
            if witness:
                if winner != side or witness[0] != action:
                    raise ValueError('Winning turn does not match the exact root action')
                legal_turn(current, witness)
            elif winner == side and game.remaining == 2:
                raise ValueError('Winning root has no complete turn witness')
            result['stones'].append(dict(history=current, move=action, win_probability=probability,
                                         exact_winner=winner, completed=found['root_completed'],
                                         scheduler_completed=found['completed'],
                                         root_completed=found['root_completed'], context=found['context']))
            game.play(*action)
            selected.append(action)
            if witness:
                for second in witness[1:]:
                    result['stones'].append(dict(history=[list(cell[:2]) for cell in game.cells],
                        move=second, win_probability=1., exact_winner=side, completed=0,
                        scheduler_completed=0, root_completed=0, context=None, source='proof_witness'))
                    game.play(*second)
                    selected.append(second)
                result['winning_turn'] = list(selected)
                emit(selected)
                break
            emit(complete_candidate(history, selected))
        result['stop_reason'] = 'stop' if cancel.is_set() else 'deadline' if time.monotonic() >= normal else 'budget'
    finally:
        try:
            if service is not None:
                service.close()
                result['inference'] = service.stats()
                result['evaluated'] = result['inference']['launched_rows']
                if pool.proofs is not None:
                    after = pool.proofs.stats()
                    result['proof_work'] = {key: after[key]-before[key] for key in
                        ('submitted', 'finished', 'installed', 'unknown', 'fresh_nodes', 'missing_fresh', 'worker_service_ms')}
                    result['solver_nodes'] = result['proof_work']['fresh_nodes']
        finally:
            game.close()
    result['elapsed_ms'] = (time.monotonic()-started)*1000
    return result


def _worker(connection, cancellation, config):
    """Own the model and its mutable tree in one spawned process."""
    player = None
    try:
        kind = config.get('kind', 'bubble')
        if kind == 'bubble':
            import torch
            from dense_player import DensePlayer
            torch.set_num_threads(2)
            run = Path(config.get('run', '.'))
            model = config.get('model')
            if not model and config.get('checkpoint'):
                checkpoint = config['checkpoint']
                if checkpoint == 'champion':
                    import json
                    checkpoint = json.loads((run/'champion.json').read_text(encoding='utf-8'))['checkpoint']
                elif checkpoint == 'newest':
                    checkpoint = max((p.parent for p in (run/'checkpoints').glob('*/*/ema.pt')),
                                     key=lambda p: int(p.name)).relative_to(run/'checkpoints').as_posix()
                model = run/'checkpoints'/checkpoint/'ema.pt'
            search, solver = config.get('search', {}), config.get('solver', {})
            player = DensePlayer(run, config.get('device', 'cpu'), model=model,
                                 tactical_package=Path(config['tactical_package']) if config.get('tactical_package') else None,
                                 net_kernels=config.get('net_kernels', 'fused'),
                                 native_scheduler=search.get('native_scheduler', False))
            if player.prover is not None:
                player.prover.stamps = bool(solver.get('stamps', False))
            options = dict(search=search.get('enabled', True), solver=solver.get('enabled', player.prover is not None))
            if 'simulations' in search:
                options['simulations'] = search['simulations']
            if 'nodes' in solver:
                options['solver_nodes'] = solver['nodes']
            player.configure(options)
            t0 = time.monotonic()
            player.evaluator.evaluate([[(0, 0)]])
            player.batch_seconds = time.monotonic()-t0
            identity = dict(checkpoint=player.checkpoint, model_sha256=player.model_sha256)
            search_limits = dict(simulations=search.get('max_simulations', search.get('simulations')),
                                 root_samples=search.get('root_samples', 16),
                                 q_range_floor=search.get('q_range_floor', 0.),
                                 leaf_solver=solver.get('leaf', False) and player.options['solver'])
        elif kind == 'six':
            from six_engine import SixEngine
            player = SixEngine(config['command'], cancel=cancellation, mirrored=config.get('mirrored', False),
                               cwd=config.get('cwd'), path=config.get('path', ()), startup=120)
            identity = dict(checkpoint='six', command=config['command'])
        elif kind == 'seal':
            from legacy.arena import Seal
            player = Seal(config.get('library'))
            identity = dict(checkpoint='seal')
        elif kind == 'htttx':
            player = HTTTXEngine(config['url'])
            identity = dict(checkpoint='htttx', url=config['url'], capabilities=player.capabilities,
                            clock_allocation='host')
        elif kind == 'native':
            identity = dict(checkpoint='native')
        else:
            raise ValueError(f'Unsupported engine kind {kind}')
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        identity.update(source_revision=revision, engine_options=config,
                        native_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        if kind == 'bubble':
            gumbel = library.with_name(library.name.replace('hexo', 'hexo_gumbel'))
            identity['gumbel_sha256'] = hashlib.sha256(gumbel.read_bytes()).hexdigest()
            identity['tactical_build_hash'] = player.prover.metadata['binary_sha256'] if player.prover else None
        connection.send(('ready', identity))
        while True:
            request = connection.recv()
            if request is None:
                break
            if request[0] == 'reset':
                if kind == 'bubble':
                    player.set_history(request[1])
                    if hasattr(player, 'simulations_per_second'):
                        del player.simulations_per_second
                connection.send(('reset', None))
                continue
            generation, history, limits = request
            def publish(result):
                connection.send((generation, 'progress', result))
            try:
                if cancellation.is_set():
                    result = dict(moves=legal_turn(history))
                elif kind == 'bubble':
                    result = dense_turn(player, history, limits | search_limits, cancellation, publish)
                else:
                    game = Game(history)
                    try:
                        budget = max(1, int(min(limits['normal_ms'], config.get('max_ms', limits['normal_ms']))))
                        if kind == 'six':
                            result = dict(moves=player(game, budget, nodes=config.get('nodes'), clock=limits.get('clock')),
                                          nodes=player.info.get('nodes'))
                        elif kind == 'seal':
                            result = dict(moves=player(game, budget))
                        elif kind == 'htttx':
                            result = dict(moves=player.turn(game, budget))
                        else:
                            result = game.search(ms=budget, width=config.get('width', 16),
                                                 depth=config.get('depth', 12))
                    finally:
                        game.close()
                connection.send((generation, 'done', result))
            except Exception as error:
                reason = ('timeout' if isinstance(error, TimeoutError) else
                          'illegal' if kind in ('six', 'htttx') and isinstance(error, (ValueError, KeyError, TypeError))
                          else 'crash')
                connection.send((generation, 'error', dict(reason=reason, message=str(error))))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send(('error', str(error)))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if player and config.get('kind') != 'seal':
            player.close()
        connection.close()


class TimedEngine:
    """Return a completed legal candidate by the controller deadline, including on stop."""
    def __init__(self, config, *, startup_timeout=120):
        search, solver = config.get('search', {}), config.get('solver', {})
        if config.get('kind') == 'bubble' and search.get('native_scheduler') and search.get('enabled', True) and solver.get('enabled', True):
            if 'nodes' in solver:
                raise ValueError('Native timed solving uses time slices; omit solver.nodes or disable native_scheduler')
            if solver.get('leaf'):
                raise ValueError('Native timed solving uses a proof frontier; disable leaf solver queries')
        cap = search.get('max_simulations', search.get('simulations'))
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise ValueError('simulation cap must be a positive integer')
        self.external = config.get('kind') in ('six', 'htttx')
        self.config = dict(config)
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.cancellation = context.Event()
        self.lock = threading.Lock()
        self.generation = 0
        self.busy = False
        self.process = context.Process(target=_worker, args=(child, self.cancellation, self.config), daemon=True)
        self.process.start()
        child.close()
        try:
            if not self.connection.poll(startup_timeout):
                raise TimeoutError('Engine initialization timed out')
            status, identity = self.connection.recv()
            if status != 'ready':
                raise RuntimeError(identity)
            self.identity = identity
            self.checkpoint = identity['checkpoint']
            self.model_sha256 = identity.get('model_sha256', '')
        except BaseException:
            self.close()
            raise

    def set_history(self, history=()):
        pass  # Each request carries the complete confirmed history.

    def wait_idle(self, timeout=5):
        """Drain a cancelled turn before another engine takes the shared hardware."""
        started = time.monotonic()
        deadline = started + timeout
        if not self.lock.acquire(timeout=max(0, timeout)):
            raise TimeoutError('Engine controller did not become idle')
        try:
            if self.busy:
                self.cancellation.set()
            while self.busy:
                left = deadline-time.monotonic()
                if left <= 0:
                    raise TimeoutError('Engine worker did not stop after its turn')
                if self.connection.poll(min(.01, left)):
                    ident, status, result = self.connection.recv()
                    if ident != self.generation:
                        continue
                    if status in ('done', 'error'):
                        self.busy = False
                    if status == 'error':
                        raise RuntimeError(result['message'])
                elif not self.process.is_alive():
                    raise RuntimeError('Engine worker exited')
            return (time.monotonic()-started)*1000
        finally:
            self.lock.release()

    def reset(self, history=(), timeout=5):
        """Start a benchmark game with empty Bubble search and prediction caches."""
        self.wait_idle(timeout)
        with self.lock:
            self.connection.send(('reset', history))
            if not self.connection.poll(timeout):
                raise TimeoutError('Engine game reset timed out')
            if self.connection.recv() != ('reset', None):
                raise RuntimeError('Engine game reset failed')

    def turn(self, game, milliseconds=None, *, clock=None, cancel=None, publish=None):
        history = [list(cell[:2]) for cell in game.cells]
        started = time.monotonic()
        limits = allowance(clock, game.player, milliseconds)
        if clock and self.config.get('kind') == 'six':
            # A clock-aware external engine owns its allocation, bounded by the host's remaining clock.
            remaining = clock['cross_ms' if game.player == 0 else 'circle_ms']
            limits['hard_ms'] = remaining if milliseconds is None else min(remaining, milliseconds)
            limits['clock'] = dict(clock)
        native_clocked = (self.config.get('kind', 'bubble') == 'bubble' and
                          self.config.get('search', {}).get('native_scheduler') and
                          self.config.get('search', {}).get('enabled', True))
        # Native search stops before the reserve; its final result is delivered
        # during it. Keep listening through the inclusive response deadline.
        response_ms = limits['hard_ms'] if native_clocked else limits['hard_ms']-limits['reserve_ms']
        deadline = started + max(0, response_ms)/1000
        best = dict(moves=legal_turn(history), backend='timed', checkpoint=self.checkpoint,
                    model_sha256=self.model_sha256, stop_reason='deadline', elapsed_ms=0,
                    allowance=limits)
        if not self.lock.acquire(timeout=max(0, deadline-time.monotonic())):
            if self.external:
                raise TimeoutError('Opponent worker is busy')
            best.update(stop_reason='busy', elapsed_ms=(time.monotonic()-started)*1000)
            return best
        try:
            def drain():
                while self.connection.poll():
                    message = self.connection.recv()
                    if len(message) == 3 and message[1] in ('done', 'error'):
                        self.busy = False
            drain()
            while self.busy and time.monotonic() < deadline:
                if cancel is not None and cancel.is_set():
                    break
                if not self.process.is_alive():
                    raise RuntimeError('Engine worker exited')
                self.connection.poll(min(.005, max(0, deadline-time.monotonic())))
                drain()
            if self.busy or time.monotonic() >= deadline:
                if self.external:
                    raise TimeoutError('Opponent did not return a move within its allowance')
                best.update(stop_reason='busy' if self.busy else 'deadline',
                            elapsed_ms=(time.monotonic()-started)*1000)
                return best
            if not self.process.is_alive():
                raise RuntimeError('Engine worker exited')
            self.cancellation.clear()
            self.generation += 1
            generation = self.generation
            # Dispatch/fallback time consumes the worker's allowance too.
            elapsed = (time.monotonic()-started)*1000
            worker_limits = limits | dict(hard_ms=max(0, limits['hard_ms']-elapsed),
                                         normal_ms=max(0, limits['normal_ms']-elapsed))
            if native_clocked:
                # Monotonic time is shared by local processes. Queue/IPC delay
                # consumes the turn instead of restarting its clock on receipt.
                worker_limits.update(response_deadline=deadline,
                                     search_deadline=started+limits['normal_ms']/1000)
            self.connection.send((generation, history, worker_limits))
            self.busy = True
            complete = False
            try:
                while time.monotonic() < deadline:
                    if cancel is not None and cancel.is_set():
                        best['stop_reason'] = 'stop'
                        break
                    if not self.connection.poll(min(.005, max(0, deadline-time.monotonic()))):
                        if not self.process.is_alive():
                            raise RuntimeError('Engine worker exited')
                        continue
                    ident, status, result = self.connection.recv()
                    if ident != generation:
                        continue
                    if status in ('done', 'error'):
                        self.busy = False
                    if status == 'error':
                        error = {'timeout': TimeoutError, 'illegal': ValueError}.get(result['reason'], RuntimeError)
                        raise error(result['message'])
                    result['moves'] = legal_turn(history, result['moves'])
                    best.update(result)
                    if publish:
                        publish(dict(best))
                    if status == 'done':
                        complete = True
                        best['stop_reason'] = 'budget'
                        break
            finally:
                if self.busy:
                    self.cancellation.set()
            best['elapsed_ms'] = (time.monotonic()-started)*1000
            best['allowance'] = limits
            if self.external and not complete and (cancel is None or not cancel.is_set()):
                raise TimeoutError('Opponent did not return a move within its allowance')
            return best
        finally:
            self.lock.release()

    def close(self):
        self.cancellation.set()
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, OSError):
                pass
            self.process.join(.2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
