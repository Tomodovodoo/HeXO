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
                                proof_solver=proof_budget if limits.get('leaf_solver') else None)
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
            if available <= 0:
                break
            current = [cell[:2] for cell in game.cells]
            if player.options['search']:
                rate = getattr(player, 'simulations_per_second', 200.)
                sims = max(2, min(16384, int(rate*available)))
                if limits.get('simulations') is not None:
                    sims = min(sims, max(0, limits['simulations']-result['completed']))
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
                        extra = min(extra, max(0, limits['simulations']-result['completed']))
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
            player = DensePlayer(run, config.get('device', 'cpu'), model=model,
                                 net_kernels=config.get('net_kernels', 'fused'))
            search, solver = config.get('search', {}), config.get('solver', {})
            player.configure(dict(search=search.get('enabled', True),
                                  simulations=search.get('simulations', 128),
                                  solver=solver.get('enabled', player.prover is not None),
                                  solver_nodes=solver.get('nodes', 32768)))
            t0 = time.monotonic()
            player.evaluator.evaluate([[(0, 0)]])
            player.batch_seconds = time.monotonic()-t0
            identity = dict(checkpoint=player.checkpoint, model_sha256=player.model_sha256)
            search_limits = dict(simulations=search.get('max_simulations', search.get('simulations')),
                                 root_samples=search.get('root_samples', 16),
                                 leaf_solver=solver.get('leaf', False) and player.options['solver'])
        elif kind == 'six':
            from six_engine import SixEngine
            player = SixEngine(config['command'], cancel=cancellation)
            identity = dict(checkpoint='six', command=config['command'])
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
                        budget = max(1, int(limits['normal_ms']))
                        if kind == 'six':
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
        if player:
            player.close()
        connection.close()


class TimedEngine:
    """Return a completed legal candidate by the controller deadline, including on stop."""
    def __init__(self, config, *, startup_timeout=120):
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

    def turn(self, game, milliseconds=None, *, clock=None, cancel=None, publish=None):
        history = [list(cell[:2]) for cell in game.cells]
        started = time.monotonic()
        limits = allowance(clock, game.player, milliseconds)
        deadline = started + max(0, limits['hard_ms']-limits['reserve_ms'])/1000
        best = dict(moves=legal_turn(history), backend='timed', checkpoint=self.checkpoint,
                    model_sha256=self.model_sha256, stop_reason='deadline', elapsed_ms=0,
                    allowance=limits)
        if not self.lock.acquire(timeout=max(0, deadline-time.monotonic())):
            if self.external:
                raise TimeoutError('Opponent worker is busy')
            best.update(stop_reason='busy', elapsed_ms=(time.monotonic()-started)*1000)
            return best
        try:
            while self.connection.poll():
                message = self.connection.recv()
                if len(message) == 3 and message[1] in ('done', 'error'):
                    self.busy = False
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
