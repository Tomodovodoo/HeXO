"""Dense player shared by browser play, Six and the clocked engine worker."""
import json
from pathlib import Path
import sys
import time
from hexo import Game


class DensePlayer:
    """Play exported dense checkpoints; analysis searches a copy of the browser game."""
    mode = 'dense'

    def __init__(self, run, device, tactical_package=None, model=None, net_kernels='fused', native_scheduler=False):
        if type(native_scheduler) is not bool:
            raise ValueError('native_scheduler must be on or off')
        self.native_scheduler = native_scheduler
        self.run, self.device = run, device
        self.model_path = Path(model).resolve() if model else None
        self.tactical_package = tactical_package
        self.net_kernels = net_kernels
        self.evaluator = self.prover = None
        self.checkpoint = None
        try:
            from tactical_proof import NativeTactics
            self.prover = NativeTactics(**({'package': tactical_package} if tactical_package else {}))
        except (OSError, ValueError, KeyError) as error:
            print(f'solver off: {error}', file=sys.stderr)
        self.options = dict(search=True, simulations=128, solver=self.prover is not None, solver_nodes=32768)
        available = self.models()
        if not available:
            raise ValueError('No playable dense exports found')
        self.select(available[0]['id'])

    def models(self):
        if self.model_path:
            path = self.model_path
            name = (f'{path.parent.parent.name}/{path.parent.name}'
                    if path.name == 'ema.pt' and path.parent.parent.parent.name == 'checkpoints' else path.stem)
            if not hasattr(self, 'model_digest'):
                from legacy.train import digest
                self.model_digest = digest(path)
            checkpoint = f'{name}@{self.model_digest[:12]}'  # notes and caches are keyed by the weights, not the file name
            return [dict(id=checkpoint, label=checkpoint)]
        exports = sorted((p for p in (self.run/'checkpoints').glob('*/*/ema.pt') if p.parent.name.isdigit()),
                         key=lambda p: (int(p.parent.name), p.parent.parent.name), reverse=True)
        ids = [p.parent.relative_to(self.run/'checkpoints').as_posix() for p in exports]
        if not ids:
            return []
        champion_file = self.run/'champion.json'
        champion = json.loads(champion_file.read_text(encoding='utf-8'))['checkpoint'] if champion_file.exists() else None
        reference = 'main/065000' if 'main/065000' in ids else ids[-1]
        selected = []
        for checkpoint in (champion, ids[0], reference, *ids):
            if checkpoint in ids and checkpoint not in selected:
                selected.append(checkpoint)
            if len(selected) == 4:
                break
        labels = {champion: 'champion', ids[0]: 'newest', reference: 'reference'}
        if champion == ids[0]:
            labels[champion] = 'champion, newest'
        return [dict(id=k, label=f'{k} · {labels.get(k, "earlier export")}') for k in selected]

    def select(self, checkpoint):
        if checkpoint not in {m['id'] for m in self.models()}:
            raise ValueError('Choose one of the available checkpoints')
        if checkpoint == self.checkpoint:
            return
        import hexnet
        from legacy.train import digest
        path = self.model_path or self.run/'checkpoints'/checkpoint/'ema.pt'
        model = hexnet.load_model(path, net_kernels=self.net_kernels)
        self.evaluator = hexnet.DenseEvaluator(model, self.device, digest(path),
            max_batch=128 if self.native_scheduler else 16,
            cuda_graphs=self.native_scheduler and self.net_kernels == 'fused')
        if self.native_scheduler:
            self.evaluator.free = []  # Packed forwards own staging until their completion fence.
        self.checkpoint, self.model_sha256 = checkpoint, digest(path)
        self.set_history()

    def configure(self, options):
        updated = self.options | options
        if any(type(updated[k]) is not bool for k in ('search', 'solver')):
            raise ValueError('Search and solver must be on or off')
        if updated['solver'] and self.prover is None:
            raise ValueError('The tactical solver is not built; run python tools/build_tactical.py')
        for key, maximum in (('simulations', 4096), ('solver_nodes', 1000000)):
            if type(updated[key]) is not int or not 1 <= updated[key] <= maximum:
                raise ValueError(f'{key} must be 1..{maximum}')
        self.options = updated

    def set_history(self, history=()):
        from neural_search import EvaluationCache
        if getattr(self, '_timed_native', None):
            graph, pool, _ = self._timed_native
            pool.close()
            graph.close()
            self._timed_native = None
        if getattr(self, '_timed_tree', None):
            self._timed_tree.close()
            self._timed_tree = None
        self.cache = EvaluationCache(4096)

    def close(self):
        self.set_history()
        if self.evaluator is not None and self.evaluator.graph is not None:
            self.evaluator.graph.close()
        self.evaluator = None

    def solve(self, history, attacker='mover'):
        return self.prover.history(history, attacker=attacker, nodes=self.options['solver_nodes'], ms=10000)

    @staticmethod
    def winning_line(history, result):
        """One legal continuation of a verified strategy, choosing its first covered defender reply."""
        from dense_solver import Proof
        certificate = result.get('certificate') or json.loads(result['certificate_json'])
        proof = Proof(list(map(tuple, history)), certificate)
        local, line = Game(history), []
        try:
            while local.winner < 0:
                current = [cell[:2] for cell in local.cells]
                move = proof.path(current)[1]
                if move is None:
                    break
                actions = move[0] or proof.reply(current) or local.legal_moves()[:local.remaining]
                for action in actions:
                    line.append([*action, local.player])
                    local.play(*action)
                    if local.winner >= 0:
                        break
            return line
        finally:
            local.close()

    def turn(self, game, milliseconds=None, analyze=False):
        """Keep the first stone's search continuation, with a fresh budget for each placement."""
        if milliseconds is not None:
            from threading import Event
            from timed_engine import dense_turn
            from time_control import allowance
            return dense_turn(self, [cell[:2] for cell in game.cells], allowance(movetime=milliseconds),
                              Event(), analyze=analyze)
        import numpy as np
        from neural_search import NeuralSearch
        from dense_selfplay import root_value
        if game.winner >= 0:
            raise ValueError('This game has finished')
        history = [cell[:2] for cell in game.cells]
        local, moves, suggestions = Game(history), [], []
        player, start, proof, line, threat = local.player, time.perf_counter(), None, [], None
        win_probability = tree = None
        try:
            if self.options['solver']:
                proof = self.solve(history)
                if proof['status'] == 'PROVEN_WIN' and proof.get('native_verified'):
                    moves = proof['moves']
                    line = self.winning_line(history, proof)
                if analyze:
                    danger = self.solve(history, 'opponent')
                    if danger['status'] == 'PROVEN_WIN' and danger.get('native_verified'):
                        threat = dict(moves=danger['moves'], turns=danger['proof_turns'])
            proven = bool(proof and proof['status'] == 'PROVEN_WIN' and proof.get('native_verified'))
            while not proven and local.player == player and local.winner < 0:
                current = [cell[:2] for cell in local.cells]
                if self.options['search']:
                    if tree is None:
                        tree = NeuralSearch(self.evaluator, self.model_sha256, current, seed=1740,
                                            cache=self.cache, tactics=True)
                    result = tree.search(self.options['simulations'], root_samples=16, batch_size=16)
                    action, policy, actions = result['action'], result['policy'], result['actions']
                    value = root_value(result, local.player)
                else:
                    result = self.evaluator.evaluate([current])[0]
                    actions = result['actions']
                    policy = np.exp(result['logits']-result['logits'].max()); policy /= policy.sum()
                    action, value = actions[policy.argmax()].tolist(), float(result['q'][0])
                if not suggestions:
                    suggestions = [dict(move=actions[i].tolist(), probability=float(policy[i]))
                                   for i in np.argsort(-policy)[:5]]
                    win_probability = (value+1)/2
                moves.append(action)
                local.play(*action)
                if tree is not None:
                    tree.advance(action)
            return dict(moves=moves, backend='dense', checkpoint=self.checkpoint,
                        elapsed_ms=(time.perf_counter()-start)*1000, suggestions=suggestions,
                        player=player, win_probability=1. if proven else win_probability,
                        proof_status='PROVEN_WIN' if proven else 'UNKNOWN', winning_line=line, threat=threat,
                        threat_checked=analyze and self.options['solver'],
                        solver_status=proof['status'] if proof else 'off', settings=dict(self.options))
        finally:
            if tree is not None:
                tree.close()
            local.close()
