"""Direct saved relational policy/Q players, without NNUE export or fallback."""
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from hexo import Game
from legacy.relational_model import NeuralEvaluator
from legacy.relational_train import load_model


MODES = ('pi', 'mu', 'gumbel', 'gumbel-proof')


class RelationalPlayer:
    def __init__(self, checkpoint, *, mode='gumbel', device='cuda', simulations=16,
                 root_samples=8, batch_size=4, max_nodes=12000, max_edges=1000000,
                 milliseconds=10000, seed=0, alpha=.03, beta=.1, proof_ms=1000):
        if mode not in MODES or min(simulations, root_samples, batch_size, milliseconds) < 1:
            raise ValueError('Invalid relational player mode or search budget')
        if alpha < 0 or beta < 0 or alpha+beta <= 0:
            raise ValueError('KLENT coefficients require nonnegative values and positive sum')
        if not 1 <= proof_ms <= 60000:
            raise ValueError('Proof budget must be between 1 and 60000 milliseconds')
        self.mode, self.seed = mode, seed
        self.simulations, self.root_samples, self.batch_size = simulations, root_samples, batch_size
        self.milliseconds, self.alpha, self.beta = milliseconds, alpha, beta
        self.proof_ms = proof_ms
        self.model_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        model = load_model(checkpoint, 'cpu', expected_sha256=self.model_sha256)
        if model.config.head == 'value' and mode == 'mu':
            raise ValueError('Policy/value models use pi or Gumbel; they have no action-Q head for KLENT')
        self.evaluator = NeuralEvaluator(model, device, max_nodes=max_nodes, max_edges=max_edges)
        self.tree, self.cache, self.prover = None, None, None
        if mode == 'gumbel-proof':
            from tactical_proof import NativeTactics
            self.prover = NativeTactics()
        self.set_history()

    def close(self):
        if self.tree is not None:
            self.tree.close()
            self.tree = None

    def set_history(self, history=()):
        self.close()
        self.history = [tuple(map(int, point)) for point in history]
        if self.mode.startswith('gumbel'):
            from neural_search import EvaluationCache, NeuralSearch
            self.cache = EvaluationCache()
            self.tree = NeuralSearch(self.evaluator, self.model_sha256, self.history, self.seed, self.cache,
                                     tactics=True, proof_solver=self.prover, proof_ms=self.proof_ms)

    def _sync(self, game):
        current = [tuple(cell[:2]) for cell in game.cells]
        if current[:len(self.history)] != self.history:
            self.set_history(current)
        else:
            for action in current[len(self.history):]:
                self._advance(action)

    def _advance(self, action):
        point = tuple(map(int, action))
        if self.tree is not None:
            self.tree.advance(point)
        self.history.append(point)

    def _reactive(self, game):
        prediction = self.evaluator.evaluate([self.history])[0]
        actions = prediction['actions']
        if {tuple(p) for p in actions} != set(game.legal_moves()) or len(actions) != len(set(map(tuple, actions))):
            raise ValueError('Relational action support disagrees with exact legal placements')
        logits = torch.as_tensor(prediction['logits'], dtype=torch.float32)
        q = torch.as_tensor(prediction['q'], dtype=torch.float32)
        pi = torch.softmax(logits, 0)
        if self.mode == 'mu':
            from legacy.klent import improved_policy
            policy, _, _, _ = improved_policy(logits, q, torch.zeros(len(q), dtype=torch.int64), 1, self.alpha, self.beta)
        else:
            policy = pi
        selected = int(torch.argmax(policy))
        return actions[selected].tolist(), dict(decision_source=self.mode, legal_actions=len(actions),
            selected_q=float(q[selected]), value_pi=float(torch.dot(pi, q)),
            max_probability=float(policy.max()), entropy=float(-(policy*policy.clamp_min(1e-30).log()).sum()))

    def turn(self, game, milliseconds=None):
        """Choose a complete turn without changing game; deadline is cooperative."""
        budget = self.milliseconds if milliseconds is None else milliseconds
        if budget <= 0 or game.winner >= 0:
            raise ValueError('Positive deadline and nonterminal game required')
        self._sync(game)
        local = Game(self.history)
        side, moves, diagnostics = local.player, [], []
        start = time.perf_counter()
        proof = None
        try:
            if self.prover is not None:
                from tactical_proof import MAX_NODES
                proof_ms = min(self.proof_ms, max(1, int(budget/4)))
                proof = self.prover.solve(local, ms=proof_ms, nodes=MAX_NODES)
                if proof.get('status') == 'PROVEN_WIN' and proof.get('native_verified'):
                    for action in proof['moves']:
                        if local.winner >= 0 or local.player != side:
                            raise ValueError('Proof continued beyond the current turn')
                        local.play(*action)
                        moves.append(list(action))
                        self._advance(action)
                    if local.winner < 0 and local.player == side:
                        raise ValueError('Verified root action does not complete its turn')
            while local.winner < 0 and local.player == side:
                remaining = budget-(time.perf_counter()-start)*1000
                if remaining <= 0:
                    raise TimeoutError('Relational complete-turn deadline expired')
                decision_start = time.perf_counter()
                if self.tree is None:
                    action, detail = self._reactive(local)
                else:
                    result = self.tree.search(simulations=self.simulations, root_samples=self.root_samples,
                                              batch_size=self.batch_size, milliseconds=remaining/local.remaining, choice='gumbel')
                    action = result['action']
                    if action is None:
                        raise TimeoutError('Gumbel deadline produced no completed neural action; no fallback')
                    detail = {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in result.items() if key != 'action'}
                    detail['decision_source'] = 'gumbel'
                local.play(*action)
                moves.append(list(action))
                self._advance(action)
                diagnostics.append({**detail, 'action': list(action), 'elapsed_ms': (time.perf_counter()-decision_start)*1000})
            status = 'UNKNOWN'
            if proof and proof.get('status') == 'PROVEN_WIN' and proof.get('native_verified'):
                status = 'PROVEN_WIN'
            elif any(item.get('proof_status') == 'PROVEN_WIN' for item in diagnostics):
                status = 'PROVEN_WIN'
            elif diagnostics and diagnostics[0].get('proof_status') == 'PROVEN_LOSS':
                status = 'PROVEN_LOSS'
            elapsed = (time.perf_counter()-start)*1000
            return dict(moves=moves, elapsed_ms=elapsed, backend=self.mode, model_sha256=self.model_sha256,
                        placements=diagnostics, deadline_ms=budget, deadline_scope='complete-turn-cooperative',
                        overrun_ms=max(0, elapsed-budget),
                        proof_status=status,
                        proof=proof, proof_budget_ms=self.proof_ms if self.prover is not None else None,
                        proof_scope='verified-root-and-tree-tactics' if self.prover is not None else 'disabled')
        except Exception:
            self.set_history([cell[:2] for cell in game.cells])
            raise
        finally:
            local.close()
