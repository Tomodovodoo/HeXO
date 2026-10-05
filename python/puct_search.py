"""CPU PUCT scheduler for fixed-budget comparisons, using the existing neural evaluator.

Q is the edge's mean value for its parent mover; unvisited edges use the node value.
Selection is Q + cpuct * prior * sqrt(1 + visits) / (1 + edge visits).
There is no root noise or temperature, and the final move is most visited.
Native Gumbel expansion supplies only immediate tactical eligibility and exact
results, without running a Gumbel simulation. Graph nodes share the network's
complete turn-context key. This is an experimental Python backend, not Six's
implementation or a replacement for the production native scheduler.
"""
import numpy as np
from hexo import Game
from neural_search import EvaluationCache, NeuralSearch, checked, native


class Node:
    def __init__(self, history, player, winner=-1):
        self.history, self.player, self.winner = history, player, winner
        self.expanded, self.value = False, 0.
        self.children, self.parents = {}, set()

    def settle(self):
        if not self.expanded or self.winner >= 0:
            return
        winning = [i for i, c in self.children.items() if c.winner == self.player]
        if winning:
            self.winner = self.player
            self.eligible[:] = False
            self.eligible[winning] = True
        else:
            for i, c in self.children.items():
                if c.winner == 1-self.player:
                    self.eligible[i] = False
            if not self.eligible.any():
                self.winner = 1-self.player
                self.eligible[:] = True
        if self.winner >= 0:
            for parent, _ in self.parents:
                parent.settle()

    def q(self):
        out = np.full(len(self.actions), self.value)
        np.divide(self.sums, self.visits, out=out, where=self.visits > 0)
        for i, child in self.children.items():
            if child.winner >= 0:
                out[i] = 1. if child.winner == self.player else -1.
        return out


class PUCTSearch:
    def __init__(self, evaluator, model_version, history=(), seed=0, cache=None,
                 tactics=False, graph=False, cpuct=1.5):
        if cpuct <= 0:
            raise ValueError('cpuct must be positive')
        self.evaluator, self.model_version = evaluator, model_version
        self.cache = cache if cache is not None else EvaluationCache()
        self.history = [tuple(map(int, m)) for m in history]
        self.game = Game(self.history)
        self.tactics, self.graph, self.cpuct = tactics, graph, cpuct
        self.nodes = {}
        self.root = self.node(self.history)

    def node(self, history):
        key = self.cache.key(history, self.model_version)
        if self.graph and key in self.nodes:
            return self.nodes[key]
        node = Node(tuple(history), self.game.player, self.game.winner)
        if self.graph:
            self.nodes[key] = node
        return node

    def close(self):
        self.game.close()
        self.nodes.clear()

    def advance(self, action):
        action = tuple(map(int, action))
        old = self.root
        self.game.play(*action)
        self.history.append(action)
        self.root = next((old.children[i] for i, m in enumerate(old.actions)
                          if tuple(m) == action and i in old.children), None) if old.expanded else None
        self.root = self.root or self.node(self.history)
        reachable, todo = set(), [self.root]
        while todo:
            node = todo.pop()
            if node not in reachable:
                reachable.add(node)
                todo.extend(node.children.values())
        for node in reachable:
            node.parents = {(p, i) for p, i in node.parents if p in reachable}
        if self.graph:
            self.nodes = {k: n for k, n in self.nodes.items() if n in reachable}

    def begin(self, simulations):
        if simulations < 1:
            raise ValueError('simulations must be positive')
        self.budget, self.completed = simulations, 0

    def done(self):
        return self.completed >= self.budget or self.root.winner >= 0

    def request(self):
        node, path, played = self.root, [], 0
        try:
            while node.expanded and node.winner < 0:
                score = node.q() + self.cpuct*node.prior*np.sqrt(1+node.visits.sum())/(1+node.visits)
                i = int(np.argmax(np.where(node.eligible, score, -np.inf)))
                self.game.play(*map(int, node.actions[i]))
                played += 1
                if i not in node.children:
                    history = self.history + [tuple(map(int, p.actions[j])) for p, j in path] + [tuple(map(int, node.actions[i]))]
                    node.children[i] = self.node(history)
                    node.children[i].parents.add((node, i))
                path.append((node, i))
                node = node.children[i]
            if node.winner >= 0:
                self.backup(node, path, 1. if node.winner == node.player else -1.)
                return None
            return node, path
        finally:
            for _ in range(played):
                self.game.undo()

    def fulfill(self, request, prediction):
        node, path = request
        oracle = NeuralSearch(None, self.model_version, node.history, cache=self.cache, tactics=self.tactics)
        try:
            checked(native.hxg_begin(oracle.ptr, 1, 1))
            rid, _ = oracle.request()
            if rid > 0:
                oracle.fulfill(rid, prediction)
            result = oracle.result(0., 0., 0, 0)
            if rid <= 0:
                actions = np.asarray(prediction['actions'])
                logits, values = (np.asarray(prediction[k], np.float64) for k in ('logits', 'q'))
                if actions.dtype.kind not in 'iu':
                    raise ValueError('Evaluator coordinates must be integers within +/- 10^12')
                if (not np.array_equal(actions, result['actions']) or logits.shape != (len(result['actions']),)
                        or values.shape != logits.shape or not np.isfinite(logits).all()
                        or not np.isfinite(values).all() or (np.abs(values) > 1).any()):
                    raise ValueError('Invalid evaluation')
                prediction = dict(prediction, logits=logits, q=values)
        finally:
            oracle.close()
        node.actions = result['actions']
        node.prior = np.exp(prediction['logits']-np.max(prediction['logits']))
        node.prior /= node.prior.sum()
        node.value = float(np.dot(node.prior, prediction['q']))
        node.visits = np.zeros(len(node.actions), np.int64)
        node.sums = np.zeros(len(node.actions))
        node.eligible = result['policy'] > 0
        node.prior *= node.eligible
        node.prior /= node.prior.sum()
        node.expanded, node.winner = True, result['exact_winner']
        node.exact_action = result['action']
        if node.winner >= 0:
            for parent, _ in node.parents:
                parent.settle()
        value = node.value if node.winner < 0 else 1. if node.winner == node.player else -1.
        self.backup(node, path, value)

    def backup(self, child, path, value):
        for parent, i in reversed(path):
            if parent.player != child.player:
                value = -value
            parent.visits[i] += 1
            parent.sums[i] += value
            parent.settle()
            if parent.winner >= 0:
                value = 1. if parent.winner == parent.player else -1.
            child = parent
        if path:
            self.completed += 1

    def result(self):
        node = self.root
        score = np.where(node.eligible, node.visits + node.prior*.5, -np.inf)
        action = node.actions[int(np.argmax(score))].tolist()
        if node.winner >= 0 and node.exact_action is not None and not node.visits.any():
            action = node.exact_action
        return dict(action=action, actions=node.actions, visits=node.visits.copy(), values=node.q(),
                    completed=self.completed, proven=0 if node.winner < 0 else 1 if node.winner == node.player else -1)


def search_many(searches, simulations=64):
    """One outstanding leaf per tree, batched across games; cached expansions still consume simulations."""
    searches = list(searches)
    for search in searches:
        search.begin(simulations)
    while any(not s.done() for s in searches):
        pending = {}
        for search in searches:
            if search.done():
                continue
            request = search.request()
            if request is None:
                continue
            node, _ = request
            key = search.cache.key(node.history, search.model_version)
            prediction = search.cache.get(key)
            if prediction is None:
                pending.setdefault(key, []).append((search, request))
            else:
                search.fulfill(request, prediction)
        if pending:
            groups = list(pending.values())
            predictions = searches[0].evaluator.evaluate([group[0][1][0].history for group in groups])
            if len(predictions) != len(groups):
                raise ValueError('Evaluator returned the wrong batch size')
            for group, prediction in zip(groups, predictions):
                for search, request in group:
                    search.fulfill(request, prediction)
                    search.cache.put(search.cache.key(request[0].history, search.model_version), prediction)
    return [s.result() for s in searches]
