"""Sparring: the actor's model plays games against an external Six-protocol engine, published as actor shards.

Games. `games` games are in flight. Each draws the trained side uniformly and the external engine's positions per
complete turn (go nodes) uniformly from `nodes`, and with probability `book_fraction` starts from an opening of the
live book's training openings (dense_selfplay.BookStarts, without the actors' training-start pool, whose positions
come from the evaluator's games against Six), played in a random orientation as a forced prefix without rows, the
episode recording origin 'book' and `book` as actor book games do. The trained model searches every one of its
placements for `sims` simulations with `root_samples` (a transposition graph and the actors' tactics), the games'
searches batched together, and plays the highest improved policy, except that its first exponential(`random_plies`)
own placements are sampled from that policy, so games against a near-deterministic engine differ. The engine
answers each of its turns in one process, one game at a time, while the model searches. A game ends at six in a
row, at max_plies (cap), or when a placement or one of the model's searches no longer fits the largest crop (span);
a game the engine fails is dropped. Sparring starts once the run has a checkpoint.

Output. Every `shard_games` finished games form a shard of origin 'actor' (dense_data) whose episodes record the
engine as `opponent` (`name`). The model's placements are full searches with their improved policy. The engine's
plies have a null root value and full_search False. With `imitate` they are policy targets too: one-hot on the
engine's placement (search source 'engine' with its node budget), and the episode records no `trained_side`, so
both colours' rows enter the replay window and the value targets of the engine's positions come from the result.
Without `imitate` the episode records the model's `trained_side` and the engine's rows have no policy, so only the
model's placements train. The episode's `sparring` records the engine's node budget. The model follows the actors' model_source, and the service
pauses while a phased learner trains (dense_selfplay.Yield).
<run>/sparring-status.json records counters.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import time

import numpy as np

import dense_config
import dense_data
from dense_selfplay import BookStarts, Yield, policy_target, resolve, root_value, shard_name
import hexcrop
from hexo import Game
import hexnet
from legacy.klent import digest
from legacy.train import write_json
from neural_search import EvaluationCache, NeuralSearch, SearchCoordinator
from six_engine import ProtocolError, SixEngine


@dataclass(frozen=True)
class SparringSettings:
    engine: str = ''           # Six-protocol command line of the opponent
    name: str = 'six'          # its `opponent` label in episodes
    nodes: str = '240,480,960'  # the engine's positions per complete turn, one drawn per game
    sims: int = 64
    root_samples: int = 16
    random_plies: float = 3.   # mean number of the model's own placements sampled from its improved policy
    book_fraction: float = .75
    games: int = 16            # games in flight
    shard_games: int = 32
    seed: int = 0              # 0 draws fresh entropy
    imitate: bool = True       # the engine's placements are one-hot policy targets

    def budgets(self):
        return [int(n) for n in self.nodes.split(',')]

    def __post_init__(self):
        if not self.engine or not self.name or self.sims < 1 or not 1 <= self.root_samples <= 1024 \
                or self.random_plies < 0 or not 0 <= self.book_fraction <= 1 or self.games < 1 or self.shard_games < 1 \
                or not self.budgets() or min(self.budgets()) < 1:
            raise ValueError('Invalid sparring settings')


class Model:
    def __init__(self, run, config):
        self.checkpoint, path = resolve(run, source=config.actor.model_source, variant=config.learner.variant)
        net = hexnet.load_model(path)
        net.set_kernels(config.actor.net_kernels)
        self.sha, self.config = digest(path), net.config
        self.evaluator = hexnet.DenseEvaluator(net, config.device, self.sha, max_batch=config.actor.leaf_batch)
        self.cache = EvaluationCache(config.actor.cache_positions)


class Match:
    """One game between `model` (colour `side`) and the engine at `nodes` per turn."""

    def __init__(self, model, side, nodes, sims, random_plies, seed, max_plies, book=None, imitate=False):
        self.model, self.side, self.nodes, self.sims, self.seed, self.max_plies = model, side, nodes, sims, seed, max_plies
        self.imitate = imitate
        self.rng = np.random.default_rng(seed)
        self.random_plies = int(round(self.rng.exponential(random_plies))) if random_plies > 0 else 0
        self.book, forced = (None, []) if book is None else (book[0], [[int(q), int(r)] for q, r in book[1]])
        self.forced, self.sampled, self.opening_plies = len(forced), 0, len(forced)
        self.game, self.moves, self.rows, self.reason = Game([tuple(m) for m in forced]), forced, [], None
        self.values, self.full = [None]*len(forced), [False]*len(forced)

    @property
    def ours(self):
        return self.game.player == self.side

    def fits(self):
        """Whether the position fits the largest crop; ends the game as 'span' otherwise."""
        try:
            hexcrop.encode_game(self.game, np.asarray(self.moves, np.int64).reshape(-1, 2))
            return True
        except hexcrop.SpanError:
            self.reason = 'span'
            return False

    def play(self, q, r):
        self.game.play(q, r)
        self.moves.append([q, r])
        if self.game.winner < 0 and len(self.moves) >= self.max_plies:
            self.reason = 'cap'
        elif self.game.winner < 0:
            self.fits()  # a placement beyond the largest crop ends the game before the next one

    @property
    def over(self):
        return self.game.winner >= 0 or self.reason is not None

    def searched(self, result):
        """Record and play the model's search `result` at the current placement."""
        ply, player, actions = len(self.moves), self.game.player, result['actions']
        row = dict(ply=ply, player=player, remaining=self.game.remaining, legal_sha256=dense_data.legal_digest(actions),
                   policy=None, search=dict(source='sparring', requested_sims=self.sims,
                                            completed=int(result['completed'])))
        if policy_target(result, player):
            row['policy'] = result['policy'].astype(np.float32)
        if result.get('proven'):
            row.update(proven=int(result['proven']), proof_turns=0, proof_plies=result.get('proof_plies', 0))
            if result['proven'] > 0 and result.get('proof_action'):
                row['proof_action'] = result['proof_action']
        self.rows.append(row)
        self.values.append(root_value(result, player))
        self.full.append(True)
        policy = result['policy']
        if self.sampled < self.random_plies and not result.get('proven') and policy.sum() > 0:
            action = actions[self.rng.choice(len(policy), p=policy/policy.sum())]
            self.sampled += 1
            self.opening_plies = ply+1
        else:
            action = result['action']
        self.play(int(action[0]), int(action[1]))

    def answered(self, turn):
        """Record and play the engine's placements `turn`."""
        for q, r in turn:
            if self.over:
                break
            history = np.asarray(self.moves, np.int64).reshape(-1, 2)
            legal = hexcrop.legal_array(self.game, history)
            row = dict(ply=len(self.moves), player=self.game.player, remaining=self.game.remaining,
                       legal_sha256=dense_data.legal_digest(legal), policy=None)
            if self.imitate:
                hit = np.flatnonzero((legal == (q, r)).all(1))
                if len(hit) == 1:
                    row['policy'] = np.zeros(len(legal), np.float32)
                    row['policy'][hit[0]] = 1.
                    row['search'] = dict(source='engine', nodes=self.nodes)
            self.rows.append(row)
            self.values.append(None)
            self.full.append(False)
            self.play(int(q), int(r))

    def episode(self, name):
        winner = self.game.winner
        self.game.close()
        sha = self.model.sha
        episode = dict(moves=self.moves, winner=winner, reason=self.reason or ('six-in-a-row' if winner >= 0 else 'cap'),
                    opening_plies=self.opening_plies, actor=sha,
                    actors={str(self.side): sha, str(1-self.side): name}, opponent=name,
                    trained_side=None if self.imitate else self.side,
                    root_values=self.values, network_values=[None]*len(self.moves), full_search=self.full,
                    sparring=dict(nodes=self.nodes, sims=self.sims), origin='book' if self.book else 'selfplay')
        if self.book:
            episode['book'] = self.book
        return episode, self.rows


class Sparring:
    def __init__(self, run, settings, config):
        self.run, self.s, self.config = Path(run), settings, config
        self.path = self.run/'sparring-status.json'
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.stats = dict(dict(games=0, wins=0, losses=0, capped=0, dropped=0, shards=0, rows=0), **state.get('stats', {}))
        self.rng = np.random.default_rng(settings.seed or None)
        self.model, self.last_shard, self.finished = None, '', []
        self.books = BookStarts(self.run, config.actor.max_plies)
        self.engine = SixEngine(shlex.split(settings.engine), timeout=60., startup=900.)

    def save(self, stage):
        write_json(self.path, dict(stage=stage, updated_at=time.time(), settings=asdict(self.s),
                                   checkpoint=self.model and self.model.checkpoint, stats=self.stats))

    def refresh_model(self):
        """Follow the actors' model; False while the run has no checkpoint to play (sparring starts with the first)."""
        checkpoint, path = resolve(self.run, source=self.config.actor.model_source, variant=self.config.learner.variant)
        if path is None:
            return False
        if self.model is None or self.model.checkpoint != checkpoint:
            self.model = Model(self.run, self.config)
        return True

    def start(self):
        book = self.books.draw(self.rng) if self.rng.random() < self.s.book_fraction else None
        return Match(self.model, int(self.rng.integers(2)), int(self.rng.choice(self.s.budgets())), self.s.sims,
                     self.s.random_plies, int(self.rng.integers(2**31)), self.config.actor.max_plies, book, self.s.imitate)

    def answer(self, matches):
        """The engine's turn in each of `matches`, in order; None for a game the engine failed."""
        turns = []
        for match in matches:
            try:
                turns.append(self.engine(match.game, nodes=match.nodes))
            except (ProtocolError, ValueError):
                turns.append(None)
        return turns

    def search(self, matches):
        """The model's search results at the current placement of each of `matches` (one shared model)."""
        model, actor = matches[0].model, self.config.actor
        trees = [NeuralSearch(model.evaluator, model.sha, [tuple(m) for m in match.moves], match.seed+len(match.moves),
                              model.cache, actor.tactics, graph=True) for match in matches]
        try:
            return SearchCoordinator(model.evaluator, model.sha, model.cache).search_many(
                trees, self.s.sims, self.s.root_samples, actor.leaf_batch)
        finally:
            for tree in trees:
                tree.close()

    def searched(self, matches):
        """Search and play the current placement of every match in `matches`. A search reaching a leaf beyond the
        largest crop fails the batch; the matches are then searched one at a time and the failing ones end as
        'span'."""
        try:
            results = self.search(matches)
        except hexcrop.SpanError:
            if len(matches) == 1:
                matches[0].reason = 'span'
                return
            for match in matches:
                self.searched([match])
            return
        for match, result in zip(matches, results):
            match.searched(result)

    def finish(self, match):
        episode, rows = match.episode(self.s.name)
        self.stats['games'] += 1
        if episode['winner'] < 0:
            self.stats['capped'] += 1
        else:
            self.stats['wins' if episode['winner'] == match.side else 'losses'] += 1
        self.finished.append((match.model, episode, rows))
        while len(self.finished) >= self.s.shard_games:
            self.publish()

    def publish(self):
        """Publish the oldest `shard_games` finished games, one shard per trained model among them."""
        batch, self.finished = self.finished[:self.s.shard_games], self.finished[self.s.shard_games:]
        for sha in dict.fromkeys(m.sha for m, _, _ in batch):
            group = [b for b in batch if b[0].sha == sha]
            model, episodes, rows = group[0][0], [], []
            for _, episode, game_rows in group:
                rows += [dict(row, game=len(episodes)) for row in game_rows]
                episodes.append(episode)
            name = self.last_shard = shard_name(self.last_shard)
            identity = dict(actor_sha256=model.sha, actors=[model.sha], opponents=[self.s.name],
                            checkpoint=model.checkpoint, pid=os.getpid(), model=hexnet.saved_config(model.config),
                            sparring=asdict(self.s), value_targets='not stored; derive from episode root_values and winner')
            dense_data.write_shard(self.run/'shards'/name, identity, episodes, rows, 'actor')
            self.stats['shards'] += 1
            self.stats['rows'] += sum(r['policy'] is not None for r in rows)

    def loop(self):
        pause = Yield(self.run, 1., 0., 1., 2., follow=True)
        matches, engine = [], ThreadPoolExecutor(1)
        while True:
            while pause.paused():
                self.save('paused')
                time.sleep(2.)
            if not self.refresh_model():
                self.save('waiting-for-checkpoint')
                time.sleep(30.)
                continue
            if len(matches) < self.s.games:
                self.books.load()
            matches += [self.start() for _ in range(self.s.games-len(matches))]
            theirs = [m for m in matches if not m.ours]
            pending = engine.submit(self.answer, theirs) if theirs else None
            ours = [m for m in matches if m.ours and not m.over]
            for model in {m.model for m in ours}:
                self.searched([m for m in ours if m.model is model])
            if pending is not None:
                for match, turn in zip(theirs, pending.result()):
                    if turn is None:
                        match.reason = 'dropped'
                    else:
                        match.answered(turn)
            for match in [m for m in matches if m.over]:
                matches.remove(match)
                if match.reason == 'dropped':
                    match.game.close()
                    self.stats['dropped'] += 1
                else:
                    self.finish(match)
            self.save('playing')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True)
    dense_config.add_arguments(parser, SparringSettings)
    args = parser.parse_args()
    config = dense_config.load(Path(args.run))
    values = {k: getattr(args, k) for k in SparringSettings.__dataclass_fields__ if getattr(args, k) is not None}
    Sparring(args.run, SparringSettings(**values), config).loop()


if __name__ == '__main__':
    main()
