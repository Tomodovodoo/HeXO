"""Targeted reanalysis: deep searches of the turns where a self-play loser last rated itself fine, published as
extra policy rows (MuZero Reanalyze, restricted to value collapses).

Selection. In each decisive own-model game of a new actor shard (validation games excepted), the loser's turn
starts t with a searched root value v_t are scanned. t is a collapse point when v_t is inexact (below 1, which
marks an exact win) and >= `high`, t is not a tactical book position (its manual result stays with the source),
and one of the loser's next `turns` turn starts has a value <= `low` (an exact loss counts); a game gives its latest
collapse point, the last turn the loser still thought it was fine. A shard gives at most `fraction` of its policy rows.

Search. The champion searches every point for `sims` simulations with `root_samples` (a transposition graph and
the actors' tactics), `batch` trees at a time, and pauses while a phased learner trains (dense_selfplay.Yield).

Output. Shard <source>1 of origin 'reanalysis' (dense_data): one episode per point holding the whole source game,
so its outcome and future targets are the source's, with root value and full_search set only at the point and
`reanalysis` {shard, game, ply, holdout_key, checkpoint, sims}; one row at the point with the deep search's improved
policy, and proven fields when the search proved the root; an empty proof sidecar. Episodes keep the source game's
validation key.
<run>/reanalysis-status.json records the shards done (newest RECENT kept) and counters.
"""
import argparse
from collections import namedtuple
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import dense_config
import dense_data
from dense_data import forced_plies, player_at
from dense_selfplay import Yield, policy_target, resolve, root_value
import hexnet
from legacy.klent import digest
from legacy.train import write_json
from neural_search import EvaluationCache, NeuralSearch, SearchCoordinator

RECENT = 20000
Champion = namedtuple('Champion', 'checkpoint sha evaluator cache')


@dataclass(frozen=True)
class ReanalysisSettings:
    sims: int = 1024
    root_samples: int = 16
    high: float = 0.        # the loser's root value at the point is at least this
    low: float = -.5        # and at one of its next `turns` turn starts at most this
    turns: int = 3
    fraction: float = .01   # points per shard, at most this share of its policy rows
    batch: int = 32         # trees searched together
    limit: int = 200        # newest shards considered each poll
    poll_seconds: float = 30.

    def __post_init__(self):
        if self.sims < 1 or not 1 <= self.root_samples <= 1024 or not -1 <= self.low < self.high < 1 \
                or self.turns < 1 or not 0 < self.fraction <= .1 or self.batch < 1 or self.limit < 1:
            raise ValueError('Invalid reanalysis settings')


def point(episode, settings, validation_fraction):
    """The game's latest collapse point (module contract), or None."""
    values = episode['root_values']
    if episode['winner'] < 0 or episode.get('opponent') is not None or values is None \
            or dense_data.holdout(episode, validation_fraction):
        return None
    loser = 1-episode['winner']
    book = episode.get('book') or {}
    labelled = book['ply'] if book.get('tactical') is not None else None  # keeps its manual result; not reanalysed
    starts = [(t, values[t]) for t in range(max(1, forced_plies(episode)), len(values))
              if t % 2 and player_at(t) == loser and values[t] is not None]
    found = None
    for i, (t, v) in enumerate(starts):
        later = [w for _, w in starts[i+1:i+1+settings.turns]]  # exact losses count as collapse evidence
        if settings.high <= v < 1 and t != labelled and later and min(later) <= settings.low:
            found = t
    return found


def settled(path):
    """Give a published shard an empty proof sidecar: the proof pass skips reanalysis shards, and readers stop
    looking for labels once the sidecar exists."""
    sidecar = path/dense_data.SIDECAR
    if not sidecar.exists():
        temporary = path/(dense_data.SIDECAR+'.tmp')
        temporary.write_text('', encoding='utf-8')
        os.replace(temporary, sidecar)


def seed(name, game, ply):
    return int.from_bytes(hashlib.sha256(f'{name}/{game}/{ply}'.encode()).digest()[:4], 'little') >> 1


def episode_of(source, name, game, ply, value, model, settings):
    T = len(source['moves'])
    roots, full = [None]*T, [False]*T
    roots[ply], full[ply] = value, True
    return dict(moves=source['moves'], winner=source['winner'], reason=source['reason'], opening_plies=ply,
                actor=model.sha, actors={'0': model.sha, '1': model.sha}, opponent=None, trained_side=None,
                root_values=roots, network_values=[None]*T, full_search=full, origin='reanalysis',
                reanalysis=dict(shard=name, game=game, ply=ply, holdout_key=dense_data.holdout_key(source),
                                checkpoint=model.checkpoint, sims=settings.sims))


def row_of(game, ply, result, model, settings):
    player = player_at(ply)
    row = dict(game=game, ply=ply, player=player, remaining=2, legal_sha256=dense_data.legal_digest(result['actions']),
               policy=result['policy'].astype(np.float32),
               search=dict(source='reanalysis', model=model.sha, requested_sims=settings.sims,
                           completed=int(result['completed'])))
    if result.get('proven'):
        row.update(proven=int(result['proven']), proof_turns=0, proof_plies=result.get('proof_plies', 0))
        if result['proven'] > 0 and result.get('proof_action'):
            row['proof_action'] = result['proof_action']
    return row


class Reanalysis:
    def __init__(self, run, settings, config):
        self.run, self.s, self.config = Path(run), settings, config
        self.path = self.run/'reanalysis-status.json'
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.done = list(state.get('done', []))
        self.stats = dict(dict(shards=0, points=0, rows=0, search_seconds=0.), **state.get('stats', {}))
        self.model, self.origins = None, {}

    def save(self, stage):
        write_json(self.path, dict(stage=stage, updated_at=time.time(), settings=asdict(self.s), done=self.done[-RECENT:],
                                   checkpoint=self.model and self.model.checkpoint, stats=self.stats))

    def pending(self):
        """Of the newest `limit` actor shards, those not done, newest first."""
        actors = []
        for p in dense_data.shard_dirs(self.run)[::-1]:
            if p.name not in self.origins:
                self.origins[p.name] = dense_data.origin(dense_data.manifest(p))
            if self.origins[p.name] == 'actor':
                actors.append(p.name)
                if len(actors) == self.s.limit:
                    break
        done = set(self.done)
        return [n for n in actors if n not in done]

    def refresh_model(self):
        checkpoint, path = resolve(self.run, source='champion', variant=self.config.learner.variant)
        if self.model is None or self.model.checkpoint != checkpoint:
            net, sha = hexnet.load_model(path), digest(path)
            net.set_kernels(self.config.actor.net_kernels)
            evaluator = hexnet.DenseEvaluator(net, self.config.device, sha, max_batch=self.config.actor.leaf_batch)
            self.model = Champion(checkpoint, sha, evaluator, EvaluationCache(self.config.actor.cache_positions))

    def shard(self, name):
        """Reanalyse shard `name`; the number of rows published. An output already published for it (the service
        stopped before recording the source as done) completes it as it is, whichever champion made it."""
        path = self.run/'shards'/(name+'1')
        if path.exists():
            published = dense_data.verify(path)
            if published['identity'].get('reanalysis_source') != name:
                raise ValueError(f'Reanalysis shard collision: {path}')
            settled(path)
            return 0
        source = dense_data.verify(self.run/'shards'/name)
        episodes = json.loads((self.run/'shards'/name/'episodes.json').read_text(encoding='utf-8'))
        cap = int(self.s.fraction*source['counts']['policy_rows'])
        found = [(g, t) for g, e in enumerate(episodes)
                 if (t := point(e, self.s, self.config.learner.validation_fraction)) is not None]
        found = found[::-1][:cap]  # later games of the shard first; any order is fine
        if not found:
            return 0
        self.refresh_model()
        model, out_episodes, rows, actor = self.model, [], [], self.config.actor
        started = time.monotonic()
        for k in range(0, len(found), self.s.batch):
            chunk = found[k:k+self.s.batch]
            trees = [NeuralSearch(model.evaluator, model.sha, [tuple(m) for m in episodes[g]['moves'][:t]], seed(name, g, t),
                                  model.cache, actor.tactics, graph=True) for g, t in chunk]
            try:
                results = SearchCoordinator(model.evaluator, model.sha, model.cache).search_many(
                    trees, self.s.sims, self.s.root_samples, actor.leaf_batch, choice='policy')
            finally:
                for tree in trees:
                    tree.close()
            for (g, t), result in zip(chunk, results):
                if not policy_target(result, player_at(t)):
                    continue
                out_episodes.append(episode_of(episodes[g], name, g, t, root_value(result, player_at(t)), model, self.s))
                rows.append(row_of(len(out_episodes)-1, t, result, model, self.s))
        self.stats['search_seconds'] += time.monotonic()-started
        self.stats['points'] += len(found)
        if not rows:
            return 0
        identity = dict(source['identity'], actor_sha256=model.sha, reanalysis_source=name,
                        reanalysis_source_files=source['files'], reanalysis_settings=asdict(self.s),
                        reanalysis_checkpoint=model.checkpoint)
        identity.pop('process', None)
        dense_data.write_shard(path, identity, out_episodes, rows, origin='reanalysis')
        settled(path)
        return len(rows)

    def loop(self):
        pause = Yield(self.run, 1., 0., 1., 2., follow=True)
        while True:
            names = self.pending()
            if not names:
                self.save('idle')
                time.sleep(self.s.poll_seconds)
                continue
            for name in names:
                while pause.paused():
                    self.save('paused')
                    time.sleep(2.)
                self.stats['rows'] += self.shard(name)
                self.stats['shards'] += 1
                self.done.append(name)
                self.save('reanalysing')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True)
    dense_config.add_arguments(parser, ReanalysisSettings)
    args = parser.parse_args()
    config = dense_config.load(Path(args.run))
    Reanalysis(args.run, dense_config.override(ReanalysisSettings(), args), config).loop()


if __name__ == '__main__':
    main()
