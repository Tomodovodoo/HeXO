"""Replenishable fixed-checkpoint self-play through continuous native graph owners.

This writes ordinary played-root rows. It does not change the actor daemon,
checkpoint rotation, learner weights or value-target construction.
"""
import numpy as np
from native_scheduler import SearchPool, InferenceService
from dense_selfplay import record_network_values


def label_prefixes(game, prefixes):
    """Apply exact evidence to its played position, retaining the tightest witnesses."""
    rows = {r['ply']:r for r in game.rows}
    pending = list(prefixes)
    while pending:
        ply, exact, distance, witnesses = pending.pop()
        row = rows.get(ply)
        if row is None:
            continue
        proven = 1 if row['player']==exact else -1
        if row.get('proven') and row['proven']!=proven:
            raise ValueError('Graph proof contradicts an earlier exact played-root label')
        old = max(0, row.get('proof_plies', 0))
        tighter = distance>0 and (not old or distance<old)
        if proven>0:
            if tighter:
                row.pop('proof_action', None)
            if witnesses and (tighter or distance==old or (not old and distance<=0)):
                row['proof_action'] = witnesses
        else:
            row.pop('proof_action', None)
        # Graph bounds count placements, not certificate turns.
        if not row.get('proven'):
            row.update(proven=proven, proof_turns=0)
        if distance>0:
            row['proof_plies'] = min(old, distance) if old else distance
        elif not old:
            row.pop('proof_plies', None)
        previous = rows.get(ply-1)
        if previous is not None and previous['player']==exact:
            # The winner can choose the played stone leading to this exact
            # position. This does not prove a loss across unexamined replies.
            bound = row.get('proof_plies',0)
            pending.append((ply-1,exact,bound+1 if bound>0 else 0,[game.moves[ply-1]]))


class NativeGames:
    """Replenishable fixed-model slots with native graph ownership and proof retirement.

    `step` exposes a finished game only after every one of its model owners has
    returned final fresh work. `replace` requires the slot's original model set.
    Checkpoint rotation and actor daemon publication are separate integration.
    """
    def __init__(self, games, *, producers=4, quantum=64, views=8, depth=8, cache=8192,
                 batch_size=128, slice_ms=8, proof_workers=0, proof_package=None, ms=0, progress=None):
        if not games or any(not g.native_owner for g in games) or producers<1:
            raise ValueError('A nonempty cohort of native-owner games is required')
        self.games = list(games)
        self.models = list(dict.fromkeys(model for g in games for model in g.trees))
        self.slot_models = [set(g.trees) for g in games]
        self.views, self.ms, self.progress = views, ms, progress
        self.pools, self.mapping, self.lookup, self.proof_loops = [], {}, {}, []
        self.epochs, self.finished, self.closing, self.warming = {}, set(), {}, {}
        self.service, self.receipt = None, None
        members = [[(i,g.trees[m]) for i,g in enumerate(games) if m in g.trees] for m in self.models]
        if len(members)>16:
            raise ValueError('At most sixteen frozen models can share a native service')
        # One producer per model, then distribute the remaining host budget to
        # the models serving the most games. Do not multiply it by model count.
        counts = [1]*len(members)
        for _ in range(min(16,max(producers,len(members)))-len(members)):
            choices = [i for i in range(len(members)) if counts[i]<len(members[i])]
            if not choices:
                break
            i = max(choices,key=lambda i:len(members[i])/counts[i])
            counts[i] += 1
        try:
            for model, group, count in zip(self.models,members,counts):
                for start in range(count):
                    assigned = group[start::count]
                    pool = SearchPool([tree for _,tree in assigned],quantum=quantum,views=views,
                                      depth=depth,work=quantum,cache=cache,seed=games[assigned[0][0]].seed)
                    producer = len(self.pools)
                    self.pools.append(pool)
                    for owner,(slot,_) in enumerate(assigned):
                        self.mapping[slot,model] = producer,owner
                        self.lookup[producer,owner] = slot,model
                        self.epochs[producer,owner] = 0
                    self.proof_loops.append(pool.enable_proofs(proof_package,workers=proof_workers,
                                            queue=max(4,proof_workers*2),slice_ms=slice_ms,table_mb=4)
                                            if proof_workers else None)
            self.service = InferenceService(self.pools,[m.evaluator for m in self.models],batch_size=batch_size)
            self.service.start(continuous=True)
            for index in range(len(games)):
                self.next_root(index)
        except BaseException:
            self.close()
            raise

    @property
    def done(self):
        return len(self.finished)==len(self.games)

    def next_root(self, index):
        game = self.games[index]
        producer,owner = self.mapping[index,game.model]
        self.service.retarget(producer,owner,game.moves,expected=self.epochs[producer,owner],
                              work=0 if self.ms and game.is_full else game.budget,
                              ms=self.ms if game.is_full else 0,samples=game.samples,
                              views=self.views if game.is_full else 1,
                              noise=game.settings.root_noise if game.is_full else 0.)

    def retire(self, index):
        keys = {self.mapping[index,m] for m in self.slot_models[index]}
        self.closing[index] = keys
        for producer,owner in keys:
            self.service.release(producer,owner,expected=self.epochs[producer,owner])

    def replace(self, index, game):
        """Start a fresh game in a fully retired slot, retaining model predictions."""
        if index not in self.finished or not game.native_owner or set(game.trees)!=self.slot_models[index]:
            raise ValueError('Replacement needs a retired slot and its frozen model set')
        self.games[index] = game
        self.finished.remove(index)
        self.warming[index] = {self.mapping[index,m] for m in game.trees}
        for k,model in enumerate(game.trees):
            producer,owner = self.mapping[index,model]
            self.service.replace(producer,owner,game.trees[model],expected=self.epochs[producer,owner],
                                 samples=game.samples,views=self.views,seed=game.seed+k)

    def pause(self, **options):
        self.service.pause(**options)

    def resume(self):
        self.service.resume()

    def step(self):
        """Pump bulk inference and consume immutable placement/lifecycle events."""
        self.service.pump()
        finished = []
        while (event:=self.service.event()) is not None:
            key = event['producer'],event['game']
            index,model = self.lookup[key]
            game = self.games[index]
            if event['token']!=self.epochs[key]+1 or event['model']!=self.models.index(model):
                raise ValueError('Native completion does not match its model/root generation')
            self.epochs[key] = event['token']
            if event.get('kind')=='released':
                if index not in self.closing or key not in self.closing[index]:
                    raise ValueError('Unexpected native game retirement')
                effort = {generation:dict(fresh_nodes=fresh,queries=queries,missing_fresh=missing)
                          for generation,fresh,queries,missing in event['effort']}
                for row in game.rows:
                    if 'search' in row and game.sides[row['player']] is model:
                        source = row['search']
                        spent = effort.get(source['solver_generation'],dict(fresh_nodes=0,queries=0,missing_fresh=0))
                        source.update(spent)
                        row['solver_nodes'],row['solver_budget'] = spent['fresh_nodes'],0
                self.closing[index].remove(key)
                if not self.closing[index]:
                    del self.closing[index]
                    self.finished.add(index)
                    finished.append((index,game))
                continue
            if event.get('kind')=='replaced':
                if index not in self.warming or key not in self.warming[index]:
                    raise ValueError('Unexpected native graph replacement')
                self.warming[index].remove(key)
                if not self.warming[index]:
                    del self.warming[index]
                    self.next_root(index)
                continue
            if (index in self.finished or index in self.closing or index in self.warming
                    or model is not game.model or event['history']!=game.moves):
                raise ValueError('Native completion does not match the active game/position')
            if 'error' in event:
                game.reason = event['error']
                self.retire(index)
                if self.progress:
                    self.progress(index,event)
                continue
            edges = np.asarray(event.pop('edges'),np.float64)
            player,winner = game.game.player,event['exact_winner']
            proven = 0 if winner<0 else 1 if winner==player else -1
            result = dict(event,actions=edges[:,:2].astype(np.int64),visits=edges[:,6].astype(np.int64),
                          completed_q=edges[:,3],values=edges[:,4],policy=edges[:,5],
                          completed=event['root_completed'],proven=proven,proof_turns=0,
                          solver_nodes=0,solver_budget=0,
                          proof_action=edges[edges[:,8].astype(bool),:2].astype(np.int64).tolist() if proven>0 else [])
            result['search'] = dict(source='native-root',model=model.sha,context=event['context'],
                                    token=event['token'],comparison_credits=int(edges[:,7].sum()),
                                    direct_max=int(edges[:,7].max()),shared_visits=int(edges[:,6].sum()),
                                    root_completed=event['root_completed'],completed=event['completed'],
                                    root_estimate=None,solver_generation=event['solver_generation'])
            more = game.searched(result)
            result['search']['root_estimate'] = game.values[len(event['history'])]
            prefixes = event['exact_prefixes']
            if winner>=0:
                prefixes = [*prefixes,[len(event['history']),winner,event['proof_plies'],result['proof_action']]]
            label_prefixes(game,prefixes)
            if self.progress:
                self.progress(index,event)
            if more:
                self.next_root(index)
            else:
                self.retire(index)
        return finished

    def close(self):
        if self.service is not None:
            self.service.close()
            if self.receipt is None:
                self.receipt = dict(inference=self.service.stats(),
                    proofs=[dict(r,producer=p) for p,loop in enumerate(self.proof_loops) if loop for r in loop.records()],
                    proof_stats=[dict(loop.stats(),producer=p) for p,loop in enumerate(self.proof_loops) if loop])
        for pool in reversed(self.pools):
            pool.close()


def play_stream(games, next_game=None, **options):
    """Complete games and refill retired slots via `next_game(slot, finished_game)`.

    Returning None leaves a slot retired. The factory returns fresh native-owner
    SelfPlayGames with that slot's frozen model set. NativeGames exposes the same
    lifecycle incrementally for a daemon that publishes bounded shards.
    """
    engine = NativeGames(games,**options)
    finished = []
    try:
        while not engine.done:
            for index,game in engine.step():
                record_network_values([game])
                episode,rows = game.episode()
                finished.append((index,episode,rows))
                if next_game is not None:
                    replacement = next_game(index,game)
                    if replacement is not None:
                        engine.replace(index,replacement)
    finally:
        engine.close()
    if next_game is None:
        finished.sort(key=lambda item:item[0])
    episodes,rows = [],[]
    for _,episode,played in finished:
        rows.extend(dict(row,game=len(episodes)) for row in played)
        episodes.append(episode)
    return episodes,rows,engine.receipt


def play_cohort(games, **options):
    """Complete a fixed cohort through the same replenishable native lifecycle."""
    return play_stream(games,**options)
