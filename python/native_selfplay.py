"""Played-root self-play through persistent native graph owners and one GPU queue."""
from collections import deque
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
    returned final fresh work. Dynamic mode attaches frozen checkpoints while
    older games continue; fixed cohorts retain their original model set.
    """
    def __init__(self, games, *, producers=4, quantum=64, views=8, depth=8, cache=8192,
                 batch_size=128, slice_ms=8, proof_workers=0, proof_package=None, ms=0, progress=None,
                 dynamic=False):
        if not games or any(not g.native_owner for g in games) or producers<1:
            raise ValueError('A nonempty cohort of native-owner games is required')
        self.games = list(games)
        models = [model for g in games for model in g.trees]
        self.models = list({m.sha:m for m in models}.values()) if dynamic else list(dict.fromkeys(models))
        if len(self.models)>16:
            raise ValueError('At most sixteen frozen models can share a native service')
        self.slot_models = [set(g.trees) for g in games]
        self.views, self.ms, self.progress = views, ms, progress
        self.pools, self.mapping, self.lookup, self.proof_loops = [], {}, {}, []
        self.epochs, self.finished, self.closing, self.warming = {}, set(), {}, {}
        self.service, self.receipt = None, None
        self.dynamic = dynamic
        self.groups, self.idle, self.waiting, self.unbound = {}, {}, set(), set()
        self.archived_proofs, self.archived_stats = deque(maxlen=512), deque(maxlen=512)
        self.options = dict(quantum=quantum,views=views,depth=depth,work=quantum,cache=cache,
                            workers=max(1,producers//len(self.models)))
        self.proof_options = dict(package=proof_package,workers=proof_workers,
                                  queue=max(4,proof_workers*2),slice_ms=slice_ms,table_mb=4)
        if dynamic:
            try:
                initial = list(self.models)
                self.models = []
                for model in initial:
                    self.register(model, initial=True)
                self.service = InferenceService(self.pools,[m.evaluator for m in self.models],batch_size=batch_size)
                self.service.start(continuous=True)
                for key in self.idle:
                    self.service.release(*key,expected=0)
                for index in range(len(games)):
                    self.next_root(index)
            except BaseException:
                self.close()
                raise
            return
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

    def register(self, model, *, initial=False):
        """One frozen producer per model, with a reusable graph slot for each game."""
        if model.sha in self.groups:
            return True
        if sum(pool is not None for pool in self.pools)>=16:
            return False
        sources, placeholders = [], {}
        for index,game in enumerate(self.games):
            source = next((tree for m,tree in game.trees.items() if initial and m.sha==model.sha),None)
            if source is None:
                s = game.settings
                source = model.tree([],game.seed+index,s.tactics,s.search_graph,s.q_range_floor,s.game_graph)
                placeholders[index] = source
            sources.append(source)
        pool = SearchPool(sources,seed=self.games[0].seed,**self.options)
        try:
            proof = pool.enable_proofs(**self.proof_options) if self.proof_options['workers'] else None
            if self.service is None:
                producer,model_id = len(self.pools),len(self.models)
                self.pools.append(pool);self.proof_loops.append(proof);self.models.append(model)
            else:
                producer,model_id = self.service.attach(pool,model.evaluator)
                while len(self.pools)<=producer:
                    self.pools.append(None);self.proof_loops.append(None)
                self.pools[producer],self.proof_loops[producer] = pool,proof
                while len(self.models)<=model_id:
                    self.models.append(None)
                self.models[model_id] = model
            self.groups[model.sha] = dict(model=model,producer=producer,model_id=model_id,free=set())
            for index,game in enumerate(self.games):
                key = producer,index
                self.epochs[key] = 0
                if index in placeholders:
                    self.lookup[key] = None
                    self.idle[key] = placeholders[index]
                    if self.service is not None:
                        self.service.release(*key,expected=0)
                else:
                    actual = next(m for m in game.trees if m.sha==model.sha)
                    self.mapping[index,actual] = key
                    self.lookup[key] = index,actual
            return True
        except BaseException:
            if getattr(pool,'_service',None) is None:
                pool.close()
                for tree in placeholders.values():
                    tree.close()
            raise

    def unbind(self):
        for index in self.finished-self.unbound:
            if any(tree.ptr for tree in self.games[index].trees.values()):
                continue  # The publisher still owns these caller wrappers.
            for model in self.slot_models[index]:
                key = self.mapping.pop((index,model))
                self.lookup[key] = None
                self.groups[model.sha]['free'].add(key[1])
            self.unbound.add(index)

    def maintain(self):
        """Retire unused weights only after graph retirement and device work finish."""
        self.unbind()
        needed = {m.sha for i,g in enumerate(self.games) if i not in self.finished for m in g.trees}
        for sha,group in list(self.groups.items()):
            producer,model_id = group['producer'],group['model_id']
            if sha in needed or len(group['free'])!=len(self.games):
                continue
            self.service.detach(producer)
            loop = self.proof_loops[producer]
            if loop:
                self.archived_stats.append(dict(loop.stats(),model=sha))
                self.archived_proofs.extend(dict(r,model=sha) for r in loop.records())
            self.pools[producer].close()
            self.pools[producer],self.proof_loops[producer] = None,None
            del self.groups[sha]
        for model_id,model in enumerate(self.models):
            if model is not None and model.sha not in self.groups and not self.service.model_pending(model_id):
                if model.evaluator.graph is not None:
                    model.evaluator.graph.close()
                    model.evaluator.graph = None
                self.models[model_id] = None
                self.service.models[model_id] = None
        for index in sorted(self.waiting):
            game = self.games[index]
            if not all(self.register(m) for m in game.trees):
                continue
            if not all(index in self.groups[m.sha]['free'] for m in game.trees):
                continue
            self.waiting.remove(index)
            self.warming[index] = set()
            for k,model in enumerate(game.trees):
                group = self.groups[model.sha]
                producer,owner = group['producer'],index
                key = producer,owner
                group['free'].remove(owner)
                self.mapping[index,model] = key
                self.lookup[key] = index,model
                self.warming[index].add(key)
                self.service.replace(*key,game.trees[model],expected=self.epochs[key],
                                     samples=game.samples,views=self.views,seed=game.seed+k)

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
        if index not in self.finished or not game.native_owner or (not self.dynamic and set(game.trees)!=self.slot_models[index]):
            raise ValueError('Replacement needs a retired slot and its frozen model set')
        if any(tree.ptr for tree in self.games[index].trees.values()):
            raise ValueError('Publish and close the retired game before replacing its slot')
        if self.dynamic:
            self.unbind()
            self.games[index] = game
            self.slot_models[index] = set(game.trees)
            self.finished.remove(index)
            self.unbound.discard(index)
            self.waiting.add(index)
            self.maintain()
            return
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
            if key in self.idle:
                group = next(g for g in self.groups.values() if g['producer']==key[0])
                if event.get('kind')!='released' or event['token']!=self.epochs[key]+1 or event['model']!=group['model_id']:
                    raise ValueError('Unexpected idle native graph retirement')
                self.epochs[key] = event['token']
                self.idle.pop(key).close()
                group['free'].add(key[1])
                continue
            index,model = self.lookup[key]
            game = self.games[index]
            model_id = self.service.model_versions.index(model.sha)
            if event['token']!=self.epochs[key]+1 or event['model']!=model_id:
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
        if self.dynamic:
            self.maintain()
        return finished

    def close(self):
        if self.service is not None:
            self.service.close()
            if self.receipt is None:
                self.receipt = dict(inference=self.service.stats(),
                    proofs=[*self.archived_proofs,*[dict(r,producer=p) for p,loop in enumerate(self.proof_loops) if loop for r in loop.records()]],
                    proof_stats=[*self.archived_stats,*[dict(loop.stats(),producer=p) for p,loop in enumerate(self.proof_loops) if loop]])
        for pool in reversed(self.pools):
            if pool is not None:
                pool.close()
        for tree in self.idle.values():
            tree.close()
        self.idle.clear()


class ActorEngine:
    """Actor publication and source draws around continuously owned native games."""
    def __init__(self, settings):
        self.settings, self.leaf_batch = settings,settings.leaf_batch
        self.slots, self.free, self.engine = [],deque(),None
        self.evals = self.calls = self.full_calls = self.searches = 0
        self.solver = None
        self.paused, self.captures = False,[]
        self.accounted = (0,0,0)
        self.fresh_nodes = self.queries = self.missing_fresh = 0

    @property
    def closing(self):
        return self.engine.closing if self.engine else {}

    def add(self, game):
        if self.engine is not None:
            self.engine.replace(self.free.popleft(),game)
        self.slots.append(game)

    def progress(self, index, event):
        self.searches += 'error' not in event

    def account(self):
        service = self.engine.service
        current = (service.stats()['launched_rows'],service.calls,service.full_calls)
        a,b,c = (x-y for x,y in zip(current,self.accounted))
        self.evals += a;self.calls += b;self.full_calls += c
        self.accounted = current

    def summary(self):
        return dict(inference=self.engine.service.stats() if self.engine else None,
                    producers=sum(p is not None for p in self.engine.pools) if self.engine else 0,
                    waiting_games=len(self.engine.waiting) if self.engine else 0,
                    retired_fresh_nodes=self.fresh_nodes,retired_queries=self.queries,
                    retired_missing_fresh=self.missing_fresh)

    def step(self):
        if self.paused:
            raise ValueError('Resume the native actor before stepping')
        if self.engine is None:
            s = self.settings
            self.engine = NativeGames(self.slots,dynamic=True,producers=s.native_producers,
                quantum=s.native_quantum,views=s.native_views,depth=s.native_depth,
                cache=s.cache_positions,batch_size=s.leaf_batch,proof_workers=s.native_proof_workers,
                slice_ms=s.native_proof_slice_ms,progress=self.progress)
        finished = self.engine.step()
        self.account()
        for index,game in finished:
            self.slots.remove(game)
            self.free.append(index)
            for row in game.rows:
                source = row.get('search',{})
                self.fresh_nodes += source.get('fresh_nodes',0)
                self.queries += source.get('queries',0)
                self.missing_fresh += source.get('missing_fresh',0)
        return [game for _,game in finished]

    def synchronize_inflight(self):
        if self.engine is not None:
            self.engine.pause()
            self.account()
            for evaluator in self.engine.service.models:
                if evaluator is not None and evaluator.graph is not None:
                    evaluator.graph.close()
                    evaluator.graph = None
                    self.captures.append(evaluator)
        self.paused = True

    def resume(self):
        if self.engine is not None:
            from hexnet_graphs import ActorGraph
            for evaluator in self.captures:
                # A retired checkpoint may have dropped out while the phase waited.
                if evaluator in self.engine.service.models:
                    evaluator.graph = ActorGraph(evaluator.model)
            self.captures.clear()
            self.engine.resume()
        self.paused = False

    def drain(self):
        if self.engine is not None:
            self.engine.close()
            self.account()

    def close(self):
        if self.engine is not None:
            self.engine.close()
        # Source wrappers can be destroyed only after their native owners join.
        games = self.engine.games if self.engine else self.slots
        for game in games:
            game.game.close()
            for tree in game.trees.values():
                tree.close()
        self.captures.clear()


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
