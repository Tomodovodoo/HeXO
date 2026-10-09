"""Hybrid-scheduler self-play: played roots through persistent native graph owners and one GPU queue."""
from collections import deque
import numpy as np
from hybrid_scheduler import SearchPool, InferenceService, ProofWorkers
from dense_selfplay import record_network_values
from neural_search import checked, native


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


def processor_times():
    """(busy, total) seconds summed over every logical processor since boot, or None where unreadable.

    Windows reads GetSystemTimes, Linux /proc/stat. The difference of two
    readings gives the machine's busy share over that span, all processes included.
    """
    import os
    if os.name == 'nt':
        import ctypes as C
        idle, kernel, user = C.c_uint64(), C.c_uint64(), C.c_uint64()
        if not C.windll.kernel32.GetSystemTimes(C.byref(idle), C.byref(kernel), C.byref(user)):
            return None
        return (kernel.value+user.value-idle.value)/1e7, (kernel.value+user.value)/1e7
    try:
        with open('/proc/stat') as f:
            # user nice system idle iowait irq softirq steal; guest time is already inside user and nice.
            ticks = [int(v) for v in f.readline().split()[1:9]]
    except (OSError, ValueError):
        return None
    hz = os.sysconf('SC_CLK_TCK')
    return (sum(ticks)-ticks[3]-ticks[4])/hz, sum(ticks)/hz


class ProofSizer:
    """Serving proof workers sized so the GPU is fed first and spare CPU proves second.

    Each `observe` after `window` seconds differences the service's supply
    clocks over the window. `starved` is the share of the window in which a
    batch slot was free and no row was ready, `backlog` the share in which
    every slot was in flight with rows waiting, `proof_busy` the mean number
    of workers holding a job, `machine` the busy share of all logical
    processors from `processors()`, reported only. Starvation above `starved`
    parks a quarter of the serving workers, at least one, never below `floor`.
    Starvation below `fed`, with backlog of at least `queued` or at least
    `rows` ready rows at the window's end, wakes one, never above the pool.
    Anything between holds. A change starts a new window and the next comes no
    sooner than the hold, `dwell` seconds. A change against the direction of
    the last one, within `flip` seconds after its hold ended, doubles the hold
    up to `longest`; any other change resets it. A floor at the pool size fixes the count.
    Parked workers keep their solver tables. `reset` discards the window in
    progress, as after a pause. `summary` reports the count, its changes as
    (seconds since start, count), the current hold and the last full window.
    """
    def __init__(self, service, workers, floor, *, window=5., dwell=10., flip=60., longest=160., starved=.2, fed=.05,
                 queued=.5, rows=128, processors=processor_times):
        self.service, self.workers, self.processors = service, workers, processors
        self.ceiling = workers.stats()['workers']
        self.floor = max(1, min(floor, self.ceiling))
        self.window, self.dwell, self.flip, self.longest = window, dwell, flip, longest
        self.starved, self.fed, self.queued, self.rows = starved, fed, queued, rows
        self.serving, self.changes, self.changed, self.direction, self.hold = self.ceiling, 0, None, 0, dwell
        self.history, self.last = deque(maxlen=64), {}
        self.reset()
        self.origin = self.start['now']

    def clocks(self):
        clocks = dict(self.service.supply(), proof_service=self.workers.stats()['worker_service_ms']/1e3)
        times = self.processors()
        clocks['machine_busy'], clocks['machine_total'] = times if times else (None, None)
        return clocks

    def reset(self):
        self.start = self.clocks()

    def observe(self):
        """Read the clocks; at the end of a window record it and resize at most once."""
        now = self.clocks()
        span = now['now']-self.start['now']
        if span < self.window:
            return
        d = {k: now[k]-self.start[k] for k in ('starved', 'unflown', 'backlog', 'producer_wall', 'producer_wait',
                                                'producer_cpu', 'proof_service')}
        busy = d['producer_wall']-d['producer_wait']
        known = now['machine_total'] is not None and self.start['machine_total'] is not None
        total = now['machine_total']-self.start['machine_total'] if known else 0
        self.last = dict(seconds=span, starved=d['starved']/span, unflown=d['unflown']/span, backlog=d['backlog']/span,
                         machine=(now['machine_busy']-self.start['machine_busy'])/total if total > 0 else None,
                         ready_rows=now['ready_rows'], oldest_ready_ms=now['oldest_ready_s']*1e3,
                         producer_busy=busy/d['producer_wall'] if d['producer_wall'] > 0 else None,
                         producer_cpu=d['producer_cpu']/busy if busy > 0 else None,
                         proof_busy=d['proof_service']/span)
        self.start = now
        if self.changed is not None and now['now']-self.changed < self.hold:
            return
        w, count = self.last, self.serving
        if w['starved'] > self.starved:
            count = max(self.floor, count-max(1, count//4))
        elif w['starved'] < self.fed and (w['backlog'] >= self.queued or w['ready_rows'] >= self.rows):
            count = min(self.ceiling, count+1)
        if count != self.serving:
            direction = 1 if count > self.serving else -1
            reversal = direction == -self.direction and now['now']-self.changed < self.hold+self.flip
            self.hold = min(self.longest, 2*self.hold) if reversal else self.dwell
            self.workers.serve(count)
            self.serving, self.changed, self.direction = count, now['now'], direction
            self.changes += 1
            self.history.append((round(now['now']-self.origin, 1), count))

    def summary(self):
        return dict(serving=self.serving, floor=self.floor, ceiling=self.ceiling, changes=self.changes,
                    hold=self.hold, history=list(self.history), window=dict(self.last))


class HybridGames:
    """Replenishable fixed-model slots with native graph ownership and proof retirement.

    `step` exposes a finished game only after every one of its model owners has
    returned final fresh work. Dynamic mode attaches frozen checkpoints while
    older games continue; fixed cohorts retain their original model set.
    In dynamic mode `producers` bounds producer threads plus host workers, and
    each model's slots are split over `model_producers` independent producers.
    Up to `proof_workers` native proof workers serve every producer's live games;
    a ProofSizer serves between `proof_floor` (default: all of them) and all, from
    inference starvation. `proof_budget` caps each graph owner's share of time in
    proof work (ProofLoop owner_budget), and `proof_stamps` lets the proof loops
    reuse the solver's stamps (ProofLoop stamps).
    """
    def __init__(self, games, *, producers=4, quantum=64, views=8, depth=8, cache=8192,
                 batch_size=128, slice_ms=8, proof_workers=0, proof_package=None, ms=0, progress=None,
                 dynamic=False, model_producers=1, proof_budget=1., proof_stamps=False, proof_floor=None):
        if not games or any(not g.hybrid for g in games) or producers<1:
            raise ValueError('A nonempty cohort of hybrid games is required')
        if not 1<=model_producers<=producers:
            raise ValueError('Producers per model must lie between one and the host allocation')
        self.games = list(games)
        models = [model for g in games for model in g.trees]
        self.models = list({m.sha:m for m in models}.values()) if dynamic else list(dict.fromkeys(models))
        if len(self.models)>16:
            raise ValueError('At most sixteen frozen models can share a native service')
        self.slot_models = [set(g.trees) for g in games]
        self.views, self.ms, self.progress = views, ms, progress
        self.pools, self.mapping, self.lookup, self.proof_loops = [], {}, {}, []
        self.epochs, self.finished, self.closing, self.warming = {}, set(), {}, {}
        self.service, self.receipt, self.sizer = None, None, None
        self.proof_floor = proof_workers if proof_floor is None else proof_floor
        self.dynamic = dynamic
        self.batch_size = batch_size
        self.groups, self.idle, self.waiting, self.unbound = {}, {}, set(), set()
        self.archived_proofs, self.archived_stats = deque(maxlen=512), deque(maxlen=512)
        self.options = dict(quantum=quantum,views=views,depth=depth,work=quantum,cache=cache,
                            workers=max(1,producers//len(self.models)))
        self.host_budget = min(16,producers if dynamic else max(producers,len(self.models)))
        self.split = min(model_producers,len(self.games))
        self.proof_workers = ProofWorkers(proof_package,workers=proof_workers) if proof_workers else None
        self.proof_options = dict(slice_ms=slice_ms,table_mb=4,shared=self.proof_workers,owner_budget=proof_budget,
                                  stamps=proof_stamps)
        if dynamic:
            try:
                if any(not self.fits({m.sha for m in g.trees}) for g in games):
                    raise ValueError('Initial model set exceeds the native host allocation')
                initial = list(self.models)
                self.models = []
                admitted = set()
                for game in games:
                    candidate = {m.sha for m in game.trees}
                    if self.fits(admitted|candidate):
                        admitted.update(candidate)
                for model in initial:
                    if model.sha in admitted:
                        self.register(model)
                self.waiting.update(range(len(games)))
                self.service = InferenceService(self.pools,[m.evaluator for m in self.models],batch_size=batch_size,
                    quantum=min(128,batch_size),pending=4)
                self.service.start(continuous=True)
                self.service.launch()
                self.size_proofs()
                for key in self.idle:
                    self.service.release(*key,expected=0)
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
                    self.proof_loops.append(pool.enable_proofs(**self.proof_options) if proof_workers else None)
            self.service = InferenceService(self.pools,[m.evaluator for m in self.models],batch_size=batch_size)
            self.service.start(continuous=True)
            self.service.launch()
            self.size_proofs()
            for index in range(len(games)):
                self.next_root(index)
        except BaseException:
            self.close()
            raise

    @property
    def done(self):
        return len(self.finished)==len(self.games)

    def size_proofs(self):
        if self.proof_workers is not None:
            self.sizer = ProofSizer(self.service,self.proof_workers,self.proof_floor)

    def fits(self, shas):
        """Whether these models' producers fit the host allocation."""
        return len(shas)*self.split<=self.host_budget

    @staticmethod
    def shares(count, parts):
        return [count//parts+(j<count%parts) for j in range(parts)]

    def register(self, model):
        """Split each frozen model's reusable game slots over `split` producers.

        A retiring model admits no games until all of its producers have left.
        """
        if model.sha in self.groups:
            return not self.groups[model.sha]['retiring']
        if sum(len(g['producers']) for g in self.groups.values())+self.split>self.host_budget:
            return False
        if model.evaluator.graph is not None:
            graph = model.evaluator.graph
            graph.max_batch = max(b for b in graph.BATCHES if b<=min(128,self.batch_size))
        placeholders = {}
        for index,game in enumerate(self.games):
            s = game.settings
            source = model.tree([],game.seed+index,s.tactics,s.search_graph,s.q_range_floor,s.graph_nodes)
            if s.hybrid_round_barrier:
                checked(native.hxg_round_barrier(source.ptr, 1))
            placeholders[index] = source
        workers = self.allocate(model.sha).get(model.sha,self.split) if self.service else max(self.split,self.options['workers'])
        pools, attached, keys = [], [], {}
        try:
            for part,share in enumerate(self.shares(workers,self.split)):
                slots = range(part,len(self.games),self.split)
                pool = SearchPool([placeholders[i] for i in slots],seed=self.games[0].seed+part,**dict(self.options,workers=share))
                pools.append(pool)
                proof = pool.enable_proofs(**self.proof_options) if self.proof_workers else None
                if self.service is None:
                    producer,model_id = len(self.pools),len(self.models)
                    self.pools.append(pool);self.proof_loops.append(proof)
                else:
                    producer,model_id = self.service.attach(pool,model.evaluator)
                    while len(self.pools)<=producer:
                        self.pools.append(None);self.proof_loops.append(None)
                    self.pools[producer],self.proof_loops[producer] = pool,proof
                attached.append(producer)
                for owner,index in enumerate(slots):
                    keys[index] = producer,owner
            if self.service is None:
                self.models.append(model)
            else:
                while len(self.models)<=model_id:
                    self.models.append(None)
                self.models[model_id] = model
            self.groups[model.sha] = dict(model=model,producers=attached,model_id=model_id,keys=keys,
                                          slots={key:index for index,key in keys.items()},
                                          free=set(),workers=workers,retiring=False)
            for index,key in keys.items():
                self.epochs[key] = 0
                self.lookup[key] = None
                self.idle[key] = placeholders[index]
                if self.service is not None:
                    self.service.release(*key,expected=0)
            return True
        except BaseException:
            for pool in pools:
                if getattr(pool,'_service',None) is None:
                    pool.close()
            if not attached:
                for tree in placeholders.values():
                    tree.close()
            raise

    def allocate(self, extra=None):
        keys = [*self.groups,*([extra] if extra else [])]
        counts = {sha:len(self.groups[sha]['producers']) if sha in self.groups else self.split for sha in keys}
        weights = {sha:max(1,sum(any(m.sha==sha for m in game.trees) for i,game in enumerate(self.games)
                                if i not in self.finished)) for sha in keys}
        for _ in range(self.host_budget-sum(counts.values())):
            sha = max(keys,key=lambda k:weights[k]/counts[k])
            counts[sha] += 1
        # Shrink before growing so a rotation never multiplies active CPU work.
        for grow in (False,True):
            for sha,group in self.groups.items():
                count = counts[sha]
                if count!=group['workers'] and (count>group['workers'])==grow:
                    for producer,share in zip(group['producers'],self.shares(count,len(group['producers']))):
                        self.service.workers(producer,share)
                    group['workers'] = count
        return counts

    def unbind(self):
        for index in self.finished-self.unbound:
            if any(tree.ptr for tree in self.games[index].trees.values()):
                continue  # The publisher still owns these caller wrappers.
            for model in self.slot_models[index]:
                key = self.mapping.pop((index,model))
                self.lookup[key] = None
                self.groups[model.sha]['free'].add(index)
            self.unbound.add(index)

    def maintain(self):
        """Retire unused weights only after graph retirement and device work finish."""
        self.unbind()
        active = {m.sha for i,g in enumerate(self.games) if i not in self.finished and i not in self.waiting for m in g.trees}
        needed = set(active)
        for index in sorted(self.waiting):
            candidate = {m.sha for m in self.games[index].trees}
            if self.fits(active|candidate):
                needed.update(candidate)
                break
        for sha,group in list(self.groups.items()):
            if not group['retiring'] and (sha in needed or len(group['free'])!=len(self.games)):
                continue
            # Reclamation is bounded, so a model's producers may leave over several calls.
            group['retiring'] = True
            while group['producers'] and self.service.reclaim_ready():
                producer = group['producers'].pop()
                group['slots'] = {key:index for key,index in group['slots'].items() if key[0]!=producer}
                self.service.detach(producer)
                loop = self.proof_loops[producer]
                if loop:
                    self.archived_stats.append(dict(loop.stats(),model=sha))
                    self.archived_proofs.extend(dict(r,model=sha) for r in loop.records())
                self.service.reclaim(self.pools[producer])
                self.pools[producer],self.proof_loops[producer] = None,None
            if not group['producers']:
                del self.groups[sha]
        for model_id,model in enumerate(self.models):
            if model is not None and model.sha not in self.groups and not self.service.model_pending(model_id):
                if model.evaluator.graph is not None:
                    model.evaluator.graph.close()
                    model.evaluator.graph = None
                self.models[model_id] = None
                self.service.models[model_id] = None
        if self.groups:
            self.allocate()
        for index in sorted(self.waiting):
            game = self.games[index]
            candidate = {m.sha for m in game.trees}
            if not self.fits(active|candidate):
                continue
            if not all(self.register(m) for m in game.trees):
                continue
            if not all(index in self.groups[m.sha]['free'] for m in game.trees):
                continue
            self.waiting.remove(index)
            active.update(candidate)
            self.warming[index] = set()
            for k,model in enumerate(game.trees):
                group = self.groups[model.sha]
                key = group['keys'][index]
                group['free'].remove(index)
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
                              noise=game.settings.root_noise if game.is_full else 0.,
                              concentration=game.settings.root_noise_concentration,temperature=game.temperature)

    def retire(self, index):
        keys = {self.mapping[index,m] for m in self.slot_models[index]}
        self.closing[index] = keys
        for producer,owner in keys:
            self.service.release(producer,owner,expected=self.epochs[producer,owner])

    def replace(self, index, game):
        """Start a fresh game in a fully retired slot, retaining model predictions."""
        if index not in self.finished or not game.hybrid or (not self.dynamic and set(game.trees)!=self.slot_models[index]):
            raise ValueError('Replacement needs a retired slot and its frozen model set')
        if any(tree.ptr for tree in self.games[index].trees.values()):
            raise ValueError('Publish and close the retired game before replacing its slot')
        if self.dynamic:
            if not self.fits({m.sha for m in game.trees}):
                raise ValueError('Replacement model set exceeds the native host allocation')
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
        if self.sizer:
            self.sizer.reset()

    def step(self, wait_ms=50.):
        """Consume immutable placement/lifecycle events, waiting up to `wait_ms` for one.

        The service's launcher thread keeps forwards running meanwhile, so new
        events can arrive while these are handled; one step takes at most two
        per slot and returns, leaving the caller its own work between steps.
        """
        self.service.wait(wait_ms)
        finished = []
        for _ in range(2*len(self.games)):
            if (event:=self.service.event()) is None:
                break
            key = event['producer'],event['game']
            if key in self.idle:
                group = next(g for g in self.groups.values() if key in g['slots'])
                if event.get('kind')!='released' or event['token']!=self.epochs[key]+1 or event['model']!=group['model_id']:
                    raise ValueError('Unexpected idle native graph retirement')
                self.epochs[key] = event['token']
                self.idle.pop(key).close()
                group['free'].add(group['slots'][key])
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
                          completed_q=edges[:,3],values=edges[:,4],policy=edges[:,5],prior_logits=edges[:,2],
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
        if self.sizer:
            self.sizer.observe()
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
        if self.proof_workers is not None:
            self.proof_workers.close()


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
            self.engine.replace(self.free[0],game)
            self.free.popleft()
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
        running = self.engine is not None and bool(self.engine.service._ptr)
        return dict(inference=self.engine.service.stats() if self.engine else None,
                    producers=sum(p is not None for p in self.engine.pools) if running else 0,
                    host_workers=sum(g['workers'] for g in self.engine.groups.values()) if running else 0,
                    waiting_games=len(self.engine.waiting) if self.engine else 0,
                    proof_workers=self.engine.sizer.summary() if running and self.engine.sizer else None,
                    retired_fresh_nodes=self.fresh_nodes,retired_queries=self.queries,
                    retired_missing_fresh=self.missing_fresh)

    def step(self):
        if self.paused:
            raise ValueError('Resume the hybrid actor before stepping')
        if self.engine is None:
            s = self.settings
            self.engine = HybridGames(self.slots,dynamic=True,producers=s.hybrid_producers,
                quantum=s.hybrid_quantum,views=s.hybrid_views,depth=s.hybrid_depth,
                cache=s.cache_positions,batch_size=s.leaf_batch,proof_workers=s.hybrid_proof_workers,
                slice_ms=s.hybrid_proof_slice_ms,progress=self.progress,model_producers=s.hybrid_model_producers,
                proof_budget=s.hybrid_proof_budget,proof_stamps=s.hybrid_proof_stamps,proof_floor=s.hybrid_proof_floor)
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

    def synchronize_inflight(self, models=()):
        evaluators = [m.evaluator for m in models]
        evaluators.extend(m.evaluator for g in self.slots for m in g.trees)
        if self.engine is not None:
            self.engine.pause()
            self.account()
            evaluators.extend(e for e in self.engine.service.models if e is not None)
        for evaluator in dict.fromkeys(evaluators):
            if evaluator.graph is not None:
                graph = evaluator.graph
                self.captures.append((evaluator,graph.max_incremental_bytes,graph.max_batch))
                graph.close()
                evaluator.graph = None
            evaluator.free.clear()
            evaluator.staging.clear()
        self.paused = True

    def resume(self):
        from hexnet_graphs import ActorGraph
        for evaluator,memory,batch in self.captures:
            evaluator.graph = ActorGraph(evaluator.model,max_incremental_bytes=memory,max_batch=batch)
        self.captures.clear()
        if self.engine is not None:
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

    Returning None leaves a slot retired. The factory returns fresh hybrid
    SelfPlayGames with that slot's frozen model set. HybridGames exposes the same
    lifecycle incrementally for a daemon that publishes bounded shards.
    """
    engine = HybridGames(games,**options)
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
    """Complete a fixed cohort through the same replenishable hybrid lifecycle."""
    return play_stream(games,**options)
