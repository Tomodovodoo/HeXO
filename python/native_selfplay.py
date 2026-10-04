"""Bounded fixed-checkpoint self-play through continuous native graph owners.

This writes ordinary played-root rows. It does not change the actor daemon,
checkpoint rotation, pause protocol, learner weights or value-target construction.
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


def play_cohort(games, *, producers=4, quantum=64, views=8, depth=8, cache=8192,
                batch_size=128, slice_ms=8, proof_workers=0, proof_package=None, ms=0,
                progress=None):
    """Complete pre-created SelfPlayGames and return episodes, rows and work receipts.

    Models remain frozen until every game finishes. Only native owners access
    the search graphs; Python handles one immutable event per played placement.
    `ms` gives full roots a clock; cheap roots retain their smaller work ceiling.
    Solver slices run concurrently and exact evidence takes precedence.
    """
    if not games or any(not g.native_owner for g in games) or producers<1:
        raise ValueError('A nonempty cohort of native-owner games is required')
    models = list(dict.fromkeys(model for g in games for model in g.trees))
    pools, mapping, lookup, proof_loops = [], {}, {}, []
    service = None
    epochs = {}
    finished = set()
    try:
        for model in models:
            members = [(i, g.trees[model]) for i, g in enumerate(games) if model in g.trees]
            for start in range(min(producers, len(members))):
                group = members[start::min(producers, len(members))]
                pool = SearchPool([tree for _, tree in group], quantum=quantum, views=views,
                                  depth=depth, work=quantum, cache=cache, seed=games[group[0][0]].seed)
                producer = len(pools)
                pools.append(pool)
                for index, (game, _) in enumerate(group):
                    mapping[game, model] = producer, index
                    lookup[producer, index] = game, model
                    epochs[producer, index] = 0
                proof_loops.append(pool.enable_proofs(proof_package, workers=proof_workers,
                                    queue=max(4, proof_workers*2), slice_ms=slice_ms, table_mb=4)
                                   if proof_workers else None)
        service = InferenceService(pools, [model.evaluator for model in models], batch_size=batch_size)
        service.start(continuous=True)
        def next_root(index):
            game = games[index]
            producer, owner = mapping[index, game.model]
            service.retarget(producer, owner, game.moves, expected=epochs[producer, owner],
                             work=0 if ms and game.is_full else game.budget,
                             ms=ms if game.is_full else 0, samples=game.samples,
                             views=views if game.is_full else 1,
                             noise=game.settings.root_noise if game.is_full else 0.)
        for index in range(len(games)):
            next_root(index)
        while len(finished)<len(games):
            service.pump()
            while (event := service.event()) is not None:
                key = event['producer'], event['game']
                index, model = lookup[key]
                game = games[index]
                if (index in finished or model is not game.model or event['history'] != game.moves
                        or event['token'] != epochs[key]+1):
                    raise ValueError('Native completion does not match the game/model/root generation')
                epochs[key] = event['token']
                if 'error' in event:
                    game.reason = event['error']
                    finished.add(index)
                    if progress:
                        progress(index, event)
                    continue
                edges = np.asarray(event.pop('edges'), np.float64)
                player = game.game.player
                winner = event['exact_winner']
                proven = 0 if winner<0 else 1 if winner==player else -1
                result = dict(event, actions=edges[:, :2].astype(np.int64), visits=edges[:, 6].astype(np.int64),
                              completed_q=edges[:, 3], values=edges[:, 4], policy=edges[:, 5],
                              completed=event['root_completed'],
                              proven=proven, proof_turns=0, solver_nodes=0, solver_budget=0,
                              proof_action=edges[edges[:, 8].astype(bool), :2].astype(np.int64).tolist() if proven>0 else [])
                result['search'] = dict(source='native-root', model=model.sha, context=event['context'],
                                        token=event['token'], comparison_credits=int(edges[:, 7].sum()),
                                        direct_max=int(edges[:, 7].max()), shared_visits=int(edges[:, 6].sum()),
                                        root_completed=event['root_completed'], completed=event['completed'],
                                        root_estimate=None, solver_generation=event['solver_generation'])
                more = game.searched(result)
                result['search']['root_estimate'] = game.values[len(event['history'])]
                prefixes = event['exact_prefixes']
                if winner>=0:
                    prefixes = [*prefixes,[len(event['history']),winner,event['proof_plies'],result['proof_action']]]
                label_prefixes(game, prefixes)
                if progress:
                    progress(index, event)
                if more:
                    next_root(index)
                else:
                    finished.add(index)
        service.close()
        proofs, proof_stats, effort = [], [], {}
        for producer, loop in enumerate(proof_loops):
            if loop is None:
                continue
            proofs.extend(dict(r, producer=producer) for r in loop.records())
            proof_stats.append(dict(loop.stats(), producer=producer))
            for owner in range(len(pools[producer].games)):
                effort[producer, owner] = loop.effort(owner)
        for index, game in enumerate(games):
            for row in game.rows:
                if 'search' not in row:
                    continue
                source = row['search']
                key = mapping[index, game.sides[row['player']]]
                spent = effort.get(key, {}).get(source['solver_generation'], dict(fresh_nodes=0, queries=0, missing_fresh=0))
                source.update(spent)
                row['solver_nodes'] = spent['fresh_nodes']
                row['solver_budget'] = 0  # Slices are time grants, not historical node budgets.
        stats = service.stats()
    finally:
        if service is not None:
            service.close()
        for pool in reversed(pools):
            pool.close()
    record_network_values(games)
    episodes, rows = [], []
    for index, game in enumerate(games):
        episode, played = game.episode()
        episodes.append(episode)
        rows.extend(dict(row, game=index) for row in played)
    return episodes, rows, dict(inference=stats, proofs=proofs, proof_stats=proof_stats)
