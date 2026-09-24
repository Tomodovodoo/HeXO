"""Fresh-corpus KLENT for the primary relational policy/Q network."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from hexo import Game, ROOT, library
from klent import (digest, improved_policy, load_corpus, publish, save_corpus,
                   segmented_log_softmax, signed_returns, verify)
from train import write_json

MODEL_SCHEMA = 'hexo-relational-policy-q-v1'


def load_model(path, device='cpu', *, expected_sha256=None):
    from relational_model import ModelConfig, RelationalNet
    source = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('Relational checkpoint changed before loading')
    payload = torch.load(io.BytesIO(source), map_location=device, weights_only=True)
    if payload.get('schema') != MODEL_SCHEMA:
        raise ValueError('Expected a relational policy/Q checkpoint, not NNUE weights')
    model = RelationalNet(ModelConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['state'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite relational checkpoint')
    return model


def save_model(path, model):
    torch.save(dict(schema=MODEL_SCHEMA, config=asdict(model.config), state=model.state_dict()), path)


def precision(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()


def source_identity(extra=()):
    files = ('relational_train.py', 'relational_encoder.py', 'relational_model.py',
             'relational_native.py', 'klent.py', 'hexo.py', 'train.py') + tuple(extra)
    from relational_native import _load
    return dict(engine_sha256=digest(library), graph_engine_sha256=digest(Path(_load()._name)),
                sources={name: digest(ROOT/name) for name in files})


def graph(history, model, args=None):
    from relational_native import encode
    budgets = dict(max_nodes=args.max_nodes, max_edges=args.max_edges) if args else {}
    return encode(history, global_tokens=model.config.global_tokens, **budgets)


def work_batches(graphs, args):
    from relational_encoder import iter_batches
    yield from iter_batches(graphs, max_nodes=args.max_nodes, max_edges=args.max_edges)


def outputs(model, graphs, args):
    from relational_encoder import pack
    batch = pack(graphs, args.device)
    with precision(args.device):
        result = model(batch)
    logits, q = result['logits'].float(), result['q'].float()
    if not torch.isfinite(logits).all() or not torch.isfinite(q).all():
        raise FloatingPointError('Nonfinite relational policy/Q output')
    return batch, logits, q


def legal_hash(actions):
    return hashlib.sha256(np.asarray(actions, dtype=np.int64).tobytes()).hexdigest()


def collect(model, args, iteration, progress=None):
    """One frozen actor, exact full legal normalization, no tactical overrides."""
    model.eval()
    rng = np.random.default_rng(args.seed+iteration*10000)
    episodes, rows, live = [], [], []
    next_id = 0
    try:
        while live or next_id < args.games:
            while len(live) < args.envs and next_id < args.games:
                live.append(dict(id=next_id, game=Game(), moves=[], rows=[]))
                next_id += 1
            graphs = [graph(e['moves'], model, args) for e in live]
            offset = 0
            finished = []
            with torch.no_grad():
                for group in work_batches(graphs, args):
                    batch, logits, q = outputs(model, group, args)
                    mu, values, kl, entropy = improved_policy(logits, q, batch['action_owner'],
                                                            len(group), args.alpha, args.beta)
                    bounds = batch['action_offsets'].cpu().tolist()
                    probabilities = mu.cpu().numpy()
                    values, kl, entropy = values.cpu().tolist(), kl.cpu().tolist(), entropy.cpu().tolist()
                    for j, item in enumerate(group):
                        episode = live[offset+j]
                        game = episode['game']
                        if len(episode['moves']) >= args.max_plies:
                            episode.update(winner=-1, reason='cap', tail_player=game.player, tail_value=values[j])
                            finished.append(episode)
                            continue
                        probs = probabilities[bounds[j]:bounds[j+1]].astype(np.float64)
                        probs /= probs.sum()
                        chosen = int(rng.choice(len(probs), p=probs))
                        action = item.actions[chosen].tolist()
                        row = dict(game=episode['id'], ply=len(episode['moves']), player=game.player,
                                   remaining=game.remaining, action=action, chosen=chosen,
                                   legal_sha256=legal_hash(item.actions), mu=probs.astype(np.float32),
                                   vhat=values[j], kl=kl[j], entropy=entropy[j])
                        episode['rows'].append(row)
                        game.play(*action)
                        episode['moves'].append(action)
                        if game.winner >= 0:
                            episode.update(winner=game.winner, reason='six-in-a-row', tail_player=None, tail_value=None)
                            finished.append(episode)
                    offset += len(group)
            for episode in finished:
                targets = signed_returns([r['player'] for r in episode['rows']],
                                         [r['vhat'] for r in episode['rows']], episode['winner'] >= 0,
                                         episode['tail_player'], episode['tail_value'], args.gamma, args.lambda_return)
                for row, target in zip(episode['rows'], targets, strict=True):
                    row.update(target=float(target), bootstrapped=episode['winner'] < 0)
                    rows.append(row)
                episodes.append({k: episode[k] for k in ('id', 'moves', 'winner', 'reason', 'tail_player', 'tail_value')})
                episode['game'].close()
            done = {e['id'] for e in finished}
            live = [e for e in live if e['id'] not in done]
            if progress:
                progress(dict(stage='collection', games=len(episodes), games_total=args.games,
                              positions=len(rows)+sum(len(e['rows']) for e in live),
                              terminal_games=sum(e['winner'] >= 0 for e in episodes)))
    finally:
        for episode in live:
            episode['game'].close()
    return episodes, rows


def rebuild(row, episodes, model, args=None):
    item = graph(episodes[row['game']]['moves'][:row['ply']], model, args)
    if (item.player != row['player'] or item.remaining != row['remaining']
            or legal_hash(item.actions) != row['legal_sha256'] or len(item.actions) != len(row['mu'])
            or item.actions[row['chosen']].tolist() != row['action']):
        raise ValueError('Relational replay geometry/action indexing changed')
    return item


def fit(model, optimizer, episodes, rows, args, iteration, progress=None):
    """Exactly one shuffled pass; memory chunks preserve position-batch gradients."""
    model.train()
    episodes = {e['id']: e for e in episodes}
    order = np.random.default_rng(args.seed+iteration).permutation(len(rows))
    totals = np.zeros(2)
    steps = 0
    for start in range(0, len(order), args.batch):
        selected = [rows[int(i)] for i in order[start:start+args.batch]]
        graphs = [rebuild(r, episodes, model, args) for r in selected]
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        for group in work_batches(graphs, args):
            target_rows = selected[offset:offset+len(group)]
            batch, logits, q = outputs(model, group, args)
            logpi = segmented_log_softmax(logits, batch['action_owner'], len(group))
            mu = torch.as_tensor(np.concatenate([r['mu'] for r in target_rows]), device=args.device)
            target = torch.tensor([r['target'] for r in target_rows], device=args.device)
            chosen = batch['action_offsets'][:-1]+torch.tensor([r['chosen'] for r in target_rows], device=args.device)
            ce = -(mu*logpi).sum()/len(group)
            mse = (q[chosen]-target).square().mean()
            ((ce+mse)*(len(group)/len(selected))).backward()
            totals += np.array([ce.item(), mse.item()])*len(group)
            offset += len(group)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        steps += 1
        if progress:
            progress(dict(stage='fitting', positions=len(rows), fit_completed=min(start+args.batch,len(rows)),
                          optimizer_steps=steps, policy_ce=totals[0]/min(start+args.batch,len(rows)),
                          q_mse=totals[1]/min(start+args.batch,len(rows))))
    return dict(policy_ce=totals[0]/len(rows), q_mse=totals[1]/len(rows), optimizer_steps=steps)


def main(args):
    from relational_model import ModelConfig, RelationalNet
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config = {k: v for k,v in vars(args).items() if k not in ('run','iterations')}
    if args.initial_model:
        config['initial_model'] = str(Path(args.initial_model).resolve())
        config['initial_model_sha256'] = digest(Path(args.initial_model))
    identity = dict(run=str(run), backbone=MODEL_SCHEMA, config=config, **source_identity(),
                    runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda),
                    optimizer_initialization='Fresh Adam at new-run initialization; exact optimizer restored on resume')
    lock = run/'training.lock'
    import os
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        (run/'checkpoints').mkdir(exist_ok=True)
        (run/'corpus').mkdir(exist_ok=True)
        existing = sorted((run/'checkpoints').glob('[0-9][0-9][0-9][0-9]'))
        if existing:
            for checkpoint in existing:
                checkpoint_manifest = verify(checkpoint, identity)
            latest = existing[-1]
            model = load_model(latest/'model.pt', args.device, expected_sha256=checkpoint_manifest['files']['model.pt'])
            completed = int(latest.name)
        else:
            if not args.initial_model and not args.allow_cold_start:
                raise ValueError('Warm-start policy and Q, or explicitly request --allow-cold-start')
            model = load_model(args.initial_model,args.device,expected_sha256=config['initial_model_sha256']) if args.initial_model else RelationalNet(ModelConfig()).to(args.device)
            if getattr(args,'critic_initialization','checkpoint')=='zero-output':
                with torch.no_grad():
                    model.critic[-1].weight.zero_()
                    model.critic[-1].bias.zero_()
            completed = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if existing:
            optimizer.load_state_dict(torch.load(latest/'optimizer.pt', map_location=args.device, weights_only=True))
        def writer(stage):
            save_model(stage/'model.pt', model)
            torch.save(optimizer.state_dict(), stage/'optimizer.pt')
        if not existing:
            publish(run/'checkpoints/0000',identity,writer)
        for number in range(1, completed+1):
            actor = run/'checkpoints'/f'{number-1:04d}'/'model.pt'
            corpus_identity = dict(identity, iteration=number, actor_sha256=digest(actor),
                                   actions='full-legal-native-order', returns='signed-lambda-v1-cap-bootstrap')
            verify(run/'corpus'/f'{number:04d}',corpus_identity)
            checkpoint = json.loads((run/'checkpoints'/f'{number:04d}'/'manifest.json').read_text())
            if checkpoint['metrics']['corpus_manifest_sha256'] != digest(run/'corpus'/f'{number:04d}'/'manifest.json'):
                raise ValueError('Previously consumed corpus changed')
        for number in range(completed+1,args.iterations+1):
            start = time.perf_counter()
            if args.device == 'cuda':torch.cuda.reset_peak_memory_stats()
            actor = run/'checkpoints'/f'{number-1:04d}'/'model.pt'
            corpus_identity = dict(identity, iteration=number, actor_sha256=digest(actor),
                                   actions='full-legal-native-order', returns='signed-lambda-v1-cap-bootstrap')
            last = [0., None]
            def progress(data):
                if data['stage'] != last[1] or time.monotonic()-last[0] > .5:
                    write_json(run/'status.json',dict(schema=MODEL_SCHEMA,updated_at=time.time(),iteration=number,**data))
                    last[:] = [time.monotonic(),data['stage']]
            corpus = run/'corpus'/f'{number:04d}'
            if not corpus.exists():
                episodes, rows = collect(model,args,number,progress)
                save_corpus(corpus,corpus_identity,episodes,rows)
            episodes, rows, _ = load_corpus(corpus,corpus_identity)
            terminal_games=sum(e['winner']>=0 for e in episodes)
            terminal_fraction=terminal_games/len(episodes)
            minimum_terminal_fraction=getattr(args,'min_terminal_fraction',0.)
            if terminal_fraction<minimum_terminal_fraction:
                write_json(run/'status.json',dict(schema=MODEL_SCHEMA,updated_at=time.time(),
                    stage='collection_insufficient',iteration=number,games=len(episodes),positions=len(rows),
                    terminal_games=terminal_games,bootstrapped_games=len(episodes)-terminal_games,
                    terminal_fraction=terminal_fraction,min_terminal_fraction=minimum_terminal_fraction,
                    fitting_started=False,reason='Fresh corpus terminal fraction is below the declared fitting threshold'))
                return
            metrics = fit(model,optimizer,episodes,rows,args,number,progress)
            metrics.update(iteration=number,games=len(episodes),positions=len(rows),
                           terminal_games=sum(e['winner']>=0 for e in episodes),
                           bootstrapped_games=sum(e['winner']<0 for e in episodes),
                           terminal_fraction=sum(e['winner']>=0 for e in episodes)/len(episodes),
                           nonzero_return_fraction=float(np.mean([abs(r['target'])>1e-8 for r in rows])),
                           target_variance_nonzero=float(np.std([r['target'] for r in rows]))>1e-8,
                           acting_entropy=float(np.mean([r['entropy'] for r in rows])),
                           acting_normalized_entropy=float(np.mean([r['entropy']/math.log(len(r['mu'])) if len(r['mu'])>1 else 0 for r in rows])),
                           acting_kl=float(np.mean([r['kl'] for r in rows])),
                           acting_value_std=float(np.std([r['vhat'] for r in rows])),
                           target_std=float(np.std([r['target'] for r in rows])),
                           legal_action_rows=sum(len(r['mu']) for r in rows),
                           seconds=time.perf_counter()-start,
                           gpu_memory_peak_mb=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0,
                           corpus_manifest_sha256=digest(corpus/'manifest.json'),
                           ratings='Unrated; evaluate the neural player against pinned independent opponents')
            publish(run/'checkpoints'/f'{number:04d}',identity,writer,metrics)
            write_json(run/'status.json',dict(schema=MODEL_SCHEMA,updated_at=time.time(),stage='finished',**metrics))
            print(json.dumps(metrics),flush=True)
    except Exception as error:
        write_json(run/'status.json',dict(schema=MODEL_SCHEMA,updated_at=time.time(),stage='failed',error=repr(error)))
        raise
    finally:
        lock.unlink()


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--initial-model')
    parser.add_argument('--allow-cold-start',action='store_true')
    parser.add_argument('--critic-initialization',choices=['checkpoint','zero-output'],default='checkpoint',
                        help='Only new runs: retain checkpoint Q, or zero only its final projection and record that control')
    parser.add_argument('--min-terminal-fraction',type=float,default=0.,
                        help='Preserve collection but skip fitting when too many games reached the placement cap')
    for name,default in [('iterations',1),('games',16),('envs',8),('max-plies',128),('batch',8),
                         ('max-nodes',12000),('max-edges',600000),('seed',1729)]:
        parser.add_argument('--'+name,type=int,default=default)
    for name,default in [('alpha',.03),('beta',.1),('gamma',1.),('lambda-return',math.exp(-1/16)),('lr',.0001),('grad-clip',1.)]:
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if min(args.iterations,args.games,args.envs,args.max_plies,args.batch,args.max_nodes,args.max_edges)<1:
        parser.error('Counts and graph budgets must be positive')
    if not math.isfinite(args.min_terminal_fraction) or not 0<=args.min_terminal_fraction<=1:
        parser.error('Minimum terminal fraction must be between zero and one')
    if not all(math.isfinite(getattr(args,k)) for k in ('alpha','beta','gamma','lambda_return','lr','grad_clip')) or not (
        args.alpha>=0 and args.beta>=0 and args.alpha+args.beta>0 and 0<args.gamma<=1 and 0<=args.lambda_return<=1 and args.lr>0 and args.grad_clip>0):
        parser.error('Invalid optimization coefficients')
    return args


if __name__=='__main__':main(parse_args())
