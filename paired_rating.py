"""Joint checkpoint ratings from actual color-swapped opening-pair outcomes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch

from train import paired_metrics, write_json

SCALE=400/math.log(10)
METHOD='Joint likelihood of completed color-swapped pairs; conditional MAP Elo and 95% credible intervals'
NOTE=('Provisional ratings use only pairs where both games finished. Capped or unplayed outcomes remain unknown; '
      'game completion may depend on strength, so the point and credible interval can be selection-biased. '
      'Observed-score bounds include every unknown outcome and are not 95% confidence intervals. '
      'Actual paired counts have no per-match pseudo-wins. Checkpoint 0 fixes the origin. '
      'Ratings have independent Normal(0, 1000 Elo) priors. A shared fitted pair-dispersion parameter '
      'allows split pairs to be more or less common than independent games. Intervals are conditional '
      'on this model and do not cover CPU/CUDA differences or cross-comparison opening dependence.')


def completed_pairs(ids,reports):
    edges=[];coverage=[]
    for report in reports:
        a,b=report['candidate'],report['opponent']
        if a not in ids or b not in ids:continue
        if a==b:raise ValueError('A checkpoint cannot rate itself')
        games=report['games'];metrics=report['metrics'];target=report.get('target_games',metrics['planned_games'])
        if target<2 or target%2 or len(games)>target or len(games)%2 or \
           metrics!=paired_metrics(games,metrics['planned_games']):
            raise ValueError('Comparison games disagree with reported metrics')
        pairs={}
        for game in games:pairs.setdefault(game['pair'],[]).append(game)
        counts=np.zeros(3);censored=0
        for pair,games_in_pair in pairs.items():
            if len(games_in_pair)!=2 or {g['challenger_color'] for g in games_in_pair}!={0,1} or \
               games_in_pair[0]['opening']!=games_in_pair[1]['opening'] or \
               games_in_pair[0]['seed']!=games_in_pair[1]['seed'] or \
               any(g['winner'] not in (-1,0,1) for g in games_in_pair):
                raise ValueError('Rating requires intact color-swapped opening pairs')
            if any(g['winner']<0 for g in games_in_pair):censored+=1
            else:counts[sum(g['winner']==g['challenger_color'] for g in games_in_pair)]+=1
        rated=int(counts.sum());unplayed=(target-len(games))//2
        known=metrics['wins'];unknown=target-known-metrics['losses']
        coverage.append(dict(candidate=a,opponent=b,origin='background' if 'background_identity' in report else 'scheduled',
            target_games=target,known_wins=known,known_losses=metrics['losses'],rated_pairs=rated,
            censored_pairs=censored,unplayed_pairs=unplayed,capped_games=metrics['incomplete'],
            observed_score_bounds=[known/target,(known+unknown)/target]))
        if rated:edges.append((a,b,counts,2*rated))
    return edges,coverage


def fit_ratings(ids,reports,seed=1740,samples=32768):
    edges,coverage=completed_pairs(ids,reports)
    connected={0}
    while True:
        before=set(connected)
        for a,b,_,_ in edges:
            if a in connected or b in connected:connected.update((a,b))
        if before==connected:break
    variables=sorted(connected-{0});index={n:i for i,n in enumerate(variables)}
    result={n:dict(id=n,elo=None,elo_interval=None,rating_pairs=0) for n in ids}
    result[0].update(elo=0.,elo_interval=[0.,0.])
    for record in coverage:
        for number in (record['candidate'],record['opponent']):
            result[number].setdefault('rating_censored_pairs',0)
            result[number].setdefault('rating_unplayed_pairs',0)
            result[number]['rating_censored_pairs']+=record['censored_pairs']
            result[number]['rating_unplayed_pairs']+=record['unplayed_pairs']
    provisional=any(record['rated_pairs'] and (record['censored_pairs'] or record['unplayed_pairs']) and
                    record['candidate'] in connected and record['opponent'] in connected for record in coverage)
    for number in variables:
        result[number]['rating_provisional']=provisional or bool(result[number].get('rating_censored_pairs')) or \
            bool(result[number].get('rating_unplayed_pairs'))
    edges=[e for e in edges if e[0] in connected]
    if not variables:return result,dict(effective_samples=0,comparison_coverage=coverage)
    matrix=np.zeros((len(edges),len(variables)))
    for i,(a,b,alpha,n) in enumerate(edges):
        if a:matrix[i,index[a]]=1
        if b:matrix[i,index[b]]=-1
        result[a]['rating_pairs']+=int(n/2);result[b]['rating_pairs']+=int(n/2)
    matrix=torch.tensor(matrix,dtype=torch.float64)
    counts=torch.tensor(np.array([e[2] for e in edges]),dtype=torch.float64)
    def loss(x):
        d=x[...,:-1]@matrix.T;tau=x[...,-1:]
        # p=logistic(d) remains the expected game score for any dispersion.
        # Pair scores 0,1,2 have probabilities softmax(-h,tau,h).
        h=d/2+torch.asinh(torch.exp(tau)*torch.sinh(d/2)/2)
        logits=torch.stack((-h,tau.expand_as(h),h),dim=-1)
        likelihood=-(counts*torch.log_softmax(logits,dim=-1)).sum(dim=(-1,-2))
        return likelihood+.5*(x[...,:-1]/(1000/SCALE)).square().sum(-1)+.5*((x[...,-1]-math.log(2))/2).square()
    def objective(x):
        t=torch.tensor(x,dtype=torch.float64,requires_grad=True);value=loss(t)
        return value.item(),torch.autograd.grad(value,t)[0].numpy()
    initial=np.zeros(len(variables)+1);initial[-1]=math.log(2)
    optimum=minimize(objective,initial,jac=True,method='L-BFGS-B',
        options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8))
    if not optimum.success or not np.isfinite(optimum.fun):raise ValueError('Joint Elo optimization failed')
    hessian=torch.autograd.functional.hessian(loss,torch.tensor(optimum.x,dtype=torch.float64)).numpy()
    covariance=np.linalg.inv(hessian);cholesky=np.linalg.cholesky(covariance)
    # A heavy-tailed proposal retains uncertainty after sweeps and tiny samples.
    rng=np.random.default_rng(seed);dimension=len(initial);df=5.
    z=rng.normal(size=(samples,dimension))/np.sqrt(rng.chisquare(df,size=(samples,1))/df)
    draws=optimum.x+z@cholesky.T
    log_proposal=(math.lgamma((df+dimension)/2)-math.lgamma(df/2)-dimension/2*math.log(df*math.pi)
        -np.log(np.diag(cholesky)).sum()-(df+dimension)/2*np.log1p(np.square(z).sum(1)/df))
    with torch.inference_mode():
        log_target=np.concatenate([-loss(torch.tensor(batch,dtype=torch.float64)).numpy() for batch in np.array_split(draws,32)])
    log_weights=log_target-log_proposal
    weights=np.exp(log_weights-logsumexp(log_weights));ess=float(1/np.square(weights).sum())
    if not np.isfinite(weights).all() or ess<1000:raise ValueError(f'Insufficient posterior sampling precision: ESS={ess:.0f}')
    for number,i in index.items():
        values=draws[:,i]*SCALE;order=np.argsort(values);cdf=np.cumsum(weights[order])
        interval=np.interp([.025,.975],cdf,values[order]).tolist()
        result[number].update(elo=float(optimum.x[i]*SCALE),elo_interval=interval)
    return result,dict(effective_samples=ess,samples=samples,pair_dispersion=float(optimum.x[-1]),
                       comparison_coverage=coverage)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


_model_digests={}


def model_digest(path,modified,size):
    # Published models are immutable; invalidate verification when file metadata changes.
    cached=_model_digests.get(path)
    if cached is None or cached[:2]!=(modified,size):
        cached=(modified,size,sha(Path(path)));_model_digests[path]=cached
    return cached[2]


def snapshot(run):
    config_hash=sha(run/'config.json');config=read(run/'config.json');league=read(run/'league.json')
    protocol=league.get('rating_protocol')
    hashes={c['id']:read(run/'checkpoints'/f"{c['id']:04d}"/'manifest.json')['files']['model.pt'] for c in league['checkpoints']}
    for path in sorted((run/'checkpoints').glob('[0-9][0-9][0-9][0-9]/manifest.json')):
        number=int(path.parent.name)
        if number in hashes:continue
        manifest=read(path)
        if manifest['identity']!=dict(config,checkpoint=number):raise ValueError('Pending checkpoint identity changed')
        hashes[number]=manifest['files']['model.pt']
    model_paths=set()
    for number,expected in hashes.items():
        path=run/'checkpoints'/f'{number:04d}'/'model.pt';stat=path.stat()
        model_paths.add(str(path))
        if model_digest(str(path),stat.st_mtime_ns,stat.st_size)!=expected:raise ValueError('Rating checkpoint bytes changed')
    for path in set(_model_digests)-model_paths:del _model_digests[path]
    reports=[];sources={};background_identity=None
    for path in sorted((run/'evaluation').glob('*-vs-*/report.json'))+sorted((run/'background-evaluation').glob('*-vs-*.json')):
        contents=path.read_bytes();report=json.loads(contents);a,b=report['candidate'],report['opponent']
        if a not in hashes or b not in hashes:continue
        digest=hashlib.sha256(contents).hexdigest()
        if 'background_identity' in report:
            if protocol and report.get('protocol')!=protocol:continue
            if background_identity is None:background_identity=report['background_identity']
            if report['background_identity']!=background_identity:raise ValueError('Background worker identities differ')
            if report['background_identity']['config_sha256']!=config_hash or report['model_hashes']!={str(a):hashes[a],str(b):hashes[b]}:
                raise ValueError('Background rating input identity changed')
        else:
            manifest=read(path.parent/'manifest.json')
            if protocol and manifest['identity'].get('protocol')!=protocol:continue
            if manifest['files']['report.json']!=digest or any(manifest['identity'][role]!=hashes[report[role]] for role in ('candidate','opponent')):
                raise ValueError('Scheduled rating input identity changed')
        reports.append(report);sources[path.relative_to(run).as_posix()]=digest
    return config_hash,hashes,reports,sources


def run_ratings(args):
    run=Path(args.run).resolve();torch.set_num_threads(1);previous=None
    while True:
        config_hash,hashes,reports,sources=snapshot(run)
        signature=(config_hash,hashes,sources)
        if signature!=previous:
            ratings,diagnostics=fit_ratings(list(hashes),reports)
            for n,record in ratings.items():record['model_sha256']=hashes[n]
            result=dict(config_sha256=config_hash,checkpoints=list(ratings.values()),updated_at=time.time(),
                rating_method=METHOD,note=NOTE,diagnostics=diagnostics,reports=sources)
            write_json(run/'paired-ratings.json',result)
            print(json.dumps(dict(updated_at=result['updated_at'],checkpoints=len(ratings),**diagnostics)),flush=True)
            previous=signature
        if args.once:return
        time.sleep(10)


def main(args):
    lock=Path(args.run).resolve()/'paired-ratings.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:run_ratings(args)
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--once',action='store_true')
    main(parser.parse_args())
