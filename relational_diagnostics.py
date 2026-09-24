"""Read-only immediate-tactic diagnostics; masks are not eventual Q labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from klent import digest, improved_policy
from relational_train import graph, load_model, outputs, source_identity
from train import write_json


def evaluate(checkpoint, fixtures, args):
    fixture_bytes=Path(fixtures).read_bytes()
    fixture_sha=hashlib.sha256(fixture_bytes).hexdigest()
    source=json.loads(fixture_bytes)
    identity=source_identity(('relational_diagnostics.py',))
    model_sha=digest(Path(checkpoint))
    model=load_model(checkpoint,args.device,expected_sha256=model_sha).eval()
    rows=[]
    with torch.no_grad():
        for index,position in enumerate(source['positions']):
            item=graph(position['history'],model,args)
            if (item.player!=position['player'] or item.remaining!=position['remaining']
                    or not np.array_equal(item.actions,np.asarray(position['all_legal_actions']))):
                raise ValueError('Tactical fixture legal order or phase changed')
            good={tuple(a) for a in position['good_actions']}
            mask=np.asarray([tuple(a) in good for a in item.actions])
            if not mask.any() or mask.sum()!=len(good):
                raise ValueError('Tactical answer is absent from exact legal actions')
            batch,logits,q=outputs(model,[item],args)
            pi=logits.softmax(0)
            mu,*_=improved_policy(logits,q,batch['action_owner'],1,args.alpha,args.beta)
            pi,mu,q=pi.cpu().numpy(),mu.cpu().numpy(),q.cpu().numpy()
            result=dict(index=index,category=position['category'],position_key=item.position_key,
                        source=position['source'],legal_actions=len(mask),acceptable_actions=int(mask.sum()),
                        q_argmax_acceptable=bool(mask[q.argmax()]),
                        q_gap=float(q[mask].max()-q[~mask].max()) if (~mask).any() else None,
                        q_std=float(q.std()))
            for name,probabilities in (('pi',pi),('mu',mu)):
                positive=probabilities[probabilities>0]
                result[name]=dict(argmax_acceptable=bool(mask[probabilities.argmax()]),
                                  acceptable_mass=float(probabilities[mask].sum()),
                                  max_probability=float(probabilities.max()),
                                  entropy=float(-(positive*np.log(positive)).sum()))
            rows.append(result)
    groups=defaultdict(list)
    for row in rows:groups[row['category']].append(row)
    summary={category:dict(positions=len(values),q_argmax_acceptable=sum(r['q_argmax_acceptable'] for r in values),
                          **{name:dict(argmax_acceptable=sum(r[name]['argmax_acceptable'] for r in values),
                                       mean_acceptable_mass=float(np.mean([r[name]['acceptable_mass'] for r in values])))
                             for name in ('pi','mu')}) for category,values in groups.items()}
    if source_identity(('relational_diagnostics.py',))!=identity:
        raise ValueError('Diagnostic source or native library changed during evaluation')
    return dict(model_sha256=model_sha,fixtures_sha256=fixture_sha,**identity,
                scope='Development immediate-safety masks, not held-out strength tests or eventual Q labels',
                config=vars(args),summary=summary,positions=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--fixtures',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--max-nodes',type=int,default=12000)
    parser.add_argument('--max-edges',type=int,default=1000000)
    parser.add_argument('--alpha',type=float,default=.03)
    parser.add_argument('--beta',type=float,default=.1)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=evaluate(args.checkpoint,args.fixtures,args)
    write_json(Path(args.output),result)
    print(json.dumps(result['summary']),flush=True)
