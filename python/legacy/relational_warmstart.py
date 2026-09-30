"""Initialize relational policy and chosen-action Q from verified human histories."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch

from legacy.klent import digest, publish, segmented_log_softmax
from legacy.relational_data import human_examples
from legacy.relational_train import graph, outputs, save_model, source_identity, work_batches
from legacy.train import write_json


def epoch(model, histories, rows, args, optimizer=None, seed=0, progress=None):
    model.train(optimizer is not None)
    order = np.random.default_rng(seed).permutation(len(rows)) if optimizer else np.arange(len(rows))
    totals = np.zeros(4)
    for start in range(0,len(rows),args.batch):
        selected = [rows[int(i)] for i in order[start:start+args.batch]]
        graphs = [graph(histories[r['game']][:r['ply']],model,args) for r in selected]
        if optimizer:optimizer.zero_grad(set_to_none=True)
        offset=0
        with torch.set_grad_enabled(optimizer is not None):
            for group in work_batches(graphs,args):
                examples=selected[offset:offset+len(group)]
                chosen=[]
                for item,row in zip(group,examples,strict=True):
                    matches=np.flatnonzero(np.all(item.actions==np.asarray(row['action']),axis=1))
                    if len(matches)!=1 or item.player!=row['player']:
                        raise ValueError('Human action/phase disagrees with exact encoder')
                    chosen.append(int(matches[0]))
                batch,logits,q=outputs(model,group,args)
                picked=batch['action_offsets'][:-1]+torch.tensor(chosen,device=args.device)
                targets=torch.tensor([r['target'] for r in examples],device=args.device)
                logpi=segmented_log_softmax(logits,batch['action_owner'],len(group))
                ce=-logpi[picked].mean()
                mse=(q[picked]-targets).square().mean()
                if optimizer:((ce+mse)*(len(group)/len(selected))).backward()
                offsets=batch['action_offsets'].cpu().tolist()
                correct=sum(int(logits[offsets[i]:offsets[i+1]].argmax())==chosen[i] for i in range(len(group)))
                totals+=np.array([ce.item()*len(group),mse.item()*len(group),correct,
                                  int(((q[picked]>0)==(targets>0)).sum())])
                offset+=len(group)
        if optimizer:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
            optimizer.step()
        if progress:progress(min(start+args.batch,len(rows)),len(rows))
    return dict(policy_ce=totals[0]/len(rows),q_mse=totals[1]/len(rows),
                policy_accuracy=totals[2]/len(rows),chosen_q_sign_accuracy=totals[3]/len(rows))


def fit(args, config=None):
    from legacy.relational_model import ModelConfig,RelationalNet
    output=Path(args.output)
    if output.exists():raise ValueError('Warm-start output must be a new directory')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    histories,rows,data_identity=human_examples(args.corpus,args.positions,args.seed)
    model=RelationalNet(config or ModelConfig()).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    identity=dict(kind='relational-human-policy-q-v1',config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  model_config=asdict(model.config),parameters=sum(p.numel() for p in model.parameters()),
                  data=data_identity,**source_identity(('relational_warmstart.py','relational_data.py','human_corpus.py','corpus_warmstart.py')))
    output.parent.mkdir(parents=True,exist_ok=True)
    metrics=[]
    best=None
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='relational-fit-') as temporary:
        temporary=Path(temporary)
        save_model(temporary/'initial_model.pt',model)
        initial_validation=epoch(model,histories,rows['validation'],args)
        for number in range(1,args.epochs+1):
            last=[0.,None]
            def progress(stage,completed,total):
                if stage!=last[1] or time.monotonic()-last[0]>.5 or completed==total:
                    write_json(output.with_suffix('.status.json'),dict(stage=stage,epoch=number,epochs=args.epochs,
                               completed=completed,total=total,seconds=time.perf_counter()-started))
                    last[:]=[time.monotonic(),stage]
            train=epoch(model,histories,rows['train'],args,optimizer,args.seed+number,
                        lambda completed,total:progress('training',completed,total))
            validation=epoch(model,histories,rows['validation'],args,
                             progress=lambda completed,total:progress('validation',completed,total))
            metric=dict(epoch=number,train=train,validation=validation)
            metrics.append(metric)
            score=validation['policy_ce']+validation['q_mse']
            if best is None or score<best['score']:
                best=dict(epoch=number,score=score)
                save_model(temporary/'model.pt',model)
                torch.save(optimizer.state_dict(),temporary/'optimizer.pt')
            print(json.dumps(metric),flush=True)
        if (digest(Path(args.corpus)/'manifest.json')!=data_identity['manifest_sha256']
                or digest(Path(args.corpus)/'games.jsonl')!=data_identity['histories_sha256']):
            raise ValueError('Human history source changed during fitting')
        for files in data_identity['shards'].values():
            for path,expected in files.items():
                if digest(Path(path))!=expected:raise ValueError('Human split shard changed during fitting')
        report=dict(epochs=metrics,initial_validation=initial_validation,selected=best,seconds=time.perf_counter()-started,
                    train_positions=len(rows['train']),validation_positions=len(rows['validation']),
                    gpu_memory_peak_mb=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0,
                    target='Human chosen actions and terminal STM outcomes; unknown action Q values remain unlabeled',
                    promotion=False)
        def writer(stage):
            shutil.copy2(temporary/'model.pt',stage/'model.pt')
            shutil.copy2(temporary/'initial_model.pt',stage/'initial_model.pt')
            shutil.copy2(temporary/'optimizer.pt',stage/'optimizer.pt')
            (stage/'report.json').write_text(json.dumps(report,indent=2))
        publish(output,identity,writer,report)
        write_json(output.with_suffix('.status.json'),dict(stage='finished',**report))
    return report


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name,default in [('positions',0),('epochs',12),('batch',8),('max-nodes',12000),('max-edges',600000),('seed',1729)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--lr',type=float,default=.0001)
    parser.add_argument('--grad-clip',type=float,default=1.)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if min(args.epochs,args.batch,args.max_nodes,args.max_edges)<1 or args.positions<0 or args.positions==1:
        parser.error('Positive budgets required; positions0 means all available positions')
    if not math.isfinite(args.lr) or not math.isfinite(args.grad_clip) or min(args.lr,args.grad_clip)<=0:
        parser.error('Positive finite optimization coefficients required')
    return args


if __name__=='__main__':fit(parse_args())
