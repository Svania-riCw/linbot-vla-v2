"""CPU diagnostic: project real joint action chunks onto the candidate spline.

Reads only Parquet state/action columns, not images or model weights. Reports
representation error in radians and gripper units; does not evaluate a policy.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lingbotvla.action_spaces.spline import SplineActionCodec
from ysj_utils.action_experiment import verify_pair
from deploy.action_contract import digest, canonical_norm_types

ARM=[0,1,2,3,4,5,7,8,9,10,11,12]
GRIP=[6,13]


def summarize_chunks(states,actions,pad,stats,points=16,degree=3):
    if actions.ndim!=3 or actions.shape[-1]!=14 or states.shape!=(len(actions),14) or pad.shape!=actions.shape[:2] or pad.dtype!=np.bool_:
        raise ValueError('Expected states [N,14], actions [N,H,14], padding [N,H]')
    if not np.isfinite(states).all() or not np.isfinite(actions[~pad]).all():
        raise ValueError('Nonfinite valid actions/states')
    target=np.concatenate([actions[...,ARM]-states[:,None,ARM],actions[...,GRIP]],axis=-1)
    keys=['action.arm.position','action.effector.position']
    low=np.concatenate([stats[k]['q01'] for k in keys]).astype('float32')
    high=np.concatenate([stats[k]['q99'] for k in keys]).astype('float32')
    scale=high-low+1e-6
    normalized=(target-low)/scale*2-1
    codec=SplineActionCodec(actions.shape[1],points,degree)
    tensor=torch.tensor(normalized,dtype=torch.float32)
    with torch.no_grad():
        coefficients,_=codec.encode(tensor,torch.tensor(pad),torch.ones_like(tensor,dtype=torch.bool))
        reconstructed=codec.decode(coefficients).numpy()
    errors=(reconstructed-normalized)[~pad]
    physical=errors*scale/2
    def summary(value):
        return dict(mae=float(np.abs(value).mean()),p95_abs=float(np.quantile(np.abs(value),.95)),max_abs=float(np.abs(value).max()))
    changes=np.any(np.abs(np.diff(actions[...,GRIP],axis=1))>.1,axis=-1)&~pad[:,1:]&~pad[:,:-1]
    return dict(chunks=len(actions),valid_steps=int((~pad).sum()),control_points=points,degree=degree,
                normalized=dict(rmse=float(np.sqrt(np.mean(errors**2))),**summary(errors)),
                joint_radians=summary(physical[:,:12]),gripper_units=summary(physical[:,12:]),
                abrupt_gripper_transitions=int(changes.sum()),
                note='Projection error only; inspect abrupt gripper actions and contact phases before training. No universal acceptance threshold or success-rate claim.')


def sampled_chunks(dataset,count,horizon):
    import pyarrow.parquet as pq
    rows={}
    for path in sorted((Path(dataset)/'data').rglob('*.parquet')):
        table=pq.read_table(path,columns=['episode_index','frame_index','observation.state','action']).to_pydict()
        for ep,frame,state,action in zip(*(table[k] for k in ('episode_index','frame_index','observation.state','action'))):
            rows.setdefault(int(ep),[]).append((int(frame),state,action))
    episodes=[]
    for ep,values in sorted(rows.items()):
        values.sort(key=lambda row:row[0])
        if [v[0] for v in values]!=list(range(len(values))):
            raise ValueError(f'{dataset} episode {ep}: invalid frame indices')
        episodes.append((np.array([v[1] for v in values],dtype='float32'),np.array([v[2] for v in values],dtype='float32')))
    total=sum(len(s) for s,a in episodes)
    if not total: raise ValueError(f'No data: {dataset}')
    selected=set(np.random.default_rng(42).choice(total,size=min(count,total),replace=False).tolist())
    offset=0
    for states,actions in episodes:
        for frame in sorted(i-offset for i in selected if offset<=i<offset+len(states)):
            indices=frame+np.arange(horizon)
            yield states[frame],actions[np.minimum(indices,len(states)-1)],indices>=len(states)
        offset+=len(states)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=128)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    _,directory=verify_pair(args.bundle,'Spline')
    cfg=yaml.safe_load((directory/'train.yaml').read_text(encoding='utf-8'))
    norm_types=canonical_norm_types(cfg['data']['norm_type'])
    if any(norm_types.get(k)!='bounds_99_woclip' for k in ('arm.position','effector.position')):
        raise ValueError('Physical-unit diagnostic currently requires bounds_99_woclip normalization')
    stats_path=directory/'norm_stats.json'
    receipt=json.loads(Path(str(stats_path)+'.contract.json').read_text(encoding='utf-8'))
    if receipt['norm_sha256']!=digest(stats_path): raise ValueError('Normalization changed')
    if args.samples<1: raise ValueError('samples must be positive')
    datasets=[line.split()[1] for line in (args.bundle/'train.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
    samples=[]; provenance=[]
    for index,dataset in enumerate(datasets):
        remaining=args.samples-len(samples)
        if remaining<=0: break
        count=max(1,(remaining+len(datasets)-index-1)//(len(datasets)-index))
        batch=list(sampled_chunks(dataset,count,cfg['train']['chunk_size']))
        samples.extend(batch); provenance.append(dict(dataset=dataset,chunks=len(batch)))
    states,actions,pad=map(np.stack,zip(*samples))
    result=summarize_chunks(states,actions,pad,json.loads(stats_path.read_text())['norm_stats'],
                            cfg['train']['spline_control_points'],cfg['train']['spline_degree'])
    result.update(datasets=provenance,sampling_seed=42,norm_sha256=digest(stats_path),train_list_sha256=digest(args.bundle/'train.txt'))
    with args.output.open('x',encoding='utf-8') as file: json.dump(result,file,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
