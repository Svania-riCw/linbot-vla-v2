"""Sequential JDelta/JointEE preparation, smoke checks, 5k training and evaluation.

Run inside the team's Linux model environment. No credentials are accepted or
stored. Use a dedicated checkout; leave teammates' jobs and checkouts alone.
"""
import argparse
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ysj_utils.action_experiment import environment


@dataclass
class Stage:
    name: str
    command: list
    gpus: list


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def build_plan(args,python=sys.executable):
    bundle=Path(args.bundle).resolve(); run=Path(args.run_dir).resolve()
    prefix=[python,str(ROOT/'ysj_utils/action_experiment.py')]
    gpus=list(args.gpus)
    stages=[Stage('labels',[python,str(ROOT/'ysj_utils/prepare_ee_labels.py'),'--train-list',str(args.train_list),
                   '--raw-root',str(args.raw_root),'--raw-setting',args.raw_setting,'--output',str(args.ee_sidecar_root),
                   '--resume-complete'],[]),
            Stage('prepare',prefix+['prepare','--base-config',str(args.base_config),'--robot-config',str(args.robot_config),
                   '--train-list',str(args.train_list),'--ee-sidecar-root',str(args.ee_sidecar_root),'--output',str(bundle),
                   '--world-size',str(len(gpus)),'--micro-batch',str(args.micro_batch),'--steps',str(args.steps)],[])]
    for group in ('JDelta','JointEE0','JointEE'):
        stages.append(Stage('norm_'+group,prefix+['norm','--bundle',str(bundle),'--group',group],[]))
    def train(group,kind):
        return Stage(kind+'_'+group,prefix+['train','--bundle',str(bundle),'--group',group,'--stage',kind],gpus[:1] if kind=='smoke' else gpus)
    def evaluate(group,kind):
        mode='smoke' if kind=='smoke' else 'full'
        directory=bundle/group/('train-smoke' if mode=='smoke' else 'train')
        step=5 if mode=='smoke' else args.steps
        ids=gpus[:1] if mode=='smoke' else gpus
        checkpoint=directory/f'checkpoints/global_step_{step}/hf_ckpt'
        return Stage('eval_'+mode+'_'+group,[python,str(ROOT/'ysj_utils/evaluate.py'),'--mode',mode,
                     '--checkpoint',str(checkpoint),'--gpus',','.join(ids),'--seed',str(args.eval_seed),
                     '--port-base',str(args.port_base),'--summary-output',str(run/f'eval_{mode}_{group}.json')],ids)
    # Both candidates must pass short training+reload+simulation before full budget.
    for group in ('JDelta','JointEE'):
        stages.extend([train(group,'smoke'),evaluate(group,'smoke')])
    for group in ('JDelta','JointEE'):
        stages.append(train(group,'full'))
    for group in ('JDelta','JointEE'):
        stages.append(evaluate(group,'full'))
    return stages


def gpu_snapshot():
    command=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    rows=[]
    for line in result.stdout.splitlines():
        parts=[p.strip() for p in line.split(',')]
        if len(parts)!=6: raise ValueError('Unexpected nvidia-smi output')
        rows.append(dict(index=parts[0],uuid=parts[1],name=parts[2],total_mib=int(parts[3]),used_mib=int(parts[4]),utilization=int(parts[5])))
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],
                             capture_output=True,text=True,check=True).stdout
    return dict(gpus=rows,compute_processes=processes,timestamp=timestamp())


def wait_idle(ids,seconds=300):
    deadline=time.monotonic()+seconds
    while True:
        snapshot=gpu_snapshot()
        selected=[row for row in snapshot['gpus'] if row['index'] in ids]
        if len(selected)!=len(ids): raise ValueError('Requested GPU IDs are not available')
        uuids={row['uuid'] for row in selected}
        occupied=any(any(uuid in line for uuid in uuids) for line in snapshot['compute_processes'].splitlines())
        busy=occupied or any(row['used_mib']>1024 or row['utilization']>10 for row in selected)
        if not busy: return snapshot
        if time.monotonic()>=deadline:
            raise RuntimeError(f'Allocated GPUs {ids} are busy; no existing processes were stopped')
        print(json.dumps(dict(event='WAIT_GPU',gpus=selected,timestamp=timestamp())),flush=True)
        time.sleep(min(10,max(.1,deadline-time.monotonic())))


NONFINITE=re.compile(r'\b(?:[\w/.-]*loss[\w/.-]*|grad(?:ient)?_?norm)[\"\x27]?(?:\s*[:=]\s*|\s+)(?:tensor\()?\s*[\"\x27]?\s*[+-]?(?:nan|inf)\b',re.I)


def stop_owned_process(process):
    if process.poll() is not None: return
    try:
        if os.name=='posix': os.killpg(process.pid,signal.SIGTERM)
        else: process.terminate()
    except ProcessLookupError:
        return
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name=='posix': os.killpg(process.pid,signal.SIGKILL)
        else: process.kill()
        process.wait(timeout=20)


def run_stage(stage,run_dir,state,env,poll_seconds=10,check_gpu=True):
    path=Path(run_dir)/f'{stage.name}.log'
    if path.exists(): raise FileExistsError(f'Log exists for unfinished stage: {path}')
    info=dict(status='STARTING',started_at=timestamp(),command=stage.command,log=str(path))
    state['stages'][stage.name]=info; state['current_stage']=stage.name
    atomic_json(Path(run_dir)/'status.json',state)
    if stage.gpus and check_gpu:
        info['gpu_before']=wait_idle(stage.gpus)
    stage_env=env.copy()
    if stage.gpus: stage_env['CUDA_VISIBLE_DEVICES']=','.join(stage.gpus)
    stage_env['PYTHONUNBUFFERED']='1'
    print(json.dumps(dict(event='START',stage=stage.name,timestamp=timestamp())),flush=True)
    with path.open('x',encoding='utf-8') as output:
        process=subprocess.Popen(stage.command,cwd=ROOT,env=stage_env,stdout=output,stderr=subprocess.STDOUT,
                                 start_new_session=os.name=='posix')
        info.update(status='RUNNING',pid=process.pid)
        atomic_json(Path(run_dir)/'status.json',state)
        position=0; tail=''
        try:
            while True:
                with path.open(encoding='utf-8',errors='replace') as file:
                    file.seek(position); recent=file.read(); position=file.tell()
                tail=(tail+recent)[-16000:]
                if NONFINITE.search(tail):
                    raise RuntimeError(f'Nonfinite loss/gradient detected in {path}')
                code=process.poll()
                if code is not None:
                    # Drain output written between the last read and process exit.
                    with path.open(encoding='utf-8',errors='replace') as file:
                        file.seek(position); tail=(tail+file.read())[-16000:]
                    if NONFINITE.search(tail): raise RuntimeError(f'Nonfinite loss/gradient detected in {path}')
                    break
                info['heartbeat_at']=timestamp()
                atomic_json(Path(run_dir)/'status.json',state)
                print(json.dumps(dict(event='HEARTBEAT',stage=stage.name,pid=process.pid,
                                      tail=tail[-1200:],timestamp=timestamp())),flush=True)
                time.sleep(poll_seconds)
            info['exit_code']=code
            if code: raise RuntimeError(f'{stage.name} exited {code}; inspect {path}')
        except BaseException:
            stop_owned_process(process)
            info.update(status='FAILED',finished_at=timestamp())
            atomic_json(Path(run_dir)/'status.json',state)
            raise
    info.update(status='COMPLETE',finished_at=timestamp())
    atomic_json(Path(run_dir)/'status.json',state)
    print(json.dumps(dict(event='COMPLETE',stage=stage.name,timestamp=timestamp())),flush=True)


def finish_comparison(run_dir,steps):
    from ysj_utils.compare_results import compare
    run_dir=Path(run_dir)
    reports=[json.loads((run_dir/f'eval_full_{group}.json').read_text(encoding='utf-8')) for group in ('JDelta','JointEE')]
    result=compare(*(report['run_dir'] for report in reports),expected_steps=steps)
    atomic_json(run_dir/'comparison.json',result)
    lines=[f'# JDelta vs JointEE, {steps} optimizer updates','', '| Setting | JDelta | JointEE | Difference (pp) |',
           '|---|---:|---:|---:|']
    for setting,values in result['summary'].items():
        lines.append(f"| {setting} | {values['baseline']:.2%} | {values['candidate']:.2%} | {values['difference_pp']:+.2f} |")
    lines += ['','Single training seed; inspect scene consistency and failure videos before claiming a stable improvement.']
    (run_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base-config','robot-config','train-list','raw-root','ee-sidecar-root','bundle','run-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--raw-setting',default='aloha-agilex_clean_50')
    parser.add_argument('--gpus',required=True,help='Explicit allocated GPU IDs, e.g. 0,1,2,3,4,5,6,7')
    parser.add_argument('--micro-batch',type=int,default=16)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--eval-seed',type=int,default=0)
    parser.add_argument('--port-base',type=int,default=19330)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    args.gpus=[value.strip() for value in args.gpus.split(',') if value.strip()]
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or any(not value.isdigit() for value in args.gpus):
        parser.error('gpus must be distinct numeric IDs')
    if args.steps<5 or args.micro_batch<1 or 128%(len(args.gpus)*args.micro_batch):
        parser.error('steps must be >=5, and global BS128 must divide world_size * micro_batch')
    for name in ('base_config','robot_config','train_list','raw_root','ee_sidecar_root','bundle','run_dir'):
        setattr(args,name,getattr(args,name).resolve())
    stages=build_plan(args)
    plan=dict(stages=[asdict(stage) for stage in stages],steps=args.steps,global_batch=128,
              groups=['JDelta','JointEE'],total_formal_eval_episodes=700,
              code_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
    if args.dry_run:
        print(json.dumps(plan,indent=2)); return
    if os.name!='posix': raise RuntimeError('Run the sequential experiment in the configured Linux model environment')
    import fcntl
    args.run_dir.mkdir(parents=True,exist_ok=args.resume)
    with (args.run_dir/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fingerprint=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        status_path=args.run_dir/'status.json'
        if args.resume and status_path.exists():
            state=json.loads(status_path.read_text(encoding='utf-8'))
            if state['plan_sha256']!=fingerprint: raise ValueError('Resume requires identical arguments and code commit')
        else:
            state=dict(status='RUNNING',started_at=timestamp(),plan_sha256=fingerprint,stages={})
            atomic_json(args.run_dir/'plan.json',plan)
        state['status']='RUNNING'
        atomic_json(status_path,state)
        try:
            atomic_json(args.run_dir/'gpu_initial.json',wait_idle(args.gpus))
            for stage in stages:
                if state['stages'].get(stage.name,{}).get('status')=='COMPLETE': continue
                run_stage(stage,args.run_dir,state,environment())
            result=finish_comparison(args.run_dir,args.steps)
            state.update(status='COMPLETE',finished_at=timestamp(),summary=result['summary'])
        except BaseException as exc:
            state.update(status='FAILED',finished_at=timestamp(),error=f'{type(exc).__name__}: {exc}')
            if state.get('current_stage') in state['stages']:
                state['stages'][state['current_stage']].update(status='FAILED',error=state['error'])
            atomic_json(status_path,state)
            raise
        atomic_json(status_path,state)


if __name__=='__main__': main()
