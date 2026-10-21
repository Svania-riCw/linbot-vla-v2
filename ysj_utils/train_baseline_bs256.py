"""BS256 baseline, one 20k cosine schedule, strict resume and 350-episode evals."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.action_contract import action_mode, canonical_norm_types, digest, resolve_contract, supervision_config

TARGETS = (50, 5000, 10000, 20000)
EVAL_TARGETS = (5000, 10000, 20000)
STEP_RE = re.compile(r'\bStep (\d+)/\d+,.*?Loss ([\d.eE+-]+),.*?GradNorm ([\d.eE+-]+), LR ([\d.eE+-]+),.*?StepTime ([\d.]+)s')


def requested_stages(final_only):
    return ((50, 5000, 20000), (20000,)) if final_only else (TARGETS, EVAL_TARGETS)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def stage_config(source, run, target, checkpoint=None):
    """Only budget/resume/IO change; baseline action/model/precision stay intact."""
    if target not in TARGETS:
        raise ValueError('Unsupported milestone')
    config = copy.deepcopy(source)
    train = config['train']
    train.update(output_dir=str(Path(run)/f'step_{target:05d}'/'train'),
                 global_batch_size=256, micro_batch_size=16, gradient_accumulation_steps=2,
                 max_steps=target, lr_schedule_steps=20000,
                 lr=1e-4, lr_min=5e-5, lr_decay_style='cosine',
                 save_steps=1000, enable_resume=checkpoint is not None,
                 require_resume=checkpoint is not None, allow_partial_checkpoint=False,
                 load_checkpoint_path=str(checkpoint) if checkpoint else None)
    # Large collated batches include both accumulation micros. Bound shared-memory
    # prefetch in the 4GiB container; this changes resources, not the sampler.
    config['data'].update(train_path=str(Path(run)/'train.txt'),
                          robot_config_root=str(Path(run)/'robot_configs'),
                          norm_stats_file=str(Path(run)/'norm_stats.json'),
                          num_workers=2, prefetch_factor=1)
    return config


def check_source(source, robot):
    t = source['train']
    if (t['global_batch_size'], t['micro_batch_size'], t.get('gradient_accumulation_steps', 1)) != (128, 16, 1):
        raise ValueError('Expected the actual BS128 baseline snapshot')
    if not t['enable_fp32'] or not t['enable_mixed_precision'] or t['loss_type'] != 'L1_fm':
        raise ValueError('Baseline precision/loss differs')
    if t['lr'] != 1e-4 or t['lr_min'] != 5e-5 or t['lr_decay_style'] != 'cosine':
        raise ValueError('Unexpected baseline LR schedule')
    if action_mode(robot) != 'joint_absolute' or len(robot['states']) != 2:
        raise ValueError('Require baseline absolute joints/grippers without EE state')
    if (t.get('mask_action_padding', False) or t.get('training_time_rtc', False)
            or t.get('joint_ee_aux_weight', -1) != -1 or t.get('spline_control_points', 0)
            or t.get('flow_gibbs_epsilon', 0)
            or any(t.get(k) is not None for k in ('flow_dimension_weights', 'flow_horizon_weights'))):
        raise ValueError('Source contains a separate action/supervision experiment')
    if t['ckpt_manager'] != 'dcp' or t.get('enable_fsdp_offload', False):
        raise ValueError('Require DCP and no FSDP offload for gradient accumulation')


def validate_grid_report(report):
    from ysj_utils.evaluate import validate_results, scheme
    if (report['status'] != 'COMPLETE' or report['process_exit_code'] != 0
            or report['total_episodes'] != 350 or report['use_compile'] is not False):
        raise ValueError('Evaluation did not finish with the prescribed protocol')
    return validate_results(Path(report['run_dir'])/'_result.txt', **scheme('full'))


def verify_updates(text, first, target):
    import math
    from ysj_utils.run_joint_pair import NONFINITE
    if NONFINITE.search(text):
        raise ValueError('Nonfinite training loss/gradient in log')
    updates = STEP_RE.findall(text)
    if [int(x[0]) for x in updates] != list(range(first, target+1)):
        raise ValueError('Missing/duplicate optimizer updates')
    if not all(math.isfinite(float(v)) for row in updates for v in row[1:]):
        raise ValueError('Nonfinite training metric')
    if first > 1 and ('Load distributed checkpoint from' not in text
                      or 'Starting training from scratch' in text):
        raise ValueError('Continuation did not restore the required DCP')
    tail = updates[-min(20, len(updates)):]
    return dict(first_step=first, last_step=target, updates=len(updates),
                last_loss=float(updates[-1][1]), last_grad_norm=float(updates[-1][2]),
                last_lr=float(updates[-1][3]),
                tail_step_seconds_mean=sum(float(x[4]) for x in tail)/len(tail))


def prepare(args):
    from ysj_utils.action_experiment import validate_stats
    source_path = args.source_run/'lingbotvla_cli.yaml'
    source = yaml.safe_load(source_path.read_text())
    robot_path = Path(source['data']['robot_config_root'])/'robotwin.yaml'
    robot = yaml.safe_load(robot_path.read_text())
    check_source(source, robot)
    norm = Path(source['data'].get('norm_stats_file') or robot['norm_stats'])
    validate_stats(norm, robot)
    datasets = [line.split()[1] for line in args.train_list.read_text().splitlines() if line.strip()]
    if len(datasets) != 50 or len(set(datasets)) != 50 or not all(Path(p).exists() for p in datasets):
        raise ValueError('Expected the original 50 accessible clean task datasets')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir/'source_baseline.yaml').write_bytes(source_path.read_bytes())
    (args.run_dir/'train.txt').write_bytes(args.train_list.read_bytes())
    (args.run_dir/'norm_stats.json').write_bytes(norm.read_bytes())
    robot = copy.deepcopy(robot)
    robot['norm_stats'] = str(args.run_dir/'norm_stats.json')
    (args.run_dir/'robot_configs').mkdir()
    (args.run_dir/'robot_configs/robotwin.yaml').write_text(yaml.safe_dump(robot, sort_keys=False))
    first_config = stage_config(source, args.run_dir, 50)
    receipt = dict(norm_sha256=digest(args.run_dir/'norm_stats.json'),
                   robot_sha256=digest(args.run_dir/'robot_configs/robotwin.yaml'),
                   train_list_sha256=digest(args.run_dir/'train.txt'), mode='joint_absolute',
                   chunk_size=source['train'].get('chunk_size', 50),
                   norm_type=canonical_norm_types(source['data']['norm_type']),
                   supervision=supervision_config(source['train']), label_manifests={})
    atomic_json(args.run_dir/'norm_stats.json.contract.json', receipt)
    for target in TARGETS:
        directory = args.run_dir/f'step_{target:05d}'
        (directory/'train').mkdir(parents=True)
        cfg = stage_config(source, args.run_dir, target)
        (directory/'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (directory/'train/lingbotvla_cli.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        resolve_contract(directory/f'train/checkpoints/global_step_{target}/hf_ckpt')
    original_model = Path(source['model']['model_path'])
    index = original_model/'model.safetensors.index.json'
    shards = set(json.loads(index.read_text())['weight_map'].values())
    if not all((original_model/name).is_file() for name in shards):
        raise FileNotFoundError('Missing original pretrained model shard')
    protected = [args.run_dir/'source_baseline.yaml', args.run_dir/'train.txt',
                 args.run_dir/'norm_stats.json', args.run_dir/'norm_stats.json.contract.json',
                 args.run_dir/'robot_configs/robotwin.yaml']
    plan = dict(source_run=str(args.source_run), source_config_sha256=digest(source_path),
                initial_model=str(original_model), initial_model_index_sha256=digest(index),
                model=source['model'], global_batch_size=256, micro_batch_size=16,
                accumulation_steps=2, world_size=8, total_updates=20000,
                schedule_steps=20000, eval_targets=list(EVAL_TARGETS), probe_steps=50,
                gpus=args.gpus.split(','), protected_files={str(p):digest(p) for p in protected},
                created_at=datetime.now(timezone.utc).isoformat(),
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                protocol='7 tasks x clean/random x25=350; seed0, fp32, compile=False, action length50',
                matching_samples={'5000':1280000,'10000':2560000,'20000':5120000},
                resource_changes={'num_workers':2,'prefetch_factor':1},
                note='Probe updates1-50 belong to this 20k run. Every milestone strictly restores full training state; cosine never restarts.')
    atomic_json(args.run_dir/'plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, default=Path('/2025233210/xiesy/lingbot_ws/outputs/baseline_gbs128_10k'))
    parser.add_argument('--train-list', type=Path, default=ROOT/'xsy_utils/configs/robotwin_clean.txt')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--gpu-wait-seconds', type=int, default=604800)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--eval-final-only', action='store_true',
                        help='Skip intermediate evals and the10k restart; train directly to20k, then evaluate350')
    args = parser.parse_args()
    args.run_dir=args.run_dir.resolve();args.source_run=args.source_run.resolve()
    ids=args.gpus.split(',')
    if len(ids)!=8 or len(set(ids))!=8 or not all(i.isdigit() for i in ids):
        parser.error('This experiment requires eight distinct GPU IDs')
    if os.name != 'posix':
        raise RuntimeError('Run in the configured Linux model environment')
    import fcntl
    from ysj_utils.action_experiment import environment
    from ysj_utils.resume_joint_delta import inspect_checkpoint
    from ysj_utils.run_joint_pair import wait_idle
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with args.run_dir.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan_path=args.run_dir/'plan.json'
        plan=json.loads(plan_path.read_text()) if plan_path.exists() else prepare(args)
        if plan['gpus']!=ids or plan['source_run']!=str(args.source_run):
            raise ValueError('Run identity changed')
        for path,sha in plan['protected_files'].items():
            if digest(path)!=sha:raise ValueError('Training input changed: '+path)
        if args.eval_final_only and not plan.get('eval_final_only', False):
            stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            change=dict(at=datetime.now(timezone.utc).isoformat(),reason='Explicit --eval-final-only request',
                        previous_eval_targets=plan['eval_targets'],new_eval_targets=[20000])
            atomic_json(args.run_dir/f'protocol_change_{stamp}.json',change)
            plan.update(eval_final_only=True,eval_targets=[20000])
            atomic_json(plan_path,plan)
        targets,eval_targets=requested_stages(plan.get('eval_final_only',False))
        if args.prepare_only:
            print(json.dumps(plan,indent=2),flush=True);return
        source=yaml.safe_load((args.run_dir/'source_baseline.yaml').read_text())
        env=environment();env.update(CUDA_VISIBLE_DEVICES=','.join(ids),PYTHONUNBUFFERED='1',
                                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        env.pop('LINGBOT_INFERENCE_CONFIG',None);env.pop('NORM_STATS',None)
        def update(state,**values):
            record=dict(state=state,at=datetime.now(timezone.utc).isoformat(),**values)
            atomic_json(args.run_dir/'status.json',record);print(json.dumps(record),flush=True)
        previous=None;results={}
        try:
            for target in targets:
                directory=args.run_dir/f'step_{target:05d}';train_dir=directory/'train'
                final=train_dir/f'checkpoints/global_step_{target}'
                candidates=sorted((train_dir/'checkpoints').glob('global_step_*'),
                                  key=lambda p:int(p.name.removeprefix('global_step_')),reverse=True)
                selected=None
                for p in candidates:
                    try:selected=inspect_checkpoint(p,8)
                    except (OSError,ValueError,KeyError,RuntimeError,EOFError) as exc:
                        print(json.dumps(dict(event='REJECT_INCOMPLETE_CHECKPOINT',path=str(p),error=str(exc))),flush=True);continue
                    break
                if selected is None and previous is not None:selected=inspect_checkpoint(previous,8)
                if selected and selected['step']>target:raise ValueError('Checkpoint exceeds target')
                checkpoint=selected['path'] if selected else None
                start=selected['step'] if selected else 0
                if target>50 and checkpoint is None:raise RuntimeError('Required continuation checkpoint is missing')
                if start<target:
                    update('WAITING_FOR_GPU',target=target,restored_step=start)
                    wait_idle(ids,seconds=args.gpu_wait_seconds)
                    cfg=stage_config(source,args.run_dir,target,checkpoint)
                    config_path=directory/'train.yaml';config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                    log_path=directory/f'training_{stamp}.log'
                    update('PROBE' if target==50 else 'TRAINING',target=target,restored_step=start,log=str(log_path))
                    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                             str(ROOT/'tasks/vla/train_lingbotvla.py'),str(config_path)]
                    with log_path.open('x') as log:
                        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    verified=verify_updates(log_path.read_text(errors='replace'),start+1,target)
                    atomic_json(directory/f'verified_{stamp}.json',verified)
                    print(json.dumps(dict(event='VERIFIED_TRAINING',target=target,**verified)),flush=True)
                saved=inspect_checkpoint(final,8)
                hf=final/'hf_ckpt';index=hf/'model.safetensors.index.json'
                shards=set(json.loads(index.read_text())['weight_map'].values())
                if not shards or any(not(hf/name).is_file() for name in shards):raise FileNotFoundError('Incomplete HF export')
                contract=resolve_contract(hf)
                if contract['mode']!='joint_absolute':raise ValueError('Unexpected inference action space')
                previous=final
                if target not in eval_targets:continue
                complete=None
                for report_path in sorted(directory.glob('evaluation_*.json')):
                    report=json.loads(report_path.read_text())
                    if report.get('checkpoint')!=str(hf):continue
                    if report.get('status')=='COMPLETE':validate_grid_report(report);complete=report;break
                if complete is None:
                    update('WAITING_FOR_EVAL_GPU',target=target)
                    wait_idle(ids,seconds=args.gpu_wait_seconds)
                    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                    report_path=directory/f'evaluation_{stamp}.json';log_path=directory/f'evaluation_{stamp}.log'
                    command=[sys.executable,str(ROOT/'ysj_utils/evaluate.py'),'--mode','full',
                             '--checkpoint',str(hf),'--gpus',','.join(ids),'--seed','0','--port-base','19330',
                             '--use-compile','false','--summary-output',str(report_path)]
                    update('EVALUATING',target=target,checkpoint=str(hf),log=str(log_path))
                    with log_path.open('x') as log:
                        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    complete=json.loads(report_path.read_text());validate_grid_report(complete)
                results[str(target)]=complete
                atomic_json(args.run_dir/'results.json',results)
                update('MILESTONE_COMPLETE',target=target,rates=complete['rates'])
            update('COMPLETE',target=20000,rates=results['20000']['rates'])
        except BaseException as exc:
            update('FAILED',error=repr(exc));raise


if __name__=='__main__':
    main()
