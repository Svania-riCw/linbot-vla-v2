"""Continue an audited joint-delta or absolute-spline DCP run without changing its contract."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.action_contract import digest, resolve_contract
from ysj_utils.action_experiment import environment

DEFAULT_RUN = Path('/2025233210/ysj/lingbot_ws/runs/joint_pair_5k_20261017_v3/bundle/JDelta/train')


def inspect_checkpoint(path, world_size):
    """Check all saved ranks and DCP shard extents without loading model tensors."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    path = Path(path)
    for component in ('model', 'optimizer'):
        reader = FileSystemReader(path / component)
        metadata = reader.read_metadata()
        if not metadata.state_dict_metadata or not metadata.storage_data:
            raise ValueError(f'Empty {component} checkpoint: {path}')
        for entry in metadata.storage_data.values():
            shard = path / component / entry.relative_path
            if not shard.is_file() or shard.stat().st_size < entry.offset + entry.length:
                raise ValueError(f'Missing/truncated checkpoint shard: {shard}')
    expected = int(path.name.removeprefix('global_step_'))
    scheduler = None
    required = {'global_step', 'lr_scheduler', 'train_dataloader', 'environ_meter', 'torch_rng_state'}
    for rank in range(world_size):
        # Trusted checkpoints created by this project contain non-tensor loader state.
        state = torch.load(path / 'extra_state' / f'extra_state_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        if not required <= state.keys() or state['global_step'] != expected:
            raise ValueError(f'Incomplete/mismatched training state at rank {rank}: {path}')
        current = state['lr_scheduler']
        if current['last_epoch'] != expected:
            raise ValueError(f'Scheduler counter differs from optimizer step: {path}')
        if scheduler is not None and current != scheduler:
            raise ValueError(f'Ranks disagree on scheduler state: {path}')
        scheduler = current
    return dict(path=str(path), step=expected, saved_lrs=scheduler['_last_lr'])


def select_checkpoint(source, output, world_size, source_steps):
    rejected = []
    candidates = sorted((output / 'checkpoints').glob('global_step_*'),
                        key=lambda p: int(p.name.removeprefix('global_step_')), reverse=True)
    for path in candidates:
        try:
            info = inspect_checkpoint(path, world_size)
            if info['step'] < source_steps:
                raise ValueError('Continuation checkpoint predates its source')
            return info, rejected
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as exc:
            rejected.append(dict(path=str(path), error=repr(exc)))
    return inspect_checkpoint(source / 'checkpoints' / f'global_step_{source_steps}', world_size), rejected


def continuation_config(source_config, output, checkpoint, target, prefetch_factor=None):
    import copy
    config = copy.deepcopy(source_config)
    train = config['train']
    train.update(output_dir=str(output), max_steps=target, enable_resume=True,
                 load_checkpoint_path=checkpoint, require_resume=True,
                 allow_partial_checkpoint=False,
                 lr_schedule_steps=source_config['train'].get('lr_schedule_steps') or source_config['train']['max_steps'])
    if prefetch_factor is not None:
        config['data']['prefetch_factor'] = prefetch_factor
    return config


def build_commands(config, output, gpus, target, python=sys.executable, use_compile=True):
    train = [python, '-m', 'torch.distributed.run', '--standalone',
             f'--nproc-per-node={len(gpus)}', str(ROOT / 'tasks/vla/train_lingbotvla.py'), str(config)]
    evaluate = [python, str(ROOT / 'ysj_utils/evaluate.py'), '--mode', 'full',
                '--checkpoint', str(output / f'checkpoints/global_step_{target}/hf_ckpt'),
                '--gpus', ','.join(gpus), '--seed', '0', '--port-base', '19330',
                '--use-compile', str(use_compile).lower(), '--summary-output',
                str(output / f'evaluation_compile_{str(use_compile).lower()}.json')]
    return train, evaluate


def validate_source_experiment(train, action_mode, experiment):
    if experiment == 'joint_delta':
        if action_mode != 'joint_delta':
            raise ValueError('Source must be a joint-delta run')
    elif experiment == 'spline_absolute':
        expected = dict(spline_control_points=16, spline_degree=3, mask_action_padding=False,
                        enable_fp32=True, enable_mixed_precision=True, joint_ee_aux_weight=-1.)
        if action_mode != 'joint_absolute' or any(train.get(k) != v for k, v in expected.items()):
            raise ValueError('Source must be the audited fp32 absolute-joint 16-point cubic spline run with baseline padding')
        if train.get('flow_gibbs_epsilon', 0) or any(train.get(k) is not None for k in ('flow_dimension_weights','flow_horizon_weights')):
            raise ValueError('Spline continuation must not add weighted loss')
    else:
        raise ValueError('Unknown continuation experiment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/2025233210/ysj/lingbot_ws/outputs/joint_delta_5k_to_10k'))
    parser.add_argument('--additional-steps', type=int, default=5000)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--gpu-wait-seconds', type=int, default=259200)
    parser.add_argument('--prefetch-factor', type=int, default=1, help='Reduce in-flight batches for the current 4GB /dev/shm; keep worker count and sampler')
    parser.add_argument('--experiment', choices=['joint_delta','spline_absolute'], default='joint_delta')
    parser.add_argument('--inference-compile', choices=['true','false'], default='true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('Use a separate output directory outside the source run')
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or not all(x.isdigit() for x in gpus):
        parser.error('GPUs must be distinct numeric IDs')
    if args.additional_steps < 1 or args.gpu_wait_seconds < 0 or args.prefetch_factor < 1:
        parser.error('additional steps must be positive; GPU wait must be nonnegative')
    original_path = source / 'lingbotvla_cli.yaml'
    original = yaml.safe_load(original_path.read_text(encoding='utf-8'))
    t = original['train']
    source_steps = int(t['max_steps'])
    target = source_steps + args.additional_steps
    if t['global_batch_size'] != 128 or t['micro_batch_size'] * len(gpus) * t['gradient_accumulation_steps'] != 128:
        parser.error('Preserve the original world size / micro batch / global BS128')
    if t['ckpt_manager'] != 'dcp' or t.get('training_time_rtc', False):
        parser.error('This script requires the existing non-RTC DCP run')
    contract = resolve_contract(source / f'checkpoints/global_step_{source_steps}/hf_ckpt')
    validate_source_experiment(t, contract['mode'], args.experiment)
    receipt_path = Path(contract['norm_stats'] + '.contract.json')
    receipt = json.loads(receipt_path.read_text())
    for path, expected in receipt.get('label_manifests', {}).items():
        if digest(path) != expected:
            raise ValueError(f'EE labels changed: {path}')
    identity = dict(source_run=str(source), source_config_sha256=digest(original_path),
                    target_steps=target, source_steps=source_steps, gpus=gpus,
                    norm_sha256=digest(contract['norm_stats']),
                    robot_sha256=digest(contract['robot_config']),
                    receipt_sha256=digest(receipt_path),
                    train_list_sha256=digest(original['data']['train_path']),
                    lr_schedule_steps=t.get('lr_schedule_steps') or source_steps,
                    prefetch_factor=args.prefetch_factor,
                    inference_compile=args.inference_compile == 'true')
    if args.experiment != 'joint_delta':
        identity['experiment'] = args.experiment
    config_path = output / 'continuation.yaml'
    train_cmd, eval_cmd = build_commands(config_path, output, gpus, target,
                                         use_compile=args.inference_compile == 'true')

    def prepare_plan():
        if (output / 'continuation.json').exists():
            saved = json.loads((output / 'continuation.json').read_text())
            if saved != identity:
                raise ValueError('Continuation arguments or source files changed; use a new output directory')
        elif output.exists() and any(output.iterdir()):
            raise FileExistsError('Refusing to reuse an unrelated output directory')
        checkpoint, rejected = select_checkpoint(source, output, len(gpus), source_steps)
        if checkpoint['step'] > target:
            raise ValueError('Checkpoint is newer than the requested total budget')
        return dict(identity=identity, checkpoint=checkpoint, rejected_incomplete=rejected,
                    train_command=train_cmd, eval_command=eval_cmd,
                    note='5k schedule extended at its final LR; not a fresh 10k cosine run')

    if args.dry_run:
        print(json.dumps(prepare_plan(), indent=2))
        return
    if os.name != 'posix':
        raise RuntimeError('Execute in the configured Linux model environment')
    import fcntl
    # A sibling lock does not make a new output directory appear nonempty.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare_plan()
        output.mkdir(exist_ok=True)
        (output / 'continuation.json').write_text(json.dumps(identity, indent=2))
        config = continuation_config(original, output, plan['checkpoint']['path'], target, args.prefetch_factor)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        (output / f'launch_{stamp}.json').write_text(json.dumps(plan, indent=2))
        env = environment()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        env['PYTHONUNBUFFERED'] = '1'
        from ysj_utils.run_joint_pair import wait_idle
        def status(state, **extra):
            data = dict(state=state, at=datetime.now(timezone.utc).isoformat(),
                        source_checkpoint=plan['checkpoint'], target_steps=target, **extra)
            temporary = output/'status.tmp'
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(output/'status.json')
            print(json.dumps(data), flush=True)
        try:
            if plan['checkpoint']['step'] < target:
                status('WAITING_FOR_GPU')
                wait_idle(gpus, args.gpu_wait_seconds)
                status('TRAINING', log=str(output/f'train_{stamp}.log'))
                with (output / f'train_{stamp}.log').open('x') as log:
                    subprocess.run(train_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result_path = output / f'evaluation_compile_{args.inference_compile}.json'
            if not args.no_eval:
                if result_path.exists():
                    report = json.loads(result_path.read_text())
                    if report.get('status') != 'COMPLETE' or report.get('use_compile') != identity['inference_compile']:
                        raise RuntimeError('Previous evaluation incomplete; preserve its report and use a fresh summary path')
                    status('COMPLETE', rates=report['rates'])
                    return
                status('WAITING_FOR_EVAL_GPU')
                wait_idle(gpus, args.gpu_wait_seconds)
                status('EVALUATING', log=str(output/f'eval_{stamp}.log'))
                with (output / f'eval_{stamp}.log').open('x') as log:
                    subprocess.run(eval_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(result_path.read_text())
                if report['status'] != 'COMPLETE' or report['total_episodes'] != 350 or report['use_compile'] != identity['inference_compile']:
                    raise RuntimeError('Continuation evaluation incomplete or wrong compile flag')
                status('COMPLETE', rates=report['rates'])
            else:
                status('TRAIN_COMPLETE_NO_EVAL')
        except BaseException as exc:
            status('FAILED', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
