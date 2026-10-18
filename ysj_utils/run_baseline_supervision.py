"""Independent 5k Gibbs/Spline ablations on the team's absolute-joint baseline.

Linux runner: calibrate on numeric training columns, then train/evaluate in tmux.
No EE input, delta actions, or legacy dimension/time weights are introduced.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.action_contract import action_mode, canonical_norm_types, digest, resolve_contract, supervision_config
from ysj_utils.action_experiment import environment, validate_stats


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def absolute_targets(actions, stats):
    # Dataset 14-vector: left joints/gripper, right joints/gripper.
    indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    keys = ['action.arm.position', 'action.effector.position']
    low = np.concatenate([stats[k]['q01'] for k in keys]).astype('float32')
    high = np.concatenate([stats[k]['q99'] for k in keys]).astype('float32')
    return (actions[..., indices] - low) / (high - low + 1e-6) * 2 - 1, (high - low + 1e-6) / 2


def calibrate(targets, pad, source_points=128, seed=7042, tolerance=.01):
    """Paper Appendix D: choose smallest bandwidth with both marginal RV <= .01.

    Masked RMS cost handles variable valid trajectory length. Pairwise proxies
    average over contexts (not a conditional marginal preservation guarantee).
    Precomputed Z avoids rank/minibatch normalization and its gradient bias.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((source_points,) + targets.shape[1:]).astype('float32')
    valid = np.broadcast_to(~pad[..., None], targets.shape)
    # Calculate per-target valid-dimensional costs with independent source draws.
    costs = np.stack([np.sqrt(np.where(valid, (x[None] - targets)**2, 0).sum((1, 2)) /
                              valid.sum((1, 2))) for x in noise])
    records = []
    for epsilon in np.geomspace(.02, 8., 32):
        logs = -costs.astype('float64') / epsilon
        shift = logs.max()
        kernel = np.exp(logs - shift)
        row, column = kernel.mean(1), kernel.mean(0)
        rv_source = float(row.var() / row.mean()**2)
        rv_target = float(column.var() / column.mean()**2)
        z = float(np.log(kernel.mean()) + shift)
        records.append(dict(epsilon=float(epsilon), log_normalizer=z,
                            rv_source=rv_source, rv_target=rv_target,
                            ess_fraction=float(kernel.mean()**2 / np.mean(kernel**2))))
    selected = next((r for r in records if max(r['rv_source'], r['rv_target']) <= tolerance), None)
    if selected is None:
        raise ValueError('No bandwidth passes marginal-tilt diagnostic; do not train blindly')
    return dict(paper='https://arxiv.org/abs/2507.22270', seed=seed, tolerance=tolerance,
                source_points=source_points, target_chunks=len(targets), selected=selected, grid=records,
                cost='masked Euclidean norm / sqrt(valid dimensions)', potentials='constant f=g=1',
                caveat='Finite unconditional proxy over robot contexts; not proof of conditional marginal preservation')


def experiment_config(source, output, group, calibrated):
    cfg = copy.deepcopy(source)
    cfg['train'].update(output_dir=str(output), max_steps=5000, global_batch_size=128,
                        micro_batch_size=16, gradient_accumulation_steps=1,
                        enable_fp32=True, enable_mixed_precision=True, mask_action_padding=True,
                        enable_resume=False, require_resume=False, load_checkpoint_path=None,
                        lr_schedule_steps=None, joint_ee_aux_weight=-1., training_time_rtc=False,
                        flow_dimension_weights=None, flow_horizon_weights=None,
                        flow_gibbs_epsilon=0., flow_gibbs_log_normalizer=0.,
                        spline_control_points=0, spline_degree=3)
    # Required for this container's 4 GiB shared memory, not a change to sampling.
    cfg['data']['prefetch_factor'] = 1
    cfg['data'].pop('ee_sidecar_root', None)
    if group == 'Weighted':
        cfg['train'].update(flow_gibbs_epsilon=calibrated['epsilon'],
                            flow_gibbs_log_normalizer=calibrated['log_normalizer'])
    elif group == 'Spline':
        cfg['train']['spline_control_points'] = 16
    elif group != 'Control':
        raise ValueError(group)
    return cfg


def prepare(args):
    from ysj_utils.check_spline_fit import sampled_chunks
    from lingbotvla.action_spaces.spline import SplineActionCodec
    import torch

    source_path = args.source_run / 'lingbotvla_cli.yaml'
    source = yaml.safe_load(source_path.read_text())
    if (source['train']['global_batch_size'], source['train']['max_steps'], source['train']['loss_type']) != (128, 5000, 'L1_fm'):
        raise ValueError('Expected actual BS128 5k L1 baseline snapshot')
    if not source['train']['enable_fp32'] or not source['train']['enable_mixed_precision']:
        raise ValueError('Baseline precision flags changed; inspect before training')
    source_robot_path = Path(source['data']['robot_config_root']) / 'robotwin.yaml'
    robot = yaml.safe_load(source_robot_path.read_text())
    if action_mode(robot) != 'joint_absolute' or len(robot['states']) != 2:
        raise ValueError('Expected baseline absolute joints/grippers with no EE state')
    norm_source = Path(source['data'].get('norm_stats_file') or robot['norm_stats'])
    validate_stats(norm_source, robot)
    if any(canonical_norm_types(source['data']['norm_type'])[key] != 'bounds_99_woclip'
           for key in ('arm.position', 'effector.position')):
        raise ValueError('Calibration requires the baseline bounds_99_woclip normalization')
    manifest = args.train_list.read_text()
    datasets = [line.split()[1] for line in manifest.splitlines() if line.strip()]
    if len(datasets) != 50 or len(set(datasets)) != 50:
        raise ValueError('Expected original 50 clean task datasets')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / 'source_baseline.yaml').write_bytes(source_path.read_bytes())
    (args.run_dir / 'train.txt').write_text(manifest)
    # CPU diagnostics read numeric columns only; spread samples across all tasks.
    samples = []
    for dataset in datasets:
        samples.extend(sampled_chunks(dataset, 8, 50))
    _, actions, pad = map(np.stack, zip(*samples))
    stats = json.loads(norm_source.read_text())['norm_stats']
    targets, physical_scale = absolute_targets(actions, stats)
    calibration = calibrate(targets, pad)
    write_json(args.run_dir / 'gibbs_calibration.json', calibration)
    codec = SplineActionCodec(50, 16, 3)
    with torch.no_grad():
        tensor = torch.tensor(targets)
        coefficients, _ = codec.encode(tensor, torch.tensor(pad), torch.ones_like(tensor, dtype=torch.bool))
        reconstructed = codec.decode(coefficients).numpy()
    error = (reconstructed-targets)[~pad]
    physical = error * physical_scale
    fit = dict(chunks=len(targets), valid_steps=int((~pad).sum()), points=16, degree=3,
               normalized_rmse=float(np.sqrt(np.mean(error**2))),
               joint_mae_radians=float(np.abs(physical[:, :12]).mean()),
               joint_p95_radians=float(np.quantile(np.abs(physical[:, :12]), .95)),
               gripper_mae=float(np.abs(physical[:, 12:]).mean()),
               gripper_p95=float(np.quantile(np.abs(physical[:, 12:]), .95)),
               gripper_max=float(np.abs(physical[:, 12:]).max()),
               note='Compression error, not policy success; cubic smoothing can smear sharp gripper transitions')
    if not np.isfinite(coefficients.numpy()).all() or not np.isfinite(error).all():
        raise ValueError('Nonfinite spline coefficients/reconstruction')
    write_json(args.run_dir / 'spline_fit.json', fit)
    summary = dict(source_run=str(args.source_run), source_config_sha256=digest(source_path),
                   source_robot_sha256=digest(source_robot_path), norm_sha256=digest(norm_source),
                   train_list_sha256=digest(args.train_list),
                   code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   github_main=subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=ROOT, text=True).strip(),
                   precision='baseline FSDP2 fp32 policy/gradients; frozen visual teachers keep baseline bf16 autocast',
                   evaluation='7 tasks x 2 configs x 25, seed0, fp32, compile=False',
                   caveat='Historical baseline had no padding mask; it is not a fully matched control.', groups={})
    for group in ('Control', 'Weighted', 'Spline'):
        directory = args.run_dir / group
        directory.mkdir()
        cfg = experiment_config(source, directory/'train', group, calibration['selected'])
        robot_dir = directory/'robot_configs'
        robot_dir.mkdir()
        norm_path = directory/'norm_stats.json'
        norm_path.write_bytes(norm_source.read_bytes())
        group_robot = copy.deepcopy(robot)
        group_robot['norm_stats'] = str(norm_path)
        robot_path = robot_dir/'robotwin.yaml'
        robot_path.write_text(yaml.safe_dump(group_robot, sort_keys=False))
        cfg['data'].update(train_path=str(args.run_dir/'train.txt'), robot_config_root=str(robot_dir),
                           norm_stats_file=str(norm_path))
        receipt = dict(norm_sha256=digest(norm_path), robot_sha256=digest(robot_path), mode='joint_absolute',
                       chunk_size=50, norm_type=canonical_norm_types(cfg['data']['norm_type']),
                       train_list_sha256=digest(args.run_dir/'train.txt'), supervision=supervision_config(cfg['train']),
                       label_manifests={})
        write_json(Path(str(norm_path)+'.contract.json'), receipt)
        path = directory/'train.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        # Resolve the inference contract before a costly training launch.
        dry_run = directory/'train'
        dry_run.mkdir()
        (dry_run/'lingbotvla_cli.yaml').write_bytes(path.read_bytes())
        contract = resolve_contract(dry_run/'checkpoints/global_step_5000/hf_ckpt')
        summary['groups'][group] = dict(config=str(path), sha256=digest(path), contract=contract,
                                       supervision=supervision_config(cfg['train']), scheduled=group != 'Control')
    write_json(args.run_dir/'plan.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('/2025233210/xiesy/lingbot_ws/outputs/baseline_gbs128_5k'))
    parser.add_argument('--train-list', type=Path, default=ROOT/'xsy_utils/configs/robotwin_clean.txt')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    args = parser.parse_args()
    import fcntl
    args.run_dir = args.run_dir.resolve()
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with args.run_dir.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((args.run_dir/'plan.json').read_text()) if args.run_prepared else prepare(args)
        if args.prepare_only:
            print(json.dumps(plan, indent=2), flush=True)
            return
        if (args.run_dir/'status.json').exists():
            raise FileExistsError('Run already launched; inspect checkpoints before explicitly resuming')
        status = dict(state='PREPARED', updated_at=None, group=None)
        def update(**values):
            status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            write_json(args.run_dir/'status.json', status)
            print(json.dumps(status), flush=True)
        env = environment()
        env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PYTHONUNBUFFERED='1')
        env.setdefault('HF_HUB_OFFLINE', '1')
        env.setdefault('TRANSFORMERS_OFFLINE', '1')
        from ysj_utils.run_joint_pair import wait_idle, NONFINITE
        try:
            for group in ('Weighted', 'Spline'):
                directory = args.run_dir/group
                config_path = Path(plan['groups'][group]['config'])
                if digest(config_path) != plan['groups'][group]['sha256']:
                    raise ValueError('Prepared config changed')
                wait_idle([str(i) for i in range(8)], seconds=259200)
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                           str(ROOT/'tasks/vla/train_lingbotvla.py'), str(config_path)]
                update(state='TRAINING', group=group, step=0)
                with (directory/'training.log').open('x') as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                    try:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            if NONFINITE.search(line):
                                raise RuntimeError('Nonfinite loss/gradient detected: '+line.strip())
                            match = re.search(r'\bStep (\d+)/5000,', line)
                            if match:
                                step = int(match[1])
                                if step <= 5 or step % 100 == 0:
                                    update(step=step, last_step_log=line.strip())
                        if process.wait():
                            raise RuntimeError(f'{group} training failed, exit={process.returncode}')
                    except BaseException:
                        import signal
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                        raise
                checkpoint = directory/'train/checkpoints/global_step_5000/hf_ckpt'
                index = json.loads((checkpoint/'model.safetensors.index.json').read_text())
                if not all((checkpoint/shard).is_file() for shard in set(index['weight_map'].values())):
                    raise ValueError('HF checkpoint incomplete')
                wait_idle([str(i) for i in range(8)], seconds=259200)
                update(state='EVALUATING', checkpoint=str(checkpoint))
                command = [sys.executable, str(ROOT/'ysj_utils/evaluate.py'), '--mode', 'full', '--checkpoint', str(checkpoint),
                           '--gpus', '0,1,2,3,4,5,6,7', '--seed', '0', '--use-compile', 'false',
                           '--summary-output', str(directory/'evaluation.json')]
                with (directory/'evaluation.log').open('x') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads((directory/'evaluation.json').read_text())
                if report['status'] != 'COMPLETE' or report['total_episodes'] != 350:
                    raise RuntimeError('Evaluation incomplete')
                update(state='GROUP_COMPLETE', result=report.get('overall', report.get('totals')))
            update(state='COMPLETE')
        except BaseException as exc:
            update(state='FAILED', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
