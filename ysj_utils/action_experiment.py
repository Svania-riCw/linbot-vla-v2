"""Prepare a matched B0/JDelta experiment; compute stats before launching training.

All execution is explicit (norm/train subcommands). Prepare never starts training.
Run from the team's configured lingbotvla environment on Linux.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.action_contract import action_mode, canonical_norm_types, digest


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        yaml.safe_dump(value, file, allow_unicode=True, sort_keys=False)


def prepare(base_config, robot_config, train_list, output, world_size=8, micro_batch=16):
    if world_size < 1 or micro_batch < 1 or 128 % (world_size * micro_batch):
        raise ValueError('128 must be divisible by world_size * micro_batch')
    base = yaml.safe_load(Path(base_config).read_text(encoding='utf-8'))
    robot = yaml.safe_load(Path(robot_config).read_text(encoding='utf-8'))
    if action_mode(robot) != 'joint_absolute':
        raise ValueError('Start from the verified absolute joint robot config')
    reference = yaml.safe_load((ROOT / 'configs/robot_configs/robotwin.yaml').read_text(encoding='utf-8'))
    for field in ('states', 'actions', 'images'):
        if robot[field] != reference[field]:
            raise ValueError(f'Robot {field} differ from audited RoboTwin layout; review before proceeding')
    lines = [line.strip() for line in Path(train_list).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines or any(len(line.split()) != 2 or line.split()[0] != 'robotwin' for line in lines):
        raise ValueError('Expected nonempty robotwin training manifest, two columns per line')
    if len(set(lines)) != len(lines):
        raise ValueError('Duplicate dataset in training manifest')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    copied_list = out / 'train.txt'
    copied_list.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = {
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'base_config_sha256': digest(base_config), 'source_robot_sha256': digest(robot_config),
        'train_list_sha256': digest(copied_list), 'datasets': len(lines),
        'world_size': world_size, 'micro_batch': micro_batch, 'global_batch': 128,
        'steps': 10000, 'precision': 'bf16', 'status': 'PREPARED_NOT_TRAINED',
    }
    for group, delta in [('B0', False), ('JDelta', True)]:
        directory = out / group
        norm_path = directory / 'norm_stats.json'
        root = directory / 'robot_configs'
        group_robot = copy.deepcopy(robot)
        for feature in group_robot['actions']:
            if 'action.arm.position' in feature:
                feature['action.arm.position']['subtract_state'] = delta
        group_robot['norm_stats'] = str(norm_path)
        write_yaml(root / 'robotwin.yaml', group_robot)
        cfg = copy.deepcopy(base)
        cfg['data'].update(train_path=str(copied_list), robot_config_root=str(root), norm_stats_file=str(norm_path))
        cfg['train'].update(
            output_dir=str(directory / 'train'), global_batch_size=128,
            micro_batch_size=micro_batch, gradient_accumulation_steps=128//(world_size*micro_batch),
            max_steps=10000, save_steps=1000, enable_fp32=False, enable_mixed_precision=True,
            enable_resume=False, training_time_rtc=False, chunk_size=50, seed=42,
        )
        write_yaml(directory / 'train.yaml', cfg)
        norm = {'data': copy.deepcopy(cfg['data']), 'train': copy.deepcopy(cfg['train'])}
        norm['data'].update(robot_name='robotwin', norm_path=str(norm_path),
                            data_ratio_for_norm_compute=1.0, norm_merge_chunk_dim=True,
                            num_workers=4, use_future_image=False, image_augment=False)
        norm['train'].update(micro_batch_size=32, global_batch_size=32, gradient_accumulation_steps=1)
        write_yaml(directory / 'norm.yaml', norm)
        manifest[group] = {'robot_sha256': digest(root/'robotwin.yaml'),
                           'train_yaml_sha256': digest(directory/'train.yaml'),
                           'norm_yaml_sha256': digest(directory/'norm.yaml')}
    (out/'pair.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def verify_pair(bundle, group):
    bundle = Path(bundle).resolve()
    pair = json.loads((bundle/'pair.json').read_text(encoding='utf-8'))
    directory = bundle/group
    checks = {'robot_sha256': directory/'robot_configs/robotwin.yaml',
              'train_yaml_sha256': directory/'train.yaml', 'norm_yaml_sha256': directory/'norm.yaml'}
    for key, path in checks.items():
        if digest(path) != pair[group][key]:
            raise ValueError(f'Prepared file changed: {path}; create a new experiment bundle')
    if digest(bundle/'train.txt') != pair['train_list_sha256']:
        raise ValueError('Training manifest changed')
    return pair, directory


def validate_stats(path):
    import numpy as np
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    stats = payload['norm_stats']
    for key, dim in [('action.arm.position', 12), ('observation.state.arm.position', 12),
                     ('action.effector.position', 2), ('observation.state.effector.position', 2)]:
        for name in ('mean', 'std', 'q01', 'q99'):
            value = np.asarray(stats[key][name])
            if value.shape != (dim,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid merged-horizon stats: {key}.{name}')
        if np.any(np.array(stats[key]['q99']) < np.array(stats[key]['q01'])):
            raise ValueError(f'Reversed quantiles: {key}')
        if np.any(np.array(stats[key]['std']) < 0):
            raise ValueError(f'Negative std: {key}')


def record_receipt(directory, pair):
    stats_path = directory/'norm_stats.json'
    validate_stats(stats_path)
    cfg = yaml.safe_load((directory/'train.yaml').read_text(encoding='utf-8'))
    robot_path = directory/'robot_configs/robotwin.yaml'
    robot = yaml.safe_load(robot_path.read_text(encoding='utf-8'))
    receipt = dict(norm_sha256=digest(stats_path), robot_sha256=digest(robot_path),
                   mode=action_mode(robot), chunk_size=cfg['train']['chunk_size'],
                   norm_type=canonical_norm_types(cfg['data']['norm_type']),
                   train_list_sha256=pair['train_list_sha256'],
                   norm_yaml_sha256=digest(directory/'norm.yaml'))
    Path(str(stats_path)+'.contract.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def environment():
    env = os.environ.copy()
    vm = ROOT/'lingbotvla/models/vla/vision_models'
    env['PYTHONPATH'] = os.pathsep.join(map(str, [ROOT, vm/'MoGe', vm/'lingbot-depth', ROOT/'third_party/utils3d']))
    env['PYTHONNOUSERSITE'] = '1'
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--base-config', type=Path, default=ROOT/'xsy_utils/configs/robotwin_clean_baseline.yaml')
    p.add_argument('--robot-config', type=Path, required=True, help='actual training robot_config_clean/robotwin.yaml')
    p.add_argument('--train-list', type=Path, default=ROOT/'xsy_utils/configs/robotwin_clean.txt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--world-size', type=int, default=8)
    p.add_argument('--micro-batch', type=int, default=16)
    for name in ('norm', 'train'):
        p = commands.add_parser(name)
        p.add_argument('--bundle', type=Path, required=True)
        p.add_argument('--group', choices=['B0', 'JDelta'], required=True)
        p.add_argument('--dry-run', action='store_true')
        if name == 'train':
            p.add_argument('--stage', choices=['smoke', 'full'], default='smoke')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.base_config, args.robot_config, args.train_list,
                                 args.output, args.world_size, args.micro_batch), indent=2))
        return
    pair, directory = verify_pair(args.bundle, args.group)
    if args.command == 'norm':
        if (directory/'norm_stats.json').exists():
            raise FileExistsError('Stats already exist; use a fresh bundle to avoid replacing training statistics')
        command = [sys.executable, str(ROOT/'scripts/compute_norm_stats.py'), str(directory/'norm.yaml')]
    else:
        receipt_path = directory/'norm_stats.json.contract.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['norm_sha256'] != digest(directory/'norm_stats.json') or receipt['robot_sha256'] != pair[args.group]['robot_sha256']:
            raise ValueError('Stats or action representation changed after normalization')
        world = 1 if args.stage == 'smoke' else pair['world_size']
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        gpu_ids = [part.strip() for part in visible.split(',') if part.strip()]
        if len(gpu_ids) != world or len(set(gpu_ids)) != world:
            raise ValueError(f'Set CUDA_VISIBLE_DEVICES to exactly {world} allocated GPU(s)')
        out = directory/('train-smoke' if args.stage == 'smoke' else 'train')
        if out.exists():
            raise FileExistsError(f'Output already exists: {out}')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={world}',
                   str(ROOT/'tasks/vla/train_lingbotvla.py'), str(directory/'train.yaml'),
                   '--train.output_dir', str(out)]
        if args.stage == 'smoke':
            command += ['--train.max_steps', '5', '--train.save_steps', '5', '--train.micro_batch_size', '1',
                        '--train.global_batch_size', '1', '--train.gradient_accumulation_steps', '1',
                        '--train.use_compile', 'False', '--data.num_workers', '0']
    print(json.dumps({'command': command, 'cwd': str(ROOT)}, indent=2))
    if args.dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=environment(), check=True)
    if args.command == 'norm':
        record_receipt(directory, pair)


if __name__ == '__main__':
    main()
