"""Resolve eval action semantics before importing torch or loading model weights."""
import ast
import hashlib
import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_config_path(model_path):
    return Path(os.environ.get('LINGBOT_INFERENCE_CONFIG') or
                Path(model_path).parent.parent.parent / 'lingbotvla_cli.yaml')


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def action_mode(robot):
    actions = {k: v for group in robot['actions'] for k, v in group.items()}
    allowed = {'action.arm.position', 'action.end.position', 'action.effector.position'}
    if not set(actions) <= allowed or 'action.effector.position' not in actions or len(actions) < 2:
        raise ValueError('Expected joint and/or EE actions plus absolute grippers')
    if actions['action.effector.position'].get('subtract_state') is not False:
        raise ValueError('Grippers must remain absolute')
    if any(v.get('convert_from_state', False) for v in actions.values()):
        raise ValueError('Changing target timestamps via convert_from_state is not supported')
    ee = actions.get('action.end.position')
    if ee is not None:
        if ee.get('subtract_state') is not True or ee.get('relative_type') != 'quaternion_local':
            raise ValueError('EE experiments require chunk-anchor quaternion_local deltas')
        if 'action.arm.position' not in actions:
            return 'ee_delta'
    flag = actions['action.arm.position'].get('subtract_state')
    if not isinstance(flag, bool):
        raise ValueError('subtract_state must be a boolean')
    if ee is not None:
        if not flag:
            raise ValueError('Joint+EE experiment requires joint deltas')
        return 'joint_ee_delta'
    return 'joint_delta' if flag else 'joint_absolute'


def canonical_norm_types(value):
    result = {}
    for group in value:
        result.update(ast.literal_eval(group) if isinstance(group, str) else group)
    return result


def resolve_contract(model_path, robo_name='robotwin', norm_override=None):
    cfg_path = training_config_path(model_path)
    config = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    data = config['data']
    root = resolve_path(data.get('robot_config_root') or 'configs/robot_configs')
    robot_path = root / f'{robo_name}.yaml'
    # A missing explicitly configured training root must never silently fall back.
    robot = yaml.safe_load(robot_path.read_text(encoding='utf-8'))
    norm_path = resolve_path(norm_override or data.get('norm_stats_file') or robot['norm_stats'])
    if not norm_path.is_file():
        raise FileNotFoundError(norm_path)
    receipt_path = Path(str(norm_path) + '.contract.json')
    # Ordinary legacy baselines keep their existing supported action spaces.
    actions = {k: v for group in robot['actions'] if isinstance(group, dict) for k, v in group.items()}
    is_delta = any(v.get('subtract_state', False) for v in actions.values())
    if is_delta or receipt_path.exists():
        mode = action_mode(robot)
        if not receipt_path.exists():
            raise ValueError('Delta stats require a .contract.json receipt from ysj_utils/action_experiment.py norm')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        expected = {
            'norm_sha256': digest(norm_path), 'robot_sha256': digest(robot_path),
            'mode': mode, 'chunk_size': config['train'].get('chunk_size', 50),
            'norm_type': canonical_norm_types(data['norm_type']),
            'train_list_sha256': digest(resolve_path(data['train_path'])),
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f'Action contract mismatch: {key}')
    else:
        mode = 'legacy_absolute'
    return dict(robot_config_root=str(root), robot_config=str(robot_path),
                norm_stats=str(norm_path), mode=mode, config=str(cfg_path))
