"""Render an exported hf_ckpt for RoboTwin without changing its architecture."""
import argparse
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def render(checkpoint, qwen, output, norm_stats=None, data_config=None):
    checkpoint, qwen, output = map(lambda p: Path(p).resolve(), (checkpoint, qwen, output))
    if (checkpoint / 'hf_ckpt').is_dir():
        checkpoint /= 'hf_ckpt'
    model = json.loads((checkpoint / 'config.json').read_text())
    for key in ('post_training', 'max_action_dim', 'max_state_dim', 'token_moe_layers'):
        if key not in model:
            raise ValueError(f'Incomplete exported config: {key}; use original training YAML')
    if not list(checkpoint.glob('*.safetensors')) or not (qwen / 'config.json').is_file():
        raise ValueError('Checkpoint safetensors or Qwen config is missing')
    model.update(model_path=str(checkpoint), tokenizer_path=str(qwen), use_compile=False)
    # Keep the run's image resolution/interpolation when exporting for evaluation.
    # The stock defaults are only suitable when no training metadata is available.
    training_yaml = checkpoint.parent.parent.parent / 'lingbotvla_cli.yaml'
    source = Path(data_config) if data_config else (
        training_yaml if training_yaml.is_file() else ROOT / 'configs/vla/robotwin/robotwin.yaml'
    )
    data = yaml.safe_load(source.read_text())['data']
    # ROI checkpoints carry an input-representation fingerprint; losing the
    # training YAML must fail rather than silently evaluate ordinary full views.
    sys.path.insert(0, str(ROOT))
    from lingbotvla.utils.ee_roi import check_roi_identity
    check_roi_identity(model, data.get("roi_config"))
    for key in ('joints', 'norm_type'):
        data[key] = [str(item) if isinstance(item, dict) else item for item in data[key]]
    norm = norm_stats or data.get('norm_stats_file')
    if not norm and data.get('robot_config_root'):
        # Match serve.py: training may store its normalization path only in the
        # robot mapping. Falling back to bundled clean+random stats changes actions.
        robot_yaml = Path(data['robot_config_root']) / 'robotwin.yaml'
        if not robot_yaml.is_file():
            raise ValueError(f'Training robot config is unavailable: {robot_yaml}; pass --norm-stats')
        norm = yaml.safe_load(robot_yaml.read_text(encoding='utf-8')).get('norm_stats')
        if not norm:
            raise ValueError(f'Training robot config has no norm_stats: {robot_yaml}; pass --norm-stats')
    norm = Path(norm or ROOT / 'assets/norm_stats/robotwin.json').resolve()
    if not norm.is_file():
        raise ValueError(f'Normalization statistics are missing: {norm}')
    data['norm_stats_file'] = str(norm)
    content = yaml.safe_dump(dict(model=model, data=data, train={}), sort_keys=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text() != content:
        temporary = output.with_suffix('.tmp')
        temporary.write_text(content)
        os.replace(temporary, output)
    print(f'Normalization statistics: {norm}; verify these match checkpoint training.', file=sys.stderr)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--qwen', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--norm-stats', default=os.environ.get('LINGBOT_NORM_STATS_FILE'))
    parser.add_argument('--data-config', default=os.environ.get('LINGBOT_DATA_CONFIG'))
    args = parser.parse_args()
    try:
        print(render(args.checkpoint, args.qwen, args.output, args.norm_stats, args.data_config))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
