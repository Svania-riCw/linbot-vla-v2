"""Render an exported hf_ckpt for RoboTwin without changing its architecture."""
import argparse
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def render(checkpoint, qwen, output, norm_stats=None, data_config=None, image_size=None):
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
    explicit_data_config = Path(data_config).resolve() if data_config else None
    source = explicit_data_config or ROOT / 'configs/vla/robotwin/robotwin.yaml'
    data = yaml.safe_load(source.read_text())['data']
    training_data_config = next(
        (parent / 'lingbotvla_cli.yaml' for parent in checkpoint.parents
         if (parent / 'lingbotvla_cli.yaml').is_file()),
        None,
    )
    training_data = {}
    if training_data_config is not None:
        training_data = yaml.safe_load(training_data_config.read_text()).get('data', {})

    if image_size is not None:
        image_size_source = '--image_size'
        data['img_size'] = image_size
    elif explicit_data_config is not None and 'img_size' in data:
        image_size_source = explicit_data_config
    elif 'img_size' in training_data:
        data['img_size'] = training_data['img_size']
        image_size_source = training_data_config
    else:
        image_size_source = source if 'img_size' in data else 'default (256)'
    image_size = data.get('img_size', 256)
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError(f'Invalid data.img_size: {image_size!r}; expected a positive integer')
    data['img_size'] = image_size
    for key in ('joints', 'norm_type'):
        data[key] = [str(item) if isinstance(item, dict) else item for item in data[key]]
    norm = Path(norm_stats or data.get('norm_stats_file') or ROOT / 'assets/norm_stats/robotwin.json').resolve()
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
    print(f'Evaluation image input resolution: {image_size}x{image_size} (from {image_size_source})', file=sys.stderr)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--qwen', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--norm-stats', default=os.environ.get('LINGBOT_NORM_STATS_FILE'))
    parser.add_argument('--data-config', default=os.environ.get('LINGBOT_DATA_CONFIG'))
    parser.add_argument(
        '--image_size', '--image-size', dest='image_size', type=int,
        default=None, help='Override data.img_size for evaluation.'
    )
    args = parser.parse_args()
    try:
        print(render(
            args.checkpoint, args.qwen, args.output, args.norm_stats,
            args.data_config, args.image_size,
        ))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
