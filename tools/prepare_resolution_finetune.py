"""Prepare the 384-only experiment from an existing, completed 10k baseline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f'Preserving different pinned input: {path}')
    path.write_bytes(content)


def prepare(baseline, destination, run_root, data_root, clean_stats, micro=4, gpu_count=8):
    baseline, destination, run_root, data_root, clean_stats = map(
        lambda p: Path(p).resolve(), (baseline, destination, run_root, data_root, clean_stats)
    )
    if micro < 1 or gpu_count < 1 or 128 % (micro * gpu_count):
        raise ValueError('micro * GPU count must divide global batch 128')
    source = baseline / 'lingbotvla_cli.yaml'
    reference = yaml.safe_load(source.read_text())
    train, data = reference['train'], reference['data']
    if train['max_steps'] != 10000 or train['global_batch_size'] != 128 or train['enable_fp32'] is not True:
        raise ValueError('Reference baseline must actually use 10000 steps, global batch 128 and enable_fp32=true')
    hf = baseline / 'checkpoints/global_step_10000/hf_ckpt'
    if not (hf / 'config.json').is_file() or not list(hf.glob('*.safetensors')):
        raise ValueError('Reference baseline has no exported 10000-step checkpoint')
    robot_path = Path(data['robot_config_root']) / 'robotwin.yaml'
    robot = yaml.safe_load(robot_path.read_text())
    norm_hash = sha256(clean_stats)
    for p in [robot['norm_stats'], data.get('norm_stats_file')]:
        if p and sha256(p) != norm_hash:
            raise ValueError(f'Baseline normalization differs from clean statistics: {p}')
    entries, episodes = [], 0
    for line in Path(data['train_path']).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        robot_name, root = line.split()
        root = Path(root).resolve()
        if robot_name != 'robotwin' or root.parent != data_root or not root.name.startswith('robotwin_clean_'):
            raise ValueError(f'Unexpected non-clean training entry: {line}')
        info = json.loads((root / 'meta/info.json').read_text())
        if info['total_episodes'] != 50:
            raise ValueError(f'Expected 50 episodes per task: {root}')
        episodes += info['total_episodes']
        entries.append(f'robotwin {root.as_posix()}')
    if len(entries) != 50 or len(set(entries)) != 50 or episodes != 2500:
        raise ValueError('Expected 50 distinct clean tasks and 2500 episodes')
    # Training reads robot_config_root/robotwin.yaml, not data.norm_stats_file.
    # Pin both paths to the same immutable clean-only file for train and serve.
    pinned = destination / 'pinned_inputs'
    norm = pinned / f'robotwin_clean_{norm_hash[:16]}.json'
    write_once(norm, clean_stats.read_bytes())
    robot['norm_stats'] = str(norm)
    write_once(pinned / 'robotwin.yaml', yaml.safe_dump(robot, sort_keys=False).encode())
    manifest = pinned / 'robotwin_clean.txt'
    write_once(manifest, ('\n'.join(entries) + '\n').encode())
    config = copy.deepcopy(reference)
    config['data'].update(train_path=str(manifest), robot_config_root=str(pinned),
                          norm_stats_file=str(norm), img_size=384,
                          image_resize_mode='opencv_lanczos4', preserve_image_resolution=True,
                          num_workers=1, prefetch_factor=1)
    config['train'].update(output_dir=str(run_root / 'exp1_384_10k'),
                           max_steps=10000, global_batch_size=128, micro_batch_size=micro,
                           gradient_accumulation_steps=128 // (micro * gpu_count),
                           enable_fp32=True, enable_resume=False,
                           save_steps=1000, save_hf_weights=True, use_wandb=True,
                           pause_after_steps=5000)
    config['train'].pop('load_checkpoint_path', None)
    # The CLI parser serializes YAML values into command-line strings. Omit
    # null optionals so they retain their real None defaults instead of "null".
    for section in config.values():
        if isinstance(section, dict):
            for key in list(section):
                if section[key] is None:
                    section.pop(key)
    align = config['train'].get('align_params')
    if align:
        align['llm']['image_token_size'] = 12
    # max_steps remains 10000 even while pausing at 5000, preserving the LR curve.
    path = destination / f'b1_384_m{micro}.yaml'
    write_once(path, yaml.safe_dump(config, sort_keys=False).encode())
    differences = []
    def compare(a, b, prefix=''):
        for key in sorted(set(a) | set(b)):
            name = f'{prefix}.{key}' if prefix else key
            if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
                compare(a[key], b[key], name)
            elif a.get(key) != b.get(key):
                differences.append({'key': name, 'baseline': a.get(key), 'experiment': b.get(key)})
    compare(reference, config)
    report = dict(baseline=str(baseline), baseline_config_sha256=sha256(source),
                  config=str(path), data_manifest=str(manifest), clean_norm_stats=str(norm), clean_norm_sha256=norm_hash,
                  dataset_tasks=50, dataset_episodes=episodes, visual_tokens=432,
                  scheduler_total_steps=10000, pause_step=5000, checkpoint_every=1000,
                  reference_seed=train.get('seed'), reference_micro_batch=train.get('micro_batch_size'),
                  differences=differences)
    write_once(destination / f'preflight_m{micro}.json', (json.dumps(report, indent=2) + '\n').encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='/2025233210/xiesy/lingbot_ws/outputs/baseline_gbs128_10k')
    parser.add_argument('--destination', required=True)
    parser.add_argument('--run-root', default='/root/ql/outputs')
    parser.add_argument('--data-root', default='/2025233210/lmx/training-data/lerobot')
    parser.add_argument('--clean-stats', default='/2025233210/lmx/training-data/robotwin_clean_norm_stats.json')
    parser.add_argument('--micro', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare(args.baseline, args.destination, args.run_root, args.data_root,
                             args.clean_stats, args.micro)))


if __name__ == '__main__':
    main()
