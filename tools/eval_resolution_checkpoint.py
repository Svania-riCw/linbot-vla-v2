"""Evaluate a pinned-resolution checkpoint; fail on incomplete 7x2x25 results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('blocks_ranking_rgb', 'hanging_mug', 'open_laptop', 'pick_diverse_bottles',
         'rotate_qrcode', 'stack_blocks_three', 'place_shoe')


def validate_results(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        task, case, counts, *_ = line.split('\t')
        key = (task, case)
        if key in rows:
            raise ValueError(f'Duplicate evaluation result: {key}')
        success, total = map(int, counts.split('/'))
        if total != 25 or not 0 <= success <= total:
            raise ValueError(f'Invalid evaluation count: {line}')
        rows[key] = (success, total)
    expected = {(task, case) for task in TASKS for case in ['clean', 'random']}
    if set(rows) != expected:
        raise ValueError(f'Incomplete evaluation: missing={sorted(expected-set(rows))}, extra={sorted(set(rows)-expected)}')
    result = {'episodes': 350, 'units': 14, 'tasks': {}}
    for case in ['clean', 'random']:
        successes = sum(v[0] for k, v in rows.items() if k[1] == case)
        result[case] = {'successes': successes, 'episodes': 175, 'success_rate': successes / 175}
    result['success_rate'] = sum(v[0] for v in rows.values()) / 350
    for (task, case), (s, n) in sorted(rows.items()):
        result['tasks'].setdefault(task, {})[case] = {'successes': s, 'episodes': n, 'success_rate': s / n}
    return result


def log_wandb(settings_path, summary, checkpoint, group):
    settings = json.loads(Path(settings_path).read_text())
    secret = Path(settings['secret_file'])
    if secret.stat().st_mode & 0o077:
        raise ValueError('W&B credential file permissions must be restricted')
    os.environ['WANDB_API_KEY'] = json.loads(secret.read_text())['api_key']
    os.environ['WANDB_BASE_URL'] = settings['base_url']
    sys.path.insert(0, settings['sdk_path'])
    import wandb
    run = wandb.init(entity=settings['entity'], project=settings['project'], group=group,
                     name=f'{group}_eval_{checkpoint.parent.name}', job_type='evaluation',
                     dir=str(Path(settings_path).parent / 'wandb'),
                     config={'checkpoint': str(checkpoint), 'tasks': list(TASKS),
                             'episodes_per_task_case': 25, 'clean_norm_sha256': summary['clean_norm_sha256']})
    metrics = {'eval/overall_success_rate': summary['success_rate'], 'eval/episodes': 350}
    for case in ['clean', 'random']:
        metrics[f'eval/{case}_success_rate'] = summary[case]['success_rate']
    for task, cases in summary['tasks'].items():
        for case, values in cases.items():
            metrics[f'eval/{task}/{case}_success_rate'] = values['success_rate']
    run.log(metrics)
    run.summary.update({'checkpoint_path': str(checkpoint), 'results_file': summary['results_file'],
                        'evaluated_episodes': 350})
    run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--gpus', default='0 1 2 3 4 5 6 7')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--wandb-settings')
    parser.add_argument('--group')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    run = checkpoint.parent.parent.parent
    data = yaml.safe_load((run / 'lingbotvla_cli.yaml').read_text())['data']
    norm = Path(data['norm_stats_file'])
    robot = yaml.safe_load((Path(data['robot_config_root']) / 'robotwin.yaml').read_text())
    norm_hash = hashlib.sha256(norm.read_bytes()).hexdigest()
    if norm_hash != hashlib.sha256(Path(robot['norm_stats']).read_bytes()).hexdigest():
        raise ValueError('Training robot mapping and inference normalization differ')
    if data['img_size'] != 384 or data['image_resize_mode'] != 'opencv_lanczos4' or not data['preserve_image_resolution']:
        raise ValueError('Expected the pinned 384 Lanczos configuration')
    if not (checkpoint / 'model.safetensors.index.json').is_file():
        raise ValueError('Checkpoint export is incomplete')
    env = os.environ.copy()
    env.pop('LINGBOT_INFERENCE_CONFIG', None)
    model_name = f'ql_{run.name}_{checkpoint.parent.name}'
    env.update(MODEL_NAME=model_name, NORM_STATS=str(norm), TEST_NUM='25',
               TASKS=' '.join(TASKS), TASK_CONFIGS='demo_clean demo_randomized',
               GPUS=args.gpus, USE_FP32='True', USE_COMPILE='False', USE_LENGTH='50', SEED='0',
               PORT_BASE='19330', CONDA_ENVS_PATH='/2025233210/xiesy/lingbot_ws/conda_envs')
    command = ['bash', str(ROOT / 'xsy_utils/eval_baseline.sh'), str(checkpoint)]
    if args.dry_run:
        print(json.dumps({'command': command, 'tasks': list(TASKS), 'episodes': 350,
                          'norm_stats': str(norm), 'norm_sha256': norm_hash}))
        return
    results_root = ROOT / 'external/RoboTwin/eval_result' / model_name
    before = set(results_root.glob('run_*'))
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    created = set(results_root.glob('run_*')) - before
    if len(created) != 1:
        raise RuntimeError('Expected exactly one new evaluation directory')
    results_file = created.pop() / '_result.txt'
    summary = validate_results(results_file)
    summary.update(checkpoint=str(checkpoint), results_file=str(results_file), clean_norm_sha256=norm_hash)
    summary_path = results_file.parent / 'evaluation_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    if args.wandb_settings:
        log_wandb(args.wandb_settings, summary, checkpoint, args.group or run.name)
    print(json.dumps({'evaluation_complete': True, 'summary': str(summary_path), **summary}))


if __name__ == '__main__':
    main()
