"""Use the team eval engine, then reject incomplete/duplicate result grids."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.action_contract import resolve_contract, digest, supervision_config

TASKS = ['blocks_ranking_rgb', 'hanging_mug', 'open_laptop', 'pick_diverse_bottles',
         'rotate_qrcode', 'stack_blocks_three', 'place_shoe']


def scheme(mode):
    if mode == 'smoke':
        return dict(tasks=['open_laptop'], configs=['demo_clean'], episodes=1)
    if mode == 'full':
        return dict(tasks=TASKS, configs=['demo_clean', 'demo_randomized'], episodes=25)
    raise ValueError(mode)


def validate_results(path, tasks, configs, episodes):
    cases = ['random' if c == 'demo_randomized' else c.removeprefix('demo_') for c in configs]
    expected = {(task, case) for task in tasks for case in cases}
    results = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split('\t')
        if len(fields) != 4:
            raise ValueError(f'Malformed result line: {line}')
        key = tuple(fields[:2])
        if key in results or key not in expected:
            raise ValueError(f'Duplicate or unexpected result: {key}')
        success, total = map(int, fields[2].split('/'))
        if total != episodes or not 0 <= success <= total:
            raise ValueError(f'Incomplete/invalid episode count: {key} {success}/{total}')
        rate = float(fields[3])
        if not abs(rate - success/total) < 1e-4:
            raise ValueError(f'Incorrect success rate: {key}')
        results[key] = (success, total)
    if set(results) != expected:
        raise ValueError(f'Missing evaluation units: {sorted(expected-set(results))}')
    return dict(status='COMPLETE', total_episodes=sum(t for s, t in results.values()),
                rates={case: sum(results[(task, case)][0]/episodes for task in tasks)/len(tasks) for case in cases})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gpus', required=True, help='allocated GPU IDs, e.g. 0 or 0,1,2,3,4,5,6,7')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--port-base', type=int, default=19330)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--use-compile', choices=['true','false'], default='false', help='Compile inference; recorded as a separate evaluation condition')
    p.add_argument('--summary-output',type=Path,help='Write a machine-readable result including the actual run directory')
    args = p.parse_args()
    if args.summary_output and args.summary_output.exists():
        raise FileExistsError(args.summary_output)
    gpu_ids = args.gpus.replace(',', ' ').split()
    if not gpu_ids or any(not s.isdigit() for s in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        p.error('--gpus must contain distinct numeric GPU IDs')
    config = scheme(args.mode)
    model_name = 'ysj_' + args.mode + '_' + uuid.uuid4().hex[:12]
    env = os.environ.copy()
    # Clear inherited experiment overrides; use the checkpoint's own training contract.
    for key in ('NORM_STATS', 'LINGBOT_INFERENCE_CONFIG'):
        env.pop(key, None)
    env.update(MODEL_PATH=str(args.checkpoint.resolve()), MODEL_NAME=model_name,
               TEST_NUM=str(config['episodes']), TASKS=' '.join(config['tasks']),
               TASK_CONFIGS=' '.join(config['configs']), GPUS=' '.join(gpu_ids),
               SEED=str(args.seed), USE_FP32='True', USE_COMPILE=str(args.use_compile == 'true'), USE_LENGTH='50',
               ROBO_NAME='robotwin', PORT_BASE=str(args.port_base))
    command = ['bash', str(ROOT/'xsy_utils/eval_baseline.sh'), env['MODEL_PATH']]
    plan = dict(mode=args.mode, checkpoint=env['MODEL_PATH'], seed=args.seed, **config,
                total_episodes=len(config['tasks'])*len(config['configs'])*config['episodes'],
                inference_precision='fp32', use_compile=args.use_compile == 'true', use_length=50, gpus=gpu_ids, command=command)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    if os.name == 'nt':
        raise RuntimeError('Run simulation in the configured Linux GPU environment')
    if os.environ.get('LINGBOT_INFERENCE_CONFIG'):
        raise ValueError('Unset LINGBOT_INFERENCE_CONFIG; eval must use this checkpoint\'s own training config')
    index = args.checkpoint/'model.safetensors.index.json'
    shards = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
    if not shards or any(not (args.checkpoint/name).is_file() for name in shards):
        raise FileNotFoundError('Missing checkpoint shards')
    contract = resolve_contract(args.checkpoint)
    training = yaml.safe_load(Path(contract['config']).read_text(encoding='utf-8'))
    robot = yaml.safe_load(Path(contract['robot_config']).read_text(encoding='utf-8'))
    receipt_path = Path(contract['norm_stats']+'.contract.json')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
    result = subprocess.run(command, env=env, cwd=ROOT)
    output_parent = ROOT/'external/RoboTwin/eval_result'/model_name
    runs = list(output_parent.glob('run_*'))
    if len(runs) != 1:
        raise RuntimeError(f'Expected one run directory, found {runs}; eval exit={result.returncode}')
    run = runs[0]
    report = dict(plan, contract=contract, process_exit_code=result.returncode,run_dir=str(run.resolve()),
                  code_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                  norm_sha256=digest(contract['norm_stats']), robot_sha256=digest(contract['robot_config']),
                  training_config_sha256=digest(contract['config']),
                  state_features_sha256=hashlib.sha256(json.dumps(robot['states'],sort_keys=True).encode()).hexdigest(),
                  label_manifests=receipt.get('label_manifests', {}),
                  action_loss=dict(joint_ee_aux_weight=training['train'].get('joint_ee_aux_weight', -1.0),
                                   mask_action_padding=training['train'].get('mask_action_padding', False)),
                  action_supervision=supervision_config(training['train']),
                  training_summary={key:training['train'].get(key) for key in
                                    ['global_batch_size','max_steps','enable_fp32','enable_mixed_precision',
                                     'seed','chunk_size','lr','lr_min','lr_decay_style','optimizer','training_time_rtc',
                                     'loss_type','align_params','use_compile','lr_schedule_steps']},
                  initial_model=training['model']['model_path'],
                  train_list_sha256=digest(training['data']['train_path']))
    try:
        report.update(validate_results(run/'_result.txt', **config))
        if result.returncode:
            raise RuntimeError(f'eval process exited with {result.returncode}')
    except Exception as exc:
        report.update(status='FAILED_OR_INCOMPLETE', error=str(exc))
        raise
    finally:
        (run/'ysj_evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if args.summary_output:
            args.summary_output.parent.mkdir(parents=True,exist_ok=True)
            with args.summary_output.open('x',encoding='utf-8') as file:
                json.dump(report,file,indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
