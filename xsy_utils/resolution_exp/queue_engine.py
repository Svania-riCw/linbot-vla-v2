"""384-only 10k queue with a 5k checkpoint pause and 350-episode evaluation."""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time

REPO = Path(os.environ.get('QL_REPO', '/root/ql/linbot-vla-v2'))
TRAIN_ENV = Path('/2025233210/xiesy/lingbot_ws/conda_envs/lingbotvla')
EXPECTED_HEAD = os.environ.get('QL_EXPECTED_HEAD', '8b53058b22d9050581c895ebf7e07d377999325f')
GPUS = tuple(range(8))
STATE_DIR = Path(os.environ.get('QL_QUEUE_STATE_DIR', str(REPO / '.runtime/node1_queue')))
OUTPUT_ROOT = Path(os.environ.get('QL_OUTPUT_ROOT', '/root/ql/outputs'))


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=30, **kwargs).strip()


def snapshot_idle(gpu_csv, app_csv):
    rows = list(csv.reader(gpu_csv.strip().splitlines()))
    cards = {int(r[0]): (r[1].strip(), int(r[2]), int(r[3])) for r in rows if r}
    if any(g not in cards for g in GPUS):
        return False
    uuids = {cards[g][0] for g in GPUS}
    apps = list(csv.reader(app_csv.strip().splitlines()))
    if any(row and row[0].strip() in uuids for row in apps):
        return False
    return all(cards[g][1] <= 1024 and cards[g][2] <= 5 for g in GPUS)


def gpu_snapshot():
    cards = command(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    apps = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    return cards, apps


def write_state(state, phase, **updates):
    state.update(phase=phase, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **updates)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = STATE_DIR / 'status.tmp'
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temp.replace(STATE_DIR / 'status.json')
    print(json.dumps({'phase': phase, 'utc': state['updated_utc'], **updates}), flush=True)


def validate_setup():
    assert command(['git', '-C', str(REPO), 'rev-parse', 'HEAD']) == EXPECTED_HEAD
    assert command(['git', '-C', str(REPO), 'branch', '--show-current']) == 'roi'
    assert not command(['git', '-C', str(REPO), 'status', '--porcelain'])
    assert (TRAIN_ENV / 'bin/python').is_file()
    logging = json.loads((STATE_DIR / 'wandb_settings.json').read_text())
    assert logging['enabled'] and (Path(logging['sdk_path']) / 'wandb').is_dir()
    key_file = Path(logging['secret_file'])
    assert key_file.is_file() and key_file.stat().st_mode & 0o077 == 0
    # The audit contains decoded RGB from all 50 tasks, plus Parquet markers.
    audit = json.loads((STATE_DIR / 'dataset_verification.json').read_text())
    assert audit['dataset_count'] == 50 and not audit['errors']
    for item in audit['datasets']:
        root = Path(item['root'])
        assert (root / 'meta/info.json').is_file()
        files = list((root / 'data').rglob('*.parquet'))
        assert len(files) == item['parquet_files'] and all(p.stat().st_size > 12 for p in files)
    for path, exists in audit['required_paths'].items():
        assert exists and Path(path).exists(), path
    for micro in [4, 2, 1]:
        prepare_config(micro)


def prepare_config(micro):
    raw = command([str(TRAIN_ENV / 'bin/python'), str(REPO / 'tools/prepare_resolution_finetune.py'),
                   '--destination', str(STATE_DIR / 'replan384'), '--run-root', str(OUTPUT_ROOT), '--micro', str(micro)])
    return json.loads(raw)


def wait_idle(state, deadline):
    while time.time() < deadline:
        good = True
        for check in range(3):
            cards, apps = gpu_snapshot()
            if not snapshot_idle(cards, apps):
                retry_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
                write_state(state, 'waiting_for_idle_gpus', gpu_snapshot=cards, gpu_processes=apps,
                            retry_interval_seconds=3600, next_check_utc=retry_at.isoformat())
                good = False
                break
            if check < 2:
                time.sleep(15)
        if good:
            return
        # The user requested another check one hour later when the cards are busy.
        retry = min(time.time() + 3600, deadline)
        while time.time() < retry:
            time.sleep(min(60, retry - time.time()))
    raise TimeoutError('GPUs did not become idle within the queue waiting window')


def checkpoint_valid(output, steps):
    checkpoint = output / 'checkpoints' / ('global_step_' + str(steps))
    hf = checkpoint / 'hf_ckpt'
    assert (hf / 'config.json').is_file(), str(hf)
    assert any(p.stat().st_size > 0 for p in hf.glob('*.safetensors'))
    for name in ['model', 'optimizer']:
        assert (checkpoint / name / '.metadata').is_file(), str(checkpoint / name)
    assert len(list((checkpoint / 'extra_state').glob('extra_state_rank_*.pt'))) == len(GPUS)


def run_phase(state, queue_id, micro, smoke, deadline, resume=False):
    wait_idle(state, deadline)
    preflight = prepare_config(micro)
    size, label = 384, 'b1_384'
    kind = ('smoke_resume' if resume else 'smoke') if smoke else ('resume_to_10000' if resume else 'train_to_5000')
    steps = (4 if resume else 2) if smoke else (10000 if resume else 5000)
    run_name = f'{queue_id}_{"smoke" if smoke else "train"}_{label}_m{micro}'
    output = OUTPUT_ROOT / run_name
    if output.exists() and not resume:
        raise FileExistsError('Preserving existing run: ' + str(output))
    if resume:
        checkpoint_valid(output, 2 if smoke else 5000)
        snapshot = STATE_DIR / ('before_smoke_resume_m' + str(micro) if smoke else 'before_resume')
        snapshot.mkdir(exist_ok=True)
        for name in ['lingbotvla_cli.yaml', 'run_info.txt', 'config.yaml']:
            source = output / name
            destination = snapshot / name
            if destination.exists():
                assert destination.read_bytes() == source.read_bytes()
            else:
                destination.write_bytes(source.read_bytes())
    log_path = STATE_DIR / f'{run_name}_{kind}.log'
    env = os.environ.copy()
    # Do not inherit stale tmux proxy endpoints into this experiment.
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
        env.pop(key, None)
    for key in ['LINGBOT_INFERENCE_CONFIG', 'CONFIG', 'WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_DISABLED', 'WANDB_ANONYMOUS']:
        env.pop(key, None)
    env.update(WS=str(OUTPUT_ROOT.parent), TRAIN_ENV=str(TRAIN_ENV), CONDA_ENVS_PATH=str(TRAIN_ENV.parent),
               CONFIG=preflight['config'], CUDA_VISIBLE_DEVICES=','.join(map(str, GPUS)),
               MICRO_BS=str(micro), GLOBAL_BS='128', NUM_WORKERS='1', PREFETCH_FACTOR='1',
               RUN_NAME=run_name, MASTER_PORT='29581', PYTHONUNBUFFERED='1')
    logging = json.loads((STATE_DIR / 'wandb_settings.json').read_text())
    credential = json.loads(Path(logging['secret_file']).read_text())
    wandb_name = f'{run_name}_{kind}'
    env.update(WANDB_API_KEY=credential['api_key'], WANDB_BASE_URL=logging['base_url'],
               WANDB_ENTITY=logging['entity'], WANDB_PROJECT=logging['project'], WANDB_MODE='online',
               WANDB_GROUP=queue_id, WANDB_JOB_TYPE=kind, WANDB_NAME=wandb_name,
               WANDB_TAGS=f'resolution-exp1,{label},{kind}', WANDB_DISABLE_CODE='true',
               WANDB_CONSOLE='off', WANDB_INIT_TIMEOUT='60', QL_WANDB_SDK_PATH=logging['sdk_path'],
               WANDB_DIR=str(STATE_DIR / 'wandb'))
    Path(env['WANDB_DIR']).mkdir(exist_ok=True)
    env['QL_WANDB_RUN_METADATA'] = json.dumps({
        'execution_node': os.environ.get('QL_NODE_NAME', 'unspecified'), 'code_commit': EXPECTED_HEAD, 'experiment': label, 'phase': kind,
        'input_resolution': [384, 384], 'num_cameras': 3, 'expected_visual_tokens': 432,
        'image_resize_mode': 'opencv_lanczos4', 'preserve_image_resolution': True,
        'dataset_tasks': 50, 'dataset_episodes': 2500, 'data_manifest': preflight['data_manifest'],
        'reference_baseline': preflight['baseline'], 'reference_config_sha256': preflight['baseline_config_sha256'],
        'clean_norm_sha256': preflight['clean_norm_sha256'], 'precision': 'enable_fp32=true',
        'gpus': list(GPUS), 'micro_batch_size': micro, 'global_batch_size': 128,
        'gradient_accumulation_steps': 128 // (micro * len(GPUS)), 'seed': preflight['reference_seed'],
        'scheduler_total_steps': 4 if smoke else 10000, 'target_steps': steps,
        'checkpoint_every': 2 if smoke else 1000, 'output_dir': str(output)})
    if resume:
        args = ['bash', str(REPO / 'xsy_utils/resolution_exp/resume.sh'),
                str(output / ('checkpoints/global_step_2' if smoke else 'checkpoints/global_step_5000'))]
        if smoke:
            args += ['--train.max_steps', '4', '--train.save_steps', '2']
    else:
        args = ['bash', str(REPO / 'xsy_utils/resolution_exp/train.sh'), '384',
                '--train.max_steps', '4' if smoke else '10000',
                '--train.pause_after_steps', '2' if smoke else '5000',
                '--train.save_steps', '2' if smoke else '1000', '--train.enable_resume', 'false']
    args += ['--train.seed', str(preflight['reference_seed']), '--train.enable_fp32', 'true',
             '--train.use_wandb', 'true', '--train.wandb_project', logging['project'],
             '--train.wandb_name', wandb_name]
    with log_path.open('x') as log:
        child = subprocess.Popen(args, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_state(state, kind, run_name=run_name, output=str(output), log=str(log_path),
                    child_pid=child.pid, size=384, visual_tokens=432, micro_batch=micro,
                    global_batch=128, target_steps=steps, scheduler_total_steps=4 if smoke else 10000,
                    preflight=preflight)
        code = child.wait()
    text = log_path.read_text(errors='replace')
    if code:
        if smoke and re.search(r'CUDA out of memory|torch\.OutOfMemoryError', text, re.I):
            state.setdefault('failed_oom_smokes', []).append(str(log_path))
            return False
        raise RuntimeError(f'{run_name} exited {code}; inspect {log_path}')
    if re.search(r'(?:loss|gradnorm)[ =:]+[-+]?(?:nan|inf)\b', text, re.I):
        raise RuntimeError('Non-finite training metric: ' + str(log_path))
    checkpoint_valid(output, steps)
    if not smoke:
        for step in range(1000, steps + 1, 1000):
            checkpoint_valid(output, step)
    state.setdefault('completed_runs', []).append({'run_name': run_name, 'output': str(output),
        'phase': kind, 'size': 384, 'micro_batch': micro, 'steps': steps, 'log': str(log_path)})
    write_state(state, 'phase_completed', child_pid=None)
    return True


def run_evaluation(state, queue_id, micro, step, deadline):
    wait_idle(state, deadline)
    output = OUTPUT_ROOT / f'{queue_id}_train_b1_384_m{micro}'
    checkpoint_valid(output, step)
    log_path = STATE_DIR / f'{queue_id}_eval_{step}.log'
    args = [str(TRAIN_ENV / 'bin/python'), str(REPO / 'tools/eval_resolution_checkpoint.py'),
            '--checkpoint', str(output / f'checkpoints/global_step_{step}/hf_ckpt'),
            '--wandb-settings', str(STATE_DIR / 'wandb_settings.json'), '--group', queue_id]
    env = os.environ.copy()
    # Do not inherit stale tmux proxy endpoints into this experiment.
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
        env.pop(key, None)
    for key in ['WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_DISABLED', 'WANDB_ANONYMOUS']:
        env.pop(key, None)
    env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', WANDB_MODE='online', WANDB_DIR=str(STATE_DIR / 'wandb'))
    with log_path.open('x') as log:
        child = subprocess.Popen(args, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_state(state, 'evaluation', child_pid=child.pid, target_steps=step,
                    output=str(output), log=str(log_path), evaluation_episodes=350)
        code = child.wait()
    if code:
        raise RuntimeError(f'Evaluation step {step} failed ({code}); inspect {log_path}')
    state.setdefault('evaluations', []).append({'step': step, 'log': str(log_path), 'episodes': 350})
    write_state(state, 'evaluation_completed', child_pid=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--not-before', required=True, help='UTC ISO timestamp')
    parser.add_argument('--queue-id', default='qlexp384_10k_20261018')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.queue_id)
    target = dt.datetime.fromisoformat(args.not_before.replace('Z', '+00:00')).timestamp()
    validate_setup()
    if args.dry_run:
        cards, apps = gpu_snapshot()
        print(json.dumps({'setup_valid': True, 'not_before_utc': args.not_before,
                          'currently_idle': snapshot_idle(cards, apps), 'queue_id': args.queue_id}))
        return
    import fcntl
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = STATE_DIR / 'status.json'
        if status_path.exists():
            old = json.loads(status_path.read_text())
            if old.get('queue_id') == args.queue_id:
                raise RuntimeError('Queue already has saved state; inspect before restarting')
        state = {'queue_id': args.queue_id, 'pid': os.getpid(), 'head': EXPECTED_HEAD,
                 'not_before_utc': args.not_before, 'completed_runs': []}
        write_state(state, 'waiting_for_scheduled_time')
        try:
            while time.time() < target:
                time.sleep(min(60, target - time.time()))
            validate_setup()
            deadline = float('inf')
            selected_micro = None
            for micro in [4, 2, 1]:
                if (run_phase(state, args.queue_id, micro, True, deadline)
                        and run_phase(state, args.queue_id, micro, True, deadline, resume=True)):
                    selected_micro = micro
                    break
            if selected_micro is None:
                raise RuntimeError('384 smoke OOM even at micro batch 1')
            run_phase(state, args.queue_id, selected_micro, False, deadline)
            run_evaluation(state, args.queue_id, selected_micro, 5000, deadline)
            run_phase(state, args.queue_id, selected_micro, False, deadline, resume=True)
            run_evaluation(state, args.queue_id, selected_micro, 10000, deadline)
            write_state(state, 'complete', child_pid=None)
        except Exception as error:
            write_state(state, 'failed', error=type(error).__name__ + ': ' + str(error))
            raise


if __name__ == '__main__':
    main()
