"""Durable episode results and run compatibility checks (no simulator imports)."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EpisodeCheckpoint:
    def __init__(self, directory, identity, episodes, start_seed=100000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / '_progress.json'
        self.identity = identity
        self.target = episodes
        self.lock = (self.directory / '_progress.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError(f'Another evaluator is using {self.directory}')
        try:
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
            else:
                if (self.directory / '_result.txt').exists() or list(self.directory.glob('episode*.mp4')):
                    raise ValueError(f'Legacy results without resume state: {self.directory}; use a new run directory')
                self.state = dict(version=1, identity=identity, episodes=[],
                                  next_seed=start_seed, pending=None)
                self.save()
            self.validate()
        except Exception:
            self.lock.close()
            raise

    def validate(self):
        state = self.state
        if state.get('version') != 1 or state.get('identity') != self.identity:
            raise ValueError(f'Checkpoint/task/settings mismatch in {self.path}; use a new run directory')
        previous = -1
        for index, entry in enumerate(state['episodes']):
            if entry['episode'] != index or type(entry['success']) is not bool or entry['seed'] <= previous:
                raise ValueError(f'Invalid episode history: {self.path}')
            previous = entry['seed']
        if state['next_seed'] <= previous or self.completed > self.target:
            raise ValueError(f'Invalid next seed or target smaller than completed episodes: {self.path}')
        pending = state.get('pending')
        if pending and (pending['episode'] != self.completed or pending['seed'] != self.next_seed):
            raise ValueError(f'Invalid pending episode: {self.path}')

    @property
    def completed(self):
        return len(self.state['episodes'])

    @property
    def successes(self):
        return sum(item['success'] for item in self.state['episodes'])

    @property
    def next_seed(self):
        return self.state['next_seed']

    def save(self):
        atomic_json(self.path, self.state)

    def skip_seed(self, seed):
        if seed != self.next_seed or self.state['pending']:
            raise ValueError('Cannot skip a different seed or an interrupted valid episode')
        self.state['next_seed'] = seed + 1
        self.save()

    def begin(self, seed, instruction):
        if seed != self.next_seed:
            raise ValueError('Episode seed does not match saved progress')
        pending = self.state['pending']
        if pending is None:
            pending = dict(episode=self.completed, seed=seed, instruction=instruction)
            self.state['pending'] = pending
            self.save()
        # An interrupted episode may have finalized a video before committing.
        for tag in ('success', 'fail'):
            (self.directory / f'episode{self.completed}_{tag}.mp4').unlink(missing_ok=True)
        self.started = time.monotonic()
        return pending['instruction']

    def commit(self, success, steps):
        pending = self.state['pending']
        if pending is None:
            raise ValueError('No pending episode')
        self.state['episodes'].append(dict(pending, success=bool(success), steps=int(steps),
                                           seconds=time.monotonic() - self.started))
        self.state['next_seed'] = pending['seed'] + 1
        self.state['pending'] = None
        self.save()

    def close(self):
        self.lock.close()


def task_identity(run_id, task, setting, instruction, seed=0):
    return dict(run_id=run_id, task=task, task_config=setting,
                instruction_type=instruction, seed=seed)


def prepare(args):
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model = Path(args.model).resolve()
    shards = sorted(model.glob('*.safetensors'))
    if not shards:
        raise ValueError(f'No safetensors in {model}; pass the hf_ckpt directory')
    config = Path(args.config).resolve()
    identity = dict(model=str(model), shards=[dict(name=p.name, size=p.stat().st_size,
                    mtime_ns=p.stat().st_mtime_ns) for p in shards],
                    inference_config_sha256=file_digest(config),
                    task_config=args.setting, task_config_sha256=file_digest(args.task_config_file),
                    instruction_type=args.instruction, seed=0, use_length=args.length,
                    robo_name=args.robo_name, precision=args.precision,
                    denoiser=os.environ.get('ROBOTWIN_DENOISER', 'default'))
    if getattr(args, 'rtc_async', 'False') == 'True':
        identity['rtc'] = dict(mode='async_fixed_prefix_v1', replan_steps=args.rtc_replan_steps, delay=args.rtc_delay)
    prepare_run(args, identity)


def prepare_run(args, identity):
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'run_manifest.json'
    run_id = digest(identity)
    if manifest.exists():
        if json.loads(manifest.read_text())['identity'] != identity:
            raise ValueError(f'Checkpoint or evaluation settings differ from {manifest}; use a new --run_dir')
    else:
        atomic_json(manifest, dict(version=1, run_id=run_id, identity=identity))
    pending, complete = [], []
    for task in args.tasks:
        checkpoint = EpisodeCheckpoint(root / 'eval_results' / task,
            task_identity(run_id, task, args.setting, args.instruction), args.episodes)
        try:
            if checkpoint.completed == args.episodes:
                # The process may have died after its final durable commit but
                # before writing the legacy result file. Repair it when skipping.
                (checkpoint.directory / '_result.txt').write_text(
                    f'Instruction Type: {args.instruction}\n\n'
                    f'{checkpoint.successes / args.episodes}')
                complete.append(task)
            else:
                pending.append(task)
        finally:
            checkpoint.close()
    atomic_json(root / 'resume_plan.json', dict(run_id=run_id, pending=pending, complete=complete))
    print(f'Resume: {len(complete)} tasks complete, {len(pending)} tasks pending; {root}', flush=True)


def stats(args):
    root = Path(args.run_dir)
    manifest = root / 'run_manifest.json'
    identity = json.loads(manifest.read_text())['identity'] if manifest.exists() else {}
    lines = ['Eval Result Stats (saved episode results)', f'Run directory: {root}',
             f'Model path: {identity.get("model", "remote")}',
             f'Task config: {identity.get("task_config", "see task progress")}',
             f'Instruction type: {identity.get("instruction_type", "see task progress")}',
             f'Precision (bf16/fp32/compile): {identity.get("precision", "remote")}',
             f'Episodes per task: {args.episodes}', '',
             f'{"Task":30} {"Done":12} {"Success/Total":14} Rate']
    successes = completed = 0
    for task in args.tasks:
        path = root / 'eval_results' / task / '_progress.json'
        entries = json.loads(path.read_text())['episodes'] if path.exists() else []
        total = len(entries)
        good = sum(entry['success'] for entry in entries)
        successes += good
        completed += total
        rate = f'{good / total * 100:.1f}%' if total else 'N/A'
        lines.append(f'{task:30} {total}/{args.episodes:<8} {good}/{total:<12} {rate}')
    rate = f'{successes / completed * 100:.1f}%' if completed else 'N/A'
    lines.extend(['', f'Summary: success {successes}/{completed}, overall rate {rate}',
                  f'Completed episodes: {completed}/{len(args.tasks) * args.episodes}'])
    output = '\n'.join(lines) + '\n'
    (root / 'stats.txt').write_text(output)
    print(output, end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'stats'):
        item = sub.add_parser(name)
        item.add_argument('--run-dir', required=True)
        item.add_argument('--episodes', type=int, required=True)
        item.add_argument('--tasks', nargs='+', required=True)
        if name == 'prepare':
            for option in ('model', 'config', 'setting', 'instruction', 'task-config-file', 'robo-name', 'precision'):
                item.add_argument('--' + option, required=True)
            item.add_argument('--rtc-async', choices=['True', 'False'], default='False')
            item.add_argument('--rtc-replan-steps', type=int, default=8)
            item.add_argument('--rtc-delay', type=int, default=6)
            item.add_argument('--length', type=int, required=True)
    args = parser.parse_args()
    try:
        if args.episodes < 1:
            raise ValueError('Episode target must be positive')
        (prepare if args.command == 'prepare' else stats)(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
