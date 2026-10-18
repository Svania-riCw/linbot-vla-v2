"""Three idle-node candidates compete for one durable shared experiment claim."""
import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_owner(shared):
    claim = Path(shared) / 'claim'
    if not claim.exists():
        return None
    owner = claim / 'owner.json'
    # A claim directory without a published owner is still taken, never free.
    return json.loads(owner.read_text()) if owner.is_file() else {'publishing': True}


def try_claim(shared, node, pid):
    claim = Path(shared) / 'claim'
    try:
        claim.mkdir()
    except FileExistsError:
        return None
    owner = dict(node=node, pid=pid, hostname=socket.gethostname(),
                 claimed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(claim / 'owner.json', owner)
    return owner


def resume_idle_claim(shared, node, pid, pid_alive=None):
    """Explicitly continue this host's stopped idle controller, keeping its claim."""
    shared = Path(shared)
    owner = read_owner(shared)
    state = json.loads((shared / 'engine_state/status.json').read_text())
    if (not owner or owner.get('node') != node
            or owner.get('hostname') != socket.gethostname() or owner.get('result')
            or state.get('phase') != 'waiting_for_idle_gpus'
            or state.get('owner_node') != node or state.get('pid') != owner.get('pid')
            or state.get('child_pid') is not None):
        raise RuntimeError('Only the same host may explicitly continue its idle, child-free controller')
    pid_alive = pid_alive or (lambda value: Path('/proc', str(value)).exists())
    if pid_alive(owner['pid']):
        raise RuntimeError('Previous controller is still alive; refusing a duplicate')
    previous = dict(owner)
    atomic_json(shared / 'claim' / ('previous_owner_' + uuid.uuid4().hex + '.json'), previous)
    owner.update(pid=pid, previous_pid=previous['pid'],
                 continued_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(shared / 'claim/owner.json', owner)
    return owner


def next_actions(state, queue_id, micro):
    """Resume the existing successful smoke; never repeat completed stages."""
    completed = {(r.get('phase'), r.get('micro_batch'), r.get('steps'))
                 for r in state.get('completed_runs', [])}
    actions = []
    if ('smoke', micro, 2) not in completed:
        actions.append(('smoke', 2))
    if ('smoke_resume', micro, 4) not in completed:
        actions.append(('smoke_resume', 4))
    if ('train_to_5000', micro, 5000) not in completed:
        actions.append(('train_to_5000', 5000))
    evaluations = {e['step'] for e in state.get('evaluations', [])}
    if 5000 not in evaluations:
        actions.append(('evaluation', 5000))
    if ('resume_to_10000', micro, 10000) not in completed:
        actions.append(('resume_to_10000', 10000))
    if 10000 not in evaluations:
        actions.append(('evaluation', 10000))
    return actions


def run_selected(engine, state, queue_id):
    micro = next((m for m in [4, 2, 1] if any(
        r.get('phase') == 'smoke_resume' and r.get('micro_batch') == m
        for r in state.get('completed_runs', []))), None)
    if micro is None:
        for candidate in [4, 2, 1]:
            actions = next_actions(state, queue_id, candidate)
            if ('smoke', 2) in actions and not engine.run_phase(state, queue_id, candidate, True, float('inf')):
                continue
            if ('smoke_resume', 4) in actions and not engine.run_phase(
                    state, queue_id, candidate, True, float('inf'), resume=True):
                continue
            micro = candidate
            break
    if micro is None:
        raise RuntimeError('Smoke OOM even at micro batch1')
    for phase, step in next_actions(state, queue_id, micro):
        if phase in ('smoke', 'smoke_resume'):
            raise RuntimeError('Completed smoke state is inconsistent')
        if phase == 'evaluation':
            engine.run_evaluation(state, queue_id, micro, step, float('inf'))
        else:
            engine.run_phase(state, queue_id, micro, False, float('inf'),
                             resume=(phase == 'resume_to_10000'))
    engine.write_state(state, 'complete', child_pid=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--node', choices=['node0', 'node1', 'node2'], required=True)
    parser.add_argument('--repo', type=Path, default=Path('/root/ql/linbot-vla-v2'))
    parser.add_argument('--poll-seconds', type=float, default=2)
    parser.add_argument('--resume-idle-claim', action='store_true',
                        help='Explicit same-host continuation after the idle controller has stopped')
    args = parser.parse_args()
    shared = args.shared.resolve()
    job = json.loads((shared / 'job.json').read_text())
    os.environ.update(QL_REPO=str(args.repo), QL_QUEUE_STATE_DIR=str(shared / 'engine_state'),
                      QL_OUTPUT_ROOT=str(shared / 'outputs'), QL_EXPECTED_HEAD=job['expected_head'],
                      QL_NODE_NAME=args.node)
    spec = importlib.util.spec_from_file_location('priority_queue_engine', args.repo / 'xsy_utils/resolution_exp/queue_engine.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    import fcntl
    lock_path = args.repo / '.runtime/priority_worker.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as local_lock:
        fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = shared / 'nodes' / (args.node + '.json')
        def status(phase, **fields):
            atomic_json(status_path, dict(node=args.node, pid=os.getpid(), phase=phase,
                        updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **fields))
        consecutive = 0
        continued_owner = resume_idle_claim(shared, args.node, os.getpid()) if args.resume_idle_claim else None
        while True:
            if continued_owner is not None:
                owner = continued_owner
            else:
                owner = read_owner(shared)
                if owner is not None:
                    if not owner.get('publishing'):
                        status('cancelled_other_node_selected', selected_node=owner['node'])
                        print(json.dumps({'cancelled': True, 'selected_node': owner['node']}), flush=True)
                        return
                    status('waiting_for_claim_publication')
                    time.sleep(args.poll_seconds)
                    continue
                ready = shared / 'storage_ready.json'
                if not ready.is_file():
                    status('waiting_for_checkpoint_copy')
                    time.sleep(args.poll_seconds)
                    continue
                copy_state = json.loads(ready.read_text())
                if not copy_state.get('complete'):
                    status('storage_copy_failed', copy_status=copy_state)
                    raise RuntimeError('Checkpoint staging failed; preserving all outputs')
                cards, apps = engine.gpu_snapshot()
                workloads = engine.active_gpu_workloads()
                good = engine.snapshot_idle(cards, apps) and not workloads
                consecutive = consecutive + 1 if good else 0
                status('confirming_idle' if good else 'waiting_for_idle_gpus',
                       confirmations=consecutive, gpu_snapshot=cards, gpu_processes=apps,
                       active_workloads=workloads)
                if consecutive < 3:
                    time.sleep(args.poll_seconds)
                    continue
                owner = try_claim(shared, args.node, os.getpid())
                if owner is None:
                    continue
            status('selected', selected_node=args.node)
            print(json.dumps({'selected': args.node, 'owner': owner}), flush=True)
            # The durable claim is deliberately retained on failure or completion.
            # Losing candidates exit and cannot silently start duplicate training.
            state = json.loads((shared / 'engine_state/status.json').read_text())
            state.update(pid=os.getpid(), head=job['expected_head'], owner_node=args.node)
            try:
                engine.validate_setup()
                def wait_local_idle(current, deadline):
                    count = 0
                    while True:
                        cards, apps = engine.gpu_snapshot()
                        workloads = engine.active_gpu_workloads()
                        good = engine.snapshot_idle(cards, apps) and not workloads
                        count = count + 1 if good else 0
                        if count >= 3:
                            return
                        if not good:
                            engine.write_state(current, 'waiting_for_idle_gpus',
                                               gpu_snapshot=cards, gpu_processes=apps,
                                               active_workloads=workloads,
                                               retry_interval_seconds=args.poll_seconds)
                        time.sleep(args.poll_seconds)
                engine.wait_idle = wait_local_idle
                run_selected(engine, state, job['queue_id'])
                status('complete', selected_node=args.node)
                owner.update(result='complete', finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                atomic_json(shared / 'claim/owner.json', owner)
            except Exception as error:
                engine.write_state(state, 'failed', error=type(error).__name__ + ': ' + str(error))
                status('failed', error=type(error).__name__ + ': ' + str(error))
                owner.update(result='failed', error=type(error).__name__ + ': ' + str(error))
                atomic_json(shared / 'claim/owner.json', owner)
                raise
            return


if __name__ == '__main__':
    main()
