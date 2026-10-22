"""Explicit worker-count migration for stateless map-style torchdata snapshots."""
import copy


def migrate_loader_workers(state, num_workers):
    """Keep the consumed sampler position; rebuild only empty worker snapshots.

    This is intentionally limited to torchdata's ordered, multiprocess map
    loader at an exact snapshot boundary. Worker-local random streams change
    when the worker count changes; this is not a bitwise replay guarantee.
    The caller must reject iterable/dynamic/out-of-order loaders.
    """
    snapshot = state.get('_snapshot', {})
    main = snapshot.get('_main_snapshot', {})
    old_workers = main.get('_num_workers')
    if old_workers == num_workers:
        return state
    if not isinstance(old_workers, int) or old_workers < 1 or num_workers < 1:
        raise ValueError('Worker migration requires multiprocess snapshots')
    if state.get('_steps_since_snapshot') != 0 or state.get('_iterator_finished') is not False:
        raise ValueError('Worker migration requires an exact, unfinished snapshot')
    sampler = main.get('_sampler_iter_state')
    if not sampler or main.get('_IterableDataset_len_called') is not None:
        raise ValueError('Worker migration requires a stateful map sampler')
    if main.get('_sampler_iter_yielded') != snapshot.get('_snapshot_step'):
        raise ValueError('Snapshot and consumed batch positions disagree')
    workers = snapshot.get('_worker_snapshots', {})
    if set(workers) != {f'worker_{i}' for i in range(old_workers)}:
        raise ValueError('Incomplete worker snapshot')
    for i in range(old_workers):
        if workers[f'worker_{i}'] != dict(worker_id=i, dataset_state=None, fetcher_state=None):
            raise ValueError('Cannot redistribute worker-local dataset/fetcher state')
    result = copy.deepcopy(state)
    snap = result['_snapshot']
    snap['_main_snapshot']['_num_workers'] = num_workers
    snap['_last_yielded_worker_id'] = (snap['_snapshot_step'] - 1) % num_workers
    snap['_worker_snapshots'] = {
        f'worker_{i}': dict(worker_id=i, dataset_state=None, fetcher_state=None)
        for i in range(num_workers)
    }
    return result
