"""Build EE sidecars only after auditing EVERY state and action against raw HDF5.

Uses the same raw layout as xsy_utils/d4rt_data/prep_ee.py; additionally checks
action[t] == raw_joint[t+1]. Failure leaves no usable manifest for that dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.action_spaces.ee_labels import audit_episode


def prepare_dataset(dataset, raw_directory, output):
    import h5py
    import pyarrow.parquet as pq
    dataset, raw_directory, output = map(Path, (dataset, raw_directory, output))
    files = sorted((dataset/'data').rglob('*.parquet'))
    if not files:
        raise ValueError(f'No parquet files: {dataset}')
    rows = {}
    for path in files:
        table = pq.read_table(path, columns=['episode_index','frame_index','observation.state','action']).to_pydict()
        for ep, frame, state, action in zip(*(table[k] for k in ('episode_index','frame_index','observation.state','action'))):
            rows.setdefault(int(ep), []).append((int(frame),state,action))
    directory = output / dataset.name
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema='robotwin-world-tool-xyzw-v1', dataset=str(dataset.resolve()), episodes={},
                    source_parquet={str(p.relative_to(dataset)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    for ep, values in sorted(rows.items()):
        values.sort(key=lambda row: row[0])
        if [r[0] for r in values] != list(range(len(values))):
            raise ValueError(f'Episode {ep}: duplicate/missing/nonzero-based frame indices')
        states, actions = np.asarray([r[1] for r in values]), np.asarray([r[2] for r in values])
        raw_path = raw_directory/f'episode{ep}.hdf5'
        with h5py.File(raw_path, 'r') as f:
            q = f['joint_action/vector'][:]
            ee = np.concatenate([f['endpose/left_endpose'][:], f['endpose/left_gripper'][:].reshape(-1,1),
                                 f['endpose/right_endpose'][:], f['endpose/right_gripper'][:].reshape(-1,1)], axis=-1)
        audit_episode(states, actions, q, ee)
        path = directory/f'episode_{ep:06d}.npz'
        np.savez(path, joint=q.astype('float32'), ee=ee.astype('float32'))
        manifest['episodes'][str(ep)] = dict(frames=len(values), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                            raw_path=str(raw_path.resolve()))
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def verify_existing(dataset, output):
    """Reuse a complete audit only when both source Parquet and sidecars match."""
    dataset,output=Path(dataset).resolve(),Path(output)
    directory=output/dataset.name
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!='robotwin-world-tool-xyzw-v1' or Path(manifest['dataset']).resolve()!=dataset or not manifest['episodes']:
        raise ValueError('Invalid existing EE manifest')
    actual={str(p.relative_to(dataset)) for p in (dataset/'data').rglob('*.parquet')}
    if not actual or actual!=set(manifest['source_parquet']):
        raise ValueError('Existing EE audit has a different Parquet file set')
    checks=[(dataset/relative,expected) for relative,expected in manifest['source_parquet'].items()]
    checks += [(directory/f'episode_{int(ep):06d}.npz',info['sha256']) for ep,info in manifest['episodes'].items()]
    for path,expected in checks:
        hasher=hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda:file.read(1024*1024),b''): hasher.update(block)
        if hasher.hexdigest()!=expected:
            raise ValueError(f'Existing EE audit changed: {path}')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-list', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--raw-setting', default='aloha-agilex_clean_50')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-complete',action='store_true',help='Reuse fully verified datasets; partial datasets still fail')
    args = p.parse_args()
    for line in args.train_list.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        robot, location = line.split()
        dataset = Path(location)
        if robot != 'robotwin' or not dataset.name.startswith('robotwin_clean_'):
            raise ValueError('Only the audited robotwin_clean_<task> layout is supported')
        task = dataset.name.removeprefix('robotwin_clean_')
        if args.resume_complete and (args.output/dataset.name).exists():
            result=verify_existing(dataset,args.output)
        else:
            result = prepare_dataset(dataset, args.raw_root/task/args.raw_setting/'data', args.output)
        print(f'{task}: audited {len(result["episodes"])} episodes', flush=True)


if __name__ == '__main__':
    main()
