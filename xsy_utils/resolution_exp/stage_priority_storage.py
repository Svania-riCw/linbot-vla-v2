"""Stage the immutable successful smoke on shared storage before any candidate runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from priority_worker import atomic_json


def files(root):
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    args = parser.parse_args()
    shared, source = args.shared.resolve(), args.source.resolve()
    assert source.is_relative_to(Path('/root/ql/outputs').resolve())
    assert shared.is_relative_to(Path('/2025233210/ql').resolve())
    target = shared / 'outputs' / source.name
    assert target.is_relative_to(shared) and source != target
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        before = files(source)
        subprocess.run(['nice', '-n', '10', 'rsync', '-a', '--partial', str(source) + '/', str(target) + '/'], check=True)
        after = files(target)
        if before != after or files(source) != before:
            raise RuntimeError('Checkpoint copy is incomplete or source changed')
        checkpoint = target / 'checkpoints/global_step_2'
        for relative in ['model/.metadata', 'optimizer/.metadata', 'hf_ckpt/config.json', 'hf_ckpt/model.safetensors.index.json']:
            src = source / 'checkpoints/global_step_2' / relative
            dst = checkpoint / relative
            if hashlib.sha256(src.read_bytes()).digest() != hashlib.sha256(dst.read_bytes()).digest():
                raise RuntimeError('Checkpoint metadata copy differs: ' + relative)
        if len(list((checkpoint / 'extra_state').glob('extra_state_rank_*.pt'))) != 8:
            raise RuntimeError('Missing rank checkpoint state')
        atomic_json(shared / 'storage_ready.json', dict(complete=True, source=str(source),
                    target=str(target), files=len(before), bytes=sum(before.values()),
                    elapsed_seconds=time.time() - started))
    except Exception as error:
        atomic_json(shared / 'storage_ready.json', dict(complete=False,
                    error=type(error).__name__ + ': ' + str(error)))
        raise


if __name__ == '__main__':
    main()
