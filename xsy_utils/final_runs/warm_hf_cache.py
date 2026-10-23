"""Build (or just open) the HF-datasets Arrow cache of every LeRobot dataset in a config's train list,
before torch.distributed.run starts.

    python xsy_utils/final_runs/warm_hf_cache.py <config.yaml> [--train-path LIST] [--jobs 8]

Why: every rank opens its datasets through ``Dataset.from_parquet``, which converts the parquet to an Arrow
cache under $HF_HOME on first use. Within one node the ranks serialise on a FileLock; across nodes that lock
is not shared (flock on the CPFS fuse mount is node-local) and the build writes a fixed ``<dir>.incomplete``,
so two nodes building the same dataset can corrupt it. train.sh runs this on node 0 only and the other nodes
wait for it. It also builds the datasets in parallel (the ranks would build them one after another).
Loads exactly what the trainer loads (lerobot's LeRobotDataset on the same paths and features), so the cache
fingerprints match; with the cache in place it only opens it (~1 s per dataset).
"""
import argparse
import multiprocessing as mp
import pathlib
import sys
import time

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]


def warm(path: str) -> tuple[str, int, float]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    t = time.time()
    ds = LeRobotDataset(repo_id=path)  # what lingbotvla's VLADataset does (repo_id = local path, root = None)
    return path, len(ds.hf_dataset), time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--train-path", help="override data.train_path (a train list)")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    lst = args.train_path or yaml.safe_load(open(args.config))["data"]["train_path"]
    lst = pathlib.Path(lst) if str(lst).startswith("/") else REPO / lst
    paths = [line.split()[1] for line in lst.read_text().splitlines() if line.strip()]
    t0, frames = time.time(), 0
    with mp.get_context("spawn").Pool(min(args.jobs, len(paths))) as pool:
        for i, (p, n, dt) in enumerate(pool.imap_unordered(warm, paths), 1):
            frames += n
            if dt > 30:
                print(f"[warm_hf_cache] built {pathlib.Path(p).name}: {n} frames, {dt:.0f} s", flush=True)
    print(f"[warm_hf_cache] {len(paths)} datasets, {frames} frames ready in {time.time() - t0:.0f} s ({lst.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
