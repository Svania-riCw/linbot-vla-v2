"""Preflight for a final-run config: every input the run reads exists and the d4rt data matches the
LeRobot episodes (ee rows == lerobot length + 1, every per-camera file present).

    python xsy_utils/final_runs/check_data.py xsy_utils/final_runs/configs/<run>.yaml [--quick] [--compare-ee]

--compare-ee: the copied ee/ and cam/ must equal, value for value, the ones prep_ee.py regenerated on this
side from the PPU raw HDF5 (.runtime/d4rt_ee_reference) -- i.e. the H20 targets were built from the same
raw episodes as the LeRobot copy trained on here.
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "xsy_utils/spatial-exp"))
from d4rt_lingbot.data import CAMERAS, task_from_repo  # noqa: E402

REFERENCE = REPO / ".runtime/d4rt_ee_reference"
PER_CAM = (("depth", ".npy"), ("depth_meta", ".npz"), ("tracks", ".npy"), ("tvis", ".npy"), ("tmeta", ".npz"))


def lengths(ds: pathlib.Path) -> list[int]:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(ds / "meta" / "episodes" / "*" / "*.parquet")))
    t = pq.read_table(files, columns=["episode_index", "length"]) if len(files) == 1 else \
        __import__("pyarrow").concat_tables([pq.read_table(f, columns=["episode_index", "length"]) for f in files])
    order = np.argsort(t["episode_index"].to_numpy())
    return t["length"].to_numpy()[order].tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--quick", action="store_true", help="skip the per-episode ee length check")
    ap.add_argument("--compare-ee", action="store_true", help=f"compare ee/cam with {REFERENCE}")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d, t = cfg["data"], cfg["train"]
    errs = []

    def need(p, what):
        if not pathlib.Path(p).exists():
            errs.append(f"missing {what}: {p}")

    rel = lambda p: pathlib.Path(p) if str(p).startswith("/") else REPO / p  # noqa: E731
    need(cfg["model"]["model_path"] + "/model.safetensors.index.json", "base weights")
    need(cfg["model"]["tokenizer_path"] + "/config.json", "Qwen3-VL")
    a = t.get("align_params") or {}
    for p, what in ((a.get("depth", {}).get("moge_path"), "MoGe-2"), (a.get("depth", {}).get("morgbd_path"), "MoRGBD"),
                    (a.get("video", {}).get("ckpt_path"), "DINO-video"), (a.get("video", {}).get("config_path"), "DINO-video cfg")):
        if p:
            need(p, what)
    robot = rel(d["robot_config_root"]) / "robotwin.yaml"
    need(robot, "robot config")
    if robot.exists():
        need(rel(yaml.safe_load(robot.read_text())["norm_stats"]), "norm stats")
    datasets = [line.split()[1] for line in rel(d["train_path"]).read_text().splitlines() if line.strip()]
    for ds in datasets:
        need(pathlib.Path(ds) / "meta" / "info.json", "LeRobot dataset")

    d4 = t.get("d4rt_params") or {}
    n_ep = 0
    if d4.get("enable"):
        root = pathlib.Path(d4["root"])
        if not root.is_dir():
            errs.append(f"missing d4rt data root {root} (README.md: copy H20 /2025233210/xiesy/data/lingbot_robotwin_clean_d4rt here)")
        else:
            for ds in datasets:
                task = task_from_repo(ds)
                if not (pathlib.Path(ds) / "meta" / "info.json").exists():
                    continue
                lens = lengths(pathlib.Path(ds))
                for ep, n in enumerate(lens):
                    name = f"episode_{ep:06d}"
                    files = [root / task / "ee" / f"{name}.npy", root / task / "cam" / f"{name}.npz"]
                    files += [root / task / k / c / f"{name}{ext}" for k, ext in PER_CAM for c in CAMERAS]
                    miss = [str(f) for f in files if not f.exists()]
                    if miss:
                        errs.append(f"d4rt {task} ep{ep}: {len(miss)} files missing, e.g. {miss[0]}")
                        continue
                    if not args.quick:
                        rows = np.load(files[0], mmap_mode="r").shape[0]
                        if rows != n + 1:
                            errs.append(f"d4rt {task} ep{ep}: ee has {rows} rows, lerobot length {n} (+1 expected)")
                    if args.compare_ee:
                        ref = REFERENCE / task
                        if not np.array_equal(np.load(files[0]), np.load(ref / "ee" / f"{name}.npy")):
                            errs.append(f"d4rt {task} ep{ep}: ee differs from the PPU-side regeneration")
                        with np.load(files[1]) as a, np.load(ref / "cam" / f"{name}.npz") as b:
                            for k in ("intrinsic", "extrinsic", "grip"):
                                if not np.array_equal(a[k], b[k]):
                                    errs.append(f"d4rt {task} ep{ep}: cam/{k} differs from the PPU-side regeneration")
                    n_ep += 1
    if errs:
        print(f"[check_data] {len(errs)} problem(s):", *errs[:30], sep="\n  ")
        return 1
    print(f"[check_data] ok: {len(datasets)} datasets" + (f", d4rt {n_ep} episodes" if d4.get("enable") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
