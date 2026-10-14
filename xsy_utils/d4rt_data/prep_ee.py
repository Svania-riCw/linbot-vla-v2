#!/usr/bin/env python
"""End-effector poses + per-frame camera geometry from the raw RoboTwin HDF5 (CPU only).

Per (task, episode), under ``OUT_ROOT/<task>/``:

``ee/episode_XXXXXX.npy``   float32 [T_raw, 16]
    ``[Lpos(3), Lquat_wxyz(4), Lgrip(1), Rpos(3), Rquat_wxyz(4), Rgrip(1)]``, **world** frame,
    straight from ``endpose/``. Same layout as pi05-experiments' ``prep_ee.py``: quaternion is
    wxyz (``transforms3d.mat2quat``), the position includes RoboTwin's tool-axis gripper_bias,
    i.e. it is what ``take_action(action_type='ee')`` consumes. State uses row ``t``, action
    step ``k`` uses row ``t + 1 + k``.
``cam/episode_XXXXXX.npz``
    ``intrinsic`` float32 [3, T_raw, 3, 3]  OpenCV K at native 320x240, cameras in ``cameras`` order
    ``extrinsic`` float32 [3, T_raw, 3, 4]  OpenCV world->camera ``[R|t]`` (wrists move every frame)
    ``grip``      float32 [T_raw, 3, 2, 4]  per camera x (left, right) gripper: ``(u, v, z, valid)``
                  -- the endpose position projected into each camera, native pixels, ``z`` in metres
                  along the optical axis, ``valid`` = in front of the camera and inside the image.
                  A wrist camera never sees its own arm's endpose point: it sits ~4 cm *behind*
                  that camera (z < 0, u = v = -1), so only the other arm can be valid there.
    ``cameras``   str [3]

Before writing anything for an episode the lerobot data is checked against the raw file
(``T_lerobot == T_raw - 1`` and ``state == joint_action/vector[:-1]``), which is what pins the
lerobot-episode -> raw-HDF5 mapping; a mismatch is reported and nothing is written.

    python prep_ee.py                     # all 50 tasks
    python prep_ee.py --tasks place_shoe  # subset
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import time

import numpy as np

import common as C


def lerobot_states(task: str) -> dict[int, np.ndarray]:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(C.LEROBOT_ROOT / f"robotwin_clean_{task}" / "data" / "*" / "*.parquet")))
    out: dict[int, list] = {}
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pandas()
        for ep, g in t.groupby("episode_index"):
            g = g.sort_values("frame_index")
            out.setdefault(int(ep), []).append(np.stack(g["observation.state"].to_numpy()))
    return {ep: np.concatenate(v) for ep, v in out.items()}


def project(xyz: np.ndarray, intr: np.ndarray, extr: np.ndarray) -> np.ndarray:
    """world xyz [T,3], K [T,3,3], [R|t] [T,3,4] -> [T,4] (u, v, z, valid)."""
    pc = np.einsum("tij,tj->ti", extr[:, :, :3].astype(np.float64), xyz.astype(np.float64)) + extr[:, :, 3]
    z = pc[:, 2]
    uv = np.einsum("tij,tj->ti", intr.astype(np.float64), pc)[:, :2] / np.maximum(z, 1e-6)[:, None]
    ok = (z > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < C.NATIVE_W) & (uv[:, 1] >= 0) & (uv[:, 1] < C.NATIVE_H)
    uv[z <= 1e-3] = -1.0  # behind the camera: the division is meaningless
    return np.concatenate([uv, z[:, None], ok[:, None]], axis=1).astype(np.float32)


def do_task(args: tuple[str, bool]) -> tuple[str, dict[str, int], list[str]]:
    import h5py

    task, overwrite = args
    states = lerobot_states(task)
    counts: dict[str, int] = {}
    problems: list[str] = []
    for ep in C.episodes(task):
        ee_path, cam_path = C.out_file("ee", task, ep), C.out_file("cam", task, ep, ext=".npz")
        if not overwrite and ee_path.exists() and cam_path.exists():
            counts["skip"] = counts.get("skip", 0) + 1
            continue
        with h5py.File(C.raw_path(task, ep), "r") as f:
            vec = f["joint_action/vector"][:]
            lp, rp = f["endpose/left_endpose"][:], f["endpose/right_endpose"][:]
            lg, rg = f["endpose/left_gripper"][:], f["endpose/right_gripper"][:]
            intr = np.stack([f[f"observation/{g}/intrinsic_cv"][:] for g in C.CAMERAS.values()])
            extr = np.stack([f[f"observation/{g}/extrinsic_cv"][:] for g in C.CAMERAS.values()])
        s = states.get(ep)
        status = "ok"
        if s is None:
            status = "no-lerobot-episode"
        elif len(s) != len(vec) - 1:
            status = f"length(lerobot {len(s)} != raw {len(vec)} - 1)"
        elif not np.allclose(s, vec[:-1], atol=1e-5):
            status = f"state-mismatch(max {np.abs(s - vec[:-1]).max():.2e})"
        else:
            ee = np.concatenate([lp, lg[:, None], rp, rg[:, None]], axis=1).astype(np.float32)
            qn = np.linalg.norm(ee[:, [3, 4, 5, 6]], axis=1), np.linalg.norm(ee[:, [11, 12, 13, 14]], axis=1)
            if not all(np.allclose(q, 1.0, atol=1e-3) for q in qn):
                status = "bad-quat"
        if status != "ok":
            problems.append(f"{task}/ep{ep}: {status}")
            counts["bad"] = counts.get("bad", 0) + 1
            continue
        grip = np.stack(
            [np.stack([project(p[:, :3], intr[c], extr[c]) for p in (lp, rp)], axis=1) for c in range(3)], axis=1
        )
        C.atomic_save(ee_path, lambda fh: np.save(fh, ee))
        C.atomic_save(
            cam_path,
            lambda fh: np.savez(
                fh,
                intrinsic=intr.astype(np.float32),
                extrinsic=extr.astype(np.float32),
                grip=grip,
                cameras=np.array(list(C.CAMERAS)),
            ),
        )
        counts["ok"] = counts.get("ok", 0) + 1
    return task, counts, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=25)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tasks = C.tasks() if args.tasks == "all" else args.tasks.split(",")
    t0 = time.time()
    total: dict[str, int] = {}
    problems: list[str] = []
    with mp.get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        for i, (task, counts, probs) in enumerate(
            pool.imap_unordered(do_task, [(t, args.overwrite) for t in tasks]), 1
        ):
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
            problems += probs
            print(f"[prep_ee] {i}/{len(tasks)} {task}: {counts}  ({time.time() - t0:.0f}s)", flush=True)
    for p in problems:
        print(f"[prep_ee] PROBLEM {p}")
    print(f"[prep_ee] done in {time.time() - t0:.0f}s: {total}")
    (C.OUT_ROOT / "EE_INFO.json").write_text(
        json.dumps(
            {
                "ee": "[Lpos(3), Lquat_wxyz(4), Lgrip(1), Rpos(3), Rquat_wxyz(4), Rgrip(1)], world frame, incl. gripper_bias",
                "cam.intrinsic": "[3 cams, T_raw, 3, 3] OpenCV K, native 320x240 pixels",
                "cam.extrinsic": "[3 cams, T_raw, 3, 4] OpenCV world->camera [R|t]",
                "cam.grip": "[T_raw, 3 cams, 2 arms (L, R), (u, v, z_m, valid)] endpose position projected",
                "cameras": list(C.CAMERAS),
                "length": "T_raw = lerobot length + 1; row t = lerobot frame t; action step k = row t+1+k",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
