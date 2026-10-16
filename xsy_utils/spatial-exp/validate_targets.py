#!/usr/bin/env python
"""Independent checks of D4rtSampler's targets (CPU; no model, no lingbot dataset).

Each task's xyz answer is mapped back through a path that does not reuse the sampler's own
transform chain, and compared with what it must equal:

* depth  : xyz (in c_src @ t) projected with c_src's intrinsic -> the cell's own canvas pixel.
* time   : xyz (in c_src @ t) -> world via c_src @ t, -> c_src @ t+dt, projected -> the anchor
           (tracked) cell's pixel in frame t+dt.
* reg    : xyz (in c_tgt @ t) -> world via c_tgt -> c_src @ t, projected -> the cell's pixel.
* handeye: xyz (in gripper @ t) -> world via the endpose -> c_src (cam_high), projected -> pixel;
           and the gripper origin itself maps to the endpose position.
Also: target range / clip rate, supervised fraction per task.

    python validate_targets.py --tasks place_shoe,stack_blocks_three --episodes 0,7 --frames 40
"""

from __future__ import annotations

import argparse
import collections

import numpy as np

from d4rt_lingbot import data as D


def project(xyz_cam: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """camera-frame xyz -> canvas continuous (row, col) of pixel indices."""
    u = k[0, 0] * xyz_cam[..., 0] / xyz_cam[..., 2] + k[0, 2]
    v = k[1, 1] * xyz_cam[..., 1] / xyz_cam[..., 2] + k[1, 2]
    return D.native_to_canvas(v, u)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="place_shoe,stack_blocks_three,put_bottles_dustbin,open_laptop")
    ap.add_argument("--episodes", default="0,7")
    ap.add_argument("--frames", type=int, default=40, help="frames per episode, evenly spaced")
    args = ap.parse_args()

    s = D.D4rtSampler()
    p, half = s.p, s.p // 2
    err = collections.defaultdict(list)
    n_task = collections.Counter()
    n_sup = collections.Counter()
    vals = []
    imgs = np.zeros((3, 3, D.CANVAS, D.CANVAS), np.float32)
    for task in args.tasks.split(","):
        for ep in map(int, args.episodes.split(",")):
            e = s._store.get(task, ep)
            assert e is not None, (task, ep)
            T = e["depth"][0].shape[0]
            for frame in np.linspace(0, T - 2, args.frames).round().astype(int):
                out = s(task, ep, int(frame), imgs)
                rr, cc = np.meshgrid(np.arange(p) - half, np.arange(p) - half, indexing="ij")
                for j in range(s.k):
                    t = int(out["d4rt_task"][j])
                    n_task[t] += 1
                    if not out["d4rt_mask"][j]:
                        continue
                    n_sup[t] += 1
                    xyz = out["d4rt_target"][j].astype(np.float64) * s.xyz_scale
                    vals.append(out["d4rt_target"][j])
                    cs, tg, dt = int(out["d4rt_cam"][j]), int(out["d4rt_tgt"][j]), int(out["d4rt_dt"][j])
                    r0, c0 = out["d4rt_coords"][j]
                    if t == D.T_DEPTH:
                        pr, pc = project(xyz, e["intr"][cs, frame])
                        er, ec = r0 + rr, c0 + cc
                    elif t == D.T_TIME:
                        pw = D.cam_to_world(xyz, e["extr"][cs, frame])
                        pr, pc = project(D.world_to_cam(pw, e["extr"][cs, frame + dt]), e["intr"][cs, frame + dt])
                        # anchor pixel of the tracked point: recover it from the patch centre
                        er = np.floor(pr[half, half] + 0.5) + rr
                        ec = np.floor(pc[half, half] + 0.5) + cc
                        err["time_centre_disp_px"].append(np.hypot(er[half, half] - r0, ec[half, half] - c0))
                    elif t == D.T_REG:
                        pw = D.cam_to_world(xyz, e["extr"][tg, frame])
                        pr, pc = project(D.world_to_cam(pw, e["extr"][cs, frame]), e["intr"][cs, frame])
                        er, ec = r0 + rr, c0 + cc
                    else:
                        arm = tg - D.N_CAMS
                        pos = e["ee"][frame, arm * 8 : arm * 8 + 3].astype(np.float64)
                        rot = D.quat_wxyz_to_mat(e["ee"][frame, arm * 8 + 3 : arm * 8 + 7])
                        pw = xyz @ rot.T + pos  # gripper -> world
                        pr, pc = project(D.world_to_cam(pw, e["extr"][cs, frame]), e["intr"][cs, frame])
                        er, ec = r0 + rr, c0 + cc
                        o = D.world_to_gripper(pos[None], pos, rot)
                        err["handeye_origin_mm"].append(float(np.linalg.norm(o)) * 1000)
                    # cell pixel index i covers continuous [i, i+1) in canvas index units -> centre i
                    e_px = np.hypot(pr - er, pc - ec)
                    err[f"{D.TASK_NAMES[t]}_px"].append(float(e_px.max()))

    print("supervised fraction per task:")
    for t, name in enumerate(D.TASK_NAMES):
        print(f"  {name:8s} {n_sup[t]:5d}/{n_task[t]:5d} = {n_sup[t] / max(n_task[t], 1):.3f}")
    print("round-trip error (max over the 14x14 cells of each query; canvas px):")
    for key in sorted(err):
        a = np.asarray(err[key])
        print(f"  {key:22s} n={len(a):5d}  median {np.median(a):.3f}  p99 {np.percentile(a, 99):.3f}  max {a.max():.3f}")
    v = np.concatenate([x.reshape(-1, 3) for x in vals])
    print(f"normalized target range per axis: min {v.min(0).round(3)}  max {v.max(0).round(3)}  "
          f"clipped {(np.abs(v) >= 1.0).mean() * 100:.4f} %")


if __name__ == "__main__":
    main()
