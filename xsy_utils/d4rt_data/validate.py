#!/usr/bin/env python
"""Sanity checks + overlays for the d4rt prep outputs of some episodes.

Numbers (printed):
* depth scale: MoGe depth at the projected **gripper** pixel in cam_high vs the gripper's GT
  camera-frame z (from endpose + GT extrinsic). The endpose point is inside the gripper, so the
  surface MoGe sees there is a few cm in front of it -- expect a small *negative* bias, but the
  ratio should be ~1 and stable; a wrong metric scale shows up as a ratio far from 1.
* focal: MoGe focal vs GT focal (imposed via fov_x, so should match to ~0).
* cross-camera (independent of the plane fit): wrist depth on **non-plane** pixels (objects,
  arm) is unprojected to world with GT extrinsics, projected into cam_high and compared with
  cam_high's depth there. Occlusion in cam_high only makes its depth smaller, so look at the
  mode / the inlier share within 5%, not the mean.
* tracks: visibility fraction, and the drift of visible points over a window.

Overlay (``--out``/<task>_ep<N>.png): per camera, 4 frames: RGB with track tails (the window
starting at that frame, first 25 steps) and gripper markers | depth colormap.

    python validate.py --tasks place_shoe --episodes 0,1 --out /tmp/d4rt_val
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

import common as C


def depth_check(task: str, ep: int) -> dict:
    cam = np.load(C.out_file("cam", task, ep, ext=".npz"))
    depth = np.load(C.out_file("depth", task, ep, "cam_high"), mmap_mode="r")
    meta = np.load(C.out_file("depth_meta", task, ep, "cam_high", ext=".npz"))
    grip = cam["grip"][:, 0]  # cam_high, [T, 2, (u, v, z, valid)]
    gt, est = [], []
    for t in range(len(depth)):
        for a in range(2):
            u, v, z, ok = grip[t, a]
            if not ok:
                continue
            r0, c0 = int(v), int(u)
            patch = depth[t, max(r0 - 2, 0) : r0 + 3, max(c0 - 2, 0) : c0 + 3].astype(np.float32)
            patch = patch[patch > 0]
            if patch.size:
                gt.append(z)
                est.append(np.median(patch) / 1000.0)
    gt, est = np.array(gt), np.array(est)
    f_gt = cam["intrinsic"][0, 0, 0, 0]
    return {
        "n": len(gt),
        "bias_m": float(np.median(est - gt)) if len(gt) else float("nan"),
        "ratio": float(np.median(est / gt)) if len(gt) else float("nan"),
        "ratio_iqr": (float(np.percentile(est / gt, 25)), float(np.percentile(est / gt, 75))) if len(gt) else None,
        "focal_err_px": float(np.abs(meta["intrinsic"][:, 0, 0] - f_gt).max()),
        "valid_frac": float(meta["valid_frac"].mean()),
    }


def cross_check(task: str, ep: int, cam: str, stride: int = 5) -> dict:
    import scale_align as S

    geo = np.load(C.out_file("cam", task, ep, ext=".npz"))
    ci = list(C.CAMERAS).index(cam)
    dw = np.load(C.out_file("depth", task, ep, cam), mmap_mode="r")
    dh = np.load(C.out_file("depth", task, ep, "cam_high"), mmap_mode="r")
    rel_wrist = np.load(C.out_file("depth_meta", task, ep, cam, ext=".npz"))["reliable"]
    rel_high = np.load(C.out_file("depth_meta", task, ep, "cam_high", ext=".npz"))["reliable"]
    u, v = np.meshgrid(np.arange(C.NATIVE_W) + 0.5, np.arange(C.NATIVE_H) + 0.5)
    ratios = []
    for t in range(0, len(dw), stride):
        if not (rel_wrist[t] and rel_high[t]):
            continue
        Kw, Ew = geo["intrinsic"][ci, t].astype(np.float64), geo["extrinsic"][ci, t].astype(np.float64)
        Kh, Eh = geo["intrinsic"][0, t].astype(np.float64), geo["extrinsic"][0, t].astype(np.float64)
        z = dw[t].astype(np.float64) / 1000
        plane = S.plane_depth(Kw, Ew, C.NATIVE_H, C.NATIVE_W, task)
        m = (z > 0) & ~(np.isfinite(plane) & (np.abs(np.log(np.where(z > 0, plane / np.maximum(z, 1e-6), 1))) < 0.05))
        m[::2] = False  # subsample rows
        pc = np.stack([(u[m] - Kw[0, 2]) / Kw[0, 0] * z[m], (v[m] - Kw[1, 2]) / Kw[1, 1] * z[m], z[m]], -1)
        pw = (pc - Ew[:, 3]) @ Ew[:, :3]  # R^T (pc - t)
        ph = pw @ Eh[:, :3].T + Eh[:, 3]
        uh = ph[:, 0] / ph[:, 2] * Kh[0, 0] + Kh[0, 2]
        vh = ph[:, 1] / ph[:, 2] * Kh[1, 1] + Kh[1, 2]
        ok = (ph[:, 2] > 0.05) & (uh >= 0) & (uh < C.NATIVE_W) & (vh >= 0) & (vh < C.NATIVE_H)
        zh = dh[t][vh[ok].astype(int), uh[ok].astype(int)].astype(np.float64) / 1000
        good = zh > 0
        ratios.append(zh[good] / ph[ok][good, 2])
    if not ratios:
        return {"n": 0}
    r = np.concatenate(ratios)
    hist, edges = np.histogram(np.log(r), bins=200, range=(-0.5, 0.5))
    mode = float(np.exp(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])))
    return {"n": int(r.size), "mode(high/wrist)": round(mode, 3), "within5%": round(float((np.abs(r - 1) < 0.05).mean()), 3)}


def track_check(task: str, ep: int, cam: str) -> dict:
    tr = np.load(C.out_file("tracks", task, ep, cam)).astype(np.float32)
    vis = np.load(C.out_file("tvis", task, ep, cam))
    disp = np.linalg.norm(tr[:, -1] - tr[:, 0], axis=-1)  # [W, N]
    both = vis[:, 0] & vis[:, -1]
    return {"windows": tr.shape[0], "vis": float(vis.mean()), "moving_frac(>3px)": float((disp[both] > 3).mean())}


def overlay(task: str, ep: int, out: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam_npz = np.load(C.out_file("cam", task, ep, ext=".npz"))
    cams = list(C.CAMERAS)
    fig, axes = plt.subplots(len(cams) * 2, 4, figsize=(16, 6 * len(cams)))
    for ci, cam in enumerate(cams):
        rgb = C.load_rgb(task, ep, cam)
        depth = np.load(C.out_file("depth", task, ep, cam), mmap_mode="r")
        tr = np.load(C.out_file("tracks", task, ep, cam)).astype(np.float32)
        vis = np.load(C.out_file("tvis", task, ep, cam))
        starts = np.load(C.out_file("tmeta", task, ep, cam, ext=".npz"))["starts"]
        pick = np.linspace(0, len(starts) - 1, 4).round().astype(int)
        vmax = np.percentile(depth[depth > 0], 99) if (depth > 0).any() else 1
        for j, w in enumerate(pick):
            t = int(starts[w])
            ax = axes[2 * ci, j]
            ax.imshow(rgb[t])
            for n in range(0, tr.shape[2], 2):
                if vis[w, 0, n]:
                    seg = tr[w, :25, n]
                    ax.plot(seg[:, 0], seg[:, 1], "-", lw=0.8, color="yellow", alpha=0.8)
                    ax.plot(*seg[-1], ".", ms=2, color="red")
            for a, col in ((0, "cyan"), (1, "magenta")):
                u, v, _, ok = cam_npz["grip"][t, ci, a]
                if ok:
                    ax.plot(u, v, "x", ms=10, mew=2, color=col)
            ax.set_title(f"{cam} t={t} (tracks to t+24)", fontsize=8)
            ax.axis("off")
            ax = axes[2 * ci + 1, j]
            ax.imshow(np.where(depth[t] > 0, depth[t], np.nan), cmap="turbo", vmin=0, vmax=vmax)
            ax.set_title(f"depth mm, valid {(depth[t] > 0).mean():.2f}, max {vmax:.0f}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{task}_ep{ep}.png", dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="place_shoe")
    ap.add_argument("--episodes", default="0")
    ap.add_argument("--out", default=None, help="write overlay PNGs here")
    args = ap.parse_args()
    for task in args.tasks.split(","):
        for ep in map(int, args.episodes.split(",")):
            d = depth_check(task, ep)
            print(f"{task}/ep{ep} depth(cam_high @gripper): {d}")
            for cam in C.CAMERAS:
                rel = np.load(C.out_file("depth_meta", task, ep, cam, ext=".npz"))["reliable"].mean()
                print(f"{task}/ep{ep} {cam}: depth reliable {rel:.2f}  tracks {track_check(task, ep, cam)}")
            for cam in list(C.CAMERAS)[1:]:
                print(f"{task}/ep{ep} cross-camera {cam} -> cam_high: {cross_check(task, ep, cam)}")
            if args.out:
                overlay(task, ep, pathlib.Path(args.out))


if __name__ == "__main__":
    main()
