#!/usr/bin/env python
"""Fit the d4rt-only "random-domain" photometric perturbation (``d4rt_lingbot/perturb.py``).

Source / target frames are the head-camera eval videos of one checkpoint's 7-task eval: clean
(source) vs random (target), first / one-third / two-thirds frame of each of the 25 episodes per
task. Both sides are the eval recorder's mp4 at native 320x240, so compression affects them alike;
frames are squashed to 224x224 exactly as the model's input. The parameter ranges are chosen to
minimize the sum over global image statistics of the 1-D Wasserstein distance (in units of the
random-domain std) between perturbed-clean and random. Fitting runs at 112x112 (blur halved) with
common random numbers; the result is then re-checked at 224 with fresh draws.

As a sanity check, the same statistics are computed on clean *training* frames (lerobot PNG,
640x480 -> 224) — the frames the perturbation is actually applied to.

    python calibrate_perturb.py --out configs/rt_perturb.json
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import random

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize

from d4rt_lingbot import perturb as P

EVAL = "/2025233210/xiesy/lingbot_ws/linbot-vla-v2/external/RoboTwin/eval_result/baseline_gbs128_10k_global_step_10000"
LEROBOT = "/2025233210/lmx/training-data/lerobot"
STATS = ["mean_r", "mean_g", "mean_b", "lum_std", "mean_i", "mean_q", "chroma", "hsv_sat", "std_i", "std_q", "sharp",
         "dark", "bright"]


def video_frames(path: str, fracs=(0.0, 1 / 3, 2 / 3)) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in fracs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f * max(n - 1, 0)))
        ok, fr = cap.read()
        if ok:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    return out


def to_tensor(frames: list[np.ndarray], size: int) -> torch.Tensor:
    """list of HWC uint8 -> [n, 1, 3, size, size] float in [0,1] (anisotropic squash, as the model)."""
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return x[:, None]


def eval_frames(split: str, size: int) -> torch.Tensor:
    frames = []
    for d in sorted(glob.glob(f"{EVAL}/run_*/*_{split}")):
        for v in sorted(glob.glob(f"{d}/episode*.mp4")):
            frames += video_frames(v)
    return to_tensor(frames, size)


def train_frames(size: int, n_tasks: int = 50, per_task: int = 6, seed: int = 0) -> torch.Tensor:
    rng = random.Random(seed)
    frames = []
    for d in sorted(glob.glob(f"{LEROBOT}/robotwin_clean_*"))[:n_tasks]:
        files = sorted(glob.glob(f"{d}/data/chunk-*/file-*.parquet"))
        pf = pq.ParquetFile(rng.choice(files))
        tab = pf.read_row_group(0, columns=["observation.images.cam_high"])
        col = tab.column(0).to_pylist()
        for i in rng.sample(range(len(col)), min(per_task, len(col))):
            frames.append(np.asarray(Image.open(io.BytesIO(col[i]["bytes"])).convert("RGB")))
    return to_tensor(frames, size)


_YIQ = P._YIQ


def stats(x: torch.Tensor) -> torch.Tensor:
    """x [n, 1, 3, H, W] -> [n, len(STATS)]"""
    x = x[:, 0]
    yiq = torch.einsum("ij,njhw->nihw", _YIQ, x)
    y, i, q = yiq[:, 0], yiq[:, 1], yiq[:, 2]
    mx, mn = x.max(dim=1).values, x.min(dim=1).values
    sat = ((mx - mn) / mx.clamp(min=1e-4)).mean(dim=(1, 2))
    lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]
    sharp = torch.log(F.conv2d(y[:, None], lap).var(dim=(1, 2, 3)) + 1e-6)
    return torch.stack([
        x[:, 0].mean(dim=(1, 2)), x[:, 1].mean(dim=(1, 2)), x[:, 2].mean(dim=(1, 2)),
        y.std(dim=(1, 2)), i.mean(dim=(1, 2)), q.mean(dim=(1, 2)), torch.sqrt(i ** 2 + q ** 2).mean(dim=(1, 2)),
        sat, i.std(dim=(1, 2)), q.std(dim=(1, 2)), sharp,
        (y < 0.03).float().mean(dim=(1, 2)), (y > 0.97).float().mean(dim=(1, 2)),  # crushed / blown share
    ], dim=1)


def w1(a: torch.Tensor, b: torch.Tensor, nq: int = 101) -> torch.Tensor:
    qs = torch.linspace(0, 1, nq)
    return (torch.quantile(a, qs, dim=0) - torch.quantile(b, qs, dim=0)).abs().mean(dim=0)


# ---- parameter vector <-> ranges ---------------------------------------------------------
# box per bound, so the fit cannot reach degenerate transforms (e.g. x0.2 brightness then x5 contrast
# about the mean, which crushes dark objects to black). log-space for "lu".
BOUNDS = {
    "gain_r": (-0.4, 0.4), "gain_g": (-0.4, 0.4), "gain_b": (-0.4, 0.4), "brightness": (-0.7, 0.7),
    "contrast": (0.4, 2.0), "saturation": (0.0, 2.5), "hue": (0.0, math.pi), "gamma": (-0.7, 0.7), "blur": (0.0, 2.0),
}


def bounds() -> list[tuple[float, float]]:
    return [BOUNDS[n] for n, kind in P.SPEC.items() for _ in range(1 if kind in ("sym", "zero") else 2)]


def unpack(theta: np.ndarray) -> dict:
    out, j = {}, 0
    for name, kind in P.SPEC.items():
        nb = 1 if kind in ("sym", "zero") else 2
        out[name] = [float(t) for t in theta[j : j + nb]]
        j += nb
    return out


def pack(ranges: dict) -> np.ndarray:
    return np.concatenate([np.asarray(ranges[n], dtype=np.float64) for n in P.SPEC])


INIT = {
    "gain_r": [-0.1, 0.1], "gain_g": [-0.1, 0.1], "gain_b": [-0.1, 0.1], "brightness": [-0.2, 0.2],
    "contrast": [0.8, 1.2], "saturation": [0.8, 1.2], "hue": [0.1], "gamma": [-0.1, 0.1], "blur": [0.5],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/rt_perturb.json"))
    ap.add_argument("--reps", type=int, default=2, help="perturbation draws per clean frame")
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(32)
    torch.manual_seed(0)

    clean224, rand224 = eval_frames("clean", 224), eval_frames("random", 224)
    print(f"eval frames: clean {len(clean224)}, random {len(rand224)}", flush=True)
    train224 = train_frames(224)
    print(f"train frames: {len(train224)}", flush=True)
    s_rand = stats(rand224)
    scale = s_rand.std(dim=0).clamp(min=1e-6)

    def report(name, s):
        d = w1(s, s_rand) / scale
        return f"{name:16s} W1/std total {d.sum():6.3f} | " + " ".join(f"{k} {v:.2f}" for k, v in zip(STATS, d.tolist()))

    lines = [report("clean(eval)", stats(clean224)), report("clean(train)", stats(train224)),
             report("random(eval)", s_rand)]
    for l in lines:
        print(l, flush=True)

    # fit at 112 (blur in 224-px units: halve it at 112)
    clean112 = F.interpolate(clean224[:, 0], size=(112, 112), mode="bilinear", antialias=True)[:, None]
    rand112 = F.interpolate(rand224[:, 0], size=(112, 112), mode="bilinear", antialias=True)[:, None]
    s_rand112 = stats(rand112)
    scale112 = s_rand112.std(dim=0).clamp(min=1e-6)
    src = clean112.repeat(args.reps, 1, 1, 1, 1)
    u = torch.rand(len(src), len(P.SPEC), generator=torch.Generator().manual_seed(1))

    def objective(theta):
        r = unpack(theta)
        p = P.sample(r, u=u)
        p["blur"] = p["blur"] / 2
        with torch.no_grad():
            s = stats(P.apply_batch(src, p))
        return float((w1(s, s_rand112) / scale112).sum())

    best = None
    starts = [pack(INIT), pack(INIT) * 2.0, pack(INIT) * 0.5 + pack({**INIT, "contrast": [1, 1], "saturation": [1, 1]}) * 0.5]
    for restart, theta0 in enumerate(starts[: args.restarts]):
        res = minimize(objective, theta0, method="Powell", bounds=bounds(), options={"maxiter": args.maxiter, "xtol": 1e-3, "ftol": 1e-4})
        print(f"restart {restart}: objective {res.fun:.4f} ({res.nfev} evals)", flush=True)
        if best is None or res.fun < best.fun:
            best = res
    ranges = unpack(best.x)

    # re-check at 224 with fresh draws
    with torch.no_grad():
        p = P.sample(ranges, u=torch.rand(len(clean224) * 2, len(P.SPEC), generator=torch.Generator().manual_seed(2)))
        pert224 = P.apply_batch(clean224.repeat(2, 1, 1, 1, 1), p)
        pert_train = P.apply_batch(train224, P.sample(ranges, u=torch.rand(len(train224), len(P.SPEC), generator=torch.Generator().manual_seed(3))))
    lines += [report("perturbed(eval)", stats(pert224)), report("perturbed(train)", stats(pert_train))]
    print(lines[-2] + "\n" + lines[-1])
    print("ranges:", P.describe(ranges))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "ranges": ranges,
            "describe": P.describe(ranges),
            "fit_objective_112": float(best.fun),
            "report_224": lines,
            "source": EVAL,
            "stats": STATS,
        }, f, indent=1)
    # visual check: rows = clean eval, perturbed clean eval, random eval; 8 columns
    idx = torch.linspace(0, len(clean224) - 1, 8).long()
    rows = [clean224[idx, 0], pert224[idx, 0], rand224[torch.linspace(0, len(rand224) - 1, 8).long(), 0]]
    grid = torch.cat([torch.cat(list(r), dim=2) for r in rows], dim=1)
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(args.out.replace(".json", ".png"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
