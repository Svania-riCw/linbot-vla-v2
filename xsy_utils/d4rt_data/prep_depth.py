#!/usr/bin/env python
"""Metric depth for every frame of every camera with MoGe-3 (``Ruicheng/moge-3-vitl``).

The raw RoboTwin HDF5 carries no depth (unlike pi05-experiments' dataset, which had the
simulator's ``depth_mm``), so depth is estimated. The **GT horizontal FoV** of each camera
(from the HDF5 ``intrinsic_cv``; 48.1 deg for all three) is passed to MoGe, so the focal length
is fixed to the true one. That fixes the *shape*, but MoGe's metric scale on these renders is
still off and unstable: measured scale error 0.43-0.58 per frame on the *static* cam_high and
0.15-0.54 on the wrists (i.e. raw depth up to ~6x too far). So every frame is rescaled against
the GT table / wall planes (``scale_align.py``): pure scale, no shift -- after it the plane
residual is within 1% from near to far, and depth at the projected gripper is 0.86-0.90 of the
endpose z (the endpose point is inside the gripper, so the visible surface is a bit closer).

Per (task, episode, camera), under ``OUT_ROOT/<task>/``:

``depth/<cam>/episode_XXXXXX.npy``       uint16 [T_raw, 240, 320]
    **Scale-corrected** depth along the optical axis in **millimetres** at the native 320x240
    resolution (lerobot images are a 2x resize of these frames). ``0`` = invalid (MoGe mask
    off / non-finite); values clip at 65535 mm. Raw MoGe depth = ``depth / scale[t]``.
``depth_meta/<cam>/episode_XXXXXX.npz``
    ``intrinsic`` float32 [T_raw, 3, 3]  MoGe's K in native pixels (focal = GT by construction)
    ``valid_frac`` float32 [T_raw]       fraction of valid pixels per frame
    ``fov_x_deg`` float32                the GT FoV that was imposed
    ``scale`` float32 [T_raw]            applied per-frame scale (own fit; interpolated if unreliable)
    ``scale_raw`` float32 [T_raw]        per-frame plane fit (nan when no plane visible)
    ``inlier_frac`` float32 [T_raw]      fraction of plane pixels agreeing with the fit
    ``reliable`` bool [T_raw]            frame had its own fit (>= 1500 plane inlier px); else its
                                         scale is interpolated and only approximate -- mask these

Sharding: ``--rank r --world n`` takes every n-th (task, episode); ``run_all.sh`` launches one
rank per GPU. Re-running skips finished files, so it resumes after an interruption.

    python prep_depth.py --bench 2 --tasks place_shoe          # throughput, writes nothing
    python prep_depth.py --tasks place_shoe --cams cam_high    # one task, one camera
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import common as C
import scale_align as S

MODEL = C.CKPT_ROOT / "moge-3-vitl" / "model.pt"


class _Decode(torch.utils.data.Dataset):
    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        import h5py

        task, ep, cam = self.jobs[i]
        g = C.CAMERAS[cam]
        with h5py.File(C.raw_path(task, ep), "r") as f:
            rgb = C.decode_jpegs(f[f"observation/{g}/rgb"][:])
            K, E = f[f"observation/{g}/intrinsic_cv"][:], f[f"observation/{g}/extrinsic_cv"][:]
        fov_x = math.degrees(2 * math.atan(C.NATIVE_W / 2 / float(K[0, 0, 0])))
        # GT plane depth per frame (CPU, here in the worker so the GPU loop stays fed);
        # cam_high is static, so compute once and reuse.
        static = np.allclose(E, E[:1]) and np.allclose(K, K[:1])
        planes = [S.plane_depth(K[0 if static else t], E[0 if static else t], C.NATIVE_H, C.NATIVE_W, task)
                  for t in range(1 if static else len(rgb))]
        planes = np.stack(planes if not static else planes * len(rgb)).astype(np.float32)
        return task, ep, cam, fov_x, torch.from_numpy(rgb), planes


@torch.no_grad()
def infer_episode(model, rgb: torch.Tensor, fov_x: float, args, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> raw MoGe depth float32 [T,H,W] metres (0 = invalid), K [T,3,3] native px, valid frac [T]."""
    depth_out, k_out, vf_out = [], [], []
    for i in range(0, rgb.shape[0], args.batch):
        x = rgb[i : i + args.batch].to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        out = model.infer(
            x,
            num_tokens=args.num_tokens,
            fov_x=fov_x,
            refine_steps=args.refine_steps,
            use_fp16=args.fp16,
        )
        d = out["depth"]
        valid = out["mask"].bool() & torch.isfinite(d) & (d > 0)
        depth_out.append(torch.where(valid, d, torch.zeros_like(d)).float().cpu().numpy())
        k = out["intrinsics"].float().clone()
        k[:, 0] *= C.NATIVE_W  # MoGe K is normalized by image width / height
        k[:, 1] *= C.NATIVE_H
        k_out.append(k.cpu().numpy())
        vf_out.append(valid.float().mean((1, 2)).cpu().numpy())
    return np.concatenate(depth_out), np.concatenate(k_out).astype(np.float32), np.concatenate(vf_out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--cams", default=",".join(C.CAMERAS))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16, help="frames per MoGe call")
    ap.add_argument("--num-tokens", type=int, default=1200, help="MoGe-3 base ViT tokens (range 1200..3600)")
    ap.add_argument("--refine-steps", type=int, default=3, help="MoGe-3 sparse refinement steps")
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="decode threads")
    ap.add_argument("--bench", type=int, default=0, help="time this many episodes, write nothing")
    args = ap.parse_args()

    cams = args.cams.split(",")
    jobs = [(t, e, c) for t, e in C.shard(C.units(args.tasks), args.rank, args.world) for c in cams]
    if args.bench:
        jobs = jobs[: args.bench]
    else:
        todo = [j for j in jobs if not C.out_file("depth", *j).exists()]
        print(f"[prep_depth r{args.rank}] {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to do", flush=True)
        jobs = todo
    if not jobs:
        return

    from moge.model.v3 import MoGeModel

    device = torch.device(args.device)
    model = MoGeModel.from_pretrained(str(MODEL)).to(device).eval()
    loader = C.prefetch(_Decode(jobs).__getitem__, len(jobs), args.workers)
    t0, t_gpu, n_frames = time.time(), 0.0, 0
    for i, (task, ep, cam, fov_x, rgb, planes) in enumerate(loader, 1):
        g0 = time.time()
        raw, k, vf = infer_episode(model, rgb, fov_x, args, device)
        t_gpu += time.time() - g0
        fits = [S.frame_scale(planes[t], raw[t]) for t in range(len(raw))]
        scale_raw = np.array([f[0] for f in fits], np.float32)
        inlier = np.array([f[1] for f in fits], np.float32)
        n_inl = np.array([f[2] for f in fits], np.int32)
        scale, reliable = S.episode_scales(scale_raw, n_inl)
        if not reliable.any():
            print(f"[prep_depth r{args.rank}] WARNING {task}/ep{ep}/{cam}: no plane fit, writing raw MoGe scale", flush=True)
            scale = np.ones_like(scale_raw)
        depth = np.clip(np.round(raw * scale[:, None, None] * 1000.0), 0, 65535).astype(np.uint16)
        n_frames += len(depth)
        if not args.bench:
            C.atomic_save(C.out_file("depth", task, ep, cam), lambda fh: np.save(fh, depth))
            C.atomic_save(
                C.out_file("depth_meta", task, ep, cam, ext=".npz"),
                lambda fh: np.savez(
                    fh,
                    intrinsic=k,
                    valid_frac=vf,
                    fov_x_deg=np.float32(fov_x),
                    scale=scale.astype(np.float32),
                    scale_raw=scale_raw,
                    inlier_frac=inlier,
                    reliable=reliable,
                ),
            )
        if i % 10 == 0 or i == len(jobs) or args.bench:
            el = time.time() - t0
            print(
                f"[prep_depth r{args.rank}] {i}/{len(jobs)} {task}/ep{ep}/{cam}  {el:.0f}s  "
                f"{n_frames / el:.1f} fr/s (gpu {n_frames / t_gpu:.1f})  eta {el / i * (len(jobs) - i) / 3600:.2f}h  "
                f"valid {vf.mean():.3f}  median {np.median(depth[depth > 0]) if (depth > 0).any() else 0:.0f}mm  "
                f"scale {np.median(scale):.3f} reliable {reliable.mean():.2f} inlier {inlier.mean():.2f}",
                flush=True,
            )
    if not args.bench and args.rank == 0:
        (C.OUT_ROOT / "DEPTH_INFO.json").write_text(
            json.dumps(
                {
                    "model": "MoGe-3 Ruicheng/moge-3-vitl (microsoft/MoGe @74fbce0)",
                    "fov": "GT fov_x from HDF5 intrinsic_cv imposed (focal fixed)",
                    "scale": "per-frame scale vs GT table/wall planes (scale_align.py); unreliable frames interpolated, flagged in depth_meta.reliable",
                    "num_tokens": args.num_tokens,
                    "refine_steps": args.refine_steps,
                    "fp16": bool(args.fp16),
                    "depth": "uint16 millimetres, [T_raw, 240, 320] native resolution, 0 = invalid",
                    "depth_meta": "intrinsic [T_raw,3,3] native px, valid_frac [T_raw], fov_x_deg",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
