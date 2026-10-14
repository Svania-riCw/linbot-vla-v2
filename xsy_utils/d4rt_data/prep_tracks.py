#!/usr/bin/env python
"""CoTracker3 point tracks, re-seeded every chunk_size/2 frames (pi05-experiments' scheme).

A ``--window`` = 50 (= lingbot ``chunk_size``) frame clip is tracked from a fresh
``--grid-size``^2 grid every ``--stride`` = 25 (= chunk_size / 2) frames, plus a tail window
starting at ``T - window`` so every frame is covered. For a frame t the consumer takes the window
with the largest start ``s <= t`` (the freshest seeds), which leaves 25..49 frames of horizon.
Re-seeding matters: one long track drags every point the arm sweeps over along for the rest of
the episode, and objects that appear later never get a query.

**Track ids are not stable across windows** (point i of window 3 is only the same grid cell
re-seeded as point i of window 2) -- never concatenate windows.

Per (task, episode, camera), under ``OUT_ROOT/<task>/``:

``tracks/<cam>/episode_XXXXXX.npy``  float16 [W, L, N, 2]  (u, v) in **native 320x240** pixels
``tvis/<cam>/episode_XXXXXX.npy``    bool    [W, L, N]     tracker visibility
``tmeta/<cam>/episode_XXXXXX.npz``   ``starts`` int32 [W], window, stride, grid_size, num_frames

Frames are indexed like every other output here: raw frame t = lerobot frame t.

    python prep_tracks.py --bench 2 --tasks place_shoe
    python prep_tracks.py --tasks place_shoe --cams cam_high
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import common as C

MODEL = C.CKPT_ROOT / "cotracker3" / "scaled_offline.pth"


def window_starts(num_frames: int, window: int, stride: int) -> list[int]:
    if num_frames <= window:
        return [0]
    starts = list(range(0, num_frames - window + 1, stride))
    if starts[-1] + window < num_frames:
        starts.append(num_frames - window)
    return starts


class _Decode(torch.utils.data.Dataset):
    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        task, ep, cam = self.jobs[i]
        return task, ep, cam, torch.from_numpy(C.load_rgb(task, ep, cam))


@torch.no_grad()
def track_episode(model, video: torch.Tensor, starts: list[int], args, device) -> tuple[np.ndarray, np.ndarray]:
    length = min(args.window, video.shape[0])
    clips = torch.stack([video[s : s + length] for s in starts]).permute(0, 1, 4, 2, 3).contiguous()  # [W, L, 3, H, W]
    tr_out, vi_out = [], []
    for i in range(0, len(clips), args.batch):
        x = clips[i : i + args.batch].to(device, non_blocking=True).float()
        tr, vi = model(x, grid_size=args.grid_size)  # [b, L, N, 2] in input pixels
        tr_out.append(tr.to(torch.float16).cpu().numpy())
        vi_out.append(vi.bool().cpu().numpy())
    return np.concatenate(tr_out), np.concatenate(vi_out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--cams", default=",".join(C.CAMERAS))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--window", type=int, default=50, help="= lingbot chunk_size")
    ap.add_argument("--stride", type=int, default=25, help="= chunk_size / 2")
    ap.add_argument("--grid-size", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8, help="windows per CoTracker call")
    ap.add_argument("--workers", type=int, default=4, help="decode processes")
    ap.add_argument("--bench", type=int, default=0, help="time this many episodes, write nothing")
    args = ap.parse_args()

    cams = args.cams.split(",")
    jobs = [(t, e, c) for t, e in C.shard(C.units(args.tasks), args.rank, args.world) for c in cams]
    if args.bench:
        jobs = jobs[: args.bench]
    else:
        todo = [j for j in jobs if not C.out_file("tmeta", *j, ext=".npz").exists()]
        print(f"[prep_tracks r{args.rank}] {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to do", flush=True)
        jobs = todo
    if not jobs:
        return

    from cotracker.predictor import CoTrackerPredictor

    device = torch.device(args.device)
    model = CoTrackerPredictor(checkpoint=str(MODEL), v2=False, offline=True, window_len=60).to(device).eval()
    loader = torch.utils.data.DataLoader(
        _Decode(jobs), batch_size=None, num_workers=args.workers, prefetch_factor=2 if args.workers else None
    )
    t0, n_win = time.time(), 0
    for i, (task, ep, cam, video) in enumerate(loader, 1):
        starts = window_starts(video.shape[0], args.window, args.stride)
        tracks, vis = track_episode(model, video, starts, args, device)
        n_win += len(starts)
        if not args.bench:
            # tmeta last: its existence marks the unit as finished
            C.atomic_save(C.out_file("tracks", task, ep, cam), lambda fh: np.save(fh, tracks))
            C.atomic_save(C.out_file("tvis", task, ep, cam), lambda fh: np.save(fh, vis))
            C.atomic_save(
                C.out_file("tmeta", task, ep, cam, ext=".npz"),
                lambda fh: np.savez(
                    fh,
                    starts=np.asarray(starts, np.int32),
                    window=np.int32(args.window),
                    stride=np.int32(args.stride),
                    grid_size=np.int32(args.grid_size),
                    num_frames=np.int32(video.shape[0]),
                ),
            )
        if i % 10 == 0 or i == len(jobs) or args.bench:
            el = time.time() - t0
            print(
                f"[prep_tracks r{args.rank}] {i}/{len(jobs)} {task}/ep{ep}/{cam}  {el:.0f}s  {n_win / el:.1f} win/s  "
                f"eta {el / i * (len(jobs) - i) / 3600:.2f}h  vis {vis.mean():.2f}",
                flush=True,
            )
    if not args.bench and args.rank == 0:
        (C.OUT_ROOT / "TRACKS_INFO.json").write_text(
            json.dumps(
                {
                    "model": "CoTracker3 offline (facebook/cotracker3 scaled_offline.pth), window_len=60",
                    "window": args.window,
                    "stride": args.stride,
                    "grid_size": args.grid_size,
                    "coords": "native 320x240 pixels, (u, v); lerobot pixel = 2x",
                    "note": "track ids are NOT stable across windows",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
