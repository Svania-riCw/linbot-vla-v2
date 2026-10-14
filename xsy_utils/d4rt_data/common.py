"""Shared helpers for the d4rt auxiliary-data prep on lingbot's RoboTwin clean set.

Source of truth is the **raw RoboTwin HDF5** the lerobot set was converted from, not the lerobot
parquet. Facts the scripts rely on (all verified, see README.md):

* lerobot ``robotwin_clean_<task>`` episode ``i`` <- ``<RAW_ROOT>/<task>/aloha-agilex_clean_50/data/episode{i}.hdf5``
  (the converter sorts naturally; ``prep_ee.py`` re-checks every episode via ``state == vector``).
* ``state[t] = raw[t]``, ``action[t] = raw[t + 1]``, ``T_lerobot = T_raw - 1``.
* Raw RGB is 320x240 JPEG; the lerobot images are RoboTwin ``process_data.py``'s
  ``cv2.resize(.., (640, 480))`` of it, re-encoded. All outputs here are in the **native
  320x240** pixel frame (lerobot pixel = 2x).
* Cameras: cam_high = head_camera, cam_left_wrist = left_camera, cam_right_wrist = right_camera.

Every per-frame array is indexed by **raw frame** (length ``T_raw``): row ``t`` belongs to lerobot
frame ``t``; the last row has no lerobot frame of its own (it is only ever an action target).
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

LEROBOT_ROOT = pathlib.Path("/2025233210/lmx/training-data/lerobot")
RAW_ROOT = pathlib.Path("/2025233210/lmx/RoboTwin/data")
RAW_SETTING = "aloha-agilex_clean_50"
OUT_ROOT = pathlib.Path("/2025233210/xiesy/data/lingbot_robotwin_clean_d4rt")
CKPT_ROOT = pathlib.Path("/2025233210/xiesy/lingbot_ws/checkpoints")

# lerobot camera key -> raw HDF5 camera group
CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
NATIVE_H, NATIVE_W = 240, 320


def tasks() -> list[str]:
    return sorted(p.name.removeprefix("robotwin_clean_") for p in LEROBOT_ROOT.glob("robotwin_clean_*"))


def raw_path(task: str, ep: int) -> pathlib.Path:
    return RAW_ROOT / task / RAW_SETTING / "data" / f"episode{ep}.hdf5"


def episodes(task: str) -> list[int]:
    info = json.loads((LEROBOT_ROOT / f"robotwin_clean_{task}" / "meta" / "info.json").read_text())
    return list(range(info["total_episodes"]))


def units(task_spec: str = "all") -> list[tuple[str, int]]:
    """All (task, episode) pairs; ``task_spec`` is 'all' or a comma list of task names."""
    ts = tasks() if task_spec == "all" else [t.strip() for t in task_spec.split(",") if t.strip()]
    unknown = set(ts) - set(tasks())
    if unknown:
        raise SystemExit(f"unknown tasks: {sorted(unknown)}")
    return [(t, e) for t in ts for e in episodes(t)]


def shard(items: list, rank: int, world: int) -> list:
    """Round-robin, so every shard gets a mix of long and short tasks."""
    return items[rank::world]


def out_file(kind: str, task: str, ep: int, cam: str | None = None, ext: str = ".npy") -> pathlib.Path:
    d = OUT_ROOT / task / kind
    if cam is not None:
        d = d / cam
    return d / f"episode_{ep:06d}{ext}"


def decode_jpegs(blobs) -> np.ndarray:
    """HDF5 JPEG byte strings -> uint8 RGB [T, 240, 320, 3]."""
    import cv2

    out = np.empty((len(blobs), NATIVE_H, NATIVE_W, 3), np.uint8)
    for i, b in enumerate(blobs):
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (NATIVE_H, NATIVE_W, 3), img.shape
        out[i] = img[..., ::-1]
    return out


def load_rgb(task: str, ep: int, cam: str) -> np.ndarray:
    import h5py

    with h5py.File(raw_path(task, ep), "r") as f:
        return decode_jpegs(f[f"observation/{CAMERAS[cam]}/rgb"][:])


def atomic_save(path: pathlib.Path, write) -> None:
    """tmp + rename: a killed run or a concurrent reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            write(fh)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise




def prefetch(load, n: int, workers: int = 4, ahead: int = 2):
    """Yield ``load(0), .., load(n - 1)`` in order, computed ``workers`` threads ahead.

    Threads, not a torch DataLoader: the container's /dev/shm is 4 GB and a whole episode
    (RGB + per-frame plane maps, up to a few hundred MB) per in-flight item on 8 ranks overflowed
    it (worker bus errors). JPEG decode, h5py reads and the numpy plane maths release the GIL.
    """
    from concurrent.futures import ThreadPoolExecutor

    if workers <= 0:
        yield from map(load, range(n))
        return
    with ThreadPoolExecutor(workers) as pool:
        futs = {i: pool.submit(load, i) for i in range(min(workers + ahead, n))}
        for i in range(n):
            nxt = i + workers + ahead
            if nxt < n:
                futs[nxt] = pool.submit(load, nxt)
            yield futs.pop(i).result()
