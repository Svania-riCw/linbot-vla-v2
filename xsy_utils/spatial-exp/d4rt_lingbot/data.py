"""d4rt query/target construction for lingbot's RoboTwin clean set.

Port of pi05-experiments' ``D4rtQuerySample`` (src/openpi/training/depth_probe_data.py), the
data side of ``xsy_utils/d4rt_full_optimize/train/train_q8_10000.sh``: K=8 queries per frame,
four tasks a quarter each, xyz of a 14x14 patch in a named target frame, normalized by 1.2 m.

    task           query                             target frame F      anchor (where read)
    depth          (u,v) @ c_src, dt=0               c_src @ t           (c_src, t,    (u,v))
    time           (u,v) @ c_src, dt>0               c_src @ t           (c_src, t+dt, track(u,v))
    hand-eye       (u,v) @ cam_high, dt=0            gripper_{L,R} @ t   (cam_high, t, (u,v))
    registration   (u,v) @ c_src, dt=0               c_tgt != c_src @ t  (c_src, t,    (u,v))

What differs from pi05, and why:

* Canvas. lingbot feeds ``Resize((224, 224))`` of the 640x480 lerobot frame, which is itself a
  2x resize of the native 320x240 render -- an anisotropic squash, no padding. Query pixels live
  in that 224x224 canvas (the image the model sees); all geometry is done in native continuous
  coordinates, with pixel index i <-> continuous coordinate i + 0.5 in both grids
  (``xsy_utils/d4rt_data`` uses the same convention).
* Source data. ``xsy_utils/d4rt_data`` outputs (native 320x240, uint16 mm depth, per-frame
  scale-corrected MoGe-3; CoTracker3 window 50 / stride 25; endpose in pi05's ee layout).
  The canvas depth and the importance map (pi05's ``w56``: smoothed depth gradient, 4x4
  pooled, per-frame p99-scaled) are computed on the fly instead of being stored.
* Reliability. A query whose anchor (camera, frame) has ``depth_meta.reliable == False`` (the
  plane fit failed and the scale was interpolated) is masked out.
* Input resolution / observation augmentation (``hooks.dataset_fields``). The model's view is
  ``data.img_size`` (256 / 384), not 224; it is resized to the 224 canvas here, so the d4rt task
  is the same at every input resolution. The full432 augmentation's 2-D scale/translation is one
  axis-aligned affine shared by the three views; queries are sampled where the model's (augmented)
  image shows the scene, ``d4rt_coords`` / ``d4rt_patches`` are taken from the augmented canvas,
  and each target cell is the scene point under the corresponding augmented pixel.
"""

from __future__ import annotations

import json
import operator
import os
import pathlib
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

D4RT_ROOT = pathlib.Path("/2025233210/xiesy/data/lingbot_robotwin_clean_d4rt")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")  # = lingbot camera_top / wrist_left / wrist_right
CAM_HIGH = 0
N_CAMS = 3
N_TGT_FRAMES = N_CAMS + 2  # 0/1/2 cameras, 3/4 left/right gripper
T_DEPTH, T_TIME, T_HANDEYE, T_REG = range(4)
TASK_NAMES = ("depth", "time", "handeye", "reg")

NATIVE_W, NATIVE_H = 320, 240
CANVAS = 224
POOL = 4
WRES = CANVAS // POOL  # 56
SMOOTH_K = 5
SX = NATIVE_W / CANVAS  # native px per canvas px, horizontally
SY = NATIVE_H / CANVAS

# canvas pixel index -> native pixel index (nearest: the native pixel containing the canvas
# pixel's centre). Nearest, never averaged: averaging depth across a silhouette invents surfaces.
_COL2U = np.minimum(((np.arange(CANVAS) + 0.5) * SX).astype(np.int64), NATIVE_W - 1)
_ROW2V = np.minimum(((np.arange(CANVAS) + 0.5) * SY).astype(np.int64), NATIVE_H - 1)


def canvas_to_native(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canvas pixel indices -> native continuous (v, u) of the pixel centres."""
    return (rows + 0.5) * SY, (cols + 0.5) * SX


def native_to_canvas(v: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Native continuous (v, u) -> canvas continuous (row, col) pixel-index coordinates."""
    return v / SY - 0.5, u / SX - 0.5


class CanvasAffine:
    """The observation augmentation's 2-D scale/translation in canvas continuous coordinates.

    ``lingbotvla/utils/observation_augmentation.py`` warps pixel indices with
    ``x' = s x + (1 - s)(W - 1) / 2 + t W``; with continuous ``X = x + 0.5`` that is
    ``X' = s X + (1 - s) W / 2 + t W`` -- resolution independent once normalized by W, so the
    same (s, tx, ty) applies to the 224 canvas. ``None`` / no affine -> identity.
    """

    def __init__(self, scale: float = 1.0, tx: float = 0.0, ty: float = 0.0):
        self.s = float(scale)
        self.off_c = (1.0 - self.s) * CANVAS / 2 + float(tx) * CANVAS
        self.off_r = (1.0 - self.s) * CANVAS / 2 + float(ty) * CANVAS
        self.identity = self.s == 1.0 and self.off_c == 0.0 and self.off_r == 0.0

    def fwd_idx(self, rows, cols):
        """Source canvas pixel indices -> augmented canvas pixel indices (nearest)."""
        if self.identity:
            return np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        r = np.floor(self.s * (np.asarray(rows, np.float64) + 0.5) + self.off_r).astype(np.int64)
        c = np.floor(self.s * (np.asarray(cols, np.float64) + 0.5) + self.off_c).astype(np.int64)
        return r, c

    def inv_cont(self, rows_c, cols_c):
        """Augmented continuous (row, col) -> source canvas continuous (row, col)."""
        if self.identity:
            return rows_c, cols_c
        return (rows_c - self.off_r) / self.s, (cols_c - self.off_c) / self.s


IDENTITY = CanvasAffine()


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def cam_to_world(p: np.ndarray, e: np.ndarray) -> np.ndarray:
    """``e``: OpenCV world->camera [R|t] (3, 4)."""
    return (p - e[:, 3]) @ e[:, :3]


def world_to_cam(p: np.ndarray, e: np.ndarray) -> np.ndarray:
    return p @ e[:, :3].T + e[:, 3]


def world_to_gripper(p: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """``rot`` is gripper->world."""
    return (p - pos) @ rot


def valid_cell_mask(patch: int) -> np.ndarray:
    """bool[56, 56]: cells whose whole 4x4 jitter span can host an in-bounds p x p patch."""
    half = patch // 2
    lo, hi = half, CANVAS - (patch - half)
    start = np.arange(WRES) * POOL
    ok = (start >= lo) & (start + POOL - 1 < hi)
    return ok[:, None] & ok[None, :]


def in_patch_bounds(rows: np.ndarray, cols: np.ndarray, patch: int) -> np.ndarray:
    half = patch // 2
    lo, hi = half, CANVAS - (patch - half)
    return (rows >= lo) & (rows < hi) & (cols >= lo) & (cols < hi)


def canvas_depth(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 [240, 320] mm -> float32 [224, 224] metres (0 = invalid)."""
    return depth_mm[_ROW2V[:, None], _COL2U[None, :]].astype(np.float32) / 1000.0


def weight_map(z: np.ndarray, depth_max: float) -> np.ndarray:
    """[224, 224] metres -> [56, 56] importance in [0, 1] (pi05 prep_depth3d_3cam._weight_map)."""
    import cv2

    zn = np.clip(z / depth_max, 0.0, 1.0)
    gy = np.zeros_like(zn)
    gx = np.zeros_like(zn)
    gy[1:-1] = (zn[2:] - zn[:-2]) * 0.5
    gx[:, 1:-1] = (zn[:, 2:] - zn[:, :-2]) * 0.5
    g = np.sqrt(gy * gy + gx * gx)
    g = cv2.blur(g, (SMOOTH_K, SMOOTH_K), borderType=cv2.BORDER_CONSTANT)
    w = g.reshape(WRES, POOL, WRES, POOL).mean(axis=(1, 3))
    hi = max(float(np.quantile(w, 0.99)), 1e-6)
    return np.clip(w / hi, 0.0, 1.0)


def sample_cells(weight: np.ndarray, rng: np.random.Generator, num: int, alpha: float, gamma: float,
                 cell_ok: np.ndarray) -> np.ndarray:
    """pi05 ``sample_cells``: p = alpha * uniform + (1 - alpha) * norm(w ** gamma). -> int32 [num, 2] (row, col)."""
    flat_ok = cell_ok.reshape(-1)
    n_ok = int(flat_ok.sum())
    w = weight.reshape(-1)[flat_ok].astype(np.float64)
    if gamma != 1.0:
        w = np.power(w, gamma)
    total = w.sum()
    prob = w / total if total > 0 else np.full(n_ok, 1.0 / n_ok)
    prob = alpha / n_ok + (1.0 - alpha) * prob
    prob /= prob.sum()
    picked = rng.choice(n_ok, size=num, replace=num > n_ok, p=prob)
    cell_idx = np.flatnonzero(flat_ok)[picked]
    cell_r, cell_c = np.divmod(cell_idx, WRES)
    r = cell_r * POOL + rng.integers(0, POOL, size=num)
    c = cell_c * POOL + rng.integers(0, POOL, size=num)
    return np.stack([r, c], axis=-1).astype(np.int32)


_IO_POOL: tuple[int, ThreadPoolExecutor] | None = None


def _io_pool() -> ThreadPoolExecutor:
    """Per-process thread pool for file reads (dataloader workers are forked: a parent's pool has no threads)."""
    global _IO_POOL
    if _IO_POOL is None or _IO_POOL[0] != os.getpid():
        _IO_POOL = (os.getpid(), ThreadPoolExecutor(max_workers=16, thread_name_prefix="d4rt-io"))
    return _IO_POOL[1]


class _Rows:
    """Leading-axis rows of a C-order .npy file, one ``pread`` per row. On network mounts a memmap
    costs one round trip per 4 KB page it touches (~2.5 ms each on the ossfs /mnt/oss); a row is one."""

    def __init__(self, path: pathlib.Path):
        self._f = open(path, "rb")
        version = np.lib.format.read_magic(self._f)
        read_header = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        if version not in read_header:
            raise ValueError(f"{path}: unsupported .npy version {version}")
        shape, fortran_order, dtype = read_header[version](self._f)
        if fortran_order or dtype.hasobject or not shape:
            raise ValueError(f"{path}: need a C-order numeric array, got shape {shape} fortran={fortran_order}")
        self.shape, self.dtype = shape, dtype
        self._offset = self._f.tell()
        self._row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i) -> np.ndarray:
        i = operator.index(i)
        if i < 0:
            i += self.shape[0]
        if not 0 <= i < self.shape[0]:
            raise IndexError(f"row {i} out of range for {self.shape[0]}")
        buf = bytearray(self._row_bytes)
        view, got, base = memoryview(buf), 0, self._offset + i * self._row_bytes
        while got < len(buf):
            n = os.preadv(self._f.fileno(), [view[got:]], base + got)
            if n == 0:
                raise EOFError(f"{self._f.name}: truncated at row {i}")
            got += n
        return np.frombuffer(buf, self.dtype).reshape(self.shape[1:])


def _load_npz(path: pathlib.Path, keys: dict[str, type]) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k], dtype=t) for k, t in keys.items()}


class _Store:
    """Per-worker LRU of one (task, episode)'s d4rt arrays (row readers where large). The ~20 files of an
    episode are opened concurrently: on ossfs each open + small read is a round trip."""

    def __init__(self, root: pathlib.Path, max_open: int = 32):
        self.root = root
        self.max_open = max_open
        self._cache: OrderedDict[tuple[str, int], dict | None] = OrderedDict()

    def get(self, task: str, ep: int) -> dict | None:
        key = (task, ep)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        entry = self._load(task, ep)
        self._cache[key] = entry
        while len(self._cache) > self.max_open:
            self._cache.popitem(last=False)
        return entry

    def _load(self, task: str, ep: int) -> dict | None:
        d = self.root / task
        name = f"episode_{ep:06d}"
        paths = {
            "ee": d / "ee" / f"{name}.npy",
            "cam": d / "cam" / f"{name}.npz",
            **{f"depth/{c}": d / "depth" / c / f"{name}.npy" for c in CAMERAS},
            **{f"meta/{c}": d / "depth_meta" / c / f"{name}.npz" for c in CAMERAS},
            **{f"tracks/{c}": d / "tracks" / c / f"{name}.npy" for c in CAMERAS},
            **{f"tvis/{c}": d / "tvis" / c / f"{name}.npy" for c in CAMERAS},
            **{f"tmeta/{c}": d / "tmeta" / c / f"{name}.npz" for c in CAMERAS},
        }
        jobs = {
            "ee": (np.load, paths["ee"]),
            # intrinsic [3, T, 3, 3]; extrinsic [3, T, 3, 4] world->cam; grip [T, 3 cams, 2 arms, (u, v, z, valid)]
            "cam": (lambda p: _load_npz(p, {"intrinsic": np.float64, "extrinsic": np.float64, "grip": np.float64}),
                    paths["cam"]),
            **{f"meta/{c}": (lambda p: _load_npz(p, {"reliable": bool})["reliable"], paths[f"meta/{c}"]) for c in CAMERAS},
            **{f"tmeta/{c}": (lambda p: _load_npz(p, {"starts": np.int64})["starts"], paths[f"tmeta/{c}"]) for c in CAMERAS},
            **{f"{k}/{c}": (_Rows, paths[f"{k}/{c}"]) for k in ("depth", "tracks", "tvis") for c in CAMERAS},
        }
        futures = {k: _io_pool().submit(fn, p) for k, (fn, p) in jobs.items()}
        try:
            got = {k: f.result() for k, f in futures.items()}
        except FileNotFoundError:
            return None
        cam = got["cam"]
        return {
            "ee": got["ee"],  # [T, 16]
            "intr": cam["intrinsic"],
            "extr": cam["extrinsic"],
            "grip": cam["grip"],
            "depth": [got[f"depth/{c}"] for c in CAMERAS],  # u16 [T, 240, 320]
            "reliable": [got[f"meta/{c}"] for c in CAMERAS],
            "tracks": [got[f"tracks/{c}"] for c in CAMERAS],  # f16 [W, L, N, 2] (u, v)
            "tvis": [got[f"tvis/{c}"] for c in CAMERAS],  # bool [W, L, N]
            "starts": [got[f"tmeta/{c}"] for c in CAMERAS],
        }


class D4rtSampler:
    """(task, episode, frame, augmented 224 images) -> d4rt query/target fields (numpy)."""

    def __init__(
        self,
        root: str | pathlib.Path = D4RT_ROOT,
        num_queries: int = 8,
        patch_size: int = 14,
        task_mix: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        xyz_scale: float = 1.2,
        max_dt: int = 49,
        alpha: float = 0.1,
        gamma: float = 2.0,
        depth_max: float = 1.2,
        force_gripper: bool = True,
        gripper_fallback: bool = True,
        seed: int = 0,
    ):
        self.root = pathlib.Path(root)
        self.k = num_queries
        self.p = patch_size
        self.task_mix = np.asarray(task_mix, dtype=np.float64)
        if self.task_mix.shape != (4,) or (self.task_mix < 0).any() or self.task_mix.sum() <= 0:
            raise ValueError(f"task_mix must be 4 non-negative weights, got {task_mix}")
        self.xyz_scale = xyz_scale
        self.max_dt = max_dt
        self.alpha = alpha
        self.gamma = gamma
        self.depth_max = depth_max
        self.force_gripper = force_gripper
        self.gripper_fallback = gripper_fallback
        self.seed = seed
        self.cell_ok = valid_cell_mask(patch_size)
        self._store = _Store(self.root)
        self._rows_memo: dict[tuple[int, int], np.ndarray] = {}  # (id(_Rows), row) -> row, one sample's reads
        tasks = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self._task_id = {t: i for i, t in enumerate(tasks)}

    # ------------------------------------------------------------------------------------
    def empty(self) -> dict:
        k, p = self.k, self.p
        return {
            "d4rt_coords": np.zeros((k, 2), np.int64),
            "d4rt_cam": np.zeros((k,), np.int64),
            "d4rt_tgt": np.zeros((k,), np.int64),
            "d4rt_dt": np.zeros((k,), np.int64),
            "d4rt_task": np.zeros((k,), np.int64),
            "d4rt_target": np.zeros((k, p, p, 3), np.float32),
            "d4rt_mask": np.zeros((k,), bool),
            "d4rt_patches": np.zeros((k, 3, p, p), np.float32),
        }

    def _assign_tasks(self, rng: np.random.Generator) -> np.ndarray:
        exact = self.task_mix / self.task_mix.sum() * self.k
        counts = np.floor(exact).astype(np.int64)
        short = self.k - int(counts.sum())
        if short:
            counts[np.argsort(-(exact - counts))[:short]] += 1
        task = np.repeat(np.arange(4, dtype=np.int64), counts)
        rng.shuffle(task)
        return task

    def _grip_canvas(self, e: dict, frame: int, arm: int) -> tuple[int, int] | None:
        """cam_high gripper (row, col) in the canvas, or None if out of view / patch bounds."""
        if frame >= len(e["grip"]):
            return None
        u, v, _z, ok = e["grip"][frame, CAM_HIGH, arm]
        if ok <= 0:
            return None
        r, c = native_to_canvas(np.float64(v), np.float64(u))
        r, c = int(np.floor(r + 0.5)), int(np.floor(c + 0.5))
        return (r, c) if in_patch_bounds(np.int64(r), np.int64(c), self.p) else None

    def _track_slots(self, e, frame, rng, cam, slots, coords, anchor, dt, mask, weights, aff=IDENTITY) -> None:
        """pi05 ``_track_slots``: tracked (source, anchor) pairs inside camera ``cam``."""
        tracks, tvis, starts = e["tracks"][cam], e["tvis"][cam], e["starts"][cam]
        length = tracks.shape[1]
        cand = np.flatnonzero((starts <= frame) & (frame < starts + length))
        if len(cand) == 0:
            return
        w = int(cand[-1])
        i = frame - int(starts[w])
        n_frames = e["depth"][cam].shape[0]
        dt_max = min(self.max_dt, length - 1 - i, n_frames - 1 - frame)
        if dt_max < 1:
            return
        # [L, N, 2] native (u, v); CoTracker puts pixel i's centre at i, here it is at i + 0.5
        uv = self._row(tracks, w).astype(np.float64) + 0.5
        rows, cols = native_to_canvas(uv[..., 1], uv[..., 0])
        rows = np.floor(rows + 0.5).astype(np.int64)
        cols = np.floor(cols + 0.5).astype(np.int64)
        usable = self._row(tvis, w) & in_patch_bounds(rows, cols, self.p)
        horizon = usable[i + 1 : i + dt_max + 1]
        q_ok = in_patch_bounds(*aff.fwd_idx(rows[i], cols[i]), self.p)  # query visible after augmentation
        pts = np.flatnonzero(usable[i] & q_ok & horizon.any(axis=0))
        if len(pts) == 0:
            return
        q_r, q_c = rows[i, pts], cols[i, pts]
        wt = np.power(weights[cam][q_r // POOL, q_c // POOL], self.gamma)
        prob = wt / wt.sum() if wt.sum() > 0 else np.full(len(pts), 1.0 / len(pts))
        prob = self.alpha / len(pts) + (1.0 - self.alpha) * prob
        prob /= prob.sum()
        pick = rng.choice(len(pts), size=len(slots), replace=len(slots) > len(pts), p=prob)
        for j, sel in zip(slots, pick):
            n = pts[sel]
            choices = np.flatnonzero(horizon[:, n]) + 1
            if len(choices) == 0:
                continue
            d = int(rng.choice(choices))
            coords[j] = (q_r[sel], q_c[sel])
            anchor[j] = (rows[i + d, n], cols[i + d, n])
            dt[j] = d
            mask[j] = True

    def _aug_ok(self, rc, aff) -> bool:
        return rc is not None and bool(in_patch_bounds(*aff.fwd_idx(rc[0], rc[1]), self.p))

    def _gripper_time_slot(self, e, frame, rng, arm, coords, anchor, dt, mask, weights, aff=IDENTITY) -> None:
        """Gripper slot's trajectory from the exact endpose projection; tracker fallback when out of view."""
        n_frames = e["depth"][CAM_HIGH].shape[0]
        src = self._grip_canvas(e, frame, arm)
        if self._aug_ok(src, aff):
            hi = min(self.max_dt, len(e["grip"]) - 1 - frame, n_frames - 1 - frame)
            choices = [d for d in range(1, hi + 1) if self._grip_canvas(e, frame + d, arm) is not None]
            if choices:
                d = int(rng.choice(choices))
                coords[arm] = src
                anchor[arm] = self._grip_canvas(e, frame + d, arm)
                dt[arm] = d
                mask[arm] = True
                return
        if self.gripper_fallback:
            self._track_slots(e, frame, rng, CAM_HIGH, np.array([arm]), coords, anchor, dt, mask, weights, aff)

    def patch_xyz(self, e: dict, cam: int, frame: int, row: int, col: int) -> np.ndarray | None:
        """p x p canvas patch centred at (row, col) -> [p, p, 3] metres in that camera's frame, or None if any cell is invalid."""
        half = self.p // 2
        rr = np.arange(row - half, row - half + self.p) + 0.5
        cc = np.arange(col - half, col - half + self.p) + 0.5
        return self.grid_xyz(e, cam, frame, rr, cc)

    def grid_xyz(self, e: dict, cam: int, frame: int, rows_c: np.ndarray, cols_c: np.ndarray) -> np.ndarray | None:
        """Separable grid of canvas continuous coordinates ([p] rows x [p] cols) -> [p, p, 3] metres in
        that camera's frame (depth of the native pixel containing each point), or None if any cell is
        off the image or has no depth."""
        v1, u1 = rows_c * SY, cols_c * SX  # native continuous
        if v1.min() < 0 or v1.max() >= NATIVE_H or u1.min() < 0 or u1.max() >= NATIVE_W:
            return None
        iv = np.minimum(np.floor(v1).astype(np.int64), NATIVE_H - 1)
        iu = np.minimum(np.floor(u1).astype(np.int64), NATIVE_W - 1)
        z = self._row(e["depth"][cam], frame)[iv[:, None], iu[None, :]].astype(np.float64) / 1000.0
        if (z <= 0).any():
            return None
        v, u = v1[:, None], u1[None, :]
        k = e["intr"][cam, frame]
        x = (u - k[0, 2]) / k[0, 0] * z
        y = (v - k[1, 2]) / k[1, 1] * z
        return np.stack([x, y, z], axis=-1)

    def _row(self, rows: _Rows, i: int) -> np.ndarray:
        key = (id(rows), int(i))
        if key not in self._rows_memo:
            self._rows_memo[key] = rows[i]
        return self._rows_memo[key]

    def _prefetch(self, reqs) -> None:
        """Read [(rows, i)] concurrently into the per-sample memo (latency-bound on network mounts)."""
        todo = [(r, int(i)) for r, i in dict.fromkeys((r, int(i)) for r, i in reqs) if (id(r), int(i)) not in self._rows_memo]
        for (r, i), row in zip(todo, _io_pool().map(lambda q: q[0][q[1]], todo)):
            self._rows_memo[(id(r), i)] = row

    def __call__(self, task: str, episode: int, frame: int, images: np.ndarray | None,
                 affine: CanvasAffine | None = None) -> dict:
        """``images``: the model's views after augmentation, resized to the canvas, [3 cams, 3, 224, 224],
        float in [0, 1] (or uint8). ``affine``: the augmentation's 2-D scale/translation (None = none).
        Returns numpy fields; all-masked if the episode has no data."""
        self._rows_memo = {}
        try:
            return self._sample(task, episode, frame, images, affine)
        finally:
            self._rows_memo = {}

    def _sample(self, task, episode, frame, images, affine) -> dict:
        aff = affine or IDENTITY
        e = self._store.get(task, episode)
        if e is None or images is None:
            return self.empty()
        n_frames = e["depth"][0].shape[0]
        frame = min(frame, n_frames - 1)
        # this frame's depth in every camera + every camera's track window around it, in one round trip
        reqs = [(e["depth"][c], frame) for c in range(N_CAMS)]
        for c in range(N_CAMS):
            starts, length = e["starts"][c], e["tracks"][c].shape[1]
            cand = np.flatnonzero((starts <= frame) & (frame < starts + length))
            if len(cand):
                reqs += [(e["tracks"][c], int(cand[-1])), (e["tvis"][c], int(cand[-1]))]
        self._prefetch(reqs)
        rng = np.random.default_rng((self.seed, self._task_id.get(task, 0), episode, frame))
        k, p = self.k, self.p
        task_vec = self._assign_tasks(rng)

        weights = [weight_map(canvas_depth(self._row(e["depth"][c], frame)), self.depth_max) for c in range(N_CAMS)]

        # source camera: hand-eye and the gripper-pinned slots source from cam_high
        src = rng.integers(0, N_CAMS, size=k).astype(np.int64)
        src[task_vec == T_HANDEYE] = CAM_HIGH
        n_grip = min(2, k) if self.force_gripper else 0
        src[:n_grip] = CAM_HIGH

        cell_ok = self.cell_ok
        if not aff.identity:  # every pixel of the cell's 4x4 jitter span must land inside the patch bounds
            ar, ac = aff.fwd_idx(np.arange(CANVAS), np.arange(CANVAS))
            half = p // 2
            r_ok = ((ar >= half) & (ar < CANVAS - (p - half))).reshape(WRES, POOL).all(axis=1)
            c_ok = ((ac >= half) & (ac < CANVAS - (p - half))).reshape(WRES, POOL).all(axis=1)
            cell_ok = cell_ok & r_ok[:, None] & c_ok[None, :]
            if not cell_ok.any():
                return self.empty()
        coords = np.zeros((k, 2), np.int64)
        for c in range(N_CAMS):
            idx = np.flatnonzero(src == c)
            if len(idx):
                coords[idx] = sample_cells(weights[c], rng, len(idx), self.alpha, self.gamma, cell_ok)
        for arm in range(n_grip):
            g = self._grip_canvas(e, frame, arm)
            if self._aug_ok(g, aff):
                coords[arm] = g

        anchor = coords.copy()
        dt = np.zeros((k,), np.int64)
        mask = np.ones((k,), bool)
        time_slots = np.flatnonzero(task_vec == T_TIME)
        if len(time_slots):
            mask[time_slots] = False
            tracked = time_slots[time_slots >= n_grip]
            for c in range(N_CAMS):
                idx = tracked[src[tracked] == c]
                if len(idx):
                    self._track_slots(e, frame, rng, c, idx, coords, anchor, dt, mask, weights, aff)
            for j in time_slots[time_slots < n_grip]:
                self._gripper_time_slot(e, frame, rng, int(j), coords, anchor, dt, mask, weights, aff)

        tgt = src.copy()
        he = task_vec == T_HANDEYE
        if he.any():
            tgt[he] = N_CAMS + rng.integers(0, 2, size=int(he.sum()))
        rg = task_vec == T_REG
        if rg.any():
            tgt[rg] = (src[rg] + rng.integers(1, N_CAMS, size=int(rg.sum()))) % N_CAMS

        ee_row = min(frame, len(e["ee"]) - 1)
        # where the model sees each query (augmented canvas indices); the target cells are the scene
        # points under the augmented p x p patch, i.e. source-canvas offsets (aug grid - centre) / s
        # around the anchor (= the query itself unless dt > 0)
        aug_r, aug_c = aff.fwd_idx(coords[:, 0], coords[:, 1])
        offs = np.arange(-(p // 2), p - p // 2, dtype=np.float64)
        xyz = np.zeros((k, p, p, 3), np.float32)
        self._prefetch((e["depth"][int(src[j])], frame + int(dt[j])) for j in np.flatnonzero(mask))
        for j in range(k):
            if not mask[j]:
                continue
            cs, f = int(src[j]), frame + int(dt[j])
            if not (e["reliable"][cs][f] and (tgt[j] >= N_CAMS or e["reliable"][int(tgt[j])][frame])):
                mask[j] = False
                continue
            gr, gc = aff.inv_cont(aug_r[j] + 0.5 + offs, aug_c[j] + 0.5 + offs)
            gr = gr - (coords[j, 0] + 0.5) + (anchor[j, 0] + 0.5)
            gc = gc - (coords[j, 1] + 0.5) + (anchor[j, 1] + 0.5)
            pc = self.grid_xyz(e, cs, f, gr, gc)
            if pc is None:
                mask[j] = False
                continue
            # anchor camera at frame t+dt -> world (wrist extrinsics move every frame)
            pw = cam_to_world(pc, e["extr"][cs, f])
            t_id = int(tgt[j])
            if t_id < N_CAMS:
                # target camera at the OBSERVATION time t
                out = world_to_cam(pw, e["extr"][t_id, frame])
            else:
                arm = t_id - N_CAMS
                pos = e["ee"][ee_row, arm * 8 : arm * 8 + 3].astype(np.float64)
                quat = e["ee"][ee_row, arm * 8 + 3 : arm * 8 + 7].astype(np.float64)
                out = world_to_gripper(pw, pos, quat_wxyz_to_mat(quat))
            xyz[j] = np.clip(out / self.xyz_scale, -1.0, 1.0)

        # query RGB patch from the augmented model image of the source camera, in [-1, 1]
        imgs = np.asarray(images, dtype=np.float32)
        if imgs.max() > 1.5:
            imgs = imgs / 255.0
        half = p // 2
        patches = np.zeros((k, 3, p, p), np.float32)
        for j in range(k):
            r, c = int(aug_r[j]), int(aug_c[j])
            patches[j] = imgs[int(src[j]), :, r - half : r - half + p, c - half : c - half + p]
        patches = patches * 2.0 - 1.0

        return {
            "d4rt_coords": np.stack([aug_r, aug_c], axis=-1).astype(np.int64),
            "d4rt_cam": src,
            "d4rt_tgt": tgt,
            "d4rt_dt": dt,
            "d4rt_task": task_vec,
            "d4rt_target": xyz,
            "d4rt_mask": mask,
            "d4rt_patches": patches,
        }


def task_from_repo(repo_id: str) -> str:
    """'/.../lerobot/robotwin_clean_<task>' (H20) or '/.../<task>-aloha-agilex_clean_50-50'
    (PPU-side conversion, legacy_lingbot/data/robotwin_clean50_lerobot) -> '<task>'."""
    name = pathlib.Path(str(repo_id).rstrip("/")).name
    return re.sub(r"-aloha-agilex_clean_50-\d+$", "", name.removeprefix("robotwin_clean_"))


def load_info(root: str | pathlib.Path = D4RT_ROOT) -> dict:
    return json.loads((pathlib.Path(root) / "DEPTH_INFO.json").read_text())
