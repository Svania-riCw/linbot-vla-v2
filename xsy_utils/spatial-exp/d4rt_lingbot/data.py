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
"""

from __future__ import annotations

import json
import pathlib
from collections import OrderedDict

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


class _Store:
    """Per-worker LRU of one (task, episode)'s d4rt arrays (memmapped where large)."""

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
        if not all(p.exists() for p in paths.values()):
            return None
        with np.load(paths["cam"]) as cam:
            intr = np.asarray(cam["intrinsic"], dtype=np.float64)  # [3, T, 3, 3]
            extr = np.asarray(cam["extrinsic"], dtype=np.float64)  # [3, T, 3, 4] world->cam
            grip = np.asarray(cam["grip"], dtype=np.float64)  # [T, 3 cams, 2 arms, (u, v, z, valid)]
        reliable, starts = [], []
        for c in CAMERAS:
            with np.load(paths[f"meta/{c}"]) as m:
                reliable.append(np.asarray(m["reliable"], dtype=bool))
            with np.load(paths[f"tmeta/{c}"]) as m:
                starts.append(np.asarray(m["starts"], dtype=np.int64))
        return {
            "ee": np.load(paths["ee"]),  # [T, 16]
            "intr": intr,
            "extr": extr,
            "grip": grip,
            "depth": [np.load(paths[f"depth/{c}"], mmap_mode="r") for c in CAMERAS],  # u16 [T, 240, 320]
            "reliable": reliable,
            "tracks": [np.load(paths[f"tracks/{c}"], mmap_mode="r") for c in CAMERAS],  # f16 [W, L, N, 2] (u, v)
            "tvis": [np.load(paths[f"tvis/{c}"], mmap_mode="r") for c in CAMERAS],  # bool [W, L, N]
            "starts": starts,
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

    def _track_slots(self, e, frame, rng, cam, slots, coords, anchor, dt, mask, weights) -> None:
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
        uv = np.asarray(tracks[w]).astype(np.float64) + 0.5
        rows, cols = native_to_canvas(uv[..., 1], uv[..., 0])
        rows = np.floor(rows + 0.5).astype(np.int64)
        cols = np.floor(cols + 0.5).astype(np.int64)
        usable = np.asarray(tvis[w]) & in_patch_bounds(rows, cols, self.p)
        horizon = usable[i + 1 : i + dt_max + 1]
        pts = np.flatnonzero(usable[i] & horizon.any(axis=0))
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

    def _gripper_time_slot(self, e, frame, rng, arm, coords, anchor, dt, mask, weights) -> None:
        """Gripper slot's trajectory from the exact endpose projection; tracker fallback when out of view."""
        n_frames = e["depth"][CAM_HIGH].shape[0]
        src = self._grip_canvas(e, frame, arm)
        if src is not None:
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
            self._track_slots(e, frame, rng, CAM_HIGH, np.array([arm]), coords, anchor, dt, mask, weights)

    def patch_xyz(self, e: dict, cam: int, frame: int, row: int, col: int) -> np.ndarray | None:
        """p x p canvas patch centred at (row, col) -> [p, p, 3] metres in that camera's frame, or None if any cell is invalid."""
        half = self.p // 2
        rr = np.arange(row - half, row - half + self.p)
        cc = np.arange(col - half, col - half + self.p)
        z = e["depth"][cam][frame][_ROW2V[rr][:, None], _COL2U[cc][None, :]].astype(np.float64) / 1000.0
        if (z <= 0).any():
            return None
        v, u = canvas_to_native(rr[:, None].astype(np.float64), cc[None, :].astype(np.float64))
        k = e["intr"][cam, frame]
        x = (u - k[0, 2]) / k[0, 0] * z
        y = (v - k[1, 2]) / k[1, 1] * z
        return np.stack([x, y, z], axis=-1)

    def __call__(self, task: str, episode: int, frame: int, images: np.ndarray | None) -> dict:
        """``images``: the model's 224x224 views after colour augmentation, [3 cams, 3, 224, 224],
        float in [0, 1] (or uint8). Returns numpy fields; all-masked if the episode has no data."""
        e = self._store.get(task, episode)
        if e is None or images is None:
            return self.empty()
        n_frames = e["depth"][0].shape[0]
        frame = min(frame, n_frames - 1)
        rng = np.random.default_rng((self.seed, self._task_id.get(task, 0), episode, frame))
        k, p = self.k, self.p
        task_vec = self._assign_tasks(rng)

        weights = [weight_map(canvas_depth(np.asarray(e["depth"][c][frame])), self.depth_max) for c in range(N_CAMS)]

        # source camera: hand-eye and the gripper-pinned slots source from cam_high
        src = rng.integers(0, N_CAMS, size=k).astype(np.int64)
        src[task_vec == T_HANDEYE] = CAM_HIGH
        n_grip = min(2, k) if self.force_gripper else 0
        src[:n_grip] = CAM_HIGH

        coords = np.zeros((k, 2), np.int64)
        for c in range(N_CAMS):
            idx = np.flatnonzero(src == c)
            if len(idx):
                coords[idx] = sample_cells(weights[c], rng, len(idx), self.alpha, self.gamma, self.cell_ok)
        for arm in range(n_grip):
            g = self._grip_canvas(e, frame, arm)
            if g is not None:
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
                    self._track_slots(e, frame, rng, c, idx, coords, anchor, dt, mask, weights)
            for j in time_slots[time_slots < n_grip]:
                self._gripper_time_slot(e, frame, rng, int(j), coords, anchor, dt, mask, weights)

        tgt = src.copy()
        he = task_vec == T_HANDEYE
        if he.any():
            tgt[he] = N_CAMS + rng.integers(0, 2, size=int(he.sum()))
        rg = task_vec == T_REG
        if rg.any():
            tgt[rg] = (src[rg] + rng.integers(1, N_CAMS, size=int(rg.sum()))) % N_CAMS

        ee_row = min(frame, len(e["ee"]) - 1)
        xyz = np.zeros((k, p, p, 3), np.float32)
        for j in range(k):
            if not mask[j]:
                continue
            cs, f = int(src[j]), frame + int(dt[j])
            if not (e["reliable"][cs][f] and (tgt[j] >= N_CAMS or e["reliable"][int(tgt[j])][frame])):
                mask[j] = False
                continue
            pc = self.patch_xyz(e, cs, f, int(anchor[j, 0]), int(anchor[j, 1]))
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
            r, c = int(coords[j, 0]), int(coords[j, 1])
            patches[j] = imgs[int(src[j]), :, r - half : r - half + p, c - half : c - half + p]
        patches = patches * 2.0 - 1.0

        return {
            "d4rt_coords": coords,
            "d4rt_cam": src,
            "d4rt_tgt": tgt,
            "d4rt_dt": dt,
            "d4rt_task": task_vec,
            "d4rt_target": xyz,
            "d4rt_mask": mask,
            "d4rt_patches": patches,
        }


def task_from_repo(repo_id: str) -> str:
    """'/.../lerobot/robotwin_clean_<task>' -> '<task>'."""
    name = pathlib.Path(str(repo_id).rstrip("/")).name
    return name.removeprefix("robotwin_clean_")


def load_info(root: str | pathlib.Path = D4RT_ROOT) -> dict:
    return json.loads((pathlib.Path(root) / "DEPTH_INFO.json").read_text())
