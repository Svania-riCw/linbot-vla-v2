"""Per-frame metric-scale correction of MoGe depth against RoboTwin's known scene planes.

MoGe-3 (with the GT focal imposed) gets the *shape* right but over-estimates the metric scale
on RoboTwin's renders by ~1.6x (measured at the gripper). The scene contains two planes whose
pose is known exactly -- the table top and the back wall -- and every camera's per-frame
extrinsic is GT, so for every pixel whose ray hits one of them we know what the depth *would*
be if nothing stood in front of it:

    plane_depth(u, v)  (ray / plane intersection, in front of the camera, inside the plane's extent)

Pixels that really show the plane have ``plane_depth / moge_depth == s`` (the scale error);
pixels showing an object or the arm are closer than the plane, so their ratio is **larger**.
``s`` is therefore the dominant mode of the ratio histogram (log space), not its median, and
objects / arm are rejected without any colour or segmentation (pi05's DEPTH_ANCHORING.md shows
why colour-based table masks break on randomized scenes).

Scene geometry (envs/_base_task.py ``create_table_and_wall``):
  table top  z = 0.74 + height_bias,  x in [-0.6, 0.6] + x_bias,  y in [-0.35, 0.35]
  back wall  y = 0.40 (box centred at y=1 with half-size 0.6),  z in [0, 3]
Per-task overrides (``_init_task_env_`` calls): see ``TABLE_BIAS``.
"""

from __future__ import annotations

import numpy as np

TABLE_Z = 0.74
TABLE_X, TABLE_Y = 0.6, 0.35
WALL_Y = 0.40
MARGIN = 0.03  # stay this far inside the table edge
TABLE_BIAS = {  # task -> (x_bias, height_bias)
    "dump_bin_bigbin": (0.3, 0.0),
    "put_bottles_dustbin": (0.3, 0.0),
    "place_dual_shoes": (0.0, -0.1),
}


def plane_depth(K: np.ndarray, E: np.ndarray, h: int, w: int, task: str) -> np.ndarray:
    """GT depth (along the optical axis, metres) of the table/wall behind each pixel; nan where
    the ray hits neither inside its extent. K [3,3], E world->camera [3,4]."""
    xb, hb = TABLE_BIAS.get(task, (0.0, 0.0))
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)  # z = 1
    R, t = E[:, :3].astype(np.float64), E[:, 3].astype(np.float64)
    cam_c = -R.T @ t
    d = rays @ R  # = (R^T ray) per pixel, world direction for unit optical-axis depth
    out = np.full((h, w), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (TABLE_Z + hb - cam_c[2]) / d[..., 2]
        p = cam_c + lam[..., None] * d
        ok = (lam > 0) & (np.abs(p[..., 0] - xb) < TABLE_X - MARGIN) & (np.abs(p[..., 1]) < TABLE_Y - MARGIN)
        out = np.where(ok, lam, out)
        lam = (WALL_Y - cam_c[1]) / d[..., 1]
        p = cam_c + lam[..., None] * d
        ok = (lam > 0) & (p[..., 2] > TABLE_Z + hb + MARGIN) & (p[..., 2] < 3)
        out = np.where(ok & (lam < out), lam, out)
    return np.where(np.isfinite(out), out, np.nan)


def frame_scale(plane: np.ndarray, moge: np.ndarray, bin_w: float = 0.004) -> tuple[float, float, int]:
    """-> (scale, inlier_frac, n_inliers). scale = mode of log(plane / moge)."""
    m = np.isfinite(plane) & (moge > 0)
    n = int(m.sum())
    if n < 200:
        return np.nan, 0.0, 0
    lr = np.log(plane[m] / moge[m])
    lo, hi = np.percentile(lr, [1, 99])
    if hi - lo < bin_w:
        hi = lo + bin_w
    hist, edges = np.histogram(lr, bins=max(int((hi - lo) / bin_w), 1), range=(lo, hi))
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    peak = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    inl = np.abs(lr - peak) < 0.02
    s = float(np.exp(np.median(lr[inl]))) if inl.any() else float(np.exp(peak))
    return s, float(inl.mean()), int(inl.sum())


MIN_INLIERS = 1500  # ~2% of a 320x240 frame


def episode_scales(scales: np.ndarray, n_inliers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (applied scale, reliable). Reliable frames keep their **own** fit: MoGe's raw scale
    jitters from frame to frame (measured +-30% on a static wrist view), so smoothing the
    correction over time would re-introduce exactly that jitter. Frames without enough plane
    pixels get the nearest reliable frames' scale interpolated -- approximate, flagged."""
    good = np.isfinite(scales) & (n_inliers >= MIN_INLIERS)
    if not good.any():
        return np.full_like(scales, np.nan), good
    idx = np.arange(len(scales))
    return np.interp(idx, idx[good], scales[good]).astype(np.float32), good
