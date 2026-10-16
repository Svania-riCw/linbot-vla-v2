"""CPU reference contracts, NOT a replacement for LingBot's training pipeline.

Joint layout: [left q(6), left gripper, right q(6), right gripper].
EE layout: [left xyz+xyzw(7), left gripper, right xyz+xyzw(7), right gripper].
Inputs are physical units before normalization. One frozen anchor per chunk.
"""
import numpy as np
from scipy.spatial.transform import Rotation

ARM = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIP = np.array([6, 13])


def finite_array(value, name):
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.floating):
        raise ValueError(f"{name} must be floating point, got {a.dtype}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return a


def _chunk_inputs(actions, state, dim):
    a, s = finite_array(actions, "actions"), finite_array(state, "state")
    if a.ndim not in (2, 3) or a.shape[-1] != dim or a.shape[-2] == 0:
        raise ValueError(f"expected [H,{dim}] or [B,H,{dim}] nonempty chunk")
    if s.shape != a.shape[:-2] + (dim,):
        raise ValueError("exactly one current state per chunk is required")
    return a, s


def encode_joint_delta(actions, state):
    a, s = _chunk_inputs(actions, state, 14)
    out = a.copy()
    out[..., ARM] = a[..., ARM] - s[..., None, ARM]
    return out


def decode_joint_delta(delta, state):
    a, s = _chunk_inputs(delta, state, 14)
    out = a.copy()
    out[..., ARM] = a[..., ARM] + s[..., None, ARM]
    return out


def _rotation(q):
    if np.any(np.linalg.norm(q, axis=-1) < 1e-8):
        raise ValueError("zero quaternion is invalid")
    return Rotation.from_quat(q.reshape(-1, 4))


def _ee_transform(actions, state, inverse):
    a, s = _chunk_inputs(actions, state, 16)
    out = a.copy()
    for start in (0, 8):
        pose = a[..., start:start+7]
        anchor = np.broadcast_to(s[..., None, start:start+7], pose.shape)
        r0, r = _rotation(anchor[..., 3:7]), _rotation(pose[..., 3:7])
        p0, p = anchor[..., :3].reshape(-1, 3), pose[..., :3].reshape(-1, 3)
        pos = p0 + r0.apply(p) if inverse else r0.inv().apply(p - p0)
        rot = r0 * r if inverse else r0.inv() * r
        out[..., start:start+3] = pos.reshape(pose.shape[:-1] + (3,))
        out[..., start+3:start+7] = rot.as_quat(canonical=True).reshape(pose.shape[:-1] + (4,))
    return out


def encode_ee_delta(actions_xyzw, state_xyzw):
    """Local SE(3): inverse(T_state) @ T_action; grippers stay absolute."""
    return _ee_transform(actions_xyzw, state_xyzw, inverse=False)


def decode_ee_delta(delta_xyzw, state_xyzw):
    return _ee_transform(delta_xyzw, state_xyzw, inverse=True)


def reorder_ee_quaternions(actions, source, target):
    """Explicit order conversion only; DOES NOT change tool/base frames."""
    a = finite_array(actions, "EE actions")
    if a.shape[-1] != 16 or source not in ("xyzw", "wxyz") or target not in ("xyzw", "wxyz"):
        raise ValueError("expected 16-dim dual-arm EE, explicit xyzw/wxyz orders")
    out = a.copy()
    if source != target:
        order = [3, 0, 1, 2] if source == "xyzw" else [1, 2, 3, 0]
        for start in (3, 11):
            out[..., start:start+4] = a[..., start:start+4][..., order]
    return out


def weighted_flow_l1(pred, target, joint_mask, dimension_weight=None, horizon_weight=None):
    """NumPy oracle for a future torch loss; dimension mask only, no pad change.

    All-one weights recover baseline global masked L1 reduction. Zero total
    weight raises here as a preflight error (baseline itself clamps denominator).
    """
    p, t = finite_array(pred, "pred"), finite_array(target, "target")
    if p.ndim != 3 or p.shape != t.shape:
        raise ValueError("pred/target must share [B,H,D]")
    m = np.asarray(joint_mask)
    if m.shape != p.shape or m.dtype != np.bool_:
        raise ValueError("joint_mask must be bool [B,H,D]")
    d = np.ones(p.shape[2]) if dimension_weight is None else np.asarray(dimension_weight)
    h = np.ones(p.shape[1]) if horizon_weight is None else np.asarray(horizon_weight)
    if d.shape != (p.shape[2],) or h.shape != (p.shape[1],):
        raise ValueError("weight shape mismatch")
    if not np.isfinite(d).all() or not np.isfinite(h).all() or np.any(d < 0) or np.any(h < 0):
        raise ValueError("weights must be finite and nonnegative")
    w = m * d[None, None, :] * h[None, :, None]
    if w.sum() <= 0:
        raise ValueError("no effective supervision")
    return float((w * np.abs(p-t)).sum() / w.sum())


def recover_clean_action(x_s, velocity, time):
    """In normalized action coordinates, BEFORE inverse norm/delta and FK."""
    x, v = finite_array(x_s, "x_s"), finite_array(velocity, "velocity")
    s = np.asarray(time)
    if x.ndim != 3 or x.shape != v.shape or s.shape != (x.shape[0],):
        raise ValueError("expected [B,H,D] tensors and [B] time")
    if not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError("time must be in [0,1]")
    return x - s[:, None, None] * v
