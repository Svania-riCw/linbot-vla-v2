"""Photometric "random-domain" perturbation for the d4rt-only co-training steps.

A parametric colour transform whose parameter ranges are fitted (``calibrate_perturb.py``) so that
clean RoboTwin frames, once perturbed, have the same distribution of global colour statistics as
the RoboTwin *random* (domain-randomized) eval frames. It is purely photometric: pixels do not
move, so the d4rt targets stay exact.

One parameter set is sampled per training sample and shared by its 3 views (one scene, one
lighting), like the standard colour augmentation it replaces on those steps.

    x  in [0, 1], CHW
    x  = x * gain_c                        per-channel gain (colour temperature / tint), log-uniform
    x  = x * brightness                    log-uniform
    x  = (x - mean) * contrast + mean      uniform
    x  = gray + (x - gray) * saturation    uniform
    x  = hue rotation by h (YIQ)           uniform in [-hue, hue]
    x  = clamp(x)^gamma                    log-uniform
    x  = gaussian blur sigma (px @ 224)    uniform in [0, blur]
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

# name -> (kind, number of bounds). "lu" = log-uniform [lo, hi]; "u" = uniform [lo, hi];
# "sym" = uniform [-hi, hi]; "zero" = uniform [0, hi]
SPEC = {
    "gain_r": "lu",
    "gain_g": "lu",
    "gain_b": "lu",
    "brightness": "lu",
    "contrast": "u",
    "saturation": "u",
    "hue": "sym",
    "gamma": "lu",
    "blur": "zero",
}
IDENTITY = {
    "gain_r": [0.0, 0.0], "gain_g": [0.0, 0.0], "gain_b": [0.0, 0.0], "brightness": [0.0, 0.0],
    "contrast": [1.0, 1.0], "saturation": [1.0, 1.0], "hue": [0.0], "gamma": [0.0, 0.0], "blur": [0.0],
}  # log-uniform bounds are in log space

_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ_INV = torch.linalg.inv(_YIQ)


def load(path: str) -> dict:
    with open(path) as f:
        ranges = json.load(f)["ranges"]
    if set(ranges) != set(SPEC):
        raise ValueError(f"{path}: expected keys {sorted(SPEC)}, got {sorted(ranges)}")
    return ranges


def sample(ranges: dict, u: torch.Tensor | None = None, generator: torch.Generator | None = None) -> dict:
    """-> {name: float tensor [n]}. ``u`` [n, len(SPEC)] uniform(0,1) draws (common random numbers
    in calibration); otherwise one draw from ``generator``."""
    if u is None:
        u = torch.rand(1, len(SPEC), generator=generator)
    out = {}
    for j, (name, kind) in enumerate(SPEC.items()):
        b = ranges[name]
        x = u[:, j]
        if kind == "lu":
            out[name] = torch.exp(b[0] + (b[1] - b[0]) * x)
        elif kind == "u":
            out[name] = b[0] + (b[1] - b[0]) * x
        elif kind == "sym":
            out[name] = (2 * x - 1) * abs(b[0])
        else:
            out[name] = abs(b[0]) * x
    return out


def _gauss_kernels(sigma: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """[n] sigmas -> [n, 2r+1] normalized 1-D kernels (sigma ~ 0 -> delta)."""
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    s = sigma.clamp(min=1e-3)[:, None]
    k = torch.exp(-0.5 * (t[None] / s) ** 2)
    return k / k.sum(dim=1, keepdim=True)


def apply_batch(x: torch.Tensor, p: dict) -> torch.Tensor:
    """x [n, v, 3, H, W] float in [0, 1]; p: {name: [n]} -> same shape (one param set per n)."""
    n, v, c, h, w = x.shape
    sh = (n, 1, 1, 1, 1)
    gain = torch.stack([p["gain_r"], p["gain_g"], p["gain_b"]], dim=1).view(n, 1, 3, 1, 1)
    x = x * gain * p["brightness"].view(sh)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    x = (x - mean) * p["contrast"].view(sh) + mean
    gray = x.mean(dim=2, keepdim=True)
    x = gray + (x - gray) * p["saturation"].view(sh)
    # hue: rotate the IQ chroma plane
    th = p["hue"]
    cos, sin = torch.cos(th), torch.sin(th)
    rot = torch.zeros(n, 3, 3)
    rot[:, 0, 0] = 1
    rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = cos, -sin, sin, cos
    m = _YIQ_INV[None] @ rot @ _YIQ[None]  # [n, 3, 3]
    x = torch.einsum("nij,nvjhw->nvihw", m.to(x.dtype), x)
    x = x.clamp(0.0, 1.0) ** p["gamma"].view(sh)
    if bool((p["blur"] > 0.05).any()):
        k = _gauss_kernels(p["blur"])  # [n, K]
        r = k.shape[1] // 2
        y = x.reshape(n, v * c, h, w)
        y = y.reshape(1, n * v * c, h, w)
        kk = k.repeat_interleave(v * c, dim=0)  # [n*v*c, K]
        y = F.pad(y, (r, r, r, r), mode="reflect")
        y = F.conv2d(y, kk[:, None, None, :], groups=n * v * c)
        y = F.conv2d(y, kk[:, None, :, None], groups=n * v * c)
        x = y.view(n, v, c, h, w)
    return x.clamp(0.0, 1.0)


class Perturb:
    """Picklable callable used by the data pipeline (one instance per perturbed dataset)."""

    def __init__(self, path: str):
        self.path = path
        self.ranges = load(path)

    def sample(self) -> dict:
        return {k: v[0] for k, v in sample(self.ranges).items()}

    def __call__(self, img: torch.Tensor, p: dict) -> torch.Tensor:
        """img CHW uint8 or float (0..255 or 0..1) -> same dtype/scale, perturbed."""
        dtype = img.dtype
        x = img.to(torch.float32)
        scale = 255.0 if (dtype == torch.uint8 or x.max() > 1.0) else 1.0
        y = apply_batch((x / scale)[None, None], {k: torch.as_tensor(v).reshape(1) for k, v in p.items()})[0, 0]
        y = y * scale
        if dtype == torch.uint8:
            return y.round().to(torch.uint8)
        return y.to(dtype)


def describe(ranges: dict) -> str:
    parts = []
    for name, kind in SPEC.items():
        b = ranges[name]
        if kind == "lu":
            parts.append(f"{name} x[{math.exp(b[0]):.3f}, {math.exp(b[1]):.3f}]")
        elif kind == "u":
            parts.append(f"{name} [{b[0]:.3f}, {b[1]:.3f}]")
        elif kind == "sym":
            parts.append(f"{name} +-{math.degrees(abs(b[0])):.1f} deg")
        else:
            parts.append(f"{name} [0, {abs(b[0]):.2f}] px")
    return ", ".join(parts)
