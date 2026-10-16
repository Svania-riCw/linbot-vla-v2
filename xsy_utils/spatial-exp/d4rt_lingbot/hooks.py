"""The only functions LingBot core code calls (via ``lingbotvla/spatial_exp.py``).

Every hook is a no-op unless the model config carries ``d4rt_params: {enable: true, ...}``
(``train.d4rt_params`` in the YAML), so the baseline is bit-identical with it absent.

``d4rt_params`` keys (defaults = pi05 ``pi05_robotwin_new_3view_d4rt_full_unified``, K=8):
    enable            bool   false
    loss_weight       float  0.1
    root              str    /2025233210/xiesy/data/lingbot_robotwin_clean_d4rt
    num_queries       int    8
    patch_size        int    14
    task_mix          list   [0.25, 0.25, 0.25, 0.25]   depth / time / hand-eye / registration
    xyz_scale         float  1.2
    max_dt            int    49
    width / depth / num_heads / head_dim / mlp_dim      1024 / 3 / 8 / 128 / 4096
"""

from __future__ import annotations

import numpy as np
import torch

from .data import D4rtSampler, task_from_repo
from .probe import D4rtProbe

D4RT_KEYS = ("d4rt_coords", "d4rt_cam", "d4rt_tgt", "d4rt_dt", "d4rt_task", "d4rt_target", "d4rt_mask", "d4rt_patches")
PROBE_ATTR = "d4rt_probe"

_DEFAULTS = {
    "enable": False,
    "loss_weight": 0.1,
    "root": "/2025233210/xiesy/data/lingbot_robotwin_clean_d4rt",
    "num_queries": 8,
    "patch_size": 14,
    "task_mix": [0.25, 0.25, 0.25, 0.25],
    "xyz_scale": 1.2,
    "max_dt": 49,
    "width": 1024,
    "depth": 3,
    "num_heads": 8,
    "head_dim": 128,
    "mlp_dim": 4096,
    "alpha": 0.1,
    "gamma": 2.0,
    "seed": 0,
}


def params(config) -> dict:
    raw = getattr(config, "d4rt_params", None) or {}
    if not isinstance(raw, dict):
        raw = dict(raw)
    unknown = set(raw) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown d4rt_params keys: {sorted(unknown)}")
    return {**_DEFAULTS, **raw}


def enabled(config) -> bool:
    return bool(params(config)["enable"])


# ---- model -----------------------------------------------------------------------------
def build_probe(config, context_width: int) -> D4rtProbe:
    p = params(config)
    probe = D4rtProbe(
        context_width=context_width,
        width=p["width"],
        depth=p["depth"],
        num_heads=p["num_heads"],
        head_dim=p["head_dim"],
        mlp_dim=p["mlp_dim"],
        patch_size=p["patch_size"],
        max_dt=p["max_dt"],
        xyz_scale=p["xyz_scale"],
    )
    probe.reset_parameters()
    return probe


def pop_batch(kwargs: dict) -> dict | None:
    """Remove the d4rt fields from a forward()'s **kwargs; None if the batch has none."""
    batch = {k: kwargs.pop(k) for k in D4RT_KEYS if k in kwargs}
    return batch or None


def model_loss(config, probe: D4rtProbe, prefix_hidden: torch.Tensor, prefix_pad_masks: torch.Tensor,
               batch: dict | None) -> tuple[torch.Tensor | float, dict]:
    """-> (weighted loss to add to the total, metrics incl. ``d4rt/loss`` unweighted)."""
    if batch is None:
        return 0.0, {}
    loss, metrics = probe.loss(prefix_hidden, prefix_pad_masks, batch)
    w = params(config)["loss_weight"]
    metrics["d4rt/loss_weighted"] = (loss * w).detach()
    return loss * w, metrics


def init_missing(model, missing: set) -> set:
    """Loader hook: the probe is not in the pretrained checkpoint. Re-init it (the loader's
    ``to_empty`` left its storage uninitialized) and drop its names from the missing set."""
    probe_names = {n for n in missing if f".{PROBE_ATTR}." in f".{n}"}
    if not probe_names:
        return missing
    for m in model.modules():
        if isinstance(m, D4rtProbe):
            m.reset_parameters()
    return missing - probe_names


# ---- data ------------------------------------------------------------------------------
_SAMPLER: D4rtSampler | None = None


def _sampler(config) -> D4rtSampler:
    global _SAMPLER
    if _SAMPLER is None:
        p = params(config)
        _SAMPLER = D4rtSampler(
            root=p["root"],
            num_queries=p["num_queries"],
            patch_size=p["patch_size"],
            task_mix=tuple(p["task_mix"]),
            xyz_scale=p["xyz_scale"],
            max_dt=p["max_dt"],
            alpha=p["alpha"],
            gamma=p["gamma"],
            depth_max=p["xyz_scale"],
            seed=p["seed"],
        )
    return _SAMPLER


def dataset_fields(config, repo_id: str, raw_item: dict, item: dict) -> dict:
    """Called per sample after the feature transform. ``item['pil_images']`` are the model's
    3 views after colour augmentation, [3, 3, 224, 224]."""
    ep = int(np.asarray(raw_item["episode_index"]).reshape(-1)[0])
    frame = int(np.asarray(raw_item["frame_index"]).reshape(-1)[0])
    imgs = item.get("pil_images")
    if imgs is None or (hasattr(imgs, "__len__") and len(imgs) == 0):
        raise RuntimeError("d4rt needs the augmented 224x224 views (pil_images): enable align_params / use_depth_align.")
    imgs = imgs.cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs)
    fields = _sampler(config)(task_from_repo(repo_id), ep, frame, imgs)
    return {k: torch.from_numpy(v) for k, v in fields.items()}
