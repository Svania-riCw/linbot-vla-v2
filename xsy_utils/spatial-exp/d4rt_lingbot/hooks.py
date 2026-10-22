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
    context           str    prefix         prefix: every prefix token (image, text, alignment
                                            queries); visual_state: image patch tokens only, plus
                                            the raw robot state as one token (probe's own Linear)
    cotrain_every     int    0              >0: after every N-th action step run one d4rt-only
                                            step (prefix forward, d4rt loss only, action expert
                                            untouched, LR schedule not advanced) on a separate,
                                            perturbed data stream
    perturb           str    ""             JSON from calibrate_perturb.py: the random-domain
                                            photometric perturbation used on d4rt-only steps
"""

from __future__ import annotations

import numpy as np
import torch

from .data import CANVAS, CanvasAffine, D4rtSampler, task_from_repo
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
    "context": "prefix",
    "cotrain_every": 0,
    "perturb": "",
}
CONTEXTS = ("prefix", "visual_state")


def params(config) -> dict:
    raw = getattr(config, "d4rt_params", None) or {}
    if not isinstance(raw, dict):
        raw = dict(raw)
    unknown = set(raw) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown d4rt_params keys: {sorted(unknown)}")
    p = {**_DEFAULTS, **raw}
    if p["context"] not in CONTEXTS:
        raise ValueError(f"d4rt_params.context must be one of {CONTEXTS}, got {p['context']!r}")
    return p


def enabled(config) -> bool:
    return bool(params(config)["enable"])


# ---- model -----------------------------------------------------------------------------
def build_probe(config, context_width: int) -> D4rtProbe:
    p = params(config)
    state_dim = int(config.max_state_dim) if p["context"] == "visual_state" else 0
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
        state_dim=state_dim,
    )
    probe.reset_parameters()
    return probe


def pop_batch(kwargs: dict) -> dict | None:
    """Remove the d4rt fields from a forward()'s **kwargs; None if the batch has none."""
    batch = {k: kwargs.pop(k) for k in D4RT_KEYS if k in kwargs}
    return batch or None


def model_loss(config, probe: D4rtProbe, prefix_hidden: torch.Tensor, prefix_pad_masks: torch.Tensor,
               batch: dict | None, visual_pos_masks: torch.Tensor | None = None,
               state: torch.Tensor | None = None) -> tuple[torch.Tensor | float, dict]:
    """-> (weighted loss to add to the total, metrics incl. ``d4rt/loss`` unweighted).
    ``visual_pos_masks`` [b, prefix_len] marks the image patch tokens (from ``embed_prefix``)."""
    if batch is None:
        return 0.0, {}
    p = params(config)
    mask = prefix_pad_masks.bool()
    if p["context"] == "visual_state":
        mask = mask & visual_pos_masks.bool()
    loss, metrics = probe.loss(prefix_hidden, mask, batch, state=state if p["context"] == "visual_state" else None)
    w = p["loss_weight"]
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


def to_canvas(imgs: np.ndarray) -> np.ndarray:
    """The model's views [n, 3, H, W] (``data.img_size``: 256 / 384, uint8 or float [0, 1]) ->
    [n, 3, 224, 224] float32 in [0, 1], area-resized (half-pixel centres, as the sampler assumes)."""
    x = np.asarray(imgs).astype(np.float32)
    if np.asarray(imgs).dtype == np.uint8 or x.max() > 1.5:
        x = x / 255.0
    if x.shape[-2:] == (CANVAS, CANVAS):
        return x
    import cv2

    out = np.empty(x.shape[:-2] + (CANVAS, CANVAS), np.float32)
    for i in range(x.shape[0]):
        out[i] = cv2.resize(x[i].transpose(1, 2, 0), (CANVAS, CANVAS), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    return out


def canvas_affine(aug_plan: dict | None) -> CanvasAffine | None:
    """Plan of ``lingbotvla/utils/observation_augmentation.ObservationAugmenter.sample`` -> its 2-D
    scale/translation (shared by all views and frames), or None if this sample was not warped."""
    if not aug_plan or aug_plan.get("keep_original") or not aug_plan.get("affine"):
        return None
    tx, ty = aug_plan["translation"]
    return CanvasAffine(aug_plan["scale"], tx, ty)


def dataset_fields(config, repo_id: str, raw_item: dict, item: dict, aug_plan: dict | None = None) -> dict:
    """Called per sample after the feature transform. ``item['pil_images']`` are the model's
    3 views after augmentation, [3, 3, img_size, img_size]; ``aug_plan`` the observation
    augmentation's plan for this sample (None when that augmentation is off)."""
    ep = int(np.asarray(raw_item["episode_index"]).reshape(-1)[0])
    frame = int(np.asarray(raw_item["frame_index"]).reshape(-1)[0])
    imgs = item.get("pil_images")
    if imgs is None or (hasattr(imgs, "__len__") and len(imgs) == 0):
        raise RuntimeError("d4rt needs the model's augmented views (pil_images): enable align_params / use_depth_align.")
    imgs = to_canvas(imgs.cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs))
    fields = _sampler(config)(task_from_repo(repo_id), ep, frame, imgs, canvas_affine(aug_plan))
    return {k: torch.from_numpy(v) for k, v in fields.items()}


# ---- d4rt-only co-training ------------------------------------------------------------
def cotrain_every(config) -> int:
    return int(params(config)["cotrain_every"]) if enabled(config) else 0


def make_perturbed(dataset, config) -> None:
    """Give every sub-dataset's feature transform the calibrated random-domain perturbation.
    The baseline trains with ``data.image_augment: false`` (no colour augmentation at all); on this
    separate stream train-mode image processing is switched on and ``prepare_images`` applies the
    perturbation *instead of* the standard colour augmentation, so the perturbation is the only
    photometric change. The action stream is untouched."""
    from .perturb import Perturb

    path = params(config)["perturb"]
    if not path:
        raise ValueError("d4rt_params.cotrain_every > 0 needs d4rt_params.perturb (calibrate_perturb.py output)")
    pert = Perturb(path)
    subs = getattr(dataset, "_datasets", None) or [dataset]
    seen = set()
    for d in subs:
        ft = d.feature_transform
        if id(ft) in seen:
            continue
        seen.add(id(ft))
        ft.image_augment = True
        ft.visual_perturb = pert


def d4rt_only_trainable(name: str) -> bool:
    """Parameters a d4rt-only step may update: the VLM (vision + LLM), the prefix alignment-query
    embeddings the probe reads, and the probe. Everything else (action expert, state/action
    projections, alignment heads) keeps its grad None, so the optimizer skips it entirely."""
    return (
        ".qwenvl_with_expert.qwenvl." in f".{name}"
        or f".{PROBE_ATTR}." in f".{name}"
        or name.split(".")[-1].endswith("_align_embs")
        or "_align_embs." in name
    )
