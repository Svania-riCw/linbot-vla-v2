"""Spatial Forcing teacher and target utilities.

The teacher is used only while training.  VGGT is frozen and its aggregator
features are resized to the Qwen3-VL visual-token grid before they are passed
to the VLA alignment loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import warnings


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _factor_grid(num_tokens: int) -> Tuple[int, int]:
    """Return a near-square H/W factorization for a flattened token count."""
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    height = int(num_tokens**0.5)
    while height > 1 and num_tokens % height:
        height -= 1
    return height, num_tokens // height


def _position_grid_to_embed(height: int, width: int, dim: int, device, dtype) -> torch.Tensor:
    """Create the 2-D sinusoidal position embedding used by the SF target."""
    if dim % 4:
        raise ValueError(f"target_dim must be divisible by 4 for 2-D PE, got {dim}")
    aspect_ratio = width / height
    diagonal = (aspect_ratio**2 + 1.0) ** 0.5
    span_x, span_y = aspect_ratio / diagonal, 1.0 / diagonal
    x = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        device=device,
        dtype=torch.float32,
    )
    y = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        device=device,
        dtype=torch.float32,
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    half = dim // 4
    freq = torch.arange(half, device=device, dtype=torch.float64)
    freq = 1.0 / (100.0 ** (freq / half))
    x_phase = xx.reshape(-1, 1) * freq.reshape(1, -1)
    y_phase = yy.reshape(-1, 1) * freq.reshape(1, -1)
    emb = torch.cat(
        [torch.sin(x_phase), torch.cos(x_phase), torch.sin(y_phase), torch.cos(y_phase)],
        dim=-1,
    )
    return emb.reshape(height, width, dim).to(dtype=dtype)


def _select_feature_layer(features, layer: int) -> torch.Tensor:
    if layer < 0:
        for feature in reversed(features):
            if feature is not None:
                return feature
        raise RuntimeError("VGGT aggregator returned no cached features")
    if layer >= len(features) or features[layer] is None:
        available = [idx for idx, feature in enumerate(features) if feature is not None]
        raise ValueError(f"VGGT target_layer={layer} is unavailable; cached layers are {available}")
    return features[layer]


def build_spatial_forcing_teacher(config: Any, device: Optional[torch.device] = None) -> nn.Module:
    """Build a frozen VGGT aggregator teacher from a local file or HF repo."""
    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:
        raise ImportError(
            "Spatial Forcing requires the optional VGGT package. "
            "Install it with `pip install -e /path/to/facebookresearch/vggt`."
        ) from exc

    checkpoint_path = _cfg_get(config, "checkpoint_path", None)
    repo_id = _cfg_get(config, "repo_id", "facebook/VGGT-1B")
    model_kwargs = {
        "enable_camera": False,
        "enable_point": False,
        "enable_depth": False,
        "enable_track": False,
    }
    path = Path(checkpoint_path) if checkpoint_path else None
    if path is not None and path.is_dir():
        teacher = VGGT.from_pretrained(str(path), **model_kwargs)
    elif path is not None and path.is_file():
        teacher = VGGT(**model_kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        teacher.load_state_dict(state, strict=False)
    else:
        if path is not None:
            warnings.warn(
                f"Spatial Forcing checkpoint {path} is not present; falling back to Hugging Face repo {repo_id}.",
                stacklevel=2,
            )
        teacher = VGGT.from_pretrained(repo_id, **model_kwargs)

    if device is not None:
        teacher = teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def _vla_grid_from_image_grid(
    image_grid_thw: Optional[torch.Tensor],
    num_images: int,
    merge_size: int,
    fallback_tokens: Optional[int],
) -> Tuple[int, int]:
    if image_grid_thw is not None:
        grid = image_grid_thw.reshape(-1, image_grid_thw.shape[-1])
        if grid.shape[-1] < 3:
            raise ValueError(f"image_grid_thw must end with (T,H,W), got {tuple(image_grid_thw.shape)}")
        height = int(grid[0, -2].item()) // merge_size
        width = int(grid[0, -1].item()) // merge_size
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid Qwen3-VL image grid: {grid[0].tolist()}")
        expected = height * width
        if not torch.all((grid[:, -2] // merge_size) * (grid[:, -1] // merge_size) == expected):
            raise ValueError("Spatial Forcing currently requires a fixed Qwen3-VL patch grid for all images.")
        return height, width
    if fallback_tokens is None:
        raise ValueError("Spatial Forcing needs image_grid_thw or vla_patch_tokens in its config")
    return _factor_grid(int(fallback_tokens))


@torch.no_grad()
def get_spatial_forcing_target(
    teacher: nn.Module,
    pil_images: torch.Tensor,
    img_masks: Optional[torch.Tensor],
    config: Any,
    image_grid_thw: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return VGGT features resized to VLA tokens and a valid-token mask.

    Args:
        pil_images: uint8/float images with shape ``[B, S, 3, H, W]``.
        img_masks: valid-camera mask with shape ``[B, S]``.
        image_grid_thw: Qwen3-VL grid, used to recover the VLA patch grid.
    """
    if pil_images is None or pil_images.ndim != 5:
        raise ValueError(f"pil_images must have shape [B,S,3,H,W], got {None if pil_images is None else tuple(pil_images.shape)}")
    batch, num_images, channels, _, _ = pil_images.shape
    if channels != 3:
        raise ValueError(f"Spatial Forcing expects RGB images, got {channels} channels")

    input_size = int(_cfg_get(config, "input_size", 518))
    target_layer = int(_cfg_get(config, "target_layer", -1))
    use_pe = bool(_cfg_get(config, "use_positional_encoding", True))
    pe_scale = float(_cfg_get(config, "positional_encoding_scale", 0.1))
    merge_size = int(_cfg_get(config, "vla_merge_size", 2))
    fallback_tokens = _cfg_get(config, "vla_patch_tokens", None)
    vla_h, vla_w = _vla_grid_from_image_grid(image_grid_thw, num_images, merge_size, fallback_tokens)

    images = pil_images.float()
    if images.numel() and images.amax() > 1.0:
        images = images / 255.0
    images = F.interpolate(
        images.reshape(batch * num_images, 3, images.shape[-2], images.shape[-1]),
        size=(input_size, input_size),
        mode="bicubic",
        align_corners=False,
    ).reshape(batch, num_images, 3, input_size, input_size).clamp_(0.0, 1.0)

    teacher_device = next(teacher.parameters()).device
    images = images.to(device=teacher_device)
    if img_masks is None:
        img_masks = torch.ones(batch, num_images, dtype=torch.bool, device=teacher_device)
    else:
        img_masks = img_masks.to(device=teacher_device, dtype=torch.bool)
    if img_masks.shape != (batch, num_images):
        raise ValueError(f"img_masks must have shape {(batch, num_images)}, got {tuple(img_masks.shape)}")

    empty_samples = (~img_masks.any(dim=1)).nonzero(as_tuple=False).flatten()
    if empty_samples.numel():
        raise ValueError(f"Samples {empty_samples.tolist()} have no valid camera image for Spatial Forcing")

    # Group equal camera masks so VGGT can process full batches without letting
    # padded/missing cameras attend to valid views from another sample.
    grouped_samples = {}
    for sample_idx, camera_mask in enumerate(img_masks.tolist()):
        grouped_samples.setdefault(tuple(camera_mask), []).append(sample_idx)

    target_by_image = None
    target_dim = None
    for camera_mask, sample_indices in grouped_samples.items():
        sample_indices = torch.tensor(sample_indices, dtype=torch.long, device=teacher_device)
        valid_views = torch.tensor(
            [idx for idx, is_valid in enumerate(camera_mask) if is_valid],
            dtype=torch.long,
            device=teacher_device,
        )
        valid_images = images[sample_indices[:, None], valid_views[None, :]]
        autocast_enabled = teacher_device.type == "cuda"
        with torch.autocast(
            device_type=teacher_device.type,
            dtype=torch.bfloat16 if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            features, patch_start_idx = teacher.aggregator(valid_images)
        feature = _select_feature_layer(features, target_layer)
        feature = feature[:, :, patch_start_idx:, :]
        source_h, source_w = _factor_grid(feature.shape[2])
        if source_h * source_w != feature.shape[2]:
            raise RuntimeError("VGGT patch feature count cannot be represented as a 2-D grid")
        if target_dim is None:
            target_dim = feature.shape[-1]
        elif feature.shape[-1] != target_dim:
            raise ValueError("VGGT returned inconsistent feature dimensions across camera-mask groups")
        feature = feature.reshape(
            len(sample_indices) * valid_views.numel(), source_h, source_w, target_dim
        ).permute(0, 3, 1, 2)
        if use_pe:
            feature = feature + _position_grid_to_embed(
                source_h, source_w, target_dim, feature.device, feature.dtype
            ).permute(2, 0, 1).unsqueeze(0) * pe_scale
        feature = F.interpolate(feature.float(), size=(vla_h, vla_w), mode="bilinear", align_corners=True)
        feature = feature.reshape(len(sample_indices), valid_views.numel(), target_dim, vla_h, vla_w)
        feature = feature.permute(0, 1, 3, 4, 2)
        if target_by_image is None:
            target_by_image = feature.new_zeros(
                batch, num_images, vla_h, vla_w, target_dim
            )
        target_by_image[sample_indices[:, None], valid_views[None, :]] = feature
        del features, feature

    feature = target_by_image.reshape(batch, num_images * vla_h * vla_w, target_dim)
    token_mask = img_masks.unsqueeze(-1).expand(-1, -1, vla_h * vla_w).reshape(batch, -1)
    output_dtype = torch.bfloat16 if bool(_cfg_get(config, "target_bfloat16", True)) else torch.float32
    return feature.to(dtype=output_dtype), token_mask


def spatial_forcing_cosine_loss(
    projection: nn.Module,
    normalization: nn.Module,
    vla_hidden: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    spatial_targets: torch.Tensor,
    spatial_target_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked per-patch cosine alignment for a captured VLA layer."""
    if vla_hidden.ndim != 3 or visual_pos_masks.shape != vla_hidden.shape[:2]:
        raise ValueError(
            "vla_hidden and visual_pos_masks must have shapes [B,L,D] and [B,L], "
            f"got {tuple(vla_hidden.shape)} and {tuple(visual_pos_masks.shape)}"
        )
    if spatial_targets.ndim != 3 or spatial_target_mask.shape != spatial_targets.shape[:2]:
        raise ValueError(
            "spatial_targets and spatial_target_mask must have shapes [B,N,D] and [B,N], "
            f"got {tuple(spatial_targets.shape)} and {tuple(spatial_target_mask.shape)}"
        )
    if spatial_targets.shape[0] != vla_hidden.shape[0]:
        raise ValueError("VLA hidden states and Spatial Forcing targets have different batch sizes")

    predictions = []
    targets = []
    for batch_idx in range(vla_hidden.shape[0]):
        visual_tokens = vla_hidden[batch_idx][visual_pos_masks[batch_idx].bool()]
        target_tokens = spatial_targets[batch_idx][spatial_target_mask[batch_idx].bool()]
        if visual_tokens.shape[0] != target_tokens.shape[0]:
            raise ValueError(
                f"Spatial Forcing token count mismatch for sample {batch_idx}: "
                f"VLA={visual_tokens.shape[0]}, VGGT={target_tokens.shape[0]}"
            )
        if visual_tokens.numel():
            predictions.append(visual_tokens)
            targets.append(target_tokens)
    if not predictions:
        raise ValueError("Spatial Forcing batch contains no valid image patches")

    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0).detach().to(device=prediction.device)
    prediction = projection(normalization(prediction))
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return 1.0 - (prediction * target).sum(dim=-1).mean()


__all__ = [
    "build_spatial_forcing_teacher",
    "get_spatial_forcing_target",
    "spatial_forcing_cosine_loss",
]
