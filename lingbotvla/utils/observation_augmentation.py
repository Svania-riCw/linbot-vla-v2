"""Training-only RGB perturbations; no scene editing or action/state transforms.

The user-supplied recipe fixes probabilities and some ranges. Remaining strengths
are explicit implementation choices below, not claimed to reproduce its authors.
OpenCV runs on CPU. A replayable seed comes from the DataLoader worker's torch RNG.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import cv2
import numpy as np


class AugmentationError(ValueError):
    """Invalid image/mask/recipe must not trigger random dataset replacement."""


@dataclass(frozen=True)
class AugmentationConfig:
    schema_version: int = 1
    keep_original_probability: float = 0.10
    color_probability: float = 0.80
    brightness: tuple = (0.25, 1.10)
    contrast: tuple = (0.50, 1.50)
    saturation: tuple = (0.40, 1.60)
    temperature: tuple = (-0.10, 0.10)  # log red/blue gain, NOT kelvin
    affine_probability: float = 0.50
    scale: tuple = (0.90, 1.10)
    translation_fraction: float = 0.06
    occlusion_probability: float = 0.95
    occlusion_area: tuple = (0.02, 0.10)
    occlusion_aspect: tuple = (0.50, 2.00)
    occlusion_attempts: int = 20
    foreground_margin_pixels: int = 4
    missing_foreground: str = "skip"  # alternatively error, never unprotected erase
    gaussian_noise_probability: float = 0.50
    gaussian_noise_std: tuple = (0.005, 0.030)  # relative to [0,1]
    salt_pepper_probability: float = 0.15
    salt_pepper_fraction: tuple = (0.001, 0.010)
    gaussian_blur_probability: float = 0.25
    gaussian_blur_sigma: tuple = (0.10, 1.50)
    motion_blur_probability: float = 0.10
    motion_blur_sizes: tuple = (3, 5, 7)
    jpeg_probability: float = 0.30
    jpeg_quality: tuple = (35, 90)

    def __post_init__(self):
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise AugmentationError("Unsupported image augmentation schema")
        for name, value in asdict(self).items():
            if name.endswith("_probability") and (not math.isfinite(value) or not 0 <= value <= 1):
                raise AugmentationError(f"{name} must be in [0,1]")
        ranges = ("brightness", "contrast", "saturation", "temperature", "scale", "occlusion_area",
                  "occlusion_aspect", "gaussian_noise_std", "salt_pepper_fraction",
                  "gaussian_blur_sigma", "jpeg_quality")
        for name in ranges:
            value = getattr(self, name)
            if len(value) != 2 or not all(math.isfinite(v) for v in value) or value[0] > value[1]:
                raise AugmentationError(f"Invalid range: {name}")
            if name != "temperature" and value[0] < 0:
                raise AugmentationError(f"Negative range: {name}")
        if self.scale[0] <= 0 or self.occlusion_aspect[0] <= 0 or self.gaussian_blur_sigma[0] <= 0:
            raise AugmentationError("Scale, aspect and blur sigma must be positive")
        if not 0 < self.occlusion_area[0] <= self.occlusion_area[1] <= 1:
            raise AugmentationError("Occlusion area must be a fraction in (0,1]")
        if not 0 <= self.salt_pepper_fraction[1] <= 1:
            raise AugmentationError("Salt/pepper fraction must be in [0,1]")
        if not math.isfinite(self.translation_fraction) or not 0 <= self.translation_fraction <= 1:
            raise AugmentationError("Translation must be a fraction in [0,1]")
        if not 1 <= self.jpeg_quality[0] <= self.jpeg_quality[1] <= 100 or any(
                int(v) != v for v in self.jpeg_quality):
            raise AugmentationError("JPEG quality must be integer [1,100]")
        if not self.motion_blur_sizes or any(int(k) != k or k < 3 or k % 2 != 1 for k in self.motion_blur_sizes):
            raise AugmentationError("Motion kernels must be odd integers >=3")
        for name, minimum in (("occlusion_attempts", 1), ("foreground_margin_pixels", 0)):
            value = getattr(self, name)
            if int(value) != value or value < minimum:
                raise AugmentationError(f"Invalid {name}")
        if self.missing_foreground not in ("skip", "error"):
            raise AugmentationError("missing_foreground must be skip or error")
        if self.gaussian_blur_probability + self.motion_blur_probability > 1:
            raise AugmentationError("Mutually exclusive blur probabilities must sum to <=1")


def parse_config(config):
    if not isinstance(config, dict) or not config:
        raise AugmentationError("Expected a nonempty image_augmentation_config mapping")
    unknown = set(config) - set(AugmentationConfig.__dataclass_fields__)
    if unknown:
        raise AugmentationError(f"Unknown image augmentation options: {sorted(unknown)}")
    return AugmentationConfig(**config)


def config_hash(config):
    resolved = asdict(parse_config(config)) if config else None
    return hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if resolved else None


def check_augmentation_identity(checkpoint_config, config):
    if checkpoint_config.get("image_augmentation_config_sha256") != config_hash(config):
        raise AugmentationError("DCP resume augmentation recipe differs from checkpoint; use the saved training config")


class ObservationAugmenter:
    """Shared parameters across cameras/time, per-view noise and protected rectangles.

    Input: RGB uint8 arrays [H,W,3] or [T,H,W,3]. Optional masks: HW or THW,
    bool / binary 0,1 / binary 0,255. True means protected foreground. A single
    HW mask is only accepted for a single image; clips require one mask per frame.
    No detector is guessed. Missing masks skip ONLY occlusion (or raise in strict
    mode). All-frame union protects a rectangle in every frame of a teacher clip.
    """
    def __init__(self, config):
        self.config = parse_config(config)

    def sample(self, seed):
        rng = np.random.default_rng(int(seed))
        c = self.config
        plan = dict(seed=int(seed), keep_original=bool(rng.random() < c.keep_original_probability))
        if plan["keep_original"]:
            return plan
        for name in ("color", "affine", "occlusion", "gaussian_noise", "salt_pepper", "jpeg"):
            plan[name] = bool(rng.random() < getattr(c, name + "_probability"))
        for name in ("brightness", "contrast", "saturation", "temperature", "scale",
                     "gaussian_noise_std", "salt_pepper_fraction", "gaussian_blur_sigma"):
            plan[name] = float(rng.uniform(*getattr(c, name)))
        plan["translation"] = rng.uniform(-c.translation_fraction, c.translation_fraction, 2).tolist()
        blur = rng.random()
        plan["blur"] = ("gaussian" if blur < c.gaussian_blur_probability else
                        "motion" if blur < c.gaussian_blur_probability + c.motion_blur_probability else "none")
        plan["motion_size"] = int(rng.choice(c.motion_blur_sizes))
        plan["motion_angle"] = float(rng.uniform(0, np.pi))
        plan["jpeg_quality"] = int(rng.integers(c.jpeg_quality[0], c.jpeg_quality[1] + 1))
        return plan

    @staticmethod
    def affine_matrix(height, width, plan):
        scale = plan["scale"]
        tx, ty = plan["translation"]
        return np.array([[scale, 0, (1-scale)*(width-1)/2 + tx*width],
                         [0, scale, (1-scale)*(height-1)/2 + ty*height]], dtype=np.float32)

    def _protected(self, mask, shape, matrix):
        if mask is None:
            return None
        mask = np.asarray(mask)
        frames, height, width = shape
        if mask.ndim == 2 and frames == 1:
            mask = mask[None]
        if mask.shape != (frames, height, width) or not np.isin(mask, (0, 1, 255)).all():
            raise AugmentationError("Foreground masks must be aligned binary THW arrays (one for EVERY frame)")
        masks = mask.astype(bool).astype(np.uint8)
        if matrix is not None:
            # Linear interpolation + any nonzero contribution is conservative at edges.
            masks = np.stack([cv2.warpAffine(m.astype(np.float32), matrix, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 0 for m in masks])
        protected = masks.any(axis=0).astype(np.uint8)
        margin = int(self.config.foreground_margin_pixels)
        if margin:
            protected = cv2.dilate(protected, np.ones((2*margin+1, 2*margin+1), np.uint8))
        return protected.astype(bool)

    def _rectangle(self, protected, rng):
        height, width = protected.shape
        c = self.config
        for _ in range(int(c.occlusion_attempts)):
            area = rng.uniform(*c.occlusion_area) * height * width
            aspect = np.exp(rng.uniform(np.log(c.occlusion_aspect[0]), np.log(c.occlusion_aspect[1])))
            rh, rw = max(1, round(np.sqrt(area/aspect))), max(1, round(np.sqrt(area*aspect)))
            if rh > height or rw > width:
                continue
            y, x = int(rng.integers(0, height-rh+1)), int(rng.integers(0, width-rw+1))
            if not protected[y:y+rh, x:x+rw].any():
                return (x, y, rw, rh)
        return None

    def apply(self, images, plan, foreground_masks=None):
        """Never mutate the input. Reports distinguish requested/applied occlusion."""
        if plan["keep_original"]:
            return dict(images), dict(keep_original=True, views={})
        masks = foreground_masks or {}
        if set(masks) - set(images):
            raise AugmentationError("Foreground mask camera keys do not match image keys")
        out, report = {}, dict(keep_original=False, views={})
        for key in sorted(images):
            value = np.asarray(images[key])
            if value.dtype != np.uint8 or value.ndim not in (3, 4) or value.shape[-1] != 3:
                raise AugmentationError("Augmentation expects HWC/THWC RGB uint8 images")
            clip = value[None] if value.ndim == 3 else value
            _, height, width, _ = clip.shape
            # Stable key hash avoids process-randomized Python hash() and view order effects.
            view_seed = int.from_bytes(hashlib.sha256(f"{plan['seed']}:{key}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(view_seed)
            matrix = self.affine_matrix(height, width, plan) if plan["affine"] else None
            protected = self._protected(masks.get(key), clip.shape[:3], matrix)
            rect = None
            status = "not_requested"
            if plan["occlusion"]:
                if protected is None:
                    if self.config.missing_foreground == "error":
                        raise AugmentationError(f"Missing foreground mask for {key}")
                    status = "skipped_missing_mask"
                else:
                    rect = self._rectangle(protected, rng)
                    status = "applied" if rect else "skipped_no_safe_rectangle"
            fill = str(rng.choice(["solid", "mosaic", "noise"]))
            solid = rng.integers(0, 256, (3,), dtype=np.uint8)
            texture = rng.integers(0, 256, (rect[3], rect[2], 3), dtype=np.uint8) if rect and fill == "noise" else None
            gaussian = rng.normal(0, plan["gaussian_noise_std"]*255, (height, width, 3)).astype(np.float32) if plan["gaussian_noise"] else None
            salt = rng.random((height, width)) if plan["salt_pepper"] else None
            frames = []
            for frame in clip:
                image = frame.astype(np.float32)
                if plan["color"]:
                    image *= plan["brightness"]
                    mean = image.mean(axis=(0, 1, 2), keepdims=True)
                    image = (image-mean)*plan["contrast"] + mean
                    gray = (image * np.array([.299, .587, .114], np.float32)).sum(-1, keepdims=True)
                    image = gray + (image-gray)*plan["saturation"]
                    image *= np.array([np.exp(plan["temperature"]), 1, np.exp(-plan["temperature"])], np.float32)
                image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
                if matrix is not None:
                    image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                if rect:
                    x, y, rw, rh = rect
                    if fill == "mosaic":
                        small = cv2.resize(image[y:y+rh, x:x+rw], (max(1, rw//8), max(1, rh//8)), interpolation=cv2.INTER_AREA)
                        pixels = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
                    else:
                        pixels = texture if fill == "noise" else solid
                    image[y:y+rh, x:x+rw] = pixels
                if gaussian is not None:
                    image = np.clip(np.rint(image.astype(np.float32) + gaussian), 0, 255).astype(np.uint8)
                if salt is not None:
                    p = plan["salt_pepper_fraction"]
                    image[salt < p/2] = 0
                    image[(salt >= p/2) & (salt < p)] = 255
                if plan["blur"] == "gaussian":
                    image = cv2.GaussianBlur(image, (0, 0), plan["gaussian_blur_sigma"])
                elif plan["blur"] == "motion":
                    k = plan["motion_size"]
                    kernel = np.zeros((k, k), np.float32)
                    radius = (k-1)/2
                    dx, dy = radius*np.cos(plan["motion_angle"]), radius*np.sin(plan["motion_angle"])
                    cv2.line(kernel, (round(radius-dx), round(radius-dy)),
                             (round(radius+dx), round(radius+dy)), 1.0, 1)
                    kernel /= kernel.sum()
                    image = cv2.filter2D(image, -1, kernel)
                if plan["jpeg"]:
                    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, plan["jpeg_quality"]])
                    if not ok:
                        raise AugmentationError("JPEG augmentation encoding failed")
                    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if decoded is None:
                        raise AugmentationError("JPEG augmentation decoding failed")
                    image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                frames.append(image)
            result = np.stack(frames)
            out[key] = result[0] if value.ndim == 3 else result
            report["views"][key] = dict(occlusion=status, rectangle=rect, fill=fill if rect else None)
        return out, report

    def apply_tensors(self, item, image_keys, seed=None):
        """FeatureTransform adapter; preserves CHW/TCHW dtype, state/actions/text.

        Optional raw item['augmentation_foreground_masks'] maps SOURCE camera keys
        to masks at the already resized image resolution. For video clips the mask
        order must match the selected current/future frames, not whole-episode order.
        """
        import torch

        if seed is None:
            seed = int(torch.randint(0, 2**63-1, ()).item())
        plan = self.sample(seed)
        if plan["keep_original"]:
            return item, dict(plan=plan, keep_original=True, views={})
        images, originals = {}, {}
        for key in image_keys:
            if key not in item:
                raise AugmentationError(f"Missing augmentation camera: {key}")
            image = item[key]
            if not torch.is_tensor(image) or image.device.type != "cpu" or image.ndim not in (3, 4) or image.shape[-3] != 3:
                raise AugmentationError("Training augmentation expects CPU RGB CHW/TCHW tensors")
            if image.is_floating_point():
                if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise AugmentationError("Floating RGB inputs must be finite [0,1]")
                pixels = (image.float()*255).round().clamp(0, 255).byte()
            elif image.dtype == torch.uint8:
                pixels = image
            else:
                raise AugmentationError("RGB dtype must be uint8 or float [0,1]")
            images[key] = pixels.movedim(-3, -1).contiguous().numpy()
            originals[key] = image
        raw_masks = item.get("augmentation_foreground_masks", {})
        if not isinstance(raw_masks, dict):
            raise AugmentationError("augmentation_foreground_masks must map source camera names to masks")
        masks = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in raw_masks.items()}
        try:
            result, report = self.apply(images, plan, masks)
        except cv2.error as exc:
            raise AugmentationError("Image augmentation failed; do not substitute a random sample") from exc
        updated = dict(item)
        for key, image in result.items():
            tensor = torch.from_numpy(image).movedim(-1, -3).contiguous()
            original = originals[key]
            updated[key] = tensor.to(original.dtype)/255 if original.is_floating_point() else tensor
        report["plan"] = plan
        return updated, report
