"""Opt-in, fixed-budget RoboTwin ROI canvases; no model/teacher changes.

All geometry is current-frame OpenCV geometry in metres/radians. The serialized
kinematic chains come from the actual URDF-loaded articulation (see
tools/export_robotwin_roi_profile.py), not fitted end-effector labels.
"""
from collections import OrderedDict
import hashlib
import json
from functools import wraps
from pathlib import Path

import numpy as np

from .image_resize import resize_rgb_uint8


IMAGE_KEYS = ("observation.images.cam_high", "observation.images.cam_left_wrist",
              "observation.images.cam_right_wrist")
GEOMETRY_KEY = "roi_geometry"


class ROIGeometryError(ValueError):
    """Fatal data/configuration error: never replace this sample at random."""


def geometry_errors(function):
    """Keep malformed ROI inputs out of MultiVLADataset's random-sample retry."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ROIGeometryError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            raise ROIGeometryError(f"{function.__name__}: malformed ROI input: {error}") from error
    return checked


def check_roi_identity(model_config, roi_config):
    """A deployed/resumed checkpoint must retain its exact input representation."""
    expected = canonical_hash(roi_config) if roi_config else None
    if model_config.get("roi_config_sha256") != expected:
        raise ROIGeometryError("Checkpoint ROI preprocessing identity does not match the data configuration")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ROIGeometryError(f"{name}: expected finite shape {shape}, got {result.shape}")
    return result


def transform(value, name):
    result = array(value, (4, 4), name)
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ROIGeometryError(f"{name}: expected a rigid transform")
    return result


class DualArmFK:
    """Serial SAPIEN joint frames: parent_T_joint Rx(q) inverse(child_T_joint).

    The tip is the joint's moving child-side frame, matching RoboTwin's
    ee.global_pose. Tool offsets include RoboTwin's gripper bias and rotations.
    State is RoboTwin's current drive target, NOT measured joint qpos. The profile
    exporter verifies FK against SAPIEN with qpos set explicitly; raw endpose is
    only a diagnostic of control tracking/projection error.
    """
    @geometry_errors
    def __init__(self, profile):
        if profile.get("schema_version") != 1 or profile.get("state_dim") != 14:
            raise ROIGeometryError("Expected version 1 RoboTwin 14-state FK profile")
        if profile.get("state_semantics") != "robotwin_drive_target":
            raise ROIGeometryError("Profile must explicitly declare robotwin_drive_target state semantics")
        self.profile = profile
        self.sha256 = canonical_hash(profile)
        self.arms = []
        for side, expected in (("left", list(range(6))), ("right", list(range(7, 13)))):
            arm = profile["arms"][side]
            joints, indices = [], []
            for joint in arm["chain"]:
                kind = joint["type"]
                if kind not in ("fixed", "revolute", "prismatic"):
                    raise ROIGeometryError(f"Unsupported FK joint type: {kind}")
                index = joint.get("state_index")
                if kind != "fixed":
                    if not isinstance(index, int) or index not in expected:
                        raise ROIGeometryError(f"{side}: unmapped/mismatched arm joint")
                    indices.append(index)
                joints.append((kind, index, transform(joint["parent_from_joint"], "parent"),
                               np.linalg.inv(transform(joint["child_from_joint"], "child"))))
            if sorted(indices) != expected:
                raise ROIGeometryError(f"{side}: chain must contain each of its six joints once")
            self.arms.append((joints, transform(arm["parent_from_tip"], "tip"),
                              array(arm["tip_offset"], (3,), "tip_offset"),
                              transform(arm["world_from_base"], "world_from_base")))

    def positions(self, state, world_from_bases=None):
        state = array(state, (14,), "current raw joint state")
        if world_from_bases is not None:
            array(world_from_bases, (2, 4, 4), "world_from_bases")
        points = []
        for side, (joints, tip, offset, default_base) in enumerate(self.arms):
            pose = (default_base.copy() if world_from_bases is None
                    else transform(world_from_bases[side], "world_from_base").copy())
            for kind, index, parent, child_inverse in joints:
                motion = np.eye(4)
                if kind == "revolute":
                    c, s = np.cos(state[index]), np.sin(state[index])
                    motion[1:3, 1:3] = [[c, -s], [s, c]]
                elif kind == "prismatic":
                    motion[0, 3] = state[index]
                pose = pose @ parent @ motion @ child_inverse
            pose = pose @ tip
            points.append(pose[:3, :3] @ offset + pose[:3, 3])
        return np.stack(points)


@geometry_errors
def project_points(points, geometry, output_hw):
    """K is calibrated at image_hw, E maps world to OpenCV camera coordinates."""
    points = array(points, (2, 3), "end effectors")
    intr = array(geometry["intrinsic_cv"], (3, 3), "intrinsic_cv")
    extr = np.asarray(geometry["extrinsic_cv"], dtype=np.float64)
    if extr.shape == (3, 4):
        extr = np.vstack((extr, [0, 0, 0, 1]))
    extr = transform(extr, "extrinsic_cv")
    source_hw = array(geometry["image_hw"], (2,), "image_hw")
    target_hw = array(output_hw, (2,), "output_hw")
    if (source_hw <= 0).any() or (target_hw <= 0).any():
        raise ROIGeometryError("Image dimensions must be positive")
    if intr[0, 0] <= 0 or intr[1, 1] <= 0 or not np.allclose(intr[2], [0, 0, 1]):
        raise ROIGeometryError("Invalid OpenCV intrinsic matrix")
    pc = points @ extr[:3, :3].T + extr[:3, 3]
    projected = pc @ intr.T
    front = pc[:, 2] > 1e-3
    uv = np.full((2, 2), -1.0)
    uv[front] = projected[front, :2] / projected[front, 2:3]
    # Source images may have been resized during dataset conversion. Do not use
    # the old 320x240 constant; scale from the explicitly recorded calibration.
    uv[front] *= (target_hw / source_hw)[::-1]
    valid = front & (uv[:, 0] >= 0) & (uv[:, 0] < target_hw[1])
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] < target_hw[0])
    return uv, valid


def letterbox(image, size, fill=0):
    """Preserve the whole wrist field of view and aspect ratio."""
    height, width = size
    scale = min(height / image.shape[0], width / image.shape[1])
    h = max(1, min(height, int(round(image.shape[0] * scale))))
    w = max(1, min(width, int(round(image.shape[1] * scale))))
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    y, x = (height - h) // 2, (width - w) // 2
    canvas[y:y+h, x:x+w] = resize_rgb_uint8(image, (h, w))
    return canvas


def crop_with_padding(image, center, crop_hw, fill=0):
    """Keep the requested centre and size at edges; never silently shift the ROI."""
    h, w = crop_hw
    x, y = int(np.floor(center[0] - w / 2)), int(np.floor(center[1] - h / 2))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image.shape[1], x+w), min(image.shape[0], y+h)
    result = np.full((h, w, 3), fill, dtype=np.uint8)
    visible = max(0, x1-x0) * max(0, y1-y0)
    if visible:
        result[y0-y:y1-y, x0-x:x1-x] = image[y0:y1, x0:x1]
    return result, [x, y, x+w, y+h], 1 - visible / (h*w)


class ROICanvas:
    @geometry_errors
    def __init__(self, config):
        allowed = {"schema_version", "mode", "image_size", "crop_height_fraction", "fixed_centers", "profile"}
        if set(config) - allowed:
            raise ROIGeometryError(f"Unknown roi_config keys: {sorted(set(config)-allowed)}")
        if config.get("schema_version") != 1:
            raise ROIGeometryError("roi_config.schema_version must be 1")
        self.mode = config.get("mode")
        if self.mode not in ("ee", "fixed"):
            raise ROIGeometryError("ROI mode must be ee or fixed")
        self.size = int(config.get("image_size", 384))
        if self.size != 384:
            raise ROIGeometryError("This experiment is ROI-432: image_size must be 384")
        self.fraction = float(config.get("crop_height_fraction", 0.5))
        if not 0 < self.fraction <= 1:
            raise ROIGeometryError("crop_height_fraction must be in (0,1]")
        self.fixed = array(config.get("fixed_centers", [[0.35, 0.5], [0.65, 0.5]]),
                           (2, 2), "fixed_centers")
        if (self.fixed < 0).any() or (self.fixed >= 1).any():
            raise ROIGeometryError("fixed_centers must lie in normalized [0,1)")
        self.fk = DualArmFK(config["profile"])

    @geometry_errors
    def build(self, images, state, geometry):
        if len(images) != 3:
            raise ROIGeometryError("ROI-432 requires head, left wrist, right wrist")
        for image in images:
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ROIGeometryError("ROI expects original HWC uint8 RGB")
        head = images[0]
        height, width = head.shape[:2]
        if geometry.get("profile_sha256") != self.fk.sha256:
            raise ROIGeometryError("Geometry was prepared for a different robot profile")
        # Check even the fixed-crop control so both arms see the same eligible
        # dataset. A missing calibration is an error, not an invisible fallback.
        positions = self.fk.positions(state, geometry.get("world_from_bases"))
        uv, valid = project_points(positions, geometry, (height, width))
        crop_h = max(2, int(round(height * self.fraction / 2)) * 2)
        crop_w = crop_h // 2  # same 2:1 H:W as a half canvas, no crop distortion
        output = [resize_rgb_uint8(head, (self.size, self.size))]
        diagnostics = []
        for side in range(2):
            fallback = self.mode == "ee" and not valid[side]
            center = (self.fixed[side] * [width, height]
                      if self.mode == "fixed" or fallback else uv[side])
            crop, box, padding = crop_with_padding(head, center, (crop_h, crop_w))
            canvas = np.concatenate((
                letterbox(images[side+1], (self.size, self.size//2)),
                resize_rgb_uint8(crop, (self.size, self.size//2))), axis=1)
            output.append(canvas)
            diagnostics.append(dict(projected_uv=uv[side].tolist(),
                                    projection_valid=bool(valid[side]), fallback=bool(fallback),
                                    crop_xyxy=box, padding_fraction=padding))
        return output, diagnostics

    @geometry_errors
    def prepare_sample(self, item):
        """Shared training/serving entry: CPU CHW or current/future TCHW tensors.

        Only frame 0 is a policy canvas. Future images keep the baseline full
        views; their states are never used to choose current ROI coordinates.
        """
        import torch

        if GEOMETRY_KEY not in item:
            raise ROIGeometryError("ROI requires current-frame roi_geometry")
        state = item["observation.state"]
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        if np.asarray(state).ndim == 2:
            state = state[0]
        originals, current = [], []
        for key in IMAGE_KEYS:
            value = item[key]
            if not torch.is_tensor(value) or value.device.type != "cpu" or value.ndim not in (3, 4):
                raise ROIGeometryError(f"{key}: expected raw CPU CHW/TCHW tensor")
            frames = value.unsqueeze(0) if value.ndim == 3 else value
            if frames.shape[1] != 3 or min(frames.shape) <= 0:
                raise ROIGeometryError(f"{key}: expected RGB channels")
            if frames.is_floating_point():
                if not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
                    raise ROIGeometryError("Floating RGB must be in [0,1]")
                frames = (frames.float() * 255).round().to(torch.uint8)
            elif frames.dtype != torch.uint8:
                raise ROIGeometryError("RGB must be uint8 or float [0,1]")
            hwc = frames.permute(0, 2, 3, 1).contiguous().numpy()
            originals.append((value, hwc))
            current.append(hwc[0])
        canvas, diagnostics = self.build(current, state, item[GEOMETRY_KEY])
        result = dict(item)
        for key, first, (value, frames) in zip(IMAGE_KEYS, canvas, originals):
            converted = np.stack([first] + [resize_rgb_uint8(x, (self.size, self.size))
                                           for x in frames[1:]])
            tensor = torch.from_numpy(converted).permute(0, 3, 1, 2).contiguous()
            if value.is_floating_point():
                tensor = tensor.to(value.dtype) / 255
            result[key] = tensor[0] if value.ndim == 3 else tensor
        return result, diagnostics


def build_roi_processor(data_config, model_config, processor):
    config = getattr(data_config, "roi_config", None)
    if not config:
        return None
    if (getattr(data_config, "img_size", None) != 384
            or getattr(data_config, "image_resize_mode", None) != "opencv_lanczos4"
            or not getattr(data_config, "preserve_image_resolution", False)
            or list(data_config.cameras) != ["camera_top", "camera_wrist_left", "camera_wrist_right"]):
        raise ROIGeometryError("ROI-432 requires 384/Lanczos4/fixed grid and the original 3 camera slots")
    ip = processor.image_processor if processor is not None else None
    if ip is None or ip.patch_size != 16 or ip.merge_size != 2:
        raise ROIGeometryError("ROI-432 requires the actual Qwen processor with patch16/merge2")
    if not getattr(model_config, "return_image_grid_thw", False):
        raise ROIGeometryError("ROI requires actual image_grid_thw")
    align = getattr(model_config, "align_params", None) or {}
    if align and (align.get("llm", {}).get("image_token_size") != 12
                  or align.get("video", {}).get("num_future_frames", 1) != 1):
        raise ROIGeometryError("Keep the head 12x12 grid and one future teacher frame")
    return ROICanvas(config)


class EpisodeGeometry:
    """Read verified per-episode sidecars; fail on mismatched state/frame/profile."""
    def __init__(self, root, repo_id, profile_sha256):
        self.directory = Path(root) / Path(repo_id).name
        self.profile_sha256 = profile_sha256
        self.cache = OrderedDict()
        try:
            manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
            if not manifest["complete"] or manifest["profile_sha256"] != profile_sha256:
                raise ROIGeometryError("Incomplete or mismatched geometry manifest")
            entry = manifest["datasets"][Path(repo_id).name]
            if Path(entry["source"]).resolve() != Path(repo_id).resolve():
                raise ROIGeometryError("Geometry manifest belongs to a different dataset path")
            self.episodes = entry["episodes"]
        except (OSError, KeyError, ValueError) as error:
            raise ROIGeometryError(f"Cannot validate geometry manifest: {error}") from error

    @geometry_errors
    def get(self, episode, frame, state):
        if int(episode) != episode or int(frame) != frame or episode < 0:
            raise ROIGeometryError("Episode/frame indices must be nonnegative integers")
        path = self.directory / f"episode_{int(episode):06d}.npz"
        if path not in self.cache:
            try:
                report = self.episodes[str(int(episode))]
                if hashlib.sha256(path.read_bytes()).hexdigest() != report["sidecar_sha256"]:
                    raise ROIGeometryError(f"ROI sidecar digest mismatch: {path}")
                with np.load(path, allow_pickle=False) as data:
                    record = {k: data[k] for k in data.files}
                if (str(record["profile_sha256"].item()) != self.profile_sha256
                        or int(record["schema_version"]) != 1
                        or not bool(record["validated"])):
                    raise ROIGeometryError(f"Unvalidated/mismatched ROI sidecar: {path}")
                count = report["frames"]
                array(record["state"], (count, 14), "sidecar states")
                array(record["intrinsic_cv"], (count, 3, 3), "sidecar intrinsics")
                if record["extrinsic_cv"].shape not in ((count, 3, 4), (count, 4, 4)):
                    raise ROIGeometryError("Sidecar extrinsics must have one transform per frame")
                array(record["image_hw"], (2,), "sidecar image dimensions")
                self.cache[path] = record
                while len(self.cache) > 2:
                    self.cache.popitem(last=False)
            except (OSError, KeyError, ValueError) as error:
                raise ROIGeometryError(f"Cannot read ROI geometry {path}: {error}") from error
        record = self.cache[path]
        self.cache.move_to_end(path)
        frame = int(frame)
        state = np.asarray(state)
        if state.ndim == 2:
            state = state[0]
        if (not 0 <= frame < len(record["state"])
                or not np.allclose(record["state"][frame], array(state, (14,), "state"), atol=1e-5, rtol=0)):
            raise ROIGeometryError(f"ROI frame/state mismatch: {path}, frame {frame}")
        return dict(intrinsic_cv=record["intrinsic_cv"][frame],
                    extrinsic_cv=record["extrinsic_cv"][frame],
                    image_hw=record["image_hw"], profile_sha256=self.profile_sha256)


def robotwin_geometry(env, observation):
    """Metadata only: no endpose, object labels or future action are sent."""
    camera = observation["observation"]["head_camera"]
    robot = env.robot
    profile = getattr(robot, "_lingbot_roi_profile", None)
    if profile is None:
        from .roi_profile import export_robotwin_profile
        profile = export_robotwin_profile(robot)
        robot._lingbot_roi_profile = profile
    return dict(intrinsic_cv=np.asarray(camera["intrinsic_cv"]),
                extrinsic_cv=np.asarray(camera["extrinsic_cv"]),
                image_hw=list(camera["rgb"].shape[:2]),
                profile_sha256=canonical_hash(profile),
                world_from_bases=np.stack([
                    robot.left_entity.get_root_pose().to_transformation_matrix(),
                    robot.right_entity.get_root_pose().to_transformation_matrix()]))
