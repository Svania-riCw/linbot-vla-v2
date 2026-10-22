"""Align current drive-target states with raw recordings; publish camera sidecars.

Endpose is used ONLY as a validation reference, never as a policy input/crop centre.
The source LeRobot datasets and HDF5 recordings are read-only.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.utils.ee_roi import DualArmFK, project_points


def read_dataset_list(path):
    datasets = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] != "robotwin":
            raise ValueError("Expected 'robotwin /absolute/path/robotwin_clean_<task>' per line")
        directory = Path(parts[1])
        if not directory.name.startswith("robotwin_clean_") or not directory.is_absolute():
            raise ValueError(f"Expected an absolute clean dataset path: {directory}")
        datasets.append(directory)
    if not datasets or len({p.name for p in datasets}) != len(datasets):
        raise ValueError("Empty or duplicate dataset list")
    return datasets


def lerobot_states(directory):
    import pyarrow.parquet as pq
    frames = defaultdict(list)
    for path in sorted((directory / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for ep, frame, state in zip(table["episode_index"], table["frame_index"], table["observation.state"]):
            frames[int(ep)].append((int(frame), state))
    if not frames:
        raise ValueError(f"No Parquet states under {directory}")
    result = {}
    for ep, rows in frames.items():
        rows.sort(key=lambda row: row[0])
        if ep < 0 or [r[0] for r in rows] != list(range(len(rows))):
            raise ValueError(f"Duplicate/missing frame index: {directory}, episode {ep}")
        result[ep] = np.asarray([r[1] for r in rows], dtype=np.float64)
    return result


def image_hw(frame):
    """Decode just the dimensions of raw HWC or encoded JPEG/PNG RGB."""
    import cv2
    value = np.asarray(frame)
    if value.ndim == 3 and value.shape[-1] == 3:
        return value.shape[:2]
    encoded = (np.frombuffer(frame, np.uint8) if isinstance(frame, (bytes, np.bytes_))
               else value.astype(np.uint8, copy=False).reshape(-1))
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Cannot decode raw camera RGB; calibration size must not be guessed")
    return decoded.shape[:2]


def validate_episode(fk, states, raw):
    states = np.asarray(states, dtype=np.float64)
    joints = np.asarray(raw["joint_action/vector"], dtype=np.float64)
    if (states.ndim != 2 or states.shape[1] != 14 or joints.shape != (len(states)+1, 14)
            or not np.isfinite(states).all()
            or not np.allclose(states, joints[:-1], atol=1e-5, rtol=0)):
        raise ValueError("Raw/LeRobot current state mismatch or T_lerobot != T_raw-1")
    count = len(states)
    intr = np.asarray(raw["observation/head_camera/intrinsic_cv"])
    extr = np.asarray(raw["observation/head_camera/extrinsic_cv"])
    if intr.shape != (count+1, 3, 3) or extr.shape not in ((count+1, 3, 4), (count+1, 4, 4)):
        raise ValueError("Expected per-frame raw camera calibration aligned with joint_action")
    rgb = raw["observation/head_camera/rgb"]
    if len(rgb) != count+1:
        raise ValueError("Raw RGB length does not match calibration/state")
    hw = image_hw(rgb[0])
    # Resolution may not be silently inferred from historical conversion constants.
    for index in sorted({count//2, count}):
        if image_hw(rgb[index]) != hw:
            raise ValueError("Variable raw image dimensions require a different sidecar schema")
    reference = np.stack([np.asarray(raw[f"endpose/{side}_endpose"])[:-1, :3]
                          for side in ("left", "right")], axis=1)
    if reference.shape != (count, 2, 3) or not np.isfinite(reference).all():
        raise ValueError("Missing or malformed raw endpose validation reference")
    errors, pixel_errors = [], []
    valid_counts, covered_counts = np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)
    for t, state in enumerate(states):
        xyz = fk.positions(state)
        errors.extend(np.linalg.norm(xyz-reference[t], axis=-1).tolist())
        geometry = dict(intrinsic_cv=intr[t], extrinsic_cv=extr[t], image_hw=hw)
        uv, valid = project_points(xyz, geometry, hw)
        actual_uv, actual_valid = project_points(reference[t], geometry, hw)
        valid_counts += valid
        both = valid & actual_valid
        pixel_errors.extend(np.linalg.norm(uv[both]-actual_uv[both], axis=-1).tolist())
        # Diagnostics for the default crop only. Endpose never selects the crop.
        centers = np.where(valid[:, None], uv, np.array([[.35, .5], [.65, .5]]) * hw[::-1])
        crop_h = max(2, int(round(hw[0] * .5 / 2)) * 2)
        covered_counts += actual_valid & (abs(actual_uv[:, 0]-centers[:, 0]) < crop_h/4) & (
            abs(actual_uv[:, 1]-centers[:, 1]) < crop_h/2)
    record = dict(schema_version=np.array(1), validated=np.array(True),
                  profile_sha256=np.array(fk.sha256), state=states,
                  intrinsic_cv=intr[:-1], extrinsic_cv=extr[:-1], image_hw=np.array(hw))
    report = dict(frames=count, drive_target_to_recorded_ee_max_m=max(errors),
                  drive_target_to_recorded_ee_mean_m=float(np.mean(errors)),
                  projection_error_px_p95=(float(np.percentile(pixel_errors, 95)) if pixel_errors else None),
                  actual_ee_in_default_crop_frames=covered_counts.tolist(),
                  valid_projection_frames=valid_counts.tolist(),
                  fallback_frames=(count-valid_counts).tolist())
    return record, report


def prepare(profile_path, dataset_list, raw_root, output):
    import h5py
    profile_path, output = Path(profile_path), Path(output)
    fk = DualArmFK(json.loads(profile_path.read_text(encoding="utf-8")))
    validation = json.loads(profile_path.with_suffix(".validation.json").read_text(encoding="utf-8"))
    if validation["profile_sha256"] != fk.sha256 or validation["max_error_m"] > 1e-5:
        raise ValueError("Missing/mismatched SAPIEN FK validation")
    datasets = read_dataset_list(dataset_list)
    # A new directory prevents merging stale camera files from a previous preparation.
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, complete=False, profile_sha256=fk.sha256,
                    dataset_list_sha256=hashlib.sha256(Path(dataset_list).read_bytes()).hexdigest(),
                    raw_root=str(Path(raw_root).resolve()), datasets={})
    try:
        for directory in datasets:
            task = directory.name.removeprefix("robotwin_clean_")
            dest = output / directory.name
            dest.mkdir()
            manifest["datasets"][directory.name] = dict(source=str(directory), episodes={})
            for episode, states in sorted(lerobot_states(directory).items()):
                source = Path(raw_root) / task / "aloha-agilex_clean_50" / "data" / f"episode{episode}.hdf5"
                with h5py.File(source, "r") as raw:
                    record, report = validate_episode(fk, states, raw)
                target = dest / f"episode_{episode:06d}.npz"
                with target.open("xb") as stream:
                    np.savez_compressed(stream, **record)
                report.update(sidecar_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                              source_hdf5=str(source))
                manifest["datasets"][directory.name]["episodes"][str(episode)] = report
            print(json.dumps(dict(dataset=directory.name,
                                  episodes=len(manifest["datasets"][directory.name]["episodes"])),
                             ensure_ascii=True), flush=True)
        manifest["complete"] = True
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dataset-list", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.profile, args.dataset_list, args.raw_root, args.output)
    print(json.dumps(dict(complete=result["complete"], datasets=len(result["datasets"])), ensure_ascii=True))


if __name__ == "__main__":
    main()
