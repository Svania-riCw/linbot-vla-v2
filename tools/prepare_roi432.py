"""Derive matched full-view, fixed-crop and EE-ROI recipes from a verified 384 run.

Writes configs only. Does not submit jobs, acquire a claim, or change a checkpoint.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.utils.ee_roi import ROICanvas
from tools.prepare_roi_geometry import read_dataset_list


def recipes(source, roi, geometry, run_root):
    """Keep optimization/data content identical across the three input layouts."""
    result = {}
    for name in ("full432_matched", "roi432_fixed", "roi432_ee"):
        config = copy.deepcopy(source)
        config["train"].update(output_dir=str(Path(run_root)/name), enable_resume=False,
                               pause_after_steps=5000,
                               allow_dataloader_worker_change=False)
        config["train"].pop("load_checkpoint_path", None)
        if "wandb_name" in config["train"]:
            config["train"]["wandb_name"] = name
        if name != "full432_matched":
            config["data"].update(roi_config=copy.deepcopy(roi), roi_geometry_root=str(geometry))
            config["data"]["roi_config"]["mode"] = "ee" if name.endswith("_ee") else "fixed"
        # The existing CLI flattens scalar YAML null into the literal "null".
        # Omit optional unset arguments, matching prepare_resolution_finetune.py.
        for section in config.values():
            if isinstance(section, dict):
                for key in [key for key, value in section.items() if value is None]:
                    del section[key]
        result[name] = config
    return result


def prepare(base, profile, geometry, destination, run_root):
    source = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
    data, train = source["data"], source["train"]
    required_data = dict(img_size=384, image_resize_mode="opencv_lanczos4",
                         preserve_image_resolution=True, num_workers=8, prefetch_factor=4, pin_memory=True)
    required_train = dict(global_batch_size=128, micro_batch_size=4, gradient_accumulation_steps=4, max_steps=10000,
                          save_steps=1000, seed=42, enable_fp32=True)
    for section, required in ((data, required_data), (train, required_train)):
        for key, expected in required.items():
            if section.get(key) != expected:
                raise ValueError(f"Base recipe must explicitly specify {key}={expected}")
    if data.get("roi_config"):
        raise ValueError("Start with the full-view 384 recipe, not a previous ROI configuration")
    if data["cameras"] != ["camera_top", "camera_wrist_left", "camera_wrist_right"]:
        raise ValueError("Keep the original head/left/right camera order")
    if train.get("align_params", {}).get("llm", {}).get("image_token_size") != 12:
        raise ValueError("Preserve the 12x12 head student grid")
    norm = Path(data["norm_stats_file"])
    expected_norm = "bd1ada4579d9c3cd5106255f4a53385af2cd82999322b395d4204c28d85a29f6"
    if hashlib.sha256(norm.read_bytes()).hexdigest() != expected_norm:
        raise ValueError("ROI-432 requires the experiment's pinned clean normalization snapshot")
    mapping = yaml.safe_load((Path(data["robot_config_root"]) / "robotwin.yaml").read_text(encoding="utf-8"))
    if Path(mapping["norm_stats"]).resolve() != norm.resolve():
        raise ValueError("Actual robot mapping and data.norm_stats_file must use the same clean snapshot")
    profile = json.loads(Path(profile).read_text(encoding="utf-8"))
    roi = dict(schema_version=1, mode="ee", image_size=384, crop_height_fraction=0.5,
               fixed_centers=[[0.35, 0.5], [0.65, 0.5]], profile=profile)
    canvas = ROICanvas(roi)
    geometry = Path(geometry).resolve()
    manifest = json.loads((geometry / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest["profile_sha256"] != canvas.fk.sha256:
        raise ValueError("Geometry preparation must be complete and match the embedded FK profile")
    datasets = read_dataset_list(data["train_path"])
    if len(datasets) != 50 or set(manifest["datasets"]) != {p.name for p in datasets}:
        raise ValueError("Expected exactly the same 50 clean tasks and geometry dataset set")
    if manifest["dataset_list_sha256"] != hashlib.sha256(Path(data["train_path"]).read_bytes()).hexdigest():
        raise ValueError("Training dataset list differs from geometry preparation")
    for dataset in datasets:
        entry = manifest["datasets"][dataset.name]
        if Path(entry["source"]).resolve() != dataset.resolve() or len(entry["episodes"]) != 50:
            raise ValueError(f"Expected 50 verified episodes from {dataset}")
        for ep, report in entry["episodes"].items():
            path = geometry / dataset.name / f"episode_{int(ep):06d}.npz"
            if hashlib.sha256(path.read_bytes()).hexdigest() != report["sidecar_sha256"]:
                raise ValueError(f"Geometry sidecar changed: {path}")
    # Do not initialize ROI from the previously fine-tuned full-view run.
    init = Path(source["model"]["model_path"])
    if any(p.startswith("global_step_") or p == "hf_ckpt" for p in init.parts):
        raise ValueError("Use the same original 6B initialization as the 384 baseline")
    destination, run_root = Path(destination), Path(run_root).resolve()
    paths = [destination / f"{name}.yaml" for name in ("full432_matched", "roi432_fixed", "roi432_ee")]
    if any(p.exists() for p in paths):
        raise FileExistsError("Output config already exists")
    for path in paths:
        if (run_root / path.stem).exists():
            raise FileExistsError(f"Choose a fresh run directory: {run_root / path.stem}")
    destination.mkdir(parents=True, exist_ok=True)
    generated = recipes(source, roi, geometry, run_root)
    for path in paths:
        path.write_text(yaml.safe_dump(generated[path.stem], sort_keys=False), encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "profile", "geometry", "destination", "run-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    paths = prepare(args.base, args.profile, args.geometry, args.destination, args.run_root)
    print(json.dumps([str(p) for p in paths], ensure_ascii=True))


if __name__ == "__main__":
    main()
