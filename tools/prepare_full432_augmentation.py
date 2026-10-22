"""Generate a matched full432 control and augmentation recipe; never start jobs."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.utils.observation_augmentation import AugmentationConfig, parse_config, config_hash


def recipes(source, run_root, augmentation=None):
    """Use original initialization and preserve every optimizer/model/data setting."""
    data, train = source["data"], source["train"]
    required = (
        (data, dict(img_size=384, image_resize_mode="opencv_lanczos4", preserve_image_resolution=True,
                    num_workers=8, prefetch_factor=4, pin_memory=True)),
        (train, dict(global_batch_size=128, micro_batch_size=4, gradient_accumulation_steps=4,
                     max_steps=10000, save_steps=1000, seed=42, optimizer="muon", use_compile=True,
                     enable_fp32=True, enable_mixed_precision=True, action_fp32=False)),
    )
    for section, expected in required:
        for key, value in expected.items():
            if section.get(key) != value:
                raise ValueError(f"Source must explicitly retain {key}={value}; do not silently change controls")
    if data["cameras"] != ["camera_top", "camera_wrist_left", "camera_wrist_right"]:
        raise ValueError("Keep the same three original cameras")
    if train.get("align_params", {}).get("llm", {}).get("image_token_size") != 12:
        raise ValueError("Full432 requires the existing 12x12 student grid")
    if any(part.startswith("global_step_") for part in PurePosixPath(source["model"]["model_path"].replace("\\", "/")).parts):
        raise ValueError("Use the original 6B initialization, not trained full/ROI/smoke checkpoints")
    resolved = asdict(parse_config(augmentation)) if augmentation is not None else asdict(AugmentationConfig())
    root = PurePosixPath(run_root) if str(run_root).startswith("/") else Path(run_root)
    result = {}
    for name, enabled in (("full432_control", False), ("full432_aug", True)):
        cfg = copy.deepcopy(source)
        cfg["data"].update(roi_config={}, image_augment=enabled)
        cfg["data"].pop("roi_geometry_root", None)
        cfg["data"].pop("image_augmentation_config", None)
        if enabled:
            cfg["data"]["image_augmentation_config"] = copy.deepcopy(resolved)
        cfg["train"].update(output_dir=str(root/name), enable_resume=False, pause_after_steps=5000,
                                allow_dataloader_worker_change=False)
        cfg["train"].pop("load_checkpoint_path", None)
        if "wandb_name" in cfg["train"]:
            cfg["train"]["wandb_name"] = name
        for section in cfg.values():
            if isinstance(section, dict):
                for key in [k for k, v in section.items() if v is None]:
                    del section[key]
        result[name] = cfg
    return result


def prepare(base, destination, run_root, augmentation=None):
    base, destination = Path(base), Path(destination)
    source = yaml.safe_load(base.read_text(encoding="utf-8"))
    configs = recipes(source, str(run_root), augmentation)
    paths = {name: destination / f"{name}.yaml" for name in configs}
    report_path = destination / "augmentation_preparation.json"
    if report_path.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("Use a fresh config destination; existing experiments are preserved")
    if any(Path(cfg["train"]["output_dir"]).exists() for cfg in configs.values()):
        raise FileExistsError("Choose fresh output directories; never reuse completed ROI/full runs")
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        path.write_text(yaml.safe_dump(configs[name], sort_keys=False), encoding="utf-8")
    report = dict(base=str(base), source_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                  configs={k: str(v) for k, v in paths.items()}, layout="full_view_384", visual_patch_tokens=432,
                  augmentation_sha256=config_hash(configs["full432_aug"]["data"]["image_augmentation_config"]),
                  initialized_from=source["model"]["model_path"], training_launched=False,
                  foreground_detection="No detector bundled; missing masks skip occlusion",
                  remote_paths_and_clean_norm_revalidated=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Actual saved full384/ROI YAML with original initialization")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--recipe", help="Optional YAML mapping of augmentation options only")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8")) if args.recipe else None
    print(json.dumps(prepare(args.base, args.destination, args.run_root, config), ensure_ascii=True))


if __name__ == "__main__":
    main()
