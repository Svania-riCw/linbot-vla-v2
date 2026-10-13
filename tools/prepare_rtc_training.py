#!/usr/bin/env python3
"""Create a local RTC training config for the clean RoboTwin LeRobot data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT.parent


def _gpu_count() -> int:
    processes = os.environ.get("NPROC_PER_NODE")
    if processes is not None:
        count = int(processes)
        if count <= 0:
            raise ValueError("NPROC_PER_NODE must be positive")
        return count
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if not ids or ids == ["-1"]:
            raise ValueError("CUDA_VISIBLE_DEVICES does not select any GPUs")
        return len(ids)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return max(1, len([line for line in result.stdout.splitlines() if line.strip()]))
    except (OSError, subprocess.SubprocessError):
        return 1


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")
    return path.resolve()


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def prepare(root: Path, output: Path | None = None, source: Path | None = None) -> Path:
    root = root.resolve()
    source = (source or REPO_ROOT).resolve()
    base_weights = root / "checkpoints/lingbot-vla-v2-6b"
    qwen_weights = root / "checkpoints/Qwen3-VL-4B-Instruct"
    data_root = root / "training-data/lerobot"
    clean_stats = root / "training-data/robotwin_clean_norm_stats.json"
    dino_root = base_weights / "dino_video"
    depth_root = base_weights / "depth"
    use_teachers = _parse_bool(os.environ.get("RTC_USE_TEACHERS", "true"))

    _required_file(base_weights / "config.json", "Official LingBot-VLA 2.0 base config")
    _required_file(base_weights / "model.safetensors.index.json", "Base weight index")
    with (base_weights / "model.safetensors.index.json").open(encoding="utf-8") as f:
        index = json.load(f)
    for shard in set(index["weight_map"].values()):
        _required_file(base_weights / shard, "Base weight shard")
    _required_file(qwen_weights / "config.json", "Qwen3-VL tokenizer/model config")
    if use_teachers:
        moge_path = Path(
            os.environ.get("RTC_MOGE_PATH", root / "checkpoints/moge2-vitb-normal.pt")
        ).expanduser().resolve()
        _required_file(depth_root / "model.pt", "MoRGBD teacher checkpoint")
        _required_file(dino_root / "teacher_step_10000.pth", "DINO video teacher checkpoint")
        _required_file(dino_root / "config.yaml", "DINO video teacher config")
        _required_file(moge_path, "MoGe-2 ViT-B normal checkpoint")
    _required_file(clean_stats, "Clean RoboTwin normalization stats")

    recipe_path = source / "configs/vla/robotwin/robotwin.yaml"
    robot_config_path = source / "configs/robot_configs/robotwin.yaml"
    _required_file(recipe_path, "Official RoboTwin training recipe")
    _required_file(robot_config_path, "RoboTwin robot config")
    with recipe_path.open(encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    with robot_config_path.open(encoding="utf-8") as f:
        robot_config = yaml.safe_load(f)

    datasets = sorted(
        path.resolve()
        for path in data_root.iterdir()
        if path.is_dir() and (path / "meta/info.json").is_file()
    ) if data_root.is_dir() else []
    if not datasets:
        raise FileNotFoundError(
            f"No LeRobot datasets with meta/info.json found under {data_root}"
        )
    for dataset in datasets:
        if any(character.isspace() for character in str(dataset)):
            raise ValueError(f"Dataset paths with whitespace are unsupported: {dataset}")
        with (dataset / "meta/info.json").open(encoding="utf-8") as f:
            info = json.load(f)
        if info.get("codebase_version") not in {"v3.0", "v3.0.0"}:
            raise ValueError(
                f"Expected LeRobot v3.0 dataset at {dataset}, got "
                f"{info.get('codebase_version')!r}"
            )

    runtime_dir = source / ".runtime/rtc/robotwin_clean"
    robot_config_dir = runtime_dir / "robot_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    robot_config_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_dir / "robotwin_clean.txt"
    manifest_path.write_text(
        "".join(f"robotwin {dataset}\n" for dataset in datasets), encoding="utf-8"
    )

    robot_config["norm_stats"] = str(clean_stats.resolve())
    runtime_robot_config = robot_config_dir / "robotwin.yaml"
    with runtime_robot_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(robot_config, f, sort_keys=False, allow_unicode=True)

    recipe.setdefault("model", {})["model_path"] = str(base_weights.resolve())
    recipe["model"]["tokenizer_path"] = str(qwen_weights.resolve())
    data_cfg = recipe.setdefault("data", {})
    data_cfg["train_path"] = str(manifest_path.resolve())
    data_cfg["robot_config_root"] = str(robot_config_dir.resolve())
    # The data loader currently reads normalization from the robot config above.
    # Keep this explicit field in sync for readers and future upstream versions.
    data_cfg["norm_stats_file"] = str(clean_stats.resolve())
    data_cfg["use_future_image"] = use_teachers

    train_cfg = recipe.setdefault("train", {})
    default_output_dir = root / (
        "checkpoints/lingbot-vla-v2-6b-robotwin-rtc"
        if use_teachers
        else "checkpoints/lingbot-vla-v2-6b-robotwin-rtc-no-moge"
    )
    train_cfg["output_dir"] = str(
        Path(os.environ.get("RTC_OUTPUT_DIR", default_output_dir)).expanduser().resolve()
    )
    gpu_count = _gpu_count() * int(os.environ.get("NNODES", "1"))
    if gpu_count <= 0:
        raise ValueError("NNODES must be positive")
    micro_batch_size = int(os.environ.get("RTC_MICRO_BATCH_SIZE", "16"))
    global_batch_size = int(os.environ.get("RTC_GLOBAL_BATCH_SIZE", "128"))
    if micro_batch_size <= 0 or global_batch_size <= 0:
        raise ValueError("RTC_MICRO_BATCH_SIZE and RTC_GLOBAL_BATCH_SIZE must be positive")
    if global_batch_size % (micro_batch_size * gpu_count) != 0:
        raise ValueError(
            "RTC_GLOBAL_BATCH_SIZE must be divisible by "
            "RTC_MICRO_BATCH_SIZE * GPU count "
            f"({global_batch_size} % ({micro_batch_size} * {gpu_count}) != 0)"
        )
    train_cfg["micro_batch_size"] = micro_batch_size
    train_cfg["global_batch_size"] = global_batch_size
    train_cfg["gradient_accumulation_steps"] = global_batch_size // (
        micro_batch_size * gpu_count
    )
    train_cfg["enable_gradient_checkpointing"] = _parse_bool(
        os.environ.get("RTC_GRADIENT_CHECKPOINTING", "true")
    )
    train_cfg["save_steps"] = int(os.environ.get("RTC_SAVE_STEPS", "1000"))
    if train_cfg["save_steps"] <= 0:
        raise ValueError("RTC_SAVE_STEPS must be positive")
    if use_teachers:
        train_cfg.setdefault("align_params", {}).setdefault("depth", {})[
            "moge_path"
        ] = str(moge_path)
        train_cfg["align_params"]["depth"]["morgbd_path"] = str(
            (depth_root / "model.pt").resolve()
        )
        train_cfg["align_params"].setdefault("video", {})["ckpt_path"] = str(
            (dino_root / "teacher_step_10000.pth").resolve()
        )
        train_cfg["align_params"]["video"]["config_path"] = str(
            (dino_root / "config.yaml").resolve()
        )
    else:
        # Keep RTC's action loss, but omit depth/video auxiliary teachers entirely.
        train_cfg["align_params"] = {}
    train_cfg["training_time_rtc"] = True
    train_cfg["training_time_rtc_max_delay"] = int(os.environ.get("RTC_MAX_DELAY", "10"))
    if train_cfg["training_time_rtc_max_delay"] <= 0:
        raise ValueError("RTC_MAX_DELAY must be positive")

    output_path = (output or runtime_dir / "robotwin_rtc.yaml").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f, sort_keys=False, allow_unicode=True)
    print(f"Generated RTC config: {output_path}")
    print(f"Clean LeRobot datasets: {len(datasets)} ({manifest_path})")
    print(f"Clean normalization stats: {clean_stats.resolve()}")
    print(
        "Auxiliary teachers: "
        + ("enabled (MoGe, MoRGBD, DINO-video)" if use_teachers else "disabled")
    )
    print(f"Checkpoint interval: {train_cfg['save_steps']} steps")
    print(
        f"Batch setup: micro={micro_batch_size}, global={global_batch_size}, "
        f"GPUs={gpu_count}, accumulation={train_cfg['gradient_accumulation_steps']}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LMX_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--output", type=Path, help="Config output path (defaults under .runtime/rtc)")
    args = parser.parse_args()
    prepare(args.root, args.output)


if __name__ == "__main__":
    main()
