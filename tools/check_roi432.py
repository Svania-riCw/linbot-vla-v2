"""Preview real PNG LeRobot frames through ROI-432 and the real Qwen processor.

CPU only: no model weights, no simulation, no training. Writes a contact sheet,
head projection overlay, and a JSON report. Never substitutes synthetic images.
"""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.utils.ee_roi import IMAGE_KEYS, EpisodeGeometry, build_roi_processor
from lingbotvla.utils.image_resize import resize_rgb_uint8


def read_frame(dataset, episode, frame):
    import pyarrow.parquet as pq
    from PIL import Image
    rows = []
    columns = ["episode_index", "frame_index", "observation.state", *IMAGE_KEYS]
    for path in sorted((dataset / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=columns,
                              filters=[("episode_index", "=", episode), ("frame_index", "=", frame)])
        rows.extend(table.to_pylist())
    if len(rows) != 1:
        raise ValueError(f"Expected one matching LeRobot frame, found {len(rows)}")
    row = rows[0]
    images = []
    for key in IMAGE_KEYS:
        image = row[key]
        if not isinstance(image, dict):
            raise ValueError("This preview supports the experiment's PNG image dataset, not video-only datasets")
        if image.get("bytes") is not None:
            source = io.BytesIO(image["bytes"])
        elif image.get("path"):
            path = Path(image["path"])
            source = path if path.is_absolute() else dataset / path
        else:
            raise ValueError(f"No encoded image bytes/path: {key}")
        with Image.open(source) as rgb:
            images.append(np.asarray(rgb.convert("RGB")).copy())
    return images, np.asarray(row["observation.state"], dtype=np.float64)


def check(config_path, dataset, episode, frame, processor_path, output):
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True, use_fast=True)
    model_config = SimpleNamespace(**{**config["model"], **config["train"], "return_image_grid_thw": True})
    canvas = build_roi_processor(SimpleNamespace(**config["data"]), model_config,
                                 SimpleNamespace(image_processor=processor))
    if canvas is None:
        raise ValueError("This check requires an ROI configuration")
    images, state = read_frame(dataset, episode, frame)
    geometry = EpisodeGeometry(config["data"]["roi_geometry_root"], dataset, canvas.fk.sha256).get(
        episode, frame, state)
    item = {key: torch.from_numpy(image).permute(2, 0, 1) for key, image in zip(IMAGE_KEYS, images)}
    item.update({"observation.state": torch.from_numpy(state), "roi_geometry": geometry})
    actual, diagnostics = canvas.prepare_sample(item)
    spec = importlib.util.spec_from_file_location("roi_image_transform", ROOT/"lingbotvla/data/vla_data/transform.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processed, masks, teacher, grid = module.prepare_images(
        processor, {"state": torch.zeros(55), "image": {key: actual[key] for key in IMAGE_KEYS}},
        image_keys=list(IMAGE_KEYS), use_depth_align=True, return_image_grid_thw=True,
        fixed_image_size=(384, 384))
    tokens = int((grid.prod(-1)//processor.merge_size**2).sum())
    if tokens != 432 or not bool(masks.all()) or not bool(torch.isfinite(processed).all()):
        raise AssertionError("Actual processor did not preserve the ROI-432 grid")
    # Upper row is the full-view baseline, lower row is the actual policy input.
    contact = np.concatenate([
        np.concatenate([resize_rgb_uint8(x, (384,384)) for x in images], axis=1),
        np.concatenate([actual[key].permute(1,2,0).numpy() for key in IMAGE_KEYS], axis=1)], axis=0)
    overlay = images[0].copy()
    for side, color in enumerate(((255,80,50), (50,180,255))):
        d = diagnostics[side]
        x0, y0, x1, y1 = d["crop_xyxy"]
        cv2.rectangle(overlay, (x0,y0), (x1,y1), color, 2)
        if d["projection_valid"]:
            cv2.circle(overlay, tuple(int(round(x)) for x in d["projected_uv"]), 4, color, -1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(contact).save(output/"full_vs_roi.png")
    Image.fromarray(overlay).save(output/"head_projection.png")
    report = dict(dataset=str(dataset), episode=episode, frame=frame,
                  state_semantics="robotwin_drive_target", mode=canvas.mode,
                  roi=diagnostics, image_grid_thw=grid.tolist(),
                  visual_patch_tokens=tokens, visual_tokens_with_boundaries=tokens+6,
                  processor_class=type(processor).__name__, model_forward_tested=False)
    (output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset", "processor", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--frame", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.config, args.dataset, args.episode, args.frame, args.processor, args.output),
                     ensure_ascii=True))


if __name__ == "__main__":
    main()
