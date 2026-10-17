"""CPU check of real RGB -> shared resize -> Qwen processor -> prepare_images.

Loads processor metadata only, never model weights. Pass three --image arguments
to exercise real camera views; one image is repeated for a shape smoke check.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(config_path, image_paths, processor_path):
    import numpy as np
    from PIL import Image
    import torch
    import yaml
    from transformers import AutoImageProcessor

    from lingbotvla.utils.image_resize import build_image_resize, fixed_image_grid

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data = config["data"]
    if not data.get("preserve_image_resolution", False):
        raise ValueError("Baseline check requires data.preserve_image_resolution=true")
    processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True, use_fast=True)
    size = int(data["img_size"])
    grid = fixed_image_grid((size, size), processor.patch_size, processor.merge_size)
    merged_side = int(grid[0, 1]) // processor.merge_size
    if config["train"]["align_params"]["llm"]["image_token_size"] != merged_side:
        raise ValueError(f"Teacher student-grid slice must be {merged_side}")
    cameras = data["cameras"]
    if len(image_paths) not in (1, len(cameras)):
        raise ValueError("Pass one image (repeated smoke input) or one image per configured camera")
    paths = list(image_paths) * len(cameras) if len(image_paths) == 1 else list(image_paths)
    resize = build_image_resize((size, size), data.get("image_resize_mode", "torchvision_bilinear"))
    images, source_sizes = {}, []
    for camera, path in zip(cameras, paths):
        with Image.open(path) as source:
            pixels = np.asarray(source.convert("RGB")).copy()
        source_sizes.append([int(pixels.shape[1]), int(pixels.shape[0])])
        image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        images[camera] = resize(image)

    # Load the actual transform without importing the GPU/distributed/LeRobot stack
    # pulled in by lingbotvla.data.__init__.py. No transform logic is copied here.
    spec = importlib.util.spec_from_file_location("resolution_image_transform", ROOT / "lingbotvla/data/vla_data/transform.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pixels, masks, teacher_images, actual_grid = module.prepare_images(
        processor,
        {"state": torch.zeros(55), "image": images},
        image_keys=cameras,
        use_depth_align=True,
        return_image_grid_thw=True,
        fixed_image_size=(size, size),
    )
    expected_grid = torch.as_tensor(grid).repeat(len(cameras), 1)
    if not torch.equal(actual_grid, expected_grid) or not bool(masks.all()):
        raise AssertionError("Camera grid or mask mismatch")
    visual_tokens = int((actual_grid.prod(dim=-1) // processor.merge_size**2).sum())
    if pixels.shape[-2] != int(grid.prod()):
        raise AssertionError("Processor pixel patch count does not match its reported grid")
    if not bool(torch.isfinite(pixels).all()):
        raise AssertionError("Processor output contains non-finite values")
    report = {
        "config": str(config_path),
        "processor": str(processor_path),
        "processor_class": type(processor).__name__,
        "transformers_version": __import__("transformers").__version__,
        "opencv_version": __import__("cv2").__version__,
        "input_mode": "repeated_single_image_smoke" if len(image_paths) == 1 else "one_image_per_camera",
        "source_sizes_wh": source_sizes,
        "input_size_wh": [size, size],
        "image_grid_thw": actual_grid.tolist(),
        "pixel_values_shape": list(pixels.shape),
        "teacher_images_shape": list(teacher_images.shape),
        "visual_tokens": visual_tokens,
        "visual_tokens_with_boundaries": visual_tokens + 2 * len(cameras),
        "llm_image_token_size": merged_side,
        "model_forward_tested": False,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--image", required=True, action="append", type=Path)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.config, args.image, args.processor)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
