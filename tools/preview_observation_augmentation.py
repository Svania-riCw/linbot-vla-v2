"""Preview the actual CPU recipe on three RGB files; no model or training launch."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.utils.image_resize import resize_rgb_uint8
from lingbotvla.utils.observation_augmentation import ObservationAugmenter, AugmentationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs=3, required=True, help="head / left wrist / right wrist RGB files")
    parser.add_argument("--recipe", default=str(ROOT/"configs/vla/robotwin/full432_augmentation_recipe.yaml"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh preview directory")
    config = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8")) if args.recipe else asdict(AugmentationConfig())
    aug = ObservationAugmenter(config)
    images = {}
    for name, path in zip(("head", "left_wrist", "right_wrist"), args.images):
        decoded = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Unreadable image: {path}")
        images[name] = resize_rgb_uint8(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), (384, 384))
    rows = [("Original full432 (no ROI)", images)]
    records = []
    for seed in args.seeds:
        plan = aug.sample(seed)
        result, report = aug.apply(images, plan)
        label = f"seed={seed}: KEEP ORIGINAL" if plan["keep_original"] else (
            f"seed={seed}: color={plan['color']} affine={plan['affine']} blur={plan['blur']} jpeg={plan['jpeg']}; no foreground masks")
        rows.append((label, result))
        records.append(dict(plan=plan, report=report))
    sheet = Image.new("RGB", (3*384, len(rows)*(384+28)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, row) in enumerate(rows):
        y = i*(384+28)
        draw.text((8, y+5), label, fill="black")
        for j, name in enumerate(images):
            sheet.paste(Image.fromarray(row[name]), (j*384, y+28))
    args.output.mkdir(parents=True)
    sheet.save(args.output/"preview.jpg", quality=94)
    (args.output/"report.json").write_text(json.dumps(dict(images=args.images, seeds=args.seeds,
        config=asdict(aug.config), records=records, masks_provided=False, model_executed=False),
        indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(dict(preview=str(args.output/"preview.jpg"), report=str(args.output/"report.json")), ensure_ascii=True))


if __name__ == "__main__":
    main()
