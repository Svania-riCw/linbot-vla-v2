"""CPU behavior and real Qwen/FeatureTransform tests for the full432 augmentation."""
import ast
import copy
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Literal, Optional
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.utils.observation_augmentation import (
    AugmentationConfig, AugmentationError, ObservationAugmenter, check_augmentation_identity, config_hash,
)
from tools.prepare_full432_augmentation import recipes, prepare


def config(**overrides):
    values = dataclasses.asdict(AugmentationConfig())
    values.update(overrides)
    return values


def only(operation, **overrides):
    values = config()
    for key in values:
        if key.endswith("_probability"):
            values[key] = 0.
    values[operation + "_probability"] = 1.
    values.update(overrides)
    return values


def rgb(height=64, width=80):
    y, x = np.indices((height, width))
    return np.stack([(3*x) % 256, (4*y) % 256, (x+2*y) % 256], axis=-1).astype(np.uint8)


def base_recipe():
    source = yaml.safe_load((ROOT/"configs/vla/robotwin/robotwin.yaml").read_text(encoding="utf-8"))
    source["model"]["model_path"] = "/original/lingbot-vla-v2-6b"
    source["data"].update(prefetch_factor=4, pin_memory=True, roi_config={"previous_roi": True},
                          roi_geometry_root="/previous/geometry")
    source["train"].update(global_batch_size=128, micro_batch_size=4, gradient_accumulation_steps=4,
                           max_steps=10000, save_steps=1000, seed=42, enable_fp32=True,
                           enable_mixed_precision=True, action_fp32=False, load_checkpoint_path="/previous/ckpt")
    return source


class AugmentationTest(unittest.TestCase):
    def test_keep_branch_is_bitwise_noop_for_float_tensors(self):
        image = torch.rand(2, 3, 32, 32)
        sample = {"camera": image, "action": torch.arange(8), "task": "keep task"}
        result, report = ObservationAugmenter(config(keep_original_probability=1)).apply_tensors(sample, ["camera"], 5)
        self.assertIs(result, sample)
        self.assertTrue(report["keep_original"])
        self.assertTrue(torch.equal(result["camera"], image))

    def test_conditional_probabilities_ranges_and_exclusive_blur(self):
        aug = ObservationAugmenter(config())
        plans = [aug.sample(i) for i in range(20000)]
        active = [p for p in plans if not p["keep_original"]]
        self.assertAlmostEqual(len(active)/len(plans), .9, delta=.015)
        for key, expected in (("color", .8), ("affine", .5), ("occlusion", .95),
                              ("gaussian_noise", .5), ("salt_pepper", .15), ("jpeg", .3)):
            self.assertAlmostEqual(sum(p[key] for p in active)/len(active), expected, delta=.02)
        for name, expected in (("gaussian", .25), ("motion", .1), ("none", .65)):
            self.assertAlmostEqual(sum(p["blur"] == name for p in active)/len(active), expected, delta=.02)
        for p in active:
            self.assertLessEqual(max(abs(v) for v in p["translation"]), .06)
            for name in ("brightness", "contrast", "saturation", "scale", "jpeg_quality"):
                low, high = getattr(aug.config, name)
                self.assertTrue(low <= p[name] <= high)

    def test_each_operation_preserves_shape_dtype_and_input(self):
        original = rgb()
        for op in ("color", "affine", "occlusion", "gaussian_noise", "salt_pepper", "gaussian_blur", "motion_blur", "jpeg"):
            with self.subTest(op=op):
                aug = ObservationAugmenter(only(op))
                sample = original.copy()
                out, report = aug.apply({"cam": sample}, aug.sample(31), {"cam": np.zeros(sample.shape[:2], bool)})
                self.assertEqual(out["cam"].shape, original.shape)
                self.assertEqual(out["cam"].dtype, np.uint8)
                np.testing.assert_array_equal(sample, original)
                self.assertFalse(np.array_equal(out["cam"], original))
                if op == "occlusion":
                    self.assertEqual(report["views"]["cam"]["occlusion"], "applied")

    def test_affine_black_padding_and_marker_coordinates(self):
        image = np.full((40, 40, 3), 100, np.uint8)
        image[20, 20] = 255
        aug = ObservationAugmenter(only("affine", scale=(1, 1)))
        plan = aug.sample(2)
        plan["translation"] = [.05, 0]
        out, _ = aug.apply({"cam": image}, plan)
        self.assertTrue((out["cam"][:, :2] == 0).all())
        np.testing.assert_array_equal(out["cam"][20, 22], [255]*3)

    def test_missing_or_fully_protected_masks_skip_occlusion(self):
        aug = ObservationAugmenter(only("occlusion"))
        image = rgb()
        for masks, status in ((None, "skipped_missing_mask"),
                              ({"cam": np.ones(image.shape[:2], bool)}, "skipped_no_safe_rectangle")):
            out, report = aug.apply({"cam": image}, aug.sample(3), masks)
            np.testing.assert_array_equal(out["cam"], image)
            self.assertEqual(report["views"]["cam"]["occlusion"], status)
        with self.assertRaises(ValueError):
            strict = ObservationAugmenter(only("occlusion", missing_foreground="error"))
            strict.apply({"cam": image}, strict.sample(3))

    def test_rectangle_avoids_both_current_and_future_foreground(self):
        clip = np.stack([rgb(), rgb()])
        mask = np.zeros(clip.shape[:3], np.uint8)
        mask[0, 12:40, 30:50] = 1
        mask[1, 15:35, 50:70] = 255
        aug = ObservationAugmenter(only("occlusion", occlusion_area=(.02, .03)))
        applied = 0
        for seed in range(30):
            out, report = aug.apply({"cam": clip}, aug.sample(seed), {"cam": mask})
            for index in range(2):
                np.testing.assert_array_equal(out["cam"][index][mask[index] > 0], clip[index][mask[index] > 0])
            rect = report["views"]["cam"]["rectangle"]
            if rect:
                applied += 1
                x, y, w, h = rect
                self.assertFalse(mask[:, y:y+h, x:x+w].any())
        self.assertGreater(applied, 20)

    def test_foreground_transforms_with_affine(self):
        aug = ObservationAugmenter(only("affine", foreground_margin_pixels=0))
        p = aug.sample(5)
        p.update(scale=1, translation=[.05, 0])
        mask = np.zeros((1, 40, 40), bool)
        mask[0, 20, 20] = True
        protected = aug._protected(mask, mask.shape, aug.affine_matrix(40, 40, p))
        self.assertTrue(protected[20, 22])
        self.assertFalse(protected[20, 20])

    def test_bad_or_incomplete_masks_fail(self):
        aug = ObservationAugmenter(only("occlusion"))
        clip = np.stack([rgb(), rgb()])
        for mask in (np.ones((64, 80)), np.ones((2, 63, 80)), np.full((2, 64, 80), .5)):
            with self.assertRaises(ValueError):
                aug.apply({"cam": clip}, aug.sample(4), {"cam": mask})
        with self.assertRaises(ValueError):
            aug.apply({"cam": clip}, aug.sample(4), {"wrong_camera": np.zeros((64, 80))})

    def test_replay_and_temporal_consistency(self):
        clip = np.stack([rgb(), rgb()])
        aug = ObservationAugmenter(config(keep_original_probability=0))
        for seed in range(20):
            p = aug.sample(seed)
            a, _ = aug.apply({"right": clip, "left": clip}, p)
            b, _ = aug.apply({"left": clip, "right": clip}, p)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
                np.testing.assert_array_equal(a[key][0], a[key][1])

    def test_jpeg_preserves_rgb_channel_order(self):
        image = np.zeros((64, 64, 3), np.uint8)
        image[:] = [220, 20, 10]
        aug = ObservationAugmenter(only("jpeg"))
        out, _ = aug.apply({"cam": image}, aug.sample(4))
        means = out["cam"].mean((0, 1))
        self.assertGreater(means[0], means[2] + 180)

    def test_worker_rng_replay_and_nonimage_invariance(self):
        aug = ObservationAugmenter(config(keep_original_probability=0))
        source = {"cam": torch.from_numpy(rgb()).permute(2, 0, 1).float()/255,
                  "action": torch.arange(14), "observation.state": torch.ones(14), "task": "pick bottle"}
        torch.manual_seed(42)
        a, _ = aug.apply_tensors(source, ["cam"])
        torch.manual_seed(42)
        b, _ = aug.apply_tensors(source, ["cam"])
        self.assertTrue(torch.equal(a["cam"], b["cam"]))
        self.assertEqual(a["cam"].dtype, source["cam"].dtype)
        self.assertTrue(bool(((a["cam"] >= 0) & (a["cam"] <= 1)).all()))
        for key in ("action", "observation.state", "task"):
            self.assertIs(a[key], source[key])

    def test_invalid_configs_and_resume_identity(self):
        for change in (dict(color_probability=1.1), dict(gaussian_blur_probability=.95),
                       dict(scale=(0, 1)), dict(motion_blur_sizes=[4]), dict(jpeg_quality=[90, 35]),
                       dict(missing_foreground="erase"), dict(temperature=(float('nan'), 1)), dict(unknown=True)):
            with self.assertRaises(ValueError):
                ObservationAugmenter(config(**change))
        cfg = config()
        metadata = {"image_augmentation_config_sha256": config_hash(cfg)}
        check_augmentation_identity(metadata, cfg)
        check_augmentation_identity({}, {})  # old unaugmented checkpoint compatibility
        for changed in ({}, config(gaussian_noise_std=(.1, .2))):
            with self.assertRaises(ValueError):
                check_augmentation_identity(metadata, changed)
        with self.assertRaises(ValueError):
            check_augmentation_identity({}, cfg)

    def test_mask_error_is_not_retried_with_another_dataset_sample(self):
        from lingbotvla.utils.ee_roi import ROIGeometryError
        tree = ast.parse((ROOT/"lingbotvla/data/vla_data/multi_vla_dataset.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiVLADataset")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
        namespace = dict(torch=torch, np=np, ROIGeometryError=ROIGeometryError, AugmentationError=AugmentationError)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<dataset getitem>", "exec"), namespace)
        class BrokenDataset:
            calls = 0
            def __len__(self):
                return 10
            def getdata(self, index):
                self.calls += 1
                raise AugmentationError("misaligned mask")
        dataset = BrokenDataset()
        with self.assertRaisesRegex(AugmentationError, "misaligned mask"):
            namespace["__getitem__"](dataset, 0)
        self.assertEqual(dataset.calls, 1)


class ConfigTest(unittest.TestCase):
    def test_recipe_file_matches_defaults_and_export_disables_augmentation(self):
        from tools.render_robotwin_inference_config import render
        recipe = yaml.safe_load((ROOT/"configs/vla/robotwin/full432_augmentation_recipe.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config_hash(recipe), config_hash(config()))
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)/"run"
            checkpoint = run/"checkpoints/global_step_5000/hf_ckpt"
            checkpoint.mkdir(parents=True)
            metadata = dict(post_training=True, max_action_dim=55, max_state_dim=55, token_moe_layers=[],
                            image_augmentation_config_sha256=config_hash(recipe))
            (checkpoint/"config.json").write_text(json.dumps(metadata))
            (checkpoint/"model.safetensors").touch()  # exporter metadata test, not weight validation
            qwen = Path(temp)/"qwen"
            qwen.mkdir()
            (qwen/"config.json").write_text("{}")
            norm = Path(temp)/"norm.json"
            norm.write_text("{}")
            data = dict(joints=[], norm_type=[], roi_config={}, norm_stats_file=str(norm),
                        image_augment=True, image_augmentation_config=recipe, img_size=384,
                        preserve_image_resolution=True, image_resize_mode="opencv_lanczos4")
            (run/"lingbotvla_cli.yaml").write_text(yaml.safe_dump({"data": data}))
            output = Path(temp)/"eval.yaml"
            render(checkpoint, qwen, output)
            evaluated = yaml.safe_load(output.read_text())["data"]
            self.assertFalse(evaluated["image_augment"])
            self.assertEqual(evaluated["image_augmentation_config"], recipe)
            self.assertEqual(evaluated["img_size"], 384)

    def test_matched_recipes_preserve_controls_and_original(self):
        source = base_recipe()
        before = copy.deepcopy(source)
        generated = recipes(source, "/new/experiment")
        a, b = generated["full432_control"], generated["full432_aug"]
        self.assertEqual(source, before)
        self.assertFalse(a["data"]["roi_config"])
        self.assertFalse(b["data"]["roi_config"])
        for cfg in generated.values():
            self.assertNotIn("roi_geometry_root", cfg["data"])
            self.assertNotIn("load_checkpoint_path", cfg["train"])
            self.assertEqual(cfg["model"], source["model"])
            self.assertFalse(cfg["train"]["enable_resume"])
            for key in ("global_batch_size", "micro_batch_size", "gradient_accumulation_steps", "optimizer",
                        "enable_fp32", "enable_mixed_precision", "action_fp32", "align_params", "seed", "lr"):
                self.assertEqual(cfg["train"][key], source["train"][key])
        a["train"].pop("output_dir")
        b["train"].pop("output_dir")
        self.assertEqual(a["train"], b["train"])
        b["data"].pop("image_augmentation_config")
        a["data"].pop("image_augment")
        b["data"].pop("image_augment")
        self.assertEqual(a["data"], b["data"])

    def test_config_writer_no_overwrite_and_reject_changed_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root/"base.yaml"
            base.write_text(yaml.safe_dump(base_recipe()), encoding="utf-8")
            report = prepare(base, root/"config", root/"outputs")
            self.assertFalse(report["training_launched"])
            with self.assertRaises(FileExistsError):
                prepare(base, root/"config", root/"outputs")
        for key, value in (("global_batch_size", 1024), ("enable_fp32", False), ("action_fp32", True)):
            source = base_recipe()
            source["train"][key] = value
            with self.assertRaises(ValueError):
                recipes(source, "/new")

    def test_real_training_parser_accepts_augmented_recipe(self):
        from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, EvalArguments, parse_args
        namespace = dict(globals(), dataclass=dataclasses.dataclass, field=dataclasses.field,
                         DataArguments=DataArguments, ModelArguments=ModelArguments,
                         TrainingArguments=TrainingArguments, EvalArguments=EvalArguments)
        tree = ast.parse((ROOT/"tasks/vla/train_lingbotvla.py").read_text(encoding="utf-8"))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name in ("MyTrainingArguments", "MyDataArguments", "Arguments")]
        exec(compile(ast.Module(body=classes, type_ignores=[]), "<training args>", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"aug.yaml"
            cfg = recipes(base_recipe(), temp)["full432_aug"]
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with patch.dict(globals(), {n: namespace[n] for n in ("MyTrainingArguments", "MyDataArguments")}), \
                 patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "8"}), \
                 patch.object(sys, "argv", ["train", str(path)]):
                args = parse_args(namespace["Arguments"])
            self.assertEqual(config_hash(args.data.image_augmentation_config), config_hash(config()))
            self.assertTrue(args.data.image_augment)
            self.assertFalse(args.data.roi_config)
            self.assertEqual(args.train.gradient_accumulation_steps, 4)


class PipelineTest(unittest.TestCase):
    def test_actual_qwen_432_teacher_alignment_actions_and_eval_bypass(self):
        from transformers import AutoImageProcessor
        processor_path = ROOT/".runtime/resolution_processor"
        if not (processor_path/"preprocessor_config.json").is_file():
            self.skipTest("Real Qwen processor metadata unavailable")
        names = ["lingbotvla.data", "lingbotvla.data.vla_data"]
        saved = {name: sys.modules.get(name) for name in names}
        for name in names:
            package = ModuleType(name)
            package.__path__ = [str(ROOT/name.replace(".", "/"))]
            sys.modules[name] = package
        try:
            from lingbotvla.data.vla_data.utils import FeatureTransform
            processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True, use_fast=True)
            class Tokenizer:
                def __call__(self, prompts, **kwargs):
                    return {"input_ids": torch.ones((1, 4), dtype=torch.long), "attention_mask": torch.ones((1, 4), dtype=torch.long)}
            wrapper = SimpleNamespace(image_processor=processor, tokenizer=Tokenizer())
            data = SimpleNamespace(roi_config={}, img_size=384, preserve_image_resolution=True,
                image_resize_mode="opencv_lanczos4", cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
                joints=["{'arm.position':14}", "{'effector.position':2}"],
                norm_type=["{'arm.position':'identity'}", "{'effector.position':'identity'}"],
                image_augmentation_config=config(keep_original_probability=0, color_probability=1))
            model = SimpleNamespace(max_state_dim=55, max_action_dim=55, return_image_grid_thw=True,
                qwen3vl_use_vision_boundaries=True, tokenizer_max_length=4,
                align_params={"llm": {"image_token_size":12}, "video": {"num_future_frames":1}})
            # Use actual source names from the robot mapping.
            from lingbotvla.utils.ee_roi import IMAGE_KEYS
            source_keys = list(IMAGE_KEYS)
            image = torch.from_numpy(rgb(384, 384)).permute(2, 0, 1)
            def sample(future=True):
                item = {"observation.state": torch.zeros(14), "action": torch.ones((2, 14)),
                        "action_is_pad": torch.zeros(2), "task": "pick bottle"}
                item.update({k: (torch.stack([image, image]) if future else image).clone() for k in source_keys})
                return item
            with tempfile.TemporaryDirectory() as temp:
                norm = Path(temp)/"norm.json"
                norm.write_text(json.dumps({"norm_stats": {"action.arm.position": {"mean": [0]*12},
                                                           "action.effector.position": {"mean": [0]*2}}}))
                kwargs = dict(robot_config_path=str(ROOT/"configs/robot_configs/robotwin.yaml"), data_config=data,
                    model_config=model, processor=wrapper, chunk_size=2, norm_stats_path=str(norm), use_depth_align=True)
                tf = FeatureTransform(**kwargs, use_future_image=True, image_augment=True)
                control = FeatureTransform(**kwargs, use_future_image=True, image_augment=False)
                torch.manual_seed(42)
                training = tf.apply(sample())
                baseline = control.apply(sample())
                self.assertEqual(training["image_grid_thw"].tolist(), [[1, 24, 24]]*3)
                self.assertEqual(int((training["image_grid_thw"].prod(-1)//4).sum()), 432)
                self.assertTrue(torch.equal(training["pil_images"], training["future_pil_images"]))
                self.assertFalse(torch.equal(training["pil_images"], baseline["pil_images"]))
                for key in ("state", "actions", "lang_tokens", "lang_masks", "action_is_pad"):
                    self.assertTrue(torch.equal(training[key], baseline[key]), key)
                serving = FeatureTransform(**kwargs, use_future_image=False, image_augment=True)
                eval_control = FeatureTransform(**kwargs, use_future_image=False, image_augment=False)
                rng = torch.get_rng_state().clone()
                a = serving.apply(sample(False), policy_eval=True)
                b = eval_control.apply(sample(False), policy_eval=True)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(a["images"], b["images"]))
        finally:
            for name, old in saved.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old


if __name__ == "__main__":
    unittest.main()
