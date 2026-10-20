"""CPU tests for ROI geometry, input invariants and the actual Qwen processor."""
import copy
import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Optional, Literal, Dict, List, Any
import unittest
from unittest.mock import patch
import yaml

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.utils.ee_roi import (DualArmFK, ROICanvas, ROIGeometryError, EpisodeGeometry,
    IMAGE_KEYS, canonical_hash, check_roi_identity, crop_with_padding, project_points, build_roi_processor)
from lingbotvla.utils.image_resize import resize_rgb_uint8
from tools.prepare_roi_geometry import validate_episode

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch


def profile():
    """Synthetic geometry, deliberately not advertised as an actual robot profile."""
    result = dict(schema_version=1, state_dim=14, state_semantics="robotwin_drive_target", arms={})
    for side, start, x in (("left", 0, -.2), ("right", 7, .2)):
        base = np.eye(4)
        base[:3, 3] = [x, 0, 2]
        result["arms"][side] = dict(world_from_base=base.tolist(), tip_offset=[0, .1, 0],
            parent_from_tip=np.eye(4).tolist(), chain=[
                dict(name=f"{side}{i}", type="revolute", state_index=start+i,
                     parent_from_joint=np.eye(4).tolist(), child_from_joint=np.eye(4).tolist())
                for i in range(6)])
    return result


def roi_config(mode="ee"):
    return dict(schema_version=1, image_size=384, mode=mode, crop_height_fraction=.5, profile=profile())


def geometry():
    return dict(intrinsic_cv=[[200, 0, 160], [0, 200, 120], [0, 0, 1]],
                extrinsic_cv=np.eye(4), image_hw=[240, 320], profile_sha256=canonical_hash(profile()))


def images():
    y, x = np.indices((240, 320))
    rgb = np.stack([x % 256, y % 256, (x+2*y) % 256], axis=-1).astype(np.uint8)
    return [rgb, np.roll(rgb, 45, 1), np.roll(rgb, 70, 0)]


class GeometryTest(unittest.TestCase):
    def test_fk_rotates_tool_and_ignores_gripper(self):
        fk = DualArmFK(profile())
        q = np.zeros(14)
        q[0] = np.pi/2
        np.testing.assert_allclose(fk.positions(q)[0], [-.2, 0, 2.1], atol=1e-8)
        q[6], q[13] = 9, -9
        np.testing.assert_allclose(fk.positions(q)[0], [-.2, 0, 2.1], atol=1e-8)
        np.testing.assert_allclose(fk.positions(q)[1], [.2, .1, 2], atol=1e-8)

    def test_parent_child_frames_and_world_pose_override(self):
        p = profile()
        p["arms"]["left"]["chain"][0]["parent_from_joint"][0][3] = .3
        p["arms"]["left"]["chain"][0]["child_from_joint"][0][3] = .1
        fk = DualArmFK(p)
        np.testing.assert_allclose(fk.positions(np.zeros(14))[0], [0, .1, 2], atol=1e-8)
        bases = np.stack([np.eye(4), np.eye(4)])
        np.testing.assert_allclose(fk.positions(np.zeros(14), bases)[0], [.2, .1, 0], atol=1e-8)

    def test_invalid_profile_fails_closed(self):
        for mutation in ("index", "transform", "semantics", "missing"):
            p = profile()
            if mutation == "index":
                p["arms"]["left"]["chain"][0]["state_index"] = 7
            elif mutation == "transform":
                p["arms"]["left"]["world_from_base"][0][0] = 2
            elif mutation == "semantics":
                p["state_semantics"] = "measured_qpos"
            else:
                del p["arms"]["right"]
            with self.subTest(mutation=mutation), self.assertRaises(ROIGeometryError):
                DualArmFK(p)

    def test_projection_native_and_rescaled(self):
        xyz = DualArmFK(profile()).positions(np.zeros(14))
        uv, valid = project_points(xyz, geometry(), (480, 640))
        np.testing.assert_allclose(uv, [[280, 260], [360, 260]])
        self.assertTrue(valid.all())
        xyz[0, 2] = -1
        self.assertFalse(project_points(xyz, geometry(), (240, 320))[1][0])
        bad = geometry()
        del bad["intrinsic_cv"]
        with self.assertRaises(ROIGeometryError):
            project_points(xyz, bad, (240, 320))

    def test_padding_does_not_move_requested_crop(self):
        crop, box, fraction = crop_with_padding(np.full((20, 30, 3), 255, np.uint8), (0, 0), (8, 4))
        self.assertEqual(box, [-2, -4, 2, 4])
        self.assertEqual(fraction, .75)
        self.assertTrue((crop[4:, 2:] == 255).all())
        self.assertTrue((crop[:4] == 0).all())

    def test_canvas_shapes_head_rgb_and_fixed_control(self):
        inputs = images()
        saved = [x.copy() for x in inputs]
        roi = ROICanvas(roi_config())
        out, diag = roi.build(inputs, np.zeros(14), geometry())
        self.assertEqual([x.shape for x in out], [(384, 384, 3)]*3)
        np.testing.assert_array_equal(out[0], resize_rgb_uint8(inputs[0], (384, 384)))
        self.assertTrue(all(d["projection_valid"] and not d["fallback"] for d in diag))
        for before, after in zip(saved, inputs):
            np.testing.assert_array_equal(before, after)
        fixed = ROICanvas(roi_config("fixed"))
        a, _ = fixed.build(inputs, np.zeros(14), geometry())
        b, _ = fixed.build(inputs, np.ones(14), geometry())
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertTrue(np.any(out[1] != a[1]))

    def test_invalid_projection_falls_back_but_missing_calibration_raises(self):
        geo = geometry()
        geo["extrinsic_cv"][2, 3] = -3
        actual, diagnostics = ROICanvas(roi_config()).build(images(), np.zeros(14), geo)
        expected, _ = ROICanvas(roi_config("fixed")).build(images(), np.zeros(14), geo)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(all(d["fallback"] for d in diagnostics))
        for bad in ({}, {**geometry(), "profile_sha256": "wrong"}):
            with self.assertRaises(ROIGeometryError):
                ROICanvas(roi_config()).build(images(), np.zeros(14), bad)

    def test_checkpoint_preprocessing_identity(self):
        config = roi_config()
        good = {"roi_config_sha256": canonical_hash(config)}
        check_roi_identity(good, config)
        check_roi_identity({}, {})
        for model, data in (({}, config), (good, {}), (good, roi_config("fixed"))):
            with self.assertRaises(ROIGeometryError):
                check_roi_identity(model, data)

    def raw_episode(self, states):
        fk = DualArmFK(profile())
        joints = np.concatenate([states, states[-1:]])
        endposes = np.stack([fk.positions(q) for q in joints])
        # Nonzero controller tracking error is a measured diagnostic, not an FK bug.
        endposes[:, :, 0] += .01
        raw = {"joint_action/vector": joints,
               "observation/head_camera/intrinsic_cv": np.tile(geometry()["intrinsic_cv"], (len(joints), 1, 1)),
               "observation/head_camera/extrinsic_cv": np.tile(np.eye(4), (len(joints), 1, 1)),
               "observation/head_camera/rgb": np.stack([images()[0]]*len(joints))}
        for i, side in enumerate(("left", "right")):
            raw[f"endpose/{side}_endpose"] = endposes[:, i]
        return raw

    def test_raw_frame_alignment_and_tracking_error_diagnostic(self):
        states = np.zeros((3, 14))
        raw = self.raw_episode(states)
        record, report = validate_episode(DualArmFK(profile()), states, raw)
        self.assertAlmostEqual(report["drive_target_to_recorded_ee_max_m"], .01)
        self.assertEqual(report["actual_ee_in_default_crop_frames"], [3, 3])
        self.assertNotIn("endpose", record)
        raw["joint_action/vector"][1, 0] = .1
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_episode(DualArmFK(profile()), states, raw)

    def test_sidecar_round_trip_and_corruption_cannot_random_retry(self):
        states = np.zeros((3, 14))
        record, report = validate_episode(DualArmFK(profile()), states, self.raw_episode(states))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data" / "robotwin_clean_test"
            dest = root / "geometry" / source.name
            dest.mkdir(parents=True)
            path = dest / "episode_000000.npz"
            np.savez(path, **record)
            report["sidecar_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = dict(complete=True, profile_sha256=canonical_hash(profile()),
                            datasets={source.name: dict(source=str(source), episodes={"0": report})})
            (dest.parent / "manifest.json").write_text(json.dumps(manifest))
            reader = EpisodeGeometry(dest.parent, source, canonical_hash(profile()))
            geo = reader.get(0, 0, states[0])
            np.testing.assert_array_equal(geo["intrinsic_cv"], geometry()["intrinsic_cv"])
            for frame, state in ((-1, states[0]), (3, states[0]), (0, np.ones(14))):
                with self.assertRaises(ROIGeometryError):
                    reader.get(0, frame, state)
            del reader.cache[path]["intrinsic_cv"]
            with self.assertRaises(ROIGeometryError):
                reader.get(0, 0, states[0])
            path.write_bytes(b"corrupt")
            with self.assertRaises(ROIGeometryError):
                EpisodeGeometry(dest.parent, source, canonical_hash(profile())).get(0, 0, states[0])


class ConfigTest(unittest.TestCase):
    def test_layout_controls_keep_model_optimizer_and_input_data_identical(self):
        from tools.prepare_roi432 import recipes
        source = dict(model={"model_path":"/initial6b", "post_training":True},
                      data={"img_size":384, "train_path":"/clean.txt", "num_workers":8},
                      train={"lr":1e-4, "lr_min":5e-5, "max_steps":10000, "micro_batch_size":4,
                             "global_batch_size":128, "align_params":{"llm":{"image_token_size":12}},
                             "output_dir":"/old_run", "load_checkpoint_path":"/old_run/5000",
                             "enable_resume":True, "wandb_name":"old"})
        before = copy.deepcopy(source)
        generated = recipes(source, roi_config(), "/geometry", "/new_runs")
        fixed, ee = (copy.deepcopy(generated[name]) for name in ("roi432_fixed", "roi432_ee"))
        self.assertEqual(fixed["model"], before["model"])
        for config in generated.values():
            self.assertFalse(config["train"]["enable_resume"])
            self.assertNotIn("load_checkpoint_path", config["train"])
            self.assertEqual(config["train"]["pause_after_steps"], 5000)
            self.assertEqual(config["train"]["align_params"], before["train"]["align_params"])
        fixed["data"]["roi_config"]["mode"] = "ee"
        for config in (fixed, ee):
            config["train"].pop("output_dir")
            config["train"].pop("wandb_name")
        self.assertEqual(fixed, ee)
        self.assertEqual(source, before)

    def test_inference_export_retains_roi_and_rejects_wrong_layout(self):
        from tools.render_robotwin_inference_config import render
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/"run"
            checkpoint = run/"checkpoints/global_step_5000/hf_ckpt"
            checkpoint.mkdir(parents=True)
            config = roi_config()
            model = dict(post_training=True, max_action_dim=55, max_state_dim=55, token_moe_layers=[],
                         roi_config_sha256=canonical_hash(config))
            (checkpoint/"config.json").write_text(json.dumps(model))
            (checkpoint/"model.safetensors").touch()  # only metadata exporter is being tested
            qwen = root/"qwen"
            qwen.mkdir()
            (qwen/"config.json").write_text("{}")
            norm = root/"norm.json"
            norm.write_text("{}")
            data = dict(joints=[], norm_type=[], roi_config=config, norm_stats_file=str(norm))
            (run/"lingbotvla_cli.yaml").write_text(yaml.safe_dump({"data":data}))
            output = root/"eval.yaml"
            render(checkpoint, qwen, output)
            self.assertEqual(yaml.safe_load(output.read_text())["data"]["roi_config"], config)
            data["roi_config"] = roi_config("fixed")
            (run/"lingbotvla_cli.yaml").write_text(yaml.safe_dump({"data":data}))
            with self.assertRaises(ROIGeometryError):
                render(checkpoint, qwen, output)


@unittest.skipUnless(HAS_TORCH, "CPU torch runtime required")
class PipelineTest(unittest.TestCase):
    def test_roi_yaml_passes_real_training_argument_parser(self):
        from tools.prepare_roi432 import recipes
        from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, EvalArguments, parse_args
        namespace = dict(globals(), dataclass=dataclasses.dataclass, field=dataclasses.field,
                         DataArguments=DataArguments, ModelArguments=ModelArguments,
                         TrainingArguments=TrainingArguments, EvalArguments=EvalArguments)
        tree = ast.parse((ROOT/"tasks/vla/train_lingbotvla.py").read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ("MyTrainingArguments", "MyDataArguments", "Arguments")]
        exec(compile(ast.Module(body=classes, type_ignores=[]), "<training arguments>", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"roi.yaml"
            config = yaml.safe_load((ROOT/"configs/vla/robotwin/robotwin.yaml").read_text(encoding="utf-8"))
            config["data"].update(roi_config=roi_config(), roi_geometry_root="/verified/geometry",
                                  img_size=384, image_resize_mode="opencv_lanczos4", preserve_image_resolution=True)
            config["train"].update(global_batch_size=128, micro_batch_size=4, max_steps=10000,
                                   save_steps=1000, pause_after_steps=5000, output_dir=str(Path(temporary)/"output"),
                                   enable_resume=False, align_params={}, gradient_accumulation_steps=4)
            config["model"]["config_path"] = None
            config["train"]["load_checkpoint_path"] = "/previous_run/global_step_5000"
            config = recipes(config, roi_config(), "/verified/geometry", Path(temporary))["roi432_ee"]
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch.dict(globals(), {name: namespace[name] for name in ("MyTrainingArguments", "MyDataArguments")}), \
                    patch.dict(os.environ, {"RANK":"0", "LOCAL_RANK":"0", "WORLD_SIZE":"8"}), \
                    patch.object(sys, "argv", ["train", str(path)]):
                args = parse_args(namespace["Arguments"])
                with patch.object(sys, "argv", ["train", str(path),
                        "--train.load_checkpoint_path", "/roi432_ee/checkpoints/global_step_5000",
                        "--train.pause_after_steps", "0"]):
                    resumed = parse_args(namespace["Arguments"])
            self.assertEqual(args.data.roi_config, roi_config())
            self.assertEqual(args.data.roi_geometry_root, "/verified/geometry")
            self.assertEqual(args.train.gradient_accumulation_steps, 4)
            self.assertIsNone(args.train.load_checkpoint_path)
            self.assertEqual(args.model.config_path, args.model.model_path)
            self.assertEqual(resumed.train.pause_after_steps, 0)
            self.assertEqual(resumed.train.max_steps, 10000)
            self.assertEqual(resumed.data.roi_config, args.data.roi_config)

    def sample(self, future=True, floating=True):
        item = {"observation.state": torch.zeros(14), "roi_geometry": geometry(),
                "action": torch.ones((2, 14)), "action_is_pad": torch.zeros(2), "task": "test"}
        for key, image in zip(IMAGE_KEYS, images()):
            x = torch.from_numpy(image).permute(2, 0, 1).contiguous()
            if future:
                x = torch.stack([x, torch.flip(x, (-1,))])
            item[key] = x.float()/255 if floating else x
        return item

    def test_train_serve_pixels_and_future_targets_identical(self):
        roi = ROICanvas(roi_config())
        train = self.sample()
        serve = self.sample(future=False, floating=False)
        result, _ = roi.prepare_sample(train)
        inference, _ = roi.prepare_sample(serve)
        for key in IMAGE_KEYS:
            self.assertTrue(torch.equal((result[key][0]*255).round().byte(), inference[key]))
            expected = resize_rgb_uint8(train[key][1].permute(1, 2, 0).mul(255).round().byte().numpy(), (384, 384))
            np.testing.assert_array_equal((result[key][1]*255).round().byte().permute(1, 2, 0), expected)
        self.assertTrue(torch.equal(train["action"], result["action"]))
        # Changing future state/action cannot change today's ROI.
        changed = self.sample()
        changed["action"] *= 300
        changed["observation.state"] = torch.stack([torch.zeros(14), torch.full((14,), 1e6)])
        alternate, _ = roi.prepare_sample(changed)
        for key in IMAGE_KEYS:
            self.assertTrue(torch.equal(result[key], alternate[key]))

    def test_bad_sample_schema_fails_as_geometry_error(self):
        for key in (*IMAGE_KEYS, "observation.state", "roi_geometry"):
            item = self.sample()
            del item[key]
            with self.subTest(key=key), self.assertRaises(ROIGeometryError):
                ROICanvas(roi_config()).prepare_sample(item)

    def test_real_feature_transform_token_budget_and_teacher_targets(self):
        from transformers import AutoImageProcessor
        processor_path = ROOT / ".runtime/resolution_processor"
        if not (processor_path / "preprocessor_config.json").exists():
            self.skipTest("Actual local Qwen processor metadata unavailable")
        # Avoid eager LeRobot/GPU imports only; execute the actual FeatureTransform.
        names = ["lingbotvla.data", "lingbotvla.data.vla_data"]
        saved = {name: sys.modules.get(name) for name in names}
        for name in names:
            package = ModuleType(name)
            package.__path__ = [str(ROOT / name.replace(".", "/"))]
            sys.modules[name] = package
        try:
            from lingbotvla.data.vla_data.utils import FeatureTransform
            image_processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True, use_fast=True)
            class Tokenizer:
                def __call__(self, prompts, **kwargs):
                    return {"input_ids": torch.ones((1, 4), dtype=torch.long),
                            "attention_mask": torch.ones((1, 4), dtype=torch.long)}
            processor = SimpleNamespace(image_processor=image_processor, tokenizer=Tokenizer())
            data = SimpleNamespace(roi_config=roi_config(), img_size=384, preserve_image_resolution=True,
                image_resize_mode="opencv_lanczos4", cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
                joints=["{'arm.position':14}", "{'effector.position':2}"],
                norm_type=["{'arm.position':'identity'}", "{'effector.position':'identity'}"])
            model = SimpleNamespace(max_state_dim=55, max_action_dim=55, return_image_grid_thw=True,
                qwen3vl_use_vision_boundaries=True, tokenizer_max_length=4,
                align_params={"llm": {"image_token_size":12}, "video": {"num_future_frames":1}})
            with tempfile.TemporaryDirectory() as temporary:
                norm = Path(temporary)/"norm.json"
                norm.write_text(json.dumps({"norm_stats": {
                    "action.arm.position": {"mean": [0]*12},
                    "action.effector.position": {"mean": [0]*2}}}))
                kwargs = dict(robot_config_path=str(ROOT/"configs/robot_configs/robotwin.yaml"),
                    data_config=data, model_config=model, processor=processor, chunk_size=2,
                    norm_stats_path=str(norm), use_depth_align=True)
                tf = FeatureTransform(**kwargs, use_future_image=True)
                training = tf.apply(self.sample())
                sf = FeatureTransform(**kwargs, use_future_image=False)
                serving = sf.apply(self.sample(False, False), policy_eval=True)
                for key in ("images", "image_grid_thw", "state"):
                    self.assertTrue(torch.equal(training[key], serving[key]), key)
                self.assertEqual(training["image_grid_thw"].tolist(), [[1,24,24]]*3)
                self.assertEqual(int((training["image_grid_thw"].prod(-1)//4).sum()), 432)
                # Full-view reference shares the same current/future head teacher images.
                data.roi_config = {}
                baseline_tf = FeatureTransform(**kwargs, use_future_image=True)
                baseline_sample = self.sample()
                from lingbotvla.utils.image_resize import OpenCVResize
                for key in IMAGE_KEYS:
                    baseline_sample[key] = OpenCVResize((384,384))(baseline_sample[key])
                baseline = baseline_tf.apply(baseline_sample)
                for key in ("pil_images", "future_pil_images"):
                    self.assertTrue(torch.equal(training[key][0], baseline[key][0]), key)
                self.assertIsNone(build_roi_processor(data, model, processor))
        finally:
            for name, old in saved.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old


if __name__ == "__main__":
    unittest.main()
