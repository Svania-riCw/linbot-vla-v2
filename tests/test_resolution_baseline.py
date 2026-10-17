"""CPU checks for the resolution baseline, without model weights or a GPU.

Run: python -m unittest discover -s tests -p test_resolution_baseline.py -v
Only tensor/torchvision checks are skipped when those optional runtimes are absent.
"""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lingbotvla.utils.image_resize import (
    OpenCVResize,
    build_image_resize,
    fixed_image_grid,
    process_image_at_fixed_size,
    resize_rgb_uint8,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch


def rgb_sample():
    """Asymmetric, non-square RGB input makes channel/axis swaps observable."""
    y, x = np.indices((13, 19))
    return np.stack(((x * 13) % 256, (y * 21) % 256, ((x + 2 * y) * 7) % 256), axis=-1).astype(np.uint8)


class NumpyResizeTest(unittest.TestCase):
    def test_non_square_output_and_rgb_channel_identity(self):
        image = np.empty((9, 17, 3), dtype=np.uint8)
        image[...] = (13, 97, 241)
        result = resize_rgb_uint8(image, (31, 43))
        self.assertEqual(result.shape, (31, 43, 3))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, np.broadcast_to((13, 97, 241), result.shape))

    def test_uses_real_lanczos_resampling_on_a_nonconstant_image(self):
        image = rgb_sample()
        result = resize_rgb_uint8(image, (32, 48))
        expected = cv2.resize(image, (48, 32), interpolation=cv2.INTER_LANCZOS4)
        np.testing.assert_array_equal(result, expected)
        # Ensure this fixture would catch accidentally retaining the bilinear path.
        bilinear = cv2.resize(image, (48, 32), interpolation=cv2.INTER_LINEAR)
        self.assertTrue(np.any(result != bilinear))

    def test_lanczos_ringing_saturates_instead_of_wrapping(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, 4:] = 255
        unbounded = cv2.resize(image.astype(np.float32), (67, 61), interpolation=cv2.INTER_LANCZOS4)
        self.assertLess(unbounded.min(), 0)
        self.assertGreater(unbounded.max(), 255)
        result = resize_rgb_uint8(image, (61, 67))
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 255)
        np.testing.assert_array_equal(result[:, 0], np.zeros((61, 3), dtype=np.uint8))
        np.testing.assert_array_equal(result[:, -1], np.full((61, 3), 255, dtype=np.uint8))

    def test_rejects_non_rgb_or_non_byte_inputs(self):
        invalid = [np.zeros((3, 8, 8), np.uint8), np.zeros((8, 8), np.uint8),
                   np.zeros((8, 8, 4), np.uint8), np.zeros((8, 8, 3), np.float32)]
        for image in invalid:
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(ValueError):
                resize_rgb_uint8(image, (16, 16))
        for size in [(0, 16), (16, -1), (16.5, 16)]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                resize_rgb_uint8(rgb_sample(), size)

    def test_unsupported_resize_mode_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "image_resize_mode"):
            build_image_resize((256, 256), "ai_super_resolution")


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable: tensor resize checks require the CPU runtime")
class TensorResizeTest(unittest.TestCase):
    def setUp(self):
        self.image = torch.from_numpy(rgb_sample()).permute(2, 0, 1).contiguous()

    def test_float_png_and_byte_video_take_the_same_pixel_path(self):
        transform = OpenCVResize((32, 48))
        byte_output = transform(self.image)
        float_output = transform(self.image.float() / 255)
        self.assertEqual(tuple(byte_output.shape), (3, 32, 48))
        self.assertEqual(byte_output.dtype, torch.uint8)
        self.assertEqual(float_output.dtype, torch.float32)
        self.assertTrue(torch.equal((float_output * 255).round().to(torch.uint8), byte_output))
        self.assertTrue(torch.isfinite(float_output).all())
        self.assertGreaterEqual(float_output.min().item(), 0)
        self.assertLessEqual(float_output.max().item(), 1)
        expected = resize_rgb_uint8(rgb_sample(), (32, 48))
        np.testing.assert_array_equal(byte_output.permute(1, 2, 0).numpy(), expected)

    def test_current_and_future_frames_are_resized_independently(self):
        future = torch.flip(self.image, dims=(-1,))
        frames = torch.stack([self.image, future])
        transform = OpenCVResize((64, 96))
        result = transform(frames)
        self.assertEqual(tuple(result.shape), (2, 3, 64, 96))
        for index in range(2):
            self.assertTrue(torch.equal(result[index], transform(frames[index])))
        self.assertFalse(torch.equal(result[0], result[1]))
        self.assertTrue(torch.equal(frames[0], self.image))  # caller-owned frames are not mutated
        float_result = transform(frames.float() / 255)
        self.assertTrue(torch.equal((float_result * 255).round().to(torch.uint8), result))

    def test_invalid_float_ranges_nan_and_layout_are_rejected(self):
        invalid = [self.image.float(), self.image.float() / 255 - 0.1,
                   torch.full((3, 4, 4), float("nan")),
                   torch.zeros((4, 4, 3)), torch.zeros((1, 1, 3, 4, 4)),
                   self.image.to(torch.int32)]
        for image in invalid:
            with self.subTest(shape=tuple(image.shape), dtype=image.dtype), self.assertRaises(ValueError):
                OpenCVResize((32, 32))(image)

    def test_opt_in_factory_selects_opencv(self):
        transform = build_image_resize((32, 48), "opencv_lanczos4")
        self.assertIsInstance(transform, OpenCVResize)
        self.assertTrue(torch.equal(transform(self.image), OpenCVResize((32, 48))(self.image)))

    def test_default_factory_preserves_existing_torchvision_resize(self):
        if importlib.util.find_spec("torchvision") is None:
            self.skipTest("torchvision unavailable: legacy resize compatibility needs torchvision")
        from torchvision.transforms.v2 import Resize

        actual = build_image_resize((32, 48))
        self.assertIsInstance(actual, Resize)
        expected = Resize((32, 48))(self.image)
        self.assertTrue(torch.equal(actual(self.image), expected))
        self.assertTrue(torch.equal(actual(self.image.float() / 255), Resize((32, 48))(self.image.float() / 255)))


class ProcessorGridGuardTest(unittest.TestCase):
    def test_expected_grids_represent_two_distinct_token_budgets(self):
        for size, side, tokens in [(256, 16, 192), (384, 24, 432)]:
            with self.subTest(size=size):
                grid = fixed_image_grid((size, size), 16, 2)
                np.testing.assert_array_equal(grid, [[1, side, side]])
                self.assertEqual(int(grid.prod() // 4 * 3), tokens)

    def test_processor_that_silently_compresses_384_back_to_256_is_rejected(self):
        class DownscalingProcessor:
            patch_size, merge_size = 16, 2

            def __call__(self, image, **kwargs):
                self.called_with = kwargs
                return {"image_grid_thw": np.array([[1, 16, 16]])}

        processor = DownscalingProcessor()
        with self.assertRaisesRegex(ValueError, "changed the token grid"):
            process_image_at_fixed_size(processor, np.zeros((3, 384, 384), np.uint8), (384, 384))
        self.assertEqual(processor.called_with, {"do_resize": False})

    def test_missing_grid_or_mismatched_image_size_cannot_pass_as_valid(self):
        class MissingGridProcessor:
            patch_size, merge_size = 16, 2

            def __call__(self, image, **kwargs):
                return {"pixel_values": image}

        with self.assertRaisesRegex(ValueError, "did not return image_grid_thw"):
            process_image_at_fixed_size(MissingGridProcessor(), np.zeros((3, 256, 256)), (256, 256))
        with self.assertRaisesRegex(ValueError, "Expected RGB image"):
            process_image_at_fixed_size(MissingGridProcessor(), np.zeros((3, 256, 256)), (384, 384))
        with self.assertRaisesRegex(ValueError, "divisible"):
            fixed_image_grid((300, 300), 16, 2)


class ResolutionConfigTest(unittest.TestCase):
    def test_no_alignment_remains_disabled_and_outputs_can_use_own_workspace(self):
        spec = importlib.util.spec_from_file_location(
            "resolution_noalign_config_test", ROOT / "tools/prepare_resolution_baseline.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = {"model": {}, "data": {}, "train": {"align_params": {}, "output_dir": "/shared/baseline"}}
            base = directory / "base.yaml"
            base.write_text(yaml.safe_dump(source), encoding="utf-8")
            for path in module.prepare(base, directory / "configs", run_root=directory / "own_outputs"):
                result = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(result["train"]["align_params"], {})
                self.assertEqual(Path(result["train"]["output_dir"]).parent, directory / "own_outputs")

    def test_inference_export_keeps_run_resolution_and_allows_explicit_override(self):
        spec = importlib.util.spec_from_file_location(
            "resolution_inference_config_test", ROOT / "tools/render_robotwin_inference_config.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run = directory / "run"
            checkpoint = run / "checkpoints/global_step_1000/hf_ckpt"
            checkpoint.mkdir(parents=True)
            model = dict(post_training=True, max_action_dim=55, max_state_dim=55, token_moe_layers=[0])
            (checkpoint / "config.json").write_text(json.dumps(model), encoding="utf-8")
            (checkpoint / "model.safetensors").touch()  # renderer checks names; never reads weights
            qwen = directory / "qwen"
            qwen.mkdir()
            (qwen / "config.json").write_text("{}", encoding="utf-8")
            norm = directory / "norm.json"
            norm.write_text("{}", encoding="utf-8")
            data = dict(img_size=384, image_resize_mode="opencv_lanczos4", preserve_image_resolution=True,
                        joints=[{"arm.position": 14}], norm_type=[{"arm.position": "bounds_99_woclip"}])
            (run / "lingbotvla_cli.yaml").write_text(yaml.safe_dump({"data": data}), encoding="utf-8")
            output = directory / "inference.yaml"
            module.render(checkpoint, qwen, output, norm_stats=norm)
            actual = yaml.safe_load(output.read_text(encoding="utf-8"))["data"]
            self.assertEqual(actual["img_size"], 384)
            self.assertEqual(actual["image_resize_mode"], "opencv_lanczos4")
            self.assertTrue(actual["preserve_image_resolution"])
            explicit = directory / "explicit.yaml"
            explicit.write_text(yaml.safe_dump({"data": dict(data, img_size=256)}), encoding="utf-8")
            module.render(checkpoint, qwen, output, norm_stats=norm, data_config=explicit)
            self.assertEqual(yaml.safe_load(output.read_text(encoding="utf-8"))["data"]["img_size"], 256)

    def test_generated_pair_only_changes_resolution_and_corresponding_student_grid(self):
        module_path = ROOT / "tools/prepare_resolution_baseline.py"
        spec = importlib.util.spec_from_file_location("resolution_baseline_config_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = yaml.safe_load((ROOT / "xsy_utils/configs/robotwin_clean_baseline.yaml").read_text(encoding="utf-8"))
        original = copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base = directory / "base.yaml"
            base.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
            paths = module.prepare(base, directory / "generated", sizes=(256, 384))
            self.assertEqual(len(paths), 2)
            configs = [yaml.safe_load(Path(path).read_text(encoding="utf-8")) for path in paths]
            by_size = {config["data"]["img_size"]: config for config in configs}
            self.assertEqual(set(by_size), {256, 384})
            self.assertEqual(by_size[256]["train"]["align_params"]["llm"]["image_token_size"], 8)
            self.assertEqual(by_size[384]["train"]["align_params"]["llm"]["image_token_size"], 12)
            self.assertNotEqual(by_size[256]["train"]["output_dir"], by_size[384]["train"]["output_dir"])
            for config in configs:
                self.assertEqual(config["data"]["image_resize_mode"], "opencv_lanczos4")
                self.assertTrue(config["data"]["preserve_image_resolution"])
                self.assertEqual(config["train"]["align_params"]["depth"], original["train"]["align_params"]["depth"])
                self.assertEqual(config["train"]["align_params"]["video"], original["train"]["align_params"]["video"])
                self.assertEqual(config["model"], original["model"])
            left, right = (copy.deepcopy(by_size[size]) for size in (256, 384))
            for config in (left, right):
                config["data"].pop("img_size")
                config["train"].pop("output_dir")
                config["train"]["align_params"]["llm"].pop("image_token_size")
            for section in ("model", "data", "train"):
                self.assertEqual(left[section], right[section], section)
            self.assertEqual(yaml.safe_load(base.read_text(encoding="utf-8")), original)


if __name__ == "__main__":
    unittest.main()
