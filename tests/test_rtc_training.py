"""Check training preparation against the actual argument and normalizer APIs."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_rtc_training", REPO / "tools/prepare_rtc_training.py")
prepare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_module)


class TrainingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        for relative in ("configs/vla/robotwin/robotwin.yaml", "configs/robot_configs/robotwin.yaml"):
            target = self.source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, target)
        self.recipe_before = (self.source / "configs/vla/robotwin/robotwin.yaml").read_bytes()
        for relative in (
            "checkpoints/lingbot-vla-v2-6b/config.json",
            "checkpoints/lingbot-vla-v2-6b/model-00001-of-00001.safetensors",
            "checkpoints/Qwen3-VL-4B-Instruct/config.json",
            "checkpoints/lingbot-vla-v2-6b/depth/model.pt",
            "checkpoints/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth",
            "checkpoints/lingbot-vla-v2-6b/dino_video/config.yaml",
            "checkpoints/moge2-vitb-normal.pt",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture")
        index = self.root / "checkpoints/lingbot-vla-v2-6b/model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"parameter": "model-00001-of-00001.safetensors"}}))
        info_path = self.root / "training-data/lerobot/robotwin_clean_task/meta/info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps({"codebase_version": "v3.0"}))
        self.stats_path = self.root / "training-data/robotwin_clean_norm_stats.json"
        stats = {}
        for prefix in ("action", "observation.state"):
            for name, dim in (("arm.position", 12), ("effector.position", 2)):
                stats[f"{prefix}.{name}"] = {
                    "mean": [0.5] * dim, "std": [0.25] * dim,
                    "q01": [0.1] * dim, "q99": [0.9] * dim,
                }
        self.stats_path.write_text(json.dumps({"norm_stats": stats}))
        env = {key: value for key, value in os.environ.items() if not key.startswith("RTC_")}
        for key in ("NPROC_PER_NODE", "NNODES", "CUDA_VISIBLE_DEVICES"):
            env.pop(key, None)
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        env_patch = patch.dict(os.environ, env, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def generate(self):
        return prepare_module.prepare(self.root, source=self.source)

    def test_default_enables_teacher_losses_and_prepares_batch_manifest_stats(self):
        output = self.generate()
        config = yaml.safe_load(output.read_text())
        train = config["train"]
        self.assertEqual((train["micro_batch_size"], train["global_batch_size"], train["gradient_accumulation_steps"]), (16, 128, 1))
        self.assertTrue(train["training_time_rtc"])
        self.assertEqual(train["training_time_rtc_fixed_delay"], -1)
        self.assertIn("depth", train["align_params"])
        self.assertIn("video", train["align_params"])
        self.assertEqual(train["save_steps"], 1000)
        self.assertTrue(config["data"]["use_future_image"])
        self.assertEqual(
            train["output_dir"],
            str(self.root / "checkpoints/lingbot-vla-v2-6b-robotwin-rtc"),
        )
        robot = Path(config["data"]["robot_config_root"]) / "robotwin.yaml"
        self.assertEqual(yaml.safe_load(robot.read_text())["norm_stats"], str(self.stats_path))
        self.assertEqual(config["data"]["norm_stats_file"], str(self.stats_path))
        self.assertEqual(len(Path(config["data"]["train_path"]).read_text().splitlines()), 1)
        self.assertEqual((self.source / "configs/vla/robotwin/robotwin.yaml").read_bytes(), self.recipe_before)

    def test_default_requires_teacher_checkpoints(self):
        for relative in (
            "checkpoints/lingbot-vla-v2-6b/depth/model.pt",
            "checkpoints/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth",
            "checkpoints/lingbot-vla-v2-6b/dino_video/config.yaml",
            "checkpoints/moge2-vitb-normal.pt",
        ):
            (self.root / relative).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "not found or empty"):
                self.generate()
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture")

    def test_explicit_disable_does_not_require_teacher_checkpoints(self):
        for relative in (
            "checkpoints/lingbot-vla-v2-6b/depth/model.pt",
            "checkpoints/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth",
            "checkpoints/lingbot-vla-v2-6b/dino_video/config.yaml",
            "checkpoints/moge2-vitb-normal.pt",
        ):
            (self.root / relative).unlink()
        with patch.dict(os.environ, {"RTC_USE_TEACHERS": "false"}):
            config = yaml.safe_load(self.generate().read_text())
        self.assertEqual(config["train"]["align_params"], {})
        self.assertFalse(config["data"]["use_future_image"])
        self.assertTrue(config["train"]["output_dir"].endswith("robotwin-rtc-no-moge"))

    def test_teacher_mode_requires_moge(self):
        (self.root / "checkpoints/moge2-vitb-normal.pt").unlink()
        with patch.dict(os.environ, {"RTC_USE_TEACHERS": "true"}):
            with self.assertRaisesRegex(FileNotFoundError, "MoGe"):
                self.generate()

    def test_explicit_teacher_mode_uses_configured_teacher_paths(self):
        with patch.dict(os.environ, {"RTC_USE_TEACHERS": "true", "RTC_SAVE_STEPS": "250"}):
            config = yaml.safe_load(self.generate().read_text())
        self.assertTrue(config["data"]["use_future_image"])
        self.assertTrue(config["train"]["align_params"]["use_future_video"])
        self.assertEqual(config["train"]["save_steps"], 250)
        self.assertEqual(
            config["train"]["align_params"]["depth"]["moge_path"],
            str(self.root / "checkpoints/moge2-vitb-normal.pt"),
        )

    def test_fixed_delay_is_added_to_generated_config(self):
        with patch.dict(os.environ, {"RTC_FIXED_DELAY": "6"}):
            config = yaml.safe_load(self.generate().read_text())
        self.assertEqual(config["train"]["training_time_rtc_max_delay"], 10)
        self.assertEqual(config["train"]["training_time_rtc_fixed_delay"], 6)

    def test_fixed_delay_must_fit_exclusive_max_delay(self):
        with patch.dict(os.environ, {"RTC_MAX_DELAY": "6", "RTC_FIXED_DELAY": "6"}):
            with self.assertRaisesRegex(ValueError, "RTC_FIXED_DELAY"):
                self.generate()

    def test_nondivisible_global_batch_is_rejected(self):
        with patch.dict(os.environ, {"RTC_GLOBAL_BATCH_SIZE": "127"}):
            with self.assertRaisesRegex(ValueError, "divisible"):
                self.generate()

    def test_process_count_and_node_count_determine_accumulation(self):
        with patch.dict(os.environ, {"NPROC_PER_NODE": "4", "NNODES": "2"}):
            config = yaml.safe_load(self.generate().read_text())
            self.assertEqual(config["train"]["gradient_accumulation_steps"], 1)

    def test_missing_checkpoint_shard_is_rejected(self):
        (self.root / "checkpoints/lingbot-vla-v2-6b/model-00001-of-00001.safetensors").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Base weight shard"):
            self.generate()

    def test_recipe_parses_and_actual_normalizer_uses_clean_stats(self):
        from tasks.vla.train_lingbotvla import Arguments
        from lingbotvla.utils.arguments import parse_args
        from lingbotvla.data.vla_data.utils import FeatureTransform

        output = self.generate()
        with patch.dict(os.environ, {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "8"}), patch("sys.argv", ["validate", str(output)]):
            args = parse_args(Arguments)
        self.assertEqual(args.train.gradient_accumulation_steps, 1)
        self.assertTrue(args.train.training_time_rtc)
        self.assertEqual(args.train.training_time_rtc_fixed_delay, -1)
        transform = FeatureTransform(
            str(Path(args.data.robot_config_root) / "robotwin.yaml"), args.data, None, None,
        )
        self.assertEqual(transform.normalizer.norm_stats["action.arm.position"]["mean"].tolist(), [0.5] * 12)

    def launcher_fixture(self):
        shutil.copyfile(REPO / "train_rtc.sh", self.source / "train_rtc.sh")
        tool_dir = self.source / "tools"
        tool_dir.mkdir()
        shutil.copyfile(REPO / "tools/rtc_paths.sh", tool_dir / "rtc_paths.sh")
        activation = self.root / "activate.sh"
        activation.write_text('conda() { [[ "$1" == activate ]]; }\n')
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        runner = bin_dir / "torchrun"
        runner.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$RTC_TEST_ARGS"\nexit 17\n')
        runner.chmod(0o755)
        config = self.source / "custom.yaml"
        config.write_text("train: {}\n")
        args_file = self.root / "runner_args.txt"
        env = dict(os.environ, CONDA_SH=str(activation), RTC_TEST_ARGS=str(args_file),
                   LMX_ROOT=str(self.root), PATH=f"{bin_dir}:{os.environ['PATH']}")
        return config, args_file, env

    def test_launcher_propagates_failure_and_forwards_rtc_arguments(self):
        config, args_file, env = self.launcher_fixture()
        result = subprocess.run(
            ["bash", str(self.source / "train_rtc.sh"), str(config), "--train.max_steps", "1"],
            env=env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        args = args_file.read_text().splitlines()
        self.assertIn("--nproc-per-node=8", args)
        self.assertIn(str(config), args)
        self.assertIn("--train.max_steps", args)
        self.assertEqual(
            args[-10:],
            [
                "--train.training_time_rtc", "true",
                "--train.training_time_rtc_max_delay", "10",
                "--train.training_time_rtc_fixed_delay", "-1",
                "--train.output_dir", str(self.root / "checkpoints/lingbot-vla-v2-6b-robotwin-rtc"),
                "--train.enable_resume", "false",
            ],
        )

    def test_launcher_uses_named_run_directory_and_overwrites_only_that_run(self):
        config, args_file, env = self.launcher_fixture()
        run_dir = self.root / "checkpoints/rtc_named"
        run_dir.mkdir(parents=True)
        marker = run_dir / "old-run-marker"
        marker.write_text("old")
        outside_marker = self.root / "checkpoints/keep-me"
        outside_marker.mkdir(parents=True)
        (outside_marker / "marker").write_text("keep")

        result = subprocess.run(
            ["bash", str(self.source / "train_rtc.sh"), "--name", "rtc_named", "--overwrite", str(config)],
            env=env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertFalse(marker.exists())
        self.assertTrue((outside_marker / "marker").exists())
        args = args_file.read_text().splitlines()
        self.assertEqual(args[-4:], ["--train.output_dir", str(run_dir), "--train.enable_resume", "false"])

    def test_launcher_resumes_latest_named_checkpoint(self):
        config, args_file, env = self.launcher_fixture()
        run_dir = self.root / "checkpoints/rtc_resume"
        for step in (900, 1000):
            (run_dir / "checkpoints" / f"global_step_{step}").mkdir(parents=True)
        (run_dir / "checkpoints" / "not-a-checkpoint").mkdir()

        result = subprocess.run(
            ["bash", str(self.source / "train_rtc.sh"), "--name", "rtc_resume", "--resume", str(config)],
            env=env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertIn("Resuming from latest checkpoint", result.stdout)
        args = args_file.read_text().splitlines()
        self.assertEqual(args[-4:], ["--train.output_dir", str(run_dir), "--train.enable_resume", "true"])

    def test_launcher_requires_explicit_mode_for_existing_run(self):
        config, args_file, env = self.launcher_fixture()
        run_dir = self.root / "checkpoints/lingbot-vla-v2-6b-robotwin-rtc"
        run_dir.mkdir(parents=True)
        (run_dir / "partial.log").write_text("partial run")

        result = subprocess.run(
            ["bash", str(self.source / "train_rtc.sh"), str(config)],
            env=env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Use --resume or --overwrite explicitly", result.stderr)
        self.assertFalse(args_file.exists())

    def test_launcher_resume_requires_a_checkpoint(self):
        config, args_file, env = self.launcher_fixture()
        result = subprocess.run(
            ["bash", str(self.source / "train_rtc.sh"), "--name", "rtc_missing", "--resume", str(config)],
            env=env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("No global_step_N checkpoint found", result.stderr)
        self.assertFalse(args_file.exists())

    def test_prepare_only_does_not_start_distributed_training(self):
        config, args_file, env = self.launcher_fixture()
        result = subprocess.run(
            ["bash", str(self.source / "train_rtc.sh"), "--prepare-only", str(config)],
            env=env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(args_file.exists())


if __name__ == "__main__":
    unittest.main()
