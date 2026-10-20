"""Guard the user's batch/precision/comparison contract before any GPU launch."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from tools.prepare_roi432 import recipes
from xsy_utils.resolution_exp.roi432_runner import claim_once, validate_recipe, validate_smoke_continuation


class LaunchContractTest(unittest.TestCase):
    def setUp(self):
        self.source = dict(
            model=dict(model_path="/original/6b", vlm_model_name_or_path="/original/qwen"),
            data=dict(img_size=384, image_resize_mode="opencv_lanczos4",
                      preserve_image_resolution=True, num_workers=8, prefetch_factor=4,
                      pin_memory=True, train_path="/pinned/clean50.txt", norm_stats_file="/pinned/clean.json"),
            train=dict(global_batch_size=128, micro_batch_size=4, gradient_accumulation_steps=4,
                       max_steps=10000, save_steps=1000, seed=42, enable_fp32=True,
                       enable_mixed_precision=False, optimizer="adamw", lr=1e-4,
                       lr_min=5e-5, use_compile=True, output_dir="/previous/output"))
        self.config = recipes(self.source, dict(mode="ee", image_size=384), "/geometry", "/outputs")["roi432_ee"]

    def test_prepared_roi_keeps_training_contract(self):
        validate_recipe(self.config, self.source)

    def test_rejects_precision_batch_and_optimizer_shortcuts(self):
        changes = dict(global_batch_size=1024, micro_batch_size=8, gradient_accumulation_steps=2,
                       enable_fp32=False, enable_mixed_precision=True, lr=2e-4, seed=43,
                       optimizer="muon", use_compile=False, max_steps=5000)
        for key, value in changes.items():
            config = copy.deepcopy(self.config)
            config["train"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe(config, self.source)

    def test_rejects_wrong_data_or_finetuned_initialization(self):
        for section, key, value in (
            ("model", "model_path", "/previous/hf_ckpt"),
            ("data", "norm_stats_file", "/random/norm.json"),
            ("data", "train_path", "/other/data.txt"),
            ("data", "num_workers", 1),
            ("train", "load_checkpoint_path", "/smoke/global_step_4"),
        ):
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe(config, self.source)

    def test_concurrent_candidates_have_one_winner_and_no_takeover(self):
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / "claim"
            with ThreadPoolExecutor(max_workers=8) as workers:
                winners = list(workers.map(lambda _: claim_once(claim), range(8)))
            self.assertEqual(sum(winners), 1)
            self.assertFalse((claim / "owner.json").exists())
            self.assertFalse(claim_once(claim))
            (claim / "owner.json").write_text('{"result":"failed"}')
            self.assertFalse(claim_once(claim))

    def test_continuation_is_bound_to_stopped_owner_and_step(self):
        error = "Live GPU workload detected; no interference"
        plan = dict(node="node0", hostname="host0", config_sha256="config")
        owner = dict(node="node0", hostname="host0", pid=17, result="failed", error=error)
        state = dict(phase="failed", error=error, target_steps=2, child_pid=None, resumed_from_step=0)
        record = dict(node="node0", hostname="host0", previous_pid=17, config_sha256="config",
                      next_phase="smoke_resume", checkpoint_step=2, checkpoint_complete=True,
                      wandb_step=2, wandb_state="finished")
        validate_smoke_continuation(record, owner, state, plan)
        for subject, key, value in ((owner,"node","node1"),(owner,"pid",18),
                (owner,"error","CUDA OOM"),(state,"target_steps",5000),
                (state,"child_pid",19),(state,"resumed_from_step",2),
                (record,"checkpoint_complete",False),(record,"wandb_step",0),
                (record,"config_sha256","different")):
            changed = dict(subject); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_smoke_continuation(changed if subject is record else record,
                    changed if subject is owner else owner, changed if subject is state else state, plan)


if __name__ == "__main__":
    unittest.main()
