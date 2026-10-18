"""CPU regression checks for clean-only inputs, pause/resume and complete evals."""
import copy
import ast
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from typing import Any, Dict, List, Optional, Literal

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingbotvla.utils.training_schedule import training_stop_step, capture_rng_state, restore_rng_state


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('prepare_finetune_test', 'tools/prepare_resolution_finetune.py')
evaluate = load('eval_finetune_test', 'tools/eval_resolution_checkpoint.py')


class FinetuneInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'lerobot'
        self.data.mkdir()
        self.norm = self.root / 'clean_stats.json'
        self.norm.write_text('{"arm.position": {"min": -1, "max": 1}}')
        robot = self.root / 'robot_config_clean'
        robot.mkdir()
        (robot / 'robotwin.yaml').write_text(yaml.safe_dump({'norm_stats': str(self.norm), 'features': {}}))
        self.entries = []
        for i in range(50):
            task = self.data / f'robotwin_clean_task_{i}'
            (task / 'meta').mkdir(parents=True)
            (task / 'meta/info.json').write_text(json.dumps({'total_episodes': 50}))
            self.entries.append(f'robotwin {task.as_posix()}')
        self.manifest = self.root / 'clean.txt'
        self.manifest.write_text('\n'.join(self.entries))
        self.baseline = self.root / 'baseline_gbs128_10k'
        hf = self.baseline / 'checkpoints/global_step_10000/hf_ckpt'
        hf.mkdir(parents=True)
        (hf / 'config.json').write_text('{}')
        (hf / 'model.safetensors').touch()
        self.config = {'model': {'model_path': '/initial/6b'},
                       'data': {'train_path': str(self.manifest), 'robot_config_root': str(robot),
                                'norm_stats_file': None, 'img_size': 256},
                       'train': {'max_steps': 10000, 'global_batch_size': 128, 'micro_batch_size': 16,
                                 'enable_fp32': True, 'seed': 42, 'lr': 1e-4, 'optimizer': 'muon',
                                 'load_checkpoint_path': None, 'output_dir': '/baseline',
                                 'align_params': {'llm': {'image_token_size': 8},
                                                  'depth': {'input_size': 224}}}}
        self.save_reference()

    def save_reference(self):
        (self.baseline / 'lingbotvla_cli.yaml').write_text(yaml.safe_dump(self.config))

    def generate(self, micro=4):
        return prepare.prepare(self.baseline, self.root / 'plan', self.root / 'outputs',
                               self.data, self.norm, micro)

    def test_pins_real_training_mapping_and_preserves_initialization_and_lr_horizon(self):
        original = copy.deepcopy(self.config)
        report = self.generate()
        config = yaml.safe_load(Path(report['config']).read_text())
        robot = yaml.safe_load((Path(config['data']['robot_config_root']) / 'robotwin.yaml').read_text())
        self.assertEqual(config['data']['norm_stats_file'], robot['norm_stats'])
        self.assertEqual(Path(robot['norm_stats']).read_bytes(), self.norm.read_bytes())
        self.assertEqual(len(Path(config['data']['train_path']).read_text().splitlines()), 50)
        self.assertEqual(report['dataset_episodes'], 2500)
        self.assertEqual(config['train']['max_steps'], 10000)
        self.assertEqual(config['train']['pause_after_steps'], 5000)
        self.assertEqual(config['train']['save_steps'], 1000)
        self.assertEqual(config['train']['gradient_accumulation_steps'], 4)
        self.assertEqual(config['model'], original['model'])
        self.assertEqual(config['train']['lr'], original['train']['lr'])
        self.assertEqual(config['train']['align_params']['depth'], original['train']['align_params']['depth'])
        self.assertNotIn('load_checkpoint_path', config['train'])
        self.assertEqual(yaml.safe_load((self.baseline / 'lingbotvla_cli.yaml').read_text()), original)
        self.assertEqual(self.generate(), report)  # reconnect/retry is idempotent
        self.assertEqual(self.generate(micro=2)['clean_norm_sha256'], report['clean_norm_sha256'])

    def test_rejects_mixed_stats_even_with_a_clean_filename(self):
        other = self.root / 'robotwin_clean_wrong_stats.json'
        other.write_text('{"different": true}')
        self.config['data']['norm_stats_file'] = str(other)
        self.save_reference()
        with self.assertRaisesRegex(ValueError, 'normalization differs'):
            self.generate()

    def test_rejects_duplicate_task_manifest_and_wrong_episode_count(self):
        self.manifest.write_text('\n'.join(self.entries[:-1] + [self.entries[0]]))
        with self.assertRaisesRegex(ValueError, '50 distinct'):
            self.generate()
        self.manifest.write_text('\n'.join(self.entries))
        (self.data / 'robotwin_clean_task_0/meta/info.json').write_text('{"total_episodes": 49}')
        with self.assertRaisesRegex(ValueError, '50 episodes'):
            self.generate()

    def test_rejects_reference_with_wrong_training_budget(self):
        self.config['train']['max_steps'] = 5000
        self.save_reference()
        with self.assertRaisesRegex(ValueError, 'actually use 10000'):
            self.generate()

    def test_generated_yaml_passes_actual_argument_parser_for_pause_and_resume(self):
        from lingbotvla.utils.arguments import EvalArguments, DataArguments, ModelArguments, TrainingArguments, parse_args
        namespace = dict(globals(), dataclass=dataclasses.dataclass, field=dataclasses.field,
                         EvalArguments=EvalArguments, DataArguments=DataArguments,
                         ModelArguments=ModelArguments, TrainingArguments=TrainingArguments)
        tree = ast.parse((ROOT / 'tasks/vla/train_lingbotvla.py').read_text(encoding='utf-8'))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ('MyTrainingArguments', 'MyDataArguments', 'Arguments')]
        exec(compile(ast.Module(body=classes, type_ignores=[]), '<real training arguments>', 'exec'), namespace)
        # Forward annotations are resolved in this test module's globals.
        with patch.dict(globals(), {name: namespace[name] for name in ['MyTrainingArguments', 'MyDataArguments']}):
            self.config['model']['basic_modules'] = []
            self.config['train']['token_moe_layers'] = []
            self.save_reference()
            report = self.generate()
            with patch.dict(os.environ, {'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '8'}):
                with patch.object(sys, 'argv', ['train', report['config']]):
                    args = parse_args(namespace['Arguments'])
                self.assertIsNone(args.train.load_checkpoint_path)
                self.assertEqual(args.train.pause_after_steps, 5000)
                self.assertEqual(args.train.global_batch_size, 128)
                self.assertEqual(args.model.basic_modules, [])
                self.assertEqual(args.train.token_moe_layers, [])
                with patch.object(sys, 'argv', ['train', report['config'], '--train.pause_after_steps', '0',
                                                '--train.load_checkpoint_path', '/experiment/checkpoints/global_step_5000',
                                                '--model.basic_modules', 'ExplicitLayer']):
                    resumed = parse_args(namespace['Arguments'])
                self.assertEqual(resumed.train.pause_after_steps, 0)
                self.assertEqual(resumed.train.max_steps, 10000)
                self.assertEqual(resumed.model.basic_modules, ['ExplicitLayer'])


class PauseScheduleTest(unittest.TestCase):
    def test_pause_restores_all_cpu_random_streams(self):
        from io import BytesIO
        import random
        import numpy as np
        import torch
        state = capture_rng_state()
        serialized = BytesIO()
        torch.save(state, serialized)
        serialized.seek(0)
        state = torch.load(serialized, weights_only=True)
        expected = (random.random(), np.random.random(5), torch.rand(5))
        restore_rng_state(state)
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.random(5), expected[1])
        self.assertTrue(torch.equal(torch.rand(5), expected[2]))

    def test_pause_keeps_full_lr_curve_across_optimizer_scheduler_reload(self):
        import torch
        # Load the real scheduler with its package context, without the GPU entrypoint.
        from lingbotvla.optim.lr_scheduler import get_cosine_schedule_with_warmup
        def build():
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.SGD([parameter], lr=1e-4)
            scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 10000, 1e-4, min_lr=5e-5)
            return optimizer, scheduler
        full_optimizer, full_scheduler = build()
        optimizer, scheduler = build()
        saved = None
        for step in range(1, 10001):
            full_optimizer.step(); full_scheduler.step()
            optimizer.step(); scheduler.step()
            self.assertEqual(optimizer.param_groups[0]['lr'], full_optimizer.param_groups[0]['lr'])
            if step == training_stop_step(10000, 5000, 1000):
                saved = (copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict()))
                optimizer, scheduler = build()
                optimizer.load_state_dict(saved[0]); scheduler.load_state_dict(saved[1])
        self.assertIsNotNone(saved)
        self.assertEqual(training_stop_step(10000, 0, 1000), 10000)
        for pause in [-1, 5501, 10000, 10001]:
            with self.assertRaises(ValueError):
                training_stop_step(10000, pause, 1000)


class EvaluationCompletionTest(unittest.TestCase):
    def test_all_350_trials_are_required_even_if_shell_exits_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / '_result.txt'
            lines = [f'{task}\t{case}\t12/25\t0.48' for task in evaluate.TASKS for case in ['clean', 'random']]
            path.write_text('\n'.join(lines))
            self.assertEqual(evaluate.validate_results(path)['episodes'], 350)
            self.assertEqual(evaluate.validate_results(path)['clean']['successes'], 84)
            path.write_text('\n'.join(lines[:-1]))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                evaluate.validate_results(path)
            path.write_text('\n'.join(lines).replace('12/25', '12/24', 1))
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                evaluate.validate_results(path)


if __name__ == '__main__':
    unittest.main()
