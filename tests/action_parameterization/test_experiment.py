import ast
from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.action_contract import resolve_contract
from ysj_utils.action_experiment import prepare, record_receipt, verify_pair
from ysj_utils.evaluate import scheme, validate_results
from test_upstream import extract


def fake_stats(path):
    stats = {}
    for prefix in ('action.', 'observation.state.'):
        for name, dim in [('arm.position', 12), ('effector.position', 2)]:
            stats[prefix+name] = dict(mean=[0.]*dim, std=[1.]*dim, q01=[-2.]*dim, q99=[2.]*dim)
    path.write_text(json.dumps({'norm_stats': stats, 'count': 100}), encoding='utf-8')


class PairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)/'pair'
        self.pair = prepare(ROOT/'xsy_utils/configs/robotwin_clean_baseline.yaml',
                            ROOT/'configs/robot_configs/robotwin.yaml',
                            ROOT/'xsy_utils/configs/robotwin_clean.txt', self.bundle)

    def checkpoint(self, group='JDelta'):
        directory = self.bundle/group
        fake_stats(directory/'norm_stats.json')
        record_receipt(directory, self.pair)
        run = directory/'train'
        run.mkdir()
        (run/'lingbotvla_cli.yaml').write_bytes((directory/'train.yaml').read_bytes())
        return run/'checkpoints/global_step_10000/hf_ckpt'

    def test_matched_training_budget_and_data(self):
        configs = [yaml.safe_load((self.bundle/g/'train.yaml').read_text(encoding='utf-8')) for g in ('B0','JDelta')]
        for cfg in configs:
            train = cfg['train']
            self.assertEqual(train['global_batch_size'], 128)
            self.assertEqual(train['max_steps'], 10000)
            self.assertFalse(train['enable_fp32'])
            self.assertTrue(train['enable_mixed_precision'])
            self.assertFalse(train['training_time_rtc'])
            self.assertFalse(train['enable_resume'])
            self.assertEqual(train['micro_batch_size']*8*train['gradient_accumulation_steps'], 128)
            cfg['train'].pop('output_dir')
            cfg['data'].pop('robot_config_root')
            cfg['data'].pop('norm_stats_file')
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(self.pair['datasets'], 50)

    def test_five_k_preparation_keeps_both_groups_matched(self):
        out=Path(self.temp.name)/'five_k'
        pair=prepare(ROOT/'xsy_utils/configs/robotwin_clean_baseline.yaml',ROOT/'configs/robot_configs/robotwin.yaml',
                     ROOT/'xsy_utils/configs/robotwin_clean.txt',out,steps=5000)
        self.assertEqual(pair['steps'],5000)
        for group in pair['groups']:
            cfg=yaml.safe_load((out/group/'train.yaml').read_text())
            self.assertEqual(cfg['train']['max_steps'],5000)
            self.assertEqual(cfg['train']['global_batch_size'],128)

    def test_robot_diff_only_arm_delta_and_stats_path(self):
        a, b = [yaml.safe_load((self.bundle/g/'robot_configs/robotwin.yaml').read_text(encoding='utf-8')) for g in ('B0','JDelta')]
        a.pop('norm_stats'); b.pop('norm_stats')
        self.assertFalse(a['actions'][0]['action.arm.position']['subtract_state'])
        self.assertTrue(b['actions'][0]['action.arm.position']['subtract_state'])
        b['actions'][0]['action.arm.position']['subtract_state'] = False
        self.assertEqual(a, b)

    def test_no_overwrite_and_mutation_rejected(self):
        with self.assertRaises(FileExistsError):
            prepare(ROOT/'xsy_utils/configs/robotwin_clean_baseline.yaml', ROOT/'configs/robot_configs/robotwin.yaml',
                    ROOT/'xsy_utils/configs/robotwin_clean.txt', self.bundle)
        (self.bundle/'train.txt').write_text('robotwin another_dataset\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            verify_pair(self.bundle, 'B0')

    def test_delta_eval_uses_training_robot_and_stats(self):
        ckpt = self.checkpoint()
        result = resolve_contract(ckpt)
        self.assertEqual(result['mode'], 'joint_delta')
        self.assertEqual(Path(result['robot_config_root']), self.bundle/'JDelta/robot_configs')
        self.assertEqual(Path(result['norm_stats']), self.bundle/'JDelta/norm_stats.json')

    def test_stats_tampering_and_wrong_representation_rejected(self):
        ckpt = self.checkpoint()
        stats = self.bundle/'JDelta/norm_stats.json'
        stats.write_text(stats.read_text(encoding='utf-8')+' ', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'norm_sha256'):
            resolve_contract(ckpt)

    def test_robot_tampering_rejected(self):
        ckpt = self.checkpoint()
        robot = self.bundle/'JDelta/robot_configs/robotwin.yaml'
        robot.write_text(robot.read_text(encoding='utf-8').replace('subtract_state: true','subtract_state: false'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'robot_sha256'):
            resolve_contract(ckpt)

    def test_missing_delta_receipt_rejected(self):
        ckpt = self.checkpoint()
        (self.bundle/'JDelta/norm_stats.json.contract.json').unlink()
        with self.assertRaisesRegex(ValueError, 'receipt'):
            resolve_contract(ckpt)

    def test_missing_training_root_never_silently_falls_back(self):
        ckpt = self.checkpoint()
        path = self.bundle/'JDelta/train/lingbotvla_cli.yaml'
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        cfg['data']['robot_config_root'] = str(self.bundle/'nonexistent')
        path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        with self.assertRaises(FileNotFoundError):
            resolve_contract(ckpt)

    def test_legacy_baseline_without_receipt_supported(self):
        ckpt = self.checkpoint('B0')
        (self.bundle/'B0/norm_stats.json.contract.json').unlink()
        self.assertEqual(resolve_contract(ckpt)['mode'], 'legacy_absolute')

    def test_server_reset_passes_training_config_to_transform(self):
        namespace = extract(ROOT/'deploy/lingbot_vla_v2_policy.py', class_name='LingbotVLAv2Server', method_names={'reset'})
        calls = []
        def feature_transform(path, *args, **kwargs):
            calls.append((path, kwargs))
            return SimpleNamespace(org_features={'actions': ['action']}, actions=['action.arm.position'])
        namespace.update(Path=Path, yaml=yaml, FeatureTransform=feature_transform)
        obj = SimpleNamespace(robot_config_root=str(self.bundle/'JDelta/robot_configs'),
                              vla=SimpleNamespace(), data_config=None, config=SimpleNamespace(chunk_size=50),
                              processor=None, robot_norm_path='stats-for-delta.json')
        namespace['reset'](obj, 'robotwin')
        self.assertEqual(Path(calls[0][0]), self.bundle/'JDelta/robot_configs/robotwin.yaml')
        self.assertEqual(calls[0][1]['norm_stats_path'], 'stats-for-delta.json')
        self.assertIsNone(obj.last_action_chunk)


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'_result.txt'

    def test_full_protocol_is_exactly_350(self):
        full = scheme('full')
        self.assertEqual(len(set(full['tasks'])), 7)
        self.assertEqual(full['episodes'], 25)
        self.assertEqual(full['configs'], ['demo_clean','demo_randomized'])
        lines = [f'{task}\t{case}\t10/25\t0.4' for task in full['tasks'] for case in ['clean','random']]
        self.path.write_text('\n'.join(lines), encoding='utf-8')
        result = validate_results(self.path, **full)
        self.assertEqual(result['total_episodes'], 350)

    def test_missing_units_duplicates_wrong_counts_and_rates_rejected(self):
        for text in ['', 'open_laptop\tclean\t0/0\t0', 'open_laptop\tclean\t1/1\t0',
                     'open_laptop\tclean\t1/1\t1\nopen_laptop\tclean\t1/1\t1']:
            self.path.write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_results(self.path, **scheme('smoke'))

    def test_task_failure_still_counts_as_completed_rollout(self):
        self.path.write_text('open_laptop\tclean\t0/1\t0', encoding='utf-8')
        self.assertEqual(validate_results(self.path, **scheme('smoke'))['status'], 'COMPLETE')


class RelativeTypeTests(unittest.TestCase):
    def test_original_mapping_defaults_joint_to_arithmetic_and_ee_to_quaternion(self):
        ns = extract(ROOT/'lingbotvla/data/vla_data/utils.py', class_name='FeatureTransform', method_names={'get_feature_mapping'})
        ns['OrderedDict'] = OrderedDict
        for key, expected in [('arm.position', None), ('end.position','quaternion_local')]:
            robot = dict(states=[{'observation.state.'+key:{'origin_keys':[{'state':{'start':0,'end':7}}]}}],
                         actions=[{'action.'+key:{'origin_keys':[{'action':{'start':0,'end':7}}], 'subtract_state':True}}],
                         images=[])
            target={k:[] for k in robot}; original={k:set() for k in robot}
            obj=SimpleNamespace(feature_to_keep=set())
            ns['get_feature_mapping'](obj, robot, target, original)
            self.assertEqual(obj.action_relative_type['action.'+key], expected)


if __name__ == '__main__':
    unittest.main()
