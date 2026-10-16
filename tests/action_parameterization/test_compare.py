import copy
import json
from pathlib import Path
import tempfile
import unittest
from ysj_utils.compare_results import compare
from ysj_utils.evaluate import scheme


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = [Path(self.temp.name)/name for name in ('B0','JDelta')]
        full = scheme('full')
        common = dict(status='COMPLETE', mode='full', **full, seed=0,
                      inference_precision='fp32', use_length=50, code_commit='same-code',
                      initial_model='same-pretrained-model', train_list_sha256='same-training-data',
                      training_summary=dict(global_batch_size=128,max_steps=10000,
                                            enable_fp32=False,enable_mixed_precision=True))
        for run, mode, successes in zip(self.runs,['joint_absolute','joint_delta'],[10,12]):
            run.mkdir()
            report = copy.deepcopy(common)
            report.update(checkpoint=f'/run/{run.name}/checkpoints/global_step_10000/hf_ckpt',contract={'mode':mode})
            (run/'ysj_evaluation.json').write_text(json.dumps(report),encoding='utf-8')
            (run/'_result.txt').write_text('\n'.join(
                f'{task}\t{case}\t{successes}/25\t{successes/25}'
                for task in full['tasks'] for case in ('clean','random')),encoding='utf-8')

    def test_complete_matched_comparison(self):
        report = compare(*self.runs)
        self.assertEqual(len(report['rows']),14)
        self.assertAlmostEqual(report['summary']['random']['difference_pp'],8.)

    def test_explicit_five_k_budget_and_checkpoint_match(self):
        for run in self.runs:
            path=run/'ysj_evaluation.json'; report=json.loads(path.read_text())
            report['training_summary']['max_steps']=5000
            report['checkpoint']=report['checkpoint'].replace('global_step_10000','global_step_5000')
            path.write_text(json.dumps(report))
        self.assertEqual(len(compare(*self.runs,expected_steps=5000)['rows']),14)
        with self.assertRaises(ValueError): compare(*self.runs)

    def test_mismatched_training_or_eval_seed_rejected(self):
        path = self.runs[1]/'ysj_evaluation.json'
        original = json.loads(path.read_text(encoding='utf-8'))
        for mutation in ({'seed':1},{'training_summary':dict(original['training_summary'],max_steps=5000)}):
            path.write_text(json.dumps(dict(original,**mutation)),encoding='utf-8')
            with self.assertRaises(ValueError):
                compare(*self.runs)

    def test_partial_results_rejected(self):
        path = self.runs[1]/'_result.txt'
        lines = path.read_text(encoding='utf-8').splitlines()
        path.write_text('\n'.join(lines[:-1]),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Missing'):
            compare(*self.runs)

    def test_joint_ee_comparison_checks_common_inputs_labels_and_zero_weight_control(self):
        for index,run in enumerate(self.runs):
            path=run/'ysj_evaluation.json'
            report=json.loads(path.read_text())
            report.update(contract={'mode':'joint_ee_delta'},state_features_sha256='same-states',norm_sha256='same-statistics',
                          label_manifests={'labels':'same-audit'},
                          action_loss={'mask_action_padding':True,'joint_ee_aux_weight':0. if index==0 else .1})
            path.write_text(json.dumps(report))
        self.assertEqual(len(compare(*self.runs)['rows']),14)
        path=self.runs[1]/'ysj_evaluation.json'
        report=json.loads(path.read_text()); report['state_features_sha256']='different'
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'identical state inputs'):
            compare(*self.runs)

    def test_independent_supervision_comparison_and_norm_mismatch_rejection(self):
        defaults=dict(flow_dimension_weights=None,flow_horizon_weights=None,spline_control_points=0,spline_degree=3)
        for i,run in enumerate(self.runs):
            path=run/'ysj_evaluation.json'; report=json.loads(path.read_text())
            weights=[1.]*55; weights[28:30]=[2.,2.]
            report.update(contract={'mode':'joint_delta'},state_features_sha256='same',norm_sha256='same',label_manifests={},
                          action_loss=dict(mask_action_padding=True,joint_ee_aux_weight=-1.),
                          action_supervision=dict(defaults,flow_dimension_weights=weights) if i else defaults)
            path.write_text(json.dumps(report))
        self.assertEqual(len(compare(*self.runs)['rows']),14)
        path=self.runs[1]/'ysj_evaluation.json'; report=json.loads(path.read_text())
        report['norm_sha256']='different'; path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'norm_sha256'):
            compare(*self.runs)


if __name__ == '__main__':
    unittest.main()
