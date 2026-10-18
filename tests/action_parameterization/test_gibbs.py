"""Independent references for paper-inspired Gibbs loss and baseline ablations."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from lingbotvla.action_spaces.loss import gibbs_flow_loss
from ysj_utils.run_baseline_supervision import calibrate, experiment_config, absolute_targets
from test_supervision import actual_config, tiny_flow, MODEL
from test_upstream import extract


class GibbsTests(unittest.TestCase):
    def test_fixed_normalizer_reference_gradients_and_padding(self):
        errors = torch.tensor([[[2., 6.], [999., 999.]], [[8., 4.], [2., 6.]]], requires_grad=True)
        mask = torch.ones_like(errors, dtype=torch.bool); mask[0, 1] = False
        actions = torch.zeros_like(errors)
        noise = torch.ones_like(errors); noise[0] *= 2
        noise[0, 1] = float('nan')
        epsilon, logz = .7, -1.3
        expected_w = np.exp(-np.array([2., 1.]) / epsilon - logz)
        expected = (8*expected_w[0] + 20*expected_w[1]) / 6
        loss, per, metrics = gibbs_flow_loss(errors, mask, actions, noise, epsilon, logz)
        self.assertAlmostEqual(float(loss.detach()), expected, places=5)
        loss.backward()
        self.assertTrue(torch.all(errors.grad[~mask] == 0))
        self.assertAlmostEqual(float(errors.grad[0, 0, 0]), expected_w[0]/6, places=6)
        self.assertAlmostEqual(float(per[1].detach()), 5*expected_w[1], places=5)
        self.assertTrue(torch.isfinite(metrics['action/gibbs_cost_mean']))

    def test_uniform_limit_and_no_minibatch_normalization(self):
        errors = torch.arange(12.).reshape(2, 2, 3)
        mask = torch.ones_like(errors, dtype=torch.bool)
        loss, _, _ = gibbs_flow_loss(errors, mask, errors, torch.zeros_like(errors), 1e12, 0.)
        torch.testing.assert_close(loss, errors.mean())
        first, _, _ = gibbs_flow_loss(errors[:1], mask[:1], errors[:1], torch.zeros_like(errors[:1]), 1., 0.)
        per = gibbs_flow_loss(errors, mask, errors, torch.zeros_like(errors), 1., 0.)[1]
        torch.testing.assert_close(first, per[0])

    def test_actual_policy_uses_same_noise_and_preserves_baseline_l1(self):
        cfg = actual_config(flow_gibbs_epsilon=.5, flow_gibbs_log_normalizer=-2.)
        core = tiny_flow(cfg)
        seen = {}
        original = core.forward
        def record(*args, **kwargs):
            seen['actions'], seen['noise'] = args[5], args[6]
            result = original(*args, **kwargs)
            seen['losses'] = result[0]
            return result
        core.forward = record
        policy = SimpleNamespace(config=cfg, model=core)
        forward = extract(MODEL, class_name='LingbotVlaV2Policy', method_names={'forward'})['forward']
        actions = torch.randn(2, 50, 55)
        mask = torch.zeros_like(actions, dtype=torch.bool); mask[..., :12] = True; mask[..., 28:30] = True
        pad = torch.zeros(2, 50, dtype=torch.bool); pad[0, 17:] = True
        output = forward(policy, torch.zeros(2, 1), None, torch.zeros(2, 55), None, None,
                         actions, joint_mask=mask, action_is_pad=pad)
        expected = gibbs_flow_loss(seen['losses'], mask & ~pad[..., None], seen['actions'], seen['noise'], .5, -2.)[0]
        torch.testing.assert_close(output[1], expected)
        output[0].backward()
        self.assertTrue(torch.isfinite(core.action_out_proj.weight.grad).all())
        self.assertEqual(cfg.loss_type, 'L1_fm')

    def test_configuration_rejects_mixed_methods_and_bad_bandwidth(self):
        for fields in (dict(flow_gibbs_epsilon=-1), dict(flow_gibbs_epsilon=float('nan')),
                       dict(flow_gibbs_epsilon=.5, spline_control_points=16),
                       dict(flow_gibbs_epsilon=.5, training_time_rtc=True),
                       dict(flow_gibbs_epsilon=.5, flow_horizon_weights=[1.]*50)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                actual_config(**fields)

    def test_calibration_is_reproducible_and_controls_both_proxies(self):
        rng = np.random.default_rng(12)
        targets = rng.normal(size=(20, 10, 3)).astype('float32')
        pad = np.zeros((20, 10), dtype=bool); pad[0, 5:] = True
        a = calibrate(targets, pad, source_points=16)
        self.assertEqual(a, calibrate(targets, pad, source_points=16))
        self.assertLessEqual(max(a['selected'][k] for k in ('rv_source', 'rv_target')), .01)
        self.assertGreater(a['selected']['ess_fraction'], 0)

    def test_only_requested_method_differs_from_untrained_matched_control(self):
        source = dict(train=dict(global_batch_size=128, use_compile=True, loss_type='L1_fm',
                                 enable_fp32=True, max_steps=5000, align_params={'original': 1}),
                      data=dict(prefetch_factor=4), model=dict(model_path='original'))
        original = copy.deepcopy(source)
        control = experiment_config(source, Path('/same'), 'Control', {})
        for group, allowed in [('Weighted', {'flow_gibbs_epsilon', 'flow_gibbs_log_normalizer'}),
                               ('Spline', {'spline_control_points'})]:
            cfg = experiment_config(source, Path('/same'), group, {'epsilon': .5, 'log_normalizer': -2.})
            self.assertEqual(cfg['model'], control['model']); self.assertEqual(cfg['data'], control['data'])
            self.assertEqual({k for k in cfg['train'] if cfg['train'][k] != control['train'][k]}, allowed)
            self.assertTrue(cfg['train']['enable_fp32']); self.assertFalse(cfg['train']['mask_action_padding'])
        self.assertEqual(source, original)

    def test_unmasked_gibbs_uniform_limit_matches_actual_baseline_with_padding(self):
        cfg = actual_config(mask_action_padding=False)
        core = tiny_flow(cfg)
        policy = SimpleNamespace(config=cfg, model=core)
        forward = extract(MODEL, class_name='LingbotVlaV2Policy', method_names={'forward'})['forward']
        actions = torch.randn(2, 50, 55)
        actions[0, 10:] = actions[0, 9]
        mask = torch.zeros_like(actions, dtype=torch.bool); mask[..., :12] = True; mask[..., 28:30] = True
        pad = torch.zeros(2, 50, dtype=torch.bool); pad[0, 10:] = True
        def run():
            return forward(policy, torch.zeros(2, 1), None, torch.zeros(2, 55), None, None,
                           actions, joint_mask=mask, action_is_pad=pad,
                           noise=torch.zeros_like(actions), time=torch.full((2,), .5))
        baseline = run()[1].detach()
        cfg.flow_gibbs_epsilon = 1e12
        torch.testing.assert_close(run()[1], baseline)
        cfg.mask_action_padding = True
        self.assertGreater(float((run()[1].detach()-baseline).abs()), 1e-5)

    def test_baseline_padding_rows_have_loss_gradient_but_unused_dimensions_do_not(self):
        errors = torch.ones(1, 50, 55, requires_grad=True)
        mask = torch.zeros_like(errors, dtype=torch.bool); mask[..., :12] = True; mask[..., 28:30] = True
        target = torch.zeros_like(errors)
        loss, _, _ = gibbs_flow_loss(errors, mask, target, target, .5, 0.)
        loss.backward()
        self.assertTrue(torch.all(errors.grad[0, 10:, :12] > 0))
        self.assertTrue(torch.all(errors.grad[~mask] == 0))

    def test_full_chunk_calibration_includes_repeated_tail_and_is_independent_of_pad_flags(self):
        targets = np.random.default_rng(3).normal(size=(12, 50, 14)).astype('float32')
        targets[:, 10:] = targets[:, 9:10]
        pad = np.zeros((12, 50), dtype=bool); pad[:, 10:] = True
        full = calibrate(targets, pad, source_points=16, mask_action_padding=False)
        reference = calibrate(targets, np.zeros_like(pad), source_points=16, mask_action_padding=False)
        self.assertEqual(full, reference)
        self.assertEqual(full['supervised_steps'], 600)
        self.assertNotEqual(full['grid'], calibrate(targets, pad, source_points=16)['grid'])

    def test_spline_baseline_mode_fits_all_50_rows_and_preserves_dimension_mask(self):
        cfg = actual_config(spline_control_points=16, mask_action_padding=False)
        core = tiny_flow(cfg)
        policy = SimpleNamespace(config=cfg, model=core)
        forward = extract(MODEL, class_name='LingbotVlaV2Policy', method_names={'forward'})['forward']
        target = torch.zeros(2, 50, 55)
        target[:, :10, :12] = torch.linspace(0, 1, 10)[None, :, None]
        target[:, 10:, :12] = 1
        mask = torch.zeros_like(target, dtype=torch.bool); mask[..., :12] = True; mask[..., 28:30] = True
        pad = torch.zeros(2, 50, dtype=torch.bool); pad[:, 10:] = True
        expected, coefficients_mask = core.action_spline.encode(target, torch.zeros_like(pad), mask)
        original = core.forward
        seen = {}
        def record(*args, **kwargs):
            seen['coefficients'] = args[5]
            return original(*args, **kwargs)
        core.forward = record
        result = forward(policy, torch.zeros(2, 1), None, torch.zeros(2, 55), None, None,
                         target, joint_mask=mask, action_is_pad=pad,
                         noise=torch.zeros(2, 16, 55), time=torch.full((2,), .5))
        torch.testing.assert_close(seen['coefficients'], expected)
        self.assertTrue(coefficients_mask[..., :12].all())
        self.assertFalse(coefficients_mask[..., 14:28].any())
        prefix, _ = core.action_spline.encode(target, pad, mask)
        self.assertGreater(float((expected-prefix).abs().max()), .1)
        result[0].backward()
        self.assertTrue(torch.isfinite(core.action_out_proj.weight.grad).all())
        expected_curve = core.action_spline.decode(expected)
        rmse = (((expected_curve-target).square()*mask).sum()/mask.sum()).sqrt()
        torch.testing.assert_close(result[6]['spline/target_rmse'], rmse)

    def test_receipt_binds_baseline_padding_and_preserves_old_masked_receipts(self):
        from deploy.action_contract import supervision_config
        for method in (dict(spline_control_points=16), dict(flow_gibbs_epsilon=.5)):
            masked = supervision_config(dict(method, mask_action_padding=True))
            unmasked = supervision_config(dict(method, mask_action_padding=False))
            self.assertNotIn('temporal_padding', masked)
            self.assertEqual(unmasked['temporal_padding'], 'supervise_repeated_terminal_actions')
        self.assertNotIn('temporal_padding', supervision_config({}))


if __name__ == '__main__':
    unittest.main()
