import types
import unittest

import torch
from torch import nn

from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla import AdaRMSNorm, FixAdaRMSNorm
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import FlowMatchingV2
from lingbotvla.models.vla.lingbot_vla.rtc import (
    align_previous_action_chunk,
    apply_rtc_prefix,
    combine_rtc_loss_mask,
    make_rtc_flow,
    rtc_prefix_mask,
    rtc_time_for_prefix,
    sample_rtc_delay,
)


class FakeExpert(nn.Module):
    def forward(self, *, inputs_embeds, **kwargs):
        prefix, suffix = inputs_embeds
        return ( [prefix, suffix], suffix), None, []


class RTCMigrationTests(unittest.TestCase):
    def test_delay_sampling_obeys_exclusive_limit_and_episode_padding(self):
        pad = torch.tensor([[False, False, True, True], [False, False, False, True]])
        torch.manual_seed(4)
        delay = sample_rtc_delay(pad, batch_size=2, horizon=4, max_delay=10, device="cpu")
        self.assertTrue(torch.all(delay < torch.tensor([2, 3])).item())
        self.assertTrue(torch.all(delay >= 0).item())
        self.assertTrue(torch.all(sample_rtc_delay(None, 64, 50, 10, "cpu") < 10).item())

    def test_clean_prefix_flow_and_valid_loss_mask(self):
        actions = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        noise = torch.full_like(actions, -2.0)
        time = torch.full((2, 4), 0.5)
        delay = torch.tensor([0, 2])
        x_t, velocity, keep = make_rtc_flow(actions, noise, time, delay)
        self.assertTrue(torch.equal(x_t[1, :2], actions[1, :2]))
        self.assertTrue(torch.equal(x_t[1, 2:], 0.5 * noise[1, 2:] + 0.5 * actions[1, 2:]))
        self.assertTrue(torch.equal(velocity, noise - actions))
        self.assertEqual(keep.tolist(), [[True] * 4, [False, False, True, True]])

    def test_joint_prefix_and_episode_masks_share_loss_denominator(self):
        losses = torch.ones((1, 4, 2))
        joint = torch.tensor([[[True, True], [True, False], [True, True], [False, True]]])
        padded = torch.tensor([[False, False, False, True]])
        keep = torch.tensor([[False, True, True, True]])
        mask = combine_rtc_loss_mask(losses, joint, padded, keep)
        self.assertEqual(mask.sum().item(), 3)
        self.assertEqual((losses * mask).sum().item() / mask.sum().item(), 1.0)
        self.assertEqual(joint.sum().item(), 6)  # the caller's mask is not mutated

    def test_sampling_helpers_keep_normalized_previous_prefix_fixed(self):
        fresh = torch.zeros((2, 4, 3))
        previous = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        delay = torch.tensor([1, 3])
        result = apply_rtc_prefix(fresh, previous, delay)
        time = rtc_time_for_prefix(torch.tensor([0.8, 0.6]), delay, 4)
        self.assertTrue(torch.equal(result[0, :1], previous[0, :1]))
        self.assertTrue(torch.equal(result[1, :3], previous[1, :3]))
        self.assertTrue(torch.equal(result[1, 3:], fresh[1, 3:]))
        self.assertTrue(torch.allclose(time, torch.tensor([[0.0, 0.8, 0.8, 0.8], [0.0, 0.0, 0.0, 0.6]])))
        self.assertEqual(rtc_prefix_mask(delay, 4).sum(dim=1).tolist(), [1, 3])

    def test_previous_chunk_is_shifted_by_executed_ticks(self):
        previous = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
        aligned, delay = align_previous_action_chunk(previous, executed_steps=2, max_delay=3)
        self.assertEqual(delay.tolist(), [2, 2])
        self.assertTrue(torch.equal(aligned[:, :2], previous[:, 2:4]))
        self.assertTrue(torch.equal(aligned[:, 2:], torch.zeros_like(aligned[:, 2:])))

    def test_model_sampler_clamps_previous_prefix_through_all_euler_steps(self):
        batch, horizon, action_dim, width = 1, 3, 2, 4
        model = FlowMatchingV2.__new__(FlowMatchingV2)
        nn.Module.__init__(model)
        model.config = types.SimpleNamespace(
            training_time_rtc=True,
            n_action_steps=horizon,
            max_action_dim=action_dim,
            num_steps=2,
            use_cache=True,
        )
        model.qwenvl_with_expert = FakeExpert()

        def fake_prefix(self, *args, **kwargs):
            prefix = torch.zeros(batch, 2, width)
            pad = torch.ones(batch, 2, dtype=torch.bool)
            att = torch.zeros(batch, 2, dtype=torch.bool)
            positions = torch.zeros(3, batch, 2, dtype=torch.long)
            return prefix, pad, att, positions, None, []

        model.embed_prefix = types.MethodType(fake_prefix, model)
        calls = []

        def fake_predict(self, state, prefix_pad_masks, past_key_values, x_t, timestep, **kwargs):
            calls.append((x_t.detach().clone(), timestep.detach().clone()))
            return torch.ones_like(x_t)

        model.predict_velocity = types.MethodType(fake_predict, model)
        noise = torch.zeros(batch, horizon, action_dim)
        previous = torch.arange(batch * horizon * action_dim, dtype=torch.float32).reshape(batch, horizon, action_dim)
        result = model.sample_actions(
            images=None,
            img_masks=None,
            lang_tokens=None,
            lang_masks=None,
            state=torch.zeros(batch, 4),
            noise=noise,
            previous_actions=previous,
            rtc_delay=torch.tensor([2]),
        )
        self.assertTrue(torch.equal(result[:, :2], previous[:, :2]))
        self.assertTrue(torch.equal(result[:, 2:], torch.full((batch, 1, action_dim), -1.0)))
        self.assertEqual(calls[0][1].tolist(), [[0.0, 0.0, 1.0]])
        self.assertEqual(calls[1][1].tolist(), [[0.0, 0.0, 0.5]])
        self.assertTrue(torch.equal(calls[1][0][:, :2], previous[:, :2]))

    def test_per_token_embedding_and_adaptive_norm_shapes(self):
        model = FlowMatchingV2.__new__(FlowMatchingV2)
        nn.Module.__init__(model)
        model.config = types.SimpleNamespace(proj_width=8, action_fp32=False)
        model.state_proj = nn.Linear(3, 8)
        model.action_in_proj = nn.Linear(2, 8)
        model.action_time_mlp_in = nn.Linear(16, 8)
        model.action_time_mlp_out = nn.Linear(8, 8)
        state = torch.randn(2, 3)
        actions = torch.randn(2, 4, 2)
        times = torch.tensor([[0.0, 0.2, 0.3, 0.4], [0.0, 0.0, 0.5, 0.8]])
        pad = torch.tensor([[False, False, False, True], [False, False, False, False]])
        ada, suffix, suffix_pad, _ = model.embed_suffix(state, actions, times, action_is_pad=pad)
        self.assertEqual(ada.shape, (2, 5, 8))
        self.assertEqual(suffix.shape, (2, 5, 8))
        self.assertEqual(suffix_pad.tolist(), [[True, True, True, True, False], [True] * 5])
        hidden = torch.randn(2, 5, 8)
        self.assertEqual(AdaRMSNorm(8, 8)(hidden, ada).shape, hidden.shape)
        self.assertEqual(FixAdaRMSNorm(8, 8)(hidden, ada).shape, hidden.shape)
        self.assertEqual(AdaRMSNorm(8, 8)(hidden, ada[:, 0]).shape, hidden.shape)

    def test_config_roundtrip_fields_keep_default_off(self):
        default = LingbotVLAV2Config()
        enabled = LingbotVLAV2Config(training_time_rtc=True, training_time_rtc_max_delay=7)
        self.assertFalse(default.training_time_rtc)
        self.assertEqual(default.training_time_rtc_max_delay, 10)
        restored = LingbotVLAV2Config.from_dict(enabled.to_dict())
        self.assertTrue(restored.training_time_rtc)
        self.assertEqual(restored.training_time_rtc_max_delay, 7)

    def test_flow_matching_forward_uses_clean_prefix_and_returns_rtc_mask(self):
        batch, horizon, action_dim, width = 2, 4, 2, 8
        config = types.SimpleNamespace(
            training_time_rtc=True,
            training_time_rtc_max_delay=4,
            action_fp32=False,
            proj_width=width,
            max_state_dim=3,
            max_action_dim=action_dim,
            n_action_steps=horizon,
            align_params={},
            use_cache=False,
            adanorm_time=True,
            loss_type="fm",
        )
        model = FlowMatchingV2.__new__(FlowMatchingV2)
        nn.Module.__init__(model)
        model.config = config
        model.state_proj = nn.Linear(3, width)
        model.action_in_proj = nn.Linear(action_dim, width)
        model.action_out_proj = nn.Linear(width, action_dim)
        model.action_time_mlp_in = nn.Linear(width * 2, width)
        model.action_time_mlp_out = nn.Linear(width, width)
        model.qwenvl_with_expert = FakeExpert()
        model.block_future_depth_to_action = False
        model._block_suffix_to_future_video_if_enabled_ = types.MethodType(
            lambda self, mask, **kwargs: mask, model
        )
        model._moe_losses_and_metrics = types.MethodType(
            lambda self, router, losses: (losses.new_zeros(()), losses.new_zeros(()), {}), model
        )

        def fake_prefix(self, *args, **kwargs):
            prefix = torch.zeros(batch, 2, width)
            pad = torch.ones(batch, 2, dtype=torch.bool)
            att = torch.zeros(batch, 2, dtype=torch.bool)
            positions = torch.zeros(3, batch, 2, dtype=torch.long)
            return prefix, pad, att, positions, None, []

        model.embed_prefix = types.MethodType(fake_prefix, model)
        embedded = {}
        original_embed_suffix = model.embed_suffix

        def capture_suffix(self, state, noisy_actions, timestep, action_is_pad=None):
            embedded["x"] = noisy_actions.detach().clone()
            embedded["time"] = timestep.detach().clone()
            embedded["ada_shape"] = (batch, horizon + 1, width)
            return original_embed_suffix(state, noisy_actions, timestep, action_is_pad)

        model.embed_suffix = types.MethodType(capture_suffix, model)
        actions = torch.arange(batch * horizon * action_dim, dtype=torch.float32).reshape(batch, horizon, action_dim)
        noise = torch.full_like(actions, -1.0)
        time = torch.full((batch, horizon), 0.5)
        delay = torch.tensor([2, 0])
        pad = torch.tensor([[False, False, False, True], [False, False, False, False]])
        output = model.forward(
            images=None,
            img_masks=None,
            lang_tokens=None,
            lang_masks=None,
            state=torch.zeros(batch, 3),
            actions=actions,
            noise=noise,
            time=time,
            action_is_pad=pad,
            rtc_delay=delay,
        )
        rtc_mask = output[-1]
        self.assertTrue(torch.equal(embedded["x"][0, :2], actions[0, :2]))
        self.assertEqual(embedded["time"][0].tolist(), [0.0, 0.0, 0.5, 0.0])
        self.assertEqual(rtc_mask.tolist(), [[False, False, True, False], [True] * 4])
        self.assertEqual(embedded["ada_shape"], (batch, horizon + 1, width))

        config.training_time_rtc = False
        legacy_time = torch.tensor([0.25, 0.75])
        legacy_output = model.forward(
            images=None,
            img_masks=None,
            lang_tokens=None,
            lang_masks=None,
            state=torch.zeros(batch, 3),
            actions=actions,
            noise=noise,
            time=legacy_time,
            action_is_pad=None,
        )
        expected = legacy_time[:, None, None] * noise + (1 - legacy_time[:, None, None]) * actions
        self.assertTrue(torch.equal(embedded["x"], expected))
        self.assertIsNone(legacy_output[-1])


if __name__ == "__main__":
    unittest.main()
