"""Regression tests for loading the official checkpoint without teacher heads."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from lingbotvla.models import module_utils


class _TinyPolicy(nn.Module):
    def __init__(self, with_video_head=False, with_spatial_forcing=False):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.model = nn.Module()
        self.model.action_proj = nn.Linear(2, 2, bias=False)
        if with_video_head:
            self.model.current_video_align_head = nn.Linear(2, 2)
        if with_spatial_forcing:
            self.model.spatial_forcing_norm = nn.BatchNorm1d(2)
            self.model.spatial_forcing_mlp = nn.Linear(2, 2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 0.25)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            module.running_mean.zero_()
            module.running_var.fill_(1.0)


class _PassthroughWeightLoader:
    def get_vlm_para_fullnames(self, model):
        return set()

    def map_ckpt_key(self, key, load_vlm_only, post_training):
        return key


class OptionalTeacherCheckpointTests(unittest.TestCase):
    def _load(self, model, checkpoint):
        state_dict_iterators = [iter(checkpoint)]
        with (
            patch.object(module_utils, "_get_weight_loader", return_value=_PassthroughWeightLoader()),
            patch.object(module_utils, "_load_state_dict", return_value=state_dict_iterators),
            patch.object(module_utils, "empty_cache"),
        ):
            module_utils.load_model_weights(
                model,
                weights_path="unused-fixture-checkpoint",
                init_device="cpu",
                post_training=True,
            )

    def test_disabled_alignment_weights_are_skipped_but_base_weights_load(self):
        model = _TinyPolicy()
        action_weight = torch.full((2, 2), 3.0)
        optional_teacher_weight = torch.full((2, 2), 7.0)

        self._load(
            model,
            [
                ("model.action_proj.weight", action_weight),
                ("model.current_video_align_head.projector.layers.0.1.0.bias", optional_teacher_weight),
            ],
        )

        self.assertTrue(torch.equal(model.model.action_proj.weight, action_weight))

    def test_alignment_weights_load_when_head_is_enabled(self):
        model = _TinyPolicy(with_video_head=True)
        action_weight = torch.full((2, 2), 3.0)
        video_weight = torch.full((2, 2), 7.0)
        video_bias = torch.full((2,), 5.0)

        self._load(
            model,
            [
                ("model.action_proj.weight", action_weight),
                ("model.current_video_align_head.weight", video_weight),
                ("model.current_video_align_head.bias", video_bias),
            ],
        )

        self.assertTrue(torch.equal(model.model.action_proj.weight, action_weight))
        self.assertTrue(torch.equal(model.model.current_video_align_head.weight, video_weight))
        self.assertTrue(torch.equal(model.model.current_video_align_head.bias, video_bias))

    def test_new_spatial_forcing_head_is_initialized_from_base_checkpoint(self):
        model = _TinyPolicy(with_spatial_forcing=True)
        action_weight = torch.full((2, 2), 3.0)

        self._load(model, [("model.action_proj.weight", action_weight)])

        self.assertTrue(torch.equal(model.model.action_proj.weight, action_weight))
        self.assertTrue(torch.all(model.model.spatial_forcing_norm.weight == 1))
        self.assertTrue(torch.all(model.model.spatial_forcing_norm.running_mean == 0))
        self.assertTrue(torch.all(model.model.spatial_forcing_norm.running_var == 1))
        self.assertTrue(torch.all(model.model.spatial_forcing_mlp.weight == 0.25))

    def test_unrecognized_extra_weight_remains_an_error(self):
        model = _TinyPolicy()
        with self.assertRaisesRegex(KeyError, "unexpected_projector.weight"):
            self._load(
                model,
                [
                    ("model.action_proj.weight", torch.ones((2, 2))),
                    ("model.unexpected_projector.weight", torch.ones((2, 2))),
                ],
            )


if __name__ == "__main__":
    unittest.main()
