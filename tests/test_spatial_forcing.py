"""Small CPU checks for Spatial Forcing target construction and loss masking."""

import unittest

import torch
from torch import nn

from lingbotvla.models.vla.vision_models.spatial_forcing import (
    _factor_grid,
    get_spatial_forcing_target,
    spatial_forcing_cosine_loss,
)


class _FakeVGGT(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.dim = dim

    def aggregator(self, images):
        batch, views = images.shape[:2]
        # One special token + 16 patch tokens. The feature value is spatially
        # varying so interpolation and positional encoding are exercised.
        values = torch.arange(17 * self.dim, dtype=images.dtype, device=images.device)
        features = values.reshape(1, 1, 17, self.dim).expand(batch, views, -1, -1).clone()
        return [features], 1


class SpatialForcingTests(unittest.TestCase):
    def test_factor_grid_prefers_near_square_factors(self):
        self.assertEqual(_factor_grid(16), (4, 4))
        self.assertEqual(_factor_grid(12), (3, 4))

    def test_target_resizes_patches_and_respects_camera_mask(self):
        teacher = _FakeVGGT()
        images = torch.randint(0, 255, (2, 3, 3, 8, 8), dtype=torch.uint8)
        masks = torch.tensor([[True, False, True], [True, True, False]])
        targets, target_mask = get_spatial_forcing_target(
            teacher,
            images,
            masks,
            {
                "input_size": 8,
                "vla_merge_size": 2,
                "use_positional_encoding": True,
                "target_bfloat16": False,
            },
            image_grid_thw=torch.tensor([[1, 8, 8], [1, 8, 8], [1, 8, 8]]),
        )
        self.assertEqual(targets.shape, (2, 3 * 4 * 4, 8))
        self.assertEqual(target_mask.shape, (2, 3 * 4 * 4))
        self.assertEqual(target_mask.sum(dim=1).tolist(), [32, 32])
        self.assertTrue(torch.allclose(targets[0, 16:32], torch.zeros_like(targets[0, 16:32])))
        self.assertTrue(torch.isfinite(targets).all().item())

    def test_cosine_loss_is_masked_and_backpropagates(self):
        projection = nn.Linear(4, 4)
        normalization = nn.BatchNorm1d(4)
        hidden = torch.randn(2, 5, 4)
        visual_mask = torch.tensor([[True, True, False, True, False], [True, False, True, False, False]])
        targets = torch.randn(2, 3, 4)
        target_mask = torch.tensor([[True, True, True], [True, True, False]])
        loss = spatial_forcing_cosine_loss(
            projection, normalization, hidden, visual_mask, targets, target_mask
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(projection.weight.grad)

    def test_cosine_loss_rejects_token_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "token count mismatch"):
            spatial_forcing_cosine_loss(
                nn.Identity(), nn.Identity(),
                torch.randn(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
                torch.randn(1, 1, 4), torch.ones(1, 1, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
