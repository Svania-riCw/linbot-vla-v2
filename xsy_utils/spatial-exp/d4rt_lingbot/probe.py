"""Lightweight d4rt probe for LingBot-VLA v2 (PyTorch port of pi05's ``DepthProbe``).

Same module as pi05-experiments ``src/openpi/models/depth_probe.py`` in the configuration
``pi05_robotwin_new_3view_d4rt_full_unified`` (``train_q8_10000.sh``): K=8 queries per frame,
unified decoding (no offset head), xyz target space, target-frame table of 5, dt table 0..49,
3 cross-attention blocks of width 1024 (8 heads x 128, SwiGLU 4096), no query self-attention.

    query = LN( patch_proj(RGB 14x14 @ (u,v) in c_src) + fourier(u, v) + cam_embed(c_src)
                + dt_embed(dt) + tgt_embed(F) )
    context = context_proj(VLM prefix hidden states)          # all prefix tokens, padding masked
    x = query; for block: x = CrossAttnBlock(x, context)       # pre-LN attn + SwiGLU MLP
    pred = to_pred(LN(x)) -> [K, 14, 14, 3]                    # xyz / 1.2 m in frame F
    loss = masked mean |pred - target|                          # L1, x 0.1 in the total loss

The probe reads the prefix of the *same* forward pass that trains the action expert, so its
gradient shapes the VLM; it is never used at inference (``sample_actions`` does not call it).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_CAMS, N_TGT_FRAMES, TASK_NAMES


class CrossAttnBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, head_dim: int, mlp_dim: int):
        super().__init__()
        inner = num_heads * head_dim
        self.num_heads, self.head_dim = num_heads, head_dim
        self.norm_q = nn.LayerNorm(width, bias=False)
        self.norm_kv = nn.LayerNorm(width, bias=False)
        self.to_q = nn.Linear(width, inner, bias=False)
        self.to_k = nn.Linear(width, inner, bias=False)
        self.to_v = nn.Linear(width, inner, bias=False)
        self.to_out = nn.Linear(inner, width, bias=False)
        self.norm_ff = nn.LayerNorm(width, bias=False)
        self.ff_in = nn.Linear(width, mlp_dim, bias=False)
        self.ff_gate = nn.Linear(width, mlp_dim, bias=False)
        self.ff_out = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        b, k, _ = x.shape
        s = ctx.shape[1]
        n, h = self.num_heads, self.head_dim
        q = self.to_q(self.norm_q(x)).view(b, k, n, h).transpose(1, 2)
        kv = self.norm_kv(ctx)
        kk = self.to_k(kv).view(b, s, n, h).transpose(1, 2)
        vv = self.to_v(kv).view(b, s, n, h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=ctx_mask[:, None, None, :])
        x = x + self.to_out(out.transpose(1, 2).reshape(b, k, n * h))
        y = self.norm_ff(x)
        return x + self.ff_out(F.silu(self.ff_gate(y)) * self.ff_in(y))


class D4rtProbe(nn.Module):
    def __init__(
        self,
        context_width: int,
        width: int = 1024,
        depth: int = 3,
        num_heads: int = 8,
        head_dim: int = 128,
        mlp_dim: int = 4096,
        patch_size: int = 14,
        image_size: int = 224,
        max_dt: int = 49,
        xyz_scale: float = 1.2,
        fourier_seed: int = 0,
    ):
        super().__init__()
        if width % 2:
            raise ValueError("width must be even")
        self.width, self.p, self.image_size = width, patch_size, image_size
        self.xyz_scale = xyz_scale
        self.fourier_seed = fourier_seed
        self.patch_proj = nn.Linear(3 * patch_size * patch_size, width, bias=False)
        self.cam_embed = nn.Embedding(N_CAMS, width)
        self.tgt_embed = nn.Embedding(N_TGT_FRAMES, width)
        self.dt_embed = nn.Embedding(max_dt + 1, width)
        self.query_norm = nn.LayerNorm(width, bias=False)
        self.context_proj = nn.Linear(context_width, width, bias=False)
        self.blocks = nn.ModuleList(CrossAttnBlock(width, num_heads, head_dim, mlp_dim) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width, bias=False)
        self.to_pred = nn.Linear(width, patch_size * patch_size * 3, bias=False)
        self._fourier: tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Own init, applied after the weight loader's ``to_empty`` (the probe is not in the
        pretrained checkpoint), matching flax's defaults used by pi05: nnx.Linear lecun-normal
        (std 1/sqrt(fan_in)); nnx.Embed variance_scaling(1, fan_in, out_axis=0), i.e. std
        1/sqrt(features); LayerNorm scale 1."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.in_features))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.embedding_dim))
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)

    def _fourier_params(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen random Fourier map, regenerated from a fixed seed (not a parameter or buffer,
        so neither the optimizer nor the meta-device / to_empty loader path can touch it)."""
        if self._fourier is None or self._fourier[0].device != device:
            g = torch.Generator().manual_seed(self.fourier_seed)
            w = torch.randn(self.width // 2, generator=g)
            b = torch.rand(self.width // 2, generator=g)
            self._fourier = (w.to(device), b.to(device))
        return self._fourier

    def build_queries(self, patches, coords, cam, dt, tgt) -> torch.Tensor:
        b, k = coords.shape[:2]
        dtype = self.patch_proj.weight.dtype
        coords_norm = coords.float() / (self.image_size - 1) * 2.0 - 1.0  # [b, k, 2]
        w, bias = self._fourier_params(coords.device)
        pos = torch.cos(2 * math.pi * (coords_norm[..., None] * w + bias)).reshape(b, k, self.width)
        q = (
            self.patch_proj(patches.reshape(b, k, -1).to(dtype))
            + pos.to(dtype)
            + self.cam_embed(cam)
            + self.dt_embed(dt)
            + self.tgt_embed(tgt)
        )
        return self.query_norm(q)

    def forward(self, context, context_mask, patches, coords, cam, dt, tgt) -> torch.Tensor:
        """-> [b, k, p, p, 3] normalized xyz."""
        x = self.build_queries(patches, coords, cam, dt, tgt)
        ctx = self.context_proj(context.to(self.context_proj.weight.dtype))
        mask = context_mask.bool()
        for blk in self.blocks:
            x = blk(x, ctx, mask)
        raw = self.to_pred(self.out_norm(x))
        return raw.view(*raw.shape[:2], self.p, self.p, 3)

    def loss(self, context, context_mask, batch: dict) -> tuple[torch.Tensor, dict]:
        """Masked L1 over the xyz patch + reporting metrics (mm errors per task, constant baseline)."""
        pred = self(
            context, context_mask, batch["d4rt_patches"], batch["d4rt_coords"], batch["d4rt_cam"],
            batch["d4rt_dt"], batch["d4rt_tgt"],
        ).float()
        target = batch["d4rt_target"].float()
        mask = batch["d4rt_mask"].float()
        denom = mask.sum().clamp(min=1.0)
        per_query = (pred - target).abs().mean(dim=(-3, -2, -1))
        loss = (per_query * mask).sum() / denom

        with torch.no_grad():
            w = mask[..., None, None, None]
            cells = (mask.sum() * self.p * self.p).clamp(min=1.0)
            tmean = (target * w).sum(dim=(0, 1, 2, 3)) / cells  # best constant 3-vector
            baseline = (((target - tmean).abs().mean(dim=(-3, -2, -1))) * mask).sum() / denom
            dist_mm = torch.linalg.vector_norm((pred - target) * self.xyz_scale * 1000.0, dim=-1).mean(dim=(-2, -1))
            metrics = {
                "d4rt/loss": loss.detach(),
                "d4rt/baseline": baseline,
                "d4rt/mask_frac": mask.mean(),
                "d4rt/xyz_l2_mm": (dist_mm * mask).sum() / denom,
            }
            for i, name in enumerate(TASK_NAMES):
                m = mask * (batch["d4rt_task"] == i).float()
                metrics[f"d4rt/xyz_l2_mm_{name}"] = (dist_mm * m).sum() / m.sum().clamp(min=1.0)
        return loss, metrics
