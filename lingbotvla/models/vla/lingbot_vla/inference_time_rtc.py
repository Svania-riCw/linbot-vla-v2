"""Training-free inference-time inpainting from arXiv:2506.07339."""

import math

import torch
from torch import Tensor


def rtc_soft_mask(
    horizon: int,
    delay: int,
    execution_horizon: int,
    device,
    dtype,
) -> Tensor:
    """Build RTC's hard-prefix and exponentially decaying overlap weights."""
    if horizon < 1 or delay < 0 or execution_horizon < 1:
        raise ValueError("horizon and execution_horizon must be positive; delay must be nonnegative")
    if delay > execution_horizon or execution_horizon + delay > horizon:
        raise ValueError("RTC requires delay <= execution_horizon <= horizon - delay")

    indices = torch.arange(horizon, device=device)
    weights = torch.zeros(horizon, device=device, dtype=dtype)
    weights[:delay] = 1
    overlap_end = horizon - execution_horizon
    if delay < overlap_end:
        middle = indices[delay:overlap_end].to(dtype=dtype)
        c = (horizon - execution_horizon - middle) / (horizon - execution_horizon - delay + 1)
        weights[delay:overlap_end] = c * torch.expm1(c) / math.expm1(1.0)
    return weights
