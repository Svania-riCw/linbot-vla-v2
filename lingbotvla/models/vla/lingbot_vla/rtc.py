"""Small, dependency-light helpers for RTC action-chunk training."""

import torch
from torch import Tensor


def rtc_prefix_mask(delay: Tensor, horizon: int) -> Tensor:
    """Return the per-example clean-prefix mask used by RTC training/inference."""
    if delay.ndim == 0:
        delay = delay[None]
    if delay.ndim != 1:
        raise ValueError("delay must be a scalar or a one-dimensional batch tensor")
    if not torch.compiler.is_compiling() and (torch.any(delay < 0) or torch.any(delay > horizon)):
        raise ValueError(f"delay values must be in [0, {horizon}]")
    return torch.arange(horizon, device=delay.device)[None, :] < delay[:, None]


def sample_rtc_delay(action_is_pad, batch_size: int, horizon: int, max_delay: int, device) -> Tensor:
    """Sample a bounded prefix, leaving at least one valid action when available."""
    if max_delay < 1:
        raise ValueError("training_time_rtc_max_delay must be at least 1")
    high = min(max_delay, horizon)
    if high < 1:
        raise ValueError("RTC requires a non-empty action horizon")
    bounds = torch.full((batch_size,), high, device=device, dtype=torch.long)
    if action_is_pad is not None:
        if action_is_pad.shape != (batch_size, horizon):
            raise ValueError(
                f"action_is_pad must have shape {(batch_size, horizon)}, got {tuple(action_is_pad.shape)}"
            )
        valid_lengths = (~action_is_pad.to(device=device, dtype=torch.bool)).sum(dim=1)
        bounds = torch.minimum(bounds, valid_lengths.clamp_min(1))
    return torch.floor(torch.rand((batch_size,), device=device) * bounds).long()


def make_rtc_flow(actions: Tensor, noise: Tensor, time: Tensor, delay: Tensor):
    """Build clean-prefix flow inputs, velocities, and suffix loss mask."""
    batch_size, horizon = actions.shape[:2]
    if noise.shape != actions.shape:
        raise ValueError(f"noise must match actions {tuple(actions.shape)}")
    if time.ndim == 1:
        time = time[:, None].expand(batch_size, horizon)
    if time.shape != (batch_size, horizon):
        raise ValueError(f"RTC time must have shape {(batch_size, horizon)}, got {tuple(time.shape)}")
    prefix = rtc_prefix_mask(delay, horizon)
    if prefix.shape[0] != batch_size:
        raise ValueError(f"delay batch must be {batch_size}, got {prefix.shape[0]}")
    q = time.masked_fill(prefix, 0.0)
    x_t = q[..., None] * noise + (1.0 - q[..., None]) * actions
    return x_t, noise - actions, ~prefix


def combine_rtc_loss_mask(losses: Tensor, joint_mask, action_is_pad, prefix_mask: Tensor):
    """Combine active joints, episode padding, and RTC prefix exclusions."""
    if joint_mask is None:
        base = torch.ones_like(losses, dtype=torch.bool)
    else:
        if joint_mask.shape != losses.shape:
            raise ValueError(f"joint_mask must match losses {tuple(losses.shape)}, got {tuple(joint_mask.shape)}")
        base = joint_mask.to(device=losses.device, dtype=torch.bool).clone()
    if prefix_mask.shape != losses.shape[:2]:
        raise ValueError(f"prefix_mask must have shape {tuple(losses.shape[:2])}")
    base &= prefix_mask.to(device=losses.device, dtype=torch.bool)[..., None]
    if action_is_pad is not None:
        if action_is_pad.shape != losses.shape[:2]:
            raise ValueError(f"action_is_pad must have shape {tuple(losses.shape[:2])}")
        base &= ~action_is_pad.to(device=losses.device, dtype=torch.bool)[..., None]
    return base


def apply_rtc_prefix(actions: Tensor, previous_actions: Tensor, delay: Tensor) -> Tensor:
    """Hold a normalized previous action chunk fixed over each RTC prefix."""
    if actions.ndim != 3 or previous_actions.ndim != 3 or actions.shape != previous_actions.shape:
        raise ValueError("actions and previous_actions must have the same (batch, horizon, action_dim) shape")
    prefix = rtc_prefix_mask(delay, actions.shape[1])
    if prefix.shape[0] != actions.shape[0]:
        raise ValueError("delay batch does not match actions")
    return torch.where(prefix[..., None], previous_actions.to(actions), actions)


def align_previous_action_chunk(previous_actions: Tensor, executed_steps: int, max_delay: int):
    """Shift a stored normalized plan to the current control tick for hard-prefix reuse."""
    if previous_actions.ndim != 3:
        raise ValueError("previous_actions must have shape (batch, horizon, action_dim)")
    if executed_steps < 0 or max_delay < 1:
        raise ValueError("executed_steps must be non-negative and max_delay must be at least 1")
    batch_size, horizon, _ = previous_actions.shape
    delay_value = min(max(horizon - executed_steps, 0), max_delay - 1)
    aligned = torch.zeros_like(previous_actions)
    if delay_value:
        aligned[:, :delay_value] = previous_actions[:, executed_steps : executed_steps + delay_value]
    delay = torch.full((batch_size,), delay_value, dtype=torch.long, device=previous_actions.device)
    return aligned, delay


def rtc_time_for_prefix(time: Tensor, delay: Tensor, horizon: int) -> Tensor:
    """Broadcast denoising time to action tokens and set fixed-prefix time to zero."""
    if time.ndim != 1:
        raise ValueError("time must have shape (batch,)")
    prefix = rtc_prefix_mask(delay, horizon)
    if prefix.shape[0] != time.shape[0]:
        raise ValueError("delay batch does not match time batch")
    return time[:, None].expand_as(prefix).masked_fill(prefix, 0.0)
