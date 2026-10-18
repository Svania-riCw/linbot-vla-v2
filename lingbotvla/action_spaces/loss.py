"""Group-balanced joint + EE flow loss for the audited RoboTwin slot layout."""
import torch
import math


def gibbs_flow_loss(losses, mask, actions, noise, epsilon, log_normalizer):
    """W-CFM-inspired Gibbs pair weighting (arXiv:2507.22270, eq. 7).

    Cost is masked RMS endpoint distance: ||noise-action|| / sqrt(valid_d).
    Thus epsilon is the paper's kappa, not its unscaled epsilon. Z is estimated
    once from training data, never from this minibatch/rank. Keep the baseline
    element-mean reduction and its L1/MSE error unchanged. Constant f=g=1 is an
    approximation; conditional robot policies are outside the paper's tests.
    """
    if losses.ndim != 3 or any(t.shape != losses.shape for t in (mask, actions, noise)):
        raise ValueError('Gibbs loss expects matching [B,H,D] tensors')
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(log_normalizer):
        raise ValueError('Gibbs epsilon must be positive and normalization finite')
    valid = mask.bool()
    counts = valid.sum((1, 2)).clamp_min(1)
    with torch.no_grad():
        displacement = torch.where(valid, noise.float() - actions.float(), 0)
        costs = (displacement.square().sum((1, 2)) / counts).sqrt()
        weights = torch.exp(-costs / epsilon - log_normalizer)
    selected = torch.where(valid, losses.float(), 0)
    per_sample = selected.sum((1, 2)) / counts
    loss = (selected * weights[:, None, None]).sum() / valid.sum().clamp_min(1)
    metrics = {'action/gibbs_weight_mean': weights.mean(),
               'action/gibbs_weight_max': weights.max(),
               'action/gibbs_ess_fraction': weights.sum().square() / (len(weights) * weights.square().sum().clamp_min(1e-30)),
               'action/gibbs_cost_mean': costs.mean(),
               'action/unweighted_flow_loss': selected.detach().sum() / valid.sum().clamp_min(1)}
    return loss, per_sample * weights, metrics


def validate_weights(values, length, name):
    if values is None:
        return
    if len(values) != length or any(not math.isfinite(float(v)) or v < 0 for v in values) or not any(v > 0 for v in values):
        raise ValueError(f'{name} must contain {length} finite nonnegative values and at least one positive value')


def weighted_flow_loss(losses, mask, dimension_weights=None, horizon_weights=None):
    """Reduce existing L1/MSE flow errors; normalize by SUM OF VALID WEIGHTS.

    Padded dimensions/timesteps have exactly zero weight. Uniform weights give
    the ordinary masked loss, and scaling all weights preserves loss scale.
    """
    if losses.ndim != 3 or mask.shape != losses.shape:
        raise ValueError('Expected matching [B,H,D] flow errors and mask')
    validate_weights(dimension_weights, losses.shape[-1], 'dimension_weights')
    validate_weights(horizon_weights, losses.shape[1], 'horizon_weights')
    dims = torch.ones(losses.shape[-1], device=losses.device) if dimension_weights is None else torch.tensor(dimension_weights, device=losses.device, dtype=torch.float32)
    steps = torch.ones(losses.shape[1], device=losses.device) if horizon_weights is None else torch.tensor(horizon_weights, device=losses.device, dtype=torch.float32)
    weights = mask.bool().float() * dims[None,None,:] * steps[None,:,None]
    count = weights.sum()
    if not torch.compiler.is_compiling() and count <= 0:
        raise ValueError('No effective weighted supervision')
    selected = torch.where(weights > 0, losses.float(), 0) * weights
    tiny = torch.finfo(torch.float32).tiny
    per_sample = selected.sum((1,2)) / weights.sum((1,2)).clamp_min(tiny)
    return selected.sum() / count.clamp_min(tiny), per_sample


def joint_ee_loss(losses, mask, weight):
    """Primary joint/gripper mean + weight * EE mean; padding contributes zero.

Canonical slots: arm [0:12], unused [12:14], EE [14:28], grippers [28:30].
The existing 55-channel flow head predicts both groups; no FK is performed.
"""
    if weight < 0 or losses.ndim != 3 or losses.shape[-1] < 30 or mask.shape != losses.shape:
        raise ValueError('Invalid joint+EE loss inputs')
    primary = torch.zeros_like(mask, dtype=torch.bool)
    primary[..., :12] = True
    primary[..., 28:30] = True
    ee = torch.zeros_like(primary)
    ee[..., 14:28] = True
    primary = primary & mask.bool()
    ee = ee & mask.bool()

    def means(group):
        selected = torch.where(group, losses, 0).float()
        return (selected.sum() / group.sum().clamp(min=1),
                selected.sum((1,2)) / group.sum((1,2)).clamp(min=1))

    main, per_main = means(primary)
    aux, per_aux = means(ee)
    return main + weight*aux, per_main + weight*per_aux, main.detach(), aux.detach()
