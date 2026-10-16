"""Group-balanced joint + EE flow loss for the audited RoboTwin slot layout."""
import torch


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
