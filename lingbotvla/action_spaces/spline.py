"""Fixed-knot clamped B-spline control-point actions in normalized joint space.

Only joint/gripper trajectories are supported. Quaternion EE interpolation,
RTC, adaptive knots and cross-chunk continuity are separate research changes.
"""
import numpy as np
import torch


def bspline_basis(horizon, points, degree=3):
    if not all(isinstance(v, int) for v in (horizon, points, degree)) or not (1 <= degree < points <= horizon):
        raise ValueError('Require integer 1 <= degree < control points <= horizon')
    knots = np.concatenate([np.zeros(degree+1), np.arange(1,points-degree)/(points-degree), np.ones(degree+1)])
    times = np.linspace(0,1,horizon)
    basis = ((times[:,None] >= knots[:-1]) & (times[:,None] < knots[1:])).astype(np.float64)
    for order in range(1,degree+1):
        next_basis = np.zeros((horizon,len(knots)-order-1))
        for i in range(next_basis.shape[1]):
            left = knots[i+order]-knots[i]
            right = knots[i+order+1]-knots[i+1]
            if left > 0:
                next_basis[:,i] += (times-knots[i])/left*basis[:,i]
            if right > 0:
                next_basis[:,i] += (knots[i+order+1]-times)/right*basis[:,i+1]
        basis = next_basis
    basis[-1] = 0
    basis[-1,-1] = 1
    return basis, knots


class SplineActionCodec:
    def __init__(self, horizon, points, degree=3):
        self.horizon, self.points, self.degree = horizon, points, degree
        basis, _ = bspline_basis(horizon, points, degree)
        fits = np.zeros((horizon+1,points,horizon), dtype=np.float32)
        support = np.zeros((horizon+1,points), dtype=bool)
        for n in range(1,horizon+1):
            # Short terminal chunks use only observed rows, never repeated padding.
            if n == horizon:
                fits[n,:,:n] = np.linalg.pinv(basis, rcond=1e-5)
            else:
                # Prefix-only bases can be severely ill-conditioned. Ridge bounds
                # amplification; underdetermined tail coefficients are not facts.
                rows = basis[:n]
                fits[n,:,:n] = np.linalg.solve(rows.T@rows + 1e-3*np.eye(points), rows.T)
            support[n] = np.any(basis[:n] > 0, axis=0)
        # Derived constants stay outside state_dict and module dtype/meta-device
        # conversion. Build FP32 tensors lazily on each rank's actual device.
        self._arrays = basis.astype(np.float32), fits, support
        self._devices = {}

    def tensors(self, device):
        key = str(device)
        if key not in self._devices:
            self._devices[key] = tuple(torch.tensor(a,device=device) for a in self._arrays)
        return self._devices[key]

    def encode(self, actions, action_is_pad, joint_mask):
        if actions.ndim != 3 or actions.shape[1] != self.horizon or joint_mask is None or joint_mask.shape != actions.shape:
            raise ValueError('Spline encoding requires [B,H,D] actions and joint_mask')
        if action_is_pad is None or action_is_pad.shape != actions.shape[:2]:
            raise ValueError('Spline encoding requires [B,H] action_is_pad')
        pad = action_is_pad.bool()
        lengths = (~pad).sum(1)
        expected = torch.arange(self.horizon,device=actions.device)[None,:] >= lengths[:,None]
        if not torch.compiler.is_compiling():
            if torch.any(lengths == 0) or not torch.equal(pad,expected):
                raise ValueError('Spline requires nonempty valid prefix followed by padding')
            valid_mask = joint_mask.bool() & ~pad[...,None]
            active = valid_mask.any(1)
            if not torch.equal(valid_mask,active[:,None,:] & ~pad[...,None]):
                raise ValueError('Spline requires constant active action dimensions within each chunk')
        clean = torch.where(joint_mask.bool() & ~pad[...,None],actions,0)
        _, fits, support = self.tensors(actions.device)
        with torch.autocast(device_type=actions.device.type,enabled=False):
            coeff = torch.bmm(fits[lengths],clean.float())
        mask = support[lengths,:,None] & (joint_mask.bool() & ~pad[...,None]).any(1)[:,None,:]
        return coeff.to(actions.dtype), mask

    def decode(self, coefficients):
        if coefficients.ndim != 3 or coefficients.shape[1] != self.points:
            raise ValueError('Expected [B,K,D] spline coefficients')
        basis, _, _ = self.tensors(coefficients.device)
        with torch.autocast(device_type=coefficients.device.type,enabled=False):
            actions = torch.matmul(basis,coefficients.float())
        return actions.to(coefficients.dtype)
