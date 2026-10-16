"""RoboTwin world/tool poses (wxyz) <-> LingBot pose features (xyzw).

This module has no simulator/model dependencies; training and eval share it.
"""
import numpy as np


def checked_pose(value):
    pose = np.asarray(value, dtype=np.float32)
    if pose.shape[-1:] != (14,) or not np.isfinite(pose).all():
        raise ValueError('Expected finite [...,14] dual-arm xyz/xyzw pose')
    pose = pose.copy()
    for start in (3, 10):
        q = pose[..., start:start+4]
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        if np.any(norm < 1e-6):
            raise ValueError('Degenerate EE quaternion')
        pose[..., start:start+4] = q / norm
    return pose


def raw_ee_to_pose(ee):
    """[xyz,wxyz,grip] * 2 -> [xyz,xyzw] * 2, without changing frame."""
    ee = np.asarray(ee, dtype=np.float32)
    if ee.shape[-1:] != (16,) or not np.isfinite(ee).all():
        raise ValueError('Expected finite [...,16] RoboTwin EE vector')
    for sl in (slice(3, 7), slice(11, 15)):
        if not np.allclose(np.linalg.norm(ee[..., sl], axis=-1), 1, atol=1e-3, rtol=0):
            raise ValueError('Raw EE quaternion is not unit length')
    return checked_pose(ee[..., [0,1,2,4,5,6,3,8,9,10,12,13,14,11]])


def observation_pose(observation):
    end = observation['endpose']
    raw = np.concatenate([np.asarray(end['left_endpose']), [end['left_gripper']],
                          np.asarray(end['right_endpose']), [end['right_gripper']]])
    return raw_ee_to_pose(raw)


def execution_actions(result):
    """Select exactly one controller; never execute auxiliary EE predictions."""
    control = result.get('_control_space', 'qpos')
    if '_control_space' not in result and 'action.ee_pose' in result:
        raise ValueError('EE output requires explicit server control-space metadata')
    joint = np.asarray(result['action'], dtype=np.float32)
    if not np.isfinite(joint).all():
        raise ValueError('Nonfinite joint/gripper output')
    if control == 'qpos':
        if joint.shape[-1:] != (14,):
            raise ValueError('Expected [...,14] joint/gripper output')
        return np.atleast_2d(joint), control
    if control != 'ee':
        raise ValueError(f'Unknown control space: {control}')
    pose = checked_pose(result['action.ee_pose'])
    if joint.shape[-1:] != (2,):
        raise ValueError('EE-only feature reverse mapping must return [...,2] absolute grippers')
    if pose.shape[:-1] != joint.shape[:-1]:
        raise ValueError('EE and gripper chunk shapes differ')
    ee = np.concatenate([pose[..., :3], pose[..., 6:7], pose[..., 3:6], joint[..., :1],
                         pose[..., 7:10], pose[..., 13:14], pose[..., 10:13], joint[..., 1:2]], axis=-1)
    return np.atleast_2d(ee), control
