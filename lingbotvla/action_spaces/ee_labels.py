"""Audited per-episode EE labels, shared by preparation and the data loader."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from deploy.ee_actions import raw_ee_to_pose


def audit_episode(states, actions, raw_joint, raw_ee):
    states, actions, raw_joint, raw_ee = map(np.asarray, (states, actions, raw_joint, raw_ee))
    n = len(states)
    if n < 1 or states.shape != (n,14) or actions.shape != (n,14) or raw_joint.shape != (n+1,14) or raw_ee.shape != (n+1,16):
        raise ValueError('Expected N LeRobot rows and N+1 raw frames')
    if not all(np.isfinite(x).all() for x in (states, actions, raw_joint, raw_ee)):
        raise ValueError('Nonfinite episode data')
    if not np.allclose(states, raw_joint[:-1], atol=1e-5, rtol=0):
        raise ValueError('State/raw alignment mismatch')
    if not np.allclose(actions, raw_joint[1:], atol=1e-5, rtol=0):
        raise ValueError('Action/raw alignment mismatch: cannot use observed next EE as command label')
    if not np.allclose(raw_ee[:, [7,15]], raw_joint[:, [6,13]], atol=1e-5, rtol=0):
        raise ValueError('EE/joint gripper mismatch')
    return raw_ee_to_pose(raw_ee)


class EESidecar:
    def __init__(self, root, repo_id):
        self.root = Path(root) / Path(repo_id).name
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema') != 'robotwin-world-tool-xyzw-v1':
            raise ValueError('Unsupported EE sidecar schema')
        if Path(self.manifest['dataset']).resolve() != Path(repo_id).resolve():
            raise ValueError('Sidecar belongs to a different LeRobot dataset; regenerate at the training location')
        # Bounded per-worker cache, never keep all tasks/episodes in RAM.
        self.load = lru_cache(maxsize=2)(self._load)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('load')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.load = lru_cache(maxsize=2)(self._load)

    def _load(self, episode):
        info = self.manifest['episodes'][str(episode)]
        path = self.root / f'episode_{episode:06d}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('EE sidecar changed after audit')
        with np.load(path, allow_pickle=False) as data:
            q, ee = data['joint'], data['ee']
        pose = audit_episode(q[:-1], q[1:], q, ee)
        return q, pose

    def attach(self, item, chunk_size):
        ep, frame = int(item['episode_index']), int(item['frame_index'])
        q, pose = self.load(ep)
        n = len(q)-1
        if not 0 <= frame < n:
            raise ValueError('EE frame outside episode')
        indices = frame + np.arange(chunk_size)
        pad = indices >= n
        targets = np.minimum(indices, n-1) + 1
        # Check both timestamps against actual loader output, including repeated padding.
        if not np.allclose(np.asarray(item['observation.state']), q[frame], atol=1e-5, rtol=0):
            raise ValueError('Runtime state/EE alignment mismatch')
        if np.asarray(item['action']).shape != (chunk_size,14) or not np.allclose(np.asarray(item['action']), q[targets], atol=1e-5, rtol=0):
            raise ValueError('Runtime action/EE alignment mismatch')
        if not np.array_equal(np.asarray(item['action_is_pad']), pad):
            raise ValueError('Runtime padding/EE alignment mismatch')
        return {**item, 'observation.ee_pose': torch.from_numpy(pose[frame].copy()),
                'action.ee_pose': torch.from_numpy(pose[targets].copy()),
                'action.ee_pose_is_pad': item['action_is_pad']}
