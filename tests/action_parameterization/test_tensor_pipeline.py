"""Original tensor transforms including field mapping and 55-dim padding.

Only image/language processing and FeatureInfo's Pydantic validation are stubbed;
the tested action path uses complete original FeatureTransform/Normalizer bodies.
"""
import ast
from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import yaml
from test_upstream import extract, ROOT, DATA, HAS_TORCH
from test_experiment import fake_stats


@unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable: full action tensor pipeline NOT tested')
class TensorPipelineTests(unittest.TestCase):
    def test_raw_joint_to_normalized_padded_action_and_server_decode(self):
        import torch
        import torch.nn.functional as F
        transform_ns = extract(DATA/'transform.py', top_names={'dict_apply', 'Normalizer', 'prepare_state', 'prepare_action'})
        transform_ns.update(F=F)
        info_ns = extract(DATA/'utils.py', class_name='FeatureInfo', method_names={'update_info'})
        info_ns['ast'] = ast
        info_cls = type('FeatureInfoMetadata', (), {'update_info': info_ns['update_info']})
        feature_ns = extract(DATA/'utils.py', top_names={'FeatureTransform', 'compute_image_token_count'})
        feature_ns.update(os=os, yaml=yaml, json=json, ast=ast, F=F, OrderedDict=OrderedDict,
                          FeatureInfo=info_cls, Normalizer=transform_ns['Normalizer'],
                          prepare_state=transform_ns['prepare_state'], prepare_action=transform_ns['prepare_action'],
                          _is_quaternion_relative_type=extract(DATA/'ee_pose_transform.py', top_names={'_is_quaternion_relative_type'})['_is_quaternion_relative_type'],
                          prepare_language=lambda *a: (torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)))
        data = SimpleNamespace(joints=[str({'arm.position':14}), str({'end.position':14}), str({'effector.position':2})],
                               cameras=['camera_top','camera_wrist_left','camera_wrist_right'],
                               norm_type=[str({key:'bounds_99_woclip'}) for key in ['arm.position','end.position','effector.position']])
        config = SimpleNamespace(max_state_dim=55, max_action_dim=55)
        processor = SimpleNamespace(tokenizer=None, image_processor=SimpleNamespace(merge_size=2))
        rng = np.random.default_rng(31)
        state = torch.from_numpy(rng.normal(size=14).astype('float32'))
        actions = torch.from_numpy(rng.normal(size=(50,14)).astype('float32'))
        state[[6,13]] = torch.tensor([.1,.9])
        actions[:,[6,13]] = torch.linspace(0,1,50)[:,None].repeat(1,2)
        for delta in (False, True):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                stats_path = root/'stats.json'
                fake_stats(stats_path)
                robot = yaml.safe_load((ROOT/'configs/robot_configs/robotwin.yaml').read_text(encoding='utf-8'))
                robot['norm_stats'] = str(stats_path)
                robot['actions'][0]['action.arm.position']['subtract_state'] = delta
                cfg = root/'robotwin.yaml'
                cfg.write_text(yaml.safe_dump(robot),encoding='utf-8')
                transform = feature_ns['FeatureTransform'](str(cfg), data, config, processor,
                                                           disabled_image_features=True, chunk_size=50)
                raw = {'observation.state':state.clone(), 'action':actions.clone(),
                       'action_is_pad':torch.zeros(50,dtype=torch.bool), 'task':'test'}
                batch = transform.apply(raw)
                self.assertEqual(tuple(batch['actions'].shape),(50,55))
                self.assertEqual(batch['action_joint_mask'].sum().item(),14)
                self.assertTrue(torch.all(batch['actions'][:,~batch['action_joint_mask']]==0).item())
                indices = [0,1,2,3,4,5,7,8,9,10,11,12]
                q = actions[:,indices] - state[indices] if delta else actions[:,indices]
                torch.testing.assert_close(batch['actions'][:,:12],(q+2)/(4+1e-6)*2-1)
                # Inference applies only current observation; the predicted chunk
                # below is an oracle model output copied from the training target.
                observation = transform.apply({'observation.state':state.clone(),'task':'test'},policy_eval=True)
                observation['actions'] = batch['actions'].clone()
                decoded = transform.unapply(observation)
                # JSON stats are NumPy float64, so unnormalize may promote dtype.
                # Validate physical values without imposing a nonexistent dtype
                # contract on the intermediate transform.
                torch.testing.assert_close(decoded['action'],actions.to(decoded['action'].dtype),rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(decoded['observation.state'],state.to(decoded['observation.state'].dtype),rtol=1e-5,atol=1e-6)
                # Exercise the actual server conversion: it exports float32 NumPy.
                server_ns = extract(ROOT/'deploy/lingbot_vla_v2_policy.py', class_name='LingbotVLAv2Server',
                                    method_names={'_unapply_batched_actions'})
                server = SimpleNamespace(action_key=['action'], use_bf16=False,
                                         vla=SimpleNamespace(feature_transform=transform))
                output = server_ns['_unapply_batched_actions'](server, [observation], batch['actions'][None])
                self.assertEqual(output['action'].dtype,np.float32)
                np.testing.assert_allclose(output['action'][0],actions.numpy(),rtol=1e-5,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
