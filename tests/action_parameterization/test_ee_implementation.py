"""Synthetic label files -> actual tensor transforms -> controller dispatch.

The image/tokenizer and full VLM are outside this CPU test's boundary. Geometry,
loader attachment, flow-loss reduction and policy forward reduction are real.
"""
import ast
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from test_upstream import ROOT, DATA, extract, HAS_TORCH
from deploy.ee_actions import raw_ee_to_pose, execution_actions, observation_pose
from deploy.action_contract import action_mode, resolve_contract
from ysj_utils.action_experiment import ee_robot_config, prepare, verify_pair, record_receipt, equivalent_stats_source


def episode(n=5):
    rng = np.random.default_rng(81)
    q = rng.normal(size=(n+1,14)).astype('float32')/5
    q[:,[6,13]] = rng.uniform(size=(n+1,2))
    ee = np.zeros((n+1,16), dtype='float32')
    for arm in (0,8):
        ee[:,arm:arm+3] = rng.normal(size=(n+1,3))
        ee[:,arm+3:arm+7] = Rotation.from_rotvec(rng.normal(size=(n+1,3))).as_quat()[:,[3,0,1,2]]
    ee[:,[7,15]] = q[:,[6,13]]
    return q, ee


def fake_ee_stats(path, robot):
    stats = {}
    for cat in ('states','actions'):
        for group in robot[cat]:
            key = next(iter(group))
            dim = 14 if key.endswith('end.position') else 2 if key.endswith('effector.position') else 12
            stats[key] = dict(mean=[0.]*dim,std=[1.]*dim,q01=[-4.]*dim,q99=[4.]*dim)
    path.write_text(json.dumps({'norm_stats':stats}),encoding='utf-8')


def feature_namespace():
    import torch
    import torch.nn.functional as F
    ns = extract(DATA/'transform.py',top_names={'dict_apply','Normalizer','prepare_state','prepare_action'})
    ns['F'] = F
    info = extract(DATA/'utils.py',class_name='FeatureInfo',method_names={'update_info'})
    info['ast'] = ast
    features = extract(DATA/'utils.py',top_names={'FeatureTransform','compute_image_token_count'})
    features.update(os=os,yaml=yaml,json=json,ast=ast,F=F,OrderedDict=OrderedDict,
                    FeatureInfo=type('Metadata',(),{'update_info':info['update_info']}),
                    Normalizer=ns['Normalizer'], prepare_state=ns['prepare_state'],prepare_action=ns['prepare_action'],
                    prepare_language=lambda *a:(torch.zeros(1,dtype=torch.long),torch.ones(1,dtype=torch.bool)))
    geometry = extract(DATA/'ee_pose_transform.py',top_names={n.name for n in ast.parse((DATA/'ee_pose_transform.py').read_text()).body if isinstance(n,ast.FunctionDef)})
    features.update({k:v for k,v in geometry.items() if not k.startswith('__')})
    return features


class ControllerTests(unittest.TestCase):
    def test_actual_server_emits_controller_metadata_without_polluting_action_cache(self):
        namespace=extract(ROOT/'deploy/lingbot_vla_v2_policy.py',class_name='LingbotVLAv2Server',method_names={'infer'})
        for control in ('qpos','ee'):
            server=SimpleNamespace(control_space=control,chunk_ret=True,use_length=3,global_step=0,
                                   last_action_chunk=None,last_normalized_action_chunk=None,
                                   _infer_batch=lambda _: {'action':np.zeros((1,3,14))})
            for _ in range(2):
                result=namespace['infer'](server,{})
                self.assertEqual(result['_control_space'],control)
                self.assertNotIn('_control_space',server.last_action_chunk)

    def test_world_pose_wxyz_xyzw_roundtrip_and_absolute_grippers(self):
        q, ee = episode()
        pose = raw_ee_to_pose(ee)
        actual, control = execution_actions({'action':q[:,[6,13]],'action.ee_pose':pose,'_control_space':'ee'})
        self.assertEqual(control,'ee')
        np.testing.assert_allclose(actual,ee,atol=1e-6)
        obs = {'endpose':dict(left_endpose=ee[0,:7],left_gripper=ee[0,7],right_endpose=ee[0,8:15],right_gripper=ee[0,15])}
        np.testing.assert_allclose(observation_pose(obs),pose[0],atol=1e-6)

    def test_joint_ee_always_executes_joint_even_with_bad_aux_prediction(self):
        q, _ = episode()
        actual, control = execution_actions({'action':q,'action.ee_pose':np.full((len(q),14),np.nan),'_control_space':'qpos'})
        self.assertEqual(control,'qpos')
        np.testing.assert_array_equal(actual,q)

    def test_bad_or_ambiguous_controller_data_rejected(self):
        for result in ({'action':np.zeros(14),'action.ee_pose':np.zeros(14)},
                       {'action':np.zeros(2),'action.ee_pose':np.zeros(14),'_control_space':'ee'},
                       {'action':np.zeros(2),'_control_space':'qpos'}):
            with self.assertRaises(ValueError):
                execution_actions(result)

    def test_actual_sim_adapter_dispatches_ee_and_stops_on_success(self):
        q, ee = episode()
        namespace = extract(ROOT/'external/RoboTwin/policy/lingbot-vla-v2/deploy_policy.py',top_names={'eval'})
        namespace.update(execution_actions=execution_actions, encode_obs=lambda env,obs:obs)
        calls = []
        env = SimpleNamespace(eval_success=False)
        def take(action,action_type):
            calls.append((action,action_type))
            env.eval_success = True
        env.take_action = take
        model = SimpleNamespace(infer=lambda _:dict(action=q[:,[6,13]],**{'action.ee_pose':raw_ee_to_pose(ee),'_control_space':'ee'}))
        namespace['eval'](env,model,{})
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1],'ee')
        np.testing.assert_allclose(calls[0][0],ee[0],atol=1e-6)


@unittest.skipUnless(HAS_TORCH,'PyTorch required')
class EETensorTests(unittest.TestCase):
    def test_both_new_modes_through_actual_mapping_norm_padding_and_decode(self):
        import torch
        features = feature_namespace()
        q, ee = episode(7)
        pose = raw_ee_to_pose(ee)
        base = yaml.safe_load((ROOT/'configs/robot_configs/robotwin.yaml').read_text())
        data = SimpleNamespace(joints=[str({'arm.position':14}),str({'end.position':14}),str({'effector.position':2})],
                               cameras=['camera_top','camera_wrist_left','camera_wrist_right'],
                               norm_type=[str({k:'bounds_99_woclip'}) for k in ('arm.position','end.position','effector.position')])
        processor = SimpleNamespace(tokenizer=None,image_processor=SimpleNamespace(merge_size=2))
        for group in ('EEDelta','JointEE','JointEE0'):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as temp:
                directory=Path(temp)
                robot=ee_robot_config(base,group)
                robot['norm_stats']=str(directory/'stats.json')
                fake_ee_stats(directory/'stats.json',robot)
                (directory/'robot.yaml').write_text(yaml.safe_dump(robot),encoding='utf-8')
                transform=features['FeatureTransform'](str(directory/'robot.yaml'),data,SimpleNamespace(max_action_dim=55,max_state_dim=55),
                                                      processor,disabled_image_features=True,chunk_size=7)
                raw={'observation.state':torch.from_numpy(q[0].copy()),'action':torch.from_numpy(q[1:].copy()),
                     'observation.ee_pose':torch.from_numpy(pose[0].copy()),'action.ee_pose':torch.from_numpy(pose[1:].copy()),
                     'action_is_pad':torch.zeros(7,dtype=torch.bool),'action.ee_pose_is_pad':torch.zeros(7,dtype=torch.bool),'task':'test'}
                batch=transform.apply(copy.deepcopy(raw))
                self.assertEqual(batch['actions'].shape,(7,55))
                self.assertEqual(int(batch['action_joint_mask'].sum()),16 if group=='EEDelta' else 28)
                self.assertEqual(int(batch['state_joint_mask'].sum()),28)
                # Compare translation to an independent SciPy rotation implementation.
                for start in (0,7):
                    expected=Rotation.from_quat(pose[0,start+3:start+7]).inv().apply(pose[1:,start:start+3]-pose[0,start:start+3])
                    actual=(batch['actions'][:,14+start:17+start].numpy()+1)*(8+1e-6)/2-4
                    np.testing.assert_allclose(actual,expected,atol=2e-6)
                obs=transform.apply({k:v for k,v in copy.deepcopy(raw).items() if not k.startswith('action')},policy_eval=True)
                obs['actions']=batch['actions'].clone()
                server_ns=extract(ROOT/'deploy/lingbot_vla_v2_policy.py',class_name='LingbotVLAv2Server',method_names={'_unapply_batched_actions'})
                server=SimpleNamespace(action_key=transform.org_features['actions'],use_bf16=False,vla=SimpleNamespace(feature_transform=transform))
                decoded=server_ns['_unapply_batched_actions'](server,[obs],batch['actions'][None])
                result={k:v[0] for k,v in decoded.items()}
                result['_control_space']='ee' if group=='EEDelta' else 'qpos'
                actions,control=execution_actions(result)
                if control=='ee':
                    np.testing.assert_allclose(actions[:,[0,1,2,7,8,9,10,15]],ee[1:,[0,1,2,7,8,9,10,15]],atol=2e-6)
                    for offset in (3,11):
                        expected=Rotation.from_quat(ee[1:,offset:offset+4][:,[1,2,3,0]])
                        actual=Rotation.from_quat(actions[:,offset:offset+4][:,[1,2,3,0]])
                        np.testing.assert_allclose(expected.as_matrix(),actual.as_matrix(),atol=2e-6)
                else:
                    np.testing.assert_allclose(actions,q[1:],atol=2e-6)

    def test_actual_policy_forward_aux_gradient_zero_ablation_padding_and_optimizer(self):
        import torch
        import torch.nn.functional as F
        ns=extract(ROOT/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',class_name='LingbotVlaV2Policy',method_names={'forward'})
        mask=torch.zeros((2,3,55),dtype=torch.bool)
        mask[:,:,:12]=True; mask[:,:,14:30]=True
        pad=torch.tensor([[False,False,True],[False,True,True]])
        for weight in (-1.,0.,.1):
            with self.subTest(weight=weight):
                pred=torch.nn.Parameter(torch.ones((2,3,55)))
                def model_forward(*args,**kwargs):
                    loss=F.l1_loss(pred,torch.zeros_like(pred),reduction='none')
                    z=pred.new_zeros(())
                    return loss,z,z,z,None,z,z,{},None,None,None,None
                obj=SimpleNamespace(config=SimpleNamespace(loss_type='L1_fm',action_dim=55,mask_action_padding=True,
                                                           joint_ee_aux_weight=weight,training_time_rtc=False),model=SimpleNamespace(forward=model_forward))
                def forward():
                    return ns['forward'](obj,None,None,torch.zeros((2,55)),None,None,torch.zeros_like(pred),joint_mask=mask,action_is_pad=pad)
                result=forward()
                self.assertAlmostEqual(float(result[1].detach()),1.1 if weight==.1 else 1.,places=6)
                result[0].backward()
                self.assertTrue(torch.all(pred.grad[pad]==0))
                self.assertTrue(torch.all(pred.grad[...,30:]==0))
                if weight==0:
                    self.assertTrue(torch.all(pred.grad[...,14:28]==0))
                elif weight>0:
                    self.assertGreater(float(pred.grad[...,14:28].abs().sum()),0)
                    self.assertIn('action/ee_loss',result[6])
                before=float(result[0].detach())
                optimizer=torch.optim.SGD([pred],lr=.1)
                optimizer.step()
                self.assertLess(float(forward()[0].detach()),before)


@unittest.skipUnless(HAS_TORCH and importlib.util.find_spec('h5py') and importlib.util.find_spec('pyarrow'),'torch/h5py/pyarrow required')
class EELabelIntegrationTests(unittest.TestCase):
    def setUp(self):
        import h5py
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.dataset=self.root/'robotwin_clean_open_laptop'
        (self.dataset/'data/chunk-000').mkdir(parents=True)
        self.raw=self.root/'raw'; self.raw.mkdir()
        self.q,self.ee=episode()
        self.table={'episode_index':[0]*5,'frame_index':list(range(5)),
                    'observation.state':self.q[:-1].tolist(),'action':self.q[1:].tolist()}
        pq.write_table(pa.table(self.table),self.dataset/'data/chunk-000/episode_000000.parquet')
        with h5py.File(self.raw/'episode0.hdf5','w') as f:
            f['joint_action/vector']=self.q
            f['endpose/left_endpose']=self.ee[:,:7]; f['endpose/right_endpose']=self.ee[:,8:15]
            f['endpose/left_gripper']=self.ee[:,7]; f['endpose/right_gripper']=self.ee[:,15]

    def labels(self):
        from ysj_utils.prepare_ee_labels import prepare_dataset
        return prepare_dataset(self.dataset,self.raw,self.root/'labels')

    def test_real_parquet_hdf5_audit_runtime_padding_and_spawn_serialization(self):
        import torch
        from lingbotvla.action_spaces.ee_labels import EESidecar
        self.labels()
        sidecar=pickle.loads(pickle.dumps(EESidecar(self.root/'labels',self.dataset)))
        raw={'episode_index':torch.tensor(0),'frame_index':torch.tensor(3),
             'observation.state':torch.from_numpy(self.q[3].copy()),
             'action':torch.from_numpy(self.q[[4,5,5,5]].copy()),'action_is_pad':torch.tensor([False,False,True,True])}
        attached=sidecar.attach(raw,4)
        expected=raw_ee_to_pose(self.ee)
        np.testing.assert_allclose(attached['action.ee_pose'],expected[[4,5,5,5]],atol=1e-6)
        raw['action'][1,0]+=1
        with self.assertRaisesRegex(ValueError,'Runtime action'):
            sidecar.attach(raw,4)

    def test_wrong_action_timestamp_refuses_to_publish_manifest(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.table['action']=self.q[:-1].tolist()
        pq.write_table(pa.table(self.table),self.dataset/'data/chunk-000/episode_000000.parquet')
        with self.assertRaisesRegex(ValueError,'Action/raw alignment'):
            self.labels()
        self.assertFalse((self.root/'labels'/self.dataset.name/'manifest.json').exists())

    def test_completed_label_audit_reuse_rejects_changed_sources(self):
        from ysj_utils.prepare_ee_labels import verify_existing
        self.labels()
        result=verify_existing(self.dataset,self.root/'labels')
        self.assertEqual(len(result['episodes']),1)
        path=self.dataset/'data/chunk-000/episode_000000.parquet'
        with path.open('ab') as file: file.write(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            verify_existing(self.dataset,self.root/'labels')

    def test_actual_loader_attaches_before_transform_and_queries_only_real_columns(self):
        import torch
        from types import MethodType
        from lingbotvla.action_spaces.ee_labels import EESidecar
        self.labels()
        methods=extract(DATA/'base_dataset.py',class_name='VLADataset',method_names={'getitem','get_delta_timestamps','check_lerobot_item'})
        raw={'episode_index':torch.tensor(0),'frame_index':torch.tensor(3),
             'observation.state':torch.from_numpy(self.q[3].copy()),
             'action':torch.from_numpy(self.q[[4,5,5,5]].copy()),'action_is_pad':torch.tensor([False,False,True,True])}
        transform=SimpleNamespace(org_features={'actions':['action.ee_pose','action'],
                                               'states':['observation.state','observation.ee_pose']},
                                  actions_convert_from_state={},apply=lambda item:item)
        obj=SimpleNamespace(dataset=[raw],dataset_meta=SimpleNamespace(fps=20),chunk_size=4,
                            ee_sidecar=EESidecar(self.root/'labels',self.dataset),use_future_image=False,
                            feature_transform=transform,transform=None)
        obj.check_lerobot_item=MethodType(methods['check_lerobot_item'],obj)
        result=methods['getitem'](obj,0)
        self.assertIn('action.ee_pose_is_pad',result)
        np.testing.assert_allclose(result['observation.ee_pose'],raw_ee_to_pose(self.ee)[3],atol=1e-6)
        offsets=methods['get_delta_timestamps'](obj)
        self.assertEqual(set(offsets),{'action'})
        self.assertEqual(offsets['action'],[0,.05,.1,.15])

    def test_five_matched_configs_stats_receipts_and_label_mutation_rejection(self):
        self.labels()
        listing=self.root/'train.txt'; listing.write_text(f'robotwin {self.dataset}\n')
        bundle=self.root/'experiment'
        manifest=prepare(ROOT/'xsy_utils/configs/robotwin_clean_baseline.yaml',ROOT/'configs/robot_configs/robotwin.yaml',listing,
                         bundle,ee_sidecar_root=self.root/'labels')
        states=[]
        for group in manifest['groups']:
            _,directory=verify_pair(bundle,group)
            robot=yaml.safe_load((directory/'robot_configs/robotwin.yaml').read_text())
            cfg=yaml.safe_load((directory/'train.yaml').read_text())
            states.append(robot['states'])
            self.assertTrue(cfg['train']['mask_action_padding'])
            self.assertEqual(cfg['train']['global_batch_size'],128)
            self.assertEqual(cfg['train']['max_steps'],10000)
            fake_ee_stats(directory/'norm_stats.json',robot)
            record_receipt(directory,manifest)
            run=directory/'train'; run.mkdir()
            (run/'lingbotvla_cli.yaml').write_bytes((directory/'train.yaml').read_bytes())
            contract=resolve_contract(run/'checkpoints/global_step_10000/hf_ckpt')
            expected={'B0':'joint_absolute','JDelta':'joint_delta','EEDelta':'ee_delta','JointEE0':'joint_ee_delta','JointEE':'joint_ee_delta'}
            self.assertEqual(contract['mode'],expected[group])
        self.assertEqual(equivalent_stats_source(bundle,manifest,bundle/'JointEE'),bundle/'JointEE0/norm_stats.json')
        self.assertTrue(all(state==states[0] for state in states))
        path=next(iter(manifest['label_manifests']))
        with open(path,'a') as f: f.write(' ')
        with self.assertRaisesRegex(ValueError,'label manifest changed'):
            verify_pair(bundle,'JointEE')

    def test_joint_ee_norm_command_reuses_verified_zero_weight_statistics(self):
        self.labels()
        listing=self.root/'train.txt'; listing.write_text(f'robotwin {self.dataset}\n')
        bundle=self.root/'experiment'
        pair=prepare(ROOT/'xsy_utils/configs/robotwin_clean_baseline.yaml',ROOT/'configs/robot_configs/robotwin.yaml',listing,
                     bundle,ee_sidecar_root=self.root/'labels')
        source=bundle/'JointEE0'
        robot=yaml.safe_load((source/'robot_configs/robotwin.yaml').read_text())
        fake_ee_stats(source/'norm_stats.json',robot)
        record_receipt(source,pair)
        result=subprocess.run([sys.executable,str(ROOT/'ysj_utils/action_experiment.py'),'norm','--bundle',str(bundle),
                               '--group','JointEE'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((bundle/'JointEE/norm_stats.json').read_bytes(),(source/'norm_stats.json').read_bytes())
        receipt=json.loads((bundle/'JointEE/norm_stats.json.contract.json').read_text())
        self.assertEqual(receipt['stats_source_group'],'JointEE0')


if __name__=='__main__':
    unittest.main()
