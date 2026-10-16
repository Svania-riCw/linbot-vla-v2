"""Weighted flow reduction and spline training/sampling, with VLM boundaries isolated."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import yaml
from test_upstream import ROOT, HAS_TORCH, extract
from test_experiment import fake_stats
from ysj_utils.action_experiment import prepare, record_receipt, equivalent_stats_source
from deploy.action_contract import resolve_contract


MODEL = ROOT/'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py'


def actual_config(**kwargs):
    """Run actual model-config classes; only Transformers' base class is replaced."""
    path=ROOT/'lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py'
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)]
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    tree=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
    namespace={'PretrainedConfig':type('ConfigBase',(),{})}
    exec(compile(tree,str(path),'exec'),namespace)
    defaults=dict(max_action_dim=55,max_state_dim=55,action_dim=55,chunk_size=50,mask_action_padding=True,loss_type='L1_fm')
    return namespace['LingbotVLAV2Config'](**dict(defaults,**kwargs))


def tiny_flow(config):
    """Execute actual flow forward/predict_velocity/Euler sampler with tiny projections.

    Prefix/suffix embeddings and the expensive attention backbone are controlled
    substitutes. No full VLM/model loading, CUDA or simulator is exercised here.
    """
    import torch
    from torch import nn
    from types import MethodType
    ns=extract(MODEL,class_name='FlowMatchingV2',method_names={'forward','sample_actions','predict_velocity'})
    ns.update(F=torch.nn.functional,make_att_2d_masks=lambda pad,att: pad[:,:,None]&pad[:,None,:])
    obj=SimpleNamespace(config=config,action_spline=None,_use_compile_predict_velocity=False,
                        block_future_depth_to_action=False,action_in_proj=nn.Linear(55,8),action_out_proj=nn.Linear(8,55))
    if config.spline_control_points:
        from lingbotvla.action_spaces.spline import SplineActionCodec
        obj.action_spline=SplineActionCodec(50,config.spline_control_points,config.spline_degree)
    config.align_params={}; config.num_steps=2
    obj.embed_prefix=lambda images,*args,**kw: (torch.zeros(len(images),2,8),torch.ones(len(images),2,dtype=torch.bool),
                                               torch.zeros(len(images),2,dtype=torch.bool),torch.zeros(3,len(images),2,dtype=torch.long),None,None)
    obj.seen=[]
    def suffix(state,actions,time,**kwargs):
        obj.seen.append(actions.shape[1])
        shape=(len(state),actions.shape[1])
        return None,obj.action_in_proj(actions),torch.ones(shape,dtype=torch.bool),torch.zeros(shape,dtype=torch.bool)
    obj.embed_suffix=suffix
    obj.qwenvl_with_expert=SimpleNamespace(forward=lambda **kw: (kw['inputs_embeds'],None,[]))
    obj._build_full_position_ids=lambda prefix_pos,prefix_pad,suffix_pad: torch.zeros(3,len(prefix_pad),prefix_pad.shape[1]+suffix_pad.shape[1],dtype=torch.long)
    obj._block_suffix_to_future_video_if_enabled_=lambda mask,**kwargs:mask
    obj._moe_losses_and_metrics=lambda logits,loss: (loss.new_zeros(()),loss.new_zeros(()),{})
    obj.sample_time=lambda batch,device: torch.full((batch,),.5,device=device)
    for name in ('forward','sample_actions','predict_velocity'):
        setattr(obj,name,MethodType(ns[name],obj))
    return obj


@unittest.skipUnless(HAS_TORCH,'PyTorch required')
class WeightedLossTests(unittest.TestCase):
    def test_uniform_and_small_rescaled_weights_preserve_masked_loss(self):
        import torch
        from lingbotvla.action_spaces.loss import weighted_flow_loss
        errors=torch.arange(24,dtype=torch.float32).reshape(2,3,4)
        mask=torch.ones_like(errors,dtype=torch.bool); mask[0,1:]=False
        expected=errors[mask].mean()
        for scale in (1.,.0001,10.):
            actual,_=weighted_flow_loss(errors,mask,[scale]*4,[scale]*3)
            torch.testing.assert_close(actual,expected)

    def test_independent_numpy_reference_and_gradient_weights(self):
        import torch
        from action_contracts import weighted_flow_l1
        from lingbotvla.action_spaces.loss import weighted_flow_loss
        errors=torch.arange(1,25,dtype=torch.float32).reshape(2,3,4).requires_grad_()
        mask=torch.ones_like(errors,dtype=torch.bool); mask[0,2]=False
        dims=[1.,2.,0.,1.]; steps=[1.,.5,.1]
        loss,_=weighted_flow_loss(errors,mask,dims,steps)
        expected=weighted_flow_l1(errors.detach().numpy(),np.zeros((2,3,4)),mask.numpy(),dims,steps)
        self.assertAlmostEqual(float(loss.detach()),expected,places=5)
        loss.backward()
        self.assertTrue(torch.all(errors.grad[~mask]==0))
        self.assertTrue(torch.all(errors.grad[:,:,2]==0))
        self.assertAlmostEqual(float(errors.grad[0,0,1]/errors.grad[0,0,0]),2.)

    def test_bad_weights_and_no_effective_supervision_rejected(self):
        import torch
        from lingbotvla.action_spaces.loss import weighted_flow_loss
        x=torch.ones(1,2,3); mask=torch.ones_like(x,dtype=torch.bool)
        for dims,steps in (([-1,1,1],None),([np.nan,1,1],None),([0,0,0],None),([1],None),(None,[1,np.inf])):
            with self.assertRaises(ValueError): weighted_flow_loss(x,mask,dims,steps)
        with self.assertRaises(ValueError): weighted_flow_loss(x,torch.zeros_like(mask))

    def test_actual_flow_and_policy_uniform_weight_regression_and_optimizer(self):
        import torch
        cfg=actual_config()
        core=tiny_flow(cfg)
        forward=extract(MODEL,class_name='LingbotVlaV2Policy',method_names={'forward'})['forward']
        policy=SimpleNamespace(config=cfg,model=core)
        actions=torch.randn(2,50,55); mask=torch.zeros_like(actions,dtype=torch.bool)
        mask[:,:,:12]=True; mask[:,:,28:30]=True
        pad=torch.zeros(2,50,dtype=torch.bool); pad[0,10:]=True
        def run():
            return forward(policy,torch.zeros(2,1),None,torch.zeros(2,55),None,None,actions,
                           joint_mask=mask,action_is_pad=pad,noise=torch.zeros_like(actions),time=torch.full((2,),.5))
        baseline=run()[1].detach()
        cfg.flow_dimension_weights=[1.]*55; cfg.flow_horizon_weights=[1.]*50
        uniform=run()[1]
        torch.testing.assert_close(uniform,baseline)
        cfg.flow_dimension_weights[28:30]=[2.,2.]
        result=run()[0]; before=float(result.detach()); result.backward()
        self.assertGreater(float(core.action_out_proj.weight.grad.abs().sum()),0)
        optimizer=torch.optim.SGD(list(core.action_in_proj.parameters())+list(core.action_out_proj.parameters()),lr=.01)
        optimizer.step()
        self.assertLess(float(run()[0].detach()),before)
        cfg.loss_type='fm'; cfg.flow_dimension_weights=None; cfg.flow_horizon_weights=None
        mse_plain=run()[1].detach()
        cfg.flow_dimension_weights=[1.]*55
        torch.testing.assert_close(run()[1].detach(),mse_plain)


@unittest.skipUnless(HAS_TORCH,'PyTorch required')
class SplineTests(unittest.TestCase):
    def test_basis_matches_scipy_and_partition_of_unity(self):
        from scipy.interpolate import BSpline
        from lingbotvla.action_spaces.spline import bspline_basis
        for h,k,degree in ((50,16,3),(20,6,2),(10,4,1)):
            basis,knots=bspline_basis(h,k,degree)
            expected=BSpline.design_matrix(np.linspace(0,1,h),knots,degree).toarray()
            np.testing.assert_allclose(basis,expected,atol=1e-12)
            np.testing.assert_allclose(basis.sum(1),1,atol=1e-12)
            self.assertTrue(np.all(basis>=0))

    def test_control_point_roundtrip_gradient_and_bfloat16(self):
        import torch
        from lingbotvla.action_spaces.spline import SplineActionCodec
        codec=SplineActionCodec(50,16)
        control=torch.randn(2,16,3,requires_grad=True)
        curve=codec.decode(control)
        fitted,mask=codec.encode(curve,torch.zeros(2,50,dtype=torch.bool),torch.ones_like(curve,dtype=torch.bool))
        torch.testing.assert_close(fitted,control,atol=2e-6,rtol=1e-5)
        self.assertTrue(mask.all())
        codec.decode(fitted).square().mean().backward()
        self.assertTrue(torch.isfinite(control.grad).all())
        with torch.autocast('cpu',dtype=torch.bfloat16):
            fitted,_=codec.encode(curve.to(torch.bfloat16),torch.zeros(2,50,dtype=torch.bool),torch.ones_like(curve,dtype=torch.bool))
        self.assertEqual(codec.decode(fitted).dtype,torch.bfloat16)
        self.assertTrue(all(t.dtype!=torch.bfloat16 for t in codec.tensors(torch.device('cpu'))))
        with torch.device('meta'):
            meta_codec=SplineActionCodec(50,16)
        decoded=meta_codec.decode(torch.ones(1,16,3,device='cpu'))
        torch.testing.assert_close(decoded,torch.ones(1,50,3))

    def test_padding_poison_does_not_change_coefficients_or_gradients(self):
        import torch
        from lingbotvla.action_spaces.spline import SplineActionCodec
        codec=SplineActionCodec(50,16)
        actions=torch.randn(2,50,4,requires_grad=True)
        pad=torch.arange(50)[None,:]>=torch.tensor([1,17])[:,None]
        mask=torch.ones_like(actions,dtype=torch.bool)
        a,coeff_mask=codec.encode(actions,pad,mask)
        poisoned=actions.detach().clone(); poisoned[pad]=float('nan')
        b,_=codec.encode(poisoned,pad,mask)
        torch.testing.assert_close(a,b)
        self.assertTrue(torch.isfinite(a).all())
        self.assertLess(float(np.abs(codec._arrays[1]).max()),16.)
        self.assertEqual(int(coeff_mask[0,:,0].sum()),1)
        a.square().sum().backward()
        self.assertTrue(torch.all(actions.grad[pad]==0))
        bad=pad.clone(); bad[1,2]=True
        with self.assertRaises(ValueError): codec.encode(actions,bad,mask)
        with self.assertRaises(ValueError): codec.encode(actions,torch.ones_like(pad),mask)

    def test_actual_coefficient_flow_training_and_euler_decoder(self):
        import torch
        cfg=actual_config(spline_control_points=16)
        core=tiny_flow(cfg)
        policy=SimpleNamespace(config=cfg,model=core)
        forward=extract(MODEL,class_name='LingbotVlaV2Policy',method_names={'forward'})['forward']
        target=core.action_spline.decode(torch.randn(2,16,55))
        mask=torch.zeros_like(target,dtype=torch.bool); mask[:,:,:12]=True; mask[:,:,28:30]=True
        target[~mask]=0
        pad=torch.zeros(2,50,dtype=torch.bool); pad[0,25:]=True
        result=forward(policy,torch.zeros(2,1),None,torch.zeros(2,55),None,None,target,
                       joint_mask=mask,action_is_pad=pad,time=torch.full((2,),.5),noise=torch.zeros(2,16,55))
        self.assertEqual(core.seen[-1],16)
        self.assertIn('spline/gripper_target_mae',result[6])
        self.assertTrue(torch.isfinite(result[0]))
        result[0].backward()
        self.assertGreater(float(core.action_out_proj.weight.grad.abs().sum()),0)
        with torch.no_grad():
            core.action_out_proj.weight.zero_(); core.action_out_proj.bias.zero_()
        coefficients=torch.randn(2,16,55)
        output=core.sample_actions(torch.zeros(2,1),None,None,None,torch.zeros(2,55),noise=coefficients.clone())
        self.assertEqual(output.shape,(2,50,55))
        torch.testing.assert_close(output,core.action_spline.decode(coefficients))
        self.assertTrue(all(n==16 for n in core.seen))

    def test_spline_to_actual_normalizer_joint_anchor_and_controller(self):
        import torch
        from test_ee_implementation import feature_namespace
        from lingbotvla.action_spaces.spline import SplineActionCodec
        from deploy.ee_actions import execution_actions
        data=SimpleNamespace(joints=[str({'arm.position':14}),str({'end.position':14}),str({'effector.position':2})],
                             cameras=['camera_top','camera_wrist_left','camera_wrist_right'],
                             norm_type=[str({k:'bounds_99_woclip'}) for k in ('arm.position','end.position','effector.position')])
        processor=SimpleNamespace(tokenizer=None,image_processor=SimpleNamespace(merge_size=2))
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); fake_stats(directory/'stats.json')
            robot=yaml.safe_load((ROOT/'configs/robot_configs/robotwin.yaml').read_text())
            robot['actions'][0]['action.arm.position']['subtract_state']=True
            robot['norm_stats']=str(directory/'stats.json')
            (directory/'robot.yaml').write_text(yaml.safe_dump(robot))
            cfg=actual_config(spline_control_points=16)
            transform=feature_namespace()['FeatureTransform'](str(directory/'robot.yaml'),data,cfg,processor,disabled_image_features=True)
            state=torch.linspace(-.2,.3,14)
            time=torch.linspace(0,1,50)[:,None]
            target=state+time*.2+time.square()*.03
            batch=transform.apply({'observation.state':state.clone(),'action':target.clone(),'action_is_pad':torch.zeros(50,dtype=torch.bool),'task':'test'})
            codec=SplineActionCodec(50,16)
            coeff,_=codec.encode(batch['actions'][None],batch['action_is_pad'][None],batch['joint_mask'][None])
            prediction=codec.decode(coeff)
            obs=transform.apply({'observation.state':state.clone(),'task':'test'},policy_eval=True)
            server=SimpleNamespace(action_key=['action'],use_bf16=False,vla=SimpleNamespace(feature_transform=transform))
            unapply=extract(ROOT/'deploy/lingbot_vla_v2_policy.py',class_name='LingbotVLAv2Server',method_names={'_unapply_batched_actions'})['_unapply_batched_actions']
            result=unapply(server,[obs],prediction)
            actions,control=execution_actions({'action':result['action'][0],'_control_space':'qpos'})
            self.assertEqual(control,'qpos')
            np.testing.assert_allclose(actions,target.numpy(),atol=2e-6)

    def test_physical_unit_diagnostic_exposes_abrupt_gripper_error(self):
        from ysj_utils.check_spline_fit import summarize_chunks
        actions=np.zeros((2,50,14),dtype='float32')
        actions[:,20:,[6,13]]=1
        stats={key:dict(q01=[-2.]*dim,q99=[2.]*dim) for key,dim in [('action.arm.position',12),('action.effector.position',2)]}
        result=summarize_chunks(np.zeros((2,14),dtype='float32'),actions,np.zeros((2,50),dtype=bool),stats,8)
        self.assertEqual(result['abrupt_gripper_transitions'],2)
        self.assertGreater(result['gripper_units']['max_abs'],.1)
        self.assertLess(result['joint_radians']['max_abs'],1e-5)

    def test_real_parquet_sampling_stays_inside_episode_and_marks_padding(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ysj_utils.check_spline_fit import sampled_chunks
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'data/chunk-000').mkdir(parents=True)
            table={'episode_index':[0]*3+[1]*3,'frame_index':[0,1,2]*2,
                   'observation.state':[[float(i)]*14 for i in range(6)],
                   'action':[[float(i+1)]*14 for i in range(6)]}
            pq.write_table(pa.table(table),root/'data/chunk-000/part.parquet')
            samples=list(sampled_chunks(root,6,50))
            self.assertEqual(len(samples),6)
            for state,actions,pad in samples:
                i=int(state[0]); remaining=3-i%3
                self.assertEqual(int((~pad).sum()),remaining)
                self.assertEqual(actions[-1,0],3. if i<3 else 6.)


class SupervisionConfigurationTests(unittest.TestCase):
    def test_config_rejects_mixed_methods_bad_weights_and_bad_spline_shapes(self):
        for kwargs in (dict(flow_dimension_weights=[-1.]*55),dict(flow_horizon_weights=[1.]),
                       dict(spline_control_points=3),dict(spline_control_points=-1),
                       dict(spline_control_points=16,training_time_rtc=True),
                       dict(spline_control_points=16,flow_horizon_weights=[1.]*50),
                       dict(flow_horizon_weights=[1.]*50,joint_ee_aux_weight=.1)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): actual_config(**kwargs)
        self.assertEqual(actual_config(spline_control_points=16).n_action_steps,16)
        self.assertEqual(actual_config().n_action_steps,50)

    def test_prepared_groups_share_stats_budget_and_bind_codec_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp)/'bundle'
            pair=prepare(ROOT/'xsy_utils/configs/robotwin_clean_baseline.yaml',ROOT/'configs/robot_configs/robotwin.yaml',
                         ROOT/'xsy_utils/configs/robotwin_clean.txt',bundle,family='supervision')
            self.assertEqual(pair['groups'],['JDelta','WUniform','WGrip','WTime','Spline'])
            source=bundle/'JDelta'; fake_stats(source/'norm_stats.json'); record_receipt(source,pair)
            for group in pair['groups']:
                directory=bundle/group
                cfg=yaml.safe_load((directory/'train.yaml').read_text())
                self.assertTrue(cfg['train']['mask_action_padding']); self.assertFalse(cfg['train']['use_compile'])
                self.assertNotIn('ee_sidecar_root',cfg['data'])
                self.assertEqual(cfg['train']['global_batch_size'],128); self.assertEqual(cfg['train']['max_steps'],10000)
                self.assertFalse(cfg['train']['enable_fp32']); self.assertTrue(cfg['train']['enable_mixed_precision'])
                if group!='JDelta':
                    self.assertEqual(equivalent_stats_source(bundle,pair,directory),source/'norm_stats.json')
                    (directory/'norm_stats.json').write_bytes((source/'norm_stats.json').read_bytes())
                    record_receipt(directory,pair)
                run=directory/'train'; run.mkdir()
                path=run/'lingbotvla_cli.yaml'; path.write_bytes((directory/'train.yaml').read_bytes())
                ckpt=run/'checkpoints/global_step_10000/hf_ckpt'
                self.assertEqual(resolve_contract(ckpt)['mode'],'joint_delta')
                if group=='Spline':
                    cfg['train']['spline_control_points']=8
                    path.write_text(yaml.safe_dump(cfg))
                    with self.assertRaisesRegex(ValueError,'supervision/codec'): resolve_contract(ckpt)

    def test_reordered_canonical_channels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            cfg=yaml.safe_load((ROOT/'xsy_utils/configs/robotwin_clean_baseline.yaml').read_text())
            cfg['data']['joints'].reverse()
            path=directory/'base.yaml'; path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError,'ordered'):
                prepare(path,ROOT/'configs/robot_configs/robotwin.yaml',ROOT/'xsy_utils/configs/robotwin_clean.txt',directory/'out',family='supervision')


if __name__=='__main__':
    unittest.main()
