import copy
from pathlib import Path
import unittest

from ysj_utils.train_baseline_bs256 import TARGETS, EVAL_TARGETS, check_source, stage_config, verify_updates, requested_stages


class BaselineBatchTests(unittest.TestCase):
    def setUp(self):
        self.source={'model':{'model_path':'original_pretrained'},
                     'data':{'num_workers':8,'prefetch_factor':4,'joints':['original']},
                     'train':dict(global_batch_size=128,micro_batch_size=16,gradient_accumulation_steps=1,
                                  enable_fp32=True,enable_mixed_precision=True,loss_type='L1_fm',
                                  lr=1e-4,lr_min=5e-5,lr_decay_style='cosine',ckpt_manager='dcp',
                                  use_compile=True,seed=42,align_params={'original':True})}
        self.robot={'states':[{},{}],'actions':[{'action.arm.position':{'subtract_state':False}},
                                               {'action.effector.position':{'subtract_state':False}}]}

    def test_accumulation_budget_and_fixed_horizon(self):
        original=copy.deepcopy(self.source)
        for target in TARGETS:
            cfg=stage_config(self.source,Path('/runs/bs256'),target)
            t=cfg['train']
            self.assertEqual(t['global_batch_size'],8*t['micro_batch_size']*t['gradient_accumulation_steps'])
            self.assertEqual(t['max_steps'],target)
            self.assertEqual(t['lr_schedule_steps'],20000)
            self.assertEqual(cfg['model'],self.source['model'])
            for key in ['enable_fp32','enable_mixed_precision','align_params','loss_type','use_compile','seed']:
                self.assertEqual(t[key],self.source['train'][key])
        self.assertEqual(self.source,original)
        self.assertEqual([256*t for t in EVAL_TARGETS],[1280000,2560000,5120000])

    def test_strict_resume_only_when_checkpoint_supplied(self):
        t=stage_config(self.source,Path('/runs'),5000,Path('/runs/probe/global_step_50'))['train']
        self.assertTrue(t['require_resume']);self.assertTrue(t['enable_resume'])
        self.assertFalse(t['allow_partial_checkpoint'])
        self.assertTrue(t['load_checkpoint_path'].endswith('global_step_50'))
        self.assertFalse(stage_config(self.source,Path('/runs'),50)['train']['enable_resume'])

    def test_refuse_confounding_action_changes(self):
        check_source(self.source,self.robot)
        for key,value in [('mask_action_padding',True),('joint_ee_aux_weight',0.1),
                          ('spline_control_points',16),('flow_gibbs_epsilon',0.5),
                          ('enable_fp32',False),('enable_fsdp_offload',True)]:
            cfg=copy.deepcopy(self.source);cfg['train'][key]=value
            with self.assertRaises(ValueError,msg=key):check_source(cfg,self.robot)

    def test_update_audit_refuses_fresh_or_missing_resume(self):
        line='Step 51/999, Epoch 1, Loss 0.1, GradNorm 0.2, LR 1.00e-04, Expert_LR 2e-4, StepTime 8.5s'
        with self.assertRaises(ValueError):verify_updates(line,51,51)
        text='Load distributed checkpoint from /source successfully!\n'+line
        self.assertEqual(verify_updates(text,51,51)['last_step'],51)
        with self.assertRaises(ValueError):verify_updates(text,51,52)
        with self.assertRaises(ValueError):verify_updates(text+'\nLoss nan',51,51)
        with self.assertRaises(ValueError):verify_updates(text+'\nStarting training from scratch',51,51)

    def test_final_only_skips_intermediate_eval_and_restart(self):
        stages,evaluations=requested_stages(True)
        self.assertEqual(stages,(50,5000,20000))
        self.assertEqual(evaluations,(20000,))
        self.assertEqual(requested_stages(False),(TARGETS,EVAL_TARGETS))
        cfg=stage_config(self.source,Path('/runs'),20000,Path('/runs/step_05000/train/checkpoints/global_step_5000'))
        self.assertEqual(cfg['train']['max_steps'],20000)
        self.assertEqual(cfg['train']['lr_schedule_steps'],20000)
        self.assertTrue(cfg['train']['require_resume'])


if __name__=='__main__':unittest.main()
