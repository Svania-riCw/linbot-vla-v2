"""Queue ordering, failure detection and result recording, without GPU workloads."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ysj_utils.run_joint_pair import Stage,build_plan,run_stage,wait_idle,finish_comparison,NONFINITE
from ysj_utils.evaluate import scheme


class JointQueueTests(unittest.TestCase):
    def test_actual_training_counter_does_not_count_exhausted_loader(self):
        from ysj_utils.run_joint_pair import ROOT
        tree=ast.parse((ROOT/'tasks/vla/train_lingbotvla.py').read_text(encoding='utf-8'))
        loop=next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                  and isinstance(node.target,ast.Name) and node.target.id=='epoch_step')
        prefix=[]
        for node in loop.body:
            if isinstance(node,ast.If): break
            prefix.append(node)
        loop.body=prefix+ast.parse('updates.append(global_step)').body
        module=ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[]))
        scope=dict(start_step=0,args=SimpleNamespace(train=SimpleNamespace(train_steps=5),data=SimpleNamespace(drop_last=False)),
                   global_step=0,data_iterator=iter([['a'],['b']]),epoch=0,
                   logger=SimpleNamespace(info=lambda message:None),updates=[],List=list,Dict=dict,Any=object)
        exec(compile(module,'actual_training_loop','exec'),scope)
        self.assertEqual(scope['updates'],[1,2])
        self.assertEqual(scope['global_step'],2)

    def test_actual_training_log_nonfinite_format_is_detected(self):
        for text in ('Step 1/5000, Epoch 1, Loss nan, VLA_Loss 0.1, GradNorm 1.0',
                     'Step 2/5000, Loss 1.0, VLA_Loss nan, GradNorm 1.0',
                     'Step 3/5000, Loss 1.0, GradNorm inf, LR 1e-4',
                     '{"training/vla_loss": NaN}'):
            self.assertIsNotNone(NONFINITE.search(text))
        self.assertIsNone(NONFINITE.search('Loss 1.23, GradNorm 2.4, nan_guard=True'))

    def test_two_smoke_gates_before_five_k_training_and_evaluation(self):
        root=Path(tempfile.gettempdir())/'ysj_queue_plan'
        args=SimpleNamespace(bundle=root/'bundle',run_dir=root/'queue',train_list=root/'train.txt',raw_root=root/'raw',
                             raw_setting='aloha-agilex_clean_50',ee_sidecar_root=root/'labels',base_config=root/'base.yaml',
                             robot_config=root/'robot.yaml',gpus=['0','1'],micro_batch=16,steps=5000,eval_seed=0,port_base=19330)
        plan=build_plan(args)
        names=[stage.name for stage in plan]
        self.assertLess(names.index('eval_smoke_JointEE'),names.index('full_JDelta'))
        self.assertLess(names.index('eval_smoke_JDelta'),names.index('full_JDelta'))
        self.assertNotIn('full_JointEE0',names)
        self.assertEqual(sum(name.startswith('full_') for name in names),2)
        for stage in plan:
            if stage.name.startswith('eval_full'):
                self.assertTrue(any('global_step_5000' in value for value in stage.command))
                self.assertEqual(stage.gpus,['0','1'])
        prepare=next(stage for stage in plan if stage.name=='prepare')
        self.assertEqual(prepare.command[prepare.command.index('--steps')+1],'5000')

    def test_stage_records_success_and_subprocess_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state={'stages':{}}
            run_stage(Stage('ok',[sys.executable,'-c','print("loss_vla: 0.25")'],[]),root,state,os.environ.copy(),.02,False)
            self.assertEqual(state['stages']['ok']['status'],'COMPLETE')
            with self.assertRaisesRegex(RuntimeError,'exited 3'):
                run_stage(Stage('bad',[sys.executable,'-c','import sys; sys.exit(3)'],[]),root,state,os.environ.copy(),.02,False)
            self.assertEqual(json.loads((root/'status.json').read_text())['stages']['bad']['status'],'FAILED')

    def test_nonfinite_loss_stops_stage_even_when_process_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state={'stages':{}}
            with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
                run_stage(Stage('nan',[sys.executable,'-c','print("loss_vla: nan")'],[]),root,state,os.environ.copy(),.02,False)
            self.assertEqual(state['stages']['nan']['status'],'FAILED')

    def test_busy_gpu_is_rejected_without_stopping_existing_jobs(self):
        snapshot={'gpus':[dict(index='0',uuid='GPU-test',used_mib=0,utilization=0)],'compute_processes':''}
        with patch('ysj_utils.run_joint_pair.gpu_snapshot',return_value=snapshot):
            self.assertEqual(wait_idle(['0'],0),snapshot)
        snapshot['compute_processes']='123, GPU-test, 100'
        with patch('ysj_utils.run_joint_pair.gpu_snapshot',return_value=snapshot),self.assertRaisesRegex(RuntimeError,'busy'):
            wait_idle(['0'],0)

    def test_final_five_k_pair_produces_json_and_readable_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol=scheme('full')
            for group,mode,success in [('JDelta','joint_delta',10),('JointEE','joint_ee_delta',12)]:
                run=root/group; run.mkdir()
                report=dict(status='COMPLETE',mode='full',**protocol,seed=0,inference_precision='fp32',use_length=50,
                            code_commit='same',initial_model='same',train_list_sha256='same',state_features_sha256='same',
                            label_manifests={'labels':'same'},contract={'mode':mode},
                            action_loss=dict(mask_action_padding=True,joint_ee_aux_weight=-1 if group=='JDelta' else .1),
                            training_summary=dict(global_batch_size=128,max_steps=5000,enable_fp32=False,enable_mixed_precision=True),
                            checkpoint=f'/test/{group}/checkpoints/global_step_5000/hf_ckpt')
                (run/'ysj_evaluation.json').write_text(json.dumps(report))
                (run/'_result.txt').write_text('\n'.join(f'{task}\t{setting}\t{success}/25\t{success/25}'
                                                       for task in protocol['tasks'] for setting in ('clean','random')))
                (root/f'eval_full_{group}.json').write_text(json.dumps(dict(report,run_dir=str(run))))
            result=finish_comparison(root,5000)
            self.assertAlmostEqual(result['summary']['clean']['difference_pp'],8)
            self.assertTrue((root/'comparison.json').exists())
            self.assertIn('5000 optimizer updates',(root/'RESULTS.md').read_text())


if __name__=='__main__': unittest.main()
