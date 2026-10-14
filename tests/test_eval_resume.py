"""Exercise recovery after actual control-loop interruption without a GPU."""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import random
import subprocess
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiment/robotwin'))
import eval_checkpoint as resume


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.identity = resume.task_identity('checkpoint-A', 'stack_blocks_three', 'demo_clean', 'seen')

    def tearDown(self):
        self.temporary.cleanup()

    def checkpoint(self, target=3, identity=None):
        return resume.EpisodeCheckpoint(self.directory, identity or self.identity, target)

    def test_success_failure_and_partial_episode_recovery(self):
        cp = self.checkpoint()
        cp.skip_seed(100000)
        cp.begin(100001, 'first')
        cp.commit(True, 20)
        cp.begin(100002, 'second')
        cp.commit(False, 1200)
        cp.begin(100003, 'saved language')
        (self.directory / 'episode2_fail.mp4').write_bytes(b'orphan before commit')
        cp.close()  # process interrupted before completing episode 2
        cp = self.checkpoint()
        try:
            self.assertEqual((cp.completed, cp.successes, cp.next_seed), (2, 1, 100003))
            self.assertEqual(cp.begin(100003, 'new random language'), 'saved language')
            self.assertFalse((self.directory / 'episode2_fail.mp4').exists())
            cp.commit(True, 10)
            self.assertEqual((cp.completed, cp.successes, cp.next_seed), (3, 2, 100004))
        finally:
            cp.close()

    def test_lock_mismatch_and_target_extension(self):
        cp = self.checkpoint(1)
        with self.assertRaisesRegex(ValueError, 'Another evaluator'):
            self.checkpoint(1)
        cp.begin(100000, 'instruction')
        cp.commit(False, 1200)
        cp.close()
        other = dict(self.identity, run_id='checkpoint-B')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.checkpoint(1, other)
        cp = self.checkpoint(2)
        try:
            self.assertEqual((cp.completed, cp.successes), (1, 0))
        finally:
            cp.close()

    def test_corruption_and_legacy_results_are_not_silently_reset(self):
        (self.directory / '_progress.json').write_text('{broken')
        with self.assertRaises(ValueError):
            self.checkpoint()
        (self.directory / '_progress.json').unlink()
        (self.directory / '_result.txt').write_text('0.5')
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            self.checkpoint()

    def test_completed_plan_and_stats_include_saved_failures(self):
        model = self.directory / 'model'
        model.mkdir()
        (model / 'weights.safetensors').write_bytes(b'model-A')
        config = self.directory / 'config.yaml'
        config.write_text('model: {}')
        setting = self.directory / 'demo_clean.yml'
        setting.write_text('render_freq: 0')
        run = self.directory / 'run'
        args = argparse.Namespace(run_dir=str(run), model=str(model), config=str(config),
            setting='demo_clean', task_config_file=str(setting), instruction='seen', length=50,
            robo_name='robotwin', precision='True/False/False', episodes=1, tasks=['one','two'])
        with contextlib.redirect_stdout(io.StringIO()):
            resume.prepare(args)
        identity = json.loads((run / 'run_manifest.json').read_text())['run_id']
        for task, success in [('one',True),('two',False)]:
            cp = resume.EpisodeCheckpoint(run/'eval_results'/task,
                 resume.task_identity(identity,task,'demo_clean','seen'),1)
            cp.begin(100000,task)
            cp.commit(success,10)
            cp.close()
        with contextlib.redirect_stdout(io.StringIO()):
            resume.prepare(args)
            resume.stats(args)
        plan = json.loads((run / 'resume_plan.json').read_text())
        self.assertEqual(plan['pending'], [])
        self.assertEqual((run/'eval_results/two/_result.txt').read_text().splitlines()[-1],'0.0')
        self.assertIn('success 1/2, overall rate 50.0%', (run/'stats.txt').read_text())
        config.write_text('model: changed')
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            resume.prepare(args)

    def test_control_loop_restart_reuses_seed_and_does_not_double_count(self):
        # Load the real eval_policy function, excluding heavy simulator imports.
        tree = ast.parse((ROOT/'experiment/robotwin/eval_policy_client_lingbotvla.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='eval_policy')
        namespace = dict(np=np, random=random, os=__import__('os'),
            UnStableError=type('UnStableError',(Exception,),{}), report_progress=lambda *a,**k: None,
            generate_episode_descriptions=lambda *a: [dict(seen=['instruction'])])
        exec(compile(ast.Module(body=[function],type_ignores=[]),'real_eval_policy','exec'),namespace)
        calls=[]
        class Environment:
            plan_success=True
            eval_video_path=None
            render_freq=0
            step_lim=1
            def setup_demo(self, now_ep_num, seed, **kwargs):
                calls.append((now_ep_num,seed))
                self.take_action_cnt=0
                self.eval_success=False
            def play_once(self): return {'info':{}}
            def close_env(self, **kwargs): pass
            def check_success(self): return True
            def set_instruction(self, instruction): self.instruction=instruction
            def get_instruction(self): return self.instruction
            def get_obs(self):
                camera={'rgb':np.zeros((2,2,3))}
                return {'observation':dict(head_camera=camera,left_camera=camera,right_camera=camera),
                        'joint_action':{'vector':np.zeros(14)}}
            def take_action(self, action):
                self.take_action_cnt+=1
                self.eval_success=True
        class Model:
            resets=0
            fail=True
            def infer(self, obs):
                if obs.get('reset'):
                    self.resets+=1
                    return {}
                if self.fail and self.resets==2:
                    raise RuntimeError('simulated connection loss')
                return {'action':np.zeros((1,14)), 'server_timing':0}
        env,model=Environment(),Model()
        args=dict(task_name='stack_blocks_three',policy_name='ACT',task_config='demo_clean',
                  ckpt_setting=None,render_freq=0,clear_cache_freq=5)
        cp=self.checkpoint(2)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'connection loss'):
                namespace['eval_policy']('stack_blocks_three',env,args,model,100000,test_num=2,
                    instruction_type='seen',usr_args=dict(_checkpoint=cp,robo_name='robotwin'))
        cp.close()

        cp=self.checkpoint(2)
        self.assertEqual((cp.completed,cp.next_seed),(1,100001))
        model.fail=False
        calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            namespace['eval_policy']('stack_blocks_three',env,args,model,100000,test_num=2,
                instruction_type='seen',usr_args=dict(_checkpoint=cp,robo_name='robotwin'))
        self.assertEqual(calls,[(1,100001),(1,100001)])
        self.assertEqual((cp.completed,cp.successes),(2,2))
        cp.close()

    def test_launcher_skips_completed_tasks_without_loading_a_model(self):
        model = self.directory / 'hf_ckpt'
        model.mkdir()
        (model/'model.safetensors').write_bytes(b'test model')
        config = self.directory/'inference.yaml'
        config.write_text('model: {}')
        sim = self.directory/'RoboTwin'
        (sim/'task_config').mkdir(parents=True)
        (sim/'task_config/demo_clean.yml').write_text('render_freq: 0')
        run = self.directory/'run'
        task = 'stack_blocks_three'
        args = argparse.Namespace(run_dir=str(run),model=str(model),config=str(config),
            setting='demo_clean',task_config_file=str(sim/'task_config/demo_clean.yml'),
            instruction='seen',length=50,robo_name='robotwin',precision='True/False/False',
            episodes=1,tasks=[task])
        with contextlib.redirect_stdout(io.StringIO()):
            resume.prepare(args)
        run_id = json.loads((run/'run_manifest.json').read_text())['run_id']
        cp = resume.EpisodeCheckpoint(run/'eval_results'/task,
            resume.task_identity(run_id,task,'demo_clean','seen'),1)
        cp.begin(100000,'instruction')
        cp.commit(False,1200)
        cp.close()
        # No conda.sh or GPU is available in this fake simulator; an unnecessary
        # server/simulator start would fail rather than pass this test.
        launcher = ROOT/'experiment/robotwin/start_robotwin_infer_and_eval.sh'
        if not launcher.exists():
            self.skipTest('Bootstrap the pinned model checkout for launcher integration')
        environment = dict(os.environ,LINGBOT_INFERENCE_CONFIG=str(config))
        result = subprocess.run(['bash',str(launcher),'--model_path',str(model),
            '--eval_workdir',str(sim),'--task_config','demo_clean','--task_name',task,
            '--num_episodes','1','--run_dir',str(run),'--use_bf16','True',
            '--use_fp32','False','--use_compile','False','--conda_sh','/missing/conda.sh'],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr+result.stdout)
        self.assertIn('inference server is not started',result.stdout)
        self.assertIn('success 0/1',result.stdout)
        rejected = subprocess.run(['bash',str(launcher),'--model_path',str(model),
            '--eval_workdir',str(sim),'--task_config','demo_clean','--task_name',task,
            '--num_episodes','1','--run_dir',str(run),'--use_bf16','False'],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        self.assertNotEqual(rejected.returncode,0)
        self.assertIn('settings differ',rejected.stderr)


if __name__ == '__main__':
    unittest.main()
