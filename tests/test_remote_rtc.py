"""Real websocket transport and remote-only launcher checks without model weights."""
import argparse
import contextlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from deploy.async_rtc import AsyncRTC
from deploy.rtc_protocol import server_metadata
from deploy.websocket_client_policy import WebsocketClientPolicy
from experiment.robotwin.remote_eval import prepare, validate_server
from experiment.robotwin.eval_checkpoint import EpisodeCheckpoint, task_identity

ROOT=Path(__file__).resolve().parents[1]
METADATA=dict(rtc_protocol=2,rtc_modes=['training_time','inference_time'],training_time_rtc=True,rtc_max_delay=10,action_horizon=50,
              normalized_action_dim=55,checkpoint_signature='checkpoint-signature-A',
              checkpoint='/remote/global_step_5000/hf_ckpt',norm_stats_sha256='stats-A')


class ToyPolicy:
    def infer(self, obs):
        if obs.get('reset'):return {'action':None}
        rtc=obs['_rtc']
        start=int(obs['observation.state'][0])
        norm=np.repeat(np.arange(start,start+50,dtype=np.float32)[:,None],55,axis=1)
        if rtc.get('previous_actions') is not None:
            norm[:rtc['delay']]=rtc['previous_actions'][:rtc['delay']]
        return dict(action=norm[:,:14].copy(),_normalized_actions=norm,_rtc_request_id=rtc['request_id'])


def serve(port):
    from deploy.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(ToyPolicy(),host='127.0.0.1',port=port,metadata=METADATA).serve_forever()


class RemoteRTCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sock=socket.socket();sock.bind(('127.0.0.1',0));cls.port=sock.getsockname()[1];sock.close()
        cls.process=multiprocessing.get_context('spawn').Process(target=serve,args=(cls.port,))
        cls.process.start()
        client=WebsocketClientPolicy('127.0.0.1',cls.port,startup_timeout=10)
        client.close()

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate();cls.process.join(5)

    def arguments(self, root):
        task=root/'demo_clean.yml';task.write_text('render_freq: 0')
        return argparse.Namespace(run_dir=str(root/'run'),checkpoint_id='step5000-rtc',
            setting='demo_clean',task_config_file=str(task),instruction='seen',episodes=1,
            tasks=['stack_bowls_three'],rtc_delay=6,rtc_replan_steps=8,
            rtc_mode='training_time',rtc_guidance_beta=5.0)

    def test_real_websocket_client_runs_async_plans_and_reset(self):
        client=WebsocketClientPolicy('127.0.0.1',self.port,startup_timeout=5)
        try:
            self.assertEqual(client.get_server_metadata(),METADATA)
            client.reset('robotwin')
            with AsyncRTC(client,log=lambda *a,**k:None) as rtc:
                for step in range(40):
                    self.assertEqual(rtc.next_action(lambda:{'observation.state':np.full(14,step)})[0],step)
            client.reset('robotwin')
            with AsyncRTC(client,log=lambda *a,**k:None) as fresh:
                self.assertEqual(fresh.next_action(lambda:{'observation.state':np.zeros(14)})[0],0)
        finally:
            client.close()

    def test_protocol_and_trained_delay_are_validated_before_simulation(self):
        for metadata,delay,replan in (({},6,8),(dict(METADATA,training_time_rtc=False),6,8),
            (METADATA,10,8),(METADATA,6,45),(METADATA,6,5),(METADATA,6,0)):
            with self.assertRaises(ValueError):validate_server(metadata,delay,replan)
        # The final valid committed prefix may end exactly at the plan horizon.
        validate_server(METADATA,6,44)
        with self.assertRaisesRegex(ValueError, 'protocol-2'):
            validate_server(dict(METADATA, rtc_protocol=1), 6, 8, 'inference_time')
        baseline_metadata = dict(METADATA, training_time_rtc=False,
                                 rtc_modes=['inference_time'], rtc_max_delay=0)
        validate_server(baseline_metadata, 6, 8, 'inference_time')

    def test_remote_manifest_rejects_changed_server_checkpoint_and_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.arguments(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args,METADATA);prepare(args,METADATA)
            with self.assertRaisesRegex(ValueError,'settings differ'):
                prepare(args,dict(METADATA,checkpoint_signature='checkpoint-B'))
            args.rtc_delay=5
            with self.assertRaisesRegex(ValueError,'settings differ'):prepare(args,METADATA)

    def test_inference_time_manifest_records_its_guidance_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.arguments(Path(directory))
            args.rtc_mode='inference_time'
            args.rtc_guidance_beta=4.5
            metadata=dict(METADATA, training_time_rtc=False,
                          rtc_modes=['inference_time'], rtc_max_delay=0)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args,metadata)
            rtc=json.loads((Path(directory)/'run/run_manifest.json').read_text())['identity']['rtc']
            self.assertEqual(rtc,dict(mode='inference_time_softmask_vjp_v1',delay=6,
                replan_steps=8,guidance_beta=4.5,delay_buffer_size=4,
                scheduler='asap_adaptive_delay'))

    def test_metadata_signs_normalization_and_inference_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'weights.safetensors').write_bytes(b'weights')
            norm=root/'stats.json';norm.write_text('{}')
            config=root/'inference.yaml';config.write_text('model: {}')
            model_config=types.SimpleNamespace(training_time_rtc=True,training_time_rtc_max_delay=10,
                n_action_steps=50,max_action_dim=55,to_dict=lambda:{'rtc':True})
            policy=types.SimpleNamespace(config=model_config,robot_norm_path=str(norm))
            with patch.dict(os.environ,LINGBOT_INFERENCE_CONFIG=str(config)):
                policy.use_compile=False
                self.assertEqual(server_metadata(policy,root)['rtc_modes'],['training_time','inference_time'])
                policy.config.training_time_rtc=False
                self.assertEqual(server_metadata(policy,root)['rtc_modes'],['inference_time'])
                policy.use_compile=True
                self.assertEqual(server_metadata(policy,root)['rtc_modes'],[])
                policy.config.training_time_rtc=True
                policy.use_compile=False
                first=server_metadata(policy,root)
                self.assertEqual(first,server_metadata(policy,root))
                norm.write_text('{"new":true}')
                self.assertNotEqual(first['checkpoint_signature'],server_metadata(policy,root)['checkpoint_signature'])
                norm.write_text('{}');config.write_text('model: {new: true}')
                self.assertNotEqual(first['checkpoint_signature'],server_metadata(policy,root)['checkpoint_signature'])

    def test_remote_launcher_skips_complete_task_without_inference_or_sim_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.arguments(root)
            with contextlib.redirect_stdout(io.StringIO()):prepare(args,METADATA)
            run_id=json.loads((root/'run/run_manifest.json').read_text())['run_id']
            cp=EpisodeCheckpoint(root/'run/eval_results/stack_bowls_three',
                task_identity(run_id,'stack_bowls_three','demo_clean','seen'),1)
            cp.begin(100000,'saved');cp.commit(False,12);cp.close()
            sim=root/'sim';(sim/'envs').mkdir(parents=True);(sim/'task_config').mkdir()
            (sim/'envs/stack_bowls_three.py').touch()
            (sim/'task_config/demo_clean.yml').write_text('render_freq: 0')
            conda=root/'conda.sh';conda.write_text('conda() { :; }\n')
            env=dict(os.environ,CONDA_SH=str(conda),EVAL_WORKDIR=str(sim),
                     PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'])
            result=subprocess.run(['bash',str(ROOT/'eval_rtc_remote.sh'),'127.0.0.1',str(self.port),
                'demo_clean','--checkpoint_id','step5000-rtc','--task_name','stack_bowls_three',
                '--num_episodes','1','--run_dir',str(root/'run')],env=env,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('All selected tasks already complete',result.stdout)
            self.assertIn('success 0/1',result.stdout)
            self.assertFalse((sim/'script').exists())


if __name__=='__main__':unittest.main()
