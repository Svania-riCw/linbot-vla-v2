import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest

import numpy as np

from deploy.async_rtc import AsyncRTC
from deploy.msgpack_numpy import packb, unpackb
from deploy.rtc_protocol import validate_rtc_request
from experiment.robotwin import eval_checkpoint


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.thread_ids = []
        self.release = None
        self.started = threading.Event()
        self.stale = self.corrupt = False
        self.closed = False

    def infer(self, obs):
        self.thread_ids.append(threading.get_ident())
        self.calls.append(obs)
        rtc = obs['_rtc']
        start = int(obs['observation.state'][0])
        if rtc['delay'] and self.release is not None:
            self.started.set()
            if not self.release.wait(3):
                raise RuntimeError('mock was not released')
        norm = np.repeat(np.arange(start, start+50, dtype=np.float32)[:, None], 55, axis=1)
        if rtc.get('previous_actions') is not None:
            norm[:rtc['delay']] = rtc['previous_actions'][:rtc['delay']]
        if self.corrupt and rtc['delay']:
            norm[0] += 10
        return dict(action=norm[:, :14].copy(), _normalized_actions=norm,
                    _rtc_request_id='stale' if self.stale else rtc['request_id'])

    def close(self):
        self.closed = True
        if self.release:
            self.release.set()


class AsyncRTCTest(unittest.TestCase):
    def test_multiple_plans_align_without_replay_and_snapshot_copies_buffers(self):
        model = FakePolicy()
        state = np.zeros(14, dtype=np.float32)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        main = threading.get_ident()
        def observe():
            self.assertEqual(threading.get_ident(), main)
            return {'observation.state': state, 'image': image}
        with AsyncRTC(model, log=lambda *a, **k: None) as rtc:
            actions = []
            for step in range(60):
                state[0] = step
                actions.append(rtc.next_action(observe)[0])
            np.testing.assert_array_equal(actions, np.arange(60))
            self.assertEqual(rtc.step, 60)
        self.assertTrue(all(t != main for t in model.thread_ids))
        self.assertEqual([int(c['observation.state'][0]) for c in model.calls], [0,8,16,24,32,40,48,56])
        image[:] = 255
        self.assertEqual(model.calls[-1]['image'].max(), 0)
        for call in model.calls[1:]:
            start = int(call['observation.state'][0])
            np.testing.assert_array_equal(call['_rtc']['previous_actions'][:6,0], np.arange(start,start+6))

    def test_old_actions_execute_while_inference_is_blocked_and_reset_drains(self):
        model = FakePolicy()
        model.release = threading.Event()
        step = 0
        def observe(): return {'observation.state': np.full(14, step, dtype=np.float32)}
        rtc = AsyncRTC(model, log=lambda *a, **k: None)
        try:
            for step in range(14):
                self.assertEqual(rtc.next_action(observe)[0], step)
            self.assertTrue(model.started.wait(1))
            self.assertFalse(rtc.pending[0].done())  # Six old actions executed while request was blocked.
            model.release.set()
            step = 14
            self.assertEqual(rtc.next_action(observe)[0], 14)
        finally:
            model.release.set()
            rtc.close()
        previous_id = model.calls[-1]['_rtc']['request_id']
        step = 0
        with AsyncRTC(model, log=lambda *a, **k: None) as fresh:
            self.assertEqual(fresh.next_action(observe)[0], 0)
        self.assertNotEqual(previous_id.split(':')[0], model.calls[-1]['_rtc']['request_id'].split(':')[0])
        self.assertEqual(model.calls[-1]['_rtc']['delay'], 0)

    def test_zero_delay_skips_no_actions(self):
        model = FakePolicy()
        with AsyncRTC(model, delay=0, log=lambda *a, **k: None) as rtc:
            for step in range(25):
                self.assertEqual(rtc.next_action(lambda: {'observation.state':np.full(14,step)})[0],step)

    def test_delay_actions_count_toward_request_interval_across_multiple_plans(self):
        model = FakePolicy()
        with AsyncRTC(model, replan_steps=8, delay=6, log=lambda *a, **k: None) as rtc:
            actions = [rtc.next_action(lambda step=step: {
                'observation.state': np.full(14, step, dtype=np.float32)
            })[0] for step in range(31)]

        np.testing.assert_array_equal(actions, np.arange(31))
        self.assertEqual([int(call['observation.state'][0]) for call in model.calls],
                         [0, 8, 16, 24])
        self.assertEqual([call['_rtc']['delay'] for call in model.calls], [0, 6, 6, 6])
        for call in model.calls[1:]:
            start = int(call['observation.state'][0])
            np.testing.assert_array_equal(
                call['_rtc']['previous_actions'][:6, 0], np.arange(start, start + 6)
            )

    def test_paper_mode_uses_full_soft_overlap_payload_with_baseline_checkpoint(self):
        model = FakePolicy()
        model.release = threading.Event()
        rtc = AsyncRTC(model, replan_steps=8, delay=6,
                       mode='paper', guidance_beta=5.0,
                       log=lambda *a, **k: None)
        actions = []
        try:
            for step in range(14):
                actions.append(rtc.next_action(lambda step=step: {
                    'observation.state': np.full(14, step, dtype=np.float32)
                })[0])
            self.assertTrue(model.started.wait(1))
            self.assertFalse(rtc.pending[0].done())
            request = model.calls[1]['_rtc']
            self.assertEqual((request['mode'], request['delay'], request['execution_horizon']),
                             ('paper', 6, 8))
            np.testing.assert_array_equal(request['previous_actions'][:42, 0], np.arange(8, 50))
            np.testing.assert_array_equal(request['previous_actions'][42:, 0], np.zeros(8))
            model.release.set()
            for step in range(14, 24):
                actions.append(rtc.next_action(lambda step=step: {
                    'observation.state': np.full(14, step, dtype=np.float32)
                })[0])
        finally:
            model.release.set()
            rtc.close()

        np.testing.assert_array_equal(actions, np.arange(24))

    def test_delay_cannot_exceed_request_interval(self):
        with self.assertRaisesRegex(ValueError, 'cannot exceed'):
            AsyncRTC(FakePolicy(), replan_steps=5, delay=6,
                     log=lambda *a, **k: None)

    def test_request_is_due_immediately_when_delay_equals_interval(self):
        model = FakePolicy()
        with AsyncRTC(model, replan_steps=2, delay=2,
                      log=lambda *a, **k: None) as rtc:
            actions = [rtc.next_action(lambda step=step: {
                'observation.state': np.full(14, step, dtype=np.float32)
            })[0] for step in range(8)]

        np.testing.assert_array_equal(actions, np.arange(8))
        self.assertEqual([int(call['observation.state'][0]) for call in model.calls],
                         [0, 2, 4, 6])

    def test_stale_response_and_corrupt_prefix_are_errors(self):
        for mode in ('stale','corrupt'):
            model = FakePolicy()
            setattr(model,mode,True)
            rtc = AsyncRTC(model, log=lambda *a, **k: None)
            try:
                with self.assertRaisesRegex(RuntimeError,'Stale|preserve'):
                    for step in range(15):
                        rtc.next_action(lambda: {'observation.state':np.full(14,step)})
            finally:
                rtc.close()

    def test_timeout_closes_transport_and_joins_worker(self):
        model = FakePolicy()
        model.release = threading.Event()
        rtc = AsyncRTC(model, timeout=0.05, log=lambda *a, **k: None)
        for step in range(14):
            rtc.next_action(lambda: {'observation.state':np.full(14,step)})
        with self.assertRaises(TimeoutError):
            rtc.next_action(lambda: {'observation.state':np.full(14,14)})
        with self.assertRaises(TimeoutError):
            rtc.close()
        self.assertTrue(model.closed)
        self.assertTrue(rtc.closed)
        self.assertTrue(rtc.pending[0].done())

    def test_msgpack_roundtrip_and_protocol_validation(self):
        config = types.SimpleNamespace(training_time_rtc=True,training_time_rtc_max_delay=10,
                                       n_action_steps=50,max_action_dim=55)
        request = dict(request_id='episode:2',delay=6,previous_actions=np.zeros((50,55),np.float32))
        previous, delay, request_id = validate_rtc_request(unpackb(packb(request)),config,1)
        self.assertEqual((previous.shape,delay,request_id),((1,50,55),6,'episode:2'))
        for bad in (dict(delay=10),dict(delay=-1),dict(delay=True),dict(delay=6.0),
                    dict(previous_actions=np.zeros((50,14))),dict(previous_actions=None),
                    dict(previous_actions=np.full((50,55),np.nan))):
            with self.assertRaises(ValueError): validate_rtc_request(dict(request,**bad),config,1)
        config.training_time_rtc=False
        with self.assertRaisesRegex(ValueError,'RTC-trained'): validate_rtc_request(request,config,1)

        paper_request = dict(request, mode='paper', execution_horizon=8, guidance_beta=5.0)
        previous, delay, request_id = validate_rtc_request(paper_request, config, 1)
        self.assertEqual((previous.shape, delay, request_id), ((1, 50, 55), 6, 'episode:2'))

    def test_resume_rejects_changed_rtc_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=root/'model';model.mkdir();(model/'weights.safetensors').write_bytes(b'weights')
            config=root/'config';config.write_text('model: {}')
            task=root/'demo_clean.yml';task.write_text('render_freq: 0')
            args=argparse.Namespace(run_dir=str(root/'run'),model=str(model),config=str(config),
                setting='demo_clean',task_config_file=str(task),instruction='seen',length=50,
                robo_name='robotwin',precision='True/False/False',episodes=1,tasks=['stack_bowls_three'],
                rtc_async='True',rtc_replan_steps=8,rtc_delay=6)
            with contextlib.redirect_stdout(io.StringIO()):
                eval_checkpoint.prepare(args)
                eval_checkpoint.prepare(args)
            args.rtc_delay=5
            with self.assertRaisesRegex(ValueError,'settings differ'):eval_checkpoint.prepare(args)
            args.rtc_delay=6;args.rtc_async='False'
            with self.assertRaisesRegex(ValueError,'settings differ'):eval_checkpoint.prepare(args)


if __name__ == '__main__':
    unittest.main()
