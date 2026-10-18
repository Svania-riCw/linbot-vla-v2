"""No-GPU checks for first-winner dispatch and completed-stage preservation."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('priority_worker_test', ROOT / 'xsy_utils/resolution_exp/priority_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
spec = importlib.util.spec_from_file_location('queue_engine_test', ROOT / 'xsy_utils/resolution_exp/queue_engine.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


class PriorityQueueTest(unittest.TestCase):
    def test_cold_starting_trainer_is_visible_before_gpu_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pid, argv in [('1', ['private-init']),
                              ('200', ['python', 'tasks/vla/train_lingbotvla.py', 'job.yaml']),
                              ('201', ['python', 'prepare_config.py'])]:
                path = root / pid
                path.mkdir()
                (path / 'cmdline').write_bytes(b'\0'.join(v.encode() for v in argv) + b'\0')
            (root / '202').mkdir()  # A process that exits during enumeration.
            workloads = engine.active_gpu_workloads(root)
            self.assertEqual([item['pid'] for item in workloads], [200])
            self.assertEqual(workloads[0]['entry'], 'train_lingbotvla.py')
            self.assertNotIn('argv', workloads[0])

    def test_late_workload_prevents_spawn_after_config_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / 'wandb_settings.json'
            secret = root / 'dummy_test_credential.json'
            secret.write_text(json.dumps({'api_key': 'test-only'}))
            settings.write_text(json.dumps({'project': 'test', 'entity': 'test',
                'base_url': 'https://api.wandb.ai', 'secret_file': str(secret), 'sdk_path': str(root)}))
            preflight = {'config': 'test.yaml', 'reference_seed': 42, 'data_manifest': 'clean.txt',
                'baseline': 'reference', 'baseline_config_sha256': 'abc', 'clean_norm_sha256': 'def'}
            with patch.object(engine, 'STATE_DIR', root), patch.object(engine, 'OUTPUT_ROOT', root / 'outputs'), \
                 patch.object(engine, 'prepare_config', return_value=preflight), \
                 patch.object(engine, 'wait_idle', side_effect=[None, RuntimeError('late GPU workload')]) as wait, \
                 patch.object(engine.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'late GPU workload'):
                    engine.run_phase({}, 'test', 4, False, float('inf'))
                self.assertEqual(wait.call_count, 2)
                spawn.assert_not_called()
                self.assertFalse((root / 'test_train_b1_384_m4_train_to_5000.log').exists())

    def idle_claim(self, root):
        owner = worker.try_claim(root, 'node2', 100)
        worker.atomic_json(root / 'engine_state/status.json', {
            'phase': 'waiting_for_idle_gpus', 'pid': 100, 'owner_node': 'node2',
            'child_pid': None, 'completed_runs': [{'phase': 'smoke_resume', 'steps': 4}]})
        return owner

    def test_explicit_idle_continuation_preserves_claim_and_completed_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.idle_claim(root)
            state_before = (root / 'engine_state/status.json').read_bytes()
            owner = worker.resume_idle_claim(root, 'node2', 200, pid_alive=lambda pid: False)
            self.assertEqual(owner['pid'], 200)
            self.assertEqual(owner['previous_pid'], 100)
            self.assertEqual(owner['claimed_utc'], original['claimed_utc'])
            self.assertIsNone(worker.try_claim(root, 'node0', 300))
            self.assertEqual((root / 'engine_state/status.json').read_bytes(), state_before)
            self.assertEqual(len(list((root / 'claim').glob('previous_owner_*.json'))), 1)

    def test_idle_continuation_rejects_live_or_foreign_controller(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.idle_claim(root)
            for node, alive in [('node2', True), ('node1', False)]:
                with self.assertRaises(RuntimeError):
                    worker.resume_idle_claim(root, node, 200, pid_alive=lambda pid: alive)
            with patch.object(worker.socket, 'gethostname', return_value='another-host'):
                with self.assertRaises(RuntimeError):
                    worker.resume_idle_claim(root, 'node2', 200, pid_alive=lambda pid: False)
            self.assertEqual(worker.read_owner(root)['pid'], 100)

    def test_idle_continuation_rejects_child_process_or_running_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.idle_claim(root)
            state_path = root / 'engine_state/status.json'
            state = json.loads(state_path.read_text())
            for changes in [{'child_pid': 123}, {'child_pid': None, 'phase': 'train_to_5000'}]:
                state.update(changes)
                worker.atomic_json(state_path, state)
                with self.assertRaises(RuntimeError):
                    worker.resume_idle_claim(root, 'node2', 200, pid_alive=lambda pid: False)
            self.assertEqual(worker.read_owner(root)['pid'], 100)

    def test_simultaneous_candidates_have_exactly_one_durable_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            barrier = threading.Barrier(3)
            def claim(node):
                barrier.wait()
                return worker.try_claim(root, node, 100)
            with ThreadPoolExecutor(3) as pool:
                results = list(pool.map(claim, ['node0', 'node1', 'node2']))
            self.assertEqual(sum(r is not None for r in results), 1)
            winner = worker.read_owner(root)
            self.assertIn(winner['node'], ['node0', 'node1', 'node2'])
            self.assertIsNone(worker.try_claim(root, 'another', 200))

    def test_incomplete_claim_publication_never_looks_unclaimed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'claim').mkdir()
            self.assertTrue(worker.read_owner(root)['publishing'])
            self.assertIsNone(worker.try_claim(root, 'node1', 100))

    def test_successful_two_step_smoke_is_not_repeated(self):
        state = {'completed_runs': [{'phase': 'smoke', 'micro_batch': 4, 'steps': 2}]}
        actions = worker.next_actions(state, 'test', 4)
        self.assertEqual(actions[0], ('smoke_resume', 4))
        self.assertNotIn(('smoke', 2), actions)
        self.assertEqual(actions[-1], ('evaluation', 10000))

    def test_finished_5k_training_and_eval_resume_at_10k(self):
        state = {'completed_runs': [{'phase': phase, 'micro_batch': 4, 'steps': steps}
                 for phase, steps in [('smoke', 2), ('smoke_resume', 4), ('train_to_5000', 5000)]],
                 'evaluations': [{'step': 5000}]}
        self.assertEqual(worker.next_actions(state, 'test', 4),
                         [('resume_to_10000', 10000), ('evaluation', 10000)])


if __name__ == '__main__':
    unittest.main()
