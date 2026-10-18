"""No-GPU checks for first-winner dispatch and completed-stage preservation."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('priority_worker_test', ROOT / 'xsy_utils/resolution_exp/priority_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class PriorityQueueTest(unittest.TestCase):
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
