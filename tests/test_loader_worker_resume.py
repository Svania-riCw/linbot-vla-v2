"""Run on the training environment with torchdata; does not allocate CUDA."""
import copy
import importlib.util
from pathlib import Path
import unittest

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

spec = importlib.util.spec_from_file_location('loader_resume', Path(__file__).parents[1] / 'lingbotvla/utils/loader_resume.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
migrate = module.migrate_loader_workers


class Indices(Dataset):
    def __len__(self):
        return 4096

    def __getitem__(self, index):
        return index


def loader(workers, prefetch):
    dataset = Indices()
    sampler = StatefulDistributedSampler(dataset, num_replicas=8, rank=3, seed=42)
    return StatefulDataLoader(dataset, batch_size=16, sampler=sampler, num_workers=workers,
                              prefetch_factor=prefetch, drop_last=True)


class WorkerResumeTests(unittest.TestCase):
    def test_next_samples_and_second_resume_match(self):
        first = loader(1, 1)
        it = iter(first)
        for _ in range(11):
            next(it)
        original = first.state_dict()
        old_copy = copy.deepcopy(original)
        expected = [next(it).tolist() for _ in range(12)]
        migrated = migrate(original, 8)
        self.assertEqual(original, old_copy)
        second = loader(8, 4)
        second.load_state_dict(migrated)
        it2 = iter(second)
        actual = [next(it2).tolist() for _ in range(5)]
        saved = second.state_dict()
        third = loader(8, 4)
        third.load_state_dict(saved)
        it3 = iter(third)
        actual += [next(it3).tolist() for _ in range(7)]
        self.assertEqual(actual, expected)
        for iterator in [it, it2, it3]:
            iterator._shutdown_workers()

    def test_reject_worker_local_state_or_uncommitted_progress(self):
        first = loader(1, 1)
        it = iter(first)
        next(it)
        state = first.state_dict()
        bad = copy.deepcopy(state)
        bad['_snapshot']['_worker_snapshots']['worker_0']['dataset_state'] = {'position': 7}
        with self.assertRaises(ValueError):
            migrate(bad, 8)
        bad = copy.deepcopy(state)
        bad['_steps_since_snapshot'] = 1
        with self.assertRaises(ValueError):
            migrate(bad, 8)
        self.assertIs(migrate(state, 1), state)
        it._shutdown_workers()


if __name__ == '__main__':
    unittest.main()
