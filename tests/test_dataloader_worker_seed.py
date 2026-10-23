"""Loader workers of different data-parallel ranks must not share random streams (the observation
augmentation draws its per-sample seed from the worker's torch RNG), while the sampler keeps one shared
seed so the ranks still partition the data."""
import types
import unittest

import torch
from torch.utils.data import Dataset


class _Draws(Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        return {"idx": torch.tensor(i), "aug_seed": torch.randint(0, 2**31, ())}


def _first_batch(rank: int):
    import lingbotvla.data.data_loader as DL
    from lingbotvla.utils import helper

    real = DL.get_parallel_state
    DL.get_parallel_state = lambda: types.SimpleNamespace(dp_size=2, dp_rank=rank, sp_size=1, sp_enabled=False)
    try:
        helper.set_seed(42)  # what every rank does: identical global RNG state
        dl = DL.build_dataloader(
            dataset=_Draws(), micro_batch_size=4, global_batch_size=8, dataloader_batch_size=4, max_seq_len=1024,
            train_steps=8, rmpad=False, seed=42, num_workers=2,
            collate_fn=lambda xs: {k: torch.stack([x[k] for x in xs]) for k in xs[0]},
        )
        batches = []
        for b in dl:
            batches.append(b[0] if isinstance(b, list) else b)
            if len(batches) == 4:
                break
        return torch.cat([b["idx"] for b in batches]), torch.cat([b["aug_seed"] for b in batches])
    finally:
        DL.get_parallel_state = real


class WorkerSeedTest(unittest.TestCase):
    def test_ranks_draw_different_streams_on_disjoint_samples(self):
        idx0, seed0 = _first_batch(0)
        idx1, seed1 = _first_batch(1)
        self.assertEqual(set(idx0.tolist()) & set(idx1.tolist()), set())  # sampler still partitions
        self.assertFalse(torch.equal(seed0, seed1))  # was identical before the per-rank generator
        self.assertEqual(len(set(seed0.tolist()) & set(seed1.tolist())), 0)

    def test_same_rank_is_reproducible(self):
        self.assertTrue(torch.equal(_first_batch(1)[1], _first_batch(1)[1]))


if __name__ == "__main__":
    unittest.main()
