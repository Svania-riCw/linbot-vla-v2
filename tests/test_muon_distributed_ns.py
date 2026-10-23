"""DistributedMuon on FSDP2-style row-sharded params vs Muon computed on the full tensors.

Covers the distributed Newton-Schulz path (all-to-all, each rank orthogonalises 1/world of a batch)
and the all-gather path, with row counts that do not divide the world size (the d4rt head's 588-row
``to_pred`` on 8 ranks is 7 x 74 + 70 under DTensor's torch.chunk layout), ranks with empty shards,
uneven numbers of same-shape matrices, and a parameter without a gradient. CPU + gloo, 8 and 3 ranks.
"""
import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# (rows, cols, how many params of that shape)
SHAPES = [(588, 16, 3), (55, 12, 1), (64, 32, 5), (7, 5, 2), (5, 9, 1), (16, 48, 4)]
LR, WD, MOMENTUM, STEPS = 0.02, 0.1, 0.95, 2


def _full_tensors(seed: int):
    g = torch.Generator().manual_seed(seed)
    params, grads = [], []
    for rows, cols, n in SHAPES:
        for _ in range(n):
            params.append(torch.randn(rows, cols, generator=g))
            grads.append([torch.randn(rows, cols, generator=g) for _ in range(STEPS)])
    return params, grads


def _reference(params, grads, no_grad_index):
    """Plain Muon (nesterov, match the module's NS) on full tensors."""
    from lingbotvla.optim.muon import _adjust_lr, batched_newton_schulz

    params = [p.clone() for p in params]
    bufs = [torch.zeros_like(p) for p in params]
    for step in range(STEPS):
        for i, p in enumerate(params):
            if i == no_grad_index:
                continue
            g = grads[i][step]
            bufs[i].lerp_(g, 1 - MOMENTUM)
            update = g.lerp(bufs[i], MOMENTUM)
            ortho = batched_newton_schulz(update.unsqueeze(0))[0]
            p.mul_(1 - LR * WD).add_(ortho, alpha=-_adjust_lr(LR, None, p.shape))
    return params


def _worker(rank, world_size, port, distribute_ns, out_q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        from lingbotvla.optim.muon import DistributedMuon

        mesh = init_device_mesh("cpu", (world_size,))
        full_params, full_grads = _full_tensors(seed=0)
        no_grad_index = 4
        params = [torch.nn.Parameter(distribute_tensor(p.clone(), mesh, [Shard(0)])) for p in full_params]
        opt = DistributedMuon(params, lr=LR, weight_decay=WD, momentum=MOMENTUM, nesterov=True,
                              adjust_lr_fn=None, distribute_ns=distribute_ns)
        for step in range(STEPS):
            for i, p in enumerate(params):
                p.grad = None if i == no_grad_index else distribute_tensor(full_grads[i][step], mesh, [Shard(0)])
            opt.step()
        got = [p.full_tensor() for p in params]
        if rank == 0:
            ref = _reference(full_params, full_grads, no_grad_index)
            # error of the applied update in units of lr (an ortho entry is O(1/sqrt(rows)))
            err = max(((a - b).abs().max() / LR).item() for a, b in zip(got, ref))
            out_q.put(err)
    finally:
        dist.destroy_process_group()


def _run(world_size: int, distribute_ns: bool) -> float:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    mp.start_processes(_worker, args=(world_size, port, distribute_ns, q), nprocs=world_size, start_method="spawn")
    return q.get()


class DistributedMuonShardingTest(unittest.TestCase):
    def test_distributed_ns_8_ranks(self):
        self.assertLess(_run(8, distribute_ns=True), 1e-3)

    def test_distributed_ns_3_ranks(self):
        self.assertLess(_run(3, distribute_ns=True), 1e-3)

    def test_allgather_path_8_ranks(self):
        self.assertLess(_run(8, distribute_ns=False), 1e-3)

    def test_layout_matches_dtensor(self):
        from lingbotvla.optim.muon import _shard_layout

        for rows, world in ((588, 8), (55, 8), (55, 16), (5, 8), (64, 8), (7, 3)):
            chunks = torch.arange(rows).chunk(world)
            sizes = [len(c) for c in chunks] + [0] * (world - len(chunks))
            chunk, offsets, got = _shard_layout(rows, world)
            self.assertEqual(got, sizes, (rows, world))
            self.assertEqual(chunk, max(sizes))
            self.assertEqual(offsets[: len(chunks)], [int(c[0]) for c in chunks])


if __name__ == "__main__":
    unittest.main()
