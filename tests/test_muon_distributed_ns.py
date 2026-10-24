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


def _moe_worker(rank, world_size, port, num_experts, out_q):
    """MoE expert stacks [E, out, in] (qwen2_action_expert), sharded on the expert dim like FSDP2: with more ranks
    than experts some ranks hold no expert (32 experts on 64 GPUs)."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        import functools

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        import lingbotvla.optim.muon as M

        # Newton-Schulz in fp32 on both sides: in bf16 a last-bit difference of the momentum lerp (vectorised over
        # the whole stack vs one expert) can flip one rounding and show up as ~1e-2 lr in a single entry.
        M.batched_newton_schulz = batched_newton_schulz = functools.partial(M.batched_newton_schulz,
                                                                            compute_dtype=torch.float32)
        DistributedMuon, _adjust_lr = M.DistributedMuon, M._adjust_lr

        mesh = init_device_mesh("cpu", (world_size,))
        g = torch.Generator().manual_seed(1)
        shapes = [(num_experts, 24, 16), (num_experts, 24, 16), (num_experts, 16, 24)]
        full = [torch.randn(s, generator=g) for s in shapes]
        grads = [[torch.randn(s, generator=g) for _ in range(STEPS)] for s in shapes]
        params = [torch.nn.Parameter(distribute_tensor(p.clone(), mesh, [Shard(0)])) for p in full]
        opt = DistributedMuon(params, lr=LR, weight_decay=WD, momentum=MOMENTUM, nesterov=True, adjust_lr_fn=None)
        for step in range(STEPS):
            for p, gr in zip(params, grads):
                p.grad = distribute_tensor(gr[step], mesh, [Shard(0)])
            opt.step()
        got = [p.full_tensor() for p in params]
        if rank == 0:
            ref, bufs = [p.clone() for p in full], [torch.zeros_like(p) for p in full]
            for step in range(STEPS):
                for i, p in enumerate(ref):
                    gr = grads[i][step]
                    bufs[i].lerp_(gr, 1 - MOMENTUM)
                    ortho = batched_newton_schulz(gr.lerp(bufs[i], MOMENTUM))  # per expert matrix
                    p.mul_(1 - LR * WD).add_(ortho, alpha=-_adjust_lr(LR, None, p.shape[-2:]))
            out_q.put(max(((a - b).abs().max() / LR).item() for a, b in zip(got, ref)))
    finally:
        dist.destroy_process_group()


def _run(world_size: int, distribute_ns: bool = True, num_experts: int = 0) -> float:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    if num_experts:
        mp.start_processes(_moe_worker, args=(world_size, port, num_experts, q), nprocs=world_size, start_method="spawn")
    else:
        mp.start_processes(_worker, args=(world_size, port, distribute_ns, q), nprocs=world_size, start_method="spawn")
    return q.get()


class DistributedMuonShardingTest(unittest.TestCase):
    def test_distributed_ns_8_ranks(self):
        self.assertLess(_run(8, distribute_ns=True), 1e-3)

    def test_distributed_ns_3_ranks(self):
        self.assertLess(_run(3, distribute_ns=True), 1e-3)

    def test_allgather_path_8_ranks(self):
        self.assertLess(_run(8, distribute_ns=False), 1e-3)

    def test_moe_experts_fewer_than_ranks(self):
        self.assertLess(_run(8, num_experts=4), 1e-5)  # 4 of 8 ranks hold no expert (as 32 experts on 64 GPUs)

    def test_layout_matches_dtensor(self):
        from lingbotvla.optim.muon import _shard_layout

        for rows, world in ((588, 8), (55, 8), (55, 16), (588, 32), (55, 32), (588, 64), (55, 64), (5, 8), (64, 8), (7, 3)):
            chunks = torch.arange(rows).chunk(world)
            sizes = [len(c) for c in chunks] + [0] * (world - len(chunks))
            chunk, offsets, got = _shard_layout(rows, world)
            self.assertEqual(got, sizes, (rows, world))
            self.assertEqual(chunk, max(sizes))
            self.assertEqual(offsets[: len(chunks)], [int(c[0]) for c in chunks])


if __name__ == "__main__":
    unittest.main()
