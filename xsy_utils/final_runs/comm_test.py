"""Collective correctness + bandwidth, run under torch.distributed.run on every node (test_multinode.sh).

For the whole world, the ranks of one node, and the same local rank across nodes (= the inter-node links),
prints algbw / busbw (nccl-tests convention) of all_gather / reduce_scatter (FSDP's collectives) and
all_reduce (HSDP's cross-node gradient sync). A TCP-socket fallback between nodes shows up as an
inter-node busbw of ~1-3 GB/s instead of tens of GB/s.
"""
import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist

SIZES_MB = [int(x) for x in os.environ.get("COMM_SIZES_MB", "64,512").split(",")]


def bench(op: str, group, n: int, mb: int, iters: int) -> tuple[float, float]:
    numel = mb * 2**20 // 2  # bf16; = the full (gathered) tensor for all_gather / reduce_scatter
    numel -= numel % n
    full = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
    shard = torch.ones(numel // n, dtype=torch.bfloat16, device="cuda")
    call = {
        "all_gather": lambda: dist.all_gather_into_tensor(full, shard, group=group),
        "reduce_scatter": lambda: dist.reduce_scatter_tensor(shard, full, group=group),
        "all_reduce": lambda: dist.all_reduce(full, group=group),
    }[op]
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    t = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    dt = torch.tensor([(time.perf_counter() - t) / iters], device="cuda")
    dist.all_reduce(dt, op=dist.ReduceOp.MAX, group=group)
    algbw = numel * 2 / dt.item() / 1e9
    factor = 2 * (n - 1) / n if op == "all_reduce" else (n - 1) / n
    return algbw, algbw * factor


def main() -> None:
    local_rank, rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    ngpu = int(os.environ["LOCAL_WORLD_SIZE"])
    nnodes = world // ngpu
    torch.cuda.set_device(local_rank)
    t0 = time.time()
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    # who is where (catches a wrong NNODES / NODE_RANK mapping)
    hosts = [None] * world
    dist.all_gather_object(hosts, (socket.gethostname(), local_rank))
    if rank == 0:
        per_node = {}
        for r, (h, lr) in enumerate(hosts):
            per_node.setdefault(h, []).append(r)
        print(f"[comm] init {time.time() - t0:.1f} s; world {world} = {nnodes} nodes x {ngpu}: "
              + "; ".join(f"{h}: ranks {v[0]}-{v[-1]}" for h, v in per_node.items()), flush=True)
        if len(per_node) != nnodes and os.environ.get("COMM_SIMULATED") != "1":
            print(f"[comm] WARNING: {len(per_node)} distinct hosts for {nnodes} nodes", flush=True)

    # correctness: sum of ranks, on every rank
    x = torch.full((1 << 20,), float(rank), device="cuda")
    dist.all_reduce(x)
    ok = torch.allclose(x, torch.full_like(x, world * (world - 1) / 2))
    flags = torch.tensor([int(ok)], device="cuda")
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"[comm] all_reduce correctness: {'ok' if flags.item() else 'FAILED'}", flush=True)
    if not flags.item():
        raise SystemExit("all_reduce result wrong")

    groups = [("world", None, world)]
    node_groups = [dist.new_group(list(range(n * ngpu, (n + 1) * ngpu))) for n in range(nnodes)]
    if ngpu > 1:
        groups.append((f"intra-node ({ngpu})", node_groups[rank // ngpu], ngpu))
    if nnodes > 1:
        cross = [dist.new_group([lr + n * ngpu for n in range(nnodes)]) for lr in range(ngpu)]
        groups.append((f"inter-node ({nnodes}, same local rank)", cross[local_rank], nnodes))
    for name, group, n in groups:
        for op in ("all_gather", "reduce_scatter", "all_reduce"):
            for mb in SIZES_MB:
                algbw, busbw = bench(op, group, n, mb, iters=10 if mb <= 128 else 5)
                if rank == 0:
                    print(f"[comm] {name:34s} {op:15s} {mb:5d} MB  algbw {algbw:7.1f} GB/s  busbw {busbw:7.1f} GB/s", flush=True)
    dist.barrier()
    if rank == 0:
        print("[comm] done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
