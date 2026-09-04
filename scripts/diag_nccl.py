#!/usr/bin/env python
"""Minimal multi-GPU collective test — isolates NCCL health from the training stack.

M2's first launch deadlocked in a `dist.barrier()` inside HF's `main_process_first`, with all
ranks spinning at 100% GPU util / 93W / 700MB (i.e. NCCL busy-wait, not compute). This script
reproduces just the collective, so we can tell an NCCL/topology problem from a LLaMA-Factory one.

This node has no NVLink: GPUs 0-3 sit on NUMA node 0 and 4-7 on node 1, linked by `SYS`
(across the inter-socket interconnect) — a configuration where NCCL P2P frequently hangs.

Run (short init timeout so it fails fast instead of hanging):
    .venv-train/bin/torchrun --nproc_per_node=8 scripts/diag_nccl.py
    NCCL_P2P_DISABLE=1 .venv-train/bin/torchrun --nproc_per_node=8 scripts/diag_nccl.py
"""

import datetime
import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    # Short timeout: we want a fast failure, not a 30-minute hang.
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=90),
        device_id=dev,  # avoids the "devices used by this process are currently unknown" guess
    )

    if rank == 0:
        print(f"[init] world_size={world}", flush=True)

    # 1) barrier — the exact collective that deadlocked
    dist.barrier()
    if rank == 0:
        print("[ok] barrier", flush=True)

    # 2) small all_reduce
    t = torch.ones(1024, device=dev) * rank
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = world * (world - 1) / 2
    assert t[0].item() == expected, f"all_reduce wrong: {t[0].item()} != {expected}"
    if rank == 0:
        print("[ok] all_reduce (small)", flush=True)

    # 3) large all_reduce — ZeRO-3 all-gathers parameters every step, so bandwidth matters
    big = torch.ones(64 * 1024 * 1024, device=dev)  # 256MB fp32
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        dist.all_reduce(big, op=dist.ReduceOp.SUM)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 5
    gb = big.numel() * 4 / 2**30
    if rank == 0:
        # busbw for ring all-reduce ~ 2*(N-1)/N * size / time
        busbw = 2 * (world - 1) / world * gb / (ms / 1000)
        print(f"[ok] all_reduce 256MB: {ms:.1f} ms/iter, bus bw ~{busbw:.1f} GB/s", flush=True)

    dist.barrier()
    if rank == 0:
        print("[PASS] all collectives completed", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
