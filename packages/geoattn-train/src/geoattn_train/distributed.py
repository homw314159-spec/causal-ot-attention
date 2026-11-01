"""Data parallelism for the LM trainer (ADR-0008).

A step's global batch is defined by ``(data.seed, step)`` alone (lm_data), so the
world size only decides who computes which rows: rank r takes the r-th contiguous
slice of the global batch. Tokens per step, data order and the LR schedule are the
same for any number of processes, and a W-process run is comparable to a
single-process one up to float reduction order.

Launch with ``torchrun --nproc_per_node W -m geoattn_train.cli --config ...``; without
torchrun's environment everything here degrades to a single process.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    backend: str | None = None

    @property
    def main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world > 1


def init_distributed(device: torch.device) -> tuple[Dist, torch.device]:
    """Join the process group torchrun described (if any); returns the rank's device."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return Dist(), device
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if device.type == "cuda":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    # NCCL is Linux-only; gloo covers CPU and Windows (slower, but correct).
    backend = "nccl" if device.type == "cuda" and sys.platform != "win32" else "gloo"
    dist.init_process_group(backend)
    return Dist(rank=rank, world=world, local_rank=local, backend=backend), device


def shutdown(d: Dist) -> None:
    if d.enabled and dist.is_initialized():
        dist.destroy_process_group()


def shard(x: Tensor, d: Dist) -> Tensor:
    """Rank ``d.rank``'s contiguous slice of a global batch (rows must divide evenly)."""
    rows = x.shape[0]
    if rows % d.world:
        raise ValueError(f"global batch of {rows} sequences does not split over {d.world} ranks")
    n = rows // d.world
    return x[d.rank * n : (d.rank + 1) * n]


def all_reduce_sum(values: list[float], d: Dist, device: torch.device) -> list[float]:
    """Sum python floats over ranks (identity in a single process)."""
    if not d.enabled:
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()
