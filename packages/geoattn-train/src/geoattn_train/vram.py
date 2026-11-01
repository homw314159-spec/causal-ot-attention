"""VRAM guards.

On Windows (WDDM) the NVIDIA driver does not raise OOM when device memory runs
out: by default it spills into shared system RAM ("sysmem fallback"), which
freezes the whole machine instead of failing one run. So every training
process caps its own allocator (an allocation past the cap raises
``torch.OutOfMemoryError`` cleanly), and schedulers admit jobs against a VRAM
budget measured with ``nvidia-smi`` (no CUDA context in the scheduler).
"""

from __future__ import annotations

import subprocess

import torch

GB = 2**30
CONTEXT_GB = 0.5  # CUDA context + cuDNN/cuBLAS workspaces, outside the allocator cap


def cap_process_memory(limit_gb: float, device: torch.device | int = 0) -> float:
    """Cap this process's PyTorch allocator at ``limit_gb``; returns the fraction set."""
    if isinstance(device, torch.device):
        # torch >= 2.14 rejects an index-less torch.device("cuda") (what device=auto gives).
        device = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(limit_gb * GB / total, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return fraction


def device_memory_gb(index: int = 0) -> tuple[float, float]:
    """``(used, total)`` in GB for GPU ``index``, all processes, via nvidia-smi.

    On unified-memory devices (nvidia-smi reports ``[N/A]``) this is system RAM.
    """
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    fields = [v.strip() for v in out.strip().split(",")]
    if fields == ["[N/A]", "[N/A]"]:
        # Unified memory (GB10 / DGX Spark): the GPU allocates from system RAM, so the
        # budget that matters is the system's, and nvidia-smi does not report it.
        return _system_memory_gb()
    try:
        used, total = (float(v) / 1024 for v in fields)
    except ValueError:
        raise RuntimeError(f"unexpected nvidia-smi memory output: {out!r}") from None
    return used, total


def _read_meminfo() -> str:
    with open("/proc/meminfo", encoding="ascii") as f:
        return f.read()


def _system_memory_gb() -> tuple[float, float]:
    """``(used, total)`` in GB, with "used" = everything not available to new allocations."""
    kb = {}
    for line in _read_meminfo().splitlines():
        key, _, rest = line.partition(":")
        kb[key] = float(rest.split()[0])
    total = kb["MemTotal"] * 1024 / GB
    return total - kb["MemAvailable"] * 1024 / GB, total
