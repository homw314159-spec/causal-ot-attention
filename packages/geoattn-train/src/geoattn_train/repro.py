"""Reproducibility metadata without cloud-account discovery."""

from __future__ import annotations

import platform
import random
import subprocess
from datetime import UTC, datetime
from typing import Any

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def nvidia_driver() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def run_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "driver": nvidia_driver() if device.type == "cuda" else None,
        "cloud": None,
        "platform": platform.platform(),
    }
