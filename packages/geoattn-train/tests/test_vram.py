import pytest
import torch

from geoattn_train import vram
from geoattn_train.vram import GB, cap_process_memory, device_memory_gb


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cap_turns_overallocation_into_oom() -> None:
    """Past the cap PyTorch must raise, not spill into system RAM (WDDM sysmem fallback)."""
    try:
        cap_process_memory(1.0)
        torch.empty(int(0.5 * GB), dtype=torch.uint8, device="cuda")  # within the cap
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(int(2 * GB), dtype=torch.uint8, device="cuda")
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_memory_reads_nvidia_smi() -> None:
    used, total = device_memory_gb()
    assert 0 < used < total


def test_unified_memory_falls_back_to_meminfo(monkeypatch) -> None:
    """GB10 (DGX Spark) shares system RAM: nvidia-smi reports "[N/A]" for memory."""
    monkeypatch.setattr(vram.subprocess, "check_output", lambda *a, **k: "[N/A], [N/A]\n")
    meminfo = "MemTotal:       125829120 kB\nMemFree:  1 kB\nMemAvailable:   94371840 kB\n"
    monkeypatch.setattr(vram, "_read_meminfo", lambda: meminfo)
    used, total = device_memory_gb()
    assert total == pytest.approx(120.0)
    assert used == pytest.approx(30.0)


def test_unparseable_nvidia_smi_output_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(vram.subprocess, "check_output", lambda *a, **k: "garbage\n")
    with pytest.raises(RuntimeError, match="nvidia-smi"):
        device_memory_gb()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cap_accepts_device_without_index() -> None:
    """train.device=auto resolves to torch.device("cuda"), which has no index."""
    try:
        fraction = cap_process_memory(1.0, torch.device("cuda"))
        assert 0 < fraction <= 1
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
