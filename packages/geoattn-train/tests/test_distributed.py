"""Data parallelism (ADR-0008): W processes train the same model as one process.

The multi-process test launches ranks the way torchrun would (``env://`` variables)
but without torchrun, whose rendezvous needs libuv, absent from Windows builds.
gloo on CPU, so it runs anywhere; it takes seconds, hence ``slow``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from geoattn_train.distributed import Dist, shard

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "smoke_lm.yaml"


def test_rank_slices_partition_the_global_batch() -> None:
    x = torch.arange(24).view(8, 3)
    parts = [shard(x, Dist(rank=r, world=4)) for r in range(4)]
    torch.testing.assert_close(torch.cat(parts), x)
    with pytest.raises(ValueError, match="does not split"):
        shard(x, Dist(rank=0, world=3))


def _train(out: Path, world: int, *extra: str) -> Path:
    args = [
        sys.executable,
        "-m",
        "geoattn_train.cli",
        "--config",
        str(CONFIG),
        f"output_dir={out.as_posix()}",
        "train.device=cpu",
        "data.batch_size=4",
        "lm.grad_accum=2",  # global batch 8 sequences in both runs
        *extra,
    ]
    env = {**os.environ, "USE_LIBUV": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29531"}
    procs = [
        subprocess.Popen(
            args,
            env={**env, "WORLD_SIZE": str(world), "RANK": str(r), "LOCAL_RANK": str(r)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for r in range(world)
    ]
    for p in procs:
        out_text, _ = p.communicate(timeout=600)
        assert p.returncode == 0, out_text
    (run_dir,) = (out / "smoke_lm").iterdir()
    return run_dir


@pytest.mark.slow
def test_two_ranks_match_one_process(tmp_path) -> None:
    one = _train(tmp_path / "w1", world=1)
    two = _train(tmp_path / "w2", world=2)
    a = torch.load(one / "last.pt", weights_only=True)["model"]
    b = torch.load(two / "last.pt", weights_only=True)["model"]
    for name in a:
        # Same global batches and loss scaling; only float reduction order differs.
        torch.testing.assert_close(b[name], a[name], atol=1e-4, rtol=1e-3, msg=name)
    # Counter-example: other batches (data seed) must exceed the tolerance, or the
    # comparison above could not tell a mis-sharded run from a correct one.
    other = torch.load(
        _train(tmp_path / "w1_seed1", 1, "data.seed=1") / "last.pt", weights_only=True
    )
    assert max((other["model"][n] - a[n]).abs().max().item() for n in a) > 1e-2
    meta = (two / "meta.json").read_text()
    assert '"world_size": 2' in meta and '"backend": "gloo"' in meta
