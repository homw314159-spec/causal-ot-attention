"""Exploratory: does the rho extrapolation advantage survive evaluation-time RoPE scaling?

    uv run python scripts/probe_rope_scaling.py "runs/exp0001_screen_rho*_s?" \
        "runs/exp0001_screen_softmax_eps*_s?" --out runs/rope_scaling/screen.json

RoPE fails beyond the training length partly because it meets rotation angles it never saw.
The standard fixes rescale the angles at evaluation time; none needs training:

* ``pi``      position interpolation: positions / s
* ``ntk``     NTK-aware: base * s^(d / (d - 2))
* ``yarn``    YaRN frequency ramp (alpha 1, beta 32): long wavelengths interpolated, short
              ones kept, linear in between
* ``yarn_t``  YaRN plus its attention temperature: q and k scaled by 0.1 ln s + 1

with s = n / L (training length L), so every method is the identity at n <= L.

Predictions, written down before running (2026-10-03): if the rho advantage at 2x / 4x
came mainly from RoPE meeting unseen angles, it would shrink to about zero once the angles
are rescaled; if it comes from attention dilution (report 0009, probe P1-P4), it persists
under pi / ntk / yarn and shrinks only partly under yarn_t, whose temperature term is itself
a global anti-dilution correction (like log-n softmax).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geoattn_core.pe.rope import RoPE, _rotate
from geoattn_train.config import load_config
from geoattn_train.lm_data import load_corpus, val_batches
from geoattn_train.lm_trainer import build_lm, compile_normalizer
from geoattn_train.models import GPTBlock

METHODS = ("none", "pi", "ntk", "yarn", "yarn_t")


class ScaledRoPE(nn.Module):
    """``rope`` with evaluation-time angle rescaling beyond ``train_len``."""

    def __init__(self, rope: RoPE, method: str, train_len: int) -> None:
        super().__init__()
        self.rope, self.method, self.train_len = rope, method, train_len

    def _inv_freq_and_scale(self, n: int) -> tuple[Tensor, Tensor, float]:
        """(inverse frequencies, position divisor, q/k multiplier) for length n."""
        inv = self.rope.inv_freq
        s = max(1.0, n / self.train_len)
        one = torch.ones((), device=inv.device)
        if s == 1.0 or self.method == "none":
            return inv, one, 1.0
        d = self.rope.head_dim
        if self.method == "pi":
            return inv, one * s, 1.0
        if self.method == "ntk":
            base = self.rope.base * s ** (d / (d - 2))
            idx = torch.arange(0, d, 2, device=inv.device, dtype=torch.float32)
            return base ** (-idx / d), one, 1.0
        # YaRN: ramp over r = train_len / wavelength between alpha = 1 and beta = 32.
        r = self.train_len * inv / (2 * math.pi)
        keep = ((r - 1.0) / (32.0 - 1.0)).clamp(0.0, 1.0)
        yarn = (1 - keep) * inv / s + keep * inv
        mult = 0.1 * math.log(s) + 1.0 if self.method == "yarn_t" else 1.0
        return yarn, one, mult

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        inv, div, mult = self._inv_freq_and_scale(positions.shape[-1])
        angles = (positions.to(torch.float32) / div)[..., None] * inv
        cos, sin = angles.cos(), angles.sin()
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        return (q * mult, k * mult) if mult != 1.0 else (q, k)

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None:
        return None


@torch.no_grad()
def mean_loss(model: nn.Module, batches: list[tuple[Tensor, Tensor]]) -> float:
    total, count = 0.0, 0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda())
        total += F.cross_entropy(logits.float().flatten(0, 1), y.cuda().flatten(), reduction="sum")
        count += y.numel()
    return float(total / count)


def probe_run(run_dir: Path, mults: list[int], tokens: int) -> dict[str, dict[str, float]]:
    cfg = load_config(run_dir / "config.yaml", ["train.compile_normalizer=false"])
    model = build_lm(cfg).cuda().eval()
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    blocks = [b for b in model.modules() if isinstance(b, GPTBlock)]
    for b in blocks:
        assert isinstance(b.attn.normalizer, nn.Module)
        compile_normalizer(b.attn.normalizer)
    ropes = [b.attn.pe for b in blocks]
    assert all(isinstance(r, RoPE) for r in ropes)
    corpus = load_corpus(cfg.data, cfg.lm, cfg.model)
    out: dict[str, dict[str, float]] = {}
    for m in mults:
        length = m * cfg.lm.seq_len
        batches = val_batches(corpus.val, length, tokens, max(1, 8 // m))
        out[str(length)] = {}
        for method in METHODS if m > 1 else ("none",):
            for b, rope in zip(blocks, ropes, strict=True):
                assert isinstance(rope, RoPE)
                b.attn.pe = ScaledRoPE(rope, method, cfg.lm.seq_len)
            out[str(length)][method] = mean_loss(model, batches)
        print(f"  {run_dir.parent.name} L={length}: {out[str(length)]}", flush=True)
    return out


def main() -> None:
    torch._dynamo.config.cache_size_limit = 256
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("patterns", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--tokens", type=int, default=1_048_576)
    args = p.parse_args()
    runs = sorted(
        Path(r) for pat in args.patterns for exp in glob.glob(pat) for r in glob.glob(f"{exp}/*")
    )
    runs = [r for r in runs if (r / "last.pt").exists()]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    result: dict[str, object] = {"git": {"sha": sha.strip(), "dirty": bool(dirty.strip())}}
    per_run: dict[str, object] = {}
    for run_dir in runs:
        key = f"{run_dir.parent.name}/{run_dir.name}"
        per_run[key] = probe_run(run_dir, args.mults, args.tokens)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**result, "runs": per_run}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
