"""Exploratory: is the length-extrapolation gap a temperature effect? (report 0005)

    uv run python scripts/probe_length_temperature.py runs/<exp>/<stamp> [...] \
        --out runs/length_probe/<name>.json

For each trained GPT, on CPU, at twice the training length (2L) on the first
``--tokens`` validation tokens:

1. Loss with the attention logits scaled at evaluation time only: a constant factor
   f (normalizer eps -> eps / f) and a length-aware per-row factor
   max(1, log(t + 1) / log(L)) (row t, 0-based), the log-n rule of Scalable-Softmax.
2. Loss split by position: [0, L) and [L, 2L). A causal model computes positions
   < L exactly as at length L (prefix consistency), so the extrapolation gap lives
   entirely in the second half.
3. Mean row entropy of every layer's attention as a function of position, in
   buckets of 64 rows, at f = 1 (computed in a hook; no matrices are kept).

Results go to a JSON file with the code version, for docs/experiments.md
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geoattn_core import AttentionProbe
from geoattn_train.config import load_config
from geoattn_train.lm_data import load_corpus, val_batches
from geoattn_train.lm_trainer import build_lm
from geoattn_train.models import GPT, GPTBlock

FACTORS = (1.0, 1.05, 1.1, 1.15, 1.2, 1.3)
BUCKET = 64


class RowScaled(nn.Module):
    """Multiplies row t of the scores by ``scale[t]`` before the wrapped normalizer."""

    def __init__(self, inner: nn.Module, scale: Tensor) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("scale", scale, persistent=False)
        self.scale: Tensor

    def forward(self, scores: Tensor, mask: Tensor | None = None) -> Tensor:
        n = scores.shape[-2]
        return self.inner(scores * self.scale[:n, None], mask)


def load_model(run_dir: Path) -> GPT:
    cfg = load_config(run_dir / "config.yaml", ["train.compile_normalizer=false"])
    model = build_lm(cfg).eval()
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def direct_first_half(model: GPT, batches: list[tuple[Tensor, Tensor]], seq_len: int) -> float:
    """Loss of a length-L forward on the first halves: must equal ``first_half`` at f = 1."""
    total, count = 0.0, 0
    for x, y in batches:
        logits = model(x[:, :seq_len])
        total += F.cross_entropy(
            logits.flatten(0, 1), y[:, :seq_len].flatten(), reduction="sum"
        ).item()
        count += y[:, :seq_len].numel()
    return total / count


def normalizers(model: GPT) -> list[tuple[GPTBlock, nn.Module]]:
    out = []
    for b in model.modules():
        if isinstance(b, GPTBlock) and isinstance(b.attn.normalizer, nn.Module):
            out.append((b, b.attn.normalizer))
    return out


def set_eps_factor(model: GPT, f: float, originals: list[float]) -> None:
    for (_, norm), eps in zip(normalizers(model), originals, strict=True):
        norm.eps = eps / f  # type: ignore[assignment]


@torch.no_grad()
def split_loss(model: GPT, batches: list[tuple[Tensor, Tensor]], seq_len: int) -> dict[str, float]:
    """Mean nats/token over [0, L), [L, 2L) and all positions."""
    sums = torch.zeros(2, dtype=torch.float64)
    counts = torch.zeros(2, dtype=torch.float64)
    for x, y in batches:
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none").view(y.shape)
        for h, part in enumerate((loss[:, :seq_len], loss[:, seq_len:])):
            sums[h] += part.double().sum()
            counts[h] += part.numel()
    first, second = (sums / counts).tolist()
    return {"first_half": first, "second_half": second, "all": (sums.sum() / counts.sum()).item()}


@torch.no_grad()
def entropy_by_position(model: GPT, batches: list[tuple[Tensor, Tensor]]) -> list[list[float]]:
    """Per layer: mean row entropy (nats) per bucket of BUCKET positions."""
    probes = [m for m in model.modules() if isinstance(m, AttentionProbe)]
    sums: list[Tensor | None] = [None] * len(probes)
    count = 0

    def hook_for(i: int):
        def hook(_m: nn.Module, _inp: tuple[Tensor, ...], p: Tensor) -> None:
            ent = -(p * p.clamp_min(1e-30).log()).sum(-1)  # (B, H, N)
            per_pos = ent.double().sum(dim=(0, 1))  # (N,)
            prev = sums[i]
            sums[i] = per_pos if prev is None else prev + per_pos

        return hook

    handles = [p.register_forward_hook(hook_for(i)) for i, p in enumerate(probes)]
    for x, _ in batches:
        model(x)
        count += x.shape[0] * model.blocks[0].attn.num_heads  # type: ignore[union-attr]
    for h in handles:
        h.remove()
    out = []
    for s in sums:
        assert s is not None
        mean = (s / count).view(-1, BUCKET).mean(-1)
        out.append([round(v, 5) for v in mean.tolist()])
    return out


def probe_run(run_dir: Path, tokens: int, batch: int) -> dict[str, object]:
    model = load_model(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seq_len = cfg.lm.seq_len
    corpus = load_corpus(cfg.data, cfg.lm, cfg.model)
    batches = val_batches(corpus.val, 2 * seq_len, tokens, batch)
    originals = [float(norm.eps) for _, norm in normalizers(model)]  # type: ignore[arg-type]

    constant = {}
    for f in FACTORS:
        set_eps_factor(model, f, originals)
        constant[str(f)] = split_loss(model, batches, seq_len)
        print(f"  {run_dir.parent.name} f={f}: {constant[str(f)]}", flush=True)
    set_eps_factor(model, 1.0, originals)

    t = torch.arange(2 * seq_len, dtype=torch.float32)
    scale = torch.clamp(torch.log(t + 1) / math.log(seq_len), min=1.0)
    for block, norm in normalizers(model):
        block.attn.normalizer = RowScaled(norm, scale)
    log_n = split_loss(model, batches, seq_len)
    print(f"  {run_dir.parent.name} log-n: {log_n}", flush=True)
    for block, norm in normalizers(model):
        block.attn.normalizer = norm.inner  # type: ignore[assignment]

    return {
        "seq_len": seq_len,
        "prefix_check": {
            "first_half_at_2L": constant["1.0"]["first_half"],
            "direct_length_L": direct_first_half(model, batches, seq_len),
        },
        "tokens": sum(y.numel() for _, y in batches),
        "constant_factor": constant,
        "log_n_rows": log_n,
        "entropy_by_position": entropy_by_position(model, batches),
    }


def git_state() -> dict[str, object]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    porcelain = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"sha": sha.strip(), "dirty": bool(porcelain.stdout.strip())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=262_144, help="predictions per length")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    result: dict[str, object] = {"git": git_state(), "factors": FACTORS, "bucket": BUCKET}
    per_run = {}
    for run_dir in args.runs:
        per_run[f"{run_dir.parent.name}/{run_dir.name}"] = probe_run(
            run_dir, args.tokens, args.batch
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**result, "runs": per_run}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
