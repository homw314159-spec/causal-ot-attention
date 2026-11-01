"""Exploratory: how the attention distribution changes beyond the training length.

    uv run python scripts/probe_length_attention.py "runs/exp0001_screen_*" \
        --out runs/length_attention/screen.json

Predictions, written down before running (2026-10-03; derived from the fixed point of
ADR-0006, log v* = -gamma log M with gamma = rho / (1 - rho), i.e. P_tj ~ K_tj M_tj^-gamma:
a key that has already received much attention is down-weighted):

* P1  attention entropy per query position t stays roughly flat past the training length
      for rho > 0, while for softmax it keeps growing (dilution, ~ log t)
* P2  "noise mass" (1 - mass of the top-8 keys) grows with t for softmax, not (or much
      less) for rho > 0
* P3  the column cumulative mass M_tj = sum_{s<=t} P_sj stays more evenly spread for
      rho > 0: its dispersion over keys j <= t (std, the mean being exactly 1) and the share
      of the most-attended key (max_j M_tj / t) grow less with t than for softmax
* P4  all three effects get stronger monotonically in gamma (rho 0.1 < 0.25 < 0.5 < 0.75)

For every run directory (config.yaml + last.pt), every length (multiples of the training
length) and every layer, the normalizer output P is captured by a forward hook and
reduced per query position, then averaged over sequences and heads and bucketed. Eager,
uncompiled normalizers; bf16 autocast as in training. Inference only.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import torch
from torch import Tensor, nn

from geoattn_train.config import load_config
from geoattn_train.lm_data import load_corpus, val_batches
from geoattn_train.lm_trainer import build_lm
from geoattn_train.models import GPTBlock

TOP_K = 8
STATS = ("entropy", "noise_mass", "colmass_sd", "top_col_share", "sink")


def row_stats(p: Tensor) -> dict[str, Tensor]:
    """Per-query-position statistics of one layer's P (B, H, N, N), averaged over B and H."""
    p = p.float()
    n = p.shape[-1]
    t = torch.arange(1, n + 1, device=p.device, dtype=torch.float32)
    logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    entropy = -(p * logp).sum(-1)
    noise = 1 - p.topk(min(TOP_K, n), dim=-1).values.sum(-1)
    m = p.cumsum(dim=-2)  # M_tj, zero for j > t under the causal mask
    # Mean of M_tj over j <= t is exactly 1 (t rows of mass 1 over t keys).
    colmass_sd = ((m.square().sum(-1) / t) - 1).clamp_min(0).sqrt()
    top_share = m.amax(-1) / t
    sink = p[..., 0]
    out = {
        "entropy": entropy,
        "noise_mass": noise,
        "colmass_sd": colmass_sd,
        "top_col_share": top_share,
        "sink": sink,
    }
    return {k: v.mean(dim=(0, 1)) for k, v in out.items()}


@torch.no_grad()
def probe_length(
    model: nn.Module, batches: list[tuple[Tensor, Tensor]], device: torch.device
) -> dict[str, list[list[float]]]:
    """stat -> layer -> per-position mean over all sequences."""
    norms = [
        b.attn.normalizer
        for b in model.modules()
        if isinstance(b, GPTBlock) and isinstance(b.attn.normalizer, nn.Module)
    ]
    sums: dict[int, dict[str, Tensor]] = {}
    count = 0

    def hook(index: int):
        def fn(_m: nn.Module, _i: tuple, out: Tensor) -> None:
            s = row_stats(out)
            acc = sums.setdefault(index, {k: torch.zeros_like(v) for k, v in s.items()})
            for k, v in s.items():
                acc[k] += v * out.shape[0]

        return fn

    handles = [n.register_forward_hook(hook(i)) for i, n in enumerate(norms)]
    try:
        for x, _y in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x.to(device))
            count += x.shape[0]
    finally:
        for h in handles:
            h.remove()
    return {k: [(sums[i][k] / count).tolist() for i in sorted(sums)] for k in STATS}


def probe_run(run_dir: Path, mults: list[int], tokens: int) -> dict[str, object]:
    cfg = load_config(run_dir / "config.yaml", ["train.compile_normalizer=false"])
    device = torch.device("cuda")
    model = build_lm(cfg).to(device).eval()
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    corpus = load_corpus(cfg.data, cfg.lm, cfg.model)
    out: dict[str, object] = {"normalizer": cfg.attention.normalizer.__dict__}
    for m in mults:
        length = m * cfg.lm.seq_len
        batches = val_batches(corpus.val, length, tokens, max(1, 8 // m))
        out[str(length)] = probe_length(model, batches, device)
        print(f"  {run_dir.parent.name} L={length}", flush=True)
    return out


def git_state() -> dict[str, object]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    porcelain = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"sha": sha.strip(), "dirty": bool(porcelain.stdout.strip())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("patterns", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--tokens", type=int, default=65_536)
    args = p.parse_args()
    runs = sorted(
        Path(r) for pat in args.patterns for exp in glob.glob(pat) for r in glob.glob(f"{exp}/*")
    )
    runs = [r for r in runs if (r / "last.pt").exists()]
    result: dict[str, object] = {"git": git_state(), "mults": args.mults, "tokens": args.tokens}
    per_run: dict[str, object] = {}
    for run_dir in runs:
        per_run[f"{run_dir.parent.name}/{run_dir.name}"] = probe_run(
            run_dir, args.mults, args.tokens
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**result, "runs": per_run}), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
