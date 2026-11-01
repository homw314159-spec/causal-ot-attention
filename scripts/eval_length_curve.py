"""Exploratory: validation loss of trained LMs at 1x-8x the training length (report 0007).

    uv run python scripts/eval_length_curve.py "runs/exp0001_screen_*" \
        --out runs/length_curve/x.json

For every run directory (config.yaml + last.pt) and every length L x m (m in --mults):

* loss on the first ``--tokens`` predictions of the validation stream, split into
  buckets of ``--bucket`` positions (the same stream prefix at every length), so the
  curve shows whether far positions get worse (dilution) or better (more context);
* for unbalanced normalizers, the fixed-point residual max |P_k - P_(k-1)| at the
  training k on a few sequences of that length, so a failure to extrapolate can be
  told apart from an operator that no longer converges at longer lengths.

Inference only (bf16 autocast, compiled normalizer as in training).
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geoattn_core import UnbalancedSinkhorn
from geoattn_train.config import load_config
from geoattn_train.lm_data import load_corpus, val_batches
from geoattn_train.lm_trainer import build_lm
from geoattn_train.models import GPTBlock


@torch.no_grad()
def bucket_losses(
    model: nn.Module, batches: list[tuple[Tensor, Tensor]], bucket: int, device: torch.device
) -> list[float]:
    sums: Tensor | None = None
    count = 0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.to(device))
        loss = F.cross_entropy(
            logits.float().flatten(0, 1), y.to(device).flatten(), reduction="none"
        )
        per_pos = loss.view(y.shape).double().sum(0)  # (N,)
        sums = per_pos if sums is None else sums + per_pos
        count += y.shape[0]
    assert sums is not None
    return (sums / count).view(-1, bucket).mean(-1).tolist()


@torch.no_grad()
def residual_at_length(model: nn.Module, x: Tensor, device: torch.device) -> float | None:
    """max |P_k - P_(k-1)| over layers/heads at the model's own k (None for softmax)."""
    captured: list[tuple[nn.Module, Tensor]] = []
    norms = [
        b.attn.normalizer
        for b in model.modules()
        if isinstance(b, GPTBlock) and isinstance(b.attn.normalizer, nn.Module)
    ]
    if not all(isinstance(n, UnbalancedSinkhorn) for n in norms):
        return None
    hooks = [
        n.register_forward_hook(lambda m, i, _o: captured.append((m, i[0].float()))) for n in norms
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(x.to(device))
    for h in hooks:
        h.remove()
    worst = 0.0
    for m, scores in captured:
        assert isinstance(m, UnbalancedSinkhorn)
        n = scores.shape[-1]
        mask = torch.ones(n, n, dtype=torch.bool, device=device).tril()
        a = UnbalancedSinkhorn(rho=m.rho, n_iters=m.n_iters)(scores, mask)
        b = UnbalancedSinkhorn(rho=m.rho, n_iters=m.n_iters - 1)(scores, mask)
        worst = max(worst, (a - b).abs().max().item())
    return worst


def evaluate_run(run_dir: Path, mults: list[int], tokens: int, bucket: int) -> dict[str, object]:
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device("cuda")
    model = build_lm(cfg).to(device).eval()
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    corpus = load_corpus(cfg.data, cfg.lm, cfg.model)
    out: dict[str, object] = {}
    for m in mults:
        length = m * cfg.lm.seq_len
        batch = max(1, 8 // m)
        batches = val_batches(corpus.val, length, tokens, batch)
        buckets = bucket_losses(model, batches, bucket, device)
        mean = sum(buckets) / len(buckets)
        out[str(length)] = {
            "mean": mean,
            "buckets": buckets,
            "residual": residual_at_length(model, batches[0][0][:1], device),
        }
        print(f"  {run_dir.parent.name} L={length}: {mean:.4f}", flush=True)
    return out


def git_state() -> dict[str, object]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    porcelain = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"sha": sha.strip(), "dirty": bool(porcelain.stdout.strip())}


def main() -> None:
    # One compiled normalizer per layer x length: lift dynamo's recompile limit so it does
    # not silently fall back to eager (slower, same numerics).
    torch._dynamo.config.cache_size_limit = 256
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("patterns", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mults", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--tokens", type=int, default=1_048_576)
    p.add_argument("--bucket", type=int, default=512)
    args = p.parse_args()
    runs = sorted(
        Path(r) for pat in args.patterns for exp in glob.glob(pat) for r in glob.glob(f"{exp}/*")
    )
    runs = [r for r in runs if (r / "last.pt").exists()]
    result: dict[str, object] = {"git": git_state(), "mults": args.mults, "bucket": args.bucket}
    per_run: dict[str, object] = {}
    for run_dir in runs:
        per_run[f"{run_dir.parent.name}/{run_dir.name}"] = evaluate_run(
            run_dir, args.mults, args.tokens, args.bucket
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**result, "runs": per_run}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
