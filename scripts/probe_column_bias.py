"""Exploratory: is the rho correction an implicit, content-adaptive position bias?

    uv run python scripts/probe_column_bias.py "runs/exp0001_screen_rho*_s?" \
        --out runs/column_bias/screen.json

At the fixed point of the prefix update (ADR-0006), log v = -gamma log M with
gamma = rho / (1 - rho) and M_tj = sum_{s<=t} P_sj, so

    log P_tj = log K_tj  - gamma log M_tj  - lse_t.

The middle term sits where an additive position encoding would (pe.bias; ALiBi is the
distance-only special case). This script measures, per layer, how much of
B_tj = -gamma log M_tj (keys j <= t with M_tj > 0) is a function of the distance d = t - j:

* ``r2_distance``  1 - (within-distance variance) / (total variance)
* ``mean_by_d``    the mean of B at each distance (the implied bias curve)
* ``r2_log`` / ``r2_linear``  how well a + b log(1 + d) / a + b d fits that curve

Predictions, written down before running (2026-10-03): B is largely a function of
distance (r2_distance >= 0.5) and decreases with d (far keys have collected more mass, so
are penalized more), closer to logarithmic than linear in d; i.e. rho acts like an ALiBi-
type decay that adapts to content.
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


def distance_stats(p: Tensor, gamma: float) -> tuple[Tensor, Tensor, Tensor]:
    """Per distance d: (count, sum, sum of squares) of B = -gamma log M over (B, H, t)."""
    p = p.float()
    n = p.shape[-1]
    m = p.cumsum(dim=-2)
    b = -gamma * m.clamp_min(1e-30).log()
    valid = torch.ones(n, n, dtype=torch.bool, device=p.device).tril() & (m > 0)
    counts, sums, sqs = (torch.zeros(n, device=p.device, dtype=torch.float64) for _ in range(3))
    for d in range(n):
        diag_b = torch.diagonal(b, offset=-d, dim1=-2, dim2=-1)
        diag_v = torch.diagonal(valid, offset=-d, dim1=-2, dim2=-1)
        vals = diag_b[diag_v].double()
        counts[d], sums[d], sqs[d] = vals.numel(), vals.sum(), vals.square().sum()
    return counts, sums, sqs


def summarize(counts: Tensor, sums: Tensor, sqs: Tensor) -> dict[str, object]:
    keep = counts > 0
    c, s, q = counts[keep], sums[keep], sqs[keep]
    mean_d = s / c
    total_mean = s.sum() / c.sum()
    total_var = q.sum() / c.sum() - total_mean**2
    within = (q - c * mean_d**2).sum() / c.sum()
    d = torch.arange(counts.numel(), device=counts.device, dtype=torch.float64)[keep]

    def r2(x: Tensor) -> float:
        a = torch.stack([torch.ones_like(x), x], dim=1) * c.sqrt()[:, None]
        coef = torch.linalg.lstsq(a, mean_d * c.sqrt()).solution
        fit = coef[0] + coef[1] * x
        resid = (c * (mean_d - fit) ** 2).sum() / c.sum()
        between = (c * (mean_d - total_mean) ** 2).sum() / c.sum()
        return float(1 - resid / between)

    return {
        "r2_distance": float(1 - within / total_var),
        "r2_log": r2(torch.log1p(d)),
        "r2_linear": r2(d),
        "mean_by_d": mean_d.tolist(),
    }


@torch.no_grad()
def probe_length(model: nn.Module, batches: list, gamma: float) -> list[dict[str, object]]:
    norms = [
        b.attn.normalizer
        for b in model.modules()
        if isinstance(b, GPTBlock) and isinstance(b.attn.normalizer, nn.Module)
    ]
    acc: dict[int, list[Tensor]] = {}

    def hook(i: int):
        def fn(_m: nn.Module, _inp: tuple, out: Tensor) -> None:
            stats = distance_stats(out, gamma)
            if i in acc:
                acc[i] = [a + s for a, s in zip(acc[i], stats, strict=True)]
            else:
                acc[i] = list(stats)

        return fn

    handles = [n.register_forward_hook(hook(i)) for i, n in enumerate(norms)]
    try:
        for x, _y in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x.cuda())
    finally:
        for h in handles:
            h.remove()
    return [summarize(*acc[i]) for i in sorted(acc)]


def probe_run(run_dir: Path, mults: list[int], tokens: int) -> dict[str, object]:
    cfg = load_config(run_dir / "config.yaml", ["train.compile_normalizer=false"])
    rho = cfg.attention.normalizer.rho
    gamma = rho / (1 - rho)
    model = build_lm(cfg).cuda().eval()
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    corpus = load_corpus(cfg.data, cfg.lm, cfg.model)
    out: dict[str, object] = {"rho": rho, "gamma": gamma}
    for m in mults:
        length = m * cfg.lm.seq_len
        batches = val_batches(corpus.val, length, tokens, max(1, 8 // m))
        layers = probe_length(model, batches, gamma)
        out[str(length)] = layers
        r2 = [round(float(x["r2_distance"]), 3) for x in layers]  # type: ignore[arg-type]
        print(f"  {run_dir.parent.name} L={length} r2_distance per layer {r2}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("patterns", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mults", type=int, nargs="+", default=[1, 2])
    p.add_argument("--tokens", type=int, default=32_768)
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
        per_run[f"{run_dir.parent.name}/{run_dir.name}"] = probe_run(
            run_dir, args.mults, args.tokens
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**result, "runs": per_run}), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
