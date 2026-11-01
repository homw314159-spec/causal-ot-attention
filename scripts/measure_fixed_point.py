"""Gate G4 of experiment 0001: fixed-point residual of UnbalancedSinkhorn on real scores.

    uv run python scripts/measure_fixed_point.py runs/<name>/<stamp> [runs/... more]

For each run directory (a trained GPT: ``config.yaml`` + ``last.pt``), capture the
attention scores every layer feeds its normalizer on a few validation sequences,
then for every rho in the preregistered grid and every iteration count k report

    residual(k) = max |P_k - P_{k-1}|        (P_k = UnbalancedSinkhorn(rho, n_iters=k))

as the max over all layers, heads and sequences, and its p99 over (layer, sequence,
head) matrices. The smallest k with max residual < --tol is the n_iters to fix for
screening; the table goes into the experiment's change log. JSON is written next to
each run as ``fixed_point.json``.

The step residual understates the error when convergence is slow (report 0003), so
the distance to the fixed point, max |P_k - P*| with P* after ``--reference-iters``
iterations, is reported as well (``dist`` / ``needed_dist``); 0 turns it off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from geoattn_core import UnbalancedSinkhorn
from geoattn_train.config import load_config
from geoattn_train.lm_data import load_corpus, val_batches
from geoattn_train.lm_trainer import build_lm
from geoattn_train.models import GPTBlock

RHOS = (0.1, 0.25, 0.5, 0.75, 0.9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--max-iters", type=int, default=40)
    p.add_argument("--sequences", type=int, default=8)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--reference-iters", type=int, default=400)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    for run_dir in args.runs:
        scores = capture_scores(run_dir, args.sequences, torch.device(args.device))
        table = {rho: residuals(scores, rho, args.max_iters, args.reference_iters) for rho in RHOS}
        report(run_dir, table, args.tol)


def capture_scores(run_dir: Path, sequences: int, device: torch.device) -> list[Tensor]:
    """Per layer: the ``(B, H, N, N)`` scores its normalizer received (float32, CPU)."""
    # Eager: capturing scores needs no compilation (which on CPU needs a host C++ compiler).
    cfg = load_config(run_dir / "config.yaml", ["train.compile_normalizer=false"])
    model = build_lm(cfg)
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    corpus = load_corpus(cfg.data, cfg.lm, cfg.model)
    x, _ = val_batches(corpus.val, cfg.lm.seq_len, sequences * cfg.lm.seq_len, sequences)[0]

    captured: list[Tensor] = []
    handles = [
        block.attn.normalizer.register_forward_hook(
            lambda _m, inputs, _out: captured.append(inputs[0].detach().float().cpu())
        )
        for block in model.modules()
        if isinstance(block, GPTBlock) and isinstance(block.attn.normalizer, nn.Module)
    ]
    with torch.no_grad():
        model(x.to(device))
    for h in handles:
        h.remove()
    return captured


def residuals(
    scores: list[Tensor], rho: float, max_iters: int, reference_iters: int = 0
) -> dict[str, list[float]]:
    """``{"max", "p99"[, "dist"]: [...]}`` indexed by k - 2 (k = 2 .. max_iters)."""
    out: dict[str, list[float]] = {"max": [], "p99": []}
    n = scores[0].shape[-1]
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [s.to(device) for s in scores]
    mask = causal.to(device)
    ref = [
        UnbalancedSinkhorn(rho=rho, n_iters=reference_iters)(s, mask) if reference_iters else None
        for s in layers
    ]
    prev = [UnbalancedSinkhorn(rho=rho, n_iters=1)(s, mask) for s in layers]
    for k in range(2, max_iters + 1):
        cur = [UnbalancedSinkhorn(rho=rho, n_iters=k)(s, mask) for s in layers]
        per_matrix = torch.cat(
            [(c - p).abs().amax((-2, -1)).flatten() for c, p in zip(cur, prev, strict=True)]
        )
        out["max"].append(per_matrix.max().item())
        out["p99"].append(torch.quantile(per_matrix, 0.99).item())
        if reference_iters:
            out.setdefault("dist", []).append(
                max(
                    (c - r).abs().max().item()
                    for c, r in zip(cur, ref, strict=True)
                    if r is not None
                )
            )
        prev = cur
    return out


def report(run_dir: Path, table: dict[float, dict[str, list[float]]], tol: float) -> None:
    shown = [
        k
        for k in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40)
        if k - 2 < len(table[RHOS[0]]["max"])
    ]
    print(f"\n{run_dir}  (max |P_k - P_(k-1)| over layers, heads, sequences)")
    print(
        "| rho | "
        + " | ".join(f"k={k}" for k in shown)
        + f" | first k (step) < {tol:g} | first k (distance) < {tol:g} |"
    )
    print("|---|" + "---|" * (len(shown) + 2))
    needed: dict[str, int | None] = {}
    needed_dist: dict[str, int | None] = {}
    for rho, res in table.items():
        first = next((i + 2 for i, r in enumerate(res["max"]) if r < tol), None)
        needed[str(rho)] = first
        dist = res.get("dist", [])
        needed_dist[str(rho)] = next((i + 2 for i, r in enumerate(dist) if r < tol), None)
        cells = " | ".join(f"{res['max'][k - 2]:.1e}" for k in shown)
        verdict = str(first) if first is not None else f">{len(res['max']) + 1}"
        d = needed_dist[str(rho)]
        dist_verdict = "-" if not dist else (str(d) if d is not None else f">{len(dist) + 1}")
        print(f"| {rho} | {cells} | {verdict} | {dist_verdict} |")
    common = None if None in needed.values() else max(v for v in needed.values() if v is not None)
    print(f"common n_iters (max over rho): {common}")
    payload = {
        "tol": tol,
        "needed": needed,
        "needed_dist": needed_dist,
        "common_n_iters": common,
        "residuals": {str(r): v for r, v in table.items()},
    }
    (run_dir / "fixed_point.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
