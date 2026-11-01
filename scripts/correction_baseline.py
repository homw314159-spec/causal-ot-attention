"""Distance decay of the OT correction: trained models against the constant-score fixed point.

    uv run python scripts/correction_baseline.py
    uv run python scripts/correction_baseline.py --input runs/column_bias/screen_bd8a861.json

``probe_column_bias.py`` records, for each trained model and layer, the mean of the
correction B = -gamma log M_tj at every distance d = t - j, averaged over batches, heads and
rows. This script computes the same statistic for the constant-score fixed point at the
same length (no content), which is the theory's prediction, and prints both side by side:
the drop of the mean correction from d = 1 to d = L/2 and its local exponent, the slope of
-B against log d on 8 <= d <= 64. Section 7 of the paper states the comparison
qualitatively; no training is needed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from geoattn_core import UnbalancedSinkhorn

FIT = np.arange(8, 65)


def fixed_point_curve(rho: float, n: int, n_iters: int = 60) -> np.ndarray:
    """Mean of -gamma log M_{t,t-d} over the rows t > d of the constant-score fixed point."""
    gamma = rho / (1 - rho)
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    p = UnbalancedSinkhorn(rho, n_iters=n_iters)(torch.zeros(n, n, dtype=torch.float64), causal)
    b = -gamma * p.cumsum(dim=0).clamp_min(1e-300).log()
    return np.array([torch.diagonal(b, offset=-d).mean().item() for d in range(n)])


def local_exponent(curve: np.ndarray) -> float:
    return float(-np.polyfit(np.log(FIT), curve[FIT], 1)[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("runs/column_bias/screen_bd8a861.json"))
    p.add_argument("--length", type=int, default=512, help="training length L of the models")
    args = p.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    n, half = args.length, args.length // 2
    by_rho: dict[float, list[np.ndarray]] = {}
    for run in data["runs"].values():
        layers = run[str(n)]
        by_rho.setdefault(run["rho"], []).extend(np.array(x["mean_by_d"]) for x in layers)
    print(f"| rho | drop d=1..{half}: trained (layers) | fixed point "
          f"| exponent 8..64: trained | fixed point |")  # fmt: skip
    print("|---|---|---|---|---|")
    for rho in sorted(by_rho):
        curves = by_rho[rho]
        drops = [c[1] - c[half] for c in curves]
        ref = fixed_point_curve(rho, n)
        trained = np.mean(curves, axis=0)
        print(f"| {rho} | {statistics.fmean(drops):.2f} ({min(drops):.2f}-{max(drops):.2f}) "
              f"| {ref[1] - ref[half]:.2f} | {local_exponent(trained):.3f} "
              f"| {local_exponent(ref):.3f} |")  # fmt: skip


if __name__ == "__main__":
    main()
