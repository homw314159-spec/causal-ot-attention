"""Rate at which finitely many iterations approach the limit shape (paper, Appendix D).

    uv run python scripts/finite_iteration_rate.py --out runs/numerics/finite_iteration_rate.json

Constant scores, ``UnbalancedSinkhorn(rho, n_iters=k)`` in float64 on the full causal
matrix. For every n the ratio r(s) = n P[n-1, floor(s n)] / f(s) on the last row, with
f(s) = (log 1/s)^-rho / Gamma(1 - rho) (Theorem ``thm:iters``: r -> 1 for every k >= 2).
Reported per (rho, k, n): the amplitude error r(0.3) - 1, the tilt max_s r - min_s r over
s in {0.05, 0.3, 0.7}, and between successive lengths the local exponent
-log(err(n1) / err(n0)) / log(n1 / n0) of each, to compare with 1 - rho. For k = 2 also
the first-order expansion r ~ 1 / (1 + zeta(rho) n^-(1 - rho) / Gamma(1 - rho)) of the
row normalizer (docs/methods.md §7; derived, not proved).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import torch

from geoattn_core import UnbalancedSinkhorn

S = (0.05, 0.3, 0.7)


def ratios(n: int, rho: float, k: int) -> list[float]:
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    p = UnbalancedSinkhorn(rho, n_iters=k)(torch.zeros(n, n, dtype=torch.float64), causal)
    f = [math.log(1 / s) ** -rho / math.gamma(1 - rho) for s in S]
    return [n * float(p[-1, int(s * n)]) / fs for s, fs in zip(S, f, strict=True)]


def zeta(s: float, n: int = 10_000) -> float:
    """Riemann zeta for 0 < s < 1 by Euler-Maclaurin: the partial sum minus its growth,
    error O(n^-(s + 3))."""
    partial = math.fsum(m**-s for m in range(1, n + 1))
    return partial - n ** (1 - s) / (1 - s) - n**-s / 2 + s * n ** (-s - 1) / 12


def expansion(n: int, rho: float) -> float:
    """First-order amplitude error of two iterations, r(s) - 1."""
    return 1 / (1 + zeta(rho) / math.gamma(1 - rho) * n ** -(1 - rho)) - 1


def _exponent(n0: int, e0: float, n1: int, e1: float) -> float | None:
    """Local decay exponent of |error| from n0 to n1; None when the error changes sign."""
    if e0 * e1 <= 0:
        return None
    return -math.log(e1 / e0) / math.log(n1 / n0)


def _fmt(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    p.add_argument("--iters", type=int, nargs="+", default=[2, 3])
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    args = p.parse_args()
    out: dict[str, dict] = {}
    for rho in args.rhos:
        for k in args.iters:
            rows = []
            for n in args.lengths:
                r = ratios(n, rho, k)
                rows.append({"n": n, "ratios": r, "amplitude": r[1] - 1, "tilt": max(r) - min(r)})
                if k == 2:
                    rows[-1]["expansion"] = expansion(n, rho)
            for a, b in itertools.pairwise(rows):
                for key in ("amplitude", "tilt"):
                    b[f"{key}_exponent"] = _exponent(a["n"], a[key], b["n"], b[key])
            out[f"rho={rho},k={k}"] = {"rho": rho, "k": k, "rows": rows}
            for row in rows:
                print(
                    f"rho={rho} k={k} n={row['n']:>6} amplitude={row['amplitude']:+.4f} "
                    f"({_fmt(row.get('amplitude_exponent'))}) tilt={row['tilt']:.4f} "
                    f"({_fmt(row.get('tilt_exponent'))}) expansion={_fmt(row.get('expansion'))}",
                    flush=True,
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
