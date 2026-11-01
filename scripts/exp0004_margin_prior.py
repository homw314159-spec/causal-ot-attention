"""Experiment 0004: prior-only far mass of a log-distance bias with exponent c = 1 + delta.

    uv run python scripts/exp0004_margin_prior.py

Training pushes constrained exponents to their bound (experiment 0003: ~40% of the ``gt1``
heads at c ~ 1.001), so the worst case for a margin delta_0 is every head at c = 1 + delta_0.
Far mass at length n is the mean over queries t in [n/2, n) of the prior mass on keys at
distance > L, with w(d) = (1 + d)^(-c) and no content, as in ``probe_summability.py``.
The printed table fixes the margins of part B before any run (preregistration section 1).
"""

import numpy as np

L = 512  # training length of the 25M screening configuration


def far_mass(c: float, n: int) -> float:
    w = (1.0 + np.arange(n)) ** (-c)
    cum = np.cumsum(w)
    t = np.arange(n // 2, n)
    total = cum[t]
    near = cum[np.minimum(t, L)]
    return float(np.mean((total - near) / total))


def main() -> None:
    for delta in [0.0, 0.1, 0.25, 0.3, 0.5, 1.0]:
        f = {k: far_mass(1 + delta, k * L) for k in (4, 16, 32)}
        print(
            f"delta={delta:<5} far4={f[4]:.3f} far16={f[16]:.3f} far32={f[32]:.3f} "
            f"growth4-16={f[16] - f[4]:.3f}"
        )


if __name__ == "__main__":
    main()
