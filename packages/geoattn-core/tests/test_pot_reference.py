"""Numerical benchmark against POT's unbalanced Sinkhorn (gate G3 of experiment 0001).

Non-causal, unmasked, ``column_mass="full"``: the operator is the semi-relaxed
entropic OT problem

    min_P  <C, P> + eps * KL(P | a b^T) + lambda * KL(P^T 1 | b)    s.t.  P 1 = a

with C = -S. POT solves it with the scaling iteration (``fi = reg_m / (reg_m + reg)``,
and ``fi = 1`` for an infinite ``reg_m``)::

    u = (a / K v) ** 1        # hard row marginal
    v = (b / K^T u) ** rho    # rho = lambda / (lambda + eps)

Note ``v`` is the damped *absolute* column scaling. Damping a *relative* update of
the current iterate (P <- P * (b / colsum P)^rho) instead has fixed point
colsum = b, i.e. balanced Sinkhorn; the POT comparison is what tells them apart.

``reg_type="kl"`` (POT's default) uses K = exp(-C / eps) * a b^T, and
``"entropy"`` uses K = exp(-C / eps). For uniform a, b these differ by the
constant 1 / n^2; if (u, v) is a fixed point for K, then (u / c, v) is one for
cK with the same P, because the hard row constraint absorbs the constant. So both
regularizers give the same P (asserted below). POT's rows sum to 1/n; ours sum to
1, hence the factor n. Our column target n_q / n_k = 1 matches b = 1/n after scaling.
"""

import numpy as np
import ot
import pytest
import torch

from geoattn_core import UnbalancedSinkhorn


def _pot(scores: torch.Tensor, eps: float, lam: float, reg_type: str) -> torch.Tensor:
    n = scores.shape[-1]
    a = b = np.full(n, 1.0 / n)
    plan = ot.unbalanced.sinkhorn_unbalanced(
        a,
        b,
        -scores.double().numpy(),
        reg=eps,
        reg_m=(float("inf"), lam),
        reg_type=reg_type,
        numItermax=100_000,
        stopThr=1e-14,
    )
    return torch.from_numpy(n * plan)


@pytest.mark.filterwarnings("ignore:If reg_type = entropy")
@pytest.mark.parametrize("eps", [0.5, 1.0])
@pytest.mark.parametrize("lam", [0.1, 1.0, 10.0])
def test_full_column_fixed_point_matches_pot(lam: float, eps: float) -> None:
    torch.manual_seed(0)
    scores = torch.randn(6, 6)
    rho = lam / (lam + eps)
    ours = UnbalancedSinkhorn(rho=rho, n_iters=3000, eps=eps, column_mass="full")(scores[None])
    for reg_type in ("kl", "entropy"):
        torch.testing.assert_close(
            ours[0].double(), _pot(scores, eps, lam, reg_type), atol=1e-5, rtol=0
        )


def test_benchmark_distinguishes_balanced_from_unbalanced() -> None:
    """Counter-example control: the balanced fixed point must NOT pass the benchmark."""
    torch.manual_seed(0)
    scores = torch.randn(6, 6)
    balanced = UnbalancedSinkhorn(rho=1.0, n_iters=3000, column_mass="full")(scores[None])[0]
    assert (balanced.double() - _pot(scores, 1.0, 1.0, "kl")).abs().max() > 1e-2
