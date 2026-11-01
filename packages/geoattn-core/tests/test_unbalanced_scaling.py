"""Scaling limit of the prefix unbalanced fixed point on flat scores (plan.md §3.5).

With equal scores the fixed point is self-similar: t * P[t, s t] -> f(s), where
f(s) = (log 1/s)^-rho / Gamma(1 - rho). Near the diagonal f(s) ~ (1 - s)^-rho, i.e. a
power law in the distance with exponent rho, and the mass within a fixed distance L
decays like (L / t)^(1 - rho) / Gamma(2 - rho): the implied prior is not summable for
any rho < 1, so attention spreads ever further as the context grows.
"""

import math

import pytest
import torch

from geoattn_core import Softmax, UnbalancedSinkhorn

N_ITERS = 60  # converged: 30 and 60 iterations agree to 1e-4 at these sizes


def _fixed_point(n: int, rho: float) -> torch.Tensor:
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    flat = torch.zeros(n, n, dtype=torch.float64)
    return UnbalancedSinkhorn(rho, n_iters=N_ITERS)(flat, causal)


def _limit(s: float, rho: float) -> float:
    return math.log(1 / s) ** -rho / math.gamma(1 - rho)


def _near_mass(p: torch.Tensor, window: int) -> float:
    return float(p[-1, -window:].sum())


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75])
def test_last_row_follows_the_scaling_limit(rho: float) -> None:
    n = 1024
    p = _fixed_point(n, rho)
    for s in (0.05, 0.3, 0.7):
        got = n * float(p[-1, int(s * n)])
        assert got == pytest.approx(_limit(s, rho), rel=0.02), s


def test_the_error_shrinks_with_length() -> None:
    """A limit, not a coincidence at one size."""
    rho, s = 0.75, 0.7
    errors = []
    for n in (256, 1024):
        got = n * float(_fixed_point(n, rho)[-1, int(s * n)])
        errors.append(abs(got / _limit(s, rho) - 1))
    assert errors[1] < errors[0] / 2


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
def test_near_mass_decays_like_a_non_summable_prior(rho: float) -> None:
    """Mass within distance L shrinks by 2^-(1 - rho) per doubling of the length."""
    window = 64
    near = [_near_mass(_fixed_point(n, rho), window) for n in (512, 1024)]
    assert near[1] / near[0] == pytest.approx(2 ** -(1 - rho), rel=0.02)
    want = (window / 1024) ** (1 - rho) / math.gamma(2 - rho)
    assert near[1] == pytest.approx(want, rel=0.05)


def test_control_a_summable_prior_keeps_its_near_mass() -> None:
    """Control: a linear distance bias (ALiBi-like, exponential decay) is summable, so
    the near mass stays put as the length doubles; the check above would fail on it."""
    window, slope = 64, 0.05
    near = []
    for n in (512, 1024):
        i = torch.arange(n, dtype=torch.float64)
        bias = -slope * (i[:, None] - i[None, :]).clamp(min=0)
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        near.append(_near_mass(Softmax()(bias, causal), window))
    assert near[1] / near[0] == pytest.approx(1.0, abs=1e-3)
    assert abs(near[1] / near[0] - 2 ** -(1 - 0.5)) > 0.2


def _a_star(rho: float) -> float:
    gamma = rho / (1 - rho)
    return ((1 + gamma) ** rho / math.gamma(1 - rho)) ** (1 / (1 - rho))


def _z_profile(rho: float, depth: int) -> list[float]:
    """z_d^gamma (z_d - z_{d-1}) = 1, z_{-1} = 0: a column's mass after d + 1 rows with lam = 1."""
    gamma, z, prev = rho / (1 - rho), [], 0.0
    for _ in range(depth + 1):
        lo, hi = prev, prev + 2.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mid**gamma * (mid - prev) < 1 else (lo, mid)
        z.append(lo)
        prev = lo
    return z


@pytest.mark.parametrize("rho", [0.5, 0.75])
def test_near_diagonal_follows_the_discrete_profile(rho: float) -> None:
    """Theorem 4: t^(1 - rho) P[t, t - d] -> a*^(1 - rho) (z_d - z_{d-1}) for fixed d, and
    the profile turns into the power law t^(rho-1) d^-rho / Gamma(1 - rho) for large d."""
    n = 1024
    p = _fixed_point(n, rho)
    z = _z_profile(rho, 64)
    for d in (0, 1, 2, 8):
        want = _a_star(rho) ** (1 - rho) * (z[d] - (z[d - 1] if d else 0.0))
        assert n ** (1 - rho) * float(p[-1, -1 - d]) == pytest.approx(want, rel=0.01), d
    step = z[64] - z[63]
    assert _a_star(rho) ** (1 - rho) * step == pytest.approx(
        64**-rho / math.gamma(1 - rho), rel=0.03
    )


def _zeta(s: float, n: int = 10_000) -> float:
    """Riemann zeta on (0, 1) by Euler-Maclaurin, error O(n^-(s + 3))."""
    partial = math.fsum(m**-s for m in range(1, n + 1))
    return partial - n ** (1 - s) / (1 - s) - n**-s / 2 + s * n ** (-s - 1) / 12


def test_zeta_matches_a_known_value_and_has_converged() -> None:
    assert _zeta(0.5) == pytest.approx(-1.4603545088, abs=1e-8)  # zeta(1/2)
    for s in (0.75, 0.9):
        assert _zeta(s) == pytest.approx(_zeta(s, n=20_000), abs=1e-9)


@pytest.mark.parametrize("rho", [0.5, 0.75, 0.9])
def test_two_iterations_already_have_the_limit_shape(rho: float) -> None:
    """Theorem 5: every n_iters >= 2 has the same limit f. With two iterations the shape is
    exact at every length and only the amplitude is off, by the first-order expansion
    1 / (1 + zeta(rho) n^-(1 - rho) / Gamma(1 - rho)) - 1 of the row normalizer (proof
    notes, docs/methods.md §7): an error of order n^-(1 - rho)."""
    n = 1024
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    p = UnbalancedSinkhorn(rho, n_iters=2)(torch.zeros(n, n, dtype=torch.float64), causal)
    r = [n * float(p[-1, int(s * n)]) / _limit(s, rho) for s in (0.05, 0.3, 0.7)]
    assert max(r) - min(r) < 0.01 * max(r)
    expansion = 1 / (1 + _zeta(rho) / math.gamma(1 - rho) * n ** -(1 - rho)) - 1
    assert r[1] - 1 == pytest.approx(expansion, abs=1e-3)


def test_control_the_expansion_rejects_a_wrong_rate() -> None:
    """Counter-example: the same constant with the rate n^-(1 - rho) / 2 misses the
    computed amplitude by far more than the tolerance above."""
    rho, n = 0.75, 1024
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    p = UnbalancedSinkhorn(rho, n_iters=2)(torch.zeros(n, n, dtype=torch.float64), causal)
    amplitude = n * float(p[-1, int(0.3 * n)]) / _limit(0.3, rho) - 1
    wrong = 1 / (1 + _zeta(rho) / math.gamma(1 - rho) * n ** (-(1 - rho) / 2)) - 1
    assert abs(amplitude - wrong) > 0.1


@pytest.mark.parametrize("rho", [0.5, 0.75])
def test_iid_scores_factor_into_prior_times_content(rho: float) -> None:
    """Proposition 6: with i.i.d. bounded scores, t P_tj ~ (K_tj / E K) f(s): the column
    balancing averages the content out and leaves the universal position prior."""
    n, bound = 1024, 1.0
    torch.manual_seed(0)
    scores = (torch.rand(n, n, dtype=torch.float64) * 2 - 1) * bound
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    p = UnbalancedSinkhorn(rho, n_iters=N_ITERS)(scores, causal).double()
    mean_k = math.sinh(bound) / bound
    for lo, hi in ((0.05, 0.15), (0.3, 0.4), (0.6, 0.7)):
        ratios = []
        for t in range(n - 32, n):
            js = torch.arange(int(lo * (t + 1)), int(hi * (t + 1)))
            s = ((js + 1) / (t + 1)).tolist()
            prior = torch.tensor([_limit(x, rho) for x in s], dtype=torch.float64)
            ratios.append(p[t, js] * (t + 1) * mean_k / (scores[t, js].exp() * prior))
        assert float(torch.cat(ratios).mean()) == pytest.approx(1.0, abs=0.02), lo
