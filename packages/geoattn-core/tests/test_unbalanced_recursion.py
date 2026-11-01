"""The prefix unbalanced fixed point as a row recursion (docs/methods.md).

The fixed point is P_tj = K_tj M_tj^-gamma / Z_t with the inclusive prefix column mass M
(gamma = rho / (1 - rho)). Prefix consistency makes it solvable row by row: with C_j the
mass column j received from earlier rows, row t solves x_j (C_j + x_j)^gamma = lam_t K_tj,
sum_j x_j = 1 (lemma A). These tests pin the lemmas of the note and theorem 3
(t lam_t -> a*), including the observed but unproved monotone approach from above.
"""

import functools
import math

import pytest
import torch

from geoattn_core import UnbalancedSinkhorn


def _solve_row(
    c: torch.Tensor, k: torch.Tensor, gamma: float, log_lam: float
) -> tuple[torch.Tensor, float]:
    """x_j (c_j + x_j)^gamma = lam k_j with sum x = 1: Newton on log lam outside, Newton on
    log x inside (both monotone)."""
    lx = torch.full_like(c, -math.log(len(c)))
    x = lx.exp()
    for _ in range(100):
        for _ in range(40):
            f = lx + gamma * torch.log(c + x) - log_lam - k.log()
            lx = lx - f / (1 + gamma * x / (c + x))
            x = lx.exp()
        s = float(x.sum())
        if abs(s - 1) < 1e-14:
            break
        log_lam -= (s - 1) / float((x / (1 + gamma * x / (c + x))).sum())
    return x, log_lam


def _recursion(kernel: torch.Tensor, rho: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed point P and the multipliers lam_t, row by row (lemma A)."""
    n, gamma = kernel.shape[0], rho / (1 - rho)
    p = torch.zeros(n, n, dtype=torch.float64)
    lam = torch.zeros(n, dtype=torch.float64)
    c = torch.zeros(n, dtype=torch.float64)
    log_lam = 0.0
    for t in range(n):
        x, log_lam = _solve_row(c[: t + 1], kernel[t, : t + 1], gamma, log_lam)
        p[t, : t + 1], lam[t] = x, math.exp(log_lam)
        c[: t + 1] += x
    return p, lam


def _malthusian(rho: float) -> float:
    """a* = 1 / mu with k_hat(mu) = 1 for k(tau) = ((1 + gamma) tau)^-rho."""
    gamma = rho / (1 - rho)
    return ((1 + gamma) ** rho / math.gamma(1 - rho)) ** (1 / (1 - rho))


@functools.cache
def _flat(rho: float, n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    return _recursion(torch.ones(n, n, dtype=torch.float64), rho)


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("flat", [True, False])
def test_lemma_a_row_recursion_is_the_operator_fixed_point(rho: float, flat: bool) -> None:
    n = 96
    torch.manual_seed(0)
    scores = torch.zeros(n, n) if flat else torch.randn(n, n)
    scores = scores.double()
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    p, _ = _recursion(scores.exp(), rho)
    ref = UnbalancedSinkhorn(rho, n_iters=400)(scores, causal)
    # the operator works in float32 by design (ADR-0002); the recursion is float64
    torch.testing.assert_close(p, ref.masked_fill(~causal, 0.0).double(), atol=1e-6, rtol=0)


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
def test_lemma_b_sandwich_holds_for_every_entry(rho: float) -> None:
    """(1+g) sum_i lam_i (C_{i-1}/C_i)^g + diagonal correction <= C_t(j)^(1+g)
    <= (1+g) (Lam_t - Lam_{j-1}), and the diagonal C_t(t)^(1+g) = lam_t exactly."""
    gamma = rho / (1 - rho)
    p, lam = _flat(rho)
    cum = p.cumsum(0)  # cum[t, j] = C_t(j)
    big_lam = torch.cat([torch.zeros(1, dtype=torch.float64), lam.cumsum(0)])
    n = len(lam)
    t = torch.arange(n)[:, None]
    j = torch.arange(n)[None, :]
    lower_t = t >= j
    upper = (1 + gamma) * (big_lam[t + 1] - big_lam[j])
    assert bool((cum.pow(1 + gamma) <= upper * (1 + 1e-10))[lower_t].all())
    prev = torch.cat([torch.zeros(1, n, dtype=torch.float64), cum[:-1]])
    ratio = torch.where(cum > 0, prev / cum, torch.zeros_like(cum))
    increments = (1 + gamma) * lam[:, None] * ratio.pow(gamma)
    increments.diagonal().copy_(lam)  # first row of each column: x^(1+g) = lam exactly
    lower = increments.tril().cumsum(0)
    assert bool((cum.pow(1 + gamma) >= lower * (1 - 1e-10))[lower_t].all())
    torch.testing.assert_close(cum.diagonal().pow(1 + gamma), lam, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
def test_lemma_d1_t_lambda_is_at_most_one(rho: float) -> None:
    """Jensen: the t columns hold total mass t, so sum_j C_j^-gamma >= t and lam_t <= 1/t."""
    _, lam = _flat(rho)
    t = torch.arange(1, len(lam) + 1, dtype=torch.float64)
    assert bool((t * lam <= 1 + 1e-12).all())


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
def test_t_lambda_decreases_towards_the_malthusian_constant(rho: float) -> None:
    _, lam = _flat(rho)
    a = _malthusian(rho)
    tl = [(t + 1) * float(lam[t]) for t in (63, 127, 255)]
    assert tl[0] > tl[1] > tl[2] > a
    assert tl[2] - a < 0.01


@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
def test_t_lambda_approaches_the_malthusian_constant(rho: float) -> None:
    """Theorem 3: t lam_t -> a* (the monotone approach from above is observed, not proved)."""
    _, lam = _flat(rho, 1024)
    a = _malthusian(rho)
    tl = [(t + 1) * float(lam[t]) for t in (127, 511, 1023)]
    assert tl[0] > tl[1] > tl[2] > a
    assert tl[2] - a < 0.005
    assert (tl[2] - a) < (tl[0] - a) / 3


def test_malthusian_constant_at_one_half_is_two_over_pi() -> None:
    assert _malthusian(0.5) == pytest.approx(2 / math.pi, rel=1e-12)


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("bound", [0.0, 1.0, 3.0])
def test_theorem_1_no_entry_exceeds_the_dilution_bound(rho: float, bound: float) -> None:
    """|S| <= B: lam_t <= e^B / t (Jensen), so every entry is at most
    (e^(2B) / t)^(1 - rho); any fixed window loses its mass as t grows."""
    n = 128
    torch.manual_seed(1)
    scores = (torch.rand(n, n, dtype=torch.float64) * 2 - 1) * bound
    p, lam = _recursion(scores.exp(), rho)
    t = torch.arange(1, n + 1, dtype=torch.float64)
    assert bool((t * lam <= math.exp(bound) * (1 + 1e-12)).all())
    cap = (math.exp(2 * bound) / t) ** (1 - rho)
    assert bool((p.max(dim=1).values <= cap * (1 + 1e-12)).all())


def _alpha0(rho: float) -> float:
    """Lower constant of lemma D2: alpha0^(1 - rho) = 1 / (2 Gamma(1 - rho))."""
    return (2 * math.gamma(1 - rho)) ** (-1 / (1 - rho))


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
def test_lemma_b2_two_sided_sandwich_with_constant_one_below(rho: float) -> None:
    """Column mass only grows, so x_ij >= lam_i M_t(j)^-gamma and summing over the rows,
    Lam_t - Lam_{j-1} <= M_t(j)^(1+gamma) <= (1+gamma) (Lam_t - Lam_{j-1})."""
    gamma = rho / (1 - rho)
    p, lam = _flat(rho)
    n = len(lam)
    cum = p.cumsum(0).pow(1 + gamma)
    big_lam = torch.cat([torch.zeros(1, dtype=torch.float64), lam.cumsum(0)])
    t, j = torch.arange(n)[:, None], torch.arange(n)[None, :]
    span = big_lam[t + 1] - big_lam[j]
    lower_t = t >= j
    assert bool((cum >= span * (1 - 1e-10))[lower_t].all())
    assert bool((cum <= (1 + gamma) * span * (1 + 1e-10))[lower_t].all())


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75, 0.9])
def test_lemma_d2_t_lambda_is_bounded_below(rho: float) -> None:
    _, lam = _flat(rho)
    t = torch.arange(1, len(lam) + 1, dtype=torch.float64)
    assert bool((t * lam >= _alpha0(rho)).all())


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
def test_lemma_d3_increment_ratio_decays_with_distance(rho: float) -> None:
    """r_i = P_ij / M_i(j) <= lam_i / (Lam_i - Lam_{j-1}) <= 1 / (alpha0 (i - j + 1))."""
    p, lam = _flat(rho)
    n = len(lam)
    cum = p.cumsum(0)
    big_lam = torch.cat([torch.zeros(1, dtype=torch.float64), lam.cumsum(0)])
    i, j = torch.arange(n)[:, None], torch.arange(n)[None, :]
    lower_t = i >= j
    r = torch.where(lower_t, p / cum.clamp_min(1e-300), torch.zeros_like(p))
    middle = lam[:, None] / (big_lam[i + 1] - big_lam[j]).clamp_min(1e-300)
    assert bool((r <= middle * (1 + 1e-10))[lower_t].all())
    assert bool((r <= 1 / (_alpha0(rho) * (i - j + 1)) * (1 + 1e-10))[lower_t].all())


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75, 0.9])
def test_linear_bias_ot_fixed_point_has_a_nonvacuous_far_mass_bound(rho: float) -> None:
    """Use the row solver, not a finite iterate; removing decay must violate the bound."""
    n, window, bound, slope = 48, 8, 0.2, 4.0
    generator = torch.Generator().manual_seed(17)
    content = bound * (2 * torch.rand(n, n, generator=generator, dtype=torch.float64) - 1)
    distance = torch.arange(n)[:, None] - torch.arange(n)[None, :]
    p, _ = _recursion((content - slope * distance.clamp_min(0)).exp(), rho)
    far = (p * (distance > window)).sum(1)
    rate = (1 - rho) * slope
    upper = math.exp(2 * (1 - rho) * bound - rate * (window + 1)) / -math.expm1(-rate)
    assert upper < 0.1
    assert bool((far <= upper * (1 + 1e-10)).all())
    no_decay, _ = _recursion(content.exp(), rho)
    assert float((no_decay[-1] * (distance[-1] > window)).sum()) > upper
