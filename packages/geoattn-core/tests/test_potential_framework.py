"""Propositions of docs/methods.md: which position terms are equivalent,
which relative kernels a content clock can realize, and which encodings are guaranteed to
keep a bounded far mass."""

import math

import pytest
import torch
from torch.nn import functional as F

from geoattn_core import Softmax, UnbalancedSinkhorn
from geoattn_core.pe.grape import GrapeA
from geoattn_core.pe.path_bias import Cable, ForgetGate, GrapeAP


def _causal(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool).tril()


@pytest.mark.parametrize(
    "normalizer", [Softmax(), UnbalancedSinkhorn(0.5, n_iters=40)], ids=["softmax", "ot"]
)
def test_prop1_row_terms_do_not_change_the_coupling(normalizer) -> None:
    """b + r_t gives the same P as b, for softmax and for the prefix OT fixed point."""
    torch.manual_seed(0)
    n = 24
    scores = torch.randn(2, n, n, dtype=torch.float64)
    rows = torch.randn(2, n, 1, dtype=torch.float64) * 5
    p = normalizer(scores, _causal(n))
    torch.testing.assert_close(normalizer(scores + rows, _causal(n)), p, atol=1e-5, rtol=0)
    # control: a column term does change it
    cols = torch.randn(2, 1, n, dtype=torch.float64)
    assert (normalizer(scores + cols, _causal(n)) - p).abs().max() > 1e-2


def _geometric_clock_bias(n: int, q: float, b1: float) -> torch.Tensor:
    """b_tj = -g_t (tau_t - tau_j) with tau_t - tau_{t-1} = q^t and g_t = -b1 / q^t."""
    steps = torch.tensor([q**t for t in range(n)], dtype=torch.float64)
    tau = steps.cumsum(0)
    g = -b1 / steps
    return -g[:, None] * (tau[:, None] - tau[None, :])


@pytest.mark.parametrize("q", [1.0, 1.3, 0.8, -0.9])
def test_prop3_geometric_clocks_give_the_three_relative_kernels(q: float) -> None:
    n, b1 = 12, -0.7
    b = _geometric_clock_bias(n, q, b1)
    for d in range(n - 1):
        want = b1 * d if q == 1 else b1 * (1 - q**-d) / (1 - 1 / q)
        diag = torch.diagonal(b, offset=-d)
        torch.testing.assert_close(diag, torch.full_like(diag, want), atol=1e-9, rtol=1e-9)


def test_prop3_no_linear_clock_realizes_a_log_kernel() -> None:
    """Corollary 3.1: d = 1, 2 fix q; the log kernel then disagrees at d = 3."""
    b = [-math.log(1 + d) for d in range(4)]
    inv_q = b[2] / b[1] - 1
    predicted = b[1] * (1 + inv_q + inv_q**2)
    assert abs(predicted - b[3]) > 0.05
    # control: the same check passes on a kernel that a clock does realize
    lin = [-0.3 * d for d in range(4)]
    inv_q = lin[2] / lin[1] - 1
    assert lin[1] * (1 + inv_q + inv_q**2) == pytest.approx(lin[3])


def test_prop4_grape_a_is_two_sided_linear_without_input_bounds() -> None:
    torch.manual_seed(0)
    heads, dim, n = 4, 16, 32
    pe = GrapeA(heads, dim)
    q = torch.randn(1, heads, n, dim) * 100  # no bound on q, k
    k = torch.randn(1, heads, n, dim) * 100
    pos = torch.arange(n)
    b = pe.bias_ctx(q, k, torch.zeros(1, n, 1), pos)[0]
    rate = (F.softplus(pe.omega) * pe.slopes)[:, None, None]
    d = (pos[:, None] - pos[None, :]).clamp_min(0).float()
    lo, hi = 2 * F.softplus(torch.tensor(-1.0)), 2 * F.softplus(torch.tensor(1.0))
    lower = _causal(n)
    assert bool((b <= -lo * rate * d + 1e-5)[:, lower].all())
    assert bool((b >= -hi * rate * d - 1e-5)[:, lower].all())


def test_prop4_fox_decays_at_least_linearly_for_bounded_inputs() -> None:
    torch.manual_seed(0)
    dim, heads, n = 16, 4, 32
    pe = ForgetGate(dim, heads)
    x = F.layer_norm(torch.randn(1, n, dim), (dim,))  # bounded, as after the block's LayerNorm
    b = pe.bias_ctx(None, None, x, torch.arange(n))[0]  # type: ignore[arg-type]
    top = pe.proj(x).max(dim=1).values[0]  # the largest gate logit per head
    eps = -F.logsigmoid(top)[:, None, None]
    d = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).clamp_min(0).float()
    assert bool((b <= -eps * d + 1e-5)[:, _causal(n)].all())


def test_prop4_grape_ap_decays_at_least_linearly_for_bounded_inputs() -> None:
    torch.manual_seed(0)
    dim, heads, n, p_dim = 16, 4, 32, 8
    pe = GrapeAP(dim, heads, p_dim)
    x = F.layer_norm(torch.randn(1, n, dim), (dim,))
    b = pe.bias_ctx(None, None, x, torch.arange(n))[0]  # type: ignore[arg-type]
    p = pe.proj(x).view(1, n, heads, p_dim).transpose(1, 2)[0]  # (H, N, p_dim)
    a = p.norm(dim=-1) / math.sqrt(p_dim)  # a_t per head and query
    eps = pe.slopes[:, None] * F.softplus(-a)  # -m log sigmoid(a) = m softplus(-a)
    d = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).clamp_min(0).float()
    assert bool((b <= -eps[..., None] * d + 1e-5)[:, _causal(n)].all())


def test_prop4_cable_is_not_guaranteed_to_decay() -> None:
    """Counterexample: a clock whose ReLU is always 0 removes the position term."""
    dim, heads, n = 16, 4, 32
    pe = Cable(dim, heads)
    with torch.no_grad():
        pe.clock.weight.zero_()
        pe.clock.bias.fill_(-1.0)
    x = F.layer_norm(torch.randn(1, n, dim), (dim,))
    b = pe.bias_ctx(None, None, x, torch.arange(n))  # type: ignore[arg-type]
    assert bool((b == 0).all())


def test_lemma_40_linear_bias_keeps_far_mass_bounded_uniformly_in_length() -> None:
    torch.manual_seed(0)
    bound, eps, window = 2.0, 0.1, 64
    cap = math.exp(2 * bound) * math.exp(-eps * (window + 1)) / (1 - math.exp(-eps))
    for n in (256, 1024):
        i = torch.arange(n, dtype=torch.float64)
        bias = -eps * (i[:, None] - i[None, :]).clamp_min(0)
        scores = (torch.rand(n, n, dtype=torch.float64) * 2 - 1) * bound
        p = Softmax()(scores + bias, _causal(n))
        far = p.tril(-window - 1).sum(-1)
        assert float(far.max()) <= cap


@pytest.mark.parametrize(
    ("c", "rate"),
    [(0.5, lambda t: t**-0.5), (1.0, lambda t: 1 / math.log(t)), (2.0, lambda t: 1.0)],
)
def test_prop5_dilution_rates_of_the_power_prior(c: float, rate) -> None:
    """Near mass N_t(L) = S_L / Z_t scales like t^-(1-c), 1/log t, or a constant."""
    window = 8
    near = []
    for t in (2**12, 2**16):
        w = (1 + torch.arange(t, dtype=torch.float64)) ** -c
        near.append(float(w[:window].sum() / w.sum()))
    want = rate(2**16) / rate(2**12)
    assert near[1] / near[0] == pytest.approx(want, rel=0.1)


def test_cor51_log_n_on_a_log_bias_collapses_onto_the_diagonal() -> None:
    """Multiplying -c log(1+d) by s log t makes the diagonal take all the mass."""
    c, s = 1.0, 1 / math.log(512)
    diag = []
    for t in (512, 2**14, 2**20):
        w = (1 + torch.arange(t, dtype=torch.float64)) ** (-c * s * math.log(t))
        diag.append(float(1 / w.sum()))
    assert diag[0] < diag[1] < diag[2]
    assert diag[0] < 0.5 < diag[2]


def _near_mass(c: float, t: int, window: int) -> float:
    w = (1 + torch.arange(t, dtype=torch.float64)) ** -c
    return float(w[:window].sum() / w.sum())


@pytest.mark.parametrize("delta", [1e-3, 1e-2, 0.1, 0.5, 1.0])
def test_cor_boundary_bounds_on_the_near_mass(delta: float) -> None:
    """S_L / (1 + phi(t)) <= N_t(L) <= S_L / phi(t + 1), phi(x) = (1 - x^-delta) / delta, and
    log x (1 - delta log x / 2) <= phi(x) <= min(log x, 1 / delta)."""

    def phi(x: float) -> float:
        return -math.expm1(-delta * math.log(x)) / delta

    for t in (2, 10, 1000, 10**5):
        for window in (1, 8, 512):
            if window > t:
                continue
            s_l = float(((1 + torch.arange(window, dtype=torch.float64)) ** -(1 + delta)).sum())
            near = _near_mass(1 + delta, t, window)
            assert s_l / (1 + phi(t)) <= near <= s_l / phi(t + 1)
            if delta * math.log(t + 1) <= 1:
                assert near <= 2 * s_l / math.log(t + 1)
        log_t = math.log(t)
        assert log_t * (1 - delta * log_t / 2) <= phi(t) <= min(log_t, 1 / delta) + 1e-12
    # not vacuous: the two sides of the first pair are within 15% at t = 10^5, delta = 0.01
    if delta == 1e-2:
        assert (1 + phi(10**5)) / phi(10**5 + 1) < 1.15


def _zeta(c: float) -> float:
    # Euler-Maclaurin tail after K terms; relative error far below 1e-12 for c > 1
    k = 10**5
    head = float(((1 + torch.arange(k - 1, dtype=torch.float64)) ** -c).sum())
    return head + k ** (1 - c) / (c - 1) + k**-c / 2 - c * k ** (-c - 1) / 12


@pytest.mark.parametrize("delta", [1e-3, 1e-2, 0.1, 0.5, 1.0])
def test_cor_boundary_statement(delta: float) -> None:
    """(i) S_L / (1 + log t) <= N_t(L) <= 2 S_L / log(t + 1) if delta log(t + 1) <= 1;
    (ii) S_L / zeta(c) <= N_t(L) <= S_L / (zeta(c) (1 - t^-delta))."""
    c = 1 + delta
    zeta = _zeta(c)
    for t in (2, 10, 1000, 10**5):
        for window in (1, 8, 512):
            if window > t:
                continue
            s_l = float(((1 + torch.arange(window, dtype=torch.float64)) ** -c).sum())
            near = _near_mass(c, t, window)
            if delta * math.log(t + 1) <= 1:
                assert s_l / (1 + math.log(t)) <= near <= 2 * s_l / math.log(t + 1)
            assert s_l / zeta <= near * (1 + 1e-12)
            assert near <= s_l / (zeta * (1 - t**-delta))


def test_cor_boundary_statement_is_not_vacuous() -> None:
    """(ii) is within 1% once delta log t >> 1, and its correction factor is needed."""
    t, window, c = 10**4, 512, 1.5
    s_l = float(((1 + torch.arange(window, dtype=torch.float64)) ** -c).sum())
    near = _near_mass(c, t, window)
    assert s_l / (_zeta(c) * (1 - t**-0.5)) / near < 1.01
    # counterexample control: without the factor 1 / (1 - t^-delta) the upper bound fails
    assert near > s_l / _zeta(c)


def test_cor_boundary_exponent_just_above_one_behaves_like_one() -> None:
    """At practical lengths c = 1.001 dilutes as c = 1, while c = 1.5 sits at its limit."""
    t, window = 2**16, 512
    assert _near_mass(1.001, t, window) == pytest.approx(_near_mass(1.0, t, window), rel=0.02)
    zeta_15 = 2.612375348685488
    s_l = float(((1 + torch.arange(window, dtype=torch.float64)) ** -1.5).sum())
    assert _near_mass(1.5, t, window) == pytest.approx(s_l / zeta_15, rel=0.005)
