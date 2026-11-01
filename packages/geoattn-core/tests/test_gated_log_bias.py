"""GatedLogDistanceBias (experiment 0002, P6): a per-query exponent on the log-distance bias."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core import GeometricAttention, RoPE, Softmax
from geoattn_core.pe import (
    ClockLogDistanceBias,
    ComposedPE,
    GatedLogDistanceBias,
    LogDistanceBias,
)


def _bias(pe: GatedLogDistanceBias, x: torch.Tensor, positions=None) -> torch.Tensor:
    pos = torch.arange(x.shape[1]) if positions is None else positions
    return pe.bias_ctx(None, None, x, pos)  # type: ignore[arg-type]  # q, k unused


def test_zero_gate_is_the_ungated_bias() -> None:
    pe = GatedLogDistanceBias(16, 4, c_min=0.2, c_max=0.8)
    with torch.no_grad():
        pe.gate.weight.zero_()
    x = torch.randn(2, 7, 16)
    expected = LogDistanceBias(4, c_min=0.2, c_max=0.8).bias(torch.arange(7), torch.arange(7))
    torch.testing.assert_close(_bias(pe, x), expected.expand(2, -1, -1, -1))


def test_exponent_follows_the_query_content() -> None:
    """Control for the test above: with a gate, the same positions decay at different
    rates for different inputs, and every exponent stays positive."""
    torch.manual_seed(0)
    pe = GatedLogDistanceBias(16, 4)
    c = pe.coef(torch.randn(3, 9, 16) * 5)
    assert (c > 0).all() and c.std(dim=-1).min() > 1e-3


def _row_leak(make_bias, t: int = 4) -> float:
    """Max change of row t when every token except t changes."""
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 8, 16, generator=g)
    y = torch.randn(1, 8, 16, generator=g)
    y[:, t] = x[:, t]
    return (make_bias(x)[..., t, :] - make_bias(y)[..., t, :]).abs().max().item()


def test_row_reads_only_its_own_query() -> None:
    """Row t must depend on x_t alone (causal, and no attention-dependent leak); a gate
    read from the key (an exponent per column) is the counter-example the check catches."""
    torch.manual_seed(2)
    pe = GatedLogDistanceBias(16, 2)
    assert _row_leak(lambda x: _bias(pe, x)) == 0

    def key_gated(x: torch.Tensor) -> torch.Tensor:
        log_d = torch.log1p((torch.arange(8)[:, None] - torch.arange(8)[None, :]).abs())
        return -pe.coef(x)[..., None, :] * log_d

    assert _row_leak(key_gated) > 1e-3


@settings(max_examples=25, deadline=None)
@given(shift=st.integers(-5000, 5000), seed=st.integers(0, 999))
def test_bias_is_relative(shift: int, seed: int) -> None:
    torch.manual_seed(seed)
    pe = GatedLogDistanceBias(16, 2)
    x = torch.randn(1, 6, 16)
    pos = torch.arange(6) * 2
    torch.testing.assert_close(_bias(pe, x, pos + shift), _bias(pe, x, pos))


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 999))
def test_bias_decays_with_distance(seed: int) -> None:
    torch.manual_seed(seed)
    b = _bias(GatedLogDistanceBias(16, 2), torch.randn(1, 10, 16) * 3)
    assert b.diagonal(dim1=-2, dim2=-1).abs().max() == 0
    step = b[..., :, 1:] - b[..., :, :-1]  # b(t, j + 1) - b(t, j): nearer keys score higher
    causal = torch.ones(10, 10, dtype=torch.bool).tril()
    assert (step[..., causal[:, 1:]] >= -1e-6).all()


def test_per_batch_positions() -> None:
    pe = GatedLogDistanceBias(16, 2)
    x = torch.randn(3, 5, 16)
    shared = _bias(pe, x)
    torch.testing.assert_close(_bias(pe, x, torch.arange(5).expand(3, 5)), shared)


def test_gradcheck() -> None:
    torch.manual_seed(3)
    pe = GatedLogDistanceBias(8, 2).double()
    x = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: _bias(pe, x), (x,))


def test_attention_with_mask_and_fully_masked_row() -> None:
    torch.manual_seed(4)
    pe = ComposedPE(RoPE(8), GatedLogDistanceBias(16, 2))
    attn = GeometricAttention(dim=16, num_heads=2, pe=pe, normalizer=Softmax())
    mask = torch.ones(6, 6, dtype=torch.bool).tril()
    mask[2] = False
    out = attn(torch.randn(2, 6, 16), mask=mask)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- LC: on a content clock


def test_still_clock_is_the_gated_bias() -> None:
    """P1 within P6 within LC: with v = 0 the clock ticks once per token."""
    torch.manual_seed(5)
    lc = ClockLogDistanceBias(16, 4)
    p6 = GatedLogDistanceBias(16, 4)
    p6.load_state_dict({k: v for k, v in lc.state_dict().items() if not k.startswith("clock")})
    with torch.no_grad():
        lc.clock.weight.zero_()
    x = torch.randn(2, 9, 16)
    torch.testing.assert_close(_bias(lc, x).tril(), _bias(p6, x).tril())


def test_clock_speed_follows_content() -> None:
    """Control for the test above: with a clock, the same query and positions see
    different decays when the tokens in between change."""
    torch.manual_seed(6)
    lc = ClockLogDistanceBias(16, 2)
    with torch.no_grad():
        lc.clock.weight.mul_(50)
    x = torch.randn(1, 8, 16)
    y = x.clone()
    y[:, 2:6] = torch.randn(1, 4, 16)
    assert (_bias(lc, x)[..., 7, 0] - _bias(lc, y)[..., 7, 0]).abs().max() > 1e-3


def test_clock_row_reads_only_the_past() -> None:
    """Row t may depend on x_1..x_t, never later; a clock normalized by its total over
    the whole sequence (the counter-example) is caught."""
    torch.manual_seed(7)
    lc = ClockLogDistanceBias(16, 2)
    with torch.no_grad():
        lc.clock.weight.mul_(50)
    g = torch.Generator().manual_seed(8)
    x = torch.randn(1, 8, 16, generator=g)
    y = x.clone()
    y[:, 5:] = torch.randn(1, 3, 16, generator=g)
    assert (_bias(lc, x)[..., :5, :] - _bias(lc, y)[..., :5, :]).abs().max() == 0

    def normalized_clock(x: torch.Tensor) -> torch.Tensor:
        tick = 2 * torch.sigmoid(lc.clock(x)).transpose(1, 2)
        s = tick.cumsum(-1) / tick.sum(-1, keepdim=True) * x.shape[1]  # total sees the future
        dist = (s[..., :, None] - s[..., None, :]).clamp_min(0)
        return -lc.coef(x)[..., None] * torch.log1p(dist)

    leak = (normalized_clock(x)[..., :5, :] - normalized_clock(y)[..., :5, :]).abs().max()
    assert leak > 1e-3


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 999))
def test_clock_bias_decays_with_distance(seed: int) -> None:
    torch.manual_seed(seed)
    lc = ClockLogDistanceBias(16, 2)
    with torch.no_grad():
        lc.clock.weight.mul_(20)
    b = _bias(lc, torch.randn(1, 10, 16) * 3)
    assert b.diagonal(dim1=-2, dim2=-1).abs().max() == 0
    step = b[..., :, 1:] - b[..., :, :-1]
    causal = torch.ones(10, 10, dtype=torch.bool).tril()
    assert (step[..., causal[:, 1:]] >= -1e-6).all()


def test_clock_gradcheck() -> None:
    torch.manual_seed(9)
    lc = ClockLogDistanceBias(8, 2).double()
    x = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
    lower = torch.ones(5, 5, dtype=torch.bool).tril()
    # the clamp at 0 only touches masked entries; check the kept ones
    assert torch.autograd.gradcheck(lambda x: _bias(lc, x)[..., lower], (x,))
