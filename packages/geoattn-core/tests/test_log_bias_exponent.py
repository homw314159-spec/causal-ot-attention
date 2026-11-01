"""Exponent constraints of LogDistanceBias (experiment 0003, gate G1).

``gt1`` must keep every exponent above 1 (summable tail) and ``lt1`` below 1 at every
parameter value, hence at every training step; ``free`` must be the P1 of experiment 0002
unchanged, so its checkpoints still load and give the same bias.
"""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core.pe.log_bias import EXPONENT_RANGES, LogDistanceBias

H = 6


def _p1_coef(raw: torch.Tensor) -> torch.Tensor:
    """Experiment 0002's P1: softplus of the parameter, no constraint."""
    return torch.nn.functional.softplus(raw)


def test_free_is_p1_elementwise() -> None:
    pe = LogDistanceBias(H)
    init = torch.logspace(-1, 0, H)
    torch.testing.assert_close(pe.raw.detach(), init + torch.log(-torch.expm1(-init)))
    with torch.no_grad():
        pe.raw.normal_()
    assert torch.equal(pe.coef, _p1_coef(pe.raw))
    pos = torch.arange(9)
    expected = -_p1_coef(pe.raw)[:, None, None] * torch.log1p((pos[:, None] - pos).abs().float())
    assert torch.equal(pe.bias(pos, pos), expected)
    assert list(pe.state_dict()) == ["raw"]  # experiment 0002 checkpoints load as before


@pytest.mark.parametrize("exponent", ["gt1", "lt1"])
def test_init_is_log_spaced_over_the_range(exponent: str) -> None:
    lo, hi = EXPONENT_RANGES[exponent]
    pe = LogDistanceBias(H, exponent=exponent)
    torch.testing.assert_close(pe.coef, torch.logspace(math.log10(lo), math.log10(hi), H))


@settings(max_examples=60, deadline=None)
@given(raw=st.lists(st.floats(-10, 10), min_size=H, max_size=H))
def test_constraint_holds_for_any_parameter(raw: list[float]) -> None:
    for exponent, inside in (("gt1", lambda c: c > 1), ("lt1", lambda c: (c > 0) & (c < 1))):
        pe = LogDistanceBias(H, exponent=exponent)
        with torch.no_grad():
            pe.raw.copy_(torch.tensor(raw))
        assert inside(pe.coef).all(), (exponent, pe.coef)


@pytest.mark.parametrize(("exponent", "sign"), [("gt1", 1.0), ("lt1", -1.0)])
def test_constraint_holds_during_training(exponent: str, sign: float) -> None:
    """AdamW steps that push every exponent towards the forbidden side: the constraint
    holds after each one. Strictly only while |raw| stays below about 15: beyond that
    ``1 + softplus`` and ``sigmoid`` round to exactly 1 in float32. AdamW moves the
    parameter by about lr per step, up to 8 here; a B1 run (lr 3e-3, cosine, ~15k steps)
    pushing one way throughout could move it by ~25 and reach c = 1 in float32, so the
    report lists the trained exponents."""
    pe = LogDistanceBias(H, exponent=exponent)
    opt = torch.optim.AdamW(pe.parameters(), lr=0.04, weight_decay=0.0)
    pos = torch.arange(32)
    for _ in range(200):
        opt.zero_grad()
        (-sign * pe.bias(pos, pos).sum()).backward()  # gt1: lowers c; lt1: raises c
        opt.step()
        c = pe.coef
        assert ((c > 1) if exponent == "gt1" else (c > 0) & (c < 1)).all(), c
    moved = pe.coef - LogDistanceBias(H, exponent=exponent).coef
    assert (sign * moved < 0).all()  # the steps did push towards the boundary


@pytest.mark.parametrize("exponent", ["free", "gt1", "lt1"])
def test_gradcheck(exponent: str) -> None:
    wrapper = _Bias(LogDistanceBias(4, exponent=exponent).double())
    raw = wrapper.pe.raw.detach().clone().requires_grad_()
    pos = torch.arange(7)
    assert torch.autograd.gradcheck(
        lambda r: torch.func.functional_call(wrapper, {"pe.raw": r}, (pos,)), (raw,)
    )


class _Bias(torch.nn.Module):
    def __init__(self, pe: LogDistanceBias) -> None:
        super().__init__()
        self.pe = pe

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return self.pe.bias(pos, pos)


@pytest.mark.parametrize(
    ("exponent", "c_min", "c_max"),
    [("gt1", 0.5, 2.0), ("gt1", 1.0, 2.0), ("lt1", 0.1, 1.0), ("lt1", 0.5, 1.5)],
)
def test_init_outside_the_range_is_rejected(exponent: str, c_min: float, c_max: float) -> None:
    with pytest.raises(ValueError, match="cannot start"):
        LogDistanceBias(H, c_min, c_max, exponent=exponent)


def test_unknown_exponent_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown exponent"):
        LogDistanceBias(H, exponent="ge1")
