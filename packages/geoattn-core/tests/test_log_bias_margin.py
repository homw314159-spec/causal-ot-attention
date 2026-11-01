"""Margin of LogDistanceBias's ``gt1`` constraint (experiment 0004, gate G1).

With margin m every exponent must stay above 1 + m at every parameter value, hence at
every training step; ``margin=0`` must be experiment 0003's ``gt1`` unchanged, so its
checkpoints still load and give the same bias.
"""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core.pe.log_bias import LogDistanceBias

H = 6
MARGINS = [0.25, 0.5]


def test_zero_margin_is_gt1_elementwise() -> None:
    plain, zero = LogDistanceBias(H, exponent="gt1"), LogDistanceBias(H, exponent="gt1", margin=0)
    assert torch.equal(plain.raw, zero.raw)
    with torch.no_grad():
        plain.raw.normal_()
        zero.raw.copy_(plain.raw)
    pos = torch.arange(9)
    assert torch.equal(plain.bias(pos, pos), zero.bias(pos, pos))
    assert list(zero.state_dict()) == ["raw"]  # experiment 0003 checkpoints load as before


@pytest.mark.parametrize("margin", MARGINS)
def test_init_is_log_spaced_over_the_shifted_range(margin: float) -> None:
    pe = LogDistanceBias(H, exponent="gt1", margin=margin)
    lo, hi = 1.05 + margin, 2.0 + margin  # preregistration §2, G1
    torch.testing.assert_close(pe.coef, torch.logspace(math.log10(lo), math.log10(hi), H))


@settings(max_examples=60, deadline=None)
@given(raw=st.lists(st.floats(-10, 10), min_size=H, max_size=H))
def test_margin_holds_for_any_parameter(raw: list[float]) -> None:
    for margin in MARGINS:
        pe = LogDistanceBias(H, exponent="gt1", margin=margin)
        with torch.no_grad():
            pe.raw.copy_(torch.tensor(raw))
        assert (pe.coef > 1 + margin).all(), (margin, pe.coef)


@pytest.mark.parametrize("margin", MARGINS)
def test_margin_holds_during_training(margin: float) -> None:
    """AdamW steps that lower every exponent: they approach 1 + m and never cross it."""
    pe = LogDistanceBias(H, exponent="gt1", margin=margin)
    opt = torch.optim.AdamW(pe.parameters(), lr=0.04, weight_decay=0.0)
    pos = torch.arange(32)
    for _ in range(200):
        opt.zero_grad()
        (-pe.bias(pos, pos).sum()).backward()
        opt.step()
        assert (pe.coef > 1 + margin).all(), pe.coef
    assert (pe.coef < 1 + margin + 0.05).all()  # the steps did reach the boundary


@pytest.mark.parametrize("margin", MARGINS)
def test_gradcheck(margin: float) -> None:
    wrapper = _Bias(LogDistanceBias(4, exponent="gt1", margin=margin).double())
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


@pytest.mark.parametrize(("exponent", "margin"), [("free", 0.5), ("lt1", 0.5), ("gt1", -0.1)])
def test_invalid_margin_is_rejected(exponent: str, margin: float) -> None:
    with pytest.raises(ValueError, match="margin"):
        LogDistanceBias(H, exponent=exponent, margin=margin)


def test_init_below_the_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot start"):
        LogDistanceBias(H, 1.2, 2.0, exponent="gt1", margin=0.5)
