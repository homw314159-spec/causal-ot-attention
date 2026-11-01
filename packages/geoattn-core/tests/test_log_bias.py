"""LogDistanceBias and ComposedPE (docs/methods.md §2: multiplicative + additive actions)."""

import math

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core import ALiBi, GeometricAttention, PositionEncoding, RoPE, Softmax
from geoattn_core.pe.log_bias import ComposedPE, LogDistanceBias


def test_protocols() -> None:
    assert isinstance(LogDistanceBias(4), PositionEncoding)
    assert isinstance(ComposedPE(RoPE(8), LogDistanceBias(4)), PositionEncoding)


def test_init_exponents_and_values() -> None:
    pe = LogDistanceBias(4, c_min=0.1, c_max=1.0)
    torch.testing.assert_close(pe.coef, torch.logspace(-1, 0, 4))
    b = pe.bias(torch.arange(5), torch.arange(5))
    assert b.shape == (4, 5, 5)
    torch.testing.assert_close(b[3, 4, 0], torch.tensor(-math.log(5.0)))
    assert (b.diagonal(dim1=-2, dim2=-1) == 0).all()


@settings(max_examples=30, deadline=None)
@given(shift=st.integers(-5000, 5000), n=st.integers(1, 10), seed=st.integers(0, 999))
def test_bias_depends_only_on_relative_position(shift: int, n: int, seed: int) -> None:
    pos = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) * 3
    pe = LogDistanceBias(3)
    torch.testing.assert_close(pe.bias(pos + shift, pos + shift), pe.bias(pos, pos))


def test_learnable_flag_and_gradient() -> None:
    assert not LogDistanceBias(2, learnable=False).raw.requires_grad
    pe = LogDistanceBias(2)
    pe.bias(torch.arange(6), torch.arange(6)).sum().backward()
    assert pe.raw.grad is not None and (pe.raw.grad < 0).all()  # larger c, lower scores


def test_composed_adds_biases_and_rotates() -> None:
    torch.manual_seed(0)
    rope, alibi, logb = RoPE(8), ALiBi(2), LogDistanceBias(2)
    pos = torch.arange(6)
    q, k = torch.randn(2, 1, 2, 6, 8)
    cq, ck = ComposedPE(rope, logb).act(q, k, pos)
    rq, rk = rope.act(q, k, pos)
    torch.testing.assert_close(cq, rq)
    torch.testing.assert_close(ck, rk)
    both = ComposedPE(alibi, logb).bias(pos, pos)
    assert both is not None
    torch.testing.assert_close(both, alibi.bias(pos, pos) + logb.bias(pos, pos))
    assert ComposedPE(rope, RoPE(8)).bias(pos, pos) is None


def test_parameters_are_registered_in_attention() -> None:
    pe = ComposedPE(RoPE(8), LogDistanceBias(2))
    attn = GeometricAttention(dim=16, num_heads=2, pe=pe, normalizer=Softmax())
    assert any(n.endswith("additive.raw") for n, _ in attn.named_parameters())
