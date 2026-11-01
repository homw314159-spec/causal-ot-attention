"""Properties of ALiBi: the distance-only additive action of docs/methods.md §2."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core import ALiBi, GeometricAttention, PositionEncoding, Softmax
from geoattn_core.pe.alibi import alibi_slopes

SETTINGS = settings(max_examples=30, deadline=None)


def test_satisfies_protocol() -> None:
    assert isinstance(ALiBi(4), PositionEncoding)


def test_act_is_identity() -> None:
    q, k = torch.randn(2, 1, 4, 5, 8)
    rq, rk = ALiBi(4).act(q, k, torch.arange(5))
    assert rq is q and rk is k


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 12, 16])
def test_slopes_are_positive_distinct_and_match_the_paper(heads: int) -> None:
    s = alibi_slopes(heads)
    assert s.shape == (heads,) and (s > 0).all() and (s < 1).all()
    assert len(set(s.tolist())) == heads
    if heads == 8:  # the paper's example: 1/2, 1/4, ..., 1/256
        torch.testing.assert_close(s, torch.tensor([2.0**-i for i in range(1, 9)]))


@SETTINGS
@given(shift=st.integers(-5000, 5000), n=st.integers(1, 12), seed=st.integers(0, 2**31 - 1))
def test_bias_depends_only_on_relative_position(shift: int, n: int, seed: int) -> None:
    pos = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) * 3
    pe = ALiBi(6)
    torch.testing.assert_close(pe.bias(pos + shift, pos + shift), pe.bias(pos, pos))


def test_detector_catches_absolute_bias() -> None:
    """Control: a bias of absolute key position is not shift invariant."""

    def absolute(pq: torch.Tensor, pk: torch.Tensor) -> torch.Tensor:
        return -0.1 * pk.to(torch.float32)[None, None, :].expand(1, pq.numel(), -1)

    pos = torch.arange(5)
    assert (absolute(pos + 3, pos + 3) - absolute(pos, pos)).abs().max() > 0.1


def test_bias_values_and_shapes() -> None:
    pe = ALiBi(2)
    b = pe.bias(torch.arange(4), torch.arange(4))
    assert b.shape == (2, 4, 4)
    torch.testing.assert_close(b[0, 3], -alibi_slopes(2)[0] * torch.tensor([3.0, 2.0, 1.0, 0.0]))
    assert (b.diagonal(dim1=-2, dim2=-1) == 0).all()
    batched = pe.bias(torch.arange(4).repeat(3, 1), torch.arange(4).repeat(3, 1))
    assert batched.shape == (3, 2, 4, 4)
    torch.testing.assert_close(batched[1], b)


def test_penalizes_far_keys_in_attention() -> None:
    """With constant scores, ALiBi makes a causal row decay with distance."""
    attn = GeometricAttention(dim=8, num_heads=2, pe=ALiBi(2), normalizer=Softmax())
    with torch.no_grad():
        attn.qkv.weight.zero_()
    captured: list = []
    attn.probe.register_forward_hook(lambda _m, _i, out: captured.append(out))
    mask = torch.ones(6, 6, dtype=torch.bool).tril()
    attn(torch.randn(1, 6, 8), mask=mask)
    row = captured[0][0, 0, -1]
    assert (row.diff() > 0).all()  # nearer keys (larger j) get more weight
