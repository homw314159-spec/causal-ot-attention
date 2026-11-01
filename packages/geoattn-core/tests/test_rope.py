"""Properties of RoPE: the SO(2)^{d/2} action of docs/methods.md §2."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core import GeometricAttention, PositionEncoding, RoPE, Softmax

SETTINGS = settings(max_examples=30, deadline=None)


def _scores(pe, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    q, k = pe.act(q, k, positions)
    return q @ k.transpose(-2, -1)


def test_satisfies_protocol() -> None:
    assert isinstance(RoPE(8), PositionEncoding)


@SETTINGS
@given(shift=st.integers(-5000, 5000), n=st.integers(1, 12), seed=st.integers(0, 2**31 - 1))
def test_scores_depend_only_on_relative_position(shift: int, n: int, seed: int) -> None:
    """<R_m q, R_n k> = <q, R_{n-m} k>: shifting every position leaves scores unchanged."""
    g = torch.Generator().manual_seed(seed)
    q, k = torch.randn(2, 3, 2, n, 16, generator=g, dtype=torch.float64)
    pos = torch.randperm(n, generator=g) * 7
    pe = RoPE(16)
    # Angles are float32, so the angle error grows like eps32 * |position|; each score
    # moves by at most |q| |k| times that.
    max_pos = (pos.abs().max() + abs(shift)).item()
    bound = 4 * torch.finfo(torch.float32).eps * max(max_pos, 1)
    scale = q.norm(dim=-1)[..., :, None] * k.norm(dim=-1)[..., None, :]
    gap = _scores(pe, q, k, pos + shift) - _scores(pe, q, k, pos)
    assert (gap.abs() <= bound * scale + 1e-6).all()


def test_detector_catches_absolute_encoding() -> None:
    """Counter-example control: an additive absolute embedding is NOT shift invariant."""

    class AdditiveAbsolute(torch.nn.Module):
        def act(self, q, k, positions):
            emb = torch.sin(positions.to(q.dtype))[:, None] * 0.5
            return q + emb, k + emb

    torch.manual_seed(0)
    q, k = torch.randn(2, 1, 1, 6, 8)
    pos = torch.arange(6)
    gap = _scores(AdditiveAbsolute(), q, k, pos + 3) - _scores(AdditiveAbsolute(), q, k, pos)
    assert gap.abs().max() > 1e-2


def test_preserves_norms_and_is_identity_at_zero() -> None:
    torch.manual_seed(1)
    q, k = torch.randn(2, 2, 4, 5, 32)
    pos = torch.arange(5) * 100
    rq, rk = RoPE(32).act(q, k, pos)
    torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1))
    torch.testing.assert_close(rk.norm(dim=-1), k.norm(dim=-1))
    zq, zk = RoPE(32).act(q, k, torch.zeros(5, dtype=torch.long))
    torch.testing.assert_close((zq, zk), (q, k))


def test_matches_complex_multiplication_reference() -> None:
    """Independent reference: channel pair (i, i + d/2) as a complex number times e^{i m θ_i}."""
    torch.manual_seed(2)
    d, n = 12, 7
    x = torch.randn(1, 1, n, d, dtype=torch.float64)
    pos = torch.arange(n, dtype=torch.float64)
    theta = 10_000.0 ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    z = torch.complex(x[..., : d // 2], x[..., d // 2 :]) * torch.polar(
        torch.ones(n, d // 2, dtype=torch.float64), pos[:, None] * theta
    )
    ref = torch.cat((z.real, z.imag), dim=-1)
    out, _ = RoPE(d).act(x, x, pos.long())
    torch.testing.assert_close(out, ref.to(out.dtype), atol=1e-5, rtol=1e-5)


def test_per_sample_positions_match_shared_positions() -> None:
    torch.manual_seed(3)
    q, k = torch.randn(2, 3, 2, 5, 8)
    pos = torch.stack([torch.arange(5), torch.arange(5) + 10, torch.arange(5) * 2])
    rq, rk = RoPE(8).act(q, k, pos)
    for b in range(3):
        eq, ek = RoPE(8).act(q[b : b + 1], k[b : b + 1], pos[b])
        torch.testing.assert_close((rq[b : b + 1], rk[b : b + 1]), (eq, ek))


def test_bf16_input_keeps_dtype_and_accurate_angles() -> None:
    """Angles are formed in float32: at position 4000 a bf16 angle would be off by radians."""
    torch.manual_seed(4)
    q = torch.randn(1, 1, 3, 16)
    pos = torch.tensor([4000, 4001, 4002])
    exact, _ = RoPE(16).act(q, q, pos)
    low, _ = RoPE(16).act(q.bfloat16(), q.bfloat16(), pos)
    assert low.dtype == torch.bfloat16
    torch.testing.assert_close(low.float(), exact, atol=3e-2, rtol=3e-2)


def test_gradcheck() -> None:
    torch.manual_seed(5)
    q = torch.randn(1, 2, 4, 6, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 4, 6, dtype=torch.float64, requires_grad=True)
    pe, pos = RoPE(6), torch.arange(4)
    # Rotation runs in float32 internally, as in the normalizers: float32 tolerance.
    assert torch.autograd.gradcheck(
        lambda a, b: pe.act(a, b, pos), (q, k), eps=1e-3, atol=1e-3, rtol=1e-2
    )


def test_odd_head_dim_rejected() -> None:
    with pytest.raises(ValueError):
        RoPE(7)


def test_attention_with_rope_is_translation_invariant() -> None:
    """End to end through GeometricAttention: shifted positions give the same output."""
    torch.manual_seed(6)
    attn = GeometricAttention(32, 4, pe=RoPE(8), normalizer=Softmax()).eval()
    x = torch.randn(2, 9, 32)
    pos = torch.arange(9)
    torch.testing.assert_close(
        attn(x, positions=pos + 1234), attn(x, positions=pos), atol=1e-4, rtol=1e-4
    )
