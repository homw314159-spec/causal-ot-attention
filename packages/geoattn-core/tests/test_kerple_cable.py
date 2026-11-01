"""KERPLE's log kernel and CABLE (experiment 0002, P3 and the CABLE arm).

CABLE is checked against the official ``cable6.py`` (``AlgonetLabs/Cable`` @ ``05d2c62``)
when it is cloned into ``third_party/cable``; that test skips otherwise.
"""

import importlib.util
import math
import types
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core import GeometricAttention, PositionEncoding, RoPE, Softmax
from geoattn_core.pe import Cable, ComposedPE, KerpleLogBias, LogDistanceBias

CABLE_SRC = Path(__file__).resolve().parents[3] / "third_party/cable/src/pos_methods/cable6.py"


# ---------------------------------------------------------------- KERPLE


def test_kerple_protocol_init_and_formula() -> None:
    torch.manual_seed(0)
    pe = KerpleLogBias(8)
    assert isinstance(pe, PositionEncoding)
    assert (pe.p >= 0).all() and (pe.p < 2).all() and (pe.a >= 0).all() and (pe.a < 1).all()
    b = pe.bias(torch.arange(6), torch.arange(6))
    p, a = pe.p.detach().clamp_min(1e-2), pe.a.detach().clamp_min(1e-2)
    torch.testing.assert_close(b[3, 5, 1], -p[3] * torch.log(1 + a[3] * 4))
    assert (b.diagonal(dim1=-2, dim2=-1) == 0).all()


def test_kerple_with_unit_scale_is_the_log_distance_bias() -> None:
    pe = KerpleLogBias(4)
    ref = LogDistanceBias(4)
    with torch.no_grad():
        pe.a.fill_(1.0)
        pe.p.copy_(ref.coef)
    pos = torch.arange(9)
    torch.testing.assert_close(pe.bias(pos, pos), ref.bias(pos, pos))


def test_kerple_projects_parameters_like_the_official_code() -> None:
    """In-place projection onto [eps, inf): the parameter itself moves back to eps, so its
    gradient is live again. A clamp inside the computation (the control) would leave the
    parameter below eps with a zero gradient."""
    pe = KerpleLogBias(2)
    with torch.no_grad():
        pe.p.fill_(-0.5)
        pe.a.fill_(-0.5)
    pe.bias(torch.arange(4), torch.arange(4)).sum().backward()
    assert (pe.p == 1e-2).all() and (pe.a == 1e-2).all()
    assert pe.p.grad is not None and (pe.p.grad != 0).all()

    stuck = torch.tensor([-0.5], requires_grad=True)
    (-stuck.clamp(min=1e-2) * math.log(2)).sum().backward()
    assert stuck.grad is not None and stuck.grad.item() == 0  # the control


@settings(max_examples=25, deadline=None)
@given(shift=st.integers(-5000, 5000), seed=st.integers(0, 999))
def test_kerple_is_relative_and_decays(shift: int, seed: int) -> None:
    torch.manual_seed(seed)
    pe = KerpleLogBias(3)
    pos = torch.arange(7) * 2
    b = pe.bias(pos, pos)
    torch.testing.assert_close(pe.bias(pos + shift, pos + shift), b)
    lower = torch.ones(7, 7, dtype=torch.bool).tril()
    step = b[..., :, 1:] - b[..., :, :-1]  # nearer keys score higher below the diagonal
    assert (step[..., lower[:, 1:]] >= -1e-6).all()


def test_kerple_gradcheck() -> None:
    torch.manual_seed(1)
    pe = KerpleLogBias(2).double()
    pos = torch.arange(5)

    def f(p: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dist = (pos[:, None] - pos[None, :]).abs().double()
        return -p[:, None, None] * torch.log1p(a[:, None, None] * dist)

    p, a = pe.p.detach().clone().requires_grad_(), pe.a.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(f, (p, a))
    # the module computes the same function (projection is a no-op inside [eps, inf))
    torch.testing.assert_close(pe.bias(pos, pos), f(p, a))


def test_kerple_with_rope_and_a_fully_masked_row() -> None:
    attn = GeometricAttention(16, 2, ComposedPE(RoPE(8), KerpleLogBias(2)), Softmax())
    mask = torch.ones(6, 6, dtype=torch.bool).tril()
    mask[3] = False
    out = attn(torch.randn(2, 6, 16), mask=mask)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- CABLE


def test_cable_bias_by_definition() -> None:
    torch.manual_seed(2)
    pe = Cable(12, 3).double()
    x = torch.randn(2, 7, 12, dtype=torch.float64)
    f = torch.relu(pe.clock(x)).transpose(1, 2)  # (B, H, N)
    g = torch.nn.functional.softplus(pe.scale(x)).transpose(1, 2)
    expected = torch.zeros(2, 3, 7, 7, dtype=torch.float64)
    for t in range(7):
        for j in range(t + 1):
            expected[..., t, j] = -g[..., t] * f[..., j + 1 : t + 1].sum(-1)
    got = pe.bias_ctx(None, None, x, torch.arange(7))  # type: ignore[arg-type]  # q, k unused
    torch.testing.assert_close(got, expected)


def test_cable_attention_matches_official() -> None:
    if not CABLE_SRC.exists():
        pytest.skip("third_party/cable not cloned (git clone AlgonetLabs/Cable)")
    spec = importlib.util.spec_from_file_location("cable6_official", CABLE_SRC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    h, c, n = 4, 32, 10
    off = mod.CABLE6(types.SimpleNamespace(n_embd=c, n_head=h, block_size=64))
    ours = GeometricAttention(c, h, Cable(c, h), Softmax())
    with torch.no_grad():
        for dst, src in [
            (ours.qkv, off.c_attn),
            (ours.proj, off.c_proj),
            (ours.pe.clock, off.cable_layer),  # type: ignore[union-attr]
            (ours.pe.scale, off.cable_layer_scale),  # type: ignore[union-attr]
        ]:
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
    torch.manual_seed(3)
    x = torch.randn(2, n, c)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    torch.testing.assert_close(ours(x, mask=mask), off(x), atol=1e-5, rtol=1e-5)
