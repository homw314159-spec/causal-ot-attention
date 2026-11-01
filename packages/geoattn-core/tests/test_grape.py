"""GRAPE-M and GRAPE-A against their definitions and the official implementation.

The official modules are loaded from ``third_party/grape`` (model-architectures/GRAPE @
f5b539f, Apache-2.0) when it is cloned; those tests skip otherwise, like the Sinkformers
equivalence test. Their package imports ``transformers`` only for model base classes,
which are stubbed here.
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.nn import functional as F

from geoattn_core import ALiBi, GeometricAttention, PositionEncoding, RoPE, Softmax
from geoattn_core.pe import Cable, ForgetGate, GrapeA, GrapeAP, GrapeM

OFFICIAL = Path(__file__).resolve().parents[3] / "third_party/grape/model"


def _official(module: str):
    if not (OFFICIAL / f"{module}.py").exists():
        pytest.skip("third_party/grape not cloned (git clone model-architectures/GRAPE)")
    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")
        tf.PreTrainedModel = type("PreTrainedModel", (torch.nn.Module,), {})  # type: ignore[attr-defined]
        cu, mu = (
            types.ModuleType("transformers.configuration_utils"),
            types.ModuleType("transformers.modeling_utils"),
        )
        cu.PretrainedConfig = type("PretrainedConfig", (), {})  # type: ignore[attr-defined]
        mu.PreTrainedModel = tf.PreTrainedModel  # type: ignore[attr-defined]
        sys.modules.update(
            {"transformers": tf, "transformers.configuration_utils": cu,
             "transformers.modeling_utils": mu}
        )  # fmt: skip
    pkg = types.ModuleType("grape_official")
    pkg.__path__ = [str(OFFICIAL)]  # type: ignore[attr-defined]
    sys.modules.setdefault("grape_official", pkg)
    spec = importlib.util.spec_from_file_location(
        f"grape_official.{module}", OFFICIAL / f"{module}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scores(pe, q, k, pos):
    q, k = pe.act(q, k, pos)
    return q @ k.transpose(-2, -1)


# ---------------------------------------------------------------- GRAPE-M


def test_grape_m_protocol_and_init_is_rope() -> None:
    assert isinstance(GrapeM(16), PositionEncoding)
    torch.manual_seed(0)
    q, k = torch.randn(2, 1, 2, 9, 16).unbind(0)
    pos = torch.arange(9)
    torch.testing.assert_close(_scores(GrapeM(16), q, k, pos), _scores(RoPE(16), q, k, pos))


@settings(max_examples=25, deadline=None)
@given(shift=st.integers(-2000, 2000), seed=st.integers(0, 999))
def test_grape_m_group_law_keeps_scores_relative(shift: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    q, k = torch.randn(2, 1, 2, 7, 16, generator=g, dtype=torch.float64).unbind(0)
    pe = GrapeM(16)
    with torch.no_grad():  # move away from the RoPE spectrum
        pe.log_freq.add_(torch.randn(8, generator=g) * 4)
    pos = torch.arange(7) * 3 + 100
    gap = _scores(pe, q, k, pos + shift) - _scores(pe, q, k, pos)
    # Angles are formed in float32 (as in RoPE): rounding grows with the angle, about
    # angle * 1e-7, so the score error scales with the largest angle and |q| |k|. A broken
    # group law instead moves scores by O(|q| |k|).
    max_angle = (pos.abs().max() + abs(shift)) * pe.freq.max()
    tol = 1e-6 + 1e-6 * max_angle.item() * (q.norm(dim=-1).max() * k.norm(dim=-1).max()).item()
    assert gap.abs().max() < tol


def test_grape_m_frequencies_learn() -> None:
    pe = GrapeM(8)
    q, k = torch.randn(2, 1, 1, 5, 8).unbind(0)
    _scores(pe, q, k, torch.arange(5)).sum().backward()
    assert pe.log_freq.grad is not None and pe.log_freq.grad.abs().sum() > 0
    assert not GrapeM(8, learnable=False).log_freq.requires_grad


def test_grape_m_matches_official() -> None:
    off = _official("llama-mha-grape-m-nonctx").MSGRAPECommuting(
        n_head=2, head_dim=16, log_freq_scale=16.0, share_across_heads=True
    )
    ours = GrapeM(16, log_freq_scale=16.0)
    with torch.no_grad():
        off.log_freq.add_(torch.linspace(-3, 3, 8))
        ours.log_freq.copy_(off.log_freq)
    torch.manual_seed(1)
    q, k = torch.randn(2, 2, 11, 2, 16).unbind(0)  # official layout (B, T, H, D)
    oq, ok = off(q, k)
    # The official rotation goes the other way: identical to ours at negated positions.
    rq, rk = ours.act(q.transpose(1, 2), k.transpose(1, 2), -torch.arange(11))
    torch.testing.assert_close(rq.transpose(1, 2), oq, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rk.transpose(1, 2), ok, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------- GRAPE-A


def _ctx_bias(pe: GrapeA, q: torch.Tensor, k: torch.Tensor, n: int) -> torch.Tensor:
    return pe.bias_ctx(q, k, torch.zeros(q.shape[0], n, 1), torch.arange(n))


def test_grape_a_constant_gates_is_alibi() -> None:
    """Zero gate vectors give softplus(0) = ln 2 per gate; with softplus(omega) = 1/(2 ln 2)
    the penalty is exactly ALiBi's on and below the diagonal."""
    n = 8
    pe = GrapeA(4, 16, omega_init=1 / (2 * math.log(2)))
    with torch.no_grad():
        pe.u.zero_()
        pe.v.zero_()
    q, k = torch.randn(2, 2, 4, n, 16).unbind(0)
    b = _ctx_bias(pe, q, k, n)
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    expected = ALiBi(4).bias(torch.arange(n), torch.arange(n)).expand_as(b)
    torch.testing.assert_close(b[..., causal], expected[..., causal], atol=1e-6, rtol=1e-6)


def test_grape_a_slope_depends_on_content() -> None:
    """Control for the test above: with learned gate vectors, two inputs at the same
    positions get different penalties."""
    torch.manual_seed(2)
    pe = GrapeA(2, 16)
    q1, k1, q2, k2 = torch.randn(4, 1, 2, 6, 16).unbind(0)
    assert (_ctx_bias(pe, q1, k1, 6) - _ctx_bias(pe, q2, k2, 6)).abs().max() > 1e-3


@settings(max_examples=20, deadline=None)
@given(shift=st.integers(-1000, 1000), seed=st.integers(0, 999))
def test_grape_a_is_relative(shift: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    q, k = torch.randn(2, 1, 2, 5, 8, generator=g).unbind(0)
    pe = GrapeA(2, 8)
    x = torch.zeros(1, 5, 1)
    pos = torch.arange(5) * 2
    torch.testing.assert_close(pe.bias_ctx(q, k, x, pos + shift), pe.bias_ctx(q, k, x, pos))


def test_grape_a_matches_official() -> None:
    off = _official("llama-mha-grape-a-querykeygated").QueryKeyGatedAdditiveBias(4, 16)
    ours = GrapeA(4, 16)
    with torch.no_grad():
        for name in ("omega", "u", "v"):
            getattr(ours, name).copy_(getattr(off, name))
    torch.manual_seed(3)
    q, k = torch.randn(2, 2, 10, 4, 16).unbind(0)  # (B, T, H, D)
    # The official model applies QK-RMSNorm before the bias; ours normalizes internally.
    theirs = off(F.rms_norm(q, (16,)), F.rms_norm(k, (16,)))
    mine = _ctx_bias(ours, q.transpose(1, 2), k.transpose(1, 2), 10)
    torch.testing.assert_close(mine, theirs, atol=1e-5, rtol=1e-5)


def test_attention_uses_bias_ctx() -> None:
    torch.manual_seed(4)
    attn = GeometricAttention(dim=32, num_heads=2, pe=GrapeA(2, 16), normalizer=Softmax())
    mask = torch.ones(7, 7, dtype=torch.bool).tril()
    out = attn(torch.randn(2, 7, 32), mask=mask)
    assert out.shape == (2, 7, 32) and torch.isfinite(out).all()
    with pytest.raises(TypeError):
        GrapeA(2, 16).bias(torch.arange(3), torch.arange(3))


# ---------------------------------------------------------------- GRAPE-AP and FoX

PATH_BIASES = {
    "grape_ap": lambda: GrapeAP(12, 2, p_dim=4),
    "fox": lambda: ForgetGate(12, 2),
    "cable": lambda: Cable(12, 2),
}


def _path_bias(pe, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
    n = x.shape[1]
    pos = torch.arange(n) if positions is None else positions
    return pe.bias_ctx(None, None, x, pos)


def _fox_by_definition(pe: ForgetGate, x: torch.Tensor) -> torch.Tensor:
    log_f = F.logsigmoid(pe.proj(x)).transpose(1, 2)  # (B, H, N)
    n = x.shape[1]
    out = torch.zeros(*log_f.shape, n, dtype=x.dtype)
    for t in range(n):
        for j in range(t + 1):
            out[..., t, j] = log_f[..., j + 1 : t + 1].sum(-1)
    return out


def test_fox_matches_its_definition() -> None:
    torch.manual_seed(5)
    pe = ForgetGate(12, 3).double()
    x = torch.randn(2, 9, 12, dtype=torch.float64)
    torch.testing.assert_close(_path_bias(pe, x), _fox_by_definition(pe, x))


@settings(max_examples=20, deadline=None)
@given(name=st.sampled_from(sorted(PATH_BIASES)), seed=st.integers(0, 999))
def test_path_bias_decays_with_distance(name: str, seed: int) -> None:
    """psi <= 0, so the bias is 0 on the diagonal and non-increasing away from it."""
    torch.manual_seed(seed)
    b = _path_bias(PATH_BIASES[name](), torch.randn(2, 10, 12) * 3)
    causal = torch.ones(10, 10, dtype=torch.bool).tril()
    assert b.diagonal(dim1=-2, dim2=-1).abs().max() < 1e-5  # total - cumsum rounding
    assert (b[..., ~causal] == 0).all()
    step = b[..., :, 1:] - b[..., :, :-1]  # b(t, j + 1) - b(t, j) = -psi(t, j + 1) >= 0
    assert (step[..., causal[:, 1:]] >= -1e-6).all()


def test_fox_is_relative_and_grape_ap_is_not() -> None:
    """FoX's psi depends on l only through x_l; GRAPE-AP v1 rotates by the absolute
    position l, so shifting all positions changes its bias (its definition, kept)."""
    torch.manual_seed(6)
    x = torch.randn(1, 8, 12)
    pos = torch.arange(8)
    fox, ap = ForgetGate(12, 2), GrapeAP(12, 2, p_dim=4)
    torch.testing.assert_close(_path_bias(fox, x, pos + 37), _path_bias(fox, x, pos))
    assert (_path_bias(ap, x, pos + 37) - _path_bias(ap, x, pos)).abs().max() > 1e-4


@pytest.mark.parametrize("name", sorted(PATH_BIASES))
def test_path_bias_gradcheck(name: str) -> None:
    torch.manual_seed(7)
    pe = PATH_BIASES[name]().double()
    x = torch.randn(1, 5, 12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: _path_bias(pe, x), (x,))


def test_grape_a_gradcheck() -> None:
    torch.manual_seed(8)
    pe = GrapeA(2, 8).double()
    q, k = torch.randn(2, 1, 2, 5, 8, dtype=torch.float64).unbind(0)
    x = torch.zeros(1, 5, 1, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda q, k: pe.bias_ctx(q, k, x, torch.arange(5)),
        (q.requires_grad_(), k.requires_grad_()),
    )


def test_grape_ap_attention_matches_official() -> None:
    """Whole attention layer: same q/k/v/p/out weights, official QK-RMSNorm off."""
    pytest.importorskip("triton")  # imported by the official kv_shift module
    mod = _official("llama-mha-grape-ap-v1")
    h, d, c, n = 4, 8, 32, 12
    cfg = types.SimpleNamespace(
        n_head=h, n_embd=c, head_dim=d, p_head_dim=4, p_tie_mode="none", n_layer=1,
        use_qk_rmsnorm=False, using_groupnorm=False,
    )  # fmt: skip
    off = mod.CausalSelfAttention(cfg)
    ours = GeometricAttention(c, h, GrapeAP(c, h, p_dim=4), Softmax(), qkv_bias=False)
    with torch.no_grad():
        ours.qkv.weight.copy_(torch.cat([off.c_q.weight, off.c_k.weight, off.c_v.weight]))
        ours.proj.weight.copy_(off.c_proj.weight)
        ours.proj.bias.zero_()
        ours.pe.proj.weight.copy_(off.c_p.weight)  # type: ignore[union-attr]
    torch.manual_seed(9)
    x = torch.randn(2, n, c)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    torch.testing.assert_close(ours(x, mask=mask), off(x), atol=1e-5, rtol=1e-5)
