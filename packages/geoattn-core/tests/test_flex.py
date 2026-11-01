"""FlexAttention backend (ADR-0012; experiment 0002 gate G2).

The flex path must give the reference values (outputs and every gradient) for each fusable
encoding, must actually be taken when it should (a counter on the flex call guards against a
silent fallback making the equivalence test vacuous), and must fall back when the call does
not allow it.
"""

import copy

import pytest
import torch

from geoattn_core import ALiBi, GeometricAttention, IdentityPE, RoPE, Softmax, flex
from geoattn_core.pe import (
    Cable,
    ClockLogDistanceBias,
    ComposedPE,
    ForgetGate,
    GatedLogDistanceBias,
    GrapeA,
    GrapeAP,
    GrapeM,
    KerpleLogBias,
    LogDistanceBias,
)

DIM, H, D = 64, 4, 16

FUSABLE = {
    "identity": lambda: IdentityPE(),
    "rope": lambda: RoPE(D),
    "alibi": lambda: ALiBi(H),
    "logbias": lambda: LogDistanceBias(H),
    "logbias-gt1": lambda: LogDistanceBias(H, exponent="gt1"),
    "logbias-lt1": lambda: LogDistanceBias(H, exponent="lt1"),
    "logbias-margin": lambda: LogDistanceBias(H, exponent="gt1", margin=0.5),
    "rope+kerple": lambda: ComposedPE(RoPE(D), KerpleLogBias(H)),
    "rope+gated": lambda: ComposedPE(RoPE(D), GatedLogDistanceBias(DIM, H)),
    "rope+clock": lambda: ComposedPE(RoPE(D), ClockLogDistanceBias(DIM, H)),
    "cable": lambda: Cable(DIM, H),
    "fox": lambda: ForgetGate(DIM, H),
    "grape_a": lambda: GrapeA(H, D),
    "grape_m": lambda: GrapeM(D),
}


def _pair(make_pe, seed: int = 0) -> tuple[GeometricAttention, GeometricAttention]:
    torch.manual_seed(seed)
    ref = GeometricAttention(DIM, H, make_pe(), Softmax()).cuda()
    with torch.no_grad():  # move the encodings' parameters off their (near-trivial) init
        for p in (p for name, p in ref.named_parameters() if name.startswith("pe.")):
            if p.ndim < 2:  # scales, exponents, offsets: perturbed, kept positive
                p.add_(0.3 * torch.randn_like(p)).abs_()
            else:  # content gates: make them matter
                p.mul_(20)
    fused = copy.deepcopy(ref)
    fused.backend = "flex"
    return ref, fused


def _run(attn: GeometricAttention, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    n = x.shape[1]
    x = x.clone().requires_grad_()
    mask = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
    out = attn(x, mask=mask, causal=True)
    out.square().sum().backward()
    grads = {"x": x.grad} | {k: p.grad for k, p in attn.named_parameters()}
    return out.detach(), grads


@pytest.fixture
def flex_calls(monkeypatch):
    calls = []
    real = flex.causal_attention

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(flex, "causal_attention", counted)
    return calls


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(FUSABLE))
def test_flex_matches_reference(name: str, flex_calls: list) -> None:
    ref, fused = _pair(FUSABLE[name])
    x = torch.randn(2, 96, DIM, device="cuda")
    assert fused.uses_flex(x) and not ref.uses_flex(x)
    out_r, g_r = _run(ref, x)
    out_f, g_f = _run(fused, x)
    assert len(flex_calls) == 1  # the fused module really took the flex path
    torch.testing.assert_close(out_f, out_r, atol=2e-5, rtol=1e-4)
    for k, g in g_r.items():
        assert g_f[k] is not None, k
        scale = g.abs().max().clamp_min(1e-6)
        assert ((g_f[k] - g).abs().max() / scale).item() < 1e-4, k


@pytest.mark.gpu
@pytest.mark.parametrize("n", [2048, 4096])
def test_flex_runs_at_long_lengths_in_bf16(n: int) -> None:
    ref, fused = _pair(FUSABLE["rope+clock"])
    x = torch.randn(1, n, DIM, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_f, _ = _run(fused, x)
        out_r, _ = _run(ref, x)
    assert torch.isfinite(out_f).all()
    torch.testing.assert_close(out_f.float(), out_r.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.gpu
def test_falls_back_when_the_call_does_not_allow_flex(flex_calls: list) -> None:
    x = torch.randn(1, 32, DIM, device="cuda")
    _, unfusable = _pair(lambda: GrapeAP(DIM, H))
    _, composed = _pair(lambda: ComposedPE(RoPE(D), GrapeAP(DIM, H)))
    _, hot = _pair(lambda: RoPE(D))
    hot.normalizer = Softmax(eps=0.7)
    _, probed = _pair(lambda: RoPE(D))
    seen = []
    probed.probe.register_forward_hook(lambda m, i, p: seen.append(p))
    for attn in (unfusable, composed, hot, probed):
        assert not attn.uses_flex(x)
        _run(attn, x)
    _, fused = _pair(lambda: RoPE(D))
    fused(x, mask=None, causal=False)  # not declared causal
    assert flex_calls == [] and len(seen) == 1


def test_cpu_input_uses_the_reference() -> None:
    torch.manual_seed(1)
    attn = GeometricAttention(DIM, H, RoPE(D), Softmax())
    fused = copy.deepcopy(attn)
    fused.backend = "flex"
    x = torch.randn(1, 8, DIM)
    mask = torch.ones(8, 8, dtype=torch.bool).tril()
    assert not fused.uses_flex(x)
    torch.testing.assert_close(fused(x, mask=mask, causal=True), attn(x, mask=mask))


def test_backend_is_validated() -> None:
    attn = GeometricAttention(DIM, H, RoPE(D), Softmax())
    assert attn.backend == "reference"
    with pytest.raises(ValueError, match="backend"):
        attn.backend = "cuda"


@pytest.mark.gpu
def test_recompile_limit_is_an_error_not_a_silent_fallback(monkeypatch) -> None:
    """Past the limit dynamo would run the unfused eager kernel without a word; the flex
    path must refuse instead. With a limit of 1, a second length cannot compile."""
    monkeypatch.setattr(flex, "RECOMPILE_LIMIT", 1)
    torch._dynamo.reset()
    _, fused = _pair(lambda: LogDistanceBias(H))
    _run(fused, torch.randn(1, 64, DIM, device="cuda"))
    with pytest.raises(Exception, match=r"(?i)recompile"):
        _run(fused, torch.randn(1, 80, DIM, device="cuda"))
    torch._dynamo.reset()
