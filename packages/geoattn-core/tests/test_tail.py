"""Evaluation-time tail interventions (experiment 0003, gate G2).

At distances d <= L the bias must be the wrapped encoding's bit for bit (so the 1x loss is
unchanged); at d > L it must follow the preregistered definition, read at the same query.
The fused path must agree with the reference. The counter-example shifts the boundary by
one and must be caught by the same check.
"""

import copy
import math

import pytest
import torch

from geoattn_core import ALiBi, GeometricAttention, RoPE, Softmax
from geoattn_core.pe import (
    Cable,
    ClockLogDistanceBias,
    ComposedPE,
    ForgetGate,
    LogDistanceBias,
    TailIntervention,
)

DIM, H, D, N, L = 32, 4, 8, 14, 5
SLOPE = 0.07

PES = {
    "rope": lambda dim=DIM, d=D: RoPE(d),
    "alibi": lambda dim=DIM, d=D: ALiBi(H),
    "logbias": lambda dim=DIM, d=D: LogDistanceBias(H),
    "rope+logbias": lambda dim=DIM, d=D: ComposedPE(RoPE(d), LogDistanceBias(H)),
    "rope+clock": lambda dim=DIM, d=D: ComposedPE(RoPE(d), ClockLogDistanceBias(dim, H)),
    "cable": lambda dim=DIM, d=D: Cable(dim, H),
    "fox": lambda dim=DIM, d=D: ForgetGate(dim, H),
}
MODES = ("cap", "repair", "window")


def _total_bias(pe, x: torch.Tensor) -> torch.Tensor:
    """The additive bias the attention adds, as (B, H, N, N); zeros when there is none."""
    torch.manual_seed(1)
    q, k = torch.randn(2, x.shape[0], H, x.shape[1], D)
    pos = torch.arange(x.shape[1])
    ctx = getattr(pe, "bias_ctx", None)
    b = ctx(q, k, x, pos) if ctx else pe.bias(pos, pos)
    return torch.zeros(x.shape[0], H, x.shape[1], x.shape[1]) + (0.0 if b is None else b)


def _expected(old: torch.Tensor, mode: str, big_l: int) -> torch.Tensor:
    """The definition of experiment 0003 §3.3, entry by entry."""
    out = old.clone()
    for t in range(old.shape[-2]):
        for j in range(t - big_l):  # d = t - j > L
            d = t - j
            if mode == "cap":
                out[..., t, j] = old[..., t, t - big_l]
            elif mode == "repair":
                decay = math.log1p(d) - math.log1p(big_l)
                out[..., t, j] = old[..., t, t - big_l] - 1.5 * decay
            else:
                out[..., t, j] = old[..., t, j] - SLOPE * (d - big_l)
    return out


def _unchanged_within(new: torch.Tensor, old: torch.Tensor, big_l: int) -> bool:
    d = torch.arange(new.shape[-2])[:, None] - torch.arange(new.shape[-1])
    near = d <= big_l
    return torch.equal(new[..., near], old[..., near])


def _x() -> torch.Tensor:
    return torch.randn(2, N, DIM, generator=torch.Generator().manual_seed(0)) * 2


def _perturbed(make):
    torch.manual_seed(0)
    pe = make()
    with torch.no_grad():  # content gates and exponents away from their init
        for p in pe.parameters():
            p.add_(0.5 * torch.randn_like(p))
    return pe


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(PES))
def test_bias_is_unchanged_within_and_follows_the_definition_beyond(name: str, mode: str) -> None:
    pe = _perturbed(PES[name])
    old = _total_bias(pe, _x())
    new = _total_bias(TailIntervention(pe, mode, L, slope=SLOPE), _x())
    assert _unchanged_within(new, old, L)
    torch.testing.assert_close(new, _expected(old, mode, L), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_counter_example_boundary_off_by_one_is_caught(mode: str) -> None:
    pe = _perturbed(PES["rope+logbias"])
    old = _total_bias(pe, _x())
    wrong = _total_bias(TailIntervention(pe, mode, L - 1, slope=SLOPE), _x())
    assert not _unchanged_within(wrong, old, L)


def test_cap_bounds_a_decaying_bias_and_repair_makes_it_summable() -> None:
    pe = LogDistanceBias(H)  # exponents 0.1-1: not summable
    pos = torch.arange(200)
    cap = TailIntervention(pe, "cap", L).bias(pos, pos)
    rep = TailIntervention(pe, "repair", L).bias(pos, pos)
    assert cap is not None and rep is not None
    assert torch.equal(cap[:, -1, :-L], cap[:, -1, -L - 1 : -L].expand(H, 200 - L))
    far = torch.arange(200 - L - 1)  # keys of the last row at d > L
    weights = rep[:, -1, far].exp()
    # tail exponent 1.5: the weight at distance d is (1+L)^1.5 b(L) / (1+d)^1.5
    d = (199 - far).float()
    scaled = weights * (1 + d) ** 1.5
    torch.testing.assert_close(scaled, scaled[:, -1:].expand_as(scaled))


def test_short_sequences_are_untouched() -> None:
    pe = _perturbed(PES["cable"])
    x = _x()[:, :L]
    for mode in MODES:
        assert torch.equal(
            _total_bias(TailIntervention(pe, mode, L, slope=SLOPE), x), _total_bias(pe, x)
        )


def test_checkpoint_of_the_unwrapped_encoding_loads() -> None:
    pe = _perturbed(PES["rope+clock"])
    wrapped = TailIntervention(copy.deepcopy(PES["rope+clock"]()), "cap", L)
    wrapped.load_state_dict(pe.state_dict(), strict=True)
    for (name, p), (_, q) in zip(pe.named_parameters(), wrapped.pe.named_parameters(), strict=True):
        assert torch.equal(p, q), name


def test_fused_path_refuses_gradients() -> None:
    wrapped = TailIntervention(LogDistanceBias(H), "cap", L)
    q = torch.randn(1, H, N, D)
    with pytest.raises(RuntimeError, match="evaluation"):
        wrapped.score_mod(q, q, _x()[:1], torch.arange(N))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "clip", "tail_from": L}, "unknown tail mode"),
        ({"mode": "cap", "tail_from": 0}, "tail_from"),
        ({"mode": "window", "tail_from": L}, "positive slope"),
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TailIntervention(LogDistanceBias(H), **kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(PES))
def test_flex_matches_reference(name: str, mode: str) -> None:
    pe = TailIntervention(_perturbed(lambda: PES[name](64, 16)), mode, 32, slope=SLOPE)
    if name == "fox":  # gates near 1, else keys beyond 32 tokens carry ~e^-23 of the weight
        with torch.no_grad():
            pe.pe.proj.bias.add_(4.0)  # type: ignore[union-attr]
    torch.manual_seed(0)
    ref = GeometricAttention(64, H, pe, Softmax()).cuda().eval()
    fused = copy.deepcopy(ref)
    fused.backend = "flex"
    x = torch.randn(2, 128, 64, device="cuda")
    assert fused.uses_flex(x)
    mask = torch.ones(128, 128, dtype=torch.bool, device="cuda").tril()
    plain = copy.deepcopy(fused)
    plain.pe = pe.pe
    with torch.no_grad():
        out_r, out_f = ref(x, mask=mask, causal=True), fused(x, mask=mask, causal=True)
        out_p = plain(x, mask=mask, causal=True)
    torch.testing.assert_close(out_f, out_r, atol=2e-5, rtol=1e-4)
    if (name, mode) != ("rope", "cap"):  # capping a zero bias is the identity
        assert (out_f - out_p).abs().max() > 1e-3  # the intervention took effect
    # rows t <= L have no key beyond L: same values as without the intervention
    torch.testing.assert_close(out_f[:, :33], out_p[:, :33], atol=2e-5, rtol=1e-4)
