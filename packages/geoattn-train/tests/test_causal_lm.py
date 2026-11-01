"""Model-level causality of the GPT (gate G1 of experiment 0001, test-plan §4.2).

Operator-level prefix consistency (geoattn-core/tests/test_causal.py) does not
cover leaks from elsewhere in the model: position encoding, masks, or activation
checkpointing. Here the whole model must satisfy: changing tokens after position
t leaves the logits at positions <= t unchanged. The full-column normalizer is
the counter-example that the detector must catch.
"""

import pytest
import torch

from geoattn_core import ALiBi, RoPE, ScaledSoftmax, Softmax, UnbalancedSinkhorn
from geoattn_train.config import ModelConfig, NormalizerConfig
from geoattn_train.factory import build_normalizer
from geoattn_train.models import GPT

CFG = ModelConfig(dim=32, depth=2, heads=2, mlp_ratio=2.0, dropout=0.0, vocab_size=97)
N, T = 16, 6  # sequence length, last position that must not see the change

NORMALIZERS = {
    "softmax": lambda: Softmax(),
    "unbalanced-prefix": lambda: UnbalancedSinkhorn(rho=0.5, n_iters=4),
    "scaled-softmax": lambda: ScaledSoftmax(num_heads=2, base_len=16),
    "unbalanced-implicit": lambda: UnbalancedSinkhorn(
        rho=0.5, n_iters=12, grad_mode="implicit", backward_tol=1e-6
    ),
}


def _gpt(make_normalizer, checkpoint: bool = False) -> GPT:
    torch.manual_seed(0)
    cfg = ModelConfig(**{**CFG.__dict__, "checkpoint": checkpoint})
    return GPT(cfg, make_pe=lambda: RoPE(cfg.dim // cfg.heads), make_normalizer=make_normalizer)


def _past_logit_change(model: GPT) -> float:
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, CFG.vocab_size, (3, N), generator=g)
    changed = idx.clone()
    changed[:, T + 1 :] = torch.randint(0, CFG.vocab_size, (3, N - T - 1), generator=g)
    with torch.set_grad_enabled(model.training):  # checkpointing only acts in training
        a, b = model(idx)[:, : T + 1], model(changed)[:, : T + 1]
    return (a - b).abs().max().item()


@pytest.mark.parametrize("name", NORMALIZERS)
def test_future_tokens_do_not_change_past_logits(name: str) -> None:
    model = _gpt(NORMALIZERS[name]).eval()
    assert _past_logit_change(model) <= 1e-6


@pytest.mark.parametrize("name", NORMALIZERS)
def test_future_tokens_do_not_change_past_logits_with_checkpointing(name: str) -> None:
    model = _gpt(NORMALIZERS[name], checkpoint=True).train()  # dropout is 0
    assert _past_logit_change(model) <= 1e-6


def test_detector_catches_leaky_normalizer() -> None:
    """Counter-example control: full-column scaling under the causal mask leaks."""
    leaky = _gpt(lambda: UnbalancedSinkhorn(rho=0.5, n_iters=4, column_mass="full")).eval()
    assert _past_logit_change(leaky) > 1e-4


@pytest.mark.parametrize("name", NORMALIZERS)
def test_checkpointing_matches_plain(name: str) -> None:
    idx = torch.randint(0, CFG.vocab_size, (2, N), generator=torch.Generator().manual_seed(2))
    results = []
    for ckpt in (False, True):
        model = _gpt(NORMALIZERS[name], checkpoint=ckpt).train()
        logits = model(idx[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), idx[:, 1:].flatten())
        loss.backward()
        results.append((loss.detach(), {n: p.grad for n, p in model.named_parameters()}))
    (loss0, grads0), (loss1, grads1) = results
    torch.testing.assert_close(loss1, loss0, atol=1e-6, rtol=0)
    for n, g in grads0.items():
        torch.testing.assert_close(grads1[n], g, atol=1e-6, rtol=1e-5, msg=n)


def test_weight_tying_and_param_count() -> None:
    model = _gpt(NORMALIZERS["softmax"])
    assert model.head.weight is model.tok_embed.weight
    per_block = 4 * 32 * 32 + 3 * 32 + 32 + 2 * 32 * 64 + 64 + 32 + 4 * 32  # qkv/proj/mlp/norms
    assert model.num_params(embedding=False) == 2 * per_block + 2 * 32
    assert model.num_params() == model.num_params(embedding=False) + 97 * 32


def test_factory_refuses_leaky_normalizers_for_causal_models() -> None:

    ok = build_normalizer(NormalizerConfig(name="unbalanced", rho=0.3), causal=True)
    assert isinstance(ok, UnbalancedSinkhorn) and ok.column_mass == "prefix"
    for bad in (
        NormalizerConfig(name="sinkhorn"),
        NormalizerConfig(name="unbalanced", column_mass="full"),
    ):
        with pytest.raises(ValueError, match="leak"):
            build_normalizer(bad, causal=True)


def test_factory_builds_scaled_softmax_for_causal_models() -> None:
    norm = build_normalizer(
        NormalizerConfig(name="scaled_softmax", scale_base_len=64), causal=True, num_heads=3
    )
    assert isinstance(norm, ScaledSoftmax) and norm.scale.shape == (3, 1, 1)


@pytest.mark.parametrize("name", ["softmax", "unbalanced-prefix"])
def test_alibi_model_is_causal(name: str) -> None:
    """The additive action enters the scores before the mask: still no leak from the future."""
    torch.manual_seed(0)
    cfg = ModelConfig(**{**CFG.__dict__, "checkpoint": False})
    model = GPT(cfg, make_pe=lambda: ALiBi(cfg.heads), make_normalizer=NORMALIZERS[name]).eval()
    assert _past_logit_change(model) <= 1e-6


def test_logbias_model_is_causal() -> None:
    from geoattn_core.pe import ComposedPE, LogDistanceBias

    torch.manual_seed(0)
    cfg = ModelConfig(**{**CFG.__dict__, "checkpoint": False})
    model = GPT(
        cfg,
        make_pe=lambda: ComposedPE(RoPE(cfg.dim // cfg.heads), LogDistanceBias(cfg.heads)),
        make_normalizer=NORMALIZERS["softmax"],
    ).eval()
    assert _past_logit_change(model) <= 1e-6


@pytest.mark.parametrize(
    ("pe", "pe_bias"),
    [
        ("grape_m", "none"),
        ("grape_a", "none"),
        ("grape_a", "logbias"),
        ("grape_ap", "none"),
        ("fox", "none"),
        ("rope", "gated_logbias"),
        ("rope", "kerple"),
        ("rope", "clock_logbias"),
        ("cable", "none"),
    ],
)
def test_grape_model_is_causal(pe: str, pe_bias: str) -> None:
    """These biases read q, k or x (ADR-0011): row t must still only see tokens <= t."""
    from geoattn_train.config import AttentionConfig
    from geoattn_train.factory import build_pe

    torch.manual_seed(0)
    cfg = ModelConfig(**{**CFG.__dict__, "checkpoint": False})
    acfg = AttentionConfig(pe=pe, pe_bias=pe_bias)
    model = GPT(
        cfg,
        make_pe=lambda: build_pe(acfg, cfg.dim // cfg.heads, cfg.heads, dim=cfg.dim, causal=True),
        make_normalizer=NORMALIZERS["softmax"],
    ).eval()
    assert _past_logit_change(model) <= 1e-6


@pytest.mark.parametrize(("exponent", "margin"), [("gt1", 0.0), ("lt1", 0.0), ("gt1", 0.5)])
def test_constrained_logbias_model_is_causal(exponent: str, margin: float) -> None:
    """Experiments 0003 and 0004, gate G1: the constraints change only the coefficient."""
    from geoattn_core.pe import LogDistanceBias
    from geoattn_train.config import AttentionConfig
    from geoattn_train.factory import build_pe

    torch.manual_seed(0)
    cfg = ModelConfig(**{**CFG.__dict__, "checkpoint": False})
    acfg = AttentionConfig(
        pe="rope", pe_bias="logbias", logbias_exponent=exponent, logbias_margin=margin
    )
    model = GPT(
        cfg,
        make_pe=lambda: build_pe(acfg, cfg.dim // cfg.heads, cfg.heads, causal=True),
        make_normalizer=NORMALIZERS["softmax"],
    ).eval()
    logb = model.blocks[0].attn.pe.additive  # type: ignore[union-attr]
    assert isinstance(logb, LogDistanceBias) and logb.exponent == exponent
    assert logb.margin == margin and (exponent == "lt1" or (logb.coef > 1 + margin).all())
    assert _past_logit_change(model) <= 1e-6
