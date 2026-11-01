"""The GPT with ``normalizer.impl=fused`` (ADR-0009) against the reference implementation.

Operator-level equivalence is tested in geoattn-kernels; here the kernel sits inside the
model: GPT masks, RoPE, bf16 autocast and activation checkpointing around it.
"""

import pytest
import torch
from torch.nn import functional as F

from geoattn_core import ALiBi, RoPE, UnbalancedSinkhorn
from geoattn_core.pe import ForgetGate
from geoattn_kernels import FusedUnbalancedSinkhorn
from geoattn_train.config import ModelConfig, NormalizerConfig
from geoattn_train.factory import build_normalizer
from geoattn_train.models import GPT, GPTBlock

CFG = ModelConfig(dim=64, depth=2, heads=4, mlp_ratio=2.0, dropout=0.0, vocab_size=97)
N = 96


def _cfg(**kw) -> NormalizerConfig:
    return NormalizerConfig(**{"name": "unbalanced", "rho": 0.5, "n_iters": 6, **kw})


def test_factory_builds_fused() -> None:
    norm = build_normalizer(_cfg(impl="fused"), causal=True)
    assert isinstance(norm, FusedUnbalancedSinkhorn)
    assert (norm.rho, norm.n_iters, norm.eps) == (0.5, 6, 1.0)


@pytest.mark.parametrize(
    "kw",
    [
        {"impl": "cuda"},
        {"impl": "fused", "grad_mode": "implicit"},
        {"impl": "fused", "column_mass": "full"},
    ],
)
def test_factory_rejects_uncovered_fused_settings(kw: dict) -> None:
    with pytest.raises(ValueError):
        build_normalizer(_cfg(**kw), causal=False)


def _pe(name: str, cfg: ModelConfig):
    match name:
        case "rope":
            return RoPE(cfg.dim // cfg.heads)
        case "alibi":
            return ALiBi(cfg.heads)
        case _:  # a content-dependent bias: its gradient flows back through the kernel
            return ForgetGate(cfg.dim, cfg.heads)


def _gpt(impl: str, checkpoint: bool, pe: str = "rope") -> GPT:
    torch.manual_seed(0)
    cfg = ModelConfig(**{**CFG.__dict__, "checkpoint": checkpoint})
    return GPT(
        cfg,
        make_pe=lambda: _pe(pe, cfg),
        make_normalizer=lambda: build_normalizer(_cfg(impl=impl), causal=True),
    ).cuda()


def _loss_and_grads(model: GPT, idx: torch.Tensor) -> tuple[torch.Tensor, dict]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(idx[:, :-1])
    loss = F.cross_entropy(logits.float().flatten(0, 1), idx[:, 1:].flatten())
    loss.backward()
    grads = {}
    for n, p in model.named_parameters():
        assert p.grad is not None, n
        grads[n] = p.grad.clone()
    return loss.detach(), grads


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("checkpoint", "pe"), [(False, "rope"), (True, "rope"), (False, "alibi"), (False, "fox")]
)
def test_fused_model_matches_reference(checkpoint: bool, pe: str) -> None:
    idx = torch.randint(0, CFG.vocab_size, (3, N + 1), generator=torch.Generator().manual_seed(1))
    idx = idx.cuda()
    fused = _gpt("fused", checkpoint, pe).train()
    blocks = [b for b in fused.blocks if isinstance(b, GPTBlock)]
    assert len(blocks) == CFG.depth
    assert all(isinstance(b.attn.normalizer, FusedUnbalancedSinkhorn) for b in blocks)
    loss_f, grads_f = _loss_and_grads(fused, idx)
    loss_r, grads_r = _loss_and_grads(_gpt("torch", checkpoint, pe).train(), idx)
    # bf16 autocast around identical fp32 normalizer math: only rounding-order differences.
    torch.testing.assert_close(loss_f, loss_r, atol=2e-3, rtol=0)
    for name, g in grads_r.items():
        scale = g.abs().max().clamp_min(1e-6)
        assert (grads_f[name] - g).abs().max() / scale <= 5e-2, name


@pytest.mark.gpu
def test_fused_model_is_causal() -> None:
    """Changing future tokens must not change past logits; the leaky model is the control."""
    g = torch.Generator().manual_seed(2)
    idx = torch.randint(0, CFG.vocab_size, (2, N), generator=g)
    changed = idx.clone()
    changed[:, 40:] = torch.randint(0, CFG.vocab_size, (2, N - 40), generator=g)
    idx, changed = idx.cuda(), changed.cuda()

    def past_change(model: GPT) -> float:
        with torch.no_grad():
            return (model(idx)[:, :40] - model(changed)[:, :40]).abs().max().item()

    assert past_change(_gpt("fused", checkpoint=False).eval()) <= 1e-5
    torch.manual_seed(0)
    leaky = GPT(
        CFG,
        make_pe=lambda: RoPE(CFG.dim // CFG.heads),
        make_normalizer=lambda: UnbalancedSinkhorn(0.5, n_iters=6, column_mass="full"),
    ).cuda()
    assert past_change(leaky.eval()) > 1e-4
