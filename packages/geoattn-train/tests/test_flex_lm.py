"""The GPT with ``model.attn_backend: flex`` (ADR-0012): same loss and gradients as the
reference backend, and still causal."""

import pytest
import torch
from torch.nn import functional as F

from geoattn_train.config import AttentionConfig, ModelConfig, NormalizerConfig
from geoattn_train.factory import build_normalizer, build_pe
from geoattn_train.models import GPT

CFG = ModelConfig(dim=64, depth=2, heads=4, mlp_ratio=2.0, dropout=0.0, vocab_size=101)
N = 128


def _gpt(backend: str, pe: str, pe_bias: str, checkpoint: bool = False) -> GPT:
    torch.manual_seed(0)
    cfg = ModelConfig(**{**CFG.__dict__, "attn_backend": backend, "checkpoint": checkpoint})
    acfg = AttentionConfig(pe=pe, pe_bias=pe_bias)
    return GPT(
        cfg,
        make_pe=lambda: build_pe(acfg, cfg.dim // cfg.heads, cfg.heads, dim=cfg.dim, causal=True),
        make_normalizer=lambda: build_normalizer(NormalizerConfig(), causal=True),
    ).cuda()


def _loss_and_grads(model: GPT, idx: torch.Tensor, bf16: bool) -> tuple[torch.Tensor, dict]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        logits = model(idx[:, :-1])
    loss = F.cross_entropy(logits.float().flatten(0, 1), idx[:, 1:].flatten())
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("pe", "pe_bias", "checkpoint"),
    [("rope", "logbias", False), ("rope", "clock_logbias", True), ("cable", "none", False)],
)
def test_flex_gpt_matches_reference(pe: str, pe_bias: str, checkpoint: bool) -> None:
    idx = torch.randint(0, CFG.vocab_size, (2, N + 1), generator=torch.Generator().manual_seed(1))
    idx = idx.cuda()
    fused = _gpt("flex", pe, pe_bias, checkpoint).train()
    assert all(b.attn.backend == "flex" for b in fused.blocks)  # type: ignore[union-attr]
    reference = _gpt("reference", pe, pe_bias, checkpoint).train()
    # Gradients in fp32, where the two backends agree to rounding (<= 5e-6 relative on the
    # 4090): in bf16 the rounding order alone moves a small parameter's gradient by several
    # percent, and by how much depends on the GPU (GB10 vs 4090).
    _, grads_f = _loss_and_grads(fused, idx, bf16=False)
    _, grads_r = _loss_and_grads(reference, idx, bf16=False)
    for name, g in grads_r.items():
        scale = g.abs().max().clamp_min(1e-6)
        assert ((grads_f[name] - g).abs().max() / scale).item() <= 1e-4, name
    # Training runs in bf16: the loss must still match there.
    loss_f, _ = _loss_and_grads(fused, idx, bf16=True)
    loss_r, _ = _loss_and_grads(reference, idx, bf16=True)
    torch.testing.assert_close(loss_f, loss_r, atol=2e-3, rtol=0)


@pytest.mark.gpu
def test_flex_gpt_is_causal() -> None:
    g = torch.Generator().manual_seed(2)
    idx = torch.randint(0, CFG.vocab_size, (2, N), generator=g)
    changed = idx.clone()
    changed[:, 70:] = torch.randint(0, CFG.vocab_size, (2, N - 70), generator=g)
    model = _gpt("flex", "rope", "clock_logbias").eval()
    with torch.no_grad():
        diff = (model(idx.cuda())[:, :70] - model(changed.cuda())[:, :70]).abs().max().item()
    assert diff <= 1e-5
