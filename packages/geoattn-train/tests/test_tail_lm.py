"""Tail interventions at model level (experiment 0003, gate G2): a trained checkpoint loads
into the intervened model, whose logits are identical at the training length (the 1x loss
is unchanged) and change only at positions with keys beyond it."""

import pytest
import torch

from geoattn_core import Softmax
from geoattn_core.pe import TailIntervention
from geoattn_core.pe.alibi import alibi_slopes
from geoattn_train.config import AttentionConfig, ModelConfig
from geoattn_train.factory import build_pe
from geoattn_train.models import GPT

CFG = ModelConfig(dim=32, depth=2, heads=4, mlp_ratio=2.0, dropout=0.0, vocab_size=97)
L = 12


def _gpt(acfg: AttentionConfig) -> GPT:
    return GPT(
        CFG,
        make_pe=lambda: build_pe(
            acfg, CFG.dim // CFG.heads, CFG.heads, train_len=L, dim=CFG.dim, causal=True
        ),
        make_normalizer=Softmax,
    ).eval()


@pytest.mark.parametrize("mode", ["cap", "repair", "window"])
@pytest.mark.parametrize(("pe", "pe_bias"), [("rope", "logbias"), ("cable", "none")])
def test_checkpoint_loads_and_only_the_tail_changes(pe: str, pe_bias: str, mode: str) -> None:
    torch.manual_seed(0)
    trained = _gpt(AttentionConfig(pe=pe, pe_bias=pe_bias))
    with torch.no_grad():  # move every parameter away from its init
        for p in trained.parameters():
            p.add_(0.3 * torch.randn_like(p))
    model = _gpt(AttentionConfig(pe=pe, pe_bias=pe_bias, tail_mode=mode))
    model.load_state_dict(trained.state_dict(), strict=True)
    assert isinstance(model.blocks[0].attn.pe, TailIntervention)  # type: ignore[union-attr]

    idx = torch.randint(0, CFG.vocab_size, (2, 4 * L), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(model(idx[:, :L]), trained(idx[:, :L]))  # 1x: identical
        new, old = model(idx), trained(idx)
    assert torch.equal(new[:, : L + 1], old[:, : L + 1])  # rows t <= L see no key beyond L
    assert (new[:, L + 1 :] - old[:, L + 1 :]).abs().max() > 1e-4


def test_factory_defaults() -> None:
    pe = build_pe(AttentionConfig(pe="rope", tail_mode="window"), 8, 4, train_len=L)
    assert isinstance(pe, TailIntervention) and pe.tail_from == L
    geo_mean = alibi_slopes(4).log().mean().exp().item()
    assert pe.slope == pytest.approx(geo_mean)
    explicit = build_pe(
        AttentionConfig(pe="rope", tail_mode="cap", tail_from=30, tail_slope=0.2), 8, 4
    )
    assert isinstance(explicit, TailIntervention) and explicit.tail_from == 30
    assert not isinstance(build_pe(AttentionConfig(pe="rope"), 8, 4, train_len=L), TailIntervention)
    with pytest.raises(ValueError, match="tail_from"):
        build_pe(AttentionConfig(pe="rope", tail_mode="cap"), 8, 4)
