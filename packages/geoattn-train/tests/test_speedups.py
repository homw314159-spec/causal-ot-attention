"""train.compile_model and train.fused_optimizer change speed, not values."""

import sys

import pytest
import torch
from torch.nn import functional as F

from geoattn_train.config import load_config
from geoattn_train.lm_trainer import build_lm
from geoattn_train.trainer import _make_optimizer

BASE = [
    "task=lm",
    "model.dim=64",
    "model.depth=2",
    "model.heads=4",
    "model.vocab_size=101",
    "model.dropout=0.0",
    "lm.seq_len=64",
]


def _model(*over: str):
    torch.manual_seed(0)
    cfg = load_config(None, [*BASE, *over])
    return cfg, build_lm(cfg).cuda().train()


def _loss_and_grads(model, idx):
    logits = model(idx[:, :-1])
    loss = F.cross_entropy(logits.float().flatten(0, 1), idx[:, 1:].flatten())
    loss.backward()
    return loss.detach(), {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }


@pytest.mark.gpu
@pytest.mark.parametrize(
    "arm",
    [
        ["attention.pe=rope"],
        ["attention.pe=cable", "model.attn_backend=flex"],
        ["attention.pe=rope", "attention.pe_bias=alibi", "model.checkpoint=true"],
        [
            "attention.pe=rope",
            "attention.normalizer.name=unbalanced",
            "attention.normalizer.n_iters=4",
        ],
        pytest.param(
            [
                "attention.pe=rope",
                "attention.pe_bias=alibi",
                "attention.normalizer.name=unbalanced",
                "attention.normalizer.n_iters=13",
                "attention.normalizer.impl=fused",
            ],
            id="ot-fused",  # O2's setting; Inductor needs a C++ compiler for it (none on Windows)
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="no C++ compiler for Inductor"
            ),
        ),
    ],
    ids=["rope", "cable-flex", "alibi-checkpoint", "ot", "ot-fused"],
)
def test_compiled_blocks_match_eager(arm: list[str]) -> None:
    idx = torch.randint(0, 101, (2, 65), generator=torch.Generator().manual_seed(1)).cuda()
    _, eager = _model(*arm)
    _, compiled = _model(*arm, "train.compile_model=true")
    loss_e, grads_e = _loss_and_grads(eager, idx)
    loss_c, grads_c = _loss_and_grads(compiled, idx)
    torch.testing.assert_close(loss_c, loss_e, atol=1e-5, rtol=1e-5)
    for name, g in grads_e.items():
        scale = g.abs().max().clamp_min(1e-6)
        assert ((grads_c[name] - g).abs().max() / scale).item() < 1e-4, name


@pytest.mark.gpu
def test_fused_adamw_takes_the_same_step() -> None:
    idx = torch.randint(0, 101, (2, 65), generator=torch.Generator().manual_seed(2)).cuda()
    params = {}
    for fused in (False, True):
        cfg, model = _model("attention.pe=rope", f"train.fused_optimizer={str(fused).lower()}")
        opt = _make_optimizer(model, cfg)
        _loss_and_grads(model, idx)
        opt.step()
        params[fused] = {n: p.detach().clone() for n, p in model.named_parameters()}
    for name, p in params[False].items():
        torch.testing.assert_close(params[True][name], p, atol=1e-6, rtol=1e-5)
