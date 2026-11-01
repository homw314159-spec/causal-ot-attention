"""train.compile_normalizer: the compiled operator must be the same operator.

torch.compile fuses the per-iteration elementwise kernels of UnbalancedSinkhorn
(~5x faster, experiment 0001 change log) but may reorder float32 arithmetic, so
outputs and gradients are pinned to eager within float32 tolerance, and the
model-level causality gate G1 is re-checked with compilation on. CUDA + Triton only.
"""

import pytest
import torch

from geoattn_core import UnbalancedSinkhorn
from geoattn_train.config import load_config
from geoattn_train.lm_trainer import build_lm, compile_normalizer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def _mask(n: int, pad: int) -> torch.Tensor:
    valid = torch.arange(n) >= pad
    return (torch.ones(n, n, dtype=torch.bool).tril() & valid[:, None] & valid[None, :]).cuda()


@pytest.mark.parametrize("rho", [0.25, 0.9])
def test_compiled_matches_eager(rho: float) -> None:
    torch.manual_seed(0)
    n = 64
    scores = (torch.randn(2, 4, n, n, device="cuda") * 3).requires_grad_()
    weights = torch.randn(2, 4, n, n, device="cuda")
    eager = UnbalancedSinkhorn(rho=rho, n_iters=16)
    compiled = UnbalancedSinkhorn(rho=rho, n_iters=16)
    compile_normalizer(compiled)
    for pad in (0, 5):
        outs = []
        for op in (eager, compiled):
            scores.grad = None
            p = op(scores, _mask(n, pad))
            (p * weights).sum().backward()
            assert scores.grad is not None
            outs.append((p.detach(), scores.grad.clone()))
        (p0, g0), (p1, g1) = outs
        torch.testing.assert_close(p1, p0, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(g1, g0, atol=1e-5, rtol=1e-4)
        assert torch.isfinite(g1).all()


def test_compiled_gpt_is_still_causal() -> None:
    """Gate G1 at model level with compilation on (float32, no autocast)."""
    cfg = load_config(
        "configs/smoke_lm.yaml",
        ["train.compile_normalizer=true", "attention.normalizer.rho=0.9", "lm.seq_len=32"],
    )
    torch.manual_seed(0)
    model = build_lm(cfg).cuda().eval()
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, cfg.model.vocab_size, (3, 32), generator=g)
    changed = idx.clone()
    changed[:, 11:] = torch.randint(0, cfg.model.vocab_size, (3, 21), generator=g)
    with torch.no_grad():
        a, b = model(idx.cuda())[:, :11], model(changed.cuda())[:, :11]
    assert (a - b).abs().max().item() <= 1e-6
