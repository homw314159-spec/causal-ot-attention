"""Masks: padding safety, and the causal-mask theorem of docs/methods.md §4.3."""

import pytest
import torch

from geoattn_core import ScaledSoftmax, Sinkhorn, Softmax, UnbalancedSinkhorn

NORMALIZERS = [
    Softmax(),
    Sinkhorn(n_iters=1),
    Sinkhorn(n_iters=5),
    ScaledSoftmax(base_len=8),
    UnbalancedSinkhorn(rho=0.5, n_iters=5),
    UnbalancedSinkhorn(rho=0.5, n_iters=5, column_mass="full"),
    UnbalancedSinkhorn(rho=0.5, n_iters=5, column_target="uniform"),
]


@pytest.mark.parametrize("norm", NORMALIZERS, ids=repr)
def test_masked_entries_are_exactly_zero(norm) -> None:
    torch.manual_seed(0)
    scores = torch.randn(2, 6, 6)
    mask = torch.rand(2, 6, 6) > 0.4
    mask[..., 0] = True
    p = norm(scores, mask)
    assert (p[~mask] == 0).all()


@pytest.mark.parametrize("norm", NORMALIZERS, ids=repr)
def test_padding_rows_and_columns_give_no_nan(norm) -> None:
    """Last two tokens are padding: their rows AND columns are fully masked."""
    n, pad = 6, 2
    valid = torch.arange(n) < n - pad
    mask = valid[:, None] & valid[None, :]
    scores = torch.randn(1, n, n, requires_grad=True)

    p = norm(scores, mask)
    assert torch.isfinite(p).all()
    assert (p[..., n - pad :, :] == 0).all(), "fully masked rows must be zero"
    p.sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_padding_matches_unpadded_computation() -> None:
    """Sinkhorn on a padded matrix equals Sinkhorn on the valid sub-matrix."""
    torch.manual_seed(1)
    n, n_valid = 7, 4
    scores = torch.randn(1, n, n)
    valid = torch.arange(n) < n_valid
    mask = valid[:, None] & valid[None, :]
    padded = Sinkhorn(n_iters=6)(scores, mask)[..., :n_valid, :n_valid]
    direct = Sinkhorn(n_iters=6)(scores[..., :n_valid, :n_valid])
    torch.testing.assert_close(padded, direct)


def test_causal_mask_forces_identity() -> None:
    """Theorem: the only doubly stochastic lower-triangular matrix is I.

    Numerically, Sinkhorn under a causal mask drifts to the identity — and
    only sublinearly (the support has no total support off the diagonal), so
    this needs many iterations. This is why strict doubly stochastic
    attention is unusable for decoder-only LMs (paths A/B/C relax it).
    """
    torch.manual_seed(0)
    n = 6
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    p = Sinkhorn(n_iters=1000)(torch.randn(1, n, n), causal)
    assert p[0].diagonal().min() > 0.99


@pytest.mark.parametrize("column_mass", ["prefix", "full"])
def test_unbalanced_causal_with_left_padding(column_mass) -> None:
    """Left padding + causal mask: the first rows are empty and the padded columns are
    never visible, so every column-mass prefix over them is log 0 — no NaN, fwd or bwd."""
    n, pad = 7, 3
    valid = torch.arange(n) >= pad
    mask = torch.ones(n, n, dtype=torch.bool).tril() & valid[:, None] & valid[None, :]
    scores = torch.randn(2, n, n, requires_grad=True)

    p = UnbalancedSinkhorn(rho=0.7, n_iters=6, column_mass=column_mass)(scores, mask)
    assert torch.isfinite(p).all()
    assert (p[~mask.expand_as(p)] == 0).all()
    torch.testing.assert_close(p[:, pad:].sum(-1), torch.ones(2, n - pad))
    (p * torch.randn_like(p)).sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
