"""Prefix consistency of causal normalizers (ADR-0005; gate G1 of experiment 0001).

A normalizer used under a causal mask must satisfy

    op(S)[:t, :t] == op(S[:t, :t])    for every t,

i.e. row t sees only rows 1..t. Every causal operator goes into
``CAUSAL_NORMALIZERS``. ``LEAKY`` — full-column unbalanced scaling, the original
path-C definition — is the known-bad control: if the detector tests stop
flagging it, they have lost their power.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core import ScaledSoftmax, Softmax, UnbalancedSinkhorn

CAUSAL_NORMALIZERS = [
    Softmax(),
    Softmax(eps=0.7),
    ScaledSoftmax(base_len=16),
    ScaledSoftmax(base_len=4, eps=0.7),
    UnbalancedSinkhorn(rho=0.0, n_iters=5),
    UnbalancedSinkhorn(rho=0.25, n_iters=5),
    UnbalancedSinkhorn(rho=0.5, n_iters=5),
    UnbalancedSinkhorn(rho=0.9, n_iters=10, eps=0.7),
    UnbalancedSinkhorn(rho=0.5, n_iters=5, column_target="uniform"),
    UnbalancedSinkhorn(rho=0.75, n_iters=8, column_target="uniform"),
]
LEAKY = UnbalancedSinkhorn(rho=0.5, n_iters=5, column_mass="full")


def _causal(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool).tril()


def _prefix_gap(norm, scores: torch.Tensor, t: int) -> float:
    n = scores.shape[-1]
    full = norm(scores, _causal(n))[..., :t, :t]
    prefix = norm(scores[..., :t, :t], _causal(t))
    return (full - prefix).abs().max().item()


def _max_future_grad(norm, n: int = 8, row: int = 2, seed: int = 0) -> float:
    """max |d P[row] / d S[i]| over rows i > row: 0 for a prefix-consistent operator."""
    g = torch.Generator().manual_seed(seed)
    scores = (torch.randn(1, n, n, generator=g) * 2).requires_grad_()
    p = norm(scores, _causal(n))
    weights = torch.randn(n, generator=g)
    (p[0, row] * weights).sum().backward()
    assert scores.grad is not None
    return scores.grad[0, row + 1 :].abs().max().item()


@settings(max_examples=40, deadline=None)
@given(n=st.integers(1, 16), data=st.data())
@pytest.mark.parametrize("norm", CAUSAL_NORMALIZERS, ids=repr)
def test_prefix_consistency(norm, n: int, data) -> None:
    t = data.draw(st.integers(1, n))
    seed = data.draw(st.integers(0, 2**31 - 1))
    scores = torch.randn(2, n, n, generator=torch.Generator().manual_seed(seed)) * 3
    assert _prefix_gap(norm, scores, t) <= 1e-6


@pytest.mark.parametrize("norm", CAUSAL_NORMALIZERS, ids=repr)
@pytest.mark.parametrize("row", [0, 2, 6])
def test_no_gradient_from_future_rows(norm, row: int) -> None:
    assert _max_future_grad(norm, row=row) == 0.0


def test_detector_catches_known_leak() -> None:
    """Counter-example control: both detectors must flag the full-column operator."""
    assert _max_future_grad(LEAKY) > 1e-4
    scores = torch.randn(2, 8, 8, generator=torch.Generator().manual_seed(0)) * 3
    assert _prefix_gap(LEAKY, scores, t=3) > 1e-4
