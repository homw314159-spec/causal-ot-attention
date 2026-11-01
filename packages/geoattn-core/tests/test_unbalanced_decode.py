"""Token-by-token decoding of the prefix UnbalancedSinkhorn equals the full causal pass."""

import pytest
import torch

from geoattn_core import UnbalancedSinkhorn
from geoattn_core.normalizers.unbalanced_decode import PrefixMassCache, decode_row, prefill

CASES = [(0.0, 3, 1.0), (0.25, 7, 1.0), (0.5, 13, 0.7), (0.75, 20, 1.0), (0.5, 1, 1.0)]


def _full(op: UnbalancedSinkhorn, scores: torch.Tensor) -> torch.Tensor:
    n = scores.shape[-1]
    return op(scores, torch.ones(n, n, dtype=torch.bool).tril())


def _decode_all(op, scores, cache=None, start=0) -> torch.Tensor:
    rows = []
    for t in range(start + 1, scores.shape[-1] + 1):
        row, cache = decode_row(op, scores[..., t - 1 : t, :t], cache)
        rows.append(torch.nn.functional.pad(row, (0, scores.shape[-1] - t)))
    return torch.cat(rows, dim=-2)


@pytest.mark.parametrize(("rho", "k", "eps"), CASES)
def test_decoding_from_scratch_matches_the_full_pass(rho: float, k: int, eps: float) -> None:
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 24, 24) * 3
    op = UnbalancedSinkhorn(rho, n_iters=k, eps=eps)
    torch.testing.assert_close(_decode_all(op, scores), _full(op, scores), atol=1e-5, rtol=0)


@pytest.mark.parametrize(("rho", "k", "eps"), CASES)
def test_prefill_then_decode_matches_the_full_pass(rho: float, k: int, eps: float) -> None:
    torch.manual_seed(1)
    scores = torch.randn(2, 3, 30, 30) * 3
    op = UnbalancedSinkhorn(rho, n_iters=k, eps=eps)
    m = 17
    head, cache = prefill(op, scores[..., :m, :m])
    torch.testing.assert_close(head, _full(op, scores[..., :m, :m]), atol=1e-5, rtol=0)
    tail = _decode_all(op, scores, cache, start=m)
    torch.testing.assert_close(tail, _full(op, scores)[..., m:, :], atol=1e-5, rtol=0)


def test_cache_holds_k_minus_one_vectors_per_key() -> None:
    op = UnbalancedSinkhorn(0.5, n_iters=13)
    _, cache = prefill(op, torch.randn(1, 2, 10, 10))
    assert len(cache.log_mass) == 12 and cache.length == 10
    assert all(m.shape == (1, 2, 10) for m in cache.log_mass)
    _, cache1 = prefill(UnbalancedSinkhorn(0.5, n_iters=1), torch.randn(1, 2, 10, 10))
    assert cache1.log_mass == [] and cache1.length == 0


def test_stale_cache_is_detected() -> None:
    """Control: decoding with a cache that was not advanced gives a different row."""
    torch.manual_seed(2)
    scores = torch.randn(1, 2, 12, 12) * 3
    op = UnbalancedSinkhorn(0.5, n_iters=8)
    _, cache = prefill(op, scores[..., :10, :10])
    good, _ = decode_row(op, scores[..., 10:11, :11], cache)
    stale = PrefixMassCache([m[..., :9] for m in cache.log_mass])  # lost the last row
    bad, _ = decode_row(op, scores[..., 10:11, :11], stale)
    assert (good - bad).abs().max() > 1e-3


def test_rejects_full_column_mass_and_a_short_row() -> None:
    with pytest.raises(ValueError):
        decode_row(UnbalancedSinkhorn(0.5, column_mass="full"), torch.randn(1, 1, 1, 3), None)
    with pytest.raises(ValueError):
        decode_row(UnbalancedSinkhorn(0.5, column_target="uniform"), torch.randn(1, 1, 1, 3), None)
    op = UnbalancedSinkhorn(0.5)
    _, cache = prefill(op, torch.randn(1, 1, 4, 4))
    with pytest.raises(ValueError):
        decode_row(op, torch.randn(1, 1, 1, 4), cache)
