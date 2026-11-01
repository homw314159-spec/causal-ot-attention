"""ScaledSoftmax: identities that pin the log-n scaling (report 0005, experiment 0001)."""

import math

import pytest
import torch

from geoattn_core import ScaledSoftmax, Softmax


def _causal(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool).tril()


def test_constant_row_length_is_a_softmax_temperature() -> None:
    """No mask: every row sees n_k keys, so it is softmax at eps / (s log n_k)."""
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 5, 9)
    norm = ScaledSoftmax(num_heads=3, base_len=64, eps=0.8)
    s = 1 / math.log(64)
    expected = Softmax(eps=0.8 / (s * math.log(9)))(scores)
    torch.testing.assert_close(norm(scores), expected)


def test_causal_rows_use_their_own_key_count() -> None:
    """Row t (0-based) of a causal mask sees t + 1 keys and is scaled by s log(t + 1)."""
    torch.manual_seed(1)
    n, s = 7, 0.3
    scores = torch.randn(1, 1, n, n)
    norm = ScaledSoftmax(base_len=16)
    with torch.no_grad():
        norm.scale.fill_(s)
    p = norm(scores, _causal(n))
    for t in range(n):
        factor = s * math.log(t + 1)
        row = torch.softmax(scores[0, 0, t, : t + 1] * factor, dim=-1)
        torch.testing.assert_close(p[0, 0, t, : t + 1], row)
    assert p[0, 0, 0, 0] == 1.0  # one visible key: log 1 = 0, all mass on it


def test_scale_is_learnable_per_head() -> None:
    torch.manual_seed(2)
    norm = ScaledSoftmax(num_heads=4, base_len=32)
    scores = torch.randn(2, 4, 6, 6)
    (norm(scores, _causal(6)) * torch.randn(2, 4, 6, 6)).sum().backward()
    assert norm.scale.grad is not None and norm.scale.grad.shape == (4, 1, 1)
    assert (norm.scale.grad.flatten() != 0).all()
    fixed = ScaledSoftmax(num_heads=4, base_len=32, learnable=False)
    assert not list(fixed.parameters())


def test_initial_factor_is_one_at_the_base_length() -> None:
    norm = ScaledSoftmax(base_len=512)
    assert norm.scale.item() * math.log(512) == pytest.approx(1.0)


def test_output_dtype_preserved() -> None:
    scores = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16)
    assert ScaledSoftmax(num_heads=2)(scores, _causal(4)).dtype == torch.bfloat16


def test_invalid_base_len() -> None:
    with pytest.raises(ValueError):
        ScaledSoftmax(base_len=1)
