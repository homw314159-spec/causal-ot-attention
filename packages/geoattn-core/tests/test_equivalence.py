"""The project's anchor identity: Sinkhorn with one iteration IS softmax."""

import pytest
import torch

from geoattn_core import Sinkhorn, Softmax


@pytest.mark.parametrize("eps", [0.1, 1.0, 5.0])
def test_single_iteration_equals_softmax(eps: float) -> None:
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 7, 9) * 3
    torch.testing.assert_close(Sinkhorn(n_iters=1, eps=eps)(scores), Softmax(eps=eps)(scores))


def test_softmax_matches_torch() -> None:
    torch.manual_seed(1)
    scores = torch.randn(4, 5, 5)
    torch.testing.assert_close(Softmax()(scores), torch.softmax(scores, dim=-1))


def test_single_iteration_equals_softmax_with_mask() -> None:
    torch.manual_seed(2)
    scores = torch.randn(3, 6, 6)
    mask = torch.rand(3, 6, 6) > 0.3
    mask[..., 0] = True  # keep every row non-empty
    torch.testing.assert_close(Sinkhorn(n_iters=1)(scores, mask), Softmax()(scores, mask))


def test_output_dtype_preserved() -> None:
    scores = torch.randn(2, 4, 4, dtype=torch.bfloat16)
    assert Sinkhorn(n_iters=5)(scores).dtype == torch.bfloat16
    assert Softmax()(scores).dtype == torch.bfloat16
