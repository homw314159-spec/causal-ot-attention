from __future__ import annotations

from torch import Tensor, nn

from geoattn_core.normalizers.base import log_normalize, prepare_log_kernel


class Softmax(nn.Module):
    """Row-wise softmax: the one-sided (row-marginal-only) entropic OT solution.

    Identical, by construction, to ``Sinkhorn(n_iters=1)``; that identity is
    the anchor of the whole project and is pinned by ``test_equivalence.py``.
    """

    def __init__(self, eps: float = 1.0) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, scores: Tensor, mask: Tensor | None = None) -> Tensor:
        log_p = log_normalize(prepare_log_kernel(scores, mask, self.eps), dim=-1)
        return log_p.exp().to(scores.dtype)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
