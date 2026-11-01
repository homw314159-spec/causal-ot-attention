"""Sinkhorn normalization (Sander et al., Sinkformers, AISTATS 2022).

Solves the entropic OT problem between queries and keys with uniform
marginals, by alternating row and column normalization of ``K = exp(S/eps)``
in the log domain.

Iteration convention (ADR-0002)::

    step 1                : row-normalize          -> equals Softmax
    steps 2 .. n_iters    : column-normalize, then row-normalize

So ``n_iters`` counts row normalizations, ``n_iters=1`` is exactly softmax,
and the output is always exactly row-stochastic (each output token is still
a convex combination of values); the column marginal is only approximately
matched and converges as ``n_iters`` grows.

Column target: rows sum to 1 and total mass is ``n_q``, so each of the
``n_k`` valid columns targets ``n_q / n_k`` (= 1 for square self-attention).
With masks, ``n_q`` / ``n_k`` are the counts of non-empty rows / columns.

Gradients flow through the unrolled iterations (plain autograd). Implicit
differentiation at the fixed point is deferred to P5.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from geoattn_core.normalizers.base import log_normalize, prepare_log_kernel


def column_log_target(log_k: Tensor) -> Tensor:
    """``log(n_q / n_k)`` per matrix, shape ``(..., 1, 1)``, from the finite pattern."""
    valid = torch.isfinite(log_k)
    n_rows = valid.any(dim=-1).sum(dim=-1).clamp_min(1).to(log_k.dtype)
    n_cols = valid.any(dim=-2).sum(dim=-1).clamp_min(1).to(log_k.dtype)
    return (n_rows.log() - n_cols.log())[..., None, None]


class Sinkhorn(nn.Module):
    def __init__(self, n_iters: int = 3, eps: float = 1.0) -> None:
        super().__init__()
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        self.n_iters = n_iters
        self.eps = eps

    def forward(self, scores: Tensor, mask: Tensor | None = None) -> Tensor:
        log_p = prepare_log_kernel(scores, mask, self.eps)
        col_target = column_log_target(log_p) if self.n_iters > 1 else 0.0
        log_p = log_normalize(log_p, dim=-1)
        for _ in range(self.n_iters - 1):
            log_p = log_normalize(log_p, dim=-2, log_target=col_target)
            log_p = log_normalize(log_p, dim=-1)
        return log_p.exp().to(scores.dtype)

    def extra_repr(self) -> str:
        return f"n_iters={self.n_iters}, eps={self.eps}"
