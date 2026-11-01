"""Pure metric functions. Inputs are batches of matrices ``(..., n, m)``;
each function returns a scalar tensor averaged over all leading dims.
All computation is done in float32 on detached inputs.
"""

from __future__ import annotations

import torch
from torch import Tensor


def row_marginal_error(p: Tensor) -> Tensor:
    """Mean over matrices of max_i |sum_j P_ij - 1|."""
    p = p.detach().float()
    return (p.sum(-1) - 1).abs().amax(-1).mean()


def column_marginal_error(p: Tensor) -> Tensor:
    """Mean over matrices of max_j |sum_i P_ij - n_q/n_k|. Zero iff OT-balanced."""
    p = p.detach().float()
    n_q, n_k = p.shape[-2:]
    return (p.sum(-2) - n_q / n_k).abs().amax(-1).mean()


def row_entropy(p: Tensor) -> Tensor:
    """Mean Shannon entropy (nats) of the rows. Entropy collapse -> 0."""
    p = p.detach().float()
    return -(p * p.clamp_min(1e-12).log()).sum(-1).mean()


def effective_rank(x: Tensor) -> Tensor:
    """Roy & Vetterli (2007): exp(H(σ / Σσ)). Identity -> n, rank-1 -> 1."""
    s = torch.linalg.svdvals(x.detach().float())
    q = s / s.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.exp(-(q * q.clamp_min(1e-12).log()).sum(-1)).mean()


def rank_collapse_residual(x: Tensor) -> Tensor:
    """Relative distance of token matrix X (…, N, D) from rank-1 ``1 xᵀ``.

    ||X - 1·mean(X)ᵀ||_F / ||X||_F  (Dong et al. 2021, relative form; the
    minimizing x is the token mean). -> 0 means all tokens collapsed.
    """
    x = x.detach().float()
    resid = x - x.mean(dim=-2, keepdim=True)
    num = torch.linalg.matrix_norm(resid)
    den = torch.linalg.matrix_norm(x).clamp_min(1e-12)
    return (num / den).mean()


def first_token_mass(p: Tensor) -> Tensor:
    """Share of the total attention mass that lands on key 0 (the attention sink).

    sum_i P_i0 / sum_ij P_ij, averaged over matrices. Softmax under a causal mask
    with uniform scores gives H_n / n (key 0 is the only key every query sees);
    the identity gives 1 / n. Path C's prefix column scaling is hypothesized to
    lower it (experiment 0001).
    """
    p = p.detach().float()
    return (p[..., :, 0].sum(-1) / p.sum((-2, -1)).clamp_min(1e-12)).mean()
