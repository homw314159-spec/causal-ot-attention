"""Incremental (token-by-token) evaluation of the prefix UnbalancedSinkhorn.

Prefix consistency (ADR-0005) makes the operator decodable like softmax attention: row t
needs the scores of query t against keys 1..t and, for each iteration i = 1..k-1, the
column mass of iteration i-1 over rows 1..t-1. ``log v`` is local to the row, so nothing
else is carried. The state is a :class:`PrefixMassCache`: k-1 log-mass vectors of length t
per (batch, head), next to the KV cache's 2 d values per key; for k = 13 and d = 64 that is
12 / 128 of the KV cache. Each decoded row costs k row normalizations instead of one.

Values match ``UnbalancedSinkhorn(column_mass="prefix")`` applied to the full causal
matrix (float32 log domain, the same update), which the tests check token by token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from geoattn_core.normalizers.base import log_normalize
from geoattn_core.normalizers.unbalanced import _LOG_ZERO, UnbalancedSinkhorn


@dataclass
class PrefixMassCache:
    """``log_mass[i]``: log column mass of iteration i (i = 0..k-2) over the rows seen so
    far, shape ``(..., t)``; an empty list for k = 1 (plain softmax needs no state)."""

    log_mass: list[Tensor]

    @property
    def length(self) -> int:
        return self.log_mass[0].shape[-1] if self.log_mass else 0


def _check(op: UnbalancedSinkhorn) -> None:
    if op.column_mass != "prefix":
        raise ValueError("incremental evaluation needs column_mass='prefix' (ADR-0005)")
    if op.column_target != "one":
        raise ValueError("incremental evaluation supports column_target='one' only (ADR-0010)")


def _pad(log_mass: Tensor, t: int) -> Tensor:
    """Extend to t keys; keys not seen by any earlier row have zero mass."""
    extra = t - log_mass.shape[-1]
    return torch.nn.functional.pad(log_mass, (0, extra), value=float("-inf"))


def decode_row(
    op: UnbalancedSinkhorn, scores: Tensor, cache: PrefixMassCache | None
) -> tuple[Tensor, PrefixMassCache]:
    """Attention weights of one new query over all ``t`` keys so far.

    ``scores``: ``(..., 1, t)``, every key visible (causal decoding). Returns the row
    ``(..., 1, t)`` in ``scores.dtype`` and the cache extended to ``t`` keys.
    """
    _check(op)
    log_k = scores.float() / op.eps
    t = log_k.shape[-1]
    if cache is None:
        lead = log_k.shape[:-2]
        cache = PrefixMassCache(
            [log_k.new_full((*lead, 0), float("-inf")) for _ in range(op.n_iters - 1)]
        )
    if cache.length >= t:
        raise ValueError(f"cache already covers {cache.length} keys, row has {t}")
    log_v = torch.zeros((), dtype=log_k.dtype, device=log_k.device)
    log_p = log_normalize(log_k, dim=-1)
    new_mass = []
    for i in range(op.n_iters - 1):
        log_m = torch.logaddexp(_pad(cache.log_mass[i], t)[..., None, :], log_p)
        new_mass.append(log_m[..., 0, :])
        log_v = op.rho * (log_v - log_m)
        log_v = log_v.masked_fill(log_m < _LOG_ZERO / 2, 0.0)
        log_p = log_normalize(log_k + log_v, dim=-1)
    return log_p.exp().to(scores.dtype), PrefixMassCache(new_mass)


def prefill(op: UnbalancedSinkhorn, scores: Tensor) -> tuple[Tensor, PrefixMassCache]:
    """Full causal matrix ``(..., n, n)`` at once (the training path) plus the cache that
    decoding row n + 1 needs: each iteration's column mass summed over all n rows."""
    _check(op)
    n = scores.shape[-1]
    mask = torch.ones(n, n, dtype=torch.bool, device=scores.device).tril()
    log_k = scores.float() / op.eps
    log_k = log_k.masked_fill(~mask, float("-inf"))
    log_v = torch.zeros((), dtype=log_k.dtype, device=log_k.device)
    log_p = log_normalize(log_k, dim=-1)
    totals = []
    for _ in range(op.n_iters - 1):
        safe = log_p.masked_fill(~torch.isfinite(log_p), _LOG_ZERO)
        log_m = torch.logcumsumexp(safe, dim=-2)
        totals.append(log_m[..., -1, :])
        log_v = op._update(log_v, 0.0, log_m)
        log_p = log_normalize(log_k + log_v, dim=-1)
    return log_p.exp().to(scores.dtype), PrefixMassCache(totals)
