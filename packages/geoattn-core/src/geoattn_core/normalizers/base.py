"""The HOW-MUCH axis: turning a score (negative cost) matrix into a coupling.

Contract shared by every normalizer (enforced by tests/test_properties.py):

* Input ``scores``: shape ``(..., n_q, n_k)``, any float dtype.
* Input ``mask``: optional bool tensor broadcastable to ``scores``;
  ``True`` = keep, ``False`` = masked out.
* Output: same shape and dtype as ``scores``.
  - masked entries are exactly 0;
  - every row with at least one unmasked entry sums to 1 (row-stochastic);
  - fully-masked rows are all zeros (never NaN), with finite gradients.
* Internal arithmetic runs in float32 in the log domain, regardless of the
  input dtype (bf16 Sinkhorn iterations are not accurate enough).
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Normalizer(Protocol):
    def __call__(self, scores: Tensor, mask: Tensor | None = None) -> Tensor: ...


def prepare_log_kernel(scores: Tensor, mask: Tensor | None, eps: float) -> Tensor:
    """``log K = scores / eps`` in float32, with masked entries set to ``-inf``."""
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    log_k = scores.float() / eps
    if mask is not None:
        log_k = log_k.masked_fill(~mask, -math.inf)
    return log_k


def log_normalize(log_p: Tensor, dim: int, log_target: Tensor | float = 0.0) -> Tensor:
    """Normalize ``exp(log_p)`` along ``dim`` so each slice sums to ``exp(log_target)``.

    Slices that are entirely ``-inf`` (fully masked) are left as ``-inf``
    instead of producing NaN, and — importantly — without NaN gradients:
    ``logsumexp`` is evaluated on a sanitized copy, because its backward
    computes ``exp(x - lse)``, which is ``exp(nan)`` on an all ``-inf`` slice
    even when the forward result is later discarded by ``torch.where``.
    """
    empty = ~torch.isfinite(log_p).any(dim=dim, keepdim=True)
    safe = log_p.masked_fill(empty, 0.0)
    lse = torch.logsumexp(safe, dim=dim, keepdim=True)
    return torch.where(empty, log_p, log_p - lse + log_target)
