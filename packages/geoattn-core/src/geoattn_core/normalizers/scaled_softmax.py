"""Softmax with a length-aware temperature (Scalable-Softmax, Nakanishi 2025).

A row that sees ``n`` keys has its logits multiplied by ``s_h * log(n)`` before the
softmax, with a learnable scalar ``s_h`` per head. Plain softmax flattens as ``n``
grows (attention entropy grows like log n), which is one known cause of poor length
extrapolation; the log-n factor counteracts it. Experiment 0001 uses it as the
control that a length-extrapolation gain of path C must beat (report 0005).

``n`` is counted from the mask row itself, so row t depends on row t only: the
operator is prefix-consistent under a causal mask (ADR-0005). Without a mask every
row sees all ``n_k`` keys. ``s`` starts at ``1 / log(base_len)``: at the training
length the last row starts with factor 1, i.e. as plain softmax.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from geoattn_core.normalizers.base import log_normalize, prepare_log_kernel


class ScaledSoftmax(nn.Module):
    def __init__(
        self,
        num_heads: int = 1,
        base_len: int = 512,
        eps: float = 1.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if base_len < 2:
            raise ValueError(f"base_len must be >= 2, got {base_len}")
        self.eps = eps
        self.base_len = base_len
        # Shape (H, 1, 1) broadcasts against scores (..., H, n_q, n_k).
        init = torch.full((num_heads, 1, 1), 1.0 / math.log(base_len))
        if learnable:
            self.scale = nn.Parameter(init)
        else:
            self.register_buffer("scale", init)

    def forward(self, scores: Tensor, mask: Tensor | None = None) -> Tensor:
        log_n = self._log_visible_keys(scores, mask)
        factor = self.scale * log_n  # (H, n_q, 1) or broadcastable to it
        log_p = log_normalize(prepare_log_kernel(scores.float() * factor, mask, self.eps), dim=-1)
        return log_p.exp().to(scores.dtype)

    @staticmethod
    def _log_visible_keys(scores: Tensor, mask: Tensor | None) -> Tensor:
        """``log(n)`` per row, shape (..., n_q, 1); empty rows count as 1 (log 1 = 0)."""
        if mask is None:
            n_k = scores.shape[-1]
            return torch.full((scores.shape[-2], 1), math.log(n_k), device=scores.device)
        n = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        return n.to(torch.float32).log()

    def extra_repr(self) -> str:
        learnable = isinstance(self.scale, nn.Parameter)
        heads = self.scale.shape[0]
        return f"heads={heads}, base_len={self.base_len}, eps={self.eps}, learnable={learnable}"
