"""GRAPE (Zhang et al., ICLR 2026, arXiv:2512.07805): group-representational encodings.

Implemented to match the official code (``model-architectures/GRAPE`` @ ``f5b539f``):

* :class:`GrapeM` — multiplicative GRAPE, the commuting (non-contextual) variant: RoPE's
  canonical rotation planes with *learnable* per-plane frequencies, shared across heads,
  stored as ``log_freq * log_freq_scale`` (the official learning-rate trick, scale 16).
  The one-parameter group law ``G(n + m) = G(n) G(m)`` keeps scores relative.
* :class:`GrapeA` — additive GRAPE, query-key gated: a linear distance penalty whose
  slope depends on content, ``b(i, j) = -(i - j) softplus(w_h) m_h (softplus(v_h q_i / sqrt
  D) + softplus(u_h k_j / sqrt D))`` with ALiBi slopes ``m_h`` and unit vectors ``u_h,
  v_h``. The gates read RMS-normalized q and k (the official model applies QK-RMSNorm
  first; this repository's attention does not, so the encoding normalizes for the gates
  only). It needs q and k, so it implements ``bias_ctx`` (ADR-0011).

Rotation direction: the official code rotates by ``-angle``; this repository's RoPE by
``+angle``. Both are relative encodings; they differ by reversing positions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geoattn_core.flex import ScoreMod, per_query
from geoattn_core.pe.alibi import alibi_slopes
from geoattn_core.pe.rope import _rotate


def at_least_f32(t: Tensor) -> Tensor:
    """Compute gates and biases in float32 under autocast, but keep float64 for gradcheck."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


class GrapeM(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
        learnable: bool = True,
        log_freq_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"GRAPE-M needs an even head_dim, got {head_dim}")
        if log_freq_scale <= 0:
            raise ValueError(f"log_freq_scale must be positive, got {log_freq_scale}")
        self.head_dim = head_dim
        self.log_freq_scale = log_freq_scale
        inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.log_freq = nn.Parameter(inv_freq.log() * log_freq_scale, requires_grad=learnable)

    @property
    def freq(self) -> Tensor:
        return torch.exp(self.log_freq.float() / self.log_freq_scale)

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        angles = positions.to(torch.float32)[..., None] * self.freq
        if angles.ndim == 3:  # (B, N, d/2): add the head axis
            angles = angles[:, None]
        cos, sin = angles.cos(), angles.sin()
        return _rotate(q, cos, sin), _rotate(k, cos, sin)

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None:
        return None

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        return None

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, learnable={self.log_freq.requires_grad}"


class GrapeA(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, omega_init: float = 1.0) -> None:
        super().__init__()
        if omega_init <= 0:
            raise ValueError(f"omega_init must be positive, got {omega_init}")
        self.num_heads, self.head_dim = num_heads, head_dim
        self.register_buffer("slopes", alibi_slopes(num_heads), persistent=False)
        self.slopes: Tensor
        self.omega = nn.Parameter(torch.full((num_heads,), math.log(math.expm1(omega_init))))
        self.u = nn.Parameter(torch.randn(num_heads, head_dim) / math.sqrt(head_dim))
        self.v = nn.Parameter(torch.randn(num_heads, head_dim) / math.sqrt(head_dim))

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return q, k

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None:
        raise TypeError("GrapeA depends on q and k: attention calls bias_ctx (ADR-0011)")

    def gates(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Per-token gates ``(B, H, N)`` for queries and keys; q, k are ``(B, H, N, D)``."""
        qn = F.rms_norm(at_least_f32(q), (q.shape[-1],))
        kn = F.rms_norm(at_least_f32(k), (k.shape[-1],))
        v = F.normalize(at_least_f32(self.v), dim=-1)[None, :, None, :]
        u = F.normalize(at_least_f32(self.u), dim=-1)[None, :, None, :]
        scale = math.sqrt(self.head_dim)
        return F.softplus((qn * v).sum(-1) / scale), F.softplus((kn * u).sum(-1) / scale)

    def bias_ctx(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> Tensor:
        """``(B, H, N, N)``; future entries (j > i) are 0 here and masked by the caller."""
        gate_q, gate_k = self.gates(q, k)
        pos = positions.to(torch.float32)
        dist = (pos[..., :, None] - pos[..., None, :]).clamp_min(0)
        if dist.ndim == 2:
            dist = dist[None]
        rate = F.softplus(at_least_f32(self.omega)) * self.slopes  # (H,)
        return (
            -dist[:, None] * rate[None, :, None, None] * (gate_q[..., None] + gate_k[..., None, :])
        )

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        gate_q, gate_k = self.gates(q, k)
        rate = per_query(F.softplus(at_least_f32(self.omega)) * self.slopes, q)
        return lambda s, b, h, qi, kv: (
            s - (qi - kv).float() * rate[b, h, qi] * (gate_q[b, h, qi] + gate_k[b, h, kv])
        )

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}"
