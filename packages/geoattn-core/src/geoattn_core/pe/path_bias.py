"""Path-integral additive biases: ``b(t, j) = sum_{l=j+1}^{t} psi(t, l)``, causal only.

* :class:`GrapeAP` — GRAPE-AP v1 as in the official code (``model-architectures/GRAPE`` @
  ``f5b539f``, ``llama-mha-grape-ap-v1.py``): ``psi_h(t, l) = m_h log sigmoid(<p_t,
  R_l rms(p_l)> / d_p)`` with ALiBi slopes ``m_h``, a separate per-head projection ``p``
  of the layer input (``d_p = 8`` in the paper's runs), and ``R_l`` the rotation by
  angle ``l`` in every plane (rotary base 1). ``p_t`` itself is neither normalized nor
  rotated, so ``psi`` depends on the endpoint ``t`` and on the absolute position ``l``:
  the encoding is not relative.
* :class:`Cable` — CABLE: a content clock with a per-query weight (see the class).
* :class:`ForgetGate` — the Forgetting Transformer's (FoX, Lin et al., ICLR 2025)
  forget gate: ``psi(t, l) = log sigmoid(w_h x_l + b_h)``, the endpoint-independent case,
  so ``b(t, j) = c_t - c_j`` with ``c`` the cumulative sum of the log gates.

All three need the layer input ``x`` and implement ``bias_ctx`` (ADR-0011). Entries with
``j > t`` are 0 and must be masked by the caller: none of the encodings defines them.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geoattn_core.flex import ScoreMod
from geoattn_core.pe.alibi import alibi_slopes
from geoattn_core.pe.grape import at_least_f32


class _ContentBias(nn.Module):
    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return q, k

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None:
        raise TypeError(f"{type(self).__name__} depends on x: attention calls bias_ctx")


def _lower(n: int, device: torch.device) -> Tensor:
    return torch.ones(n, n, dtype=torch.bool, device=device).tril()


class GrapeAP(_ContentBias):
    def __init__(self, dim: int, num_heads: int, p_dim: int = 8) -> None:
        super().__init__()
        if p_dim % 2:
            raise ValueError(f"p_dim must be even (rotation planes), got {p_dim}")
        self.num_heads, self.p_dim = num_heads, p_dim
        self.proj = nn.Linear(dim, num_heads * p_dim, bias=False)
        self.register_buffer("slopes", alibi_slopes(num_heads), persistent=False)
        self.slopes: Tensor

    def bias_ctx(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> Tensor:
        bsz, n, _ = x.shape
        p = self.proj(x).view(bsz, n, self.num_heads, self.p_dim).transpose(1, 2)
        p = at_least_f32(p)
        angle = positions.to(torch.float32)[..., None]  # base 1: one frequency for all planes
        if angle.ndim == 3:
            angle = angle[:, None]
        cos, sin = angle.cos(), angle.sin()
        # As ``apply_rotary_emb`` there: half-split planes, rotation by -angle.
        p1, p2 = F.rms_norm(p, (self.p_dim,)).chunk(2, dim=-1)
        p_rot = torch.cat((p1 * cos + p2 * sin, p2 * cos - p1 * sin), dim=-1)
        psi = F.logsigmoid(p @ p_rot.transpose(-2, -1) / self.p_dim)
        future = ~_lower(n, x.device)
        psi = (psi * self.slopes[:, None, None]).masked_fill(future, 0.0)
        # sum over l in (j, t] = row total - prefix sum up to j
        b = psi.sum(-1, keepdim=True) - psi.cumsum(-1)
        return b.masked_fill(future, 0.0)  # exact zeros, not rounding residue

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, p_dim={self.p_dim}"


class Cable(_ContentBias):
    """CABLE (Veisi et al., EMNLP 2025) as in the paper's equations and the official
    ``cable6.py`` (``AlgonetLabs/Cable`` @ ``05d2c62``): a content clock
    ``S_t = -sum_{l<=t} relu(w_c x_l + b_c)`` and a per-query weight
    ``g_t = softplus(w_s x_t + b_s)``; ``b(t, j) = g_t (S_t - S_j) <= 0``.

    The minus sign inside ``S`` is in the code, not in the paper's equations; it is what makes
    the bias a penalty growing with (clock) distance. The repository's default ``cable.py``
    drops ``g`` and uses ``-softplus``, which is FoX's ``log sigmoid`` with negated weights.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.clock = nn.Linear(dim, num_heads)
        self.scale = nn.Linear(dim, num_heads)

    def terms(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """The clock ``S`` and the query weights ``g``, each ``(B, H, N)``."""
        s = (-F.relu(at_least_f32(self.clock(x)))).cumsum(1).transpose(1, 2)
        return s, F.softplus(at_least_f32(self.scale(x))).transpose(1, 2)

    def bias_ctx(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> Tensor:
        s, g = self.terms(x)
        b = g[..., :, None] * (s[..., :, None] - s[..., None, :])
        return b.masked_fill(~_lower(x.shape[1], x.device), 0.0)

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        s_q, g = self.terms(x)
        s_k = s_q.clone()  # read at kv_idx: a second index needs its own tensor
        return lambda s, b, h, qi, kv: s + g[b, h, qi] * (s_q[b, h, qi] - s_k[b, h, kv])

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}"


class ForgetGate(_ContentBias):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, num_heads, bias=True)

    def cumulative(self, x: Tensor) -> Tensor:
        """``c_t``, the cumulative log forget gate, ``(B, H, N)``."""
        return F.logsigmoid(at_least_f32(self.proj(x))).transpose(1, 2).cumsum(-1)

    def bias_ctx(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> Tensor:
        c = self.cumulative(x)
        b = c[..., :, None] - c[..., None, :]
        return b.masked_fill(~_lower(x.shape[1], x.device), 0.0)

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        c_q = self.cumulative(x)
        c_k = c_q.clone()  # read at kv_idx: a second index needs its own tensor
        return lambda s, b, h, qi, kv: s + c_q[b, h, qi] - c_k[b, h, kv]

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}"
