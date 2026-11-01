"""Log-distance bias, and composing a multiplicative with an additive position encoding.

``LogDistanceBias`` adds ``-c_h log(1 + |i - j|)`` to the scores: a power-law decay
``(1 + d)^-c_h`` of the attention weight with distance d, one exponent per head. It is
the distance-only part of the prefix unbalanced operator's correction (log v* = -gamma
log M, whose average over keys at distance d is close to logarithmic in d; ADR-0010 and
the column-bias probe), so "softmax + RoPE + log-distance bias" isolates that part.

``ComposedPE`` takes ``act`` from one encoding and adds the biases of both, e.g. RoPE's
rotation together with an additive bias (docs/methods.md §2: the two actions commute).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from geoattn_core.flex import ScoreMod, compose, per_query
from geoattn_core.pe.base import PositionEncoding

# Exponent constraints (experiment 0003, B1): ``gt1`` keeps every head's tail summable,
# ``lt1`` keeps every head's tail non-summable; ``free`` is the unconstrained P1. With a
# margin m, ``gt1``'s range shifts to (1.05 + m, 2 + m) (experiment 0004, B).
EXPONENT_RANGES = {"free": (0.1, 1.0), "gt1": (1.05, 2.0), "lt1": (0.1, 0.95)}


class LogDistanceBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        c_min: float | None = None,
        c_max: float | None = None,
        learnable: bool = True,
        exponent: str = "free",
        margin: float = 0.0,
    ) -> None:
        """Exponents start log-spaced from ``c_min`` to ``c_max`` across heads (default: the
        range ``EXPONENT_RANGES[exponent]``); with ``learnable`` they are trained.

        ``exponent`` maps the parameter onto the allowed range: ``free`` is ``softplus``
        (c > 0), ``gt1`` is ``1 + margin + softplus`` (c > 1 + margin), ``lt1`` is
        ``sigmoid`` (0 < c < 1). A reparametrization rather than a clamp, so the constraint
        holds after every step and the gradient does not vanish at the boundary.

        ``margin`` keeps ``gt1`` away from c = 1: training pushes exponents to the bound,
        and c = 1 + delta dilutes like c = 1 up to length e^(1/delta) (paper, Corollary
        ``cor:boundary``; experiment 0003, H2).
        """
        super().__init__()
        if exponent not in EXPONENT_RANGES:
            raise ValueError(f"unknown exponent {exponent!r} (free | gt1 | lt1)")
        if margin < 0 or (margin > 0 and exponent != "gt1"):
            raise ValueError(f"margin must be >= 0 and needs exponent='gt1', got {margin=}")
        lo, hi = EXPONENT_RANGES[exponent]
        lo, hi = (lo + margin, hi + margin) if exponent == "gt1" else (lo, hi)
        c_min, c_max = lo if c_min is None else c_min, hi if c_max is None else c_max
        if num_heads < 1 or not 0 < c_min <= c_max:
            raise ValueError(f"need num_heads >= 1 and 0 < c_min <= c_max, got {num_heads=}")
        if (exponent == "gt1" and c_min <= 1 + margin) or (exponent == "lt1" and c_max >= 1):
            raise ValueError(f"exponent={exponent!r} cannot start at [{c_min}, {c_max}]")
        self.num_heads, self.exponent, self.margin = num_heads, exponent, margin
        init = torch.logspace(math.log10(c_min), math.log10(c_max), num_heads)
        self.raw = nn.Parameter(_inverse(exponent, init, margin), requires_grad=learnable)

    @property
    def coef(self) -> Tensor:
        if self.exponent == "lt1":
            return torch.sigmoid(self.raw)
        c = nn.functional.softplus(self.raw)
        return 1 + self.margin + c if self.exponent == "gt1" else c

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return q, k

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor:
        """``(H, N_q, N_k)`` for ``(N,)`` positions, ``(B, H, N_q, N_k)`` for ``(B, N)``."""
        dist = (positions_q[..., :, None] - positions_k[..., None, :]).abs().to(torch.float32)
        return -self.coef[:, None, None] * torch.log1p(dist).unsqueeze(-3)

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        c = per_query(self.coef, q)
        return lambda s, b, h, qi, kv: s - c[b, h, qi] * torch.log1p((qi - kv).abs().float())

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, exponent={self.exponent}, margin={self.margin}, "
            f"learnable={self.raw.requires_grad}"
        )


def _inverse(exponent: str, c: Tensor, margin: float = 0.0) -> Tensor:
    """The parameter that maps to exponent ``c``."""
    if exponent == "lt1":
        return torch.logit(c)
    c = c - 1 - margin if exponent == "gt1" else c
    return c + torch.log(-torch.expm1(-c))  # inverse softplus


class KerpleLogBias(nn.Module):
    """KERPLE's logarithmic kernel ``-p_h log(1 + a_h |i - j|)`` (Chi et al., NeurIPS 2022),
    as in the official code (``chijames/KERPLE``, ``ParallelKerpleLog``): ``p ~ U(0, 2)``,
    ``a ~ U(0, 1)`` per head, both projected onto ``[eps, inf)`` in place before every use.

    The projection keeps the official training dynamics: a clamp inside the computation
    would instead zero the gradient and leave a parameter stuck below ``eps``.
    """

    def __init__(self, num_heads: int, eps: float = 1e-2) -> None:
        super().__init__()
        self.num_heads, self.eps = num_heads, eps
        self.p = nn.Parameter(torch.rand(num_heads) * 2)
        self.a = nn.Parameter(torch.rand(num_heads))

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return q, k

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor:
        """``(H, N_q, N_k)`` for ``(N,)`` positions, ``(B, H, N_q, N_k)`` for ``(B, N)``."""
        with torch.no_grad():
            self.p.clamp_(min=self.eps)
            self.a.clamp_(min=self.eps)
        dist = (positions_q[..., :, None] - positions_k[..., None, :]).abs().to(torch.float32)
        dist = dist.unsqueeze(-3)
        return -self.p[:, None, None] * torch.log1p(self.a[:, None, None] * dist)

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        with torch.no_grad():
            self.p.clamp_(min=self.eps)
            self.a.clamp_(min=self.eps)
        p, a = per_query(self.p, q), per_query(self.a, q)
        return lambda s, b, h, qi, kv: (
            s - p[b, h, qi] * torch.log1p(a[b, h, qi] * (qi - kv).abs().float())
        )

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, eps={self.eps}"


class GatedLogDistanceBias(nn.Module):
    """``-c_h(x_t) log(1 + |t - j|)``: a power-law decay whose exponent each query sets
    from its own layer input, ``c_h(x) = softplus(w_h . x + raw_h)`` (experiment 0002, P6).

    The gate reads ``x`` rather than ``q`` because under RoPE the encoding sees ``q`` after
    rotation, which would make the exponent depend on absolute position. Row t reads only
    ``x_t``, so the bias is causal and relative. With ``w = 0`` it is ``LogDistanceBias``
    with the same initial exponents. Prior forms: CABLE and GRAPE-A gate a *linear*
    distance; K-CABLE (CABLE's section 5.6, ``kcable6.py``) gates ``log(1 + (S_t - S_j)^2)``
    on a content clock, of which this is the constant-clock case.
    """

    def __init__(self, dim: int, num_heads: int, c_min: float = 0.1, c_max: float = 1.0) -> None:
        super().__init__()
        self.base = LogDistanceBias(num_heads, c_min, c_max)
        self.gate = nn.Linear(dim, num_heads, bias=False)  # the offset lives in base.raw

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return q, k

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None:
        raise TypeError("GatedLogDistanceBias depends on x: attention calls bias_ctx")

    def coef(self, x: Tensor) -> Tensor:
        """Per-query exponents ``(B, H, N)``."""
        raw = self.gate(x).transpose(1, 2) + self.base.raw[:, None]
        return nn.functional.softplus(raw.to(torch.promote_types(raw.dtype, torch.float32)))

    def bias_ctx(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> Tensor:
        dist = (positions[..., :, None] - positions[..., None, :]).abs()
        log_d = torch.log1p(dist.to(torch.float32))
        if log_d.ndim == 3:  # per-batch positions: add the head axis
            log_d = log_d[:, None]
        return -self.coef(x)[..., None] * log_d

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        c = self.coef(x)
        return lambda s, b, h, qi, kv: s - c[b, h, qi] * torch.log1p((qi - kv).abs().float())

    def extra_repr(self) -> str:
        return f"num_heads={self.base.num_heads}"


class ClockLogDistanceBias(GatedLogDistanceBias):
    """``-c_h(x_t) log(1 + S_t - S_j)`` on a content clock ``S_t = sum_{l<=t} f_h(x_l)``,
    ``f_h(x) = 2 sigmoid(v_h . x)`` (experiment 0002, LC). Causal only.

    With ``v = 0`` the clock ticks once per token, ``S_t - S_j = t - j``, and this is
    ``GatedLogDistanceBias``: P1, P6 and LC are nested. It is a variant of K-CABLE (CABLE's
    section 5.6), which uses a ReLU clock, the squared clock distance and no RoPE.
    """

    def __init__(self, dim: int, num_heads: int, c_min: float = 0.1, c_max: float = 1.0) -> None:
        super().__init__(dim, num_heads, c_min, c_max)
        self.clock = nn.Linear(dim, num_heads, bias=False)

    def time(self, x: Tensor) -> Tensor:
        """The clock ``S`` at every token, ``(B, H, N)``."""
        tick = 2 * torch.sigmoid(self.clock(x))
        return tick.to(torch.promote_types(tick.dtype, torch.float32)).cumsum(1).transpose(1, 2)

    def bias_ctx(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> Tensor:
        s = self.time(x)
        n = x.shape[1]
        future = ~torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        # clamp: the future entries (masked anyway) would be negative inside the log
        dist = (s[..., :, None] - s[..., None, :]).clamp_min(0)
        return (-self.coef(x)[..., None] * torch.log1p(dist)).masked_fill(future, 0.0)

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        c, s_q = self.coef(x), self.time(x)
        s_k = s_q.clone()  # read at kv_idx: a second index needs its own tensor
        return lambda s, b, h, qi, kv: (
            s - c[b, h, qi] * torch.log1p((s_q[b, h, qi] - s_k[b, h, kv]).clamp_min(0))
        )


class ComposedPE(nn.Module):
    """``act`` of ``rotary`` followed by the sum of both encodings' biases."""

    def __init__(self, rotary: PositionEncoding, additive: PositionEncoding) -> None:
        super().__init__()
        if not isinstance(rotary, nn.Module) or not isinstance(additive, nn.Module):
            raise TypeError("both encodings must be nn.Modules (parameters are registered)")
        self.rotary, self.additive = rotary, additive

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        q, k = self.rotary.act(q, k, positions)
        return self.additive.act(q, k, positions)

    def bias_ctx(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> Tensor | None:
        """Sum of both biases, each through its own ``bias_ctx`` when it has one."""
        parts = [_bias(pe, q, k, x, positions) for pe in (self.rotary, self.additive)]
        parts = [b for b in parts if b is not None]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else parts[0] + parts[1]

    @property
    def fusable(self) -> bool:
        """Both parts define ``score_mod`` (the attention can use FlexAttention)."""
        return all(hasattr(pe, "score_mod") for pe in (self.rotary, self.additive))

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        mods = [pe.score_mod(q, k, x, positions) for pe in (self.rotary, self.additive)]  # type: ignore[attr-defined]
        return compose(mods[0], mods[1])

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None:
        parts = [
            b
            for b in (
                self.rotary.bias(positions_q, positions_k),
                self.additive.bias(positions_q, positions_k),
            )
            if b is not None
        ]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else parts[0] + parts[1]


def _bias(pe: PositionEncoding, q: Tensor, k: Tensor, x: Tensor, positions: Tensor):
    ctx = getattr(pe, "bias_ctx", None)
    return ctx(q, k, x, positions) if ctx else pe.bias(positions, positions)
