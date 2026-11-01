"""Config name -> component. Plain functions, no registry: a handful of normalizers
and PEs today. Revisit when P3 brings GRAPE/LieRE.
"""

from __future__ import annotations

from geoattn_core import (
    ALiBi,
    IdentityPE,
    Normalizer,
    PositionEncoding,
    RoPE,
    ScaledSoftmax,
    Sinkhorn,
    Softmax,
    UnbalancedSinkhorn,
)
from geoattn_core.pe import (
    Cable,
    ClockLogDistanceBias,
    ComposedPE,
    ForgetGate,
    GatedLogDistanceBias,
    GrapeA,
    GrapeAP,
    GrapeM,
    KerpleLogBias,
    LogDistanceBias,
    TailIntervention,
)
from geoattn_core.pe.alibi import alibi_slopes
from geoattn_train.config import AttentionConfig, NormalizerConfig


def build_normalizer(cfg: NormalizerConfig, causal: bool = False, num_heads: int = 1) -> Normalizer:
    """``causal=True`` refuses operators that leak future rows (ADR-0005)."""
    match cfg.name:
        case "softmax":
            return Softmax(eps=cfg.eps)
        case "scaled_softmax":  # row-local, hence causal-safe
            return ScaledSoftmax(
                num_heads=num_heads,
                base_len=cfg.scale_base_len,
                eps=cfg.eps,
                learnable=cfg.scale_learnable,
            )
        case "sinkhorn":
            if causal:
                raise ValueError("sinkhorn normalizes full columns and leaks under a causal mask")
            return Sinkhorn(n_iters=cfg.n_iters, eps=cfg.eps)
        case "unbalanced":
            return _build_unbalanced(cfg, causal)
        case other:
            raise ValueError(f"unknown normalizer {other!r}")


def _build_unbalanced(cfg: NormalizerConfig, causal: bool) -> Normalizer:
    if causal and cfg.column_mass != "prefix":
        raise ValueError(f"column_mass={cfg.column_mass!r} leaks under a causal mask; use 'prefix'")
    if cfg.grad_mode not in ("unroll", "implicit"):
        raise ValueError(f"unknown grad_mode {cfg.grad_mode!r}")
    if cfg.column_mass not in ("prefix", "full"):
        raise ValueError(f"unknown column_mass {cfg.column_mass!r}")
    if cfg.impl == "fused":
        if cfg.column_mass != "prefix" or cfg.grad_mode != "unroll":
            raise ValueError("impl='fused' covers column_mass='prefix' with grad_mode='unroll'")
        from geoattn_kernels import FusedUnbalancedSinkhorn  # imports triton: only when asked

        return FusedUnbalancedSinkhorn(
            rho=cfg.rho, n_iters=cfg.n_iters, eps=cfg.eps, column_target=cfg.column_target
        )
    if cfg.impl != "torch":
        raise ValueError(f"unknown impl {cfg.impl!r} (torch | fused)")
    return UnbalancedSinkhorn(
        rho=cfg.rho,
        n_iters=cfg.n_iters,
        eps=cfg.eps,
        column_mass=cfg.column_mass,
        grad_mode=cfg.grad_mode,
        backward_tol=cfg.backward_tol,
        column_target=cfg.column_target,  # type: ignore[arg-type]  # validated there
    )


# Encodings that only define the bias for j <= i and rely on the causal mask for the rest.
CAUSAL_ONLY_PE = ("grape_a", "grape_ap", "fox", "cable", "clock_logbias")


def build_pe(
    cfg: AttentionConfig,
    head_dim: int,
    num_heads: int = 1,
    train_len: int | None = None,
    dim: int | None = None,
    causal: bool = False,
) -> PositionEncoding:
    """``train_len``: the length RoPE scaling starts from when ``rope_train_len`` is 0.
    ``dim``: the attention input width, needed by encodings that project it (GRAPE-AP, FoX).
    """
    for name in (cfg.pe, cfg.pe_bias):
        if name in CAUSAL_ONLY_PE and not causal:
            raise ValueError(f"{name!r} is defined only under a causal mask")
    if cfg.logbias_exponent != "free" and cfg.pe_bias != "logbias":
        raise ValueError(f"logbias_exponent applies to pe_bias='logbias', not {cfg.pe_bias!r}")
    if cfg.logbias_margin and cfg.logbias_exponent != "gt1":
        raise ValueError(
            f"logbias_margin needs logbias_exponent='gt1', not {cfg.logbias_exponent!r}"
        )
    pe = _composed_pe(cfg, _base_pe(cfg, head_dim, num_heads, train_len, dim), num_heads, dim)
    if cfg.tail_mode == "none":
        return pe
    tail_from = cfg.tail_from or train_len
    if tail_from is None:
        raise ValueError("tail_mode needs tail_from or the training length")
    # window's default m: the geometric mean of P4's (ALiBi's) slopes (experiment 0003 §3.3)
    slope = cfg.tail_slope or float(alibi_slopes(num_heads).log().mean().exp())
    return TailIntervention(pe, cfg.tail_mode, tail_from, slope=slope)


def _composed_pe(
    cfg: AttentionConfig, base: PositionEncoding, num_heads: int, dim: int | None
) -> PositionEncoding:
    match cfg.pe_bias:
        case "none":
            return base
        case "alibi":
            return ComposedPE(base, ALiBi(num_heads))
        case "logbias":
            logb = LogDistanceBias(
                num_heads,
                learnable=cfg.logbias_learnable,
                exponent=cfg.logbias_exponent,
                margin=cfg.logbias_margin,
            )
            return ComposedPE(base, logb)
        case "kerple":
            return ComposedPE(base, KerpleLogBias(num_heads))
        case "gated_logbias":
            return ComposedPE(base, GatedLogDistanceBias(_width(dim, cfg.pe_bias), num_heads))
        case "clock_logbias":
            return ComposedPE(base, ClockLogDistanceBias(_width(dim, cfg.pe_bias), num_heads))
        case other:
            raise ValueError(f"unknown pe_bias {other!r}")


def _base_pe(
    cfg: AttentionConfig, head_dim: int, num_heads: int, train_len: int | None, dim: int | None
) -> PositionEncoding:
    match cfg.pe:
        case "identity":
            return IdentityPE()
        case "rope":
            return RoPE(
                head_dim,
                base=cfg.rope_base,
                scaling=cfg.rope_scaling,  # type: ignore[arg-type]  # validated by RoPE
                train_len=cfg.rope_train_len or train_len,
            )
        case "alibi":
            return ALiBi(num_heads)
        case "grape_m":
            return GrapeM(head_dim, base=cfg.rope_base)
        case "grape_a":
            return GrapeA(num_heads, head_dim)
        case "grape_ap":
            return GrapeAP(_width(dim, cfg.pe), num_heads, p_dim=cfg.grape_p_dim)
        case "fox":
            return ForgetGate(_width(dim, cfg.pe), num_heads)
        case "cable":
            return Cable(_width(dim, cfg.pe), num_heads)
        case other:
            raise ValueError(f"unknown position encoding {other!r}")


def _width(dim: int | None, name: str) -> int:
    if dim is None:
        raise ValueError(f"{name!r} projects the attention input: build_pe needs its width (dim)")
    return dim
