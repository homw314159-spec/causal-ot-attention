"""Triton backward kernel of the prefix UnbalancedSinkhorn under a causal mask (ADR-0009).

Blocks are visited in reverse. ``mass[t] = sum_{s<=t} p[s] + carry``, so the rows of a block
receive the mass adjoint of every later row: in-block by a reverse cumsum, beyond it through
``DCARRY`` (per iteration, the column sum of the adjoint of all later rows). Each block first
re-runs its forward from the carries saved by the forward pass and keeps log v (base 2) and
the row lse of every iteration in a small per-program scratch buffer (L2-resident).

Adjoints are in natural-log units: P and the mass are unit-free, and the natural-log update
``log v <- rho (log v - ln M)`` gives ``d M = -rho d(log v) / M`` whatever base the forward
values are stored in.
"""

from __future__ import annotations

import triton
import triton.language as tl

from geoattn_kernels._unbalanced_fwd import _row_softmax2, load_block, zero_tail


@triton.jit
def _bwd_tier(
    S,
    G,
    DS,
    saved_base,
    dcarry_base,
    scr_v,
    scr_l,
    HARM,
    FIRST: tl.constexpr,
    N,
    scale2,
    rho,
    N_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    W: tl.constexpr,
    CS: tl.constexpr,
    STABLE: tl.constexpr,
    UNIFORM: tl.constexpr,
):
    n_blocks = tl.cdiv(N, BLOCK_M)
    lo = 0 if FIRST else (W // 2) // BLOCK_M
    hi = tl.minimum(tl.cdiv(W, BLOCK_M), n_blocks)
    cols = tl.arange(0, W)
    r = tl.arange(0, BLOCK_M)
    tile = r[:, None] * CS + cols[None, :]
    for rb in range(0, hi - lo):
        blk = hi - 1 - rb
        log_k, rows, offs, keep = load_block(S, blk, N, scale2, BLOCK_M, W)
        saved = saved_base + blk * N_ITERS * CS
        # Re-run this block's forward (identical arithmetic), keeping log v and the lse.
        m0 = tl.max(log_k, axis=1)
        p, lse = _row_softmax2(log_k, m0)
        tl.store(scr_l + r, lse)
        # Column target (ADR-0009 addendum): 1, or the prefix mass of uniform attention,
        # H_t - H_(j-1) under the causal mask (HARM holds harmonic numbers, H_0 = 0).
        if UNIFORM:
            tgt = tl.load(HARM + rows + 1)[:, None] - tl.load(HARM + cols)[None, :]
            log_t = tl.where(tgt > 0, tl.log2(tl.where(tgt > 0, tgt, 1.0)), 0.0)
        else:
            log_t = 0.0
        log_v = tl.zeros((BLOCK_M, W), dtype=tl.float32)
        tl.store(scr_v + tile, log_v)
        for i in range(1, N_ITERS):
            mass = tl.cumsum(p, axis=0) + tl.load(saved + (i - 1) * CS + cols)[None, :]
            log_v = tl.where(mass > 0, rho * (log_v + log_t - tl.log2(mass)), 0.0)
            tl.store(scr_v + i * BLOCK_M * CS + tile, log_v)
            x = log_k + log_v
            m = tl.max(x, axis=1) if STABLE else m0
            p, lse = _row_softmax2(x, m)
            tl.store(scr_l + i * BLOCK_M + r, lse)
        # Reverse through the iterations; p holds P_i, dp its adjoint.
        dp = tl.load(G + offs, mask=keep, other=0.0).to(tl.float32)
        d_log_k = tl.zeros((BLOCK_M, W), dtype=tl.float32)
        d_log_v = tl.zeros((BLOCK_M, W), dtype=tl.float32)
        for j in range(1, N_ITERS):
            i = N_ITERS - j
            dx = p * (dp - tl.sum(dp * p, axis=1)[:, None])
            d_log_k += dx
            d_log_v += dx
            lv_prev = tl.load(scr_v + (i - 1) * BLOCK_M * CS + tile)
            lse_prev = tl.load(scr_l + (i - 1) * BLOCK_M + r)
            p = tl.exp2(log_k + lv_prev - lse_prev[:, None])
            mass = tl.cumsum(p, axis=0) + tl.load(saved + (i - 1) * CS + cols)[None, :]
            live = mass > 0
            d_mass = tl.where(live, -rho * d_log_v / tl.where(live, mass, 1.0), 0.0)
            d_log_v = tl.where(live, rho * d_log_v, 0.0)
            dc_ptr = dcarry_base + (i - 1) * CS + cols
            dc = tl.load(dc_ptr)
            dp = tl.cumsum(d_mass, axis=0, reverse=True) + dc[None, :]
            tl.store(dc_ptr, dc + tl.sum(d_mass, axis=0))
        d_log_k += p * (dp - tl.sum(dp * p, axis=1)[:, None])
        # d/dS = d/d(log_k natural) / eps; scale2 = log2(e) / eps.
        ds = tl.where(keep, d_log_k * (scale2 * 0.6931471805599453), 0.0)
        store_mask = (cols[None, :] < N) & (rows[:, None] < N)
        tl.store(DS + offs, ds.to(DS.dtype.element_ty), mask=store_mask)
        zero_tail(DS, rows, N, W)


@triton.jit
def bwd_kernel(
    S,
    G,
    DS,
    SAVED,
    DCARRY,
    SCR_V,
    SCR_L,
    HARM,
    N,
    scale2,
    rho,
    N_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TIERS: tl.constexpr,
    STABLE: tl.constexpr,
    UNIFORM: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * N * N
    saved_base = SAVED + pid.to(tl.int64) * tl.cdiv(N, BLOCK_M) * N_ITERS * BLOCK_N
    dcarry_base = DCARRY + pid.to(tl.int64) * N_ITERS * BLOCK_N
    scr_v = SCR_V + pid.to(tl.int64) * N_ITERS * BLOCK_M * BLOCK_N
    scr_l = SCR_L + pid.to(tl.int64) * N_ITERS * BLOCK_M
    # Widest tier first: the backward walks the blocks in reverse.
    _bwd_tier(S + base, G + base, DS + base, saved_base, dcarry_base, scr_v, scr_l, HARM,
              TIERS == 1, N, scale2, rho, N_ITERS, BLOCK_M, BLOCK_N, BLOCK_N, STABLE,
              UNIFORM)  # fmt: skip
    if TIERS >= 2:
        _bwd_tier(S + base, G + base, DS + base, saved_base, dcarry_base, scr_v, scr_l, HARM,
                  TIERS == 2, N, scale2, rho, N_ITERS, BLOCK_M, BLOCK_N >> 1, BLOCK_N,
                  STABLE, UNIFORM)  # fmt: skip
    if TIERS >= 3:
        _bwd_tier(S + base, G + base, DS + base, saved_base, dcarry_base, scr_v, scr_l, HARM,
                  TIERS == 3, N, scale2, rho, N_ITERS, BLOCK_M, BLOCK_N >> 2, BLOCK_N,
                  STABLE, UNIFORM)  # fmt: skip
    if TIERS >= 4:
        _bwd_tier(S + base, G + base, DS + base, saved_base, dcarry_base, scr_v, scr_l, HARM,
                  True, N, scale2, rho, N_ITERS, BLOCK_M, BLOCK_N >> 3, BLOCK_N, STABLE,
                  UNIFORM)  # fmt: skip
