"""Triton forward kernel of the prefix UnbalancedSinkhorn under a causal mask (ADR-0009).

One program per (batch, head) walks the query blocks in order and runs *all* iterations on a
block while it sits on chip: by prefix consistency (ADR-0005) row t of iteration i needs only
rows 1..t of iteration i-1, so the only state crossing blocks is, per iteration, the column
mass of all earlier rows (an N-vector, ``CARRY``). Blocks are grouped into tiers of width
BLOCK_N / 2^j: under a causal mask the rows of a block below W only see columns below W.

Logs are kept in base 2 (``exp2`` / ``log2`` are the native instructions); log v is linear
in the log base, so the update rule is unchanged.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _row_softmax2(x, m):
    """Row softmax of base-2 logits ``x`` shifted by ``m``; also returns the base-2 lse."""
    e = tl.exp2(x - m[:, None])
    s = tl.sum(e, axis=1)
    return e * (1.0 / s)[:, None], m + tl.log2(s)


@triton.jit
def load_block(S, blk, N, scale2, BLOCK_M: tl.constexpr, W: tl.constexpr):
    """Base-2 log kernel of one query block, masked entries -inf.

    Padding rows (beyond N) get a single 0 in column 0 so they stay finite; they come after
    every real row, so they never enter a real row's prefix mass.
    """
    cols = tl.arange(0, W)
    rows = blk * BLOCK_M + tl.arange(0, BLOCK_M)
    real = rows < N
    keep = (cols[None, :] <= rows[:, None]) & (cols[None, :] < N) & real[:, None]
    offs = rows[:, None] * N + cols[None, :]
    log_k = tl.load(S + offs, mask=keep, other=float("-inf")).to(tl.float32) * scale2
    log_k = tl.where((~real[:, None]) & (cols[None, :] == 0), 0.0, log_k)
    return log_k, rows, offs, keep


@triton.jit
def zero_tail(PTR, rows, N, W: tl.constexpr):
    """Columns beyond this block's tier width are all masked: write them as 0."""
    cols = tl.arange(0, W)
    zeros = tl.zeros((rows.shape[0], W), PTR.dtype.element_ty)
    for c0 in range(W, N, W):
        c = c0 + cols
        tl.store(
            PTR + rows[:, None] * N + c[None, :], zeros, mask=(c[None, :] < N) & (rows[:, None] < N)
        )


@triton.jit
def _fwd_tier(
    S,
    OUT,
    carry_base,
    saved_base,
    HARM,
    FIRST: tl.constexpr,
    N,
    scale2,
    rho,
    N_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    W: tl.constexpr,
    CS: tl.constexpr,
    SAVE: tl.constexpr,
    STABLE: tl.constexpr,
    UNIFORM: tl.constexpr,
):
    n_blocks = tl.cdiv(N, BLOCK_M)
    lo = 0 if FIRST else (W // 2) // BLOCK_M
    cols = tl.arange(0, W)
    for blk in range(lo, tl.minimum(tl.cdiv(W, BLOCK_M), n_blocks)):
        log_k, rows, offs, _ = load_block(S, blk, N, scale2, BLOCK_M, W)
        # Iteration 1 is plain softmax; the diagonal keeps every real row non-empty.
        m0 = tl.max(log_k, axis=1)
        p, _ = _row_softmax2(log_k, m0)
        # Column target (ADR-0009 addendum): 1, or the prefix mass of uniform attention,
        # H_t - H_(j-1) under the causal mask (HARM holds harmonic numbers, H_0 = 0).
        if UNIFORM:
            tgt = tl.load(HARM + rows + 1)[:, None] - tl.load(HARM + cols)[None, :]
            log_t = tl.where(tgt > 0, tl.log2(tl.where(tgt > 0, tgt, 1.0)), 0.0)
        else:
            log_t = 0.0
        log_v = tl.zeros((BLOCK_M, W), dtype=tl.float32)
        for i in range(N_ITERS - 1):
            carry_ptr = carry_base + i * CS + cols
            prev = tl.load(carry_ptr)
            if SAVE:
                tl.store(saved_base + (blk * N_ITERS + i) * CS + cols, prev)
            mass = tl.cumsum(p, axis=0) + prev[None, :]
            tl.store(carry_ptr, prev + tl.sum(p, axis=0))
            # Empty prefixes (columns not yet visible) keep v = 1, as in the reference.
            log_v = tl.where(mass > 0, rho * (log_v + log_t - tl.log2(mass)), 0.0)
            x = log_k + log_v
            # Without STABLE the iteration-1 row max stays the shift (see ADR-0009 for why
            # log v cannot overflow it when rho is bounded away from 1).
            m = tl.max(x, axis=1) if STABLE else m0
            p, _ = _row_softmax2(x, m)
        tl.store(
            OUT + offs, p.to(OUT.dtype.element_ty), mask=(cols[None, :] < N) & (rows[:, None] < N)
        )
        zero_tail(OUT, rows, N, W)


@triton.jit
def fwd_kernel(
    S,
    OUT,
    CARRY,
    SAVED,
    HARM,
    N,
    scale2,
    rho,
    N_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE: tl.constexpr,
    TIERS: tl.constexpr,
    STABLE: tl.constexpr,
    UNIFORM: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * N * N
    carry_base = CARRY + pid.to(tl.int64) * N_ITERS * BLOCK_N
    saved_base = SAVED + pid.to(tl.int64) * tl.cdiv(N, BLOCK_M) * N_ITERS * BLOCK_N
    # Narrowest tier first; each width is its own constexpr specialization. (The widths are
    # passed as literal expressions: a constexpr bound inside a loop loses its constexpr-ness.)
    if TIERS >= 4:
        _fwd_tier(S + base, OUT + base, carry_base, saved_base, HARM, True, N, scale2, rho,
                  N_ITERS, BLOCK_M, BLOCK_N >> 3, BLOCK_N, SAVE, STABLE, UNIFORM)  # fmt: skip
    if TIERS >= 3:
        _fwd_tier(S + base, OUT + base, carry_base, saved_base, HARM, TIERS == 3, N, scale2, rho,
                  N_ITERS, BLOCK_M, BLOCK_N >> 2, BLOCK_N, SAVE, STABLE, UNIFORM)  # fmt: skip
    if TIERS >= 2:
        _fwd_tier(S + base, OUT + base, carry_base, saved_base, HARM, TIERS == 2, N, scale2, rho,
                  N_ITERS, BLOCK_M, BLOCK_N >> 1, BLOCK_N, SAVE, STABLE, UNIFORM)  # fmt: skip
    _fwd_tier(S + base, OUT + base, carry_base, saved_base, HARM, TIERS == 1, N, scale2, rho,
              N_ITERS, BLOCK_M, BLOCK_N, BLOCK_N, SAVE, STABLE, UNIFORM)  # fmt: skip


def num_tiers(block_n: int, block_m: int, max_tiers: int = 4) -> int:
    """Causal width tiers; the narrowest must still hold a block and a reasonable row."""
    t = 1
    while t < max_tiers and (block_n >> t) >= max(block_m, 64):
        t += 1
    return t
