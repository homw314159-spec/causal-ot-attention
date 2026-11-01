"""Fused prefix UnbalancedSinkhorn for causal attention (ADR-0009).

:class:`FusedUnbalancedSinkhorn` is a drop-in replacement for
``geoattn_core.UnbalancedSinkhorn(rho, n_iters, eps, column_mass="prefix")`` under a causal
mask: same interface, same values to fp32 rounding. Inputs the kernel does not cover (no mask
or a non-causal one, CPU tensors, rows longer than ``MAX_N``) go to the reference
implementation, so the module is always correct and only sometimes fast.
"""

from __future__ import annotations

import math
import weakref
from typing import Any

import torch
import triton
from torch import Tensor, nn

from geoattn_core import UnbalancedSinkhorn
from geoattn_kernels._unbalanced_bwd import bwd_kernel
from geoattn_kernels._unbalanced_fwd import fwd_kernel, num_tiers

MAX_N = 2048  # the whole row of a query block is kept on chip
# Without the per-iteration row max the shift is the iteration-1 row max; this is safe while
# rho stays away from 1 (ADR-0009). Above the threshold the kernel recomputes the max.
STABLE_RHO = 0.9
# Tuned on the RTX 4090 (ADR-0009): one program saturates an SM, so fewer warps per
# barrier beat more parallelism inside the program.
_FWD_BLOCK_M, _BWD_BLOCK_M, _WARPS = 16, 8, 8


def _meta(n_iters: int, rho: float, block_m: int, block_n: int, **extra: Any) -> dict[str, Any]:
    """Compile-time constants and launch options, passed by keyword: the kernels annotate
    them as ``tl.constexpr``, which the Python type system cannot match with plain ints."""
    return {
        "N_ITERS": n_iters,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "TIERS": num_tiers(block_n, block_m),
        "STABLE": rho > STABLE_RHO,
        "num_warps": _WARPS,
        **extra,
    }


def _harmonic(n: int, device: torch.device) -> Tensor:
    """H_0..H_n (float32, H_0 = 0); length n + 1."""
    h = torch.cat([torch.zeros(1, dtype=torch.float64), (1.0 / torch.arange(1, n + 1)).cumsum(0)])
    return h.to(device=device, dtype=torch.float32)


def _launch_forward(
    s: Tensor, rho: float, n_iters: int, eps: float, save: bool, uniform: bool
) -> tuple:
    bh, n = s.shape[0], s.shape[-1]
    block_n = triton.next_power_of_2(n)
    k = max(n_iters, 1)
    # Carries are saved per backward block, so a saving forward uses the backward block size.
    bm = _BWD_BLOCK_M if save else _FWD_BLOCK_M
    out = torch.empty_like(s)
    carry = torch.zeros(bh, k, block_n, device=s.device)
    saved = torch.empty(bh, triton.cdiv(n, bm), k, block_n, device=s.device) if save else carry
    # Covers every key index a tile can load (up to the next power of two) and padded rows.
    harm = _harmonic(block_n + 2 * _FWD_BLOCK_M, s.device)
    fwd_kernel[(bh,)](
        s, out, carry, saved, harm, n,
        math.log2(math.e) / eps, rho,
        **_meta(n_iters, rho, bm, block_n, SAVE=save, UNIFORM=uniform),
    )  # fmt: skip
    return out, saved


class _FusedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, scores: Tensor, rho: float, n_iters: int, eps: float, uniform: bool
    ) -> Tensor:
        n = scores.shape[-1]
        s = scores.contiguous().view(-1, n, n)
        out, saved = _launch_forward(
            s, rho, n_iters, eps, save=scores.requires_grad, uniform=uniform
        )
        ctx.save_for_backward(s, saved)
        ctx.args = (rho, n_iters, eps, uniform)
        return out.view(scores.shape)

    @staticmethod
    def backward(ctx: Any, *grads: Tensor) -> tuple[Tensor, None, None, None, None]:
        s, saved = ctx.saved_tensors
        rho, n_iters, eps, uniform = ctx.args
        bh, n = s.shape[0], s.shape[-1]
        block_n, k, bm = triton.next_power_of_2(n), max(n_iters, 1), _BWD_BLOCK_M
        g = grads[0].contiguous().view(-1, n, n)
        ds = torch.empty_like(s)
        dcarry = torch.zeros(bh, k, block_n, device=s.device)
        scr_v = torch.empty(bh, k, bm, block_n, device=s.device)
        scr_l = torch.empty(bh, k, bm, device=s.device)
        harm = _harmonic(block_n + 2 * _FWD_BLOCK_M, s.device)
        bwd_kernel[(bh,)](
            s, g, ds, saved, dcarry, scr_v, scr_l, harm,
            n, math.log2(math.e) / eps, rho,
            **_meta(n_iters, rho, bm, block_n, UNIFORM=uniform),
        )  # fmt: skip
        return ds.view(grads[0].shape), None, None, None, None


def is_causal_mask(mask: Tensor | None, n: int) -> bool:
    """True iff ``mask`` is exactly the (n, n) lower-triangular causal mask (broadcast dims ok)."""
    if mask is None or mask.shape[-2:] != (n, n) or any(d != 1 for d in mask.shape[:-2]):
        return False
    tril = torch.ones(n, n, dtype=torch.bool, device=mask.device).tril()
    return bool(torch.equal(mask.reshape(n, n), tril))


class FusedUnbalancedSinkhorn(nn.Module):
    """Fused ``UnbalancedSinkhorn(column_mass="prefix")`` for causal attention on CUDA."""

    def __init__(
        self, rho: float, n_iters: int = 3, eps: float = 1.0, column_target: str = "one"
    ) -> None:
        super().__init__()
        # The reference validates the arguments and is the fallback for uncovered inputs.
        self.reference = UnbalancedSinkhorn(
            rho,
            n_iters=n_iters,
            eps=eps,
            column_mass="prefix",
            column_target=column_target,  # type: ignore[arg-type]  # validated there
        )
        self.rho, self.n_iters, self.eps = rho, n_iters, eps
        self.column_target = column_target
        self._checked: tuple[weakref.ref[Tensor], int, bool] | None = None

    def covers(self, scores: Tensor, mask: Tensor | None) -> bool:
        n = scores.shape[-1]
        shape_ok = (
            scores.is_cuda
            and scores.shape[-2] == n
            and n <= MAX_N
            and scores.dtype in (torch.float32, torch.float16, torch.bfloat16)
        )
        return shape_ok and mask is not None and self._is_causal(mask, n)

    def _is_causal(self, mask: Tensor, n: int) -> bool:
        # The check syncs with the GPU; every layer of a model forward gets the same mask
        # object, so remember the verdict for that object (and its in-place version).
        if self._checked is not None:
            ref, version, verdict = self._checked
            if ref() is mask and mask._version == version:
                return verdict
        verdict = is_causal_mask(mask, n)
        self._checked = (weakref.ref(mask), mask._version, verdict)
        return verdict

    def forward(self, scores: Tensor, mask: Tensor | None = None) -> Tensor:
        if not self.covers(scores, mask):
            return self.reference(scores, mask)
        uniform = self.column_target == "uniform"
        return _FusedFn.apply(scores, self.rho, self.n_iters, self.eps, uniform)  # type: ignore[no-any-return]

    def extra_repr(self) -> str:
        return f"rho={self.rho}, n_iters={self.n_iters}, eps={self.eps}"
