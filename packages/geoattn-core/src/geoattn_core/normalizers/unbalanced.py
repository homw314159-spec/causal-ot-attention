"""Unbalanced Sinkhorn: hard row marginal, KL-relaxed column marginal (ADR-0005, ADR-0006).

With ``P = diag(u) K diag(v)``, ``K = exp(S / eps)``, the column marginal
penalized by ``lambda * KL`` gives the scaling update ``v <- (target / K^T u)^rho``
with ``rho = lambda / (lambda + eps)`` (Chizat et al. 2018; semi-relaxed form of
Fukunaga & Kasai 2022). Since ``K^T u = M / v`` for the column mass ``M`` of the
current row-normalized ``P``, in the log domain::

    step 1                : log v = 0, P = row-normalize(K)      -> equals Softmax
    steps 2 .. n_iters    : log v <- rho * (log v + log target - log M)
                            P <- row-normalize(K * v)

``column_mass`` selects how ``M`` is summed:

* ``"prefix"`` — ``M_tj = sum_{i <= t} P_ij`` (cumulative along the query axis),
  target 1 per column. Row ``t`` then depends only on rows ``1..t``, so the
  operator is prefix-consistent and legal under a causal mask (ADR-0005). It is
  not the solution of any single OT problem (docs/methods.md §4.4).
* ``"full"`` — ``M_j = sum_i P_ij``, target ``n_q / n_k`` as in :class:`Sinkhorn`.
  This is the semi-relaxed OT scaling (checked against POT). It LEAKS future
  rows under a causal mask and is for non-causal attention only.

``rho = 0`` is exactly Softmax; ``rho = 1`` with ``"full"`` is exactly Sinkhorn.
Like Sinkhorn, every iteration ends with a row normalization, so the output is
always exactly row-stochastic.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from geoattn_core.normalizers.base import log_normalize, prepare_log_kernel
from geoattn_core.normalizers.sinkhorn import column_log_target

ColumnMass = Literal["prefix", "full"]
ColumnTarget = Literal["one", "uniform"]
GradMode = Literal["unroll", "implicit"]

# Stand-in for log(0) inside the column-mass reductions. logcumsumexp's backward
# evaluates exp(x - out), which is NaN on all -inf prefixes (padding, or columns
# not yet visible under a causal mask); a finite floor keeps it exact elsewhere
# because exp(_LOG_ZERO - anything realistic) underflows to 0.
_LOG_ZERO = -1e30


def uniform_prefix_log_target(log_k: Tensor) -> Tensor:
    """log of the prefix column mass that uniform attention would give: row s spreads 1
    over its n_s visible keys, so target_tj = sum over rows s <= t that see j of 1 / n_s
    (H_t - H_(j-1) under a causal mask). Depends only on rows 1..t (prefix consistent).
    Prefixes with no mass get 0; the update leaves them untouched anyway."""
    valid = torch.isfinite(log_k)
    uniform = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1)
    mass = uniform.to(log_k.dtype).cumsum(dim=-2)
    return torch.where(mass > 0, mass.log(), torch.zeros_like(mass))


def log_column_mass(log_p: Tensor, column_mass: ColumnMass) -> Tensor:
    """``log M`` for :class:`UnbalancedSinkhorn`; broadcastable against ``log_p``."""
    safe = log_p.masked_fill(~torch.isfinite(log_p), _LOG_ZERO)
    if column_mass == "prefix":
        return torch.logcumsumexp(safe, dim=-2)
    return torch.logsumexp(safe, dim=-2, keepdim=True)


class UnbalancedSinkhorn(nn.Module):
    def __init__(
        self,
        rho: float,
        n_iters: int = 3,
        eps: float = 1.0,
        column_mass: ColumnMass = "prefix",
        grad_mode: GradMode = "unroll",
        backward_iters: int | None = None,
        backward_tol: float = 1e-6,
        column_target: ColumnTarget = "one",
    ) -> None:
        """``grad_mode="unroll"`` (default) backpropagates through every iteration;
        ``"implicit"`` differentiates the fixed point instead (experimental; see
        ``_implicit_log_v``). The forward value is the same in both modes.

        ``column_target`` (prefix mode only): ``"one"`` (default) pulls every column's
        prefix mass toward 1; ``"uniform"`` toward the mass uniform attention would give
        (:func:`uniform_prefix_log_target`). With ``"one"`` columns seen by many rows are
        always over budget, which acts as a distance decay; with ``"uniform"`` uniform
        attention is a fixed point and only deviations from it are corrected."""
        super().__init__()
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        if column_mass not in ("prefix", "full"):
            raise ValueError(f"column_mass must be 'prefix' or 'full', got {column_mass!r}")
        if grad_mode not in ("unroll", "implicit"):
            raise ValueError(f"grad_mode must be 'unroll' or 'implicit', got {grad_mode!r}")
        if column_target not in ("one", "uniform"):
            raise ValueError(f"column_target must be 'one' or 'uniform', got {column_target!r}")
        if column_target == "uniform" and column_mass != "prefix":
            raise ValueError("column_target='uniform' is defined for column_mass='prefix'")
        self.rho = rho
        self.n_iters = n_iters
        self.eps = eps
        self.column_mass: ColumnMass = column_mass
        self.grad_mode: GradMode = grad_mode
        self.backward_iters = backward_iters
        self.backward_tol = backward_tol
        self.column_target: ColumnTarget = column_target

    def forward(self, scores: Tensor, mask: Tensor | None = None) -> Tensor:
        log_k = prepare_log_kernel(scores, mask, self.eps)
        log_target: Tensor | float = 0.0
        if self.n_iters > 1 and self.column_mass == "full":
            log_target = column_log_target(log_k)
        elif self.n_iters > 1 and self.column_target == "uniform":
            log_target = uniform_prefix_log_target(log_k)
        if self.grad_mode == "implicit" and self.n_iters > 1 and torch.is_grad_enabled():
            log_v = self._implicit_log_v(log_k, log_target)
            return log_normalize(log_k + log_v, dim=-1).exp().to(scores.dtype)
        # Unrolled path, kept op-for-op as before so gradients stay bit-identical
        # (reusing log_p between iterations fixes autograd's accumulation order).
        log_v = torch.zeros((), dtype=log_k.dtype, device=log_k.device)
        log_p = log_normalize(log_k, dim=-1)
        for _ in range(self.n_iters - 1):
            log_m = log_column_mass(log_p, self.column_mass)
            log_v = self._update(log_v, log_target, log_m)
            log_p = log_normalize(log_k + log_v, dim=-1)
        return log_p.exp().to(scores.dtype)

    def _update(self, log_v: Tensor, log_target: Tensor | float, log_m: Tensor) -> Tensor:
        # Damp the accumulated column scaling v, not a relative update of P: the
        # latter has the balanced fixed point colsum = target for any rho > 0.
        log_v = self.rho * (log_v + log_target - log_m)
        # Empty (prefixes of) columns carry no mass to rescale; keep v finite there.
        return log_v.masked_fill(log_m < _LOG_ZERO / 2, 0.0)

    def _step(self, log_v: Tensor, log_k: Tensor, log_target: Tensor | float) -> Tensor:
        """The fixed-point map F: ``log v <- rho (log v + log target - log M(P(log v)))``."""
        log_m = log_column_mass(log_normalize(log_k + log_v, dim=-1), self.column_mass)
        return self._update(log_v, log_target, log_m)

    def _implicit_log_v(self, log_k: Tensor, log_target: Tensor | float) -> Tensor:
        """Fixed point ``v* = F(v*, K)`` differentiated by the implicit function theorem.

        The first n_iters - 2 updates run without a graph and the last one re-attaches
        ``v`` to ``log_k``, so the forward value equals the unrolled mode's. A hook
        on that tensor turns the incoming gradient w into
        lambda = (I - J^T)^-1 w, J = dF/dv at v*, by the fixed-point iteration
        lambda <- J^T lambda + w (one VJP per step on a single-update graph), so
        memory does not grow with n_iters. ``backward_iters`` caps that iteration
        (default 4 n_iters); with ``backward_tol=0`` it runs exactly that many steps,
        the truncated ("phantom" / Jacobian-free) gradient.
        """
        with torch.no_grad():
            v_star = torch.zeros((), dtype=log_k.dtype, device=log_k.device)
            for _ in range(self.n_iters - 2):
                v_star = self._step(v_star, log_k, log_target)
        v = self._step(v_star, log_k, log_target)  # graph to log_k; value ~ v*
        v0 = v_star.expand_as(v).clone().requires_grad_()
        f0 = self._step(v0, log_k.detach(), log_target)

        def solve_adjoint(w: Tensor) -> Tensor:
            lam = w
            for _ in range(self.backward_iters or 4 * self.n_iters):
                (jt_lam,) = torch.autograd.grad(f0, v0, lam, retain_graph=True)
                nxt = jt_lam + w
                converged = (nxt - lam).abs().max() <= self.backward_tol * nxt.abs().max()
                lam = nxt
                if self.backward_tol > 0 and converged:
                    break
            return lam

        if v.requires_grad:
            v.register_hook(solve_adjoint)
        return v

    def extra_repr(self) -> str:
        return (
            f"rho={self.rho}, n_iters={self.n_iters}, eps={self.eps}, "
            f"column_mass={self.column_mass!r}, grad_mode={self.grad_mode!r}"
            + (f", column_target={self.column_target!r}" if self.column_target != "one" else "")
        )
