"""Evaluation-time interventions on the tail of an encoding's additive bias (experiment 0003,
part B2, gate G2).

``TailIntervention`` wraps a trained encoding and changes its total additive bias ``b(t, j)``
only at distances ``d = t - j > L`` (``L`` = ``tail_from``, the training length by default);
at ``d <= L`` the bias, hence the 1x loss, is unchanged bit for bit. For ``d > L``:

* ``cap``:    ``b(t, t - L)``, the bias the same query gives the key at distance L
* ``repair``: ``b(t, t - L) - e [log(1 + d) - log(1 + L)]`` with ``e = 1.5``: a summable tail
* ``window``: ``b(t, j) - m (d - L)``: a linear penalty on top of the bias

Reading the anchor at the same query keeps content-dependent biases (CABLE, FoX) well
defined: their bias is not a function of distance alone. Positions must be ``0..N-1`` (the
causal contract of ``GeometricAttention``), so the key at distance L from row t is column
``t - L``. Evaluation only: the fused path reads the wrapped encoding's tensors twice, which
FlexAttention does not support for tensors that need gradients.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geoattn_core.flex import ScoreMod
from geoattn_core.pe.base import PositionEncoding

TAIL_MODES = ("cap", "repair", "window")
REPAIR_EXPONENT = 1.5  # preregistered (experiment 0003 §3.3)


class TailIntervention(nn.Module):
    def __init__(
        self,
        pe: PositionEncoding,
        mode: str,
        tail_from: int,
        slope: float = 0.0,
        repair_exponent: float = REPAIR_EXPONENT,
    ) -> None:
        """``slope``: the ``m`` of ``window``. The wrapper adds no parameters; a checkpoint
        of the unwrapped model loads into it (its keys gain the ``pe.`` level on load)."""
        super().__init__()
        if mode not in TAIL_MODES:
            raise ValueError(f"unknown tail mode {mode!r} ({' | '.join(TAIL_MODES)})")
        if tail_from < 1:
            raise ValueError(f"tail_from must be >= 1, got {tail_from}")
        if mode == "window" and slope <= 0:
            raise ValueError(f"window needs a positive slope, got {slope}")
        if not isinstance(pe, nn.Module):
            raise TypeError("the wrapped encoding must be an nn.Module")
        self.pe, self.mode, self.tail_from = pe, mode, tail_from
        self.slope, self.repair_exponent = slope, repair_exponent
        self.register_load_state_dict_pre_hook(_nest_keys)

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return self.pe.act(q, k, positions)

    @property
    def fusable(self) -> bool:
        return getattr(self.pe, "fusable", hasattr(self.pe, "score_mod"))

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None:
        return self.intervene(self.pe.bias(positions_q, positions_k), positions_q, positions_k)

    def bias_ctx(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> Tensor | None:
        ctx = getattr(self.pe, "bias_ctx", None)
        b = ctx(q, k, x, positions) if ctx else self.pe.bias(positions, positions)
        return self.intervene(b, positions, positions)

    def intervene(self, b: Tensor | None, positions_q: Tensor, positions_k: Tensor):
        """The bias with its tail replaced; ``b=None`` is a zero bias (e.g. RoPE alone)."""
        big_l = self.tail_from
        d = (positions_q[..., :, None] - positions_k[..., None, :]).to(torch.float32)
        if d.ndim == 3:  # per-batch positions: add the head axis
            d = d[:, None]
        far = d > big_l
        if self.mode == "window":
            penalty = -self.slope * (d - big_l)
            return penalty.where(far, 0.0) if b is None else b + penalty.where(far, 0.0)
        if b is None:
            b = torch.zeros_like(d)
        n_k = b.shape[-1]
        if n_k <= big_l:
            return b
        # row t's anchor is b[t, t - L], the -L-th diagonal; rows t < L have no far keys
        anchor = F.pad(b.diagonal(offset=-big_l, dim1=-2, dim2=-1), (big_l, 0))[..., :, None]
        return torch.where(far, anchor + self._decay(d), b)

    def _decay(self, d: Tensor) -> Tensor | float:
        if self.mode == "cap":
            return 0.0
        return -self.repair_exponent * (torch.log1p(d.clamp_min(0)) - math.log1p(self.tail_from))

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        if torch.is_grad_enabled():
            raise RuntimeError("TailIntervention is for evaluation: call under torch.no_grad()")
        inner = self.pe.score_mod(q, k, x, positions)  # type: ignore[attr-defined]
        mode, big_l, slope, e = self.mode, self.tail_from, self.slope, self.repair_exponent
        log_l = math.log1p(big_l)

        def mod(s: Tensor, b: Tensor, h: Tensor, qi: Tensor, kv: Tensor) -> Tensor:
            near = inner(s, b, h, qi, kv) if inner else s
            dist = qi - kv
            far = dist > big_l
            d = dist.float()
            if mode == "window":
                return torch.where(far, near - slope * (d - big_l), near)
            # the clamp keeps the anchor index valid on rows t < L, whose value is unused
            anchor_kv = torch.clamp(qi - big_l, min=0)
            zero = torch.zeros_like(s)
            anchor = inner(zero, b, h, qi, anchor_kv) if inner else zero
            if mode == "repair":
                anchor = anchor - e * (torch.log1p(d.clamp(min=0)) - log_l)
            return torch.where(far, s + anchor, near)

        return mod

    def extra_repr(self) -> str:
        extra = f", slope={self.slope:.4g}" if self.mode == "window" else ""
        return f"mode={self.mode}, tail_from={self.tail_from}{extra}"


def _nest_keys(module: nn.Module, state_dict: dict, prefix: str, *args) -> None:
    """Move the wrapped encoding's entries of an unwrapped checkpoint under ``pe.``."""
    for key in [k for k in state_dict if k.startswith(prefix)]:
        if not key.startswith(prefix + "pe."):
            state_dict[prefix + "pe." + key[len(prefix) :]] = state_dict.pop(key)
