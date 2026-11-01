"""Rotary position embedding (Su et al., RoFormer 2021): the SO(2)^{d/2} action.

Channel pair ``(i, i + d/2)`` of q and k is rotated by angle ``pos * base^(-2i/d)``
(the "half-split" pairing of GPT-NeoX / LLaMA, not the interleaved one of the
paper; the two differ by a fixed channel permutation, which the learned
projections absorb). Rotations are orthogonal and ``R_m^T R_n = R_{n-m}``, so
``<R_m q, R_n k>`` depends on positions only through ``n - m``: the score is
translation invariant, which is what lets a model be evaluated beyond its
training length (experiment 0001 measures loss at 2x the training context).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from geoattn_core.flex import ScoreMod

Scaling = Literal["none", "pi", "ntk", "yarn", "yarn_t"]
# YaRN's frequency ramp (Peng et al. 2023): wavelengths that fit fewer than ALPHA times in
# the training length are interpolated, more than BETA times kept, linear in between.
YARN_ALPHA, YARN_BETA = 1.0, 32.0


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
        scaling: Scaling = "none",
        train_len: int | None = None,
    ) -> None:
        """``scaling`` rescales the angles for sequences longer than ``train_len`` (factor
        s = n / train_len; the identity for n <= train_len):

        * ``pi``      position interpolation: positions / s
        * ``ntk``     NTK-aware: base * s^(d / (d - 2))
        * ``yarn``    YaRN frequency ramp: long wavelengths / s, short ones kept
        * ``yarn_t``  YaRN and its attention temperature: q and k times 0.1 ln s + 1

        All of them keep scores a function of relative position at a given length.
        """
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        if scaling not in ("none", "pi", "ntk", "yarn", "yarn_t"):
            raise ValueError(f"unknown RoPE scaling {scaling!r}")
        if scaling != "none" and (train_len is None or train_len < 1):
            raise ValueError(f"RoPE scaling {scaling!r} needs a positive train_len")
        self.head_dim = head_dim
        self.base = base
        self.scaling: Scaling = scaling
        self.train_len = train_len
        inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        # Not a parameter, and not saved: it is a pure function of (head_dim, base).
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.inv_freq: Tensor

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        cos, sin = self._cos_sin(positions)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        mult = self._yarn_multiplier(positions.shape[-1])
        return (q * mult, k * mult) if mult != 1.0 else (q, k)

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None:
        return None

    def _cos_sin(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """``(..., N, d/2)`` angles -> cos/sin shaped to broadcast over ``(B, H, N, d/2)``.

        Angles are built in float32 even under bf16 autocast: at position ~1e3 a bf16
        angle is off by several radians.
        """
        inv_freq, divisor = self._scaled(positions.shape[-1])
        angles = (positions.to(torch.float32) / divisor)[..., None] * inv_freq
        if angles.ndim == 3:  # (B, N, d/2): per-sample positions, add the head axis
            angles = angles[:, None]
        return angles.cos(), angles.sin()

    def _factor(self, n: int) -> float:
        return 1.0 if self.train_len is None else max(1.0, n / self.train_len)

    def _scaled(self, n: int) -> tuple[Tensor, float]:
        """(inverse frequencies, position divisor) for a sequence of length n."""
        s = self._factor(n)
        if self.scaling == "none" or s == 1.0:
            return self.inv_freq, 1.0
        if self.scaling == "pi":
            return self.inv_freq, s
        if self.scaling == "ntk":
            d = self.head_dim
            idx = torch.arange(0, d, 2, device=self.inv_freq.device, dtype=torch.float32)
            return (self.base * s ** (d / (d - 2))) ** (-idx / d), 1.0
        assert self.train_len is not None
        fits = self.train_len * self.inv_freq / (2 * math.pi)  # wavelengths per train_len
        keep = ((fits - YARN_ALPHA) / (YARN_BETA - YARN_ALPHA)).clamp(0.0, 1.0)
        return (1 - keep) * self.inv_freq / s + keep * self.inv_freq, 1.0

    def _yarn_multiplier(self, n: int) -> float:
        s = self._factor(n)
        return 0.1 * math.log(s) + 1.0 if self.scaling == "yarn_t" and s > 1.0 else 1.0

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        return None

    def extra_repr(self) -> str:
        extra = f", scaling={self.scaling!r}, train_len={self.train_len}"
        return f"head_dim={self.head_dim}, base={self.base}" + (
            extra if self.scaling != "none" else ""
        )


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.to(x.dtype)
