"""ALiBi (Press et al. 2022): a per-head linear distance penalty on the scores.

The additive group action of docs/methods.md §2 in its simplest form: ``bias`` adds
``-m_h |i - j|`` to the score of query position i and key position j, with one slope
``m_h`` per head, and ``act`` leaves q and k unchanged. It depends on positions only
through ``i - j`` (translation invariant), so it applies unchanged at any length; that is
why it extrapolates. In GRAPE (arXiv:2512.07805) it is the distance-only special case of
the unipotent GL(d+k) lift.

Slopes follow the paper: a geometric sequence ``2^(-8 h / H)``, h = 1..H, extended for a
head count H that is not a power of two by interleaving the next power's sequence.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from geoattn_core.flex import ScoreMod


def alibi_slopes(num_heads: int) -> Tensor:
    def geometric(n: int) -> list[float]:
        return [2.0 ** (-8.0 * (h + 1) / n) for h in range(n)]

    closest = 2 ** math.floor(math.log2(num_heads))
    slopes = geometric(closest)
    if closest < num_heads:  # the paper's rule for H not a power of two
        slopes += geometric(2 * closest)[0::2][: num_heads - closest]
    return torch.tensor(slopes, dtype=torch.float32)


class ALiBi(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"ALiBi needs at least one head, got {num_heads}")
        self.num_heads = num_heads
        # Not a parameter, and not saved: a pure function of num_heads.
        self.register_buffer("slopes", alibi_slopes(num_heads), persistent=False)
        self.slopes: Tensor

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return q, k

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor:
        """``(H, N_q, N_k)`` for ``(N,)`` positions, ``(B, H, N_q, N_k)`` for ``(B, N)``."""
        dist = (positions_q[..., :, None] - positions_k[..., None, :]).abs().to(torch.float32)
        dist = dist.unsqueeze(-3)  # head axis
        return -self.slopes[:, None, None] * dist

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        slopes = self.slopes
        return lambda s, b, h, qi, kv: s - slopes[h] * (qi - kv)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}"
