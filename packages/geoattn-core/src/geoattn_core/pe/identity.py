from __future__ import annotations

from torch import Tensor, nn

from geoattn_core.flex import ScoreMod


class IdentityPE(nn.Module):
    """No positional action inside attention (trivial group representation).

    Used for the Sinkformer ViT reproduction, where position enters through a
    learned absolute embedding added to the tokens at the input instead.
    Other implementations (RoPE, and GRAPE / LieRE in P3) live alongside this file.
    """

    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return q, k

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None:
        return None

    def score_mod(self, q: Tensor, k: Tensor, x: Tensor, positions: Tensor) -> ScoreMod | None:
        return None
