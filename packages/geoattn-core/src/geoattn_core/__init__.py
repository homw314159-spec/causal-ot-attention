"""Reference implementations of geometric attention.

Attention is decomposed along two orthogonal axes (see docs/methods.md §2):

* WHERE    — :class:`PositionEncoding`: how position enters the cost matrix.
* HOW MUCH — :class:`Normalizer`: how the cost matrix becomes a coupling.

Everything here is plain PyTorch and is the numerical ground truth that
later fused kernels (stage 2) must match.
"""

from geoattn_core.attention import AttentionProbe, GeometricAttention
from geoattn_core.normalizers import (
    Normalizer,
    ScaledSoftmax,
    Sinkhorn,
    Softmax,
    UnbalancedSinkhorn,
)
from geoattn_core.pe import ALiBi, IdentityPE, PositionEncoding, RoPE

__all__ = [
    "ALiBi",
    "AttentionProbe",
    "GeometricAttention",
    "IdentityPE",
    "Normalizer",
    "PositionEncoding",
    "RoPE",
    "ScaledSoftmax",
    "Sinkhorn",
    "Softmax",
    "UnbalancedSinkhorn",
]
