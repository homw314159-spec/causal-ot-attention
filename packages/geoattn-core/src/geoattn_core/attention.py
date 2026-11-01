from __future__ import annotations

import torch
from torch import Tensor, nn

from geoattn_core import flex
from geoattn_core.normalizers.base import Normalizer
from geoattn_core.normalizers.softmax import Softmax
from geoattn_core.pe.base import PositionEncoding


class AttentionProbe(nn.Identity):
    """Identity module applied to the attention coupling ``P``.

    It exists only as a stable hook point: observability code registers
    forward hooks on every ``AttentionProbe`` to read ``P`` without core
    knowing anything about metrics (core must not depend on geoattn-obs).
    The probe sits *before* attention dropout, so hooks see the coupling the
    normalizer actually produced.
    """


class GeometricAttention(nn.Module):
    """Multi-head attention = WHERE (position encoding) x HOW MUCH (normalizer).

    Reference implementation: materializes the full ``(B, H, N, N)`` coupling.
    Setting ``backend = "flex"`` (ADR-0012) computes causal softmax attention with
    FlexAttention when the call allows it (see :meth:`uses_flex`) and falls back to the
    reference otherwise, so both backends give the same values.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        pe: PositionEncoding,
        normalizer: Normalizer,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        head_dim: int | None = None,
    ) -> None:
        """``head_dim=None`` means ``dim // num_heads``; set it to decouple the inner
        width ``num_heads * head_dim`` from ``dim`` (as in lucidrains' ViT, ``dim_head``)."""
        super().__init__()
        self.backend = "reference"
        if head_dim is None:
            if dim % num_heads != 0:
                raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
            head_dim = dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * self.inner_dim, bias=qkv_bias)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.pe = pe
        self.normalizer = normalizer
        self.probe = AttentionProbe()
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: Tensor,
        positions: Tensor | None = None,
        mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        """``x``: ``(B, N, dim)``; ``mask``: bool, broadcastable to ``(B, H, N, N)``.

        ``causal=True`` declares that ``mask`` is the causal mask and ``positions`` are
        ``0..N-1`` (the caller's contract; not checked, to avoid a device sync), which lets
        the flex backend skip the mask tensor."""
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, head_dim)

        if positions is None:
            positions = torch.arange(n, device=x.device)
        q, k = self.pe.act(q, k, positions)
        if causal and self.uses_flex(x):
            mod = self.pe.score_mod(q, k, x, positions)  # type: ignore[attr-defined]
            out = flex.causal_attention(q, k, v, mod, self.scale)
            return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, self.inner_dim)))

        scores = (q @ k.transpose(-2, -1)) * self.scale
        # An encoding whose bias depends on content defines bias_ctx (ADR-0011).
        bias_ctx = getattr(self.pe, "bias_ctx", None)
        bias = bias_ctx(q, k, x, positions) if bias_ctx else self.pe.bias(positions, positions)
        if bias is not None:
            scores = scores + bias

        coupling = self.probe(self.normalizer(scores, mask))
        out = self.attn_drop(coupling) @ v
        out = out.transpose(1, 2).reshape(b, n, self.inner_dim)
        return self.proj_drop(self.proj(out))

    def uses_flex(self, x: Tensor) -> bool:
        """Whether a causal call takes the FlexAttention path: CUDA input, plain softmax at
        temperature 1, no attention dropout, a fusable encoding, and no hook on the probe
        (monitoring needs the coupling, which FlexAttention does not form)."""
        pe = self.pe
        return (
            self.backend == "flex"
            and x.is_cuda
            and type(self.normalizer) is Softmax
            and self.normalizer.eps == 1.0
            and self.attn_drop.p == 0.0
            and getattr(pe, "fusable", hasattr(pe, "score_mod"))
            and not self.probe._forward_hooks
        )

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, value: str) -> None:
        if value not in ("reference", "flex"):
            raise ValueError(f"unknown attention backend {value!r}")
        self._backend = value
