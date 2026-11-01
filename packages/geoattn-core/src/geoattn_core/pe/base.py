"""The WHERE axis: how positions enter the score (negative cost) matrix.

A group-representational encoding acts in two possible ways (GRAPE,
arXiv:2512.07805), and the protocol has one hook for each:

* multiplicative — ``act``: a group element acts on q and k before the
  inner product (RoPE: SO(2)^{d/2}; LieRE: SO(n); HoPE: Lorentz boosts);
* additive — ``bias``: a term added to the scores (ALiBi, and GRAPE's
  unipotent GL(d+k) lift).

Implementations should subclass ``nn.Module`` so any learnable parameters
(e.g. LieRE generators) are registered with the owning attention layer.
The hook is deliberately NOT named ``apply``: ``nn.Module.apply(fn)`` is
PyTorch's recursive tree walk (``model.apply(init_fn)``), and overriding it
would break that for the whole model (ADR-0002).

Shapes: ``q``, ``k`` are ``(B, H, N, d_head)``; ``positions`` is ``(N,)`` or
``(B, N)``. ``bias`` returns ``None`` or a tensor broadcastable to
``(B, H, N_q, N_k)``.

Optional (ADR-0011): an encoding whose additive term depends on content defines
``bias_ctx(q, k, x, positions)`` — q and k after ``act``, ``x`` the attention layer's
input ``(B, N, dim)`` — and the attention layer calls it instead of ``bias``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class PositionEncoding(Protocol):
    def act(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]: ...

    def bias(self, positions_q: Tensor, positions_k: Tensor) -> Tensor | None: ...
