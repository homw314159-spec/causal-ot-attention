"""Causal softmax attention through FlexAttention (ADR-0012).

A position encoding that can be fused defines ``score_mod(q, k, x, positions)`` and returns
either ``None`` (no additive term) or a function ``(score, b, h, q_idx, kv_idx) -> score``
that adds exactly what its ``bias`` / ``bias_ctx`` would add at ``(b, h, q_idx, kv_idx)``
for ``kv_idx <= q_idx``, with positions ``0..N-1``. Above the diagonal it must still be
finite: blocks that straddle the diagonal evaluate it there before masking, and a NaN
(e.g. ``log1p`` of a negative distance) poisons the gradients of captured tensors.

Per-token tensors that need gradients may be indexed only once inside a ``score_mod``
(a FlexAttention restriction): an encoding that reads the same tensor at ``q_idx`` and
``kv_idx`` passes a ``clone()`` for the second read, and autograd adds the two gradients.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

ScoreMod = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]

RECOMPILE_LIMIT = 64
_compiled = torch.compile(flex_attention, dynamic=False)
_masks: dict[tuple[int, torch.device], BlockMask] = {}


def _causal(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    return q_idx >= kv_idx


def causal_block_mask(n: int, device: torch.device) -> BlockMask:
    key = (n, device)
    if key not in _masks:
        _masks[key] = create_block_mask(_causal, None, None, n, n, device=device)
    return _masks[key]


def per_query(t: Tensor, q: Tensor) -> Tensor:
    """A per-head tensor ``(H,)`` laid out per query, ``(B, H, N)`` and contiguous.

    Read inside a ``score_mod`` as ``t[h]``, a tensor that needs gradients makes every
    (query, key) pair of a head add its gradient to the same address, and the backward
    pass serializes on those atomics (3.5x slower than the reference in calibration).
    Indexed as ``per_query(t, q)[b, h, q_idx]`` the gradient is spread over B x H x N
    addresses; autograd sums it back to ``t``.
    """
    b, h, n, _ = q.shape
    return t[None, :, None].expand(b, h, n).contiguous()


def compose(first: ScoreMod | None, second: ScoreMod | None) -> ScoreMod | None:
    if first is None or second is None:
        return first or second
    return lambda s, b, h, q, kv: second(first(s, b, h, q, kv), b, h, q, kv)


def causal_attention(
    q: Tensor, k: Tensor, v: Tensor, score_mod: ScoreMod | None, scale: float
) -> Tensor:
    """``softmax(scale * q k^T + bias, causal) v`` for ``(B, H, N, D)`` inputs.

    Every encoding and every length compiles its own kernel. Past dynamo's recompile limit
    (8 by default, soon reached by an evaluation sweep over lengths) dynamo would silently
    run the unfused eager ``flex_attention``, which materializes the scores and in bf16 gave
    wrong gradients in testing. So the limit is raised for this call and hitting it is an
    error, not a fallback.
    """
    mask = causal_block_mask(q.shape[-2], q.device)
    with torch._dynamo.config.patch(
        recompile_limit=RECOMPILE_LIMIT, fail_on_recompile_limit_hit=True
    ):
        out = _compiled(q, k, v, score_mod=score_mod, block_mask=mask, scale=scale)
    assert isinstance(out, Tensor)
    return out
