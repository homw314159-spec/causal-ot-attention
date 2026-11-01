"""A small pre-norm decoder-only LM whose attention is a :class:`GeometricAttention`.

Position enters only through the attention PE (RoPE in experiment 0001), never as
an input embedding, so the model can be evaluated beyond its training context.
The causal mask is built here, so every normalizer sees the same mask; whether a
normalizer is legal under it (prefix consistency, ADR-0005) is checked by the
factory, not by the model, so tests can build a deliberately leaky GPT.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from geoattn_core import GeometricAttention, Normalizer, PositionEncoding
from geoattn_train.config import ModelConfig
from geoattn_train.models.mlp import MLP


class GPTBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, pe: PositionEncoding, normalizer: Normalizer) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = GeometricAttention(
            cfg.dim,
            cfg.heads,
            pe=pe,
            normalizer=normalizer,
            qkv_bias=cfg.qkv_bias,
            attn_dropout=cfg.attn_dropout,
            proj_dropout=cfg.dropout,
            head_dim=cfg.dim_head,
        )
        self.attn.backend = cfg.attn_backend
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg.dim, int(cfg.dim * cfg.mlp_ratio), cfg.dropout)

    def forward(self, x: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), positions=positions, mask=mask, causal=True)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        make_pe: Callable[[], PositionEncoding],
        make_normalizer: Callable[[], Normalizer],
    ) -> None:
        super().__init__()
        self.checkpoint = cfg.checkpoint
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        # One PE / normalizer instance per block: learnable PEs must not share weights.
        self.blocks = nn.ModuleList(
            GPTBlock(cfg, make_pe(), make_normalizer()) for _ in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight  # weight tying (GPT-2)
        self._init_weights(cfg.depth)

    def _init_weights(self, depth: int) -> None:
        """GPT-2: N(0, 0.02), residual output projections scaled by 1/sqrt(2 * depth)."""
        for m in self.modules():
            if isinstance(m, nn.Linear | nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        for block in self.blocks:
            assert isinstance(block, GPTBlock)
            for proj in (block.attn.proj, block.mlp.net[-2]):
                assert isinstance(proj, nn.Linear)
                nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * depth))

    def forward(self, idx: Tensor) -> Tensor:
        """``idx``: ``(B, N)`` token ids -> logits ``(B, N, vocab_size)``."""
        n = idx.shape[1]
        positions = torch.arange(n, device=idx.device)
        mask = torch.ones(n, n, dtype=torch.bool, device=idx.device).tril()
        x = self.drop(self.tok_embed(idx))
        for block in self.blocks:
            if self.checkpoint and self.training:
                x = checkpoint(block, x, positions, mask, use_reentrant=False)
            else:
                x = block(x, positions, mask)
        return self.head(self.norm(x))

    def num_params(self, embedding: bool = True) -> int:
        """Parameter count; tied weights counted once. ``embedding=False`` drops the
        token embedding (the usual "non-embedding" size)."""
        total = sum(p.numel() for p in self.parameters())
        return total if embedding else total - self.tok_embed.weight.numel()
