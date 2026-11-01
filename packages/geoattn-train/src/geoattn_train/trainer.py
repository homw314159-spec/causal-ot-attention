"""Shared training helpers; numerical bodies preserved from the original trainer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from geoattn_train.config import ExperimentConfig


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No weight decay on biases, norms, position embeddings, position-encoding or
        # normalizer parameters (e.g. ScaledSoftmax's temperature, GRAPE-M's log-frequencies:
        # decay would pull them toward 0).
        exempt = ("pos_embed", "cls_token", ".normalizer.", ".pe.")
        (no_decay if p.ndim < 2 or any(e in name for e in exempt) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    match cfg.train.optimizer:
        case "adamw":
            return torch.optim.AdamW(groups, lr=cfg.train.lr, fused=cfg.train.fused_optimizer)
        case "adam":  # plain Adam, no weight decay (upstream Sinkformers)
            if cfg.train.weight_decay:
                raise ValueError("optimizer=adam takes no weight decay; set train.weight_decay=0")
            return torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        case other:
            raise ValueError(f"unknown optimizer {other!r}")


def _make_run_dir(cfg: ExperimentConfig) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.output_dir) / cfg.name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
