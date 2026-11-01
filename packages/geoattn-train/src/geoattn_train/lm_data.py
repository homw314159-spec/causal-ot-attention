"""Token streams for causal LM training (task=lm).

A corpus is a flat 1-D token array (``train.bin`` / ``val.bin`` from
``scripts/prepare_fineweb.py``, read through ``np.memmap``). Training draws
non-overlapping windows of ``seq_len + 1`` tokens (inputs and next-token targets)
in an order that is a pure function of ``(data seed, step)``: every run of a stage
sees exactly the same batches (experiment 0001 §3.3), on any machine, and a batch
can be recomputed without replaying the loader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from geoattn_train.config import DataConfig, LMConfig, ModelConfig


@dataclass
class TokenCorpus:
    train: np.ndarray  # 1-D, integer dtype (uint16 memmap for real data)
    val: np.ndarray
    manifest: dict | None = None  # from manifest.json: revision, sha256, token counts


def load_corpus(data: DataConfig, lm: LMConfig, model: ModelConfig) -> TokenCorpus:
    if data.dataset == "synthetic":
        return synthetic_corpus(model.vocab_size, data.synthetic_size, data.seed)
    if data.dataset != "tokens":
        raise ValueError(f"unknown LM dataset {data.dataset!r} (tokens | synthetic)")
    root = Path(data.root)
    train = np.memmap(root / "train.bin", dtype=np.uint16, mode="r")
    if lm.corpus_tokens is not None:
        if lm.corpus_tokens > len(train):
            raise ValueError(f"corpus_tokens={lm.corpus_tokens} > {len(train)} in train.bin")
        train = train[: lm.corpus_tokens]
    val = np.memmap(root / "val.bin", dtype=np.uint16, mode="r")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else None
    return TokenCorpus(train=train, val=val, manifest=manifest)


def synthetic_corpus(vocab_size: int, n_tokens: int, seed: int) -> TokenCorpus:
    """A learnable stream: each token is an affine function of the previous one,
    with 10% noise, so a working model's loss falls well below log(vocab)."""
    rng = np.random.default_rng(seed)

    def stream(n: int) -> np.ndarray:
        x = np.empty(n, dtype=np.int64)
        x[0] = 0
        noise = rng.random(n) < 0.1
        rand = rng.integers(0, vocab_size, n)
        for i in range(1, n):
            x[i] = rand[i] if noise[i] else (3 * x[i - 1] + 1) % vocab_size
        return x

    return TokenCorpus(train=stream(n_tokens), val=stream(max(n_tokens // 4, 1024)))


class TrainBatches:
    """``batch(step)`` -> ``(inputs, targets)``, each ``(B, seq_len)`` int64.

    Windows are a seeded permutation of the non-overlapping windows of the corpus;
    a token budget larger than the corpus wraps into a fresh permutation (reported
    by ``epochs_needed`` so the caller can refuse or log it).
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, batch_size: int, seed: int) -> None:
        self.tokens, self.seq_len, self.batch_size, self.seed = tokens, seq_len, batch_size, seed
        self.n_windows = (len(tokens) - 1) // seq_len
        if self.n_windows < batch_size:
            raise ValueError(f"corpus of {len(tokens)} tokens has < {batch_size} windows")
        self._perms: dict[int, np.ndarray] = {}

    def epochs_needed(self, steps: int) -> float:
        return steps * self.batch_size / self.n_windows

    def batch(self, step: int) -> tuple[Tensor, Tensor]:
        per_epoch = self.n_windows // self.batch_size
        epoch, k = divmod(step, per_epoch)
        idx = self._perm(epoch)[k * self.batch_size : (k + 1) * self.batch_size]
        return _windows(self.tokens, idx * self.seq_len, self.seq_len)

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._perms:
            self._perms = {
                epoch: np.random.default_rng([self.seed, epoch]).permutation(self.n_windows)
            }
        return self._perms[epoch]


def val_batches(
    tokens: np.ndarray, seq_len: int, n_tokens: int, batch_size: int
) -> list[tuple[Tensor, Tensor]]:
    """Fixed, ordered windows from the start of ``tokens`` covering ~``n_tokens``
    predictions. The same prefix of the validation stream is used at every length."""
    n_windows = min(max(n_tokens // seq_len, 1), (len(tokens) - 1) // seq_len)
    if n_windows < 1:
        raise ValueError(f"validation stream of {len(tokens)} tokens is shorter than {seq_len}")
    starts = np.arange(n_windows) * seq_len
    return [
        _windows(tokens, starts[i : i + batch_size], seq_len)
        for i in range(0, n_windows, batch_size)
    ]


def _windows(tokens: np.ndarray, starts: np.ndarray, seq_len: int) -> tuple[Tensor, Tensor]:
    rows = np.stack([np.asarray(tokens[s : s + seq_len + 1], dtype=np.int64) for s in starts])
    x = torch.from_numpy(rows)
    return x[:, :-1], x[:, 1:]
