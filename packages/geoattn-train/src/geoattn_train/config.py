"""Typed experiment configuration.

YAML files map 1:1 onto these dataclasses. Unknown keys are an error (a
typo'd hyperparameter silently falling back to its default is the classic
way to waste a sweep). CLI overrides use dotted paths: ``train.lr=3e-4``.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str = "cifar10"  # vit: cifar10 | mnist | synthetic; lm: tokens | synthetic
    root: str = "data"  # lm + tokens: directory with train.bin / val.bin (prepare_fineweb.py)
    batch_size: int = 128
    num_workers: int = 4
    synthetic_size: int = 512  # only for dataset=synthetic (tests / smoke runs)
    # lm: data order. Shared by every run of a stage (experiment 0001 §3.3), so it is
    # separate from the top-level seed, which then only varies initialization.
    seed: int = 0


@dataclass
class ModelConfig:
    image_size: int = 32
    in_channels: int = 3
    patch_size: int = 4
    dim: int = 192
    depth: int = 6
    heads: int = 3
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    num_classes: int = 10
    dim_head: int | None = None  # None = dim // heads; upstream ViT uses a fixed 64
    mlp: bool = True  # False = attention-only blocks (upstream ViT_only_Att, MNIST)
    qkv_bias: bool = True  # upstream: False
    # Causal LM only (task=lm). GPT-2 has 50257 ids; 50304 = next multiple of 64.
    vocab_size: int = 50304
    checkpoint: bool = False  # recompute each block in backward (activation checkpointing)
    # reference | flex: FlexAttention for causal softmax with a fusable encoding (ADR-0012)
    attn_backend: str = "reference"


@dataclass
class NormalizerConfig:
    name: str = "softmax"  # softmax | sinkhorn | unbalanced | scaled_softmax
    eps: float = 1.0
    n_iters: int = 3  # sinkhorn / unbalanced; 1 == softmax
    rho: float = 0.5  # unbalanced only: lambda / (lambda + eps); 0 == softmax (ADR-0006)
    column_mass: str = "prefix"  # unbalanced only: prefix (causal-safe) | full (ADR-0005)
    # scaled_softmax only: logits x s_h log(n visible keys), s_h learnable per head,
    # initialized to 1 / log(scale_base_len) (report 0005)
    scale_base_len: int = 512
    scale_learnable: bool = True
    # unbalanced only: "unroll" (backprop through the iterations) | "implicit" (fixed-point
    # adjoint to relative backward_tol; memory independent of n_iters, report 0006)
    grad_mode: str = "unroll"
    backward_tol: float = 1e-4
    # unbalanced only: "torch" (geoattn-core reference) | "fused" (geoattn-kernels Triton kernel
    # for the causal CUDA case, reference elsewhere; same values to fp32 rounding, ADR-0009)
    impl: str = "torch"
    # unbalanced only: one (each column's prefix mass toward 1) | uniform (toward the mass
    # uniform attention gives; removes the built-in distance decay)
    column_target: str = "one"


@dataclass
class AttentionConfig:
    pe: str = "identity"  # identity | rope | alibi | grape_m | grape_a | grape_ap | fox | cable
    rope_base: float = 10_000.0
    # rope only: none | pi | ntk | yarn | yarn_t (beyond rope_train_len; 0 = the LM's seq_len)
    rope_scaling: str = "none"
    rope_train_len: int = 0
    grape_p_dim: int = 8  # grape_ap: per-head width of the p projection
    # additive bias on top of pe: none | alibi | logbias (-c_h log(1 + |i - j|), c_h per head)
    # | kerple (-p_h log(1 + a_h |i - j|)) | gated_logbias (c_h(x_i) per query; experiment 0002 P6)
    # | clock_logbias (the same on a content clock; experiment 0002 LC, causal only)
    pe_bias: str = "none"
    logbias_learnable: bool = True
    # logbias only: free (c > 0) | gt1 (c > 1, summable tail) | lt1 (c < 1); experiment 0003
    logbias_exponent: str = "free"
    # gt1 only: c > 1 + logbias_margin (experiment 0004, B); 0 is experiment 0003's gt1
    logbias_margin: float = 0.0
    # Evaluation-time change of the bias beyond distance tail_from (experiment 0003, B2):
    # none | cap | repair | window. tail_from 0 = the LM's seq_len; tail_slope is window's
    # m, 0 = the geometric mean of the ALiBi slopes for this head count.
    tail_mode: str = "none"
    tail_from: int = 0
    tail_slope: float = 0.0
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float | None = 1.0
    label_smoothing: float = 0.0
    amp: bool = True  # bf16 autocast on CUDA; ignored on CPU
    device: str = "auto"
    monitor_every: int = 1  # epochs between attention-metric snapshots; 0 = off
    max_steps_per_epoch: int | None = None  # for smoke runs
    optimizer: str = "adamw"  # adamw | adam (adam: no weight decay, as upstream)
    schedule: str = "cosine"  # cosine (warmup + cosine) | step (x lr_gamma at each milestone)
    lr_milestones: list[int] = field(default_factory=list)  # 1-based epochs, schedule=step
    lr_gamma: float = 0.1
    # Allocator cap per process (CUDA only). None = 90% of the device. Past it the run
    # fails with OutOfMemoryError instead of spilling into system RAM (see vram.py).
    vram_limit_gb: float | None = None
    # torch.compile each attention normalizer (fuses the per-iteration elementwise ops;
    # ~5x faster UnbalancedSinkhorn). Same math, float32 rounding may differ slightly.
    compile_normalizer: bool = False
    # LM only: torch.compile each GPT block (fuses LayerNorm, MLP and the attention glue).
    # Same math; bf16 rounding order may differ slightly. Off by default.
    compile_model: bool = False
    # AdamW's fused CUDA kernel (one launch for all parameters). Same update to fp32
    # rounding. Off by default; needs CUDA parameters.
    fused_optimizer: bool = False


@dataclass
class TrackingConfig:
    """Mirror of metrics.jsonl to Weights & Biases (ADR-0004). JSONL stays canonical."""

    wandb: bool = False
    project: str = "attention-geometry"
    entity: str | None = None
    group: str | None = None  # default: the experiment name (seeds of one setting together)
    tags: list[str] = field(default_factory=list)
    mode: str = "online"  # online | offline | disabled


@dataclass
class LMConfig:
    """Causal LM training (task=lm). Length is a token budget, not epochs."""

    seq_len: int = 512
    token_budget: int = 500_000_000  # training tokens; steps = budget // tokens per step
    corpus_tokens: int | None = None  # use only the first N tokens of train.bin
    grad_accum: int = 1
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1  # cosine decays to lr * min_lr_ratio
    eval_every: int = 500  # steps
    eval_tokens: int = 2_000_000  # validation tokens per evaluated length
    eval_lengths: list[int] = field(default_factory=lambda: [1, 2])  # multiples of seq_len
    monitor_batch: int = 4  # sequences in the attention-metric snapshot


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    task: str = "vit"  # vit (image classification) | lm (causal language model)
    seed: int = 0
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    lm: LMConfig = field(default_factory=LMConfig)


# ---------------------------------------------------------------- loading


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    return from_dict(ExperimentConfig, raw)


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def from_dict(cls: type, data: dict[str, Any], _path: str = ""):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown config key(s) at '{_path or '<root>'}': {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        tp = hints[name]
        where = f"{_path}.{name}" if _path else name
        if dataclasses.is_dataclass(tp):
            if not isinstance(value, dict):
                raise TypeError(f"'{where}' must be a mapping")
            kwargs[name] = from_dict(tp, value, where)  # type: ignore[arg-type]
        else:
            kwargs[name] = _coerce(tp, value, where)
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(inner, value, where)
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"'{where}': expected a list, got {value!r}")
        (inner,) = typing.get_args(tp)
        return [_coerce(inner, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if tp is float and isinstance(value, int | float | str) and not isinstance(value, bool):
        return float(value)  # YAML reads "1e-3" as a string
    if tp is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if tp is bool and isinstance(value, bool):
        return value
    if tp is str and isinstance(value, str):
        return value
    raise TypeError(f"'{where}': expected {tp}, got {value!r}")


def _set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value
