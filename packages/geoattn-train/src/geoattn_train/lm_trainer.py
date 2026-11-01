"""Causal LM training loop (task=lm), for experiment 0001.

Run directory contract (``config.yaml``, ``meta.json``,
canonical ``metrics.jsonl``, ``last.pt``), plus ``data.json`` with the corpus
manifest (revision and sha256 of the token files). Training length is a token
budget; one ``metrics.jsonl`` record per evaluation, every ``lm.eval_every``
steps and at the end, with validation loss (nats/token) at every length in
``lm.eval_lengths`` (multiples of ``seq_len``: 2 measures length extrapolation).

Under torchrun each rank computes its slice of the same global batch (ADR-0008,
``distributed.py``); rank 0 alone writes the run directory and mirrors to W&B.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from geoattn_obs import AttentionMonitor
from geoattn_train.config import ExperimentConfig, to_dict
from geoattn_train.distributed import Dist, all_reduce_sum, init_distributed, shard, shutdown
from geoattn_train.factory import build_normalizer, build_pe
from geoattn_train.lm_data import TokenCorpus, TrainBatches, load_corpus, val_batches
from geoattn_train.models import GPT, GPTBlock
from geoattn_train.repro import run_metadata, seed_everything
from geoattn_train.tracking import NullTracker, make_tracker
from geoattn_train.trainer import _make_optimizer, _make_run_dir, resolve_device
from geoattn_train.vram import GB, cap_process_memory


@dataclass
class _Run:
    """State fixed for the whole run, shared by the train step and evaluation."""

    cfg: ExperimentConfig
    model: nn.Module  # what trains: the GPT, or a DistributedDataParallel around it
    core: GPT  # the unwrapped model: evaluation, monitoring, checkpoints
    dist: Dist
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: torch.device
    autocast: AbstractContextManager[Any]
    batches: TrainBatches
    val: dict[int, list[tuple[Tensor, Tensor]]]  # eval length -> fixed batches
    monitor: AttentionMonitor


def build_lm(cfg: ExperimentConfig) -> GPT:
    head_dim = cfg.model.dim_head or cfg.model.dim // cfg.model.heads
    model = GPT(
        cfg.model,
        make_pe=lambda: build_pe(
            cfg.attention,
            head_dim,
            num_heads=cfg.model.heads,
            train_len=cfg.lm.seq_len,
            dim=cfg.model.dim,
            causal=True,
        ),
        make_normalizer=lambda: build_normalizer(
            cfg.attention.normalizer, causal=True, num_heads=cfg.model.heads
        ),
    )
    if cfg.train.compile_normalizer:
        for block in model.blocks:
            assert isinstance(block, GPTBlock) and isinstance(block.attn.normalizer, nn.Module)
            # In place: the module (and hooks on it, e.g. the G4 script's) stays the same.
            compile_normalizer(block.attn.normalizer)
    if cfg.train.compile_model:
        for block in model.blocks:
            # In place, per block: hooks on submodules (monitoring, probes) still attach,
            # and activation checkpointing wraps the compiled block unchanged.
            block.compile(dynamic=False)
    return model


def train_lm(cfg: ExperimentConfig) -> Path:
    seed_everything(cfg.seed)  # same on every rank; DDP also broadcasts rank 0's weights
    d, device = init_distributed(resolve_device(cfg.train.device))
    try:
        return _train(cfg, d, device)
    finally:
        shutdown(d)


def _train(cfg: ExperimentConfig, d: Dist, device: torch.device) -> Path:
    if device.type == "cuda":  # OOM gate: fail this run, never spill into system RAM
        total_gb = torch.cuda.get_device_properties(device).total_memory / GB
        cap_process_memory(cfg.train.vram_limit_gb or 0.9 * total_gb, device)
    corpus = load_corpus(cfg.data, cfg.lm, cfg.model)
    run = _setup(cfg, corpus, device, d)
    run_dir = _run_dir_on_all_ranks(cfg, corpus, run)

    steps = total_steps(cfg)
    tracker = make_tracker(cfg, run_dir) if d.main else NullTracker()
    exit_code = 1
    try:
        window: list[float] = []
        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            window.append(_train_step(run, step - 1))
            if step % cfg.lm.eval_every == 0 or step == steps:
                elapsed = time.perf_counter() - t0
                record = _evaluate(run, step, window, elapsed)
                if d.main:
                    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    tracker.log(record)
                    _print(cfg, record, steps)
                window, t0 = [], time.perf_counter()
        if d.main:
            torch.save({"model": run.core.state_dict(), "step": steps}, run_dir / "last.pt")
        exit_code = 0
    finally:
        tracker.finish(exit_code)
    return run_dir


def compile_normalizer(normalizer: nn.Module) -> None:
    """``torch.compile`` in place, specialized on rho and shapes (``dynamic=False``): with
    automatic dynamism a second instance turns rho into a CPU scalar, which needs a host C++
    compiler on Windows, and a symbolic float buys nothing here.

    A fused normalizer is already a kernel; only its reference fallback (inputs the kernel
    does not cover, e.g. evaluation beyond its maximum length) is compiled."""
    fallback = getattr(normalizer, "reference", None)
    (fallback if isinstance(fallback, nn.Module) else normalizer).compile(dynamic=False)


def tokens_per_step(cfg: ExperimentConfig) -> int:
    return cfg.data.batch_size * cfg.lm.grad_accum * cfg.lm.seq_len


def total_steps(cfg: ExperimentConfig) -> int:
    return max(cfg.lm.token_budget // tokens_per_step(cfg), 1)


def _setup(cfg: ExperimentConfig, corpus: TokenCorpus, device: torch.device, d: Dist) -> _Run:
    if cfg.lm.grad_accum % d.world:
        # Equal micro-batches keep the loss scaling (1 / n_micro) exactly as in one process.
        raise ValueError(f"world size {d.world} must divide lm.grad_accum={cfg.lm.grad_accum}")
    core = build_lm(cfg).to(device)
    model: nn.Module = core
    if d.enabled:
        ids = [device.index] if device.index is not None else None
        model = DistributedDataParallel(core, device_ids=ids)
    optimizer = _make_optimizer(core, cfg)
    micro = cfg.data.batch_size
    batches = TrainBatches(corpus.train, cfg.lm.seq_len, micro * cfg.lm.grad_accum, cfg.data.seed)
    if batches.epochs_needed(total_steps(cfg)) > 1:
        raise ValueError(
            f"token budget needs {batches.epochs_needed(total_steps(cfg)):.2f} passes over "
            f"the corpus; experiment 0001 assumes at most one"
        )
    val = {
        m: val_batches(corpus.val, m * cfg.lm.seq_len, cfg.lm.eval_tokens, micro)
        for m in cfg.lm.eval_lengths
    }
    return _Run(
        cfg=cfg,
        model=model,
        core=core,
        dist=d,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor(cfg)),
        device=device,
        autocast=(
            torch.autocast("cuda", dtype=torch.bfloat16)
            if cfg.train.amp and device.type == "cuda"
            else nullcontext()
        ),
        batches=batches,
        val=val,
        monitor=AttentionMonitor(core, block_type=GPTBlock),
    )


def lr_factor(cfg: ExperimentConfig):
    """Linear warmup, then cosine decay to ``min_lr_ratio`` at the last step."""
    warmup, total, floor = cfg.lm.warmup_steps, total_steps(cfg), cfg.lm.min_lr_ratio

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(total - warmup, 1), 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def _train_step(run: _Run, step: int) -> float:
    """One optimizer step on this rank's slice; returns the rank-local mean loss."""
    run.model.train()
    x, y = (shard(t, run.dist) for t in run.batches.batch(step))
    accum = run.cfg.lm.grad_accum // run.dist.world
    run.optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for i, (xm, ym) in enumerate(zip(x.chunk(accum), y.chunk(accum), strict=True)):
        xm, ym = xm.to(run.device, non_blocking=True), ym.to(run.device, non_blocking=True)
        with _grad_sync(run, last=i == accum - 1):
            with run.autocast:
                loss = F.cross_entropy(run.model(xm).flatten(0, 1), ym.flatten())
            (loss / accum).backward()
        total += loss.item() / accum
    if run.cfg.train.grad_clip is not None:
        nn.utils.clip_grad_norm_(run.model.parameters(), run.cfg.train.grad_clip)
    run.optimizer.step()
    run.scheduler.step()
    return total


def _grad_sync(run: _Run, last: bool) -> AbstractContextManager[Any]:
    """All-reduce gradients once per step: skip the sync on every micro-batch but the last."""
    if last or not isinstance(run.model, DistributedDataParallel):
        return nullcontext()
    return run.model.no_sync()


@torch.no_grad()
def _evaluate(run: _Run, step: int, window: list[float], elapsed: float) -> dict[str, Any]:
    device = run.device
    record: dict[str, Any] = {
        "step": step,
        "tokens": step * tokens_per_step(run.cfg),
        "train_loss": all_reduce_sum([sum(window) / len(window)], run.dist, device)[0]
        / run.dist.world,
        "lr": float(run.scheduler.get_last_lr()[0]),
        "step_time_ms": round(1000 * elapsed / len(window), 2),
        "tokens_per_s": round(len(window) * tokens_per_step(run.cfg) / elapsed),
    }
    if device.type == "cuda":
        record["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_reserved(device) / GB, 3)
        torch.cuda.reset_peak_memory_stats(device)
    run.core.eval()
    for mult, batches in run.val.items():
        record[f"val_loss@{mult * run.cfg.lm.seq_len}"] = _val_loss(run, batches)
    if run.dist.main:
        record.update({f"attn/{k}": v for k, v in _snapshot(run).items()})
    return record


def _val_loss(run: _Run, batches: list[tuple[Tensor, Tensor]]) -> float:
    """Validation batches are split round-robin over ranks; sums are all-reduced."""
    total, count = 0.0, 0
    for x, y in batches[run.dist.rank :: run.dist.world]:
        with run.autocast:
            logits = run.core(x.to(run.device))
        total += F.cross_entropy(
            logits.float().flatten(0, 1), y.to(run.device).flatten(), reduction="sum"
        ).item()
        count += y.numel()
    total, count = all_reduce_sum([total, count], run.dist, run.device)
    return total / count


def _snapshot(run: _Run) -> dict[str, float]:
    """Attention metrics on the first validation sequences at training length, fp32."""
    x, _ = run.val[run.cfg.lm.eval_lengths[0]][0]
    run.monitor.reset()
    with run.monitor.capture():
        run.core(x[: run.cfg.lm.monitor_batch].to(run.device))
    return run.monitor.summary()


def _run_dir_on_all_ranks(cfg: ExperimentConfig, corpus: TokenCorpus, run: _Run) -> Path:
    """Rank 0 creates the run directory; the other ranks learn its path."""
    path = [str(_write_run_header(cfg, corpus, run)) if run.dist.main else ""]
    if run.dist.enabled:
        torch.distributed.broadcast_object_list(path, src=0)
    return Path(path[0])


def _write_run_header(cfg: ExperimentConfig, corpus: TokenCorpus, run: _Run) -> Path:
    device, model = run.device, run.core
    run_dir = _make_run_dir(cfg)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(to_dict(cfg), sort_keys=False), encoding="utf-8"
    )
    meta = run_metadata(device)
    meta["params"] = {"total": model.num_params(), "non_embedding": model.num_params(False)}
    meta["steps"], meta["tokens_per_step"] = total_steps(cfg), tokens_per_step(cfg)
    meta["distributed"] = {"world_size": run.dist.world, "backend": run.dist.backend}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    data = {"dataset": cfg.data.dataset, "train_tokens_used": len(corpus.train)}
    data["manifest"] = corpus.manifest
    (run_dir / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return run_dir


def _print(cfg: ExperimentConfig, record: dict[str, Any], steps: int) -> None:
    vals = "  ".join(f"{k} {v:.4f}" for k, v in record.items() if k.startswith("val_loss@"))
    print(
        f"[{cfg.name}] step {record['step']:>6}/{steps}  train_loss {record['train_loss']:.4f}  "
        f"{vals}  ({record['tokens_per_s']:,} tok/s)"
    )
