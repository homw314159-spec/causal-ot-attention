"""Causal LM data order, schedule and an end-to-end run on synthetic tokens."""

import json
import math

import numpy as np
import pytest
import torch

from geoattn_train.config import load_config
from geoattn_train.lm_data import TrainBatches, synthetic_corpus, val_batches
from geoattn_train.lm_trainer import build_lm, lr_factor, total_steps, train_lm
from geoattn_train.trainer import _make_optimizer

SMOKE = "configs/smoke_lm.yaml"


def test_targets_are_inputs_shifted_by_one() -> None:
    """A misaligned target is a data-pipeline leak the operator tests cannot see."""
    tokens = np.arange(1000)
    x, y = TrainBatches(tokens, seq_len=16, batch_size=4, seed=0).batch(3)
    torch.testing.assert_close(y[:, :-1], x[:, 1:])
    torch.testing.assert_close(y, x + 1)  # arange: the next token is +1


def test_batches_depend_only_on_data_seed_and_step() -> None:
    tokens = np.random.default_rng(0).integers(0, 50, 5000)
    a = TrainBatches(tokens, 32, 8, seed=7)
    b = TrainBatches(tokens, 32, 8, seed=7)
    for step in (0, 5, 3, 5):  # any order, recomputable
        torch.testing.assert_close(a.batch(step), b.batch(step))
    assert not torch.equal(a.batch(0)[0], TrainBatches(tokens, 32, 8, seed=8).batch(0)[0])


def test_one_epoch_uses_each_window_once() -> None:
    tokens = np.arange(32 * 40 + 1)
    batches = TrainBatches(tokens, seq_len=32, batch_size=8, seed=1)
    starts = torch.cat([batches.batch(s)[0][:, 0] for s in range(40 // 8)])
    assert sorted(starts.tolist()) == list(range(0, 32 * 40, 32))


def test_validation_batches_are_fixed_prefix() -> None:
    tokens = np.arange(10_000)
    short = val_batches(tokens, seq_len=64, n_tokens=1024, batch_size=4)
    long = val_batches(tokens, seq_len=128, n_tokens=1024, batch_size=4)
    assert sum(x.numel() for x, _ in short) == sum(x.numel() for x, _ in long) == 1024
    assert short[0][0][0, 0] == long[0][0][0, 0] == 0


def test_schedule_warmup_then_cosine_to_floor() -> None:
    cfg = load_config(SMOKE)
    f, steps = lr_factor(cfg), total_steps(cfg)
    assert f(0) == pytest.approx(1 / cfg.lm.warmup_steps)
    assert f(cfg.lm.warmup_steps) == pytest.approx(1.0)
    assert f(steps) == pytest.approx(cfg.lm.min_lr_ratio)


def test_budget_beyond_one_epoch_is_refused(tmp_path) -> None:
    overrides = [f"output_dir={tmp_path.as_posix()}", "lm.token_budget=10000000"]
    cfg = load_config(SMOKE, [*overrides, "train.device=cpu"])
    with pytest.raises(ValueError, match="passes over"):
        train_lm(cfg)


def test_smoke_run_learns_and_writes_artifacts(tmp_path) -> None:
    cfg = load_config(SMOKE, [f"output_dir={tmp_path.as_posix()}", "train.device=cpu"])
    run_dir = train_lm(cfg)
    assert {p.name for p in run_dir.iterdir()} >= {
        "config.yaml",
        "meta.json",
        "data.json",
        "metrics.jsonl",
        "last.pt",
    }
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [20, 40, 60]
    last = records[-1]
    assert last["tokens"] == cfg.lm.token_budget
    assert last["val_loss@64"] < 0.6 * math.log(cfg.model.vocab_size)  # it learned the rule
    assert math.isfinite(last["val_loss@128"])  # evaluated at 2x the training length
    assert any(k.endswith("/sink") for k in last)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["params"]["total"] > meta["params"]["non_embedding"]


def test_synthetic_corpus_is_seeded() -> None:
    a, b = synthetic_corpus(64, 500, seed=3), synthetic_corpus(64, 500, seed=3)
    assert np.array_equal(a.train, b.train) and a.train.max() < 64


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_smoke_run_on_gpu(tmp_path) -> None:
    """device=auto, VRAM cap, bf16 autocast, checkpointing: the path real runs take."""
    overrides = [f"output_dir={tmp_path.as_posix()}", "train.device=auto", "model.checkpoint=true"]
    run_dir = train_lm(load_config(SMOKE, overrides))
    last = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    assert last["gpu_mem_peak_gb"] > 0 and math.isfinite(last["val_loss@128"])


def test_scaled_softmax_trains_and_its_scale_is_not_decayed(tmp_path) -> None:
    overrides = [
        f"output_dir={tmp_path.as_posix()}",
        "train.device=cpu",
        "attention.normalizer.name=scaled_softmax",
        "attention.normalizer.scale_base_len=64",
    ]
    cfg = load_config(SMOKE, overrides)
    model = build_lm(cfg)
    optimizer = _make_optimizer(model, cfg)
    scales = {id(p) for n, p in model.named_parameters() if n.endswith("normalizer.scale")}
    assert len(scales) == cfg.model.depth
    decayed = {id(p) for g in optimizer.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert not scales & decayed
    run_dir = train_lm(cfg)
    last = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    assert last["val_loss@64"] < 0.6 * math.log(cfg.model.vocab_size)
