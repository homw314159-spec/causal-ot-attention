"""Experiment 0002 offline evaluation: write ``eval_offline.json`` into each finished run.

    uv run python scripts/exp0002_eval.py "runs/exp0002_screen_*"
    uv run python scripts/exp0002_eval.py "runs/exp0002_confirm_*" --retrieval

For each run directory (config.yaml + last.pt), from the final checkpoint:

* ``loss``: validation loss at 1x, 2x, 4x, 8x the training length, on the first
  ``--tokens`` predictions of the validation stream (the same prefix at every length);
* ``loss_yarn_t``: the same at 2x-8x with RoPE scaled by YaRN-t at evaluation time, for
  arms whose base encoding is RoPE (no training; §3.1);
* with ``--retrieval``: passkey and needle retrieval rates at 1x-8x and five depths
  (``geoattn_train.retrieval``; the needle's haystack is validation text taken after the
  loss prefix).

``scripts/exp0002_decide.py`` reads these files. Evaluation does not depend on the GPU
(report 0009; gate G5 re-checks it between DGX Spark and A100), so runs trained on rented
GPUs are evaluated locally. Inference only, bf16 autocast.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import subprocess
from pathlib import Path

import tiktoken
import torch
from torch import nn
from torch.nn import functional as F

from geoattn_train.config import ExperimentConfig, load_config
from geoattn_train.lm_data import load_corpus, val_batches
from geoattn_train.lm_trainer import build_lm
from geoattn_train.retrieval import Grid, Sample, needle, passkey, run_grid

MULTS = (1, 2, 4, 8)


def encode(text: str) -> list[int]:
    # Load vocabulary only when retrieval needs it; importing loss helpers stays offline.
    return tiktoken.get_encoding("gpt2").encode_ordinary(text)


def load_model(cfg: ExperimentConfig, run_dir: Path, device: torch.device) -> nn.Module:
    model = build_lm(cfg).to(device).eval()
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return model


def with_yarn_t(cfg: ExperimentConfig) -> ExperimentConfig | None:
    """The run's config with YaRN-t evaluation-time scaling, or None if the base is not RoPE."""
    if cfg.attention.pe != "rope":
        return None
    out = copy.deepcopy(cfg)
    out.attention.rope_scaling = "yarn_t"
    return out


@torch.no_grad()
def mean_loss(model: nn.Module, val, length: int, tokens: int, device: torch.device) -> float:
    total, count = 0.0, 0
    for x, y in val_batches(val, length, tokens, max(1, 8192 // length)):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        loss = F.cross_entropy(
            logits.float().flatten(0, 1), y.to(device).flatten(), reduction="sum"
        )
        total, count = total + loss.item(), count + y.numel()
    return total / count


def losses(model: nn.Module, val, seq_len: int, tokens: int, mults, device) -> dict[str, float]:
    out = {}
    for m in mults:
        out[str(m)] = mean_loss(model, val, m * seq_len, tokens, device)
        print(f"    {m}x: {out[str(m)]:.4f}", flush=True)
    return out


def retrieval(model: nn.Module, val, seq_len: int, tokens: int, device) -> dict[str, object]:
    start = tokens + 8 * seq_len  # after the loss prefix
    grid = Grid(lengths=tuple(m * seq_len for m in MULTS))
    hay = val[start : start + grid.trials * max(grid.lengths)].tolist()

    def make_passkey(length: int, depth: float, key: int, trial: int) -> Sample:
        return passkey(encode, length, depth, key)

    def make_needle(length: int, depth: float, key: int, trial: int) -> Sample:
        at = trial * length  # a different stretch of text for every trial
        return needle(encode, hay[at : at + length], length, depth, key)

    out: dict[str, object] = {"depths": list(grid.depths), "trials": grid.trials}
    for task, make in (("passkey", make_passkey), ("needle", make_needle)):
        per_length: dict[str, object] = {}
        for length in grid.lengths:  # one length at a time, so progress is visible
            one = Grid((length,), grid.depths, grid.trials, grid.seed + length)
            per_length.update(run_grid(model, make, encode, one, device))
            acc = per_length[str(length)]["accuracy"]  # type: ignore[index]
            print(f"    {task} L={length}: {[round(a, 2) for a in acc]}", flush=True)
        out[task] = per_length
    return out


def git_state() -> dict[str, object]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    porcelain = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"sha": sha.strip(), "dirty": bool(porcelain.stdout.strip())}


def evaluate_run(run_dir: Path, tokens: int, with_retrieval: bool) -> dict[str, object]:
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = load_corpus(cfg.data, cfg.lm, cfg.model).val
    seq_len = cfg.lm.seq_len
    print(f"  {run_dir.parent.name}", flush=True)
    model = load_model(cfg, run_dir, device)
    out: dict[str, object] = {
        "tokens": tokens,
        "loss": losses(model, val, seq_len, tokens, MULTS, device),
    }
    yarn_cfg = with_yarn_t(cfg)
    if yarn_cfg is not None:  # before retrieval: the decision needs these numbers first
        print("    YaRN-t:", flush=True)
        scaled = load_model(yarn_cfg, run_dir, device)
        out["loss_yarn_t"] = losses(scaled, val, seq_len, tokens, MULTS[1:], device)
        del scaled
    if with_retrieval:
        print("    retrieval:", flush=True)
        out["retrieval"] = retrieval(model, val, seq_len, tokens, device)
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    return {**out, "git": git_state(), "device": gpu}


def main() -> None:
    # One compiled normalizer per layer x length x batch: lift dynamo's recompile limit so it
    # does not fall back to eager (slower, same numerics).
    torch._dynamo.config.cache_size_limit = 256
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("patterns", nargs="+")
    p.add_argument("--tokens", type=int, default=2_000_000, help="predictions per length")
    p.add_argument("--retrieval", action="store_true", help="also run passkey / needle")
    p.add_argument("--force", action="store_true", help="overwrite existing eval_offline.json")
    args = p.parse_args()
    runs = sorted(
        Path(r) for pat in args.patterns for exp in glob.glob(pat) for r in glob.glob(f"{exp}/*")
    )
    for run_dir in (r for r in runs if (r / "last.pt").exists()):
        target = run_dir / "eval_offline.json"
        if target.exists() and not args.force:
            print(f"  skip {run_dir} (exists)")
            continue
        result = evaluate_run(run_dir, args.tokens, args.retrieval)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
