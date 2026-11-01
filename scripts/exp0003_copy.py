"""Experiment 0003, gate G3: the distant-copy gain of each finished run.

    uv run python scripts/exp0003_copy.py "runs/exp0002_screen_*" --out runs/copy/screen.json

A fragment A of ``m`` validation tokens closes a sequence of ``N`` tokens of natural
validation text. Three variants differ only in the ``m`` tokens that start ``delta``
tokens before the final A (``delta`` = distance between the two starts):

* ``copy``: A itself;
* ``control``: another fragment B of the stream (the preregistered control);
* ``shuffled``: A's tokens in a random order (the counter-example: same tokens, no order
  to copy).

``gain`` = mean over trials of the control's loss minus the copy's loss on the final A,
without its first token (its m - 1 predictions after A has started); ``gain_shuffled`` is
the same with the shuffled variant in place of the copy. Every model sees the same trials
(fixed generator), so gains are paired across runs. ``delta`` runs over L/2, L, 2L, 4L
(L = the run's training length) where ``delta + m <= N``; ``N`` over ``--mults`` x L.
Inference only, bf16 autocast, the run's final checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geoattn_train.config import ExperimentConfig, load_config
from geoattn_train.lm_data import load_corpus
from geoattn_train.lm_trainer import build_lm

FRAGMENT = 128
DELTA_FRACTIONS = (0.5, 1, 2, 4)  # x L
MULTS = (1, 5)  # x L: at 1x only delta = L/2 fits
VARIANTS = ("copy", "control", "shuffled")


@dataclass(frozen=True)
class Trial:
    filler: np.ndarray  # N tokens of natural text
    a: np.ndarray  # the fragment that closes the sequence
    b: np.ndarray  # the control's fragment
    perm: np.ndarray  # the shuffled variant's order of A


def make_trials(stream: np.ndarray, start: int, n: int, m: int, trials: int, seed: int):
    """Trials from ``stream[start:]``: disjoint stretches for filler, A and B per trial."""
    rng = np.random.default_rng(seed)
    per = n + 2 * m
    if start + trials * per > len(stream):
        raise ValueError(f"the stream is too short for {trials} trials of {n} tokens")
    out = []
    for i in range(trials):
        at = start + i * per
        out.append(
            Trial(
                filler=stream[at : at + n].astype(np.int64),
                a=stream[at + n : at + n + m].astype(np.int64),
                b=stream[at + n + m : at + per].astype(np.int64),
                perm=rng.permutation(m),
            )
        )
    return out


def build(trial: Trial, delta: int, variant: str) -> np.ndarray:
    """The ``N`` tokens of one variant: filler, the first fragment at ``N - m - delta``, A
    at the end."""
    seq = trial.filler.copy()
    n, m = len(seq), len(trial.a)
    if delta < m or delta + m > n:
        raise ValueError(f"need m <= delta <= N - m, got {delta=} with {m=}, N={n}")
    first = {"copy": trial.a, "control": trial.b, "shuffled": trial.a[trial.perm]}[variant]
    seq[n - m - delta : n - delta] = first
    seq[n - m :] = trial.a
    return seq


def fragment_loss(logits: Tensor, seq: Tensor, m: int) -> Tensor:
    """Mean loss per sequence on the final fragment's tokens 2..m: logits at ``N - m`` to
    ``N - 2`` predict tokens ``N - m + 1`` to ``N - 1``."""
    pred = logits[:, -m:-1].float()
    target = seq[:, -m + 1 :]
    return F.cross_entropy(pred.transpose(1, 2), target, reduction="none").mean(-1)


@torch.no_grad()
def variant_losses(model: nn.Module, seqs: np.ndarray, m: int, device) -> np.ndarray:
    out = []
    batch = max(1, 8192 // seqs.shape[1])
    for i in range(0, len(seqs), batch):
        x = torch.from_numpy(seqs[i : i + batch]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        out.append(fragment_loss(logits, x, m).cpu().numpy())
    return np.concatenate(out)


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def copy_gains(model: nn.Module, trials: list[Trial], delta: int, device) -> dict[str, float]:
    m = len(trials[0].a)
    loss = {
        v: variant_losses(model, np.stack([build(t, delta, v) for t in trials]), m, device)
        for v in VARIANTS
    }
    gain, gain_se = _mean_se(loss["control"] - loss["copy"])
    shuffled, shuffled_se = _mean_se(loss["control"] - loss["shuffled"])
    return {
        **{v: float(loss[v].mean()) for v in VARIANTS},
        "gain": gain,
        "gain_se": gain_se,
        "gain_shuffled": shuffled,
        "gain_shuffled_se": shuffled_se,
    }


def deltas(train_len: int, n: int, m: int) -> list[int]:
    return [d for d in (int(f * train_len) for f in DELTA_FRACTIONS) if m <= d <= n - m]


def load_model(cfg: ExperimentConfig, run_dir: Path, device: torch.device) -> nn.Module:
    model = build_lm(cfg).to(device).eval()
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return model


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = load_corpus(cfg.data, cfg.lm, cfg.model).val
    train_len = cfg.lm.seq_len
    model = load_model(cfg, run_dir, device)
    start = args.skip + 8 * train_len  # after the loss evaluation's prefix
    out: dict[str, object] = {}
    for mult in args.mults:
        n = mult * train_len
        trials = make_trials(np.asarray(val), start, n, FRAGMENT, args.trials, args.seed)
        per_delta = {}
        for delta in deltas(train_len, n, FRAGMENT):
            per_delta[str(delta)] = copy_gains(model, trials, delta, device)
            r = per_delta[str(delta)]
            print(
                f"    N={n} delta={delta}: gain {r['gain']:.3f} ± {r['gain_se']:.3f}, "
                f"shuffled {r['gain_shuffled']:.3f}",
                flush=True,
            )
        out[str(n)] = per_delta
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    return {
        "train_len": train_len,
        "m": FRAGMENT,
        "trials": args.trials,
        "lengths": out,
        "device": gpu,
    }


def _git() -> dict[str, object]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    porcelain = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"sha": sha.strip(), "dirty": bool(porcelain.stdout.strip())}


def main() -> None:
    torch._dynamo.config.cache_size_limit = 256
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("patterns", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mults", type=int, nargs="+", default=list(MULTS))
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip", type=int, default=2_000_000, help="loss-evaluation tokens to skip")
    args = p.parse_args()
    runs = sorted(
        Path(r) for pat in args.patterns for exp in glob.glob(pat) for r in glob.glob(f"{exp}/*")
    )
    results: dict[str, object] = {}
    if args.out.exists():
        results = json.loads(args.out.read_text(encoding="utf-8"))["runs"]
    for run_dir in (r for r in runs if (r / "last.pt").exists()):
        key = run_dir.parent.name
        if key in results:
            print(f"  skip {key} (done)")
            continue
        print(f"  {key}", flush=True)
        results[key] = evaluate_run(run_dir, args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "git": _git(),
            "args": {k: str(v) for k, v in vars(args).items()},
            "runs": results,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
