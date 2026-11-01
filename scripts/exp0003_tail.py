"""Experiment 0003, part B2: evaluation-time tail interventions on trained checkpoints.

    uv run python scripts/exp0003_tail.py "runs/exp0002_screen_*" --out runs/tail/screen.json
    uv run python scripts/exp0003_tail.py "runs/exp0002_confirm_*" --out runs/tail/confirm.json

Each checkpoint of the preregistered set (``exp0003_decide.B2_CHECKPOINTS``, §3.3) gets its
intervention (cap, repair or window; ``geoattn_core.pe.TailIntervention``, from the training
length L on) and is evaluated with and without it:

* validation loss at 1x-16x (25M) or 1x-8x (100M; GPU memory), ``--tokens`` predictions each;
  the decisive multiple is the last one (``exp0003_decide.B2_MULT``);
* far mass (attention on keys more than L back, natural text) at 2x up to that multiple.

The intervention changes nothing at d <= L, so the 1x losses must agree; a mismatch beyond
``ONE_X_TOLERANCE`` stops the run (gate G2 on real checkpoints). Output for
``exp0003_decide.py b2``: per checkpoint ``{"mode", "mult", "loss": {"none", <mode>}}`` with
the full curves and far masses alongside. Resumable: checkpoints already in ``--out`` are kept.

Exploratory runs outside the preregistered set pass ``--mode`` and ``--mult``: every matched
run then gets that intervention, whatever its name, e.g. repair on the 100M log-bias arm::

    uv run python scripts/exp0003_tail.py "runs/exp0001x_confirm_softmax_logbias_*" \\
        --mode repair --mult 8 --out runs/tail/explore_logbias_100m.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import re
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from exp0003_decide import B2_CHECKPOINTS, B2_MULT
from probe_summability import VRAM_LIMIT_GB, _git, far_mass, mean_loss

from geoattn_core.pe.tail import TAIL_MODES
from geoattn_train.config import ExperimentConfig, load_config
from geoattn_train.lm_data import load_corpus
from geoattn_train.lm_trainer import build_lm
from geoattn_train.vram import cap_process_memory

NAME = re.compile(r"exp0002_(?P<stage>screen|confirm)_[a-z0-9_]+_s\d$")
MODE = {c: mode for mode, ckpts in B2_CHECKPOINTS.items() for c in ckpts}
ONE_X_TOLERANCE = 1e-4  # bf16 evaluation; the bias itself is identical bit for bit


def load(cfg: ExperimentConfig, run_dir: Path, device: torch.device) -> nn.Module:
    model = build_lm(cfg).to(device).eval()
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])  # the intervention's wrapper remaps the keys
    return model


def with_tail(cfg: ExperimentConfig, mode: str) -> ExperimentConfig:
    out = copy.deepcopy(cfg)
    out.attention.tail_mode = mode  # tail_from 0: the training length
    return out


def curves(model: nn.Module, val, n: int, mult: int, tokens: int, device) -> dict[str, dict]:
    mults = [m for m in (1, 2, 4, 8, 16) if m <= mult]
    loss = {str(m): mean_loss(model, val, m * n, tokens, device) for m in mults}
    far = {str(m): far_mass(model, val, m * n, n, device) for m in mults if m > 1}
    return {"loss": loss, "far_mass": far}


def evaluate(run_dir: Path, mode: str, mult: int, tokens: int) -> dict[str, object]:
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device("cuda")
    val = load_corpus(cfg.data, cfg.lm, cfg.model).val
    n = cfg.lm.seq_len
    result = {}
    for variant, variant_cfg in (("none", cfg), (mode, with_tail(cfg, mode))):
        model = load(variant_cfg, run_dir, device)
        result[variant] = curves(model, val, n, mult, tokens, device)
        print(f"    {variant}: {result[variant]}", flush=True)
        del model
        # Every model's score_mod captures new tensors, so FlexAttention recompiles for each
        # model and length; within one process these add up to the recompile limit (hit at
        # the 4th checkpoint, 2026-10-13). Nothing compiled is reused across models.
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    gap = abs(result[mode]["loss"]["1"] - result["none"]["loss"]["1"])
    if gap > ONE_X_TOLERANCE:
        raise RuntimeError(f"{run_dir}: the intervention changed the 1x loss by {gap:.2e}")
    top = str(mult)
    return {
        "mode": mode,
        "mult": mult,
        "train_len": n,
        "loss": {v: result[v]["loss"][top] for v in result},
        "curves": result,
    }


def select(name: str, mode: str | None, mult: int) -> tuple[str, int] | None:
    """The intervention and longest multiple for a run, or None to skip it."""
    if mode is not None:
        return mode, mult
    m = NAME.match(name)
    if m is None or name not in MODE:
        return None
    return MODE[name], B2_MULT[m["stage"]]


def main() -> None:
    torch._dynamo.config.cache_size_limit = 256
    cap_process_memory(VRAM_LIMIT_GB, torch.device("cuda"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("patterns", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=2_000_000, help="predictions per length (§3.5)")
    p.add_argument("--mode", choices=TAIL_MODES, help="exploratory: apply to every matched run")
    p.add_argument("--mult", type=int, default=8, help="longest multiple with --mode")
    args = p.parse_args()
    run_dirs = sorted(
        Path(r) for pat in args.patterns for e in glob.glob(pat) for r in glob.glob(f"{e}/*")
    )
    result: dict = {"tokens": args.tokens, "runs": {}}
    if args.out.exists():
        result = json.loads(args.out.read_text(encoding="utf-8"))
    for run_dir in run_dirs:
        name = run_dir.parent.name
        target = select(name, args.mode, args.mult)
        if target is None or not (run_dir / "last.pt").exists():
            continue
        if name in result["runs"]:
            print(f"  skip {name} (done)")
            continue
        mode, mult = target
        print(f"  {name}: {mode}", flush=True)
        try:
            entry = evaluate(run_dir, mode, mult, args.tokens)
        except RuntimeError as exc:  # e.g. a checkpoint that does not load (report 0014 §9)
            print(f"  {name}: skipped ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
            continue
        result["runs"][name] = entry
        result["git"] = _git()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
