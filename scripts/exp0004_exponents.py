"""Experiment 0004, part B: learned exponents and the prior-only far-mass prediction (H8).

    uv run python scripts/exp0004_exponents.py "runs/exp0004_b_p1*" \
        --out runs/summability/exp0004_b_exponents.json

For every run, the per-head exponents of the log-distance bias are read from ``last.pt``
(``c = 1 + margin + softplus(raw)`` for ``gt1``), with their distribution and the share of
heads at the constraint boundary, ``c < 1 + margin + 0.01`` (preregistration section 8).
The prediction is the far-mass growth from 4x to 16x of the prior ``(1 + d)^(-c)`` alone,
averaged over heads, as in report 0016 section 8; H8 compares it with the measured growth.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from exp0004_margin_prior import L, far_mass

BOUNDARY = 0.01  # a head within this of its bound counts as at the boundary


def exponents(state: dict, margin: float) -> np.ndarray:
    """Per-head exponents of every ``gt1`` log-distance bias in a model state dict."""
    raw = [v for k, v in state.items() if k.endswith("pe.additive.raw")]
    if not raw:
        raise ValueError("no log-distance bias in this checkpoint")
    c = 1 + margin + torch.nn.functional.softplus(torch.cat(raw).double())
    return c.numpy()


def summary(c: np.ndarray, margin: float) -> dict[str, float]:
    q = np.quantile(c, [0.1, 0.5, 0.9])
    growth = [far_mass(x, 16 * L) - far_mass(x, 4 * L) for x in c]
    return {
        "heads": int(c.size),
        "min": float(c.min()),
        "p10": float(q[0]),
        "median": float(q[1]),
        "p90": float(q[2]),
        "max": float(c.max()),
        "share_at_boundary": float(np.mean(c < 1 + margin + BOUNDARY)),
        "predicted_far_growth": float(np.mean(growth)),
    }


def run_summary(run: Path) -> dict[str, float]:
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    margin = float(config["attention"].get("logbias_margin", 0.0))
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)["model"]
    return {"margin": margin, **summary(exponents(state, margin), margin)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("patterns", nargs="+", help="run name globs under runs/")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    dirs = sorted({Path(d) for pat in args.patterns for d in glob.glob(f"{pat}/*/")})
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    out = {
        "git": {"sha": sha.strip(), "dirty": bool(dirty.stdout.strip())},
        "runs": {f"{d.parent.name}/{d.name}": run_summary(d) for d in dirs},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, s in out["runs"].items():
        print(name, {k: round(v, 4) for k, v in s.items()})


if __name__ == "__main__":
    main()
