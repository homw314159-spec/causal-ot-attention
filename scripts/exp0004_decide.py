"""Experiment 0004: the preregistered decision rules (§5) on finished evaluations.

    uv run python scripts/exp0004_decide.py a --runs runs
    uv run python scripts/exp0004_decide.py b --probe runs/summability/exp0004_b.json

Inputs:

* a: ``eval_offline.json`` (``scripts/exp0002_eval.py``) in ``exp0004_a_{b0,cable}_s<k>`` run
  directories; ``L_ext`` = mean of the 2x and 4x losses, B0 under YaRN-t
* b: ``scripts/probe_summability.py`` output for ``exp0004_b_{p1gt1,p1m25,p1m50,p4}_s<k>``
  (loss at 1x-32x, far mass at 4x and 16x)

Thresholds and directions combine as in experiment 0003 (``exp0003_decide.py``): the
per-seed differences must all have the predicted sign and their mean must pass the
threshold; the far-mass and 1x conditions are on the mean over seeds, with per-seed values
reported.

Rules:

* H1c (seeds 8-10): CABLE's ``L_ext`` below B0 + YaRN-t's by more than ``DELTA_A``, every
  seed; CABLE's mean 1x loss at most B0's + ``NONINFERIOR_1X`` (``exp0003_decide.h1``)
* H6 (seeds 3-5): P1-m50's far mass at 16x minus 4x at most ``FAR_FLAT``; P1-m50's 32x loss
  below P1-gt1's by more than ``DELTA_B``, every seed. The design rule is adopted only if
  also P1-m50's mean 1x loss is at most P1-gt1's + ``NONINFERIOR_1X``
* H7: far-mass growth ordered P1-gt1 > P1-m25 > P1-m50 on every seed
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
import statistics
from pathlib import Path

from exp0003_decide import DELTA_A, DELTA_B, FAR_FLAT, NONINFERIOR_1X, h1, paired
from exp0003_decide import _meta_problems as meta_problems

SEEDS = {"a": (8, 9, 10), "b": (3, 4, 5)}
A_ARMS = ("b0", "cable")
B_ARMS = ("p1gt1", "p1m25", "p1m50", "p4")
GROWTH_ORDER = ("p1gt1", "p1m25", "p1m50")  # H7: decreasing far-mass growth
NAME = re.compile(r"exp0004_(?P<part>a|b)_(?P<arm>[a-z0-9]+)_s(?P<seed>\d+)$")

# Shared with experiment 0003 so the results compare directly; fixed by preregistration §5
assert (DELTA_A, DELTA_B, FAR_FLAT, NONINFERIOR_1X) == (0.01, 0.01, 0.02, 0.01)


# ---------------------------------------------------------------- A: H1c


def h1c(table: dict[str, dict]) -> dict[str, object]:
    """``table``: arm -> seed -> eval_offline.json."""

    def ext(ev: dict, key: str) -> float:
        return (ev[key]["2"] + ev[key]["4"]) / 2

    b0, cable = table["b0"], table["cable"]
    return h1(
        {s: ext(e, "loss_yarn_t") for s, e in b0.items()},
        {s: ext(e, "loss") for s, e in cable.items()},
        {s: e["loss"]["1"] for s, e in b0.items()},
        {s: e["loss"]["1"] for s, e in cable.items()},
    )


def decide_a(r: dict) -> str:
    if r["holds"]:
        return "extend ADR-0014's confirmation to training length 4096 (new ADR)"
    return "reconsider the main line per ADR-0014 decision 5 (new ADR)"


# ---------------------------------------------------------------- B: H6, H7


def growth(far4: dict[str, dict], far16: dict[str, dict]) -> dict[str, dict]:
    return {arm: {s: far16[arm][s] - far4[arm][s] for s in far4[arm]} for arm in far4}


def h6(loss1: dict, loss32: dict, far4: dict, far16: dict) -> dict[str, object]:
    """Each argument maps arm -> seed -> value."""
    g = growth(far4, far16)["p1m50"]
    mean_growth = statistics.fmean(g.values())
    l32 = paired(loss32["p1gt1"], loss32["p1m50"], DELTA_B)
    margin_1x = statistics.fmean(loss1["p1m50"].values()) - statistics.fmean(
        loss1["p1gt1"].values()
    )
    holds = mean_growth <= FAR_FLAT and l32.holds
    return {
        "m50_far_growth": g,
        "m50_far_growth_mean": mean_growth,
        "m50_flat": mean_growth <= FAR_FLAT,
        "loss32": l32.summary(),
        "loss1_margin": margin_1x,
        "noninferior_1x": margin_1x <= NONINFERIOR_1X,
        "holds": holds,
        "adopt_margin_rule": holds and margin_1x <= NONINFERIOR_1X,
    }


def h7(far4: dict, far16: dict) -> dict[str, object]:
    g = growth(far4, far16)
    ordered = {
        s: all(g[a][s] > g[b][s] for a, b in itertools.pairwise(GROWTH_ORDER))
        for s in g[GROWTH_ORDER[0]]
    }
    return {
        "far_growth": {a: g[a] for a in GROWTH_ORDER},
        "ordered": ordered,
        "holds": all(ordered.values()),
    }


def decide_b(r6: dict) -> str:
    if r6["adopt_margin_rule"]:
        return "keep the margin rule with delta_0 = 0.5 (new ADR)"
    if r6["holds"]:
        return "H6 holds but 1x is worse: report; the margin rule is not adopted (new ADR)"
    return "drop the margin rule; keep only at-least-linear decay (new ADR)"


# ---------------------------------------------------------------- loading


def load_a(root: Path) -> tuple[dict[str, dict], list[str]]:
    """arm -> seed -> eval_offline.json, and the problems that block a decision."""
    table: dict[str, dict] = {}
    problems: list[str] = []
    for d in sorted(glob.glob(str(root / "exp0004_a_*" / "*"))):
        run_dir = Path(d)
        m = NAME.match(run_dir.parent.name)
        if m is None or not (run_dir / "eval_offline.json").exists():
            continue
        problems += meta_problems(run_dir, run_dir.parent.name)
        ev = json.loads((run_dir / "eval_offline.json").read_text(encoding="utf-8"))
        if int(m["seed"]) in table.setdefault(m["arm"], {}):
            problems.append(f"{run_dir.parent.name}: more than one run directory")
        table[m["arm"]][int(m["seed"])] = ev
    problems += [
        f"missing exp0004_a_{arm}_s{s}"
        for arm in A_ARMS
        for s in SEEDS["a"]
        if s not in table.get(arm, {})
    ]
    return table, problems


def load_b(probe: dict) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """value -> arm -> seed, for loss1, loss32, far4 and far16."""
    keys = {"loss1": ("loss", "1"), "loss32": ("loss", "32")}
    keys |= {"far4": ("far_mass", "4"), "far16": ("far_mass", "16")}
    out: dict[str, dict[str, dict]] = {name: {} for name in keys}
    problems = []
    for key, entry in probe["runs"].items():
        m = NAME.match(key.split("/")[0])
        if m is None or m["part"] != "b":
            continue
        if entry.get("git", {}).get("dirty", True):
            problems.append(f"{key}: probe from a dirty checkout")
        for name, (group, mult) in keys.items():
            value = entry[group].get(mult)
            if value is None:
                problems.append(f"{key}: no {name}")
            out[name].setdefault(m["arm"], {})[int(m["seed"])] = value
    problems += [
        f"missing exp0004_b_{a}_s{s}"
        for a in B_ARMS
        for s in SEEDS["b"]
        if s not in out["loss32"].get(a, {})
    ]
    return out, problems


def _run(args: argparse.Namespace) -> tuple[dict[str, object], list[str]]:
    if args.part == "a":
        table, problems = load_a(args.runs)
        if problems:
            return {}, problems
        r = h1c(table)
        return {"h1c": r, "decision": decide_a(r)}, []
    v, problems = load_b(json.loads(args.probe.read_text(encoding="utf-8")))
    if problems:
        return {}, problems
    r6 = h6(v["loss1"], v["loss32"], v["far4"], v["far16"])
    return {"h6": r6, "h7": h7(v["far4"], v["far16"]), "decision": decide_b(r6)}, []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("part", choices=("a", "b"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--probe", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result, problems = _run(args)
    if problems:
        print("cannot decide:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
