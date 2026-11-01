"""Experiment 0003: the preregistered decision rules (§5) on finished evaluations.

    uv run python scripts/exp0003_decide.py a  --runs runs
    uv run python scripts/exp0003_decide.py b1 --probe runs/summability/exp0003_b1.json
    uv run python scripts/exp0003_decide.py b2 --tail runs/tail/screen.json runs/tail/confirm.json

Inputs:

* a: ``eval_offline.json`` (``scripts/exp0002_eval.py``) in ``exp0003_{a1,a2}_{b0,cable}_s<k>``
  run directories; ``L_ext`` = mean of the 2x and 4x losses, B0 under YaRN-t
* b1: ``scripts/probe_summability.py`` output for ``exp0003_b1_{p1free,p1gt1,p1lt1,p4}_s<k>``
  (loss at 1x-32x, far mass at 4x and 16x)
* b2: ``scripts/exp0003_tail.py`` output: per checkpoint the intervention, the length
  multiple (16 at 25M, 8 at 100M) and the loss without and with it

How "threshold" and "direction" combine (§5: "all thresholds and directions paired by seed,
by checkpoint for H3"), as in experiment 0002: the per-seed (per-checkpoint) differences must
all have the predicted sign and their mean must pass the threshold. The far-mass conditions
of H2 are on the mean over seeds; per-seed values are reported.

Rules:

* H1a (A1, seeds 3-7) / H1b (A2, seeds 3-5): CABLE's ``L_ext`` below B0 + YaRN-t's by more
  than ``DELTA_A``, every seed; CABLE's mean 1x loss at most B0's + ``NONINFERIOR_1X``
* H2 (seeds 0-2): P1-gt1's 32x loss below P1-lt1's by more than ``DELTA_B``, every seed; P1-gt1's
  far mass at 16x minus 4x at most ``FAR_FLAT``, P1-lt1's above ``FAR_GROW``
* H3a cap: loss up by more than ``DELTA_C``; H3b repair: down by more than ``DELTA_C``; H3c
  window: down by more than ``WINDOW_GAIN``; each over every checkpoint of §3.3
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

DELTA_A, DELTA_B, DELTA_C = 0.01, 0.01, 0.01
NONINFERIOR_1X = 0.01
FAR_FLAT, FAR_GROW = 0.02, 0.1
WINDOW_GAIN = 0.1
SEEDS = {"a1": (3, 4, 5, 6, 7), "a2": (3, 4, 5), "b1": (0, 1, 2)}
B1_ARMS = ("p1free", "p1gt1", "p1lt1", "p4")
NAME = re.compile(r"exp0003_(?P<part>a1|a2|b1)_(?P<arm>[a-z0-9]+)_s(?P<seed>\d)$")

# §3.3: which checkpoints each intervention is applied to (LC at 100M: seeds 1 and 2, the
# checkpoint of seed 0 is corrupt; report 0014 §9)
B2_CHECKPOINTS = {
    "cap": [f"exp0002_screen_{a}_s{s}" for a in ("p4", "ga", "fox", "cable") for s in (0, 1)]
    + [f"exp0002_confirm_{a}_s{s}" for a in ("p4", "cable", "cable_lr") for s in (0, 1, 2)],
    "repair": [f"exp0002_screen_{a}_s{s}" for a in ("p1", "p2", "p6") for s in (0, 1)]
    + [f"exp0002_confirm_lc_s{s}" for s in (1, 2)],
    "window": [f"exp0002_screen_b0_s{s}" for s in (0, 1)]
    + [f"exp0002_confirm_b0_s{s}" for s in (0, 1, 2)],
}
B2_MULT = {"screen": 16, "confirm": 8}


@dataclass
class Paired:
    """Per-seed (per-checkpoint) differences, positive in the predicted direction."""

    diffs: dict[str, float]
    threshold: float

    @property
    def mean(self) -> float:
        return statistics.fmean(self.diffs.values())

    @property
    def every_one(self) -> bool:
        return all(d > 0 for d in self.diffs.values())

    @property
    def holds(self) -> bool:
        return self.every_one and self.mean > self.threshold

    def summary(self) -> dict[str, object]:
        return {**asdict(self), "mean": self.mean, "every_one": self.every_one, "holds": self.holds}


def paired(worse: dict, better: dict, threshold: float) -> Paired:
    """``worse - better`` key by key; both must have the same keys."""
    if set(worse) != set(better):
        raise ValueError(f"unpaired keys: {sorted(set(worse) ^ set(better))}")
    return Paired({str(k): worse[k] - better[k] for k in sorted(worse)}, threshold)


# ---------------------------------------------------------------- A: H1


def h1(b0_ext_yarn: dict, cable_ext: dict, b0_1x: dict, cable_1x: dict) -> dict[str, object]:
    ext = paired(b0_ext_yarn, cable_ext, DELTA_A)
    margin = statistics.fmean(cable_1x.values()) - statistics.fmean(b0_1x.values())
    noninferior = margin <= NONINFERIOR_1X
    return {
        "ext": ext.summary(),
        "loss1_margin": margin,
        "noninferior_1x": noninferior,
        "holds": ext.holds and noninferior,
    }


def decide_a(h1a: dict, h1b: dict) -> str:
    if not h1a["holds"]:
        return "revise: main line RoPE + YaRN-t, explicit priors as a control (new ADR)"
    if h1b["holds"]:
        return "confirm: holds at training lengths 1024 and 2048 (new ADR)"
    return "confirm at 1024 only: H1b fails, analyse in the report (new ADR)"


# ---------------------------------------------------------------- B1: H2


def h2(loss32: dict[str, dict], far4: dict[str, dict], far16: dict[str, dict]) -> dict:
    """Each argument maps arm -> seed -> value."""
    l32 = paired(loss32["p1lt1"], loss32["p1gt1"], DELTA_B)
    growth = {
        arm: {s: far16[arm][s] - far4[arm][s] for s in far4[arm]} for arm in ("p1gt1", "p1lt1")
    }
    mean_growth = {arm: statistics.fmean(g.values()) for arm, g in growth.items()}
    flat = mean_growth["p1gt1"] <= FAR_FLAT
    grows = mean_growth["p1lt1"] > FAR_GROW
    return {
        "loss32": l32.summary(),
        "far_growth": growth,
        "far_growth_mean": mean_growth,
        "gt1_flat": flat,
        "lt1_grows": grows,
        "holds": l32.holds and flat and grows,
    }


# ---------------------------------------------------------------- B2: H3


def h3(tail: dict[str, dict]) -> dict[str, dict]:
    """``tail``: checkpoint -> {"mode", "mult", "loss": {"none": .., mode: ..}}."""
    out = {}
    for mode, threshold in (("cap", DELTA_C), ("repair", DELTA_C), ("window", WINDOW_GAIN)):
        before = {c: tail[c]["loss"]["none"] for c in B2_CHECKPOINTS[mode]}
        after = {c: tail[c]["loss"][mode] for c in B2_CHECKPOINTS[mode]}
        cmp = (
            paired(after, before, threshold) if mode == "cap" else paired(before, after, threshold)
        )
        out[mode] = cmp.summary()
    return out


def validate_b2(tail: dict[str, dict]) -> list[str]:
    problems = []
    for mode, ckpts in B2_CHECKPOINTS.items():
        for c in ckpts:
            entry = tail.get(c)
            stage = c.split("_")[1]
            if entry is None:
                problems.append(f"missing {c}")
            elif entry["mode"] != mode or entry["mult"] != B2_MULT[stage]:
                problems.append(f"{c}: {entry['mode']} at {entry['mult']}x, expected {mode}")
            elif mode not in entry["loss"] or "none" not in entry["loss"]:
                problems.append(f"{c}: no loss for none and {mode}")
    return problems


# ---------------------------------------------------------------- loading


def _meta_problems(run_dir: Path, name: str) -> list[str]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    last = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    out = [f"{name}: dirty"] if meta["git"]["dirty"] else []
    if last["step"] != meta["steps"]:
        out.append(f"{name}: incomplete ({last['step']}/{meta['steps']})")
    return out


def load_a(root: Path, part: str) -> tuple[dict[str, dict], list[str]]:
    """arm -> seed -> eval_offline.json, and the problems that block a decision."""
    table: dict[str, dict] = {}
    problems: list[str] = []
    for d in sorted(glob.glob(str(root / f"exp0003_{part}_*" / "*"))):
        run_dir = Path(d)
        m = NAME.match(run_dir.parent.name)
        if m is None or not (run_dir / "eval_offline.json").exists():
            continue
        problems += _meta_problems(run_dir, run_dir.parent.name)
        ev = json.loads((run_dir / "eval_offline.json").read_text(encoding="utf-8"))
        if int(m["seed"]) in table.setdefault(m["arm"], {}):
            problems.append(f"{run_dir.parent.name}: more than one run directory")
        table[m["arm"]][int(m["seed"])] = ev
    for arm in ("b0", "cable"):
        problems += [
            f"missing exp0003_{part}_{arm}_s{s}" for s in SEEDS[part] if s not in table.get(arm, {})
        ]
    return table, problems


def h1_from_evals(table: dict[str, dict]) -> dict[str, object]:
    def ext(ev: dict, key: str) -> float:
        return (ev[key]["2"] + ev[key]["4"]) / 2

    b0, cable = table["b0"], table["cable"]
    return h1(
        {s: ext(e, "loss_yarn_t") for s, e in b0.items()},
        {s: ext(e, "loss") for s, e in cable.items()},
        {s: e["loss"]["1"] for s, e in b0.items()},
        {s: e["loss"]["1"] for s, e in cable.items()},
    )


def load_b1(probe: dict) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """value -> arm -> seed, for loss32, far4 and far16."""
    out: dict[str, dict[str, dict]] = {"loss32": {}, "far4": {}, "far16": {}}
    problems = []
    for key, entry in probe["runs"].items():
        m = NAME.match(key.split("/")[0])
        if m is None or m["part"] != "b1":
            continue
        arm, seed = m["arm"], int(m["seed"])
        if entry.get("git", {}).get("dirty", True):
            problems.append(f"{key}: probe from a dirty checkout")
        for name, value in (
            ("loss32", entry["loss"].get("32")),
            ("far4", entry["far_mass"].get("4")),
            ("far16", entry["far_mass"].get("16")),
        ):
            if value is None:
                problems.append(f"{key}: no {name}")
            out[name].setdefault(arm, {})[seed] = value
    problems += [
        f"missing exp0003_b1_{a}_s{s}"
        for a in B1_ARMS
        for s in SEEDS["b1"]
        if s not in out["loss32"].get(a, {})
    ]
    return out, problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("part", choices=("a", "b1", "b2"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--probe", type=Path)
    p.add_argument("--tail", type=Path, nargs="+")
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


def _run(args: argparse.Namespace) -> tuple[dict[str, object], list[str]]:
    if args.part == "a":
        a1, p1 = load_a(args.runs, "a1")
        a2, p2 = load_a(args.runs, "a2")
        if p1 or p2:
            return {}, p1 + p2
        r1, r2 = h1_from_evals(a1), h1_from_evals(a2)
        return {"h1a": r1, "h1b": r2, "decision": decide_a(r1, r2)}, []
    if args.part == "b1":
        values, problems = load_b1(json.loads(args.probe.read_text(encoding="utf-8")))
        return ({} if problems else {"h2": h2(**values)}), problems
    tail: dict[str, dict] = {}
    problems = []
    for path in args.tail:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["git"]["dirty"]:
            problems.append(f"{path}: from a dirty checkout")
        tail.update(data["runs"])
    problems += validate_b2(tail)
    return ({} if problems else {"h3": h3(tail)}), problems


if __name__ == "__main__":
    main()
