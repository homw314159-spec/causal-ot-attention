"""Experiment 0002: apply the preregistered selection and decision rules to finished runs.

    uv run python scripts/exp0002_decide.py screen                       # runs/exp0002_screen_*
    uv run python scripts/exp0002_decide.py confirm --priors p1 lc --o2-bias p3
    uv run python scripts/exp0002_decide.py confirm --priors p1 lc --o2-bias p1 \
        --lr-variant p1=p1_lr2.65e-4

``--priors`` lists the screening winners in screening order: the first is the primary prior.
``--o2-bias`` is O2's bias (screening selection); its arm without OT runs in confirmation.

Rules: docs/experiments.md §3.1 (screening selection), §3.2
(learning-rate variant) and §5 (decision), with its change log.

Numbers come from ``eval_offline.json`` in each run directory, written by the offline
evaluation (G5) from the final checkpoint::

    {"loss": {"1": .., "2": .., "4": .., "8": ..},           # validation loss by length multiple
     "loss_yarn_t": {"2": .., "4": .., "8": ..}}             # RoPE arms only

``L_ext`` is the mean of the 2x and 4x losses (§4).

* screen: the 2 best of P1-P6, G-A, FoX, CABLE, LC by mean ``L_ext`` over seeds {0, 1}; the
  bias for O2 is the best of P1-P5
* confirm (all paired by seed {0, 1, 2}; ``delta_e = max(2 sd_e, 0.01)`` and
  ``delta_1 = max(2 sd_1, 0.01)`` from B0's seeds):
  - the prior under test (the "best prior" of §5) is the screening #1, fixed before any
    confirmation run (change log 2026-10-06): selecting it on the confirmation runs that
    then test it would inflate its advantage. With a learning-rate variant it uses the
    variant with the lower mean 1x loss (§3.2). The screening #2 gets the same tests,
    reported as exploratory
  - GRAPE: if G-M's or G-AP's ``L_ext`` is lower than the primary prior's by more than
    ``delta_e``, every seed positive, the main line is GRAPE (outcome "grape") whatever the
    rows below say. "Beats GRAPE" is claimed only when the primary prior beats both
  - adopt: the primary prior's ``L_ext`` lower than B0's and B0 + YaRN-t's by more than
    ``delta_e``, every seed positive, and its mean 1x loss at most B0's + ``delta_1``
  - adopt with scaling: otherwise, the same test passes for the primary prior + YaRN-t; a
    prior without RoPE has no such row (YaRN-t is undefined for it)
  - otherwise: fall back to RoPE scaling and GRAPE
  - separately, unfreeze OT: O2's ``L_ext`` lower by more than ``delta_e``, every seed
    positive, than BOTH the primary prior's and that of O2's own bias without OT (the OT
    effect at a fixed bias, change log 2026-10-06): otherwise a gain of O2 could come from
    its bias alone
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

DELTA_FLOOR = 0.01
SCREEN_SEEDS = (0, 1)
CONFIRM_SEEDS = (0, 1, 2)
SCREEN_POOL = ("p1", "p2", "p3", "p4", "p5", "p6", "ga", "fox", "cable", "lc")
O2_POOL = ("p1", "p2", "p3", "p4", "p5")
NAME = re.compile(r"exp0002_(?P<stage>screen|confirm)_(?P<arm>[a-z0-9_.\-]+)_s(?P<seed>\d)$")


@dataclass
class Eval:
    loss1: float
    ext: float  # L_ext: mean of 2x and 4x
    ext_yarn: float | None  # the same under YaRN-t, RoPE arms only


@dataclass
class Run:
    stage: str
    arm: str
    seed: int
    eval: Eval
    step: int
    steps: int
    sha: str
    dirty: bool
    gpu: str
    path: str = ""


Table = dict[str, dict[int, Eval]]


@dataclass
class Comparison:
    """``reference - candidate`` on one metric: positive means the candidate is better."""

    mean_diff: float
    per_seed: dict[int, float]
    threshold: float

    @property
    def consistent(self) -> bool:
        return all(d > 0 for d in self.per_seed.values())

    @property
    def beats(self) -> bool:
        return self.mean_diff > self.threshold and self.consistent


@dataclass
class Selection:
    ranking: list[tuple[str, float]]  # (arm, mean L_ext), best first
    winners: list[str]
    o2_bias: str


@dataclass
class PriorTests:
    """The §5 tests of one prior; comparisons are ``reference - prior``."""

    arm: str
    vs_b0: Comparison
    vs_b0_yarn: Comparison
    loss1_gap: float  # mean 1x loss: prior - B0 (must be <= delta_1)
    scaled_vs_b0: Comparison | None  # None: the prior has no RoPE
    scaled_vs_b0_yarn: Comparison | None
    vs_gm: Comparison
    vs_gap: Comparison
    row: str  # "adopt" | "adopt-with-scaling" | "fallback", before the GRAPE rule

    @property
    def beats_grape(self) -> bool:
        return self.vs_gm.beats and self.vs_gap.beats


@dataclass
class Decision:
    outcome: str  # "grape" | "adopt" | "adopt-with-scaling" | "fallback"
    delta_e: float
    delta_1: float
    primary: PriorTests
    secondary: PriorTests  # exploratory: reported, not decided on
    grape_over_primary: dict[str, Comparison]  # prior - GRAPE arm: positive = GRAPE better
    o2_over_primary: Comparison  # primary prior - O2
    ot_effect: Comparison  # O2's bias alone - O2: the gain of OT at a fixed bias
    notes: list[str] = field(default_factory=list)

    @property
    def unfreeze_ot(self) -> bool:
        return self.o2_over_primary.beats and self.ot_effect.beats


# ---------------------------------------------------------------- loading


def _eval(d: dict) -> Eval:
    loss, yarn = d["loss"], d.get("loss_yarn_t")
    ext_yarn = (yarn["2"] + yarn["4"]) / 2 if yarn else None
    return Eval(loss1=loss["1"], ext=(loss["2"] + loss["4"]) / 2, ext_yarn=ext_yarn)


def load_run(run_dir: Path) -> Run | None:
    m = NAME.match(run_dir.parent.name)
    if m is None or not (run_dir / "eval_offline.json").exists():
        return None
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    last = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    ev = json.loads((run_dir / "eval_offline.json").read_text(encoding="utf-8"))
    return Run(
        stage=m["stage"],
        arm=m["arm"],
        seed=int(m["seed"]),
        eval=_eval(ev),
        step=last["step"],
        steps=meta["steps"],
        sha=meta["git"]["sha"],
        dirty=meta["git"]["dirty"],
        gpu=meta.get("gpu", "?"),
        path=str(run_dir),
    )


def load_runs(root: Path, stage: str) -> list[Run]:
    runs = []
    for d in sorted(glob.glob(str(root / f"exp0002_{stage}_*" / "*"))):
        run = load_run(Path(d)) if (Path(d) / "meta.json").exists() else None
        if run is not None:
            runs.append(run)
    return runs


def table(runs: list[Run]) -> Table:
    out: Table = {}
    for r in runs:
        out.setdefault(r.arm, {})[r.seed] = r.eval
    return out


def validate(runs: list[Run], arms: list[str], seeds: tuple[int, ...]) -> list[str]:
    seen = {(r.arm, r.seed) for r in runs}
    problems = [f"missing {a}_s{s}" for a in arms for s in seeds if (a, s) not in seen]
    if len(seen) != len(runs):
        problems.append("more than one run directory for some arm/seed")
    problems += [
        f"{r.arm}_s{r.seed}: incomplete ({r.step}/{r.steps})" for r in runs if r.step != r.steps
    ]
    problems += [f"{r.arm}_s{r.seed}: dirty" for r in runs if r.dirty]
    if len({r.sha for r in runs}) > 1:
        problems.append(f"runs span several commits: {sorted({r.sha[:7] for r in runs})}")
    if len({r.gpu for r in runs}) > 1:  # §8: one comparison, one hardware configuration
        problems.append(f"runs span several GPU models: {sorted({r.gpu for r in runs})}")
    return problems


# ---------------------------------------------------------------- rules


def _mean(t: Table, arm: str, seeds: tuple[int, ...], attr: str = "ext") -> float:
    return statistics.fmean(getattr(t[arm][s], attr) for s in seeds)


def _compare(ref: dict[int, float], cand: dict[int, float], threshold: float) -> Comparison:
    seeds = sorted(ref)
    per_seed = {s: ref[s] - cand[s] for s in seeds}
    return Comparison(statistics.fmean(per_seed.values()), per_seed, threshold)


def _ext(t: Table, arm: str, yarn: bool = False) -> dict[int, float]:
    out = {}
    for s, e in t[arm].items():
        value = e.ext_yarn if yarn else e.ext
        if value is None:
            raise ValueError(f"{arm}_s{s} has no YaRN-t evaluation")
        out[s] = value
    return out


def select_screening(t: Table) -> Selection:
    ranking = sorted(((a, _mean(t, a, SCREEN_SEEDS)) for a in SCREEN_POOL), key=lambda x: x[1])
    o2 = min(O2_POOL, key=lambda a: _mean(t, a, SCREEN_SEEDS))
    return Selection(ranking=ranking, winners=[a for a, _ in ranking[:2]], o2_bias=o2)


def resolve_priors(t: Table, priors: list[str], lr_variants: dict[str, str]) -> list[str]:
    """Each winner, or its learning-rate variant when that has the lower mean 1x loss."""
    out = []
    for p in priors:
        v = lr_variants.get(p)
        better = v is not None and _mean(t, v, CONFIRM_SEEDS, "loss1") < _mean(
            t, p, CONFIRM_SEEDS, "loss1"
        )
        out.append(v if better and v is not None else p)
    return out


def _thresholds(t: Table) -> tuple[float, float]:
    sd_e = statistics.stdev(t["b0"][s].ext for s in CONFIRM_SEEDS)
    sd_1 = statistics.stdev(t["b0"][s].loss1 for s in CONFIRM_SEEDS)
    return max(2 * sd_e, DELTA_FLOOR), max(2 * sd_1, DELTA_FLOOR)


def _row(adopt: bool, scaled: bool, inferior: bool) -> str:
    if adopt and not inferior:
        return "adopt"
    if scaled and not inferior:
        return "adopt-with-scaling"
    return "fallback"


def prior_tests(t: Table, arm: str, delta_e: float, delta_1: float) -> PriorTests:
    b0, b0_yarn, ext = _ext(t, "b0"), _ext(t, "b0", yarn=True), _ext(t, arm)
    vs_b0, vs_yarn = _compare(b0, ext, delta_e), _compare(b0_yarn, ext, delta_e)
    gap1 = _mean(t, arm, CONFIRM_SEEDS, "loss1") - _mean(t, "b0", CONFIRM_SEEDS, "loss1")
    scaled = scaled_yarn = None
    if all(e.ext_yarn is not None for e in t[arm].values()):
        scaled = _compare(b0, _ext(t, arm, yarn=True), delta_e)
        scaled_yarn = _compare(b0_yarn, _ext(t, arm, yarn=True), delta_e)
    row = _row(
        vs_b0.beats and vs_yarn.beats,
        bool(scaled and scaled_yarn and scaled.beats and scaled_yarn.beats),
        gap1 > delta_1,
    )
    vs_gm, vs_gap = _compare(_ext(t, "gm"), ext, delta_e), _compare(_ext(t, "gap"), ext, delta_e)
    return PriorTests(arm, vs_b0, vs_yarn, gap1, scaled, scaled_yarn, vs_gm, vs_gap, row)


def decide(t: Table, priors: list[str], o2_bias: str) -> Decision:
    """``priors``: screening #1 then #2 (after learning-rate resolution); ``o2_bias``: the
    arm of O2's bias without OT."""
    delta_e, delta_1 = _thresholds(t)
    primary, secondary = (prior_tests(t, a, delta_e, delta_1) for a in priors)
    ext = _ext(t, primary.arm)
    grape = {g: _compare(ext, _ext(t, g), delta_e) for g in ("gm", "gap")}
    notes = []
    if any(c.beats for c in grape.values()):
        outcome = "grape"
        notes.append(f"GRAPE beats {primary.arm} on L_ext: the main line is GRAPE")
    else:
        outcome = primary.row
    if outcome == "fallback" and primary.loss1_gap > delta_1:
        notes.append(f"1x loss above B0 + delta_1 ({primary.loss1_gap:+.4f} > {delta_1:.4f})")
    if outcome == "fallback" and primary.scaled_vs_b0 is None:
        notes.append(f"{primary.arm} has no RoPE: the scaling row does not apply")
    o2 = _ext(t, "o2")
    over_primary, effect = _compare(ext, o2, delta_e), _compare(_ext(t, o2_bias), o2, delta_e)
    return Decision(
        outcome, delta_e, delta_1, primary, secondary, grape, over_primary, effect, notes
    )


# ---------------------------------------------------------------- report and CLI


def _line(label: str, c: Comparison | None) -> str:
    if c is None:
        return f"{label}: not applicable"
    seeds = ", ".join(f"s{s} {v:+.4f}" for s, v in c.per_seed.items())
    verdict = "beats" if c.beats else "does not beat"
    return f"{label}: {c.mean_diff:+.4f} (threshold {c.threshold:.4f}; {seeds}) -> {verdict}"


def _prior_lines(label: str, pt: PriorTests) -> list[str]:
    return [
        f"{label} = {pt.arm}: 1x gap to B0 {pt.loss1_gap:+.4f}; rows -> {pt.row}; "
        f"beats GRAPE: {pt.beats_grape}",
        _line(f"  B0 - {pt.arm} (L_ext)", pt.vs_b0),
        _line(f"  B0 + YaRN-t - {pt.arm} (L_ext)", pt.vs_b0_yarn),
        _line(f"  B0 - {pt.arm} + YaRN-t", pt.scaled_vs_b0),
        _line(f"  B0 + YaRN-t - {pt.arm} + YaRN-t", pt.scaled_vs_b0_yarn),
        _line(f"  G-M - {pt.arm} (L_ext)", pt.vs_gm),
        _line(f"  G-AP - {pt.arm} (L_ext)", pt.vs_gap),
    ]


def report_confirm(t: Table, d: Decision) -> str:
    lines = ["| arm | 1x (mean) | L_ext (mean) | L_ext + YaRN-t (mean) |", "|---|---|---|---|"]
    for arm in sorted(t):
        e = list(t[arm].values())
        yarn = [x.ext_yarn for x in e if x.ext_yarn is not None]
        y = f"{statistics.fmean(yarn):.4f}" if len(yarn) == len(e) else "-"
        lines.append(
            f"| {arm} | {statistics.fmean(x.loss1 for x in e):.4f} | "
            f"{statistics.fmean(x.ext for x in e):.4f} | {y} |"
        )
    lines += ["", f"delta_e = {d.delta_e:.4f}, delta_1 = {d.delta_1:.4f}"]
    lines += _prior_lines("primary prior (screening #1)", d.primary)
    lines += _prior_lines("secondary prior (screening #2, exploratory)", d.secondary)
    lines += [
        _line(f"GRAPE rule: {d.primary.arm} - {g}", c) for g, c in d.grape_over_primary.items()
    ]
    lines += [
        f"outcome: {d.outcome}" + "".join(f"\n  note: {n}" for n in d.notes),
        _line(f"unfreeze OT (1/2): {d.primary.arm} - O2 (L_ext)", d.o2_over_primary),
        _line("unfreeze OT (2/2): O2's bias alone - O2 (L_ext)", d.ot_effect),
        f"unfreeze OT: {d.unfreeze_ot}",
    ]
    return "\n".join(lines)


def _parse_variants(items: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in items)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("stage", choices=["screen", "confirm"])
    p.add_argument("--root", type=Path, default=Path("runs"))
    p.add_argument("--priors", nargs=2, default=None, help="screening #1, then #2")
    p.add_argument("--o2-bias", default=None, help="O2's bias arm (screening selection)")
    p.add_argument("--lr-variant", nargs="*", default=[], help="winner=variant_arm")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    runs = load_runs(args.root, args.stage)
    t = table(runs)
    if args.stage == "screen":
        arms = ["b0", "gm", "gap", "o1", *SCREEN_POOL]
        problems = validate(runs, arms, SCREEN_SEEDS)
        if any(m.startswith("missing") for m in problems):
            raise SystemExit("\n".join(problems) + "\nnot every arm/seed is present; no selection")
        result: Selection | Decision = select_screening(t)
        text = "\n".join(f"{a}: {v:.4f}" for a, v in result.ranking)
        text += f"\nwinners: {result.winners}; O2 bias: {result.o2_bias}"
    else:
        if args.priors is None or args.o2_bias is None:
            raise SystemExit("confirm needs --priors and --o2-bias (screening selection)")
        variants = _parse_variants(args.lr_variant)
        arms = ["b0", "logn", "gm", "gap", "o2", args.o2_bias, *args.priors, *variants.values()]
        arms = list(dict.fromkeys(arms))  # O2's bias may also be a winner
        problems = validate(runs, arms, CONFIRM_SEEDS)
        if any(m.startswith("missing") for m in problems):
            raise SystemExit("\n".join(problems) + "\nnot every arm/seed is present; no decision")
        result = decide(t, resolve_priors(t, args.priors, variants), args.o2_bias)
        text = report_confirm(t, result)
    print(text + "\nvalidity: " + ("all checks passed" if not problems else "; ".join(problems)))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runs": [asdict(r) for r in runs], "problems": problems}
        payload["result"] = asdict(result)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
