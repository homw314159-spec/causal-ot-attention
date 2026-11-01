"""Experiment 0001 confirmation: apply the preregistered decision rule to the finished runs.

    uv run python scripts/exp0001_decide.py                      # runs/exp0001_confirm_*
    uv run python scripts/exp0001_decide.py --out runs/decision/exp0001_confirm.json

Rule (docs/experiments.md §5 with the change log of 2026-09-28,
2026-09-30 and 2026-10-01). The baseline is softmax (eps 1.0, the only fixed-temperature
arm at this stage); "best rho" is the rho arm with the lower mean val@1024 (primary metric).

* tier 1: mean(baseline) - mean(best rho) on val@1024 > delta = max(2 sigma, 0.01), with
  sigma the baseline's seed standard deviation of val@1024, and the difference positive
  for every seed (paired by seed)                                     -> path C
* tier 2: tier 1 not met, the val@1024 gap within delta, and on val@2048 the best rho is
  lower than BOTH softmax and log-n softmax by more than delta_2 = max(2 sigma_2, 0.01)
  (sigma_2: the baseline's seed standard deviation of val@2048), every seed positive
                                                                       -> path C (long context)
* otherwise (including rho worse than the baseline by more than delta) -> path A

Numbers are the final evaluation records in metrics.jsonl (ADR-0004). Validity checks
(complete, dirty: false, one commit, one GPU model, G4 residual at the run's k <= 1e-2)
are printed and stored; a failed check is reported, not silently skipped.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

PREFIX = "exp0001_confirm_"
ARMS = ("softmax", "logn", "rho0.5", "rho0.25")
RHO_ARMS = ("rho0.5", "rho0.25")
SEEDS = (0, 1, 2)
DELTA_FLOOR = 0.01
G4_LIMIT = 1e-2
NAME = re.compile(rf"{PREFIX}(?P<arm>softmax|logn|rho0\.5|rho0\.25)_s(?P<seed>\d)$")


@dataclass
class Run:
    arm: str
    seed: int
    val1024: float
    val2048: float
    step: int
    steps: int
    sha: str
    dirty: bool
    gpu: str
    residual: float | None = None  # G4: max |P_k - P_(k-1)| at the run's own rho and k
    path: str = ""


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
class Decision:
    outcome: str  # "C" | "C-long-context" | "A"
    best_rho: str
    sigma: float
    delta: float
    sigma2: float
    delta2: float
    tier1: Comparison
    tier2_softmax: Comparison
    tier2_logn: Comparison
    notes: list[str] = field(default_factory=list)


def _residual(run_dir: Path, rho: float, k: int) -> float | None:
    f = run_dir / "fixed_point.json"
    if not f.exists():
        return None
    series = json.loads(f.read_text(encoding="utf-8"))["residuals"].get(str(rho))
    return series["max"][k - 2] if series and len(series["max"]) >= k - 1 else None


def load_run(run_dir: Path) -> Run | None:
    import yaml

    m = NAME.match(run_dir.parent.name)
    if m is None:
        return None
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    last = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    norm = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["attention"][
        "normalizer"
    ]
    residual = None
    if norm["name"] == "unbalanced":
        residual = _residual(run_dir, norm["rho"], norm["n_iters"])
    return Run(
        arm=m["arm"],
        seed=int(m["seed"]),
        val1024=last["val_loss@1024"],
        val2048=last["val_loss@2048"],
        step=last["step"],
        steps=meta["steps"],
        sha=meta["git"]["sha"],
        dirty=meta["git"]["dirty"],
        gpu=meta.get("gpu", "?"),
        residual=residual,
        path=str(run_dir),
    )


def load_runs(root: Path) -> list[Run]:
    runs = []
    for d in sorted(glob.glob(str(root / f"{PREFIX}*" / "*"))):
        if (Path(d) / "meta.json").exists() and (Path(d) / "metrics.jsonl").exists():
            run = load_run(Path(d))
            if run is not None:
                runs.append(run)
    return runs


def validate(runs: list[Run]) -> list[str]:
    problems = []
    seen = {(r.arm, r.seed) for r in runs}
    problems += [f"missing {a}_s{s}" for a in ARMS for s in SEEDS if (a, s) not in seen]
    if len(seen) != len(runs):
        problems.append("more than one run directory for some arm/seed")
    problems += [
        f"{r.arm}_s{r.seed}: incomplete ({r.step}/{r.steps})" for r in runs if r.step != r.steps
    ]
    problems += [f"{r.arm}_s{r.seed}: dirty" for r in runs if r.dirty]
    if len({r.sha for r in runs}) > 1:
        problems.append(f"runs span several commits: {sorted({r.sha[:7] for r in runs})}")
    if len({r.gpu for r in runs}) > 1:
        problems.append(f"runs span several GPU models: {sorted({r.gpu for r in runs})}")
    for r in runs:
        if r.arm in RHO_ARMS and r.residual is None:
            problems.append(f"{r.arm}_s{r.seed}: no G4 residual (fixed_point.json)")
        elif r.residual is not None and r.residual > G4_LIMIT:
            problems.append(f"{r.arm}_s{r.seed}: not converged (residual {r.residual:.1e})")
    return problems


def _table(runs: list[Run], metric: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for r in runs:
        out.setdefault(r.arm, {})[r.seed] = getattr(r, metric)
    return out


def _compare(ref: dict[int, float], cand: dict[int, float], threshold: float) -> Comparison:
    per_seed = {s: ref[s] - cand[s] for s in SEEDS}
    mean = statistics.fmean(ref[s] for s in SEEDS) - statistics.fmean(cand[s] for s in SEEDS)
    return Comparison(mean_diff=mean, per_seed=per_seed, threshold=threshold)


def decide(runs: list[Run]) -> Decision:
    v1, v2 = _table(runs, "val1024"), _table(runs, "val2048")
    sigma = statistics.stdev(v1["softmax"][s] for s in SEEDS)
    sigma2 = statistics.stdev(v2["softmax"][s] for s in SEEDS)
    delta, delta2 = max(2 * sigma, DELTA_FLOOR), max(2 * sigma2, DELTA_FLOOR)
    best = min(RHO_ARMS, key=lambda a: statistics.fmean(v1[a][s] for s in SEEDS))
    tier1 = _compare(v1["softmax"], v1[best], delta)
    t2_softmax = _compare(v2["softmax"], v2[best], delta2)
    t2_logn = _compare(v2["logn"], v2[best], delta2)
    notes = []
    if tier1.beats:
        outcome = "C"
    elif abs(tier1.mean_diff) <= delta and t2_softmax.beats and t2_logn.beats:
        outcome = "C-long-context"
    else:
        outcome = "A"
        if tier1.mean_diff > delta and not tier1.consistent:
            notes.append("val@1024 gap exceeds delta but the seeds disagree: neither tier applies")
        if tier1.mean_diff < -delta:
            notes.append("best rho is worse than the baseline on val@1024 by more than delta")
    return Decision(outcome, best, sigma, delta, sigma2, delta2, tier1, t2_softmax, t2_logn, notes)


def report(runs: list[Run], decision: Decision, problems: list[str]) -> str:
    v1, v2 = _table(runs, "val1024"), _table(runs, "val2048")
    lines = ["| arm | val@1024 (s0, s1, s2) | mean ± sd | val@2048 (s0, s1, s2) | mean ± sd |"]
    lines.append("|---|---|---|---|---|")
    for arm in ARMS:
        if arm not in v1:
            continue
        a, b = [v1[arm].get(s) for s in SEEDS], [v2[arm].get(s) for s in SEEDS]
        lines.append(f"| {arm} | {_fmt(a)} | {_ms(a)} | {_fmt(b)} | {_ms(b)} |")
    d = decision
    lines += [
        "",
        f"baseline sigma (val@1024) = {d.sigma:.4f}, delta = {d.delta:.4f}; "
        f"sigma_2 (val@2048) = {d.sigma2:.4f}, delta_2 = {d.delta2:.4f}; best rho = {d.best_rho}",
        _line("tier 1: softmax - best rho, val@1024", d.tier1),
        _line("tier 2: softmax - best rho, val@2048", d.tier2_softmax),
        _line("tier 2: log-n - best rho, val@2048", d.tier2_logn),
        f"outcome: path {d.outcome}" + "".join(f"\n  note: {n}" for n in d.notes),
        "validity: " + ("all checks passed" if not problems else "; ".join(problems)),
    ]
    return "\n".join(lines)


def _fmt(xs: list[float | None]) -> str:
    return ", ".join("-" if x is None else f"{x:.4f}" for x in xs)


def _ms(xs: list[float | None]) -> str:
    vals = [x for x in xs if x is not None]
    if len(vals) < 2:
        return "-"
    return f"{statistics.fmean(vals):.4f} ± {statistics.stdev(vals):.4f} (n={len(vals)})"


def _line(label: str, c: Comparison) -> str:
    seeds = ", ".join(f"s{s} {v:+.4f}" for s, v in c.per_seed.items())
    verdict = "beats" if c.beats else "does not beat"
    return f"{label}: {c.mean_diff:+.4f} (threshold {c.threshold:.4f}; {seeds}) -> {verdict}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    runs = load_runs(args.root)
    problems = validate(runs)
    if any(m.startswith("missing") for m in problems):
        print("\n".join(problems))
        raise SystemExit("not every arm/seed is present; no decision")
    decision = decide(runs)
    print(report(runs, decision, problems))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runs": [asdict(r) for r in runs], "problems": problems}
        payload["decision"] = asdict(decision)
        payload["verdicts"] = {
            name: {"beats": c.beats, "consistent": c.consistent}
            for name, c in (
                ("tier1", decision.tier1),
                ("tier2_softmax", decision.tier2_softmax),
                ("tier2_logn", decision.tier2_logn),
            )
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
