"""Tables of the experiment 0004 report from the evaluation files, as markdown.

    uv run python scripts/summarize_exp0004.py a   # part A: L_ext, 1x-8x, retrieval, copy
    uv run python scripts/summarize_exp0004.py b   # part B: 1x-32x, far mass, exponents, H8

Inputs (under ``--runs``, default ``runs``): ``exp0004_a_*/*/eval_offline.json``,
``summability/exp0004_{a_rtx4090,a_spark,b,b_exponents}.json`` and
``copy/exp0004_{a_rtx4090,a_spark,b}.json``. Means are over seeds with the sample standard
deviation; the decisions themselves come from ``exp0004_decide.py``, not from these tables.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

RUN = re.compile(r"exp0004_(?P<part>[ab])_(?P<arm>[a-z0-9]+)_s(?P<seed>\d+)")
A_MULTS, B_MULTS = ("1", "2", "4", "8"), ("1", "2", "4", "8", "16", "32")
B_ARMS = ("p4", "p1gt1", "p1m25", "p1m50")


def ms(xs: list[float], digits: int = 3) -> str:
    return f"{statistics.fmean(xs):.{digits}f} ± {statistics.stdev(xs):.{digits}f}"


def row(*cells: object) -> None:
    print("| " + " | ".join(str(c) for c in cells) + " |")


def header(*cells: str) -> None:
    print()
    row(*cells)
    row(*["---"] * len(cells))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def by_arm(runs: dict) -> dict[str, dict[int, dict]]:
    """``{arm: {seed: entry}}`` for the run keys of a probe or copy file."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for key, entry in runs.items():
        m = RUN.search(key)
        if m:
            out[m["arm"]][int(m["seed"])] = entry
    return out


def merged(root: Path, kind: str, tags: tuple[str, ...]) -> dict:
    runs: dict = {}
    for tag in tags:
        runs.update(load(root / kind / f"exp0004_{tag}.json")["runs"])
    return by_arm(runs)


def a_losses(b0: dict, cable: dict) -> None:
    header("seed", "B0 1x", "CABLE 1x", "B0 L_ext", "B0+YaRN-t L_ext", "CABLE L_ext", "gap")
    for s in sorted(b0):
        lb, ly, lc = b0[s]["loss"], b0[s]["loss_yarn_t"], cable[s]["loss"]
        ext_b, ext_y, ext_c = ((x["2"] + x["4"]) / 2 for x in (lb, ly, lc))
        row(s, f"{lb['1']:.4f}", f"{lc['1']:.4f}", f"{ext_b:.4f}", f"{ext_y:.4f}",
            f"{ext_c:.4f}", f"{ext_y - ext_c:+.4f}")  # fmt: skip
    header("arm", *(f"{m}x" for m in A_MULTS))
    for name, runs, key in (("B0", b0, "loss"), ("B0 + YaRN-t", b0, "loss_yarn_t"),
                            ("CABLE", cable, "loss")):  # fmt: skip
        first = next(iter(runs.values()))[key]
        row(name, *(ms([r[key][m] for r in runs.values()]) if m in first else "—"
                    for m in A_MULTS))  # fmt: skip


def a_retrieval(b0: dict, cable: dict) -> None:
    header("arm", "task", *(f"{m}x" for m in A_MULTS))
    for name, runs in (("B0", b0), ("CABLE", cable)):
        for task in ("passkey", "needle"):
            lengths = sorted(next(iter(runs.values()))["retrieval"][task], key=int)
            acc = [[statistics.fmean(r["retrieval"][task][n]["accuracy"]) for r in runs.values()]
                   for n in lengths]  # fmt: skip
            row(name, task, *(ms(a, 2) for a in acc))


def a_far_and_copy(root: Path) -> None:
    far = merged(root, "summability", ("a_rtx4090", "a_spark"))
    header("arm", "far mass 2x", "far mass 4x")
    for arm in ("b0", "cable"):
        row(arm, *(ms([r["far_mass"][m] for r in far[arm].values()]) for m in ("2", "4")))
    copy = merged(root, "copy", ("a_rtx4090", "a_spark"))
    header("arm", "N", "delta", "gain", "shuffled")
    for arm in ("b0", "cable"):
        runs = list(copy[arm].values())
        for n, deltas in runs[0]["lengths"].items():
            for d in deltas:
                cells = [r["lengths"][n][d] for r in runs]
                row(arm, n, d, ms([c["gain"] for c in cells]),
                    ms([c["gain_shuffled"] for c in cells]))  # fmt: skip


def part_a(root: Path) -> None:
    paths = glob.glob(str(root / "exp0004_a_*/*/eval_offline.json"))
    evals = by_arm({p: load(Path(p)) for p in paths})
    a_losses(evals["b0"], evals["cable"])
    a_retrieval(evals["b0"], evals["cable"])
    a_far_and_copy(root)


def b_probe(probe: dict) -> None:
    header("arm", *(f"{m}x" for m in B_MULTS))
    for arm in B_ARMS:
        row(arm, *(ms([r["loss"][m] for r in probe[arm].values()]) for m in B_MULTS))
    far = ("2", "4", "8", "16")
    header("arm", *(f"{m}x" for m in far), "16x - 4x", "per seed")
    for arm in B_ARMS:
        rs = [probe[arm][s]["far_mass"] for s in sorted(probe[arm])]
        growth = [r["16"] - r["4"] for r in rs]
        row(arm, *(ms([r[m] for r in rs]) for m in far), ms(growth),
            " / ".join(f"{g:.3f}" for g in growth))  # fmt: skip


def b_exponents(root: Path, probe: dict) -> None:
    exps = merged(root, "summability", ("b_exponents",))
    header("arm", "seed", "min", "p10", "median", "p90", "max", "at boundary", "predicted",
           "measured")  # fmt: skip
    for arm in B_ARMS[1:]:
        for s, e in sorted(exps[arm].items()):
            fm = probe[arm][s]["far_mass"]
            row(arm, s, f"{e['min']:.4f}", f"{e['p10']:.4f}", f"{e['median']:.3f}",
                f"{e['p90']:.3f}", f"{e['max']:.3f}", f"{e['share_at_boundary']:.2f}",
                f"{e['predicted_far_growth']:.4f}", f"{fm['16'] - fm['4']:.4f}")  # fmt: skip


def b_copy(root: Path) -> None:
    copy = merged(root, "copy", ("b",))
    deltas = ("256", "512", "1024", "2048")
    header("arm", *(f"delta={d}" for d in deltas))
    for arm in B_ARMS:
        rs = [r["lengths"]["2560"] for r in copy[arm].values()]
        row(arm, *(f"{statistics.fmean(r[d]['gain'] for r in rs):.3f} "
                   f"[{statistics.fmean(r[d]['gain_shuffled'] for r in rs):+.3f}]"
                   for d in deltas))  # fmt: skip


def part_b(root: Path) -> None:
    probe = merged(root, "summability", ("b",))
    b_probe(probe)
    b_exponents(root, probe)
    b_copy(root)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("part", choices=("a", "b"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    args = p.parse_args()
    (part_a if args.part == "a" else part_b)(args.runs)


if __name__ == "__main__":
    main()
