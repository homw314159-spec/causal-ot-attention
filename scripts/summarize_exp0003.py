"""Tables of report 0016 (experiment 0003) from the evaluation files, as markdown.

    uv run python scripts/summarize_exp0003.py a      # part A: L_ext, 1x, retrieval, copy gain
    uv run python scripts/summarize_exp0003.py b1     # part B1: loss 1x-32x, far mass, exponents
    uv run python scripts/summarize_exp0003.py b2     # part B2: interventions per checkpoint
    uv run python scripts/summarize_exp0003.py c      # part C: copy gain and per-head key reading

Inputs (under ``--runs``, default ``runs``): ``exp0003_*/*/eval_offline.json``,
``summability/exp0003_b1.json``, ``summability/key_heads.json``, ``tail/{screen,confirm}.json``
and ``copy/{exp0003_a,exp0003_b1,exp0002_screen,exp0002_confirm}.json``. Means are over seeds
with the sample standard deviation and n; the decisions themselves come from
``exp0003_decide.py``, not from these tables.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

RUN = re.compile(r"(?P<exp>exp000[23])_(?P<part>[a-z0-9]+)_(?P<arm>[a-z0-9_]+?)_s(?P<seed>\d)$")


def ms(xs: list[float], digits: int = 3) -> str:
    if len(xs) == 1:
        return f"{xs[0]:.{digits}f} (n=1)"
    return f"{statistics.fmean(xs):.{digits}f} ± {statistics.stdev(xs):.{digits}f} (n={len(xs)})"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def by_arm(entries: dict[str, dict]) -> dict[str, dict[int, dict]]:
    """``{run name or name/timestamp: value}`` -> arm -> seed -> value."""
    out: dict[str, dict[int, dict]] = {}
    for key, value in entries.items():
        m = RUN.match(key.split("/")[0])
        if m:
            out.setdefault(m["arm"], {})[int(m["seed"])] = value
    return out


# ---------------------------------------------------------------- part A


def _evals(root: Path, part: str) -> dict[str, dict[int, dict]]:
    found = {}
    for f in sorted(glob.glob(str(root / f"exp0003_{part}_*" / "*" / "eval_offline.json"))):
        found[Path(f).parents[1].name] = load(Path(f))
    return by_arm(found)


def _ext(loss: dict) -> float:
    return (loss["2"] + loss["4"]) / 2


def _retrieval(ev: dict, task: str) -> dict[str, float]:
    return {n: statistics.fmean(v["accuracy"]) for n, v in ev["retrieval"][task].items()}


def table_a(root: Path) -> None:
    copy_file = root / "copy" / "exp0003_a.json"
    copy = load(copy_file)["runs"] if copy_file.exists() else {}
    for part in ("a1", "a2"):
        ev = _evals(root, part)
        seeds = sorted(ev["b0"])
        print(f"\n#### {part.upper()}: seeds {seeds}\n")
        print(
            "| seed | B0 1× | CABLE 1× | B0 L_ext | B0+YaRN-t L_ext | CABLE L_ext | "
            "(B0+YaRN-t) − CABLE |"
        )
        print("|---|---|---|---|---|---|---|")
        for s in seeds:
            b, c = ev["b0"][s], ev["cable"][s]
            gap = _ext(b["loss_yarn_t"]) - _ext(c["loss"])
            print(
                f"| {s} | {b['loss']['1']:.4f} | {c['loss']['1']:.4f} | {_ext(b['loss']):.4f} | "
                f"{_ext(b['loss_yarn_t']):.4f} | {_ext(c['loss']):.4f} | {gap:+.4f} |"
            )
        print("\n| arm | 1× | 2× | 4× | 8× |")
        print("|---|---|---|---|---|")
        for name, arm, key in (
            ("B0", "b0", "loss"),
            ("B0 + YaRN-t", "b0", "loss_yarn_t"),
            ("CABLE", "cable", "loss"),
        ):
            cells = [
                ms([ev[arm][s][key][m] for s in seeds]) if m in ev[arm][seeds[0]][key] else "—"
                for m in ("1", "2", "4", "8")
            ]
            print(f"| {name} | " + " | ".join(cells) + " |")
        print("\nRetrieval accuracy (mean over 5 depths, 20 trials each):\n")
        lengths = list(ev["b0"][seeds[0]]["retrieval"]["passkey"])
        print(
            "| arm | task | " + " | ".join(f"{int(n) // int(lengths[0])}×" for n in lengths) + " |"
        )
        print("|---|---|" + "---|" * len(lengths))
        for arm in ("b0", "cable"):
            for task in ("passkey", "needle"):
                cells = [ms([_retrieval(ev[arm][s], task)[n] for s in seeds], 2) for n in lengths]
                print(f"| {arm} | {task} | " + " | ".join(cells) + " |")
        _copy_rows(copy, part)


def _copy_rows(copy: dict, part: str) -> None:
    rows = by_arm({k: v for k, v in copy.items() if k.startswith(f"exp0003_{part}_")})
    if not rows:
        print("\n(copy gain not available)")
        return
    big = max(next(iter(rows["b0"].values()))["lengths"], key=int)
    print(f"\nCopy gain at N = {big} (nats; shuffled control in brackets):\n")
    _copy_table(rows, ("b0", "cable"), big)


# ---------------------------------------------------------------- part B1


def _exponents(root: Path, arm: str) -> list[torch.Tensor]:
    out = []
    for p in sorted(glob.glob(str(root / f"exp0003_b1_{arm}_s*" / "*" / "last.pt"))):
        sd = torch.load(p, map_location="cpu", weights_only=True)["model"]
        raw = torch.cat([v for k, v in sd.items() if k.endswith("additive.raw")])
        if arm == "p1gt1":
            out.append(1 + F.softplus(raw))
        elif arm == "p1lt1":
            out.append(torch.sigmoid(raw))
        else:
            out.append(F.softplus(raw))
    return out


def table_b1(root: Path) -> None:
    probe = by_arm(load(root / "summability" / "exp0003_b1.json")["runs"])
    copy = by_arm(load(root / "copy" / "exp0003_b1.json")["runs"])
    arms = ("p4", "p1gt1", "p1free", "p1lt1")
    mults = ("1", "2", "4", "8", "16", "32")
    print("\n| arm | " + " | ".join(f"{m}×" for m in mults) + " | Δ32 = 32× − 8× |")
    print("|---|" + "---|" * (len(mults) + 1))
    for arm in arms:
        runs = probe[arm].values()
        cells = [ms([r["loss"][m] for r in runs]) for m in mults]
        delta = ms([r["loss"]["32"] - r["loss"]["8"] for r in runs])
        print(f"| {arm} | " + " | ".join(cells) + f" | {delta} |")
    far = ("2", "4", "8", "16")
    print("\nFar mass (keys more than L back, natural text):\n")
    print("| arm | " + " | ".join(f"{m}×" for m in far) + " | 16× − 4× |")
    print("|---|" + "---|" * (len(far) + 1))
    for arm in arms:
        runs = probe[arm].values()
        cells = [ms([r["far_mass"][m] for r in runs]) for m in far]
        growth = ms([r["far_mass"]["16"] - r["far_mass"]["4"] for r in runs])
        print(f"| {arm} | " + " | ".join(cells) + f" | {growth} |")
    print("\nTrained exponents (all 64 heads of each run; per seed):\n")
    print("| arm | seed | min | p10 | median | p90 | max | share c < 1.1 |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ("p1gt1", "p1free", "p1lt1"):
        for s, c in enumerate(_exponents(root, arm)):
            q = torch.quantile(c, torch.tensor([0.1, 0.5, 0.9]))
            print(
                f"| {arm} | {s} | {c.min():.4f} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | "
                f"{c.max():.3f} | {(c < 1.1).float().mean():.2f} |"
            )
    print("\nCopy gain at N = 5L = 2560 (nats; shuffled control in brackets):\n")
    _copy_table(copy, arms, "2560")


def _copy_table(copy: dict, arms, n: str) -> None:
    some = next(iter(copy[arms[0]].values()))["lengths"][n]
    deltas = list(some)
    print("| arm | " + " | ".join(f"Δ={d}" for d in deltas) + " |")
    print("|---|" + "---|" * len(deltas))
    for arm in arms:
        if arm not in copy:
            continue
        runs = list(copy[arm].values())
        cells = [
            f"{statistics.fmean(r['lengths'][n][d]['gain'] for r in runs):.3f} "
            f"[{statistics.fmean(r['lengths'][n][d]['gain_shuffled'] for r in runs):+.3f}]"
            for d in deltas
        ]
        print(f"| {arm} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------- part B2


def table_b2(root: Path) -> None:
    for stage in ("screen", "confirm"):
        runs = load(root / "tail" / f"{stage}.json")["runs"]
        mult = str(next(iter(runs.values()))["mult"])
        print(f"\n#### {stage} (decisive length {mult}×)\n")
        print(
            f"| checkpoint | intervention | loss 1× | loss {mult}× before | after | change | "
            f"far mass {mult}× before | after |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for name in sorted(runs, key=lambda n: (runs[n]["mode"], n)):
            r = runs[name]
            mode, before, after = r["mode"], r["curves"]["none"], r["curves"][r["mode"]]
            print(
                f"| {name.split('_', 2)[2]} | {mode} | {before['loss']['1']:.4f} | "
                f"{before['loss'][mult]:.4f} | {after['loss'][mult]:.4f} | "
                f"{after['loss'][mult] - before['loss'][mult]:+.4f} | "
                f"{before['far_mass'][mult]:.3f} | {after['far_mass'][mult]:.3f} |"
            )


# ---------------------------------------------------------------- part C


def table_c(root: Path) -> None:
    for stage, n in (("screen", "2560"), ("confirm", "5120")):
        copy = by_arm(load(root / "copy" / f"exp0002_{stage}.json")["runs"])
        print(f"\n#### experiment 0002 {stage}: copy gain at N = {n}\n")
        _copy_table(copy, sorted(copy), n)
    heads = load(root / "summability" / "key_heads.json")["runs"]
    print("\n#### Per-head key reading (100M CABLE, passkey at depth 0.1)\n")
    print("| run | length | mean | max head (layer, head) | max | top-3 share of total |")
    print("|---|---|---|---|---|---|")
    for key in sorted(heads):
        r = heads[key]
        for m, table in r["key_heads"].items():
            t = torch.tensor(table)
            flat = t.flatten()
            i = int(flat.argmax())
            top3 = float(flat.sort(descending=True).values[:3].sum() / flat.sum())
            print(
                f"| {key.split('/')[0][16:]} | {m}× | {r['key_attention'][m]:.4f} | "
                f"({i // t.shape[1]}, {i % t.shape[1]}) | {float(flat.max()):.3f} | "
                f"{top3:.2f} |"
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("part", choices=("a", "b1", "b2", "c"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]  # × and − on Windows
    {"a": table_a, "b1": table_b1, "b2": table_b2, "c": table_c}[args.part](args.runs)


if __name__ == "__main__":
    main()
