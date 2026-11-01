"""Experiment 0004 decision rules on synthetic numbers (exp0004_decide.py, gate G5).

Each rule has a case just inside it and controls just outside it: the mean at the threshold,
one seed reversed, a failed side condition. A wrong threshold, a dropped direction check or
a missing condition fails a test rather than silently changing the decision.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(_SCRIPTS))  # exp0004_decide imports exp0003_decide
_SPEC = importlib.util.spec_from_file_location("exp0004_decide", _SCRIPTS / "exp0004_decide.py")
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

SA, SB = mod.SEEDS["a"], mod.SEEDS["b"]


def _by_seed(values, seeds) -> dict[int, float]:
    return dict(zip(seeds, values, strict=True))


# ---------------------------------------------------------------- H1c


def _eval(ext: float, loss1: float, yarn: bool) -> dict:
    losses = {"1": loss1, "2": ext, "4": ext}
    return {"loss": losses, "loss_yarn_t": losses} if yarn else {"loss": losses}


def _h1c(gaps, loss1_margin: float = 0.0) -> dict:
    table = {
        "b0": {s: _eval(3.0, 2.9, yarn=True) for s in SA},
        "cable": {
            s: _eval(3.0 - g, 2.9 + loss1_margin, yarn=False) for s, g in zip(SA, gaps, strict=True)
        },
    }
    return mod.h1c(table)


def test_h1c_holds_just_above_the_threshold() -> None:
    r = _h1c([0.0101] * 3)
    assert r["holds"] and mod.decide_a(r).startswith("extend")


@pytest.mark.parametrize(
    ("gaps", "margin"),
    [
        ([0.01] * 3, 0.0),  # mean at the threshold
        ([0.05, 0.05, -0.001], 0.0),  # one seed reversed
        ([0.05] * 3, 0.0101),  # 1x worse than allowed
    ],
)
def test_h1c_fails_outside(gaps, margin: float) -> None:
    r = _h1c(gaps, margin)
    assert not r["holds"] and mod.decide_a(r).startswith("reconsider")


# ---------------------------------------------------------------- H6, H7


def _b(m50_growth=0.01, l32_gaps=(0.05, 0.05, 0.05), m50_1x=0.0, growths=None) -> dict:
    """Synthetic part-B values; ``growths`` overrides arm -> per-seed far-mass growth."""
    growths = growths or {
        "p1gt1": (0.05,) * 3,
        "p1m25": (0.03,) * 3,
        "p1m50": (m50_growth,) * 3,
        "p4": (0.0,) * 3,
    }
    far4 = {a: _by_seed([0.1] * 3, SB) for a in mod.B_ARMS}
    far16 = {a: _by_seed([0.1 + g for g in growths[a]], SB) for a in mod.B_ARMS}
    loss32 = {a: _by_seed([3.5] * 3, SB) for a in mod.B_ARMS}
    loss32["p1m50"] = _by_seed([3.5 - g for g in l32_gaps], SB)
    loss1 = {a: _by_seed([3.4] * 3, SB) for a in mod.B_ARMS}
    loss1["p1m50"] = _by_seed([3.4 + m50_1x] * 3, SB)
    return {"loss1": loss1, "loss32": loss32, "far4": far4, "far16": far16}


def _h6(**kw) -> dict:
    return mod.h6(**_b(**kw))


def test_h6_holds_and_the_rule_is_adopted() -> None:
    r = _h6(m50_growth=0.0199, l32_gaps=(0.0101,) * 3, m50_1x=0.0099)
    assert r["holds"] and r["adopt_margin_rule"]
    assert mod.decide_b(r).startswith("keep")


@pytest.mark.parametrize(
    "kw",
    [
        {"m50_growth": 0.0201},  # far mass not flat
        {"l32_gaps": (0.01,) * 3},  # mean at the threshold
        {"l32_gaps": (0.05, 0.05, -0.001)},  # one seed reversed
    ],
)
def test_h6_fails_outside(kw: dict) -> None:
    r = _h6(**kw)
    assert not r["holds"] and not r["adopt_margin_rule"]
    assert mod.decide_b(r).startswith("drop")


def test_h6_holds_but_a_worse_1x_blocks_the_rule() -> None:
    r = _h6(m50_1x=0.0101)
    assert r["holds"] and not r["adopt_margin_rule"]
    assert mod.decide_b(r).startswith("H6 holds")


def test_h6_far_condition_is_on_the_seed_mean() -> None:
    growths = {"p1gt1": (0.05,) * 3, "p1m25": (0.03,) * 3, "p1m50": (0.0, 0.0, 0.05)}
    r = mod.h6(**_b(growths={**growths, "p4": (0.0,) * 3}))
    assert r["m50_flat"]  # mean 0.0167 although one seed grows by 0.05


def test_h7_holds_when_ordered_on_every_seed() -> None:
    v = _b()
    assert mod.h7(v["far4"], v["far16"])["holds"]


@pytest.mark.parametrize(
    "growths",
    [
        {"p1gt1": (0.05,) * 3, "p1m25": (0.05,) * 3, "p1m50": (0.01,) * 3},  # tie
        {"p1gt1": (0.05,) * 3, "p1m25": (0.03, 0.03, 0.005), "p1m50": (0.01,) * 3},  # one seed
    ],
)
def test_h7_fails_outside(growths: dict) -> None:
    v = _b(growths={**growths, "p4": (0.0,) * 3})
    assert not mod.h7(v["far4"], v["far16"])["holds"]


# ---------------------------------------------------------------- loading


def _probe(dirty: bool = False, drop: str | None = None) -> dict:
    v = _b()
    runs = {}
    for a in mod.B_ARMS:
        for s in SB:
            name = f"exp0004_b_{a}_s{s}"
            if name == drop:
                continue
            runs[f"{name}/20261026-000000"] = {
                "git": {"dirty": dirty},
                "loss": {"1": v["loss1"][a][s], "32": v["loss32"][a][s]},
                "far_mass": {"4": v["far4"][a][s], "16": v["far16"][a][s]},
            }
    runs["exp0003_b1_p1gt1_s0/x"] = {"git": {"dirty": False}, "loss": {}, "far_mass": {}}
    return {"runs": runs}


def test_load_b_round_trips_and_ignores_other_experiments() -> None:
    values, problems = mod.load_b(_probe())
    assert problems == []
    assert values == _b()


@pytest.mark.parametrize(
    ("kw", "message"),
    [({"dirty": True}, "dirty"), ({"drop": "exp0004_b_p1m25_s4"}, "missing exp0004_b_p1m25_s4")],
)
def test_load_b_blocks_a_decision(kw: dict, message: str) -> None:
    _, problems = mod.load_b(_probe(**kw))
    assert any(message in p for p in problems)


def _write_run(root: Path, name: str, ev: dict, dirty: bool = False, done: bool = True) -> None:
    d = root / name / "20261026-000000"
    d.mkdir(parents=True)
    meta = {"git": {"dirty": dirty}, "steps": 10}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "metrics.jsonl").write_text(json.dumps({"step": 10 if done else 5}), encoding="utf-8")
    (d / "eval_offline.json").write_text(json.dumps(ev), encoding="utf-8")


def test_load_a_finds_every_run_and_flags_problems(tmp_path: Path) -> None:
    for s in SA:
        _write_run(tmp_path, f"exp0004_a_b0_s{s}", _eval(3.0, 2.9, yarn=True))
    for s in SA[:-1]:
        _write_run(tmp_path, f"exp0004_a_cable_s{s}", _eval(2.9, 2.9, yarn=False), dirty=s == 8)
    table, problems = mod.load_a(tmp_path)
    assert set(table["b0"]) == set(SA)
    assert "exp0004_a_cable_s8: dirty" in problems
    assert "missing exp0004_a_cable_s10" in problems
