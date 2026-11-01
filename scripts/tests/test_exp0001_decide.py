"""The experiment 0001 decision rule on synthetic runs (scripts/exp0001_decide.py).

Each outcome of the rule has a case built to land just inside it, and each test of a
"beats" verdict has a control just outside it, so a wrong threshold or a missing seed
check fails a test rather than silently changing the decision.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "exp0001_decide", Path(__file__).parents[1] / "exp0001_decide.py"
)
assert _SPEC is not None and _SPEC.loader is not None
decide_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = decide_mod  # dataclasses look their module up while being created
_SPEC.loader.exec_module(decide_mod)
Run = decide_mod.Run

# Baseline seeds: val@1024 sd 0.002 (delta = floor 0.01); val@2048 sd 0.05 (delta_2 = 0.1).
SOFTMAX = {0: (3.050, 3.55), 1: (3.052, 3.60), 2: (3.048, 3.50)}


def _runs(arms: dict[str, dict[int, tuple[float, float]]], **over) -> list:
    base = {"step": 10, "steps": 10, "sha": "abc", "dirty": False, "gpu": "A100"}
    runs = []
    for arm, seeds in arms.items():
        for seed, (v1, v2) in seeds.items():
            residual = 1e-4 if arm.startswith("rho") else None
            runs.append(Run(arm, seed, v1, v2, **{**base, **over}, residual=residual))
    return runs


def _arms(rho05, logn=None, rho025=None) -> dict:
    flat = {s: (3.10, 3.60) for s in (0, 1, 2)}
    return {"softmax": SOFTMAX, "logn": logn or flat, "rho0.5": rho05, "rho0.25": rho025 or flat}


def _shift(d1: float, d2: float, seeds=(0, 1, 2)) -> dict[int, tuple[float, float]]:
    """Each baseline seed moved by (d1, d2)."""
    return {s: (SOFTMAX[s][0] + d1, SOFTMAX[s][1] + d2) for s in seeds}


def test_thresholds_follow_the_baseline_seeds() -> None:
    d = decide_mod.decide(_runs(_arms(_shift(0, 0))))
    assert d.sigma == pytest.approx(0.002)
    assert d.delta == pytest.approx(0.01)  # floor
    assert d.sigma2 == pytest.approx(0.05)
    assert d.delta2 == pytest.approx(0.10)


def test_tier1_path_c() -> None:
    d = decide_mod.decide(_runs(_arms(_shift(-0.015, 0))))
    assert d.best_rho == "rho0.5" and d.tier1.beats and d.outcome == "C"


def test_tier1_control_just_inside_delta() -> None:
    d = decide_mod.decide(_runs(_arms(_shift(-0.009, 0))))
    assert not d.tier1.beats and d.outcome == "A"


def test_tier1_needs_every_seed() -> None:
    rho = _shift(-0.03, 0)
    rho[2] = (SOFTMAX[2][0] + 0.001, SOFTMAX[2][1])  # one seed the other way
    d = decide_mod.decide(_runs(_arms(rho)))
    assert d.tier1.mean_diff > d.delta and not d.tier1.consistent
    assert d.outcome == "A" and d.notes


def test_tier2_needs_both_references() -> None:
    rho = _shift(-0.005, -0.3)
    logn_close = {s: (3.06, SOFTMAX[s][1] - 0.25) for s in (0, 1, 2)}  # within delta_2 of rho
    logn_far = {s: (3.06, SOFTMAX[s][1]) for s in (0, 1, 2)}
    assert decide_mod.decide(_runs(_arms(rho, logn=logn_far))).outcome == "C-long-context"
    d = decide_mod.decide(_runs(_arms(rho, logn=logn_close)))
    assert d.tier2_softmax.beats and not d.tier2_logn.beats and d.outcome == "A"


def test_tier2_uses_its_own_threshold() -> None:
    """0.05 on val@2048 would pass a val@1024-sized threshold but not delta_2 = 0.1."""
    d = decide_mod.decide(_runs(_arms(_shift(-0.005, -0.05))))
    assert d.tier2_softmax.mean_diff > d.delta and not d.tier2_softmax.beats
    assert d.outcome == "A"


def test_worse_rho_is_path_a() -> None:
    d = decide_mod.decide(_runs(_arms(_shift(0.05, -0.5), rho025=_shift(0.06, -0.5))))
    assert d.outcome == "A" and any("worse" in n for n in d.notes)


def test_best_rho_is_chosen_on_the_primary_metric() -> None:
    d = decide_mod.decide(_runs(_arms(_shift(-0.002, -0.5), rho025=_shift(-0.004, 0.0))))
    assert d.best_rho == "rho0.25"


def test_validate_flags_each_problem() -> None:
    runs = _runs(_arms(_shift(0, 0)))
    assert decide_mod.validate(runs) == []
    runs[0].dirty = True
    runs[1].step = 5
    runs[2].sha = "def"
    runs[-1].residual = 0.5
    problems = " ".join(decide_mod.validate(runs[:-2] + runs[-1:]))
    for needle in ("dirty", "incomplete", "commits", "not converged", "missing"):
        assert needle in problems
