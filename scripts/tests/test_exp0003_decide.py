"""Experiment 0003 decision rules on synthetic numbers (exp0003_decide.py, gate G6).

Each rule has a case just inside it and controls just outside it: the mean at the threshold,
one seed (checkpoint) reversed, a failed side condition. A wrong threshold, a dropped
direction check or a missing condition fails a test rather than silently changing the
decision.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "exp0003_decide", Path(__file__).parents[1] / "exp0003_decide.py"
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod  # dataclasses look their module up while being created
_SPEC.loader.exec_module(mod)

A1 = mod.SEEDS["a1"]


def _by_seed(values, seeds=A1) -> dict[int, float]:
    return dict(zip(seeds, values, strict=True))


# ---------------------------------------------------------------- H1


def _h1(gaps, loss1_margin: float = 0.0, seeds=A1) -> dict:
    """CABLE ``gaps`` below B0 + YaRN-t per seed; CABLE's 1x loss higher by the margin."""
    b0 = _by_seed([3.0 + 0.01 * i for i in range(len(seeds))], seeds)
    cable = {s: v - g for (s, v), g in zip(b0.items(), gaps, strict=True)}
    b0_1x = _by_seed([2.9] * len(seeds), seeds)
    cable_1x = {s: v + loss1_margin for s, v in b0_1x.items()}
    return mod.h1(b0, cable, b0_1x, cable_1x)


def test_h1_holds_just_above_the_threshold() -> None:
    r = _h1([0.0102] * 5)
    assert r["holds"] and r["ext"]["mean"] == pytest.approx(0.0102)


def test_h1_fails_at_the_threshold() -> None:
    assert not _h1([0.0099, 0.01, 0.01, 0.01, 0.0101])["holds"]  # mean 0.01: not above


def test_h1_fails_when_one_seed_reverses() -> None:
    r = _h1([0.03, 0.03, 0.03, 0.03, -0.001])
    assert r["ext"]["mean"] > mod.DELTA_A and not r["ext"]["every_one"] and not r["holds"]


def test_h1_fails_when_the_1x_loss_is_inferior() -> None:
    assert _h1([0.05] * 5, loss1_margin=0.0099)["holds"]
    r = _h1([0.05] * 5, loss1_margin=0.0101)
    assert r["ext"]["holds"] and not r["noninferior_1x"] and not r["holds"]


def test_h1_refuses_unpaired_seeds() -> None:
    with pytest.raises(ValueError, match="unpaired"):
        mod.h1({3: 3.0, 4: 3.0}, {3: 2.9}, {3: 2.0}, {3: 2.0})


@pytest.mark.parametrize(
    ("a", "b", "outcome"),
    [(True, True, "confirm: holds"), (True, False, "1024 only"), (False, True, "revise")],
)
def test_decision_a(a: bool, b: bool, outcome: str) -> None:
    assert outcome in mod.decide_a({"holds": a}, {"holds": b})


# ---------------------------------------------------------------- H2

B1 = mod.SEEDS["b1"]


def _h2(gaps=(0.02, 0.02, 0.02), gt1_growth=0.01, lt1_growth=0.2) -> dict:
    lt1 = _by_seed([4.0, 4.1, 4.2], B1)
    loss32 = {
        "p1lt1": lt1,
        "p1gt1": {s: v - g for (s, v), g in zip(lt1.items(), gaps, strict=True)},
    }
    far4 = {arm: _by_seed([0.1] * 3, B1) for arm in ("p1gt1", "p1lt1")}
    far16 = {
        "p1gt1": _by_seed([0.1 + gt1_growth] * 3, B1),
        "p1lt1": _by_seed([0.1 + lt1_growth] * 3, B1),
    }
    return mod.h2(loss32, far4, far16)


def test_h2_holds_inside_every_condition() -> None:
    assert _h2(gt1_growth=0.0199, lt1_growth=0.1001)["holds"]


@pytest.mark.parametrize(
    ("kwargs", "failed"),
    [
        ({"gaps": (0.01, 0.01, 0.01)}, "loss32"),  # mean at the threshold
        ({"gaps": (0.05, 0.05, -0.001)}, "loss32"),  # one seed reversed
        ({"gt1_growth": 0.0201}, "gt1_flat"),
        ({"lt1_growth": 0.0999}, "lt1_grows"),
    ],
)
def test_h2_fails_outside_any_condition(kwargs: dict, failed: str) -> None:
    r = _h2(**kwargs)
    assert not r["holds"]
    assert (not r[failed]["holds"]) if failed == "loss32" else not r[failed]


# ---------------------------------------------------------------- H3


def _tail(changes: dict[str, float]) -> dict[str, dict]:
    """Every §3.3 checkpoint; the loss moves by ``changes[mode]`` in the predicted direction
    (up for cap, down for repair and window)."""
    out = {}
    for mode, ckpts in mod.B2_CHECKPOINTS.items():
        sign = 1 if mode == "cap" else -1
        for c in ckpts:
            stage = c.split("_")[1]
            out[c] = {
                "mode": mode,
                "mult": mod.B2_MULT[stage],
                "loss": {"none": 4.0, mode: 4.0 + sign * changes[mode]},
            }
    return out


INSIDE = {"cap": 0.0101, "repair": 0.0101, "window": 0.1001}


def test_h3_holds_just_inside() -> None:
    tail = _tail(INSIDE)
    assert mod.validate_b2(tail) == []
    assert all(r["holds"] for r in mod.h3(tail).values())


@pytest.mark.parametrize("mode", ["cap", "repair", "window"])
def test_h3_fails_just_below_the_threshold(mode: str) -> None:
    edge = {"cap": 0.0099, "repair": 0.0099, "window": 0.0999}[mode]
    r = mod.h3(_tail({**INSIDE, mode: edge}))
    assert not r[mode]["holds"]
    assert all(r[m]["holds"] for m in r if m != mode)


@pytest.mark.parametrize("mode", ["cap", "repair", "window"])
def test_h3_fails_when_one_checkpoint_reverses(mode: str) -> None:
    tail = _tail({m: 0.5 for m in INSIDE})
    first = mod.B2_CHECKPOINTS[mode][0]
    tail[first]["loss"][mode] = 4.0 + (-0.001 if mode == "cap" else 0.001)
    r = mod.h3(tail)[mode]
    assert r["mean"] > 0.1 and not r["every_one"] and not r["holds"]


def test_b2_validation_catches_missing_and_misconfigured_checkpoints() -> None:
    tail = _tail(INSIDE)
    del tail["exp0002_confirm_lc_s2"]
    tail["exp0002_screen_p4_s0"]["mult"] = 8  # 25M checkpoints are judged at 16x
    tail["exp0002_screen_b0_s1"]["mode"] = "cap"
    problems = mod.validate_b2(tail)
    assert "missing exp0002_confirm_lc_s2" in problems
    assert any(p.startswith("exp0002_screen_p4_s0") for p in problems)
    assert any(p.startswith("exp0002_screen_b0_s1") for p in problems)


def test_b2_checkpoint_set_is_the_preregistered_one() -> None:
    sets = mod.B2_CHECKPOINTS
    assert len(sets["cap"]) == 8 + 9 and len(sets["repair"]) == 6 + 2
    assert len(sets["window"]) == 2 + 3
    assert "exp0002_confirm_lc_s0" not in sets["repair"]  # corrupt checkpoint (report 0014)


# ---------------------------------------------------------------- loading


def _write_run(root: Path, name: str, ev: dict, dirty: bool = False, step: int = 10) -> None:
    d = root / name / "20261013-000000"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"git": {"dirty": dirty}, "steps": 10}))
    (d / "metrics.jsonl").write_text(json.dumps({"step": step}) + "\n")
    (d / "eval_offline.json").write_text(json.dumps(ev))


def _ev(ext: float, yarn: float | None = None) -> dict:
    out: dict = {"loss": {"1": 3.0, "2": ext, "4": ext, "8": ext + 1}}
    if yarn is not None:
        out["loss_yarn_t"] = {"2": yarn, "4": yarn, "8": yarn + 1}
    return out


def test_part_a_end_to_end(tmp_path: Path) -> None:
    for part, seeds in (("a1", A1), ("a2", mod.SEEDS["a2"])):
        for s in seeds:
            _write_run(tmp_path, f"exp0003_{part}_b0_s{s}", _ev(3.6, yarn=3.4))
            _write_run(tmp_path, f"exp0003_{part}_cable_s{s}", _ev(3.35))
    args = type("A", (), {"part": "a", "runs": tmp_path})()
    result, problems = mod._run(args)
    assert problems == []
    assert result["h1a"]["ext"]["mean"] == pytest.approx(0.05)  # against YaRN-t, not plain B0
    assert result["decision"].startswith("confirm: holds")


def test_part_a_refuses_dirty_incomplete_and_missing_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, "exp0003_a1_b0_s3", _ev(3.6, 3.4), dirty=True)
    _write_run(tmp_path, "exp0003_a1_cable_s3", _ev(3.35), step=9)
    _, problems = mod.load_a(tmp_path, "a1")
    assert "exp0003_a1_b0_s3: dirty" in problems
    assert any("incomplete" in p for p in problems)
    assert "missing exp0003_a1_cable_s7" in problems


def test_part_b1_loading(tmp_path: Path) -> None:
    runs = {}
    for arm in mod.B1_ARMS:
        for s in B1:
            runs[f"exp0003_b1_{arm}_s{s}/20261013-000000"] = {
                "loss": {"32": 4.0 if arm == "p1gt1" else 4.1},
                "far_mass": {"4": 0.1, "16": 0.105 if arm == "p1gt1" else 0.3},
                "git": {"dirty": False},
            }
    values, problems = mod.load_b1({"runs": runs})
    assert problems == []
    assert mod.h2(**values)["holds"]
    del runs["exp0003_b1_p4_s2/20261013-000000"]
    runs["exp0003_b1_p1gt1_s0/20261013-000000"]["git"]["dirty"] = True
    _, problems = mod.load_b1({"runs": runs})
    assert "missing exp0003_b1_p4_s2" in problems and any("dirty" in p for p in problems)
