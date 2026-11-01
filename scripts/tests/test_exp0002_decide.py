"""Experiment 0002 selection and decision rules on synthetic numbers (exp0002_decide.py).

Each outcome has a case built to land just inside it and a control just outside it, so a
wrong threshold, a missing seed check or a dropped condition fails a test rather than
silently changing the decision.
"""

import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "exp0002_decide", Path(__file__).parents[1] / "exp0002_decide.py"
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod  # dataclasses look their module up while being created
_SPEC.loader.exec_module(mod)
Eval = mod.Eval

SEEDS = (0, 1, 2)


def _arm(loss1: float, ext: float, yarn: float | None = None, jitter=(0.0, 0.0, 0.0)) -> dict:
    return {
        s: Eval(loss1, ext + j, None if yarn is None else yarn + j)
        for s, j in zip(SEEDS, jitter, strict=True)
    }


# B0: L_ext sd 0.02 -> delta_e = 0.04; 1x sd 0.001 -> delta_1 = floor 0.01.
B0 = {
    0: Eval(3.000, 3.500, 3.300),
    1: Eval(3.002, 3.520, 3.320),
    2: Eval(2.998, 3.480, 3.280),
}


def _table(prior: dict, **extra) -> dict:
    t = {
        "b0": B0,
        "logn": _arm(3.0, 3.4),
        "gm": _arm(3.0, 3.5),
        "gap": _arm(3.0, 3.5),
        "o2": _arm(3.0, 3.3),
        "p1": prior,
        "lc": _arm(3.0, 3.6, 3.6),
        "p3": _arm(3.0, 3.4, 3.4),
    }
    return {**t, **extra}


def test_thresholds_from_b0() -> None:
    delta_e, delta_1 = mod._thresholds(_table(_arm(3.0, 3.0)))
    assert abs(delta_e - 0.04) < 1e-9 and delta_1 == 0.01


def test_adopt_just_inside_and_control_outside() -> None:
    # B0 + YaRN-t mean 3.300: a prior at 3.255 clears it by 0.045 > 0.04
    d = mod.decide(_table(_arm(3.005, 3.255, 3.2)), ["p1", "lc"], "p3")
    assert d.primary.arm == "p1" and d.outcome == "adopt"
    d = mod.decide(_table(_arm(3.005, 3.265, 3.25)), ["p1", "lc"], "p3")  # 0.035: not enough
    assert d.outcome != "adopt" and not d.primary.vs_b0_yarn.beats and d.primary.vs_b0.beats


def test_seeds_must_agree() -> None:
    # mean clears the threshold, but seed 2 is worse than B0 + YaRN-t
    d = mod.decide(_table(_arm(3.0, 3.2, 3.2, jitter=(-0.05, -0.05, 0.11))), ["p1", "lc"], "p3")
    c = d.primary.vs_b0_yarn
    assert c.mean_diff > d.delta_e and not c.consistent
    assert d.outcome == "fallback"


def test_non_inferiority_at_1x() -> None:
    ok = mod.decide(_table(_arm(3.009, 3.2, 3.1)), ["p1", "lc"], "p3")  # +0.009 <= 0.01
    bad = mod.decide(_table(_arm(3.011, 3.2, 3.1)), ["p1", "lc"], "p3")  # +0.011 > 0.01
    assert ok.outcome == "adopt"
    assert bad.outcome == "fallback" and any("1x loss" in n for n in bad.notes)


def test_adopt_with_scaling() -> None:
    # alone it only matches B0 + YaRN-t; with YaRN-t it clears both
    d = mod.decide(_table(_arm(3.0, 3.30, 3.20)), ["p1", "lc"], "p3")
    assert d.outcome == "adopt-with-scaling"
    d = mod.decide(_table(_arm(3.0, 3.30, 3.27)), ["p1", "lc"], "p3")  # control: 0.03 short
    assert d.outcome == "fallback"


def test_prior_without_rope_has_no_scaling_row() -> None:
    d = mod.decide(_table(_arm(3.0, 3.30)), ["p1", "lc"], "p3")
    assert d.primary.scaled_vs_b0 is None and d.outcome == "fallback"
    assert any("no RoPE" in n for n in d.notes)


def test_unfreeze_ot_needs_both_legs() -> None:
    """O2 must beat the primary prior AND its own bias without OT: a bias stronger than the
    primary prior would otherwise unfreeze OT with no OT effect at all."""
    prior = _arm(3.0, 3.2, 3.1)
    yes = mod.decide(_table(prior, o2=_arm(3.0, 3.15), p3=_arm(3.0, 3.2)), ["p1", "lc"], "p3")
    assert yes.o2_over_primary.beats and yes.ot_effect.beats and yes.unfreeze_ot
    short = mod.decide(_table(prior, o2=_arm(3.0, 3.17), p3=_arm(3.0, 3.25)), ["p1", "lc"], "p3")
    assert not short.o2_over_primary.beats and not short.unfreeze_ot  # 0.03 < 0.04
    # control: O2 beats the primary prior only because its bias alone already does
    bias_only = mod.decide(
        _table(prior, o2=_arm(3.0, 3.15), p3=_arm(3.0, 3.16)), ["p1", "lc"], "p3"
    )
    assert bias_only.o2_over_primary.beats and not bias_only.ot_effect.beats
    assert not bias_only.unfreeze_ot


def test_primary_is_the_screening_first_not_the_better_one() -> None:
    """The prior under test is fixed before confirmation: screening #1 stays primary even
    when #2 does better in confirmation (#2 is reported, exploratory)."""
    d = mod.decide(_table(_arm(3.0, 3.40, 3.4), lc=_arm(3.0, 3.25, 3.2)), ["p1", "lc"], "p3")
    assert d.primary.arm == "p1" and d.secondary.arm == "lc"
    assert d.outcome == "fallback"  # decided on p1 alone
    assert d.secondary.row == "adopt"  # what #2 would have given: reported, not used


def test_grape_rule_just_inside_and_control_outside() -> None:
    prior = _arm(3.005, 3.255, 3.2)  # adopted on its own (see above)
    d = mod.decide(_table(prior, gm=_arm(3.0, 3.21)), ["p1", "lc"], "p3")  # G-M lower by 0.045
    assert d.outcome == "grape" and d.grape_over_primary["gm"].beats
    d = mod.decide(_table(prior, gm=_arm(3.0, 3.22)), ["p1", "lc"], "p3")  # 0.035: not enough
    assert d.outcome == "adopt"
    seeds_disagree = _arm(3.0, 3.20, jitter=(0.0, 0.0, 0.06))  # seed 2 of G-AP is worse
    d = mod.decide(_table(prior, gap=seeds_disagree), ["p1", "lc"], "p3")
    assert d.outcome == "adopt" and not d.grape_over_primary["gap"].consistent


def test_beats_grape_needs_both_arms() -> None:
    prior = _arm(3.005, 3.255, 3.2)
    assert mod.decide(_table(prior), ["p1", "lc"], "p3").primary.beats_grape  # GRAPE at 3.5
    d = mod.decide(_table(prior, gap=_arm(3.0, 3.27)), ["p1", "lc"], "p3")  # ahead by only 0.015
    assert d.outcome == "adopt" and d.primary.vs_gm.beats and not d.primary.beats_grape


def test_learning_rate_variant_by_1x_loss() -> None:
    t = _table(_arm(3.00, 3.3), p1_lr=_arm(2.99, 3.4))
    assert mod.resolve_priors(t, ["p1", "lc"], {"p1": "p1_lr"}) == ["p1_lr", "lc"]
    t = _table(_arm(3.00, 3.3), p1_lr=_arm(3.01, 3.2))  # control: worse at 1x, kept out
    assert mod.resolve_priors(t, ["p1", "lc"], {"p1": "p1_lr"}) == ["p1", "lc"]


def test_screening_selection_and_o2_bias() -> None:
    s = (0, 1)
    t = {a: {k: Eval(3.0, 3.5, None) for k in s} for a in mod.SCREEN_POOL}
    t["lc"] = {k: Eval(3.0, 3.1, None) for k in s}
    t["ga"] = {k: Eval(3.0, 3.2, None) for k in s}
    t["p3"] = {k: Eval(3.0, 3.3, None) for k in s}
    t["b0"] = {k: Eval(3.0, 2.0, None) for k in s}  # not in the pool, however good
    sel = mod.select_screening(t)
    assert sel.winners == ["lc", "ga"] and sel.o2_bias == "p3"


def _write_run(root: Path, name: str, seed: int, **meta_over) -> None:
    d = root / f"exp0002_confirm_{name}_s{seed}" / "20261101-000000"
    d.mkdir(parents=True)
    meta = {"steps": 10, "git": {"sha": "abc", "dirty": False}, "gpu": "A100"}
    meta.update(meta_over)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "metrics.jsonl").write_text(json.dumps({"step": 10}) + "\n", encoding="utf-8")
    ev = {"loss": {"1": 3.0, "2": 3.2, "4": 3.4, "8": 3.9}, "loss_yarn_t": {"2": 3.1, "4": 3.3}}
    (d / "eval_offline.json").write_text(json.dumps(ev), encoding="utf-8")


def test_loading_and_validity_checks(tmp_path: Path) -> None:
    for s in SEEDS:
        _write_run(tmp_path, "b0", s)
    runs = mod.load_runs(tmp_path, "confirm")
    assert len(runs) == 3 and abs(runs[0].eval.ext - 3.3) < 1e-9
    assert abs(runs[0].eval.ext_yarn - 3.2) < 1e-9
    assert mod.validate(runs, ["b0"], SEEDS) == []
    # controls: a dirty run, another GPU, a missing seed are each reported
    _write_run(tmp_path, "p1", 0, git={"sha": "abc", "dirty": True}, gpu="RTX 4090")
    problems = mod.validate(mod.load_runs(tmp_path, "confirm"), ["b0", "p1"], SEEDS)
    assert "p1_s0: dirty" in problems and "missing p1_s1" in problems
    assert any("several GPU models" in p for p in problems)
