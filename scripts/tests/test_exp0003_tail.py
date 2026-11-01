"""Run selection of exp0003_tail.py: the preregistered set by default, any run with --mode.

The control is the default path: an exploratory checkpoint outside the preregistered set
must be skipped unless an intervention is named, so the preregistered output cannot pick up
extra runs.
"""

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(_SCRIPTS))  # exp0003_tail imports its sibling scripts
_SPEC = importlib.util.spec_from_file_location("exp0003_tail", _SCRIPTS / "exp0003_tail.py")
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

EXPLORATORY = "exp0001x_confirm_softmax_logbias_s0"


def test_preregistered_runs_keep_their_intervention_and_multiple() -> None:
    for name, mode in mod.MODE.items():
        stage = mod.NAME.match(name)["stage"]
        assert mod.select(name, None, 8) == (mode, mod.B2_MULT[stage])


def test_runs_outside_the_preregistered_set_are_skipped_by_default() -> None:
    assert mod.select(EXPLORATORY, None, 8) is None
    assert mod.select("exp0002_confirm_p4_s9", None, 8) is None


def test_mode_applies_to_any_run() -> None:
    assert mod.select(EXPLORATORY, "repair", 8) == ("repair", 8)
    name = next(iter(mod.MODE))
    assert mod.select(name, "window", 4) == ("window", 4)
