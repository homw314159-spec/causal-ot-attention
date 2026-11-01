"""Exponents and the prior-only prediction of exp0004_exponents.py.

The control for the boundary share is a head just outside the boundary band, which must not
be counted; the control for the prediction is the ordering of experiment 0004's margins.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(_SCRIPTS))  # exp0004_exponents imports exp0004_margin_prior
_PATH = _SCRIPTS / "exp0004_exponents.py"
_SPEC = importlib.util.spec_from_file_location("exp0004_exponents", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def _raw(c: torch.Tensor, margin: float) -> torch.Tensor:
    """Inverse of ``1 + margin + softplus``."""
    x = (c - 1 - margin).double()
    return x + torch.log(-torch.expm1(-x))


def test_exponents_invert_the_parameterization() -> None:
    c = torch.tensor([1.6, 2.0, 2.5])
    state = {
        "blocks.0.attn.pe.additive.raw": _raw(c[:2], 0.5),
        "blocks.1.attn.pe.additive.raw": _raw(c[2:], 0.5),
        "blocks.0.attn.qkv.weight": torch.zeros(2),
    }
    np.testing.assert_allclose(mod.exponents(state, 0.5), c.numpy(), rtol=1e-6)


def test_checkpoints_without_a_log_bias_are_refused() -> None:
    with pytest.raises(ValueError):
        mod.exponents({"blocks.0.attn.qkv.weight": torch.zeros(2)}, 0.0)


def test_boundary_share_excludes_heads_outside_the_band() -> None:
    c = np.array([1.5001, 1.509, 1.52, 2.0])
    assert mod.summary(c, 0.5)["share_at_boundary"] == 0.5


def test_predicted_growth_falls_with_the_margin() -> None:
    growth = [mod.summary(np.array([1 + m]), m)["predicted_far_growth"] for m in (0, 0.25, 0.5)]
    assert growth[0] > growth[1] > growth[2] > 0
    # the preregistered worst-case table (section 1): 0.043 for 0.25, 0.010 for 0.5
    assert growth[1] == pytest.approx(0.043, abs=5e-4)
    assert growth[2] == pytest.approx(0.010, abs=5e-4)
