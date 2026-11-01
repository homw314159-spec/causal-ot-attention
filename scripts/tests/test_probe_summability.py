"""probe_summability.py: hooked measurements after a FlexAttention pass on a compiled model.

The far mass reads the coupling through forward hooks on the AttentionProbes, which switch
the attention from FlexAttention to the reference path (``uses_flex``). With
``compile_model`` the blocks are compiled, and a loss pass at the same input shape leaves a
compiled graph that does not guard on hooks added later: reusing it, no hook fires and the
probe has nothing to average (experiment 0004, gate G2, 2026-10-25: loss and far mass at 4x
both run one sequence of 16,384 tokens).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPTS = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "probe_summability", _SCRIPTS / "probe_summability.py"
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

L = 64  # training length of the smoke config
N = 16384  # mean_loss batches one sequence from here on: the far-mass input shape


def _model(compile_model: bool) -> torch.nn.Module:
    from geoattn_train.config import load_config
    from geoattn_train.lm_trainer import build_lm

    cfg = load_config(
        _SCRIPTS.parent / "configs" / "smoke_lm.yaml",
        [
            "attention.normalizer.name=softmax",
            "model.attn_backend=flex",
            f"train.compile_model={str(compile_model).lower()}",
        ],
    )
    torch.manual_seed(0)
    return build_lm(cfg).cuda().eval()


@pytest.mark.gpu
def test_far_mass_after_a_compiled_flex_loss_pass_of_the_same_shape() -> None:
    val = np.random.default_rng(0).integers(0, 64, 4 * N).astype(np.uint16)
    device = torch.device("cuda")
    expected = mod.far_mass(_model(compile_model=False), val, N, L, device)
    model = _model(compile_model=True)
    mod.mean_loss(model, val, N, N, device)  # compiles the flex path for (1, N)
    got = mod.far_mass(model, val, N, L, device)
    assert expected > 0
    assert got == pytest.approx(expected, abs=1e-3)
    # and the flex path is compiled again afterwards, without the removed hooks
    assert mod.mean_loss(model, val, N, N, device) == pytest.approx(
        mod.mean_loss(_model(compile_model=False), val, N, N, device), abs=1e-2
    )
