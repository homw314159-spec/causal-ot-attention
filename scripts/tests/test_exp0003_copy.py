"""The distant-copy metric of experiment 0003 (gate G3, exp0003_copy.py) on synthetic inputs.

The layout is checked token by token. The gain is checked on an induction oracle, a model
that predicts the token that followed the previous occurrence of the current token: it
must show a large gain for the copy and none for the shuffled counter-example, whose
tokens are the same but whose order is not.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

_SPEC = importlib.util.spec_from_file_location(
    "exp0003_copy", Path(__file__).parents[1] / "exp0003_copy.py"
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

V, N, M = 1000, 96, 12


def _trials(count: int = 8):
    stream = np.random.default_rng(0).integers(0, V, 20_000).astype(np.uint16)
    return mod.make_trials(stream, 100, N, M, count, seed=3)


@pytest.mark.parametrize("delta", [M, 30, N - M])
def test_layout(delta: int) -> None:
    t = _trials(1)[0]
    first = N - M - delta
    for variant, expected in (("copy", t.a), ("control", t.b), ("shuffled", t.a[t.perm])):
        seq = mod.build(t, delta, variant)
        assert np.array_equal(seq[first : first + M], expected)
        assert np.array_equal(seq[-M:], t.a)
        rest = np.ones(N, dtype=bool)
        rest[first : first + M] = rest[-M:] = False
        assert np.array_equal(seq[rest], t.filler[rest])  # the filler is untouched


def test_trials_use_disjoint_text_and_fixed_permutations() -> None:
    a, b = _trials(), _trials()
    assert all(np.array_equal(x.perm, y.perm) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[0].a, a[0].b) and not np.array_equal(a[0].a, a[1].a)
    assert sorted(a[0].perm) == list(range(M)) and not np.array_equal(a[0].perm, np.arange(M))


@pytest.mark.parametrize("delta", [M - 1, N - M + 1])
def test_overlapping_or_out_of_range_delta_is_rejected(delta: int) -> None:
    with pytest.raises(ValueError, match="delta"):
        mod.build(_trials(1)[0], delta, "copy")


def test_fragment_loss_skips_the_first_token_of_the_fragment() -> None:
    seq = torch.randint(0, V, (1, N))
    logits = torch.full((1, N, V), -30.0)
    logits[0, torch.arange(N - 1), seq[0, 1:]] = 30.0  # every next token predicted
    assert mod.fragment_loss(logits, seq, M).item() < 1e-6
    wrong_first = logits.clone()
    wrong_first[0, N - M - 1] = 0.0  # the prediction of the fragment's first token: excluded
    assert mod.fragment_loss(wrong_first, seq, M).item() < 1e-6
    wrong_last = logits.clone()
    wrong_last[0, N - 2] = 0.0  # the prediction of its last token: included
    assert mod.fragment_loss(wrong_last, seq, M).item() > 0.1


class InductionOracle(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*x.shape, V)
        for b in range(x.shape[0]):
            last: dict[int, int] = {}
            for i, tok in enumerate(x[b].tolist()):
                if tok in last and last[tok] + 1 < x.shape[1]:
                    logits[b, i, x[b, last[tok] + 1]] = 8.0
                last[tok] = i
        return logits


def test_gain_on_an_induction_oracle() -> None:
    trials = _trials()
    r = mod.copy_gains(InductionOracle(), trials, 40, torch.device("cpu"))
    assert r["gain"] > 3.0 and r["gain"] > 10 * r["gain_se"]
    # the counter-example earns no credit: the oracle, confidently wrong after a shuffled
    # copy, even loses to the control
    assert r["gain_shuffled"] < 0.1 * r["gain"]


def test_deltas_fit_the_sequence() -> None:
    assert mod.deltas(512, 512, 128) == [256]
    assert mod.deltas(512, 2560, 128) == [256, 512, 1024, 2048]
    assert mod.deltas(1024, 5120, 128) == [512, 1024, 2048, 4096]
