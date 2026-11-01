"""RoPE length scaling (PI, NTK-aware, YaRN): each against its defining formula."""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core import RoPE
from geoattn_core.pe.rope import Scaling

METHODS: list[Scaling] = ["pi", "ntk", "yarn", "yarn_t"]
L = 16  # training length used throughout


def _qk(n: int, d: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    q, k = torch.randn(2, 1, 2, n, d, generator=g, dtype=torch.float64).unbind(0)
    return q, k


def _scores(pe: RoPE, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    rq, rk = pe.act(q, k, pos)
    return rq @ rk.transpose(-2, -1)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", [1, 8, L])
def test_identity_up_to_the_training_length(method: Scaling, n: int) -> None:
    q, k = _qk(n)
    pos = torch.arange(n)
    torch.testing.assert_close(
        _scores(RoPE(32, scaling=method, train_len=L), q, k, pos), _scores(RoPE(32), q, k, pos)
    )


def test_pi_divides_positions() -> None:
    n = 4 * L
    q, k = _qk(n)
    pos = torch.arange(n)
    plain = RoPE(32)
    rq, rk = plain.act(q, k, pos.to(torch.float32) / 4)
    torch.testing.assert_close(
        _scores(RoPE(32, scaling="pi", train_len=L), q, k, pos), rq @ rk.transpose(-2, -1)
    )


def test_ntk_raises_the_base() -> None:
    n, d = 4 * L, 32
    q, k = _qk(n, d)
    pos = torch.arange(n)
    expected = _scores(RoPE(d, base=10_000.0 * 4 ** (d / (d - 2))), q, k, pos)
    torch.testing.assert_close(_scores(RoPE(d, scaling="ntk", train_len=L), q, k, pos), expected)


def test_yarn_keeps_high_and_interpolates_low_frequencies() -> None:
    pe = RoPE(64, scaling="yarn", train_len=4096)
    inv, divisor = pe._scaled(4 * 4096)
    assert divisor == 1.0
    torch.testing.assert_close(inv[0], pe.inv_freq[0])  # fastest rotation: kept
    torch.testing.assert_close(inv[-1], pe.inv_freq[-1] / 4)  # slowest: interpolated
    assert ((inv <= pe.inv_freq + 1e-12) & (inv >= pe.inv_freq / 4 - 1e-12)).all()


@pytest.mark.parametrize("method", ["pi", "ntk", "yarn"])
def test_norms_preserved(method: Scaling) -> None:
    q, k = _qk(4 * L)
    rq, rk = RoPE(32, scaling=method, train_len=L).act(q, k, torch.arange(4 * L))
    torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1))
    torch.testing.assert_close(rk.norm(dim=-1), k.norm(dim=-1))


def test_yarn_temperature_scales_q_and_k() -> None:
    q, k = _qk(4 * L)
    pos = torch.arange(4 * L)
    yarn = _scores(RoPE(32, scaling="yarn", train_len=L), q, k, pos)
    yarn_t = _scores(RoPE(32, scaling="yarn_t", train_len=L), q, k, pos)
    torch.testing.assert_close(yarn_t, yarn * (0.1 * math.log(4) + 1) ** 2)


@settings(max_examples=20, deadline=None)
@given(method=st.sampled_from(METHODS), shift=st.integers(-500, 500), seed=st.integers(0, 999))
def test_scores_depend_only_on_relative_position(method: Scaling, shift: int, seed: int) -> None:
    """At a fixed length (so a fixed factor s), shifting all positions keeps the scores."""
    n = 3 * L
    q, k = _qk(n, seed=seed)
    pe = RoPE(32, scaling=method, train_len=L)
    pos = torch.arange(n) + 1000
    gap = _scores(pe, q, k, pos + shift) - _scores(pe, q, k, pos)
    scale = q.norm(dim=-1)[..., :, None] * k.norm(dim=-1)[..., None, :]
    assert (gap.abs() <= 1e-3 * scale).all()  # float32 angles at positions ~1.5e3


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        RoPE(32, scaling="linear")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RoPE(32, scaling="pi")
