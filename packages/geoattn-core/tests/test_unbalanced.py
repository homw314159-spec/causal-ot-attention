"""Endpoints and identities of UnbalancedSinkhorn (gate G2 of experiment 0001)."""

import pytest
import torch

from geoattn_core import Sinkhorn, Softmax, UnbalancedSinkhorn
from geoattn_core.normalizers.base import log_normalize, prepare_log_kernel
from geoattn_core.normalizers.unbalanced import log_column_mass


def _causal(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool).tril()


@pytest.mark.parametrize("column_mass", ["prefix", "full"])
@pytest.mark.parametrize("eps", [0.7, 1.0])
def test_rho_zero_equals_softmax(column_mass, eps: float) -> None:
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 9, 9) * 3
    norm = UnbalancedSinkhorn(rho=0.0, n_iters=7, eps=eps, column_mass=column_mass)
    for mask in (None, _causal(9), torch.rand(9, 9) > 0.3):
        torch.testing.assert_close(norm(scores, mask), Softmax(eps=eps)(scores, mask))


@pytest.mark.parametrize("rho", [0.25, 0.9])
def test_single_iteration_equals_softmax(rho: float) -> None:
    torch.manual_seed(1)
    scores = torch.randn(2, 7, 7) * 3
    torch.testing.assert_close(
        UnbalancedSinkhorn(rho=rho, n_iters=1)(scores, _causal(7)), Softmax()(scores, _causal(7))
    )


@pytest.mark.parametrize("n_iters", [2, 5])
def test_full_rho_one_equals_sinkhorn(n_iters: int) -> None:
    """rho = 1 removes the damping: the full-column update IS the balanced one."""
    torch.manual_seed(2)
    scores = torch.randn(2, 6, 8) * 2
    torch.testing.assert_close(
        UnbalancedSinkhorn(rho=1.0, n_iters=n_iters, column_mass="full")(scores),
        Sinkhorn(n_iters=n_iters)(scores),
    )


def test_prefix_suppresses_first_token_mass() -> None:
    """Uniform scores: softmax puts H_n on column 1; prefix scaling lowers it (ADR-0005)."""
    n = 8
    scores = torch.zeros(1, n, n)
    harmonic = sum(1 / k for k in range(1, n + 1))
    softmax_mass = Softmax()(scores, _causal(n))[0, :, 0].sum().item()
    assert softmax_mass == pytest.approx(harmonic, rel=1e-6)
    masses = [
        UnbalancedSinkhorn(rho=rho, n_iters=5)(scores, _causal(n))[0, :, 0].sum().item()
        for rho in (0.25, 0.5, 0.9)
    ]
    assert softmax_mass > masses[0] > masses[1] > masses[2] > 1.0


def test_output_dtype_preserved() -> None:
    scores = torch.randn(2, 4, 4, dtype=torch.bfloat16)
    assert UnbalancedSinkhorn(rho=0.5, n_iters=3)(scores).dtype == torch.bfloat16


@pytest.mark.parametrize(
    "kwargs",
    [{"rho": -0.1}, {"rho": 1.5}, {"rho": 0.5, "n_iters": 0}, {"rho": 0.5, "column_mass": "all"}],
)
def test_invalid_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        UnbalancedSinkhorn(**kwargs)


def _relative_update(scores: torch.Tensor, mask: torch.Tensor, rho: float, n_iters: int):
    """The rejected ADR-0005 draft: damp a relative update of P (see ADR-0006)."""
    log_p = log_normalize(prepare_log_kernel(scores, mask, 1.0), dim=-1)
    for _ in range(n_iters - 1):
        step = -rho * log_column_mass(log_p, "prefix")
        log_p = log_normalize(torch.where(torch.isfinite(log_p), log_p + step, log_p), dim=-1)
    return log_p.exp()


def test_prefix_fixed_point_is_not_the_causal_identity() -> None:
    """The prefix operator converges to a non-degenerate coupling, while the relative
    update (control) drifts to the identity — the only lower-triangular doubly
    stochastic matrix (design.md §4.3) — so rho would be a mere step size there."""
    torch.manual_seed(3)
    n, rho = 32, 0.9
    scores, causal = torch.randn(2, n, n) * 2, _causal(n)
    norm = UnbalancedSinkhorn(rho=rho, n_iters=300)
    p = norm(scores, causal)
    residual = (p - UnbalancedSinkhorn(rho=rho, n_iters=299)(scores, causal)).abs().max()
    assert residual < 1e-5
    assert p.diagonal(dim1=-2, dim2=-1).mean() < 0.8
    control = _relative_update(scores, causal, rho, 300)
    assert control.diagonal(dim1=-2, dim2=-1).mean() > 0.95


def test_uniform_target_makes_uniform_attention_a_fixed_point() -> None:
    """Equal scores under a causal mask: the uniform target leaves every row uniform;
    the default target of 1 does not (control: it shifts mass toward recent keys)."""
    n = 12
    scores = torch.zeros(1, 1, n, n)
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    rows = torch.ones(n, n).tril() / torch.arange(1, n + 1)[:, None]
    uniform = UnbalancedSinkhorn(0.75, n_iters=20, column_target="uniform")(scores, causal)
    torch.testing.assert_close(uniform[0, 0], rows, atol=1e-6, rtol=0)
    one = UnbalancedSinkhorn(0.75, n_iters=20)(scores, causal)
    assert (one[0, 0] - rows).abs().max() > 1e-2
    assert one[0, 0, -1, -1] > one[0, 0, -1, 0]  # the default decays with distance


def test_uniform_target_rho_zero_is_softmax() -> None:
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 9, 9)
    causal = torch.ones(9, 9, dtype=torch.bool).tril()
    out = UnbalancedSinkhorn(0.0, n_iters=6, column_target="uniform")(scores, causal)
    torch.testing.assert_close(out, torch.softmax(scores.masked_fill(~causal, -1e9), -1))


def test_uniform_target_needs_prefix_mass() -> None:
    with pytest.raises(ValueError):
        UnbalancedSinkhorn(0.5, column_mass="full", column_target="uniform")
