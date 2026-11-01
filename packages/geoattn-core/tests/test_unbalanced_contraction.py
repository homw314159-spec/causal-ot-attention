"""Fixed-length convergence: the single-row log-scaling map is a rho-contraction."""

import pytest
import torch


def _row_map(
    z: torch.Tensor, scores: torch.Tensor, prefix: torch.Tensor, rho: float
) -> torch.Tensor:
    return rho * (z - (prefix + (scores + z).softmax(0)).log())


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.75, 0.99])
def test_log_scaling_jacobian_contracts_in_sup_norm(rho: float) -> None:
    """Check the derivative independently by autograd, including zero prefix masses."""
    generator = torch.Generator().manual_seed(71)
    scores = torch.randn(6, generator=generator, dtype=torch.float64)
    z = torch.randn(6, generator=generator, dtype=torch.float64)
    prefix = torch.tensor([0.0, 0.2, 3.0, 0.0, 9.0, 0.0], dtype=torch.float64)
    p = (scores + z).softmax(0)
    alpha = p / (prefix + p)
    expected = rho * (torch.diag(1 - alpha) + alpha[:, None] * p[None, :])
    jacobian = torch.autograd.functional.jacobian(lambda x: _row_map(x, scores, prefix, rho), z)
    torch.testing.assert_close(jacobian, expected)
    assert float(jacobian.min()) >= -1e-15
    torch.testing.assert_close(jacobian.abs().sum(1), torch.full_like(z, rho))
    other = 5 * torch.randn(6, generator=generator, dtype=torch.float64)
    distance = (_row_map(z, scores, prefix, rho) - _row_map(other, scores, prefix, rho)).abs()
    assert float(distance.max()) <= rho * float((z - other).abs().max()) + 1e-14


def test_without_damping_the_row_map_is_not_a_strict_contraction() -> None:
    """Counterexample control: removing rho leaves a common-shift eigenvalue of one."""
    scores = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
    z, prefix = torch.zeros_like(scores), torch.zeros_like(scores)
    difference = _row_map(z + 1, scores, prefix, 1.0) - _row_map(z, scores, prefix, 1.0)
    torch.testing.assert_close(difference, torch.ones_like(scores))
