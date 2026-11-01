"""UnbalancedSinkhorn(grad_mode="implicit"): fixed-point gradients (experimental).

The implicit gradient is exact at the fixed point; the unrolled one is the exact
gradient of the truncated iteration. With enough iterations the two must agree,
and prefix consistency (ADR-0005) must survive the adjoint solve.
"""

import pytest
import torch

from geoattn_core import UnbalancedSinkhorn


def _causal(n: int) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool).tril()


def _grad(norm, scores: torch.Tensor, mask, weights: torch.Tensor):
    x = scores.clone().requires_grad_()
    p = norm(x, mask)
    (p * weights).sum().backward()
    assert x.grad is not None
    return p.detach(), x.grad


@pytest.mark.parametrize("column_mass", ["prefix", "full"])
@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_matches_unrolled_gradient_at_convergence(rho: float, column_mass) -> None:
    torch.manual_seed(0)
    n = 12
    scores, weights = torch.randn(2, 3, n, n) * 2, torch.randn(2, 3, n, n)
    mask = _causal(n) if column_mass == "prefix" else None
    kw = {"rho": rho, "n_iters": 300, "column_mass": column_mass}
    p_u, g_u = _grad(UnbalancedSinkhorn(**kw), scores, mask, weights)
    p_i, g_i = _grad(UnbalancedSinkhorn(**kw, grad_mode="implicit"), scores, mask, weights)
    torch.testing.assert_close(p_i, p_u, atol=1e-6, rtol=0)
    torch.testing.assert_close(g_i, g_u, atol=1e-5, rtol=1e-4)


def test_forward_value_equals_unrolled_at_any_iteration_count() -> None:
    torch.manual_seed(1)
    scores = torch.randn(1, 2, 9, 9) * 3
    for k in (1, 2, 3, 7):
        a = UnbalancedSinkhorn(rho=0.7, n_iters=k)(scores, _causal(9))
        b = UnbalancedSinkhorn(rho=0.7, n_iters=k, grad_mode="implicit")(
            scores.requires_grad_(), _causal(9)
        )
        torch.testing.assert_close(b.detach(), a, atol=1e-6, rtol=0)


@pytest.mark.parametrize("row", [0, 3, 8])
def test_no_gradient_from_future_rows(row: int) -> None:
    """ADR-0005 through the adjoint: row t's gradient never reaches score rows > t."""
    torch.manual_seed(2)
    n = 10
    x = (torch.randn(1, 2, n, n) * 2).requires_grad_()
    p = UnbalancedSinkhorn(rho=0.8, n_iters=40, grad_mode="implicit")(x, _causal(n))
    (p[..., row, :] * torch.randn(n)).sum().backward()
    assert x.grad is not None
    assert x.grad[..., row + 1 :, :].abs().max().item() == 0.0


def test_gradcheck_at_the_fixed_point() -> None:
    torch.manual_seed(3)
    x = torch.randn(1, 1, 5, 5, dtype=torch.float64, requires_grad=True)
    norm = UnbalancedSinkhorn(rho=0.6, n_iters=200, eps=0.7, grad_mode="implicit")
    # check_undefined_grad probes backward with grad=None; the adjoint is a tensor hook,
    # which autograd never calls without a gradient, so that probe does not apply.
    assert torch.autograd.gradcheck(
        lambda s: norm(s, _causal(5)),
        (x,),
        eps=1e-3,
        atol=1e-3,
        rtol=1e-2,
        check_undefined_grad=False,
    )


def test_truncated_adjoint_approaches_the_exact_gradient() -> None:
    """backward_iters=m with backward_tol=0 is the phantom gradient: its error to the
    exact implicit gradient must shrink as m grows."""
    torch.manual_seed(4)
    n = 12
    scores, weights = torch.randn(2, 2, n, n) * 2, torch.randn(2, 2, n, n)
    kw = {"rho": 0.8, "n_iters": 200, "grad_mode": "implicit"}
    _, exact = _grad(UnbalancedSinkhorn(**kw), scores, _causal(n), weights)
    errors = []
    for m in (1, 5, 40):
        norm = UnbalancedSinkhorn(**kw, backward_iters=m, backward_tol=0.0)
        _, g = _grad(norm, scores, _causal(n), weights)
        errors.append((g - exact).abs().max().item())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-3 * exact.abs().max().item()


def test_padding_gives_finite_gradients() -> None:
    n, pad = 9, 3
    valid = torch.arange(n) >= pad
    mask = _causal(n) & valid[:, None] & valid[None, :]
    x = torch.randn(2, 2, n, n, requires_grad=True)
    p = UnbalancedSinkhorn(rho=0.7, n_iters=20, grad_mode="implicit")(x, mask)
    assert torch.isfinite(p).all()
    (p * torch.randn_like(p)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (x.grad[..., :pad, :] == 0).all()


def test_invalid_grad_mode() -> None:
    with pytest.raises(ValueError):
        UnbalancedSinkhorn(rho=0.5, grad_mode="adjoint")  # type: ignore[arg-type]
