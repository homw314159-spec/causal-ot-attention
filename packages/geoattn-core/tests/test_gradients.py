import pytest
import torch

from geoattn_core import ScaledSoftmax, Sinkhorn, Softmax, UnbalancedSinkhorn


@pytest.mark.parametrize("norm", [Softmax(eps=0.7), Sinkhorn(n_iters=4, eps=0.7)], ids=repr)
def test_gradcheck(norm) -> None:
    torch.manual_seed(0)
    scores = torch.randn(1, 4, 5, dtype=torch.float64, requires_grad=True)
    # Normalizers compute in float32 internally; gradcheck needs float64, so
    # check the float64-in/float64-out path at a tolerance matching float32.
    assert torch.autograd.gradcheck(lambda s: norm(s), (scores,), eps=1e-3, atol=1e-3, rtol=1e-2)


def test_gradcheck_with_mask() -> None:
    torch.manual_seed(1)
    scores = torch.randn(1, 5, 5, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(5, 5, dtype=torch.bool)
    mask[:, -1] = False
    mask[-1, :] = False
    norm = Sinkhorn(n_iters=3)
    assert torch.autograd.gradcheck(
        lambda s: norm(s, mask), (scores,), eps=1e-3, atol=1e-3, rtol=1e-2
    )


@pytest.mark.parametrize("column_mass", ["prefix", "full"])
@pytest.mark.parametrize("rho", [0.3, 0.8])
def test_gradcheck_unbalanced(rho: float, column_mass) -> None:
    torch.manual_seed(2)
    scores = torch.randn(1, 5, 5, dtype=torch.float64, requires_grad=True)
    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    norm = UnbalancedSinkhorn(rho=rho, n_iters=4, eps=0.7, column_mass=column_mass)
    for mask in (None, causal):
        assert torch.autograd.gradcheck(
            lambda s, m=mask: norm(s, m), (scores,), eps=1e-3, atol=1e-3, rtol=1e-2
        )


def test_gradcheck_scaled_softmax() -> None:
    torch.manual_seed(3)
    scores = torch.randn(1, 2, 5, 5, dtype=torch.float64, requires_grad=True)
    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    norm = ScaledSoftmax(num_heads=2, base_len=8, eps=0.7)
    for mask in (None, causal):
        assert torch.autograd.gradcheck(
            lambda s, m=mask: norm(s, m), (scores,), eps=1e-3, atol=1e-3, rtol=1e-2
        )


@pytest.mark.parametrize("rho", [0.3, 0.8])
def test_gradcheck_unbalanced_uniform_target(rho: float) -> None:
    torch.manual_seed(4)
    scores = torch.randn(1, 5, 5, dtype=torch.float64, requires_grad=True)
    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    norm = UnbalancedSinkhorn(rho=rho, n_iters=4, eps=0.7, column_target="uniform")
    for mask in (None, causal):
        assert torch.autograd.gradcheck(
            lambda s, m=mask: norm(s, m), (scores,), eps=1e-3, atol=1e-3, rtol=1e-2
        )
