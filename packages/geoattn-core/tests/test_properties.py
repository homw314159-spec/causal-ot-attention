"""Mathematical properties of the normalizers, checked on random inputs.

Each test states the property it checks; together they are the executable
form of docs/methods.md §4.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core import ScaledSoftmax, Sinkhorn, Softmax, UnbalancedSinkhorn

SETTINGS = settings(max_examples=40, deadline=None)


@st.composite
def score_matrices(draw, max_n: int = 10, scale: float = 3.0) -> torch.Tensor:
    n_q = draw(st.integers(1, max_n))
    n_k = draw(st.integers(1, max_n))
    seed = draw(st.integers(0, 2**31 - 1))
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(2, n_q, n_k, generator=g) * 2 - 1) * scale


@SETTINGS
@given(scores=score_matrices(), n_iters=st.integers(1, 8))
def test_rows_sum_to_one_for_any_iteration_count(scores: torch.Tensor, n_iters: int) -> None:
    """Every output row is a convex combination: rows sum to 1, entries >= 0."""
    p = Sinkhorn(n_iters=n_iters)(scores)
    assert (p >= 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(p.shape[:-1]), atol=1e-5, rtol=0)


@SETTINGS
@given(scores=score_matrices())
def test_converges_to_uniform_marginals(scores: torch.Tensor) -> None:
    """With enough iterations columns reach n_q / n_k (doubly stochastic if square)."""
    n_q, n_k = scores.shape[-2:]
    p = Sinkhorn(n_iters=200)(scores)
    torch.testing.assert_close(p.sum(-2), torch.full((2, n_k), n_q / n_k), atol=1e-4, rtol=1e-4)


@SETTINGS
@given(scores=score_matrices())
def test_more_iterations_do_not_worsen_column_error(scores: torch.Tensor) -> None:
    """Column-marginal L1 error of the final iterate is no worse than softmax's."""
    n_q, n_k = scores.shape[-2:]
    target = n_q / n_k

    def col_err(k: int) -> float:
        return (Sinkhorn(n_iters=k)(scores).sum(-2) - target).abs().sum().item()

    assert col_err(20) <= col_err(1) + 1e-5


@SETTINGS
@given(scores=score_matrices(), n_iters=st.integers(1, 6))
def test_invariant_to_per_row_shifts(scores: torch.Tensor, n_iters: int) -> None:
    """S + a·1ᵀ gives the same output at every iteration (rows are normalized first)."""
    shift = torch.randn(*scores.shape[:-1], 1) * 5
    torch.testing.assert_close(
        Sinkhorn(n_iters=n_iters)(scores + shift),
        Sinkhorn(n_iters=n_iters)(scores),
        atol=1e-5,
        rtol=1e-4,
    )


@SETTINGS
@given(scores=score_matrices())
def test_fixed_point_invariant_to_per_column_shifts(scores: torch.Tensor) -> None:
    """S + 1·bᵀ changes softmax but not the Sinkhorn fixed point.

    Column shifts are diagonal rescalings of K = exp(S), and the doubly
    stochastic scaling of a positive matrix is unique (Sinkhorn 1964).
    """
    shift = torch.randn(*scores.shape[:-2], 1, scores.shape[-1]) * 2
    torch.testing.assert_close(
        Sinkhorn(n_iters=300)(scores + shift), Sinkhorn(n_iters=300)(scores), atol=1e-4, rtol=1e-3
    )


def test_small_eps_recovers_optimal_assignment() -> None:
    """As eps -> 0 entropic OT approaches unregularized OT: the optimal permutation.

    Note the iterate is NOT yet a hard permutation after 300 steps — Sinkhorn's
    contraction rate degrades like exp(-range(S)/eps), so small eps needs far
    more iterations (design.md §4.2). The argmax pattern is already correct.
    """
    from itertools import permutations

    torch.manual_seed(0)
    scores = torch.randn(5, 5)
    best = max(permutations(range(5)), key=lambda s: sum(scores[i, s[i]] for i in range(5)))
    p = Sinkhorn(n_iters=300, eps=0.02)(scores[None])[0]
    assert tuple(p.argmax(dim=-1).tolist()) == best


def test_softmax_column_marginals_are_not_uniform_in_general() -> None:
    """Sanity check that the property tests above are not vacuous."""
    torch.manual_seed(0)
    p = Softmax()(torch.randn(1, 8, 8) * 3)
    assert (p.sum(-2) - 1).abs().max() > 0.1


@SETTINGS
@given(
    scores=score_matrices(),
    rho=st.floats(0.0, 1.0),
    n_iters=st.integers(1, 8),
    column_mass=st.sampled_from(["prefix", "full"]),
)
def test_unbalanced_rows_sum_to_one(scores, rho: float, n_iters: int, column_mass) -> None:
    """Every iteration ends with a row normalization, for any rho and iteration count."""
    p = UnbalancedSinkhorn(rho=rho, n_iters=n_iters, column_mass=column_mass)(scores)
    assert (p >= 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(p.shape[:-1]), atol=1e-5, rtol=0)


@SETTINGS
@given(scores=score_matrices(), rho=st.floats(0.0, 1.0), n_iters=st.integers(1, 6))
def test_unbalanced_invariant_to_per_row_shifts(scores, rho: float, n_iters: int) -> None:
    shift = torch.randn(*scores.shape[:-1], 1) * 5
    norm = UnbalancedSinkhorn(rho=rho, n_iters=n_iters)
    torch.testing.assert_close(norm(scores + shift), norm(scores), atol=1e-5, rtol=1e-4)


@SETTINGS
@given(scores=score_matrices(), base_len=st.integers(2, 64))
def test_scaled_softmax_rows_sum_to_one(scores, base_len: int) -> None:
    n_q, n_k = scores.shape[-2:]
    p = ScaledSoftmax(base_len=base_len)(scores)
    assert (p >= 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(p.shape[:-1]), atol=1e-5, rtol=0)
    if n_q == n_k:
        causal = torch.ones(n_q, n_k, dtype=torch.bool).tril()
        pc = ScaledSoftmax(base_len=base_len)(scores, causal)
        torch.testing.assert_close(pc.sum(-1), torch.ones(pc.shape[:-1]), atol=1e-5, rtol=0)
