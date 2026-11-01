"""FusedUnbalancedSinkhorn against the geoattn-core reference (ADR-0009, test-plan §8).

The reference is the ground truth: it is gradchecked in float64 (core tests), so the fused
gradients are compared with the reference's autograd gradients rather than gradchecked in
float32, where finite differences are too noisy to mean anything at these tolerances.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from geoattn_core import UnbalancedSinkhorn
from geoattn_kernels import FusedUnbalancedSinkhorn, is_causal_mask

cuda = pytest.mark.gpu
P_ATOL = 1e-5  # test-plan §8: elementwise match in float32
GRAD_RTOL = 1e-4  # max |d_fused - d_ref| / max |d_ref|


def _causal(n: int, device: str = "cpu") -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool, device=device).tril()


def _scores(shape: tuple[int, ...], scale: float, seed: int, device: str = "cuda") -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=g, device=device) * scale


def _gaps(fused, ref, scores: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """(max |P_fused - P_ref|, max |dS_fused - dS_ref| / max |dS_ref|) for one random cotangent."""
    s = scores.detach().requires_grad_()
    w = torch.randn_like(s)
    p_ref = ref(s, mask)
    (g_ref,) = torch.autograd.grad(p_ref, s, w)
    p = fused(s, mask)
    (g,) = torch.autograd.grad(p, s, w)
    # A single-column row has an identically zero gradient: guard the relative error.
    g_scale = g_ref.abs().max().clamp_min(1e-12)
    return (p - p_ref).abs().max().item(), ((g - g_ref).abs().max() / g_scale).item()


# ---------------------------------------------------------------- CPU (run in CI)


def test_is_causal_mask() -> None:
    assert is_causal_mask(_causal(5), 5)
    assert is_causal_mask(_causal(5)[None, None], 5)
    assert not is_causal_mask(None, 5)
    assert not is_causal_mask(torch.ones(5, 5, dtype=torch.bool), 5)
    assert not is_causal_mask(_causal(4), 5)
    padded = _causal(5).clone()
    padded[:, 4] = False  # causal plus key padding is not the pure causal mask
    assert not is_causal_mask(padded, 5)
    assert not is_causal_mask(_causal(5).expand(2, 5, 5), 5)  # per-batch masks are not covered


def test_cpu_falls_back_to_reference() -> None:
    scores = _scores((2, 3, 9, 9), 2.0, 0, device="cpu")
    fused = FusedUnbalancedSinkhorn(0.5, n_iters=5)
    assert not fused.covers(scores, _causal(9))
    torch.testing.assert_close(
        fused(scores, _causal(9)), UnbalancedSinkhorn(0.5, n_iters=5)(scores, _causal(9))
    )


def test_invalid_arguments_rejected() -> None:
    with pytest.raises(ValueError):
        FusedUnbalancedSinkhorn(1.5)
    with pytest.raises(ValueError):
        FusedUnbalancedSinkhorn(0.5, n_iters=0)


# ---------------------------------------------------------------- GPU

CASES = [
    # (n, rho, n_iters, eps, scale): powers of two and not, 1 row, k=1, rho at both ends,
    # the stable path (rho > 0.9), extreme logits, and the longest covered row.
    (1, 0.5, 5, 1.0, 1.0),
    (37, 0.5, 5, 1.0, 2.0),
    (64, 0.0, 3, 1.0, 3.0),
    (96, 0.5, 1, 1.0, 1.0),
    (128, 0.25, 7, 0.7, 3.0),
    (256, 0.75, 20, 1.0, 1.0),
    (256, 0.95, 20, 1.0, 8.0),
    (64, 1.0, 5, 1.0, 2.0),
    (512, 0.75, 26, 1.0, 20.0),
    (1000, 0.5, 13, 1.0, 3.0),
    (2048, 0.5, 13, 1.0, 3.0),
]


@cuda
@pytest.mark.parametrize(("n", "rho", "n_iters", "eps", "scale"), CASES)
def test_matches_reference(n: int, rho: float, n_iters: int, eps: float, scale: float) -> None:
    batch = (1, 2) if n >= 2048 else (2, 3)
    scores = _scores((*batch, n, n), scale, seed=n)
    fused = FusedUnbalancedSinkhorn(rho, n_iters=n_iters, eps=eps)
    ref = UnbalancedSinkhorn(rho, n_iters=n_iters, eps=eps)
    assert fused.covers(scores, _causal(n, "cuda"))
    p_gap, g_gap = _gaps(fused, ref, scores, _causal(n, "cuda"))
    assert p_gap <= P_ATOL
    assert g_gap <= GRAD_RTOL


@cuda
def test_match_check_detects_a_wrong_operator() -> None:
    """Control: the tolerances above must reject a nearby but different operator."""
    n = 128
    scores = _scores((2, 3, n, n), 3.0, seed=1)
    p_gap, g_gap = _gaps(
        FusedUnbalancedSinkhorn(0.5, n_iters=13),
        UnbalancedSinkhorn(0.45, n_iters=13),
        scores,
        _causal(n, "cuda"),
    )
    assert p_gap > 100 * P_ATOL
    assert g_gap > 10 * GRAD_RTOL


@cuda
@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(1, 160),
    rho=st.floats(0.0, 1.0),
    n_iters=st.integers(1, 12),
    scale=st.floats(0.1, 10.0),
    seed=st.integers(0, 2**31 - 1),
)
def test_properties(n: int, rho: float, n_iters: int, scale: float, seed: int) -> None:
    scores = _scores((1, 2, n, n), scale, seed)
    mask = _causal(n, "cuda")
    p = FusedUnbalancedSinkhorn(rho, n_iters=n_iters)(scores, mask)
    assert torch.isfinite(p).all()
    assert (p >= 0).all()
    assert (p[..., ~mask] == 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5, rtol=0)
    ref = UnbalancedSinkhorn(rho, n_iters=n_iters)(scores, mask)
    assert (p - ref).abs().max().item() <= P_ATOL


def _prefix_gap(norm, scores: torch.Tensor, t: int) -> float:
    n = scores.shape[-1]
    full = norm(scores, _causal(n, "cuda"))[..., :t, :t]
    prefix = norm(scores[..., :t, :t].contiguous(), _causal(t, "cuda"))
    return (full - prefix).abs().max().item()


@cuda
@settings(max_examples=30, deadline=None)
@given(n=st.integers(2, 200), data=st.data())
def test_prefix_consistency(n: int, data) -> None:
    """ADR-0005 for the fused operator (the core CAUSAL_NORMALIZERS list cannot import it)."""
    t = data.draw(st.integers(1, n))
    scores = _scores((1, 2, n, n), 3.0, data.draw(st.integers(0, 2**31 - 1)))
    assert _prefix_gap(FusedUnbalancedSinkhorn(0.5, n_iters=8), scores, t) <= 1e-5


@cuda
def test_prefix_detector_catches_known_leak() -> None:
    """Control: full-column scaling leaks future rows and the same check must flag it."""
    scores = _scores((1, 2, 64, 64), 3.0, seed=2)
    leaky = UnbalancedSinkhorn(0.5, n_iters=8, column_mass="full")
    assert _prefix_gap(leaky, scores, 20) > 1e-3


@cuda
def test_uncovered_masks_fall_back() -> None:
    n = 48
    scores = _scores((2, 3, n, n), 2.0, seed=3)
    fused, ref = FusedUnbalancedSinkhorn(0.5, n_iters=5), UnbalancedSinkhorn(0.5, n_iters=5)
    padded = _causal(n, "cuda").clone()
    padded[:, -5:] = False  # causal + key padding (the last 5 rows become fully masked)
    for mask in (None, padded, torch.ones(n, n, dtype=torch.bool, device="cuda")):
        assert not fused.covers(scores, mask)
        torch.testing.assert_close(fused(scores, mask), ref(scores, mask))


@cuda
def test_bf16_input() -> None:
    n = 256
    scores = _scores((2, 3, n, n), 3.0, seed=4).bfloat16()
    mask = _causal(n, "cuda")
    p = FusedUnbalancedSinkhorn(0.5, n_iters=13)(scores, mask)
    assert p.dtype == torch.bfloat16
    ref = UnbalancedSinkhorn(0.5, n_iters=13)(scores, mask)
    assert (p.float() - ref.float()).abs().max().item() <= 1e-2  # bf16 output rounding


@cuda
def test_mask_check_is_cached_per_mask_object() -> None:
    n = 32
    scores = _scores((1, 2, n, n), 1.0, seed=5)
    fused = FusedUnbalancedSinkhorn(0.5, n_iters=3)
    mask = _causal(n, "cuda")
    assert fused.covers(scores, mask)
    mask[0, 1] = True  # in-place edit bumps the version, so the verdict is recomputed
    assert not fused.covers(scores, mask)


@cuda
@pytest.mark.parametrize(
    ("n", "rho", "n_iters"), [(1, 0.5, 5), (37, 0.5, 8), (600, 0.75, 13), (1024, 0.5, 13)]
)
def test_uniform_target_matches_reference(n: int, rho: float, n_iters: int) -> None:
    scores = _scores((2, 3, n, n), 3.0, seed=n + 7)
    fused = FusedUnbalancedSinkhorn(rho, n_iters=n_iters, column_target="uniform")
    ref = UnbalancedSinkhorn(rho, n_iters=n_iters, column_target="uniform")
    p_gap, g_gap = _gaps(fused, ref, scores, _causal(n, "cuda"))
    assert p_gap <= P_ATOL
    assert g_gap <= GRAD_RTOL


@cuda
def test_uniform_target_differs_from_the_default() -> None:
    """Control: the target check above would catch a kernel that ignored the flag."""
    scores = _scores((1, 2, 128, 128), 3.0, seed=9)
    mask = _causal(128, "cuda")
    a = FusedUnbalancedSinkhorn(0.5, n_iters=8, column_target="uniform")(scores, mask)
    b = UnbalancedSinkhorn(0.5, n_iters=8)(scores, mask)
    assert (a - b).abs().max() > 100 * P_ATOL


@cuda
@settings(max_examples=20, deadline=None)
@given(n=st.integers(2, 160), data=st.data())
def test_uniform_target_prefix_consistency(n: int, data) -> None:
    t = data.draw(st.integers(1, n))
    scores = _scores((1, 2, n, n), 3.0, data.draw(st.integers(0, 2**31 - 1)))
    op = FusedUnbalancedSinkhorn(0.5, n_iters=8, column_target="uniform")
    assert _prefix_gap(op, scores, t) <= 1e-5
