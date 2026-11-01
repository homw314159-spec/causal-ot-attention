import torch
from torch import nn

from geoattn_core import GeometricAttention, IdentityPE, Sinkhorn, Softmax
from geoattn_obs import (
    AttentionMonitor,
    column_marginal_error,
    effective_rank,
    first_token_mass,
    rank_collapse_residual,
    row_entropy,
    row_marginal_error,
)


def test_marginal_errors_on_known_matrices() -> None:
    uniform = torch.full((1, 4, 4), 0.25)
    assert row_marginal_error(uniform) < 1e-6
    assert column_marginal_error(uniform) < 1e-6

    all_to_first = torch.zeros(1, 4, 4)
    all_to_first[..., 0] = 1.0  # row-stochastic, maximally unbalanced columns
    assert row_marginal_error(all_to_first) < 1e-6
    assert torch.isclose(column_marginal_error(all_to_first), torch.tensor(3.0))


def test_entropy_bounds() -> None:
    n = 8
    assert torch.isclose(
        row_entropy(torch.full((1, n, n), 1 / n)), torch.log(torch.tensor(n * 1.0))
    )
    assert row_entropy(torch.eye(n)[None]) < 1e-6


def test_effective_rank_extremes() -> None:
    assert torch.isclose(effective_rank(torch.eye(6)[None]), torch.tensor(6.0), atol=1e-4)
    rank1 = torch.outer(torch.randn(6), torch.randn(5))[None]
    assert torch.isclose(effective_rank(rank1), torch.tensor(1.0), atol=1e-3)


def test_collapse_residual() -> None:
    collapsed = torch.randn(1, 1, 16).expand(2, 10, 16)
    assert rank_collapse_residual(collapsed) < 1e-6
    assert rank_collapse_residual(torch.randn(2, 10, 16)) > 0.5


class _Block(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
        self.attn = GeometricAttention(16, 2, pe=IdentityPE(), normalizer=norm)

    def forward(self, x):
        return x + self.attn(x)


def test_monitor_collects_per_layer_and_detaches() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(_Block(Softmax()), _Block(Sinkhorn(n_iters=30)))
    monitor = AttentionMonitor(model, block_type=_Block)

    with monitor.capture():
        model(torch.randn(3, 7, 16))
    model(torch.randn(3, 7, 16))  # outside capture: must not be recorded

    s = monitor.summary()
    assert set(s) == {
        f"{i}{suffix}"
        for i in ("0.attn.probe", "1.attn.probe")
        for suffix in ("/row_err", "/col_err", "/entropy", "/eff_rank", "/sink")
    } | {"0/collapse_resid", "1/collapse_resid"}
    assert s["1.attn.probe/col_err"] < s["0.attn.probe/col_err"]  # Sinkhorn balances columns
    assert all(len(v) == 1 for v in monitor._values.values())


def test_first_token_mass_on_known_matrices() -> None:
    """test-plan §4.6: softmax on uniform causal scores -> H_n / n; identity -> 1/n;
    all mass on key 0 -> 1."""
    n = 9
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    uniform_causal = Softmax()(torch.zeros(1, n, n), causal)
    harmonic = sum(1 / k for k in range(1, n + 1))
    assert torch.isclose(first_token_mass(uniform_causal), torch.tensor(harmonic / n))
    assert torch.isclose(first_token_mass(torch.eye(n)[None]), torch.tensor(1 / n))
    all_to_first = torch.zeros(2, n, n)
    all_to_first[..., 0] = 1.0
    assert torch.isclose(first_token_mass(all_to_first), torch.tensor(1.0))


def test_first_token_mass_ignores_empty_rows() -> None:
    """Padding rows carry no mass, so they must not dilute the share."""
    p = torch.zeros(1, 4, 4)
    p[0, :2, 0] = 1.0  # two real rows, both fully on key 0; rows 2-3 are padding
    assert torch.isclose(first_token_mass(p), torch.tensor(1.0))
