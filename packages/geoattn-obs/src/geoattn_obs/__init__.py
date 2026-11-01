"""Attention observability.

The same metric code serves two lives (docs/methods.md §5):
stage 1 — logged during training runs;
stage 3 — exported from the serving platform (Prometheus).
"""

from geoattn_obs.metrics import (
    column_marginal_error,
    effective_rank,
    first_token_mass,
    rank_collapse_residual,
    row_entropy,
    row_marginal_error,
)
from geoattn_obs.monitor import AttentionMonitor

__all__ = [
    "AttentionMonitor",
    "column_marginal_error",
    "effective_rank",
    "first_token_mass",
    "rank_collapse_residual",
    "row_entropy",
    "row_marginal_error",
]
