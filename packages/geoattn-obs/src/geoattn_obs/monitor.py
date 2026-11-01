"""Hook-based per-layer attention monitor.

Hooks attach to every :class:`geoattn_core.AttentionProbe` (the coupling P)
and, optionally, to blocks of a given type (token representations X). Metrics
are computed inside the hook and only scalars are kept, so no activations or
autograd graphs are retained. Hooks exist only inside ``capture()``, so a
model carries zero monitoring overhead outside of it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from geoattn_core import AttentionProbe
from geoattn_obs import metrics


class AttentionMonitor:
    def __init__(self, model: nn.Module, block_type: type[nn.Module] | None = None) -> None:
        self.model = model
        self.block_type = block_type
        self._values: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def capture(self) -> Iterator[AttentionMonitor]:
        handles: list[RemovableHandle] = []
        for name, module in self.model.named_modules():
            if isinstance(module, AttentionProbe):
                handles.append(module.register_forward_hook(self._coupling_hook(name)))
            elif self.block_type is not None and isinstance(module, self.block_type):
                handles.append(module.register_forward_hook(self._block_hook(name)))
        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    def reset(self) -> None:
        self._values.clear()

    def summary(self) -> dict[str, float]:
        """``{"<module path>/<metric>": mean over captured forwards}``."""
        return {k: sum(v) / len(v) for k, v in sorted(self._values.items())}

    def _coupling_hook(self, name: str):
        def hook(_m: nn.Module, _inp: tuple[Tensor, ...], p: Tensor) -> None:
            self._values[f"{name}/row_err"].append(metrics.row_marginal_error(p).item())
            self._values[f"{name}/col_err"].append(metrics.column_marginal_error(p).item())
            self._values[f"{name}/entropy"].append(metrics.row_entropy(p).item())
            self._values[f"{name}/eff_rank"].append(metrics.effective_rank(p).item())
            self._values[f"{name}/sink"].append(metrics.first_token_mass(p).item())

        return hook

    def _block_hook(self, name: str):
        def hook(_m: nn.Module, _inp: tuple[Tensor, ...], x: Tensor) -> None:
            self._values[f"{name}/collapse_resid"].append(metrics.rank_collapse_residual(x).item())

        return hook
