"""Experiment tracking sinks (ADR-0004).

``metrics.jsonl`` in the run directory is the canonical record; a tracker only
mirrors each epoch record elsewhere. W&B is imported lazily so that runs with
tracking off (tests, CI) never touch it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from geoattn_train.config import ExperimentConfig, to_dict


class Tracker(Protocol):
    def log(self, record: dict[str, Any]) -> None: ...
    def finish(self, exit_code: int = 0) -> None: ...


class NullTracker:
    def log(self, record: dict[str, Any]) -> None:
        pass

    def finish(self, exit_code: int = 0) -> None:
        pass


class WandbTracker:
    def __init__(self, cfg: ExperimentConfig, run_dir: Path) -> None:
        import wandb  # lazy

        t = cfg.tracking
        self._run = wandb.init(
            project=t.project,
            entity=t.entity,
            group=t.group or cfg.name,
            name=f"{cfg.name}/{run_dir.name}",
            tags=t.tags,
            mode=t.mode,  # type: ignore[arg-type]
            dir=str(run_dir),
            config=to_dict(cfg),
        )
        self._best = -1.0
        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["wandb"] = {"id": self._run.id, "url": self._run.url, "mode": t.mode}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._best = max(self._best, record.get("test_acc", self._best))  # ViT only
        # commit=True: with only step=, W&B holds the row open until a larger step arrives.
        self._run.log(record, step=record.get("step", record.get("epoch")), commit=True)

    def finish(self, exit_code: int = 0) -> None:
        if self._best >= 0:
            self._run.summary["best_test_acc"] = self._best
        self._run.finish(exit_code=exit_code)


def make_tracker(cfg: ExperimentConfig, run_dir: Path) -> Tracker:
    return WandbTracker(cfg, run_dir) if cfg.tracking.wandb else NullTracker()
