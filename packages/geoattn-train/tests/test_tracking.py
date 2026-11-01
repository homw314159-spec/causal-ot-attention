"""W&B mirroring: every record must reach the run as soon as it is logged."""

from geoattn_train.tracking import WandbTracker


class _FakeRun:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []

    def log(self, data, **kwargs) -> None:
        self.calls.append((data, kwargs))


def test_records_are_committed_immediately() -> None:
    """With step= but no commit=True, W&B keeps a row open until a larger step
    arrives, so every point showed up one evaluation late (the last only at finish)."""
    tracker = WandbTracker.__new__(WandbTracker)  # skip wandb.init: no network in tests
    tracker._run = _FakeRun()  # type: ignore[assignment]
    tracker._best = -1.0
    tracker.log({"step": 1000, "val_loss@512": 4.5})
    tracker.log({"epoch": 3, "test_acc": 0.9})
    (_, lm_kwargs), (_, vit_kwargs) = tracker._run.calls  # type: ignore[attr-defined]
    assert lm_kwargs == {"step": 1000, "commit": True}
    assert vit_kwargs == {"step": 3, "commit": True}
