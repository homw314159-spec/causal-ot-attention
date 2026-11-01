"""Paper recipes preserve paired seeds, global batches and the selected tail settings."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from geoattn_train.config import load_config
from geoattn_train.lm_trainer import tokens_per_step

ROOT = Path(__file__).resolve().parents[2]
SPEC = yaml.safe_load((ROOT / "configs/reproduction.yaml").read_text(encoding="utf-8"))
LOADER = importlib.util.spec_from_file_location(
    "paper_training_recipe", ROOT / "scripts/train_experiment.py"
)
assert LOADER and LOADER.loader
RUNNER = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(RUNNER)


@pytest.mark.parametrize("stage", sorted(SPEC))
def test_recipe_global_batch_is_unchanged(stage: str) -> None:
    for command in RUNNER.commands(SPEC, stage):
        cfg = load_config(ROOT / command[4], command[5:])
        assert tokens_per_step(cfg) == 32768
        assert not cfg.tracking.wandb
        assert cfg.tracking.entity is None


def test_retests_keep_disjoint_seed_sets() -> None:
    assert SPEC["0002-confirm"]["seeds"] == [0, 1, 2]
    assert SPEC["0003-a1"]["seeds"] == [3, 4, 5, 6, 7]
    assert SPEC["0004-a"]["seeds"] == [8, 9, 10]
    assert "attention.logbias_margin=0.5" in SPEC["0004-b"]["arms"]["p1m50"]


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown arms"):
        RUNNER.commands(SPEC, "0004-b", ["unknown-arm"])
