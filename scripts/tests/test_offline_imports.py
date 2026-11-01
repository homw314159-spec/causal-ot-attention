"""Importing evaluation helpers must not download retrieval vocabularies."""

import runpy
from pathlib import Path

import pytest
import tiktoken


@pytest.mark.parametrize("script", ["exp0002_eval", "probe_summability"])
def test_evaluation_import_stays_offline(script: str, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(scripts))

    def reject_load(*args, **kwargs):
        raise AssertionError("Tokenizer loaded while importing evaluation helpers")

    monkeypatch.setattr(tiktoken, "get_encoding", reject_load)
    runpy.run_path(str(scripts / f"{script}.py"))
