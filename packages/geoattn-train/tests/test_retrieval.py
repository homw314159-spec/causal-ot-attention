"""Passkey / needle probes (experiment 0002, G3) on analytic models with known behaviour.

The gate asks for low scores on a model known to fail and high scores on one known to
succeed. Three hand-built "models" over a character-level tokenizer provide that:

* an induction oracle that copies whatever followed the latest earlier occurrence of the
  last ``K`` tokens (answers every probe: the query repeats the needle's phrase);
* the same oracle restricted to the last ``W`` tokens (answers only needles within reach,
  the pattern of a model that does not use far context);
* a unigram model that always predicts the space character (answers nothing).
"""

import random

import pytest
import torch

from geoattn_train.retrieval import (
    Grid,
    Sample,
    keys_by_answer_len,
    needle,
    passkey,
    run_grid,
    score,
)

V, K, TAIL = 256, 8, 12
CPU = torch.device("cpu")


def encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def decode(ids: list[int]) -> str:
    return "".join(chr(i) for i in ids)


def induction(window: int | None = None):
    def model(ids: torch.Tensor) -> torch.Tensor:
        b, n = ids.shape
        logits = torch.zeros(b, n, V)
        for r in range(b):
            seq = ids[r].tolist()
            for i in range(max(K - 1, n - TAIL), n):  # the scorer reads only the tail
                ctx = seq[i - K + 1 : i + 1]
                lo = K - 1 if window is None else max(K - 1, i - window)
                for j in range(i - 1, lo - 1, -1):
                    if seq[j - K + 1 : j + 1] == ctx:
                        logits[r, i, seq[j + 1]] = 1.0
                        break
        return logits

    return model


def unigram(ids: torch.Tensor) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, V)
    logits[..., ord(" ")] = 1.0
    return logits


def test_passkey_layout() -> None:
    s = passkey(encode, 600, 0.5, 12345)
    text = decode(s.tokens)
    assert len(s.tokens) == 600 and text.endswith("The pass key is 12345")
    assert decode(s.tokens[s.needle_pos :]).startswith(" The pass key is 12345. Remember it.")
    assert decode(s.tokens[-s.answer_len :]) == " 12345"
    with pytest.raises(ValueError, match="too short"):
        passkey(encode, 100, 0.5, 12345)


@pytest.mark.parametrize("depth", [0.0, 0.5, 1.0])
def test_needle_depth(depth: float) -> None:
    hay = encode("abcdefghij" * 100)
    s = needle(encode, hay, 400, depth, 54321)
    assert len(s.tokens) == 400
    context_len = 400 - len(encode(" The special magic number is 54321."))
    context_len -= len(encode(" The special magic number is")) + s.answer_len
    assert s.needle_pos == round(depth * context_len)
    with pytest.raises(ValueError, match="haystack"):
        needle(encode, hay[:50], 400, depth, 1)


def test_keys_share_a_token_count() -> None:
    keys = keys_by_answer_len(encode, 10, random.Random(0))
    assert len(keys) == 10 and len({len(encode(f" {k}")) for k in keys}) == 1


def _passkey_grid(model, lengths=(400, 800), depths=(0.1, 0.9)):
    def make(length: int, depth: float, key: int, trial: int) -> Sample:
        return passkey(encode, length, depth, key)

    grid = Grid(lengths=lengths, depths=depths, trials=4)
    return run_grid(model, make, encode, grid, CPU)


def test_oracle_succeeds_everywhere() -> None:
    out = _passkey_grid(induction())
    assert all(a == 1.0 for cell in out.values() for a in cell["accuracy"])


def test_unigram_fails_everywhere() -> None:
    out = _passkey_grid(unigram)
    assert all(a == 0.0 for cell in out.values() for a in cell["accuracy"])


def test_short_reach_fails_only_on_far_needles() -> None:
    """Window 300: at length 800 the needle at depth 0.1 is ~700 tokens back (missed) and
    at depth 0.9 within reach (found); at length 400 both are within reach."""
    out = _passkey_grid(induction(window=300))
    assert out["400"]["accuracy"] == [1.0, 1.0]
    assert out["800"]["accuracy"] == [0.0, 1.0]
    assert out["800"]["logprob"][0] < out["800"]["logprob"][1]


def test_needle_on_text_haystack() -> None:
    rng = random.Random(1)
    hay = [rng.randrange(97, 123) for _ in range(2000)]  # random lowercase letters

    def make(length: int, depth: float, key: int, trial: int) -> Sample:
        return needle(encode, hay[trial * 100 :], length, depth, key)

    grid = Grid(lengths=(500,), depths=(0.2, 0.8), trials=3)
    assert run_grid(induction(), make, encode, grid, CPU)["500"]["accuracy"] == [1.0, 1.0]
    assert run_grid(unigram, make, encode, grid, CPU)["500"]["accuracy"] == [0.0, 0.0]


def test_score_batches_and_shape_checks() -> None:
    samples = [passkey(encode, 400, 0.5, k) for k in (11111, 22222, 33333)]
    whole = score(induction(), samples, CPU)
    split = score(induction(), samples, CPU, tokens_per_batch=400)  # one sample per batch
    assert whole == split == (1.0, pytest.approx(whole[1]))
    with pytest.raises(ValueError, match="one length"):
        score(induction(), [samples[0], passkey(encode, 500, 0.5, 44444)], CPU)
