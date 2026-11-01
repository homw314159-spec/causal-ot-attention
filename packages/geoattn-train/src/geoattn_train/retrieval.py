"""Long-context retrieval probes: passkey and needle-in-a-haystack (experiment 0002, G3).

Both hide a key at a chosen depth of a length-``L`` context and then ask for it by
repeating the phrase that introduced it, so a model answers by copying from far back:

* passkey (Mohtashami & Jaggi, 2023): repetitive filler sentences around
  ``The pass key is {key}. Remember it. {key} is the pass key.``
* needle: real text (a slice of the validation stream) around
  ``The special magic number is {key}.``

Scoring is teacher-forced: a sample counts as retrieved when every answer token is the
model's argmax given the prompt and the earlier answer tokens, which is exactly when greedy
decoding would reproduce the key. The mean answer log-probability is reported as well.

Tokenization is passed in (``encode``), so this module needs only torch; the evaluation
script uses GPT-2's tokenizer, the tests a character-level one.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

Encode = Callable[[str], list[int]]
Model = Callable[[Tensor], Tensor]  # (B, N) token ids -> (B, N, V) logits

PASSKEY_INTRO = (
    "There is an important info hidden inside a lot of irrelevant text. "
    "Find it and memorize them. I will quiz you about the important information there."
)
PASSKEY_FILLER = (
    " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
)


@dataclass(frozen=True)
class Sample:
    tokens: list[int]  # prompt followed by the answer; len == the target length
    answer_len: int
    needle_pos: int  # token index where the needle starts


def _place(
    context: list[int], needle: list[int], query: list[int], answer: list[int], depth: float
) -> Sample:
    at = round(depth * len(context))
    tokens = context[:at] + needle + context[at:] + query + answer
    return Sample(tokens, len(answer), at)


def passkey(encode: Encode, length: int, depth: float, key: int) -> Sample:
    intro, unit = encode(PASSKEY_INTRO), encode(PASSKEY_FILLER)
    needle = encode(f" The pass key is {key}. Remember it. {key} is the pass key.")
    query, answer = encode(" What is the pass key? The pass key is"), encode(f" {key}")
    n_fill = length - len(intro) - len(needle) - len(query) - len(answer)
    if n_fill < 0:
        raise ValueError(f"length {length} too short for the passkey template")
    filler = (unit * (n_fill // len(unit) + 1))[:n_fill]
    s = _place(filler, needle, query, answer, depth)
    return Sample(intro + s.tokens, s.answer_len, len(intro) + s.needle_pos)


def needle(encode: Encode, haystack: Sequence[int], length: int, depth: float, key: int) -> Sample:
    needle_ids = encode(f" The special magic number is {key}.")
    query, answer = encode(" The special magic number is"), encode(f" {key}")
    n_fill = length - len(needle_ids) - len(query) - len(answer)
    if not 0 <= n_fill <= len(haystack):
        raise ValueError(f"haystack of {len(haystack)} tokens cannot fill length {length}")
    return _place(list(haystack[:n_fill]), needle_ids, query, answer, depth)


@torch.no_grad()
def score(
    model: Model, samples: list[Sample], device: torch.device, tokens_per_batch: int = 8192
) -> tuple[float, float]:
    """(fraction retrieved, mean answer log-probability) for samples of one length.

    Batches hold about ``tokens_per_batch`` tokens: full-vocabulary logits of long
    contexts dominate memory."""
    if len({len(s.tokens) for s in samples}) != 1 or len({s.answer_len for s in samples}) != 1:
        raise ValueError("score one length and one answer length at a time (equal shapes)")
    a, n = samples[0].answer_len, len(samples[0].tokens)
    step = max(1, tokens_per_batch // n)
    hits, logps = [], []
    for i in range(0, len(samples), step):
        ids = torch.tensor([s.tokens for s in samples[i : i + step]], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(ids[:, :-1])
        logits = logits[:, -a:].float()  # the positions that predict the answer tokens
        target = ids[:, -a:]
        hits.append((logits.argmax(-1) == target).all(-1))
        ce = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
        logps.append(-ce.mean(-1))
    return torch.cat(hits).float().mean().item(), torch.cat(logps).mean().item()


@dataclass(frozen=True)
class Grid:
    lengths: tuple[int, ...]
    depths: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    trials: int = 20
    seed: int = 0


def keys_by_answer_len(encode: Encode, n: int, rng: random.Random) -> list[int]:
    """``n`` random 5-digit keys that tokenize to the same number of tokens (batchable)."""
    first = rng.randrange(10_000, 100_000)
    want = len(encode(f" {first}"))
    keys = [first]
    while len(keys) < n:
        k = rng.randrange(10_000, 100_000)
        if len(encode(f" {k}")) == want:
            keys.append(k)
    return keys


def run_grid(
    model: Model,
    make: Callable[[int, float, int, int], Sample],
    encode: Encode,
    grid: Grid,
    device: torch.device,
) -> dict[str, dict[str, list[float]]]:
    """``make(length, depth, key, trial)`` -> sample. Returns, per length, the retrieval
    rate and mean answer log-probability at each depth."""
    rng = random.Random(grid.seed)
    out: dict[str, dict[str, list[float]]] = {}
    for length in grid.lengths:
        acc, logp = [], []
        for depth in grid.depths:
            keys = keys_by_answer_len(encode, grid.trials, rng)
            samples = [make(length, depth, k, i) for i, k in enumerate(keys)]
            a, lp = score(model, samples, device)
            acc.append(a)
            logp.append(lp)
        out[str(length)] = {"accuracy": acc, "logprob": logp}
    return out
