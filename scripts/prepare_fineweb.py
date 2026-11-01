"""Tokenize a FineWeb-Edu subset into flat uint16 token files (experiment 0001).

    uv run python scripts/prepare_fineweb.py --out data/fineweb_edu
    uv run python scripts/prepare_fineweb.py --out data/fineweb_smoke --limit-docs 2000

Output (``--out``)::

    train.bin      uint16 GPT-2 token ids, documents concatenated, each prefixed by EOT
    val.bin        same layout, from a parquet shard disjoint from the training shards
    manifest.json  dataset revision, shards, tokenizer, token counts, sha256 of both files

Validation comes from a different shard than training, so no document is in both.
The dataset revision is resolved once and pinned, so a rerun reproduces the same
bytes (the sha256 in the manifest is what the experiment document records).
Downloads need network access; on the DGX Spark set ``HTTPS_PROXY``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceFW/fineweb-edu"
SHARD_PATH = "sample/10BT/{}_00000.parquet"
ENCODING = "gpt2"


@dataclass
class Split:
    name: str
    shards: list[str]
    max_tokens: int | None  # stop (and truncate) once this many tokens are written
    tokens: int = 0
    documents: int = 0
    sha256: str = ""
    shard_sha256: dict[str, str] = field(default_factory=dict)


@dataclass
class Job:
    """Everything fixed for one run of the script."""

    out: Path
    revision: str
    encoder: tiktoken.Encoding
    limit_docs: int | None
    threads: int


def main() -> None:
    args = _parse_args()
    job = Job(
        out=args.out,
        revision=HfApi().dataset_info(REPO, revision=args.revision).sha or args.revision,
        encoder=tiktoken.get_encoding(ENCODING),
        limit_docs=args.limit_docs,
        threads=args.threads,
    )
    job.out.mkdir(parents=True, exist_ok=True)
    splits = [
        Split("train", args.train_shards, args.train_tokens),
        Split("val", [args.val_shard], args.val_tokens),
    ]
    for split in splits:
        write_split(job, split)
    write_manifest(job, splits)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-shards", nargs="+", default=["000", "001", "002"])
    p.add_argument("--val-shard", default="013")
    p.add_argument("--train-tokens", type=int, default=None, help="default: all of the shards")
    p.add_argument("--val-tokens", type=int, default=20_000_000)
    p.add_argument("--revision", default="main", help="dataset revision; resolved to a sha")
    p.add_argument("--limit-docs", type=int, default=None, help="per split, for smoke runs")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()
    if set(args.train_shards) & {args.val_shard}:
        p.error("the validation shard must not be a training shard")
    return args


def write_split(job: Job, split: Split) -> None:
    final = job.out / f"{split.name}.bin"
    tmp = final.with_suffix(".bin.tmp")
    t0 = time.perf_counter()
    with tmp.open("wb") as f:
        for tokens in _token_batches(job, split):
            if split.max_tokens is not None:
                tokens = tokens[: split.max_tokens - split.tokens]
            f.write(tokens.tobytes())
            # Count what was written: the last batch may be cut by max_tokens.
            split.documents += int((tokens == job.encoder.eot_token).sum())
            split.tokens += len(tokens)
            if split.max_tokens is not None and split.tokens >= split.max_tokens:
                break
    tmp.replace(final)
    split.sha256 = sha256(final)
    rate = split.tokens / max(time.perf_counter() - t0, 1e-9) / 1e6
    print(f"{split.name}: {split.tokens:,} tokens, {split.documents:,} docs ({rate:.1f} M tok/s)")


def _token_batches(job: Job, split: Split) -> Iterator[np.ndarray]:
    eot = job.encoder.eot_token
    for shard in split.shards:
        path = download(shard, job.revision)
        split.shard_sha256[shard] = sha256(path)
        for texts in _text_batches(path, job.limit_docs):
            docs = job.encoder.encode_ordinary_batch(texts, num_threads=job.threads)
            yield to_uint16([tok for doc in docs for tok in (eot, *doc)])


def _text_batches(path: Path, limit_docs: int | None) -> Iterator[list[str]]:
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text"]):
        texts = batch.column("text").to_pylist()
        if limit_docs is not None:
            texts = texts[: limit_docs - seen]
        seen += len(texts)
        yield texts
        if limit_docs is not None and seen >= limit_docs:
            return


def to_uint16(ids: list[int]) -> np.ndarray:
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and arr.max() >= 2**16:  # GPT-2 has 50257 ids, but guard the cast
        raise ValueError(f"token id {arr.max()} does not fit in uint16")
    return arr.astype(np.uint16)


def download(shard: str, revision: str) -> Path:
    return Path(
        hf_hub_download(REPO, SHARD_PATH.format(shard), repo_type="dataset", revision=revision)
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(job: Job, splits: list[Split]) -> None:
    manifest = {
        "dataset": REPO,
        "revision": job.revision,
        "shard_path": SHARD_PATH,
        "tokenizer": {"encoding": ENCODING, "tiktoken": tiktoken.__version__},
        "eot_token": job.encoder.eot_token,
        "dtype": "uint16",
        "layout": "documents concatenated, each prefixed by eot_token",
        "limit_docs": job.limit_docs,
        "splits": {s.name: asdict(s) for s in splits},
    }
    path = job.out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {path}")


if __name__ == "__main__":
    main()
