# Reproduction

Run commands from the repository root. The release preserves attention implementations,
model settings, data order and experimental thresholds. It removes provider-specific
launchers and vision experiments, retains the shared optimizer helpers, disables external
tracking by default, and omits cloud-account discovery from run metadata.

## Environment

```sh
uv sync --locked --all-packages
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The lock file pins the environment. On Linux x86-64 the default `cpu` group selects CPU
PyTorch; switch to `--no-group cpu --group gpu` for CUDA. Windows and Linux ARM64 select
the CUDA 13.0 wheel index, matching the development environment. GPU training needs a
compatible NVIDIA driver. The fused kernels require Triton; the PyTorch reference path
can run on CPU. Choose `attention.normalizer.impl=torch` for an explicit reference run,
and record that override when comparing timing or floating-point results.

## Data

Obtain FineWeb-Edu from its original host; data are not included in this repository.

```sh
uv run python scripts/prepare_fineweb.py --out data/fineweb_edu --revision 87f09149
```

The script uses the `sample/10BT` subset, training shards 000--002 and validation shard
013, GPT-2 tokenization, and a separate 20-million-token validation stream. It writes
`train.bin`, `val.bin` and a manifest with resolved revision, token counts and SHA-256
hashes. Confirm the resolved dataset revision and file hashes against the paper before
claiming an exact replication. A small pipeline check is available with `--limit-docs 2000`
and a separate output directory; those files are not a substitute for the paper corpus.

## Training

`configs/reproduction.yaml` contains the final paper recipes. Inspect a stage before
execution:

```sh
uv run python scripts/train_experiment.py 0002-confirm --arms cable_lr
uv run python scripts/train_experiment.py 0002-confirm --arms cable_lr --run
```

Use `--set data.root=... train.device=...` for local paths and devices. Changing model
hyperparameters, the global batch, token budgets or evaluation subsets changes the
experiment. Preserve paired seeds and data order. The paper's 100M label counts
non-embedding parameters; tied token embeddings are additional parameters.

For a single run, `geoattn-train --config <yaml> key=value ...` accepts the same dotted
overrides. Commit the public source before formal training so `meta.json` records a clean
public commit. Public commit hashes will differ from the development hashes in the paper;
do not relabel newly generated runs as the original ones. Single-GPU recipes keep the
specified global tokens per step; multi-GPU use requires attention to per-rank batching.

## Evaluation

```sh
uv run python scripts/exp0002_eval.py "runs/exp0002_confirm_*" --retrieval
uv run python scripts/probe_summability.py "runs/exp0002_confirm_*" --out runs/summability/confirm.json
uv run python scripts/exp0003_copy.py "runs/exp0002_confirm_*" --out runs/copy/confirm.json
uv run python scripts/exp0003_tail.py "runs/exp0002_confirm_*" --out runs/tail/confirm.json
uv run python scripts/measure_fixed_point.py runs/one_run/timestamp
```

The quoted patterns are resolved by the scripts. Replace the example checkpoint directory
with an actual completed run. `eval_offline.json` contains losses, evaluation-time YaRN
results and optional retrieval rates. Other probes write JSON to their requested paths.
Some evaluation scripts initialize the GPT-2 tokenizer and may download its public
vocabulary on first use. Training checkpoints should come from a trusted source.

See [experiments](experiments.md) for decision commands and the distinction between
preregistered decisions and exploratory analyses. `paper_figures.py` regenerates the
paper's plots from evaluation JSON; it needs those results, which are not bundled.

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -m "not gpu and not slow"
uv run pytest -m gpu
```

CPU checks are offline. GPU-marked checks need compatible hardware. Optional equivalence
tests skip when their external reference implementations are absent. The mathematical
tests cover row recursion, dilution, limiting shapes, contraction and linear-tail bounds;
they do not replace the analytic proofs in the paper.
