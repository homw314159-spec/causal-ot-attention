# Paper experiments

This document maps the paper's experiments to executable configurations and scripts.
The experiments were preregistered in the development repository. This public snapshot
summarizes the final protocols; its creation date is not a new preregistration, and it
does not provide an independent timestamp for the original private records.

| Stage | Configuration | Seeds | Main comparison |
| --- | --- | --- | --- |
| `0001-confirm` | `exp0001/confirm.yaml` | 0--2 | Softmax, log-n, OT with rho 0.25 and 0.5 |
| `0002-screen` | `exp0002/screen.yaml` | 0--1 | Fifteen positional/normalization arms |
| `0002-confirm` | `exp0002/confirm.yaml` | 0--2 | Selected 100M arms and learning-rate control |
| `0003-a1` | `exp0002/confirm.yaml` | 3--7 | CABLE vs B0 + evaluation-time YaRN-t, length 1024 |
| `0003-a2` | Same base, length 2048 | 3--5 | Same paired comparison |
| `0003-b1` | `exp0002/screen.yaml` | 0--2 | Free, above-one and below-one exponents; ALiBi |
| `0004-a` | Confirmation base, length 4096 | 8--10 | Same paired comparison at length 4096 |
| `0004-b` | Screening base | 3--5 | Exponent margins 0, 0.25, 0.5; ALiBi |

The recipe includes the recorded O2 choice (OT + ALiBi). It does not rerun screening
selection or learning-rate calibration automatically. The confirmation arm `cable_lr`
uses 7.5e-4; `cable` keeps the base 5.3e-4 rate. Later CABLE retests use 7.5e-4. The O2
confirmation recipe disables model compilation, matching the recorded fallback.

The paper tests loss extrapolation through comparisons, test-time tail interventions and
training-time exponent constraints. Low far mass and stable loss do not establish
effective use of distant content. Far-mass, retrieval, copying and mechanism probes are
exploratory unless explicitly included in a decision rule.

## Outputs and decisions

- `exp0001_decide.py` checks the original per-seed loss thresholds. The training-length
  softmax comparison failed its threshold; this is not replaced by the retest results.
- `exp0002_decide.py` performs screening selection and the original confirmation decision.
  Its original CABLE adoption test used 0.117 and failed. The later 0.01 retest is separate.
- `exp0003_decide.py` checks the paired retests, exponent constraints and tail interventions.
  Positive directions must hold per seed/checkpoint; the stated mean thresholds also apply.
- `exp0004_decide.py` checks the length-4096 retest and margin experiments.
- `summarize_exp0003.py` and `summarize_exp0004.py` create descriptive tables from JSON;
  they do not replace the decision scripts.

```sh
uv run python scripts/exp0001_decide.py --help
uv run python scripts/exp0002_decide.py --help
uv run python scripts/exp0003_decide.py --help
uv run python scripts/exp0004_decide.py --help
```

Decision scripts inspect provenance and expected run names. Keep the generated names
and do not bypass clean-run or pairing checks. Per-rule thresholds are visible in the
script docstrings and constants and are pinned by synthetic counterexample tests.

## Margin and theory probes

`exp0004_margin_prior.py` calculates prior-only far mass for a window of 512, the tested
length range, and the selected growth tolerance. `exp0004_exponents.py` evaluates learned
exponents. These calculations omit trained content scores and do not determine a
universally minimal margin. `finite_iteration_rate.py` measures finite-length amplitude
errors; it does not claim a proven rate uniform in context length.

Original aggregate results, missing-checkpoint qualifications, exact evaluation budgets,
hardware assignments and the full interpretation of each outcome are in the paper's
experimental appendix. Reproducing code execution alone does not establish reproduction
of its reported numerical results.
