# Implementation scope

## Causal attention

The normalizer and position encoding are independent components of `GeometricAttention`.
Masks are boolean, with `True` meaning keep. Fully masked rows return zero without NaNs.
Sinkhorn operations use float32 log-domain arithmetic and finish with row normalization;
one iteration equals softmax.

Under a causal mask, an output row may depend only on the input prefix. The unbalanced
operator uses inclusive prefix column masses, not full-sequence column totals. It damps
the accumulated scaling: `log_v = rho * (log_v - log_prefix_mass)`. Damping only a
relative attention update would define a different operator. The admissible balancing
strength is `0 <= rho < 1` for the paper's fixed-point results.

`test_causal.py` includes prefix and future-perturbation checks with leaking controls.
The model-level causal tests check that the data and architecture preserve this property.
Tests of limits and bounds include counterexample controls where applicable.

## Positional encodings and evaluation

The implementation includes RoPE, ALiBi, logarithmic distance biases, KERPLE-style log
biases, GRAPE variants, FoX and CABLE-style clocks. Paper arm labels are recorded in
the YAML configurations. Trained exponent constraints and post-training tail
interventions are separate options; changing a tail can change attention entropy as
well as distant mass.

The theory characterizes attention mass and explicit limits under its stated score
assumptions. It provides no general guarantee for validation loss or distant retrieval.
The exact constant/i.i.d. prior shape should not be assumed for correlated trained scores.

## Packages and backends

Dependencies point from training to kernels/observation and then to the mathematical
core. The core imports PyTorch only. The fused kernel is an accelerated implementation
of the prefix iteration, with reference fallbacks for unsupported inputs. FlexAttention
is used only when the score modification is supported; the factory selects fallbacks.
Changing a backend can affect floating-point rounding and measured speed.

The public release exposes language-model training. The shared configuration schema
retains some legacy fields for checkpoint compatibility, but no vision model or image
dataset pipeline is distributed. Source comments may retain experiment or implementation
invariant identifiers from development; the paper is the scientific reference.
