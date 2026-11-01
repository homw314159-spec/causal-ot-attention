# Attribution and external materials

The MIT license covers the project code and documentation distributed in this source
release. External datasets, dependencies and reference repositories retain their own
licenses. No third-party source repository, training corpus or model checkpoint is
vendored here.

Relevant method references include:

- [Sinkformers](https://proceedings.mlr.press/v151/sander22a.html): doubly stochastic attention.
- [Scaling algorithms for unbalanced OT](https://doi.org/10.1090/mcom/3303): damped marginal scaling.
- [Causal Prefix-Sinkhorn Attention](https://doi.org/10.13140/RG.2.2.24121.74087): prefix column balancing with a different update.
- [ALiBi](https://arxiv.org/abs/2108.12409), [RoFormer](https://arxiv.org/abs/2104.09864),
  [KERPLE](https://arxiv.org/abs/2205.09921) and [YaRN](https://arxiv.org/abs/2309.00071).
- [GRAPE](https://arxiv.org/abs/2512.07805), [Forgetting Transformer](https://arxiv.org/abs/2503.02130)
  and [CABLE](https://aclanthology.org/2025.emnlp-main.1545/).
- [Exploring Transformer Extrapolation](https://ojs.aaai.org/index.php/AAAI/article/view/29855): summability and sliding-window approximation.

FineWeb-Edu is obtained from [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
Consult its dataset card and license before use or redistribution. The preparation
script downloads data from that host and records the revision and hashes. GPT-2
tokenization is supplied by tiktoken. Python package dependency names, versions and
download locations are recorded in `pyproject.toml` and `uv.lock`.

Optional tests compare with user-supplied reference implementations. Those repositories
are not required for CPU tests of the local definitions, and absent references are
reported as skipped. If adding third-party source later, review its license and preserve
its required notices rather than treating the project's MIT license as a replacement.
