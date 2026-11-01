"""Exploratory: is extrapolation decided by whether the position prior's tail is summable?

    uv run python scripts/probe_summability.py "runs/exp0002_screen_*" \
        --out runs/summability/screen.json
    uv run python scripts/probe_summability.py "runs/exp0002_confirm_*" \
        --out runs/summability/confirm.json --mults 1 2 4 8 16 --far-mults 1 2 4 8 \
        --reference-max-mult 8 --passkey

Motivation (report 0012 §8): contrary to H4's prediction, linear biases (exponential
attention decay) extrapolate at least as well as logarithmic ones, and KERPLE, whose heads
mostly learned a tail exponent > 1, beats the log-distance bias (exponents mostly < 1) at
8x. A prior (1 + d)^-c has total weight sum_d (1 + d)^-c that diverges for c <= 1: the
attention mass on far keys then grows with the context length, which a model trained at
length 512 never saw. Exponential decay and c > 1 keep it bounded.

Predictions, written down before running (2026-10-08):

* S1  far mass (attention on keys more than 512 = the training length back, averaged over
      the second half of the queries) grows with the evaluation length for the arms whose
      prior is not summable (B0, G-M: no prior; P1, P2, P6: most exponents < 1; O1) and
      stays roughly flat for the summable ones (P4 ALiBi, CABLE, G-A, FoX, G-AP, O2; P3
      mostly exponents > 1)
* S2  from 8x to 32x the validation loss keeps rising for the non-summable arms and stays
      roughly flat for the summable ones
* S3  within the logarithmic family, the share of heads (P6, LC: of head x query pairs)
      with an effective exponent > 1 goes with the long-length loss: P3 > LC ? P1 > P6
      ("?" : LC's base exponents are < 1, but its gated, per-query exponents are unknown)

Predictions for the 100M confirmation checkpoints (training length 1024), written down
before running (2026-10-13; report 0014 §8.4):

* S1' as S1 at 100M: far mass on natural text grows from 2x to 8x for B0, G-M and log-n,
      and stays flat or falls for P4, G-AP, O2 and both CABLE arms; LC lies in between
* S4  content clock: on passkey inputs (repetitive filler, key at depth 0.1), CABLE's far
      mass is much higher than on natural text of the same length, while P4's and O2's is
      not; and the attention from the last prompt position to the key span at 4x is higher
      for CABLE (lr 7.5e-4 above all) than for P4 and O2

For every run directory (config.yaml + last.pt): validation loss at each length multiple
(``--tokens`` predictions; FlexAttention where the run's config allows it), far mass at
each length (reference attention, a forward hook on every AttentionProbe), and for the
log-family encodings the exponent distribution. With ``--passkey`` also the far mass on
passkey inputs and the attention on the key span. Inference only, bf16 autocast.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import tiktoken
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geoattn_core import AttentionProbe
from geoattn_core.pe import GatedLogDistanceBias, KerpleLogBias, LogDistanceBias
from geoattn_train.config import load_config
from geoattn_train.lm_data import load_corpus, val_batches
from geoattn_train.lm_trainer import build_lm
from geoattn_train.retrieval import passkey
from geoattn_train.vram import cap_process_memory

# Allocator cap: past it an allocation raises OutOfMemoryError (recorded as a missing value)
# instead of spilling into system RAM, which made a checkpoint take 30 minutes (2026-10-09).
VRAM_LIMIT_GB = 40.0

MULTS = (1, 2, 4, 8, 16, 32)
# Far mass needs the reference attention (the full coupling): at 32x (16384) the 4090 ran out
# of memory and spilled to system RAM (2026-10-09), so it stops at 16x. The first 17 runs also
# recorded 32x; the analysis uses 1x-16x for every run.
FAR_MULTS = (1, 2, 4, 8, 16)
ROW_CHUNK = 1024
PASSKEY_DEPTH = 0.1  # the far end from the query (report 0014 §8.4)
PASSKEY_KEYS = (48213, 90517)


@torch.no_grad()
def mean_loss(model: nn.Module, val, length: int, tokens: int, device) -> float:
    total, count = 0.0, 0
    for x, y in val_batches(val, length, tokens, max(1, 16384 // length)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.to(device))
        total += F.cross_entropy(
            logits.float().flatten(0, 1), y.to(device).flatten(), reduction="sum"
        ).item()
        count += y.numel()
    return total / count


def _hooked(model: nn.Module, hook) -> list:
    # Compiled blocks (compile_model) do not guard on hooks added later: a graph compiled by a
    # FlexAttention loss pass at the same input shape would be reused and no hook would fire.
    # Recompile with the hooks on, and again without them once removed (_unhook).
    torch._dynamo.reset()
    return [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, AttentionProbe)]


def _unhook(handles: list) -> None:
    for h in handles:
        h.remove()
    torch._dynamo.reset()


@torch.no_grad()
def _far_mass_on(
    model: nn.Module, inputs: Iterable[Tensor], length: int, train_len: int, device
) -> float:
    """Mean attention mass on keys more than ``train_len`` back, over the second half of the
    queries, all layers and heads, for the given (1, length) inputs."""
    sums: list[float] = []
    idx = torch.arange(length, device=device)
    far = (idx[:, None] - idx[None, :]) > train_len  # (N, N) keys more than train_len back
    rows = slice(length // 2, length)

    def hook(_m: nn.Module, _i: tuple, p: Tensor) -> None:
        total, count = 0.0, 0
        for start in range(rows.start, rows.stop, ROW_CHUNK):  # no full fp32 copy of P
            r = slice(start, min(start + ROW_CHUNK, rows.stop))
            mass = (p[..., r, :].float() * far[r]).sum(-1)
            total, count = total + mass.sum().item(), count + mass.numel()
        sums.append(total / count)

    handles = _hooked(model, hook)
    try:
        for x in inputs:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x.to(device))
    finally:
        _unhook(handles)
    if not sums:
        raise RuntimeError("no AttentionProbe hook fired: the coupling was never formed")
    return sum(sums) / len(sums)


def far_mass(model: nn.Module, val, length: int, train_len: int, device, seqs: int = 2) -> float:
    """Far mass on natural text (the validation stream); 0 when no query has far keys."""
    if length <= train_len:
        return 0.0
    inputs = (x for x, _ in val_batches(val, length, seqs * length, 1))
    return _far_mass_on(model, inputs, length, train_len, device)


def _passkey_inputs(length: int) -> list[tuple[Tensor, int, int, int]]:
    """(tokens, needle start, needle length, last prompt position) per key."""
    enc = tiktoken.get_encoding("gpt2")
    out = []
    for key in PASSKEY_KEYS:
        s = passkey(enc.encode_ordinary, length, PASSKEY_DEPTH, key)
        needle_len = len(enc.encode_ordinary(f" The pass key is {key}."))
        last_prompt = len(s.tokens) - s.answer_len - 1
        out.append((torch.tensor([s.tokens]), s.needle_pos, needle_len, last_prompt))
    return out


def far_mass_passkey(model: nn.Module, length: int, train_len: int, device) -> float:
    """Far mass on passkey inputs (repetitive filler, key at depth 0.1)."""
    if length <= train_len:
        return 0.0
    inputs = [x for x, *_ in _passkey_inputs(length)]
    return _far_mass_on(model, inputs, length, train_len, device)


@torch.no_grad()
def key_attention(model: nn.Module, length: int, device) -> float:
    """Attention from the last prompt position to the span that states the key, averaged
    over layers, heads and keys: how much the query reads the far-away key."""
    values: list[float] = []
    for x, start, span, row in _passkey_inputs(length):

        def hook(_m: nn.Module, _i: tuple, p: Tensor, s=start, n=span, r=row) -> None:
            values.append(p[..., r, s : s + n].float().sum(-1).mean().item())

        handles = _hooked(model, hook)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x.to(device))
        finally:
            _unhook(handles)
    return sum(values) / len(values)


@torch.no_grad()
def key_attention_heads(model: nn.Module, length: int, device) -> list[list[float]]:
    """``key_attention`` per layer (rows, in forward order) and head (columns), averaged over
    keys (experiment 0003, gate G4). Its mean over layers and heads is ``key_attention``."""
    per_key: list[Tensor] = []
    for x, start, span, row in _passkey_inputs(length):
        layers: list[Tensor] = []

        def hook(_m: nn.Module, _i: tuple, p: Tensor, s=start, n=span, r=row, out=layers) -> None:
            out.append(p[..., r, s : s + n].float().sum(-1).mean(0).cpu())  # (H,)

        handles = _hooked(model, hook)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x.to(device))
        finally:
            _unhook(handles)
        per_key.append(torch.stack(layers))
    return torch.stack(per_key).mean(0).tolist()


def _collect_coef(bias: GatedLogDistanceBias, values: list[Tensor]):
    def hook(_m: nn.Module, args: tuple) -> None:
        values.append(bias.coef(args[0]).flatten().float())  # per query: from the layer input

    return hook


@torch.no_grad()
def exponents(model: nn.Module, val, length: int, device) -> dict[str, float] | None:
    """Distribution of the tail exponents of a logarithmic-family bias, or None."""
    values: list[Tensor] = []
    hooks = []
    for m in model.modules():
        pe = getattr(m, "pe", None)
        add = getattr(pe, "additive", pe)
        if isinstance(add, GatedLogDistanceBias):  # P6, LC
            hooks.append(m.register_forward_pre_hook(_collect_coef(add, values)))
        elif isinstance(add, LogDistanceBias):  # P1, P2, P5
            values.append(add.coef.detach().float())
        elif isinstance(add, KerpleLogBias):  # P3: -p log(1 + a d) ~ -p log d far away
            values.append(add.p.detach().clamp_min(add.eps).float())
    if hooks:
        x, _ = next(iter(val_batches(val, length, 4 * length, 4)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x.to(device))
        for h in hooks:
            h.remove()
    if not values:
        return None
    c = torch.cat([v.to("cpu") for v in values])
    q = torch.quantile(c, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "p10": q[0].item(),
        "median": q[1].item(),
        "p90": q[2].item(),
        "share_gt1": (c > 1).float().mean().item(),
    }


def _or_none(fn, *args) -> float | None:
    """``fn(*args)``, or None when it does not fit in GPU memory: the reference attention
    of the OT arms at 32x (16384) needs several 8 GB couplings (2026-10-09)."""
    try:
        return fn(*args)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def probe_run(run_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device("cuda")
    model = build_lm(cfg).to(device).eval()
    model.load_state_dict(
        torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)["model"]
    )
    val = load_corpus(cfg.data, cfg.lm, cfg.model).val
    n = cfg.lm.seq_len
    if args.key_heads:  # experiment 0003 C: the per-head table with the mean it must match
        return {
            name: {str(m): _or_none(fn, model, m * n, device) for m in args.far_mults}
            for name, fn in (("key_attention", key_attention), ("key_heads", key_attention_heads))
        }
    # Arms without FlexAttention (OT, GRAPE-AP, log-n) materialize the coupling: at 32x the OT
    # arm's compiled normalizer died with "CUDA driver error: device not ready" (2026-10-09)
    # rather than an OutOfMemoryError, so those arms stop at --reference-max-mult.
    reference_only = cfg.attention.normalizer.name != "softmax" or cfg.attention.pe == "grape_ap"
    mults = [m for m in args.mults if not (reference_only and m > args.reference_max_mult)]
    out: dict[str, object] = {
        "loss": {
            str(m): _or_none(mean_loss, model, val, m * n, args.tokens, device) for m in mults
        },
        "far_mass": {
            str(m): _or_none(far_mass, model, val, m * n, n, device) for m in args.far_mults
        },
        "exponents": exponents(model, val, n, device),
    }
    if args.passkey:
        out["far_mass_passkey"] = {
            str(m): _or_none(far_mass_passkey, model, m * n, n, device) for m in args.far_mults
        }
        out["key_attention"] = {
            str(m): _or_none(key_attention, model, m * n, device) for m in args.far_mults
        }
    print(f"  {run_dir.parent.name}: loss {out['loss']}  far {out['far_mass']}", flush=True)
    return out


def _git() -> dict[str, object]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    return {"sha": sha, "dirty": dirty}


def main() -> None:
    torch._dynamo.config.cache_size_limit = 256
    cap_process_memory(VRAM_LIMIT_GB, torch.device("cuda"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("patterns", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=262_144, help="predictions per length")
    p.add_argument("--mults", type=int, nargs="+", default=list(MULTS))
    p.add_argument("--far-mults", type=int, nargs="+", default=list(FAR_MULTS))
    p.add_argument("--reference-max-mult", type=int, default=16)
    p.add_argument("--passkey", action="store_true", help="far mass and key attention on passkey")
    p.add_argument(
        "--key-heads", action="store_true", help="only key attention, also per layer and head"
    )
    args = p.parse_args()
    runs = sorted(
        Path(r) for pat in args.patterns for e in glob.glob(pat) for r in glob.glob(f"{e}/*")
    )
    runs = [r for r in runs if (r / "last.pt").exists()]
    git = _git()
    # Resumable: runs already in --out are kept; every entry records the commit it came from.
    result: dict = {"tokens": args.tokens, "runs": {}}
    if args.out.exists():
        result = json.loads(args.out.read_text(encoding="utf-8"))
        for entry in result["runs"].values():
            entry.setdefault("git", result.get("git"))  # files written before per-run git
    for run_dir in runs:
        key = f"{run_dir.parent.name}/{run_dir.name}"
        if key in result["runs"]:
            continue
        try:
            entry = probe_run(run_dir, args)
        except RuntimeError as exc:  # e.g. a checkpoint that does not load (report 0014 §9)
            print(f"  {key}: skipped ({type(exc).__name__}: {str(exc)[:80]})", flush=True)
            continue
        result["runs"][key] = {**entry, "git": git}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
