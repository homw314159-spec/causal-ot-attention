"""Figures of the paper (paper/figures/*.pdf) from the evaluation files and the theory.

    uv run python scripts/paper_figures.py                 # all figures
    uv run python scripts/paper_figures.py farmass boundary  # some of them

Inputs under ``--runs`` (default ``runs``), the same files as the report tables:
``summability/screen.json`` and ``tail/{screen,confirm}.json``. ``prior`` and ``boundary``
need no runs: they evaluate the operator's fixed point and the power prior directly.
The PDFs are committed, because the paper is built in CI without access to ``runs/``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from geoattn_core import UnbalancedSinkhorn
from geoattn_core.pe.alibi import alibi_slopes

OUT = Path("paper/figures")
RUN = re.compile(r"exp000[23]_(?:screen|confirm|a1|a2|b1)_(?P<arm>[a-z0-9_]+?)_s(?P<seed>\d)$")
WIDTH = 6.5  # \textwidth of the article, inches

# Colour means the case of Proposition prop:arch in every experimental figure, and nothing
# else; within a case, line style and marker tell the arms apart. P2's exponents are fixed at
# or below one, so it dilutes.
DILUTES, LEARNED, BOUNDED = "#D55E00", "#8B5AA5", "#0072B2"
REFERENCE = "#333333"
CLASSES = {
    "dilutes for all parameters": (DILUTES, ["b0", "gm", "o1", "p2"]),
    "decided by learned parameters": (LEARNED, ["p1", "p5", "p6", "p3", "lc", "cable"]),
    "bounded for all parameters": (BOUNDED, ["p4", "ga", "fox", "gap", "o2"]),
}
NAMES = {
    "b0": "RoPE", "gm": "GRAPE-M", "o1": "OT on RoPE", "p2": "log bias, fixed $c$",
    "p1": "log bias, learned $c$", "p5": "log bias, no RoPE", "p6": "log bias, gated $c$",
    "p3": "KERPLE", "lc": "log bias on a clock", "cable": "CABLE",
    "p4": "ALiBi", "ga": "GRAPE-A", "fox": "FoX", "gap": "GRAPE-AP",
    "o2": "OT on ALiBi",
}  # fmt: skip
STYLES = [("-", "o"), ("--", "s"), (":", "D"), ("-.", "^"), ((0, (5, 1, 1, 1)), "v"),
          ((0, (1, 1)), "x")]  # fmt: skip


def color(arm: str) -> str:
    return next(c for c, arms in CLASSES.values() if arm in arms)


def setup() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })  # fmt: skip


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def by_arm(entries: dict[str, dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for key, value in entries.items():
        m = RUN.match(key.split("/")[0])
        if m:
            out.setdefault(m["arm"], []).append(value)
    return out


def curve(runs: list[dict], key: str, mults: list[str]) -> tuple[list[int], list[float]]:
    """Mean over seeds at each multiple where every seed has a value."""
    xs, ys = [], []
    for m in mults:
        vals = [r[key].get(m) for r in runs]
        if vals and all(v is not None for v in vals):
            xs.append(int(m))
            ys.append(statistics.fmean(vals))
    return xs, ys


def log2_axis(ax: Axes, mults: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(mults, [f"{m}×" for m in mults])
    ax.minorticks_off()


# ---------------------------------------------------------------- Figure: far mass

# The probe's far mass stops at 16x (probe_summability.py); at 1x it is zero by definition.
MULTS = ["1", "2", "4", "8", "16"]


def fig_farmass(root: Path) -> None:
    """Far mass (top) and loss (bottom) at 25M, one column per case, every curve unclipped."""
    screen = by_arm(load(root / "summability" / "screen.json")["runs"])
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 2.8), sharex=True, sharey="row")
    for col, (case, (c, arms)) in enumerate(CLASSES.items()):
        for arm, (ls, marker) in zip(arms, STYLES, strict=False):
            kw = {"color": color(arm), "ls": ls, "marker": marker, "ms": 3, "mfc": "none",
                  "mew": 0.8, "label": NAMES[arm]}  # fmt: skip
            axes[0, col].plot(*curve(screen[arm], "far_mass", MULTS), **kw)
            axes[1, col].plot(*curve(screen[arm], "loss", MULTS), **kw)
        # the loss row is empty at the upper left in every case; the far-mass row is not
        axes[1, col].legend(frameon=False, fontsize=6, loc="upper left")
        axes[0, col].set_title(case, color=c)
    for ax in axes.flat:
        log2_axis(ax, [int(m) for m in MULTS])
    for ax in axes[1]:
        ax.set_xlabel("evaluation length / training length")
    axes[0, 0].set(ylabel="far mass", ylim=(0, 1))
    axes[1, 0].set(ylabel="validation loss (nats)", ylim=(3.25, 5.9))  # headroom for legends
    fig.tight_layout()
    save(fig, "farmass")


# ---------------------------------------------------------------- Figure: interventions

# Each intervention is coloured by the case of the arms it is applied to: caps to bounded
# arms, repairs to log biases decided by learned exponents, windows to RoPE, which dilutes.
MODES = {"cap": BOUNDED, "repair": LEARNED, "window": DILUTES}
EXAMPLES = (
    ("exp0002_screen_p4_s0", "cap", "ALiBi, cap"),
    ("exp0002_screen_p1_s0", "repair", "log bias (learned $c$), repair"),
    ("exp0002_screen_b0_s0", "window", "RoPE, window"),
)


def fig_interventions(root: Path) -> None:
    """Loss change at the longest length for every checkpoint, one panel per intervention."""
    stages = {"25M": load(root / "tail" / "screen.json")["runs"],
              "100M": load(root / "tail" / "confirm.json")["runs"]}  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.6))
    for ax, (mode, c) in zip(axes, MODES.items(), strict=True):
        for y, runs in enumerate(stages.values()):
            # "loss" is at the decisive length: 16x at 25M, 8x at 100M
            changes = sorted(r["loss"][mode] - r["loss"]["none"] for r in runs.values()
                             if r["mode"] == mode)  # fmt: skip
            ys = y + np.linspace(-0.2, 0.2, len(changes)) if len(changes) > 1 else [y]
            ax.scatter(changes, ys, s=12, color=c, marker="os"[y], linewidths=0)
            mean = statistics.fmean(changes)
            ax.plot([mean, mean], [y - 0.3, y + 0.3], color=REFERENCE, lw=1.2)
            ax.text(1.0, y, f" n={len(changes)}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=6)  # fmt: skip
        ax.axvline(0, color=REFERENCE, lw=0.6, ls="--")
        ax.set(title=mode, ylim=(-0.6, 1.6), xlabel="loss change (nats)")
        ax.title.set_color(c)
        ax.set_yticks([0, 1], list(stages))
    fig.tight_layout(w_pad=2.5)
    save(fig, "interventions")


def fig_intervention_examples(root: Path) -> None:
    """Loss against the evaluation length before and after, one 25M checkpoint each."""
    screen = load(root / "tail" / "screen.json")["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.6))
    for ax, (name, mode, title) in zip(axes, EXAMPLES, strict=True):
        curves = screen[name]["curves"]
        for variant, c in (("none", REFERENCE), (mode, MODES[mode])):
            loss = curves[variant]["loss"]
            ms = sorted(loss, key=int)
            ax.plot([int(m) for m in ms], [loss[m] for m in ms], marker="o", ms=2.5, color=c,
                    label="trained" if variant == "none" else mode)  # fmt: skip
        ax.axvline(1, color="grey", lw=0.6, ls=":")
        log2_axis(ax, [1, 4, 16])
        ax.set_title(title)
        ax.set_xlabel("eval length / train length")
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("validation loss (nats)")
    fig.tight_layout()
    save(fig, "intervention_examples")


# ---------------------------------------------------------------- Figure: boundary


def _zeta(c: float, terms: int = 1000) -> float:
    k = np.arange(1, terms, dtype=np.float64)
    n = float(terms)
    return float((k**-c).sum() + n ** (1 - c) / (c - 1) + n**-c / 2 + c * n ** (-c - 1) / 12)


def _partial_zeta(c: float, t: np.ndarray) -> np.ndarray:
    """sum_{k=1}^t k^-c by Euler-Maclaurin (exact enough for t >= 1000)."""
    if c == 1.0:
        return np.log(t) + 0.5772156649015329 + 1 / (2 * t)
    tail = t ** (1 - c) / (c - 1) - t**-c / 2 + c * t ** (-c - 1) / 12
    return _zeta(c) - tail


def fig_boundary(_: Path) -> None:
    """Corollary cor:boundary: far mass of (1 + d)^-c at row t, for exponents near one."""
    window = 512
    ratio = np.logspace(0.3, 6, 200)
    t = ratio * window
    fig, ax = plt.subplots(figsize=(WIDTH * 0.75, 2.3))
    s_l = {c: float(((1 + np.arange(window)) ** -c).sum()) for c in (1.0, 1.001, 1.05, 1.2, 1.5)}
    # c is ordered, so one hue from dark (c = 1) to light; c = 1.001 is dashed on top of c = 1
    shades = plt.get_cmap("Purples")(np.linspace(1.0, 0.45, len(s_l) - 1))
    for (c, s), shade in zip(s_l.items(), [shades[0], *shades], strict=True):
        far = 1 - s / _partial_zeta(c, t)
        ax.plot(ratio, far, color=shade, ls="--" if c == 1.001 else "-", label=f"$c={c:g}$")
    # prior e^(-m d), d < t: the geometric sums give (e^(-mL) - e^(-mt)) / (1 - e^(-mt)) per head
    m = alibi_slopes(8).double().numpy()[:, None]
    alibi = (np.exp(-m * window) * -np.expm1(-m * (t - window)) / -np.expm1(-m * t)).mean(0)
    ax.plot(ratio, alibi, color=REFERENCE, ls=":", label="ALiBi (mean of 8 heads)")
    ax.axvspan(1, 32, color="grey", alpha=0.07, lw=0)
    ax.text(5.6, 0.92, "lengths used", ha="center", fontsize=6)
    ax.annotate("$c=1$ and $c=1.001$ nearly overlap at this scale", (3e4, 0.56),
                xytext=(60, 0.8), fontsize=6,
                arrowprops={"arrowstyle": "-", "lw": 0.5})  # fmt: skip
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set(xlabel="context length / window $L$ ($L=512$)", ylabel="mass beyond $L$ at row $t$",
           ylim=(-0.02, 1))  # fmt: skip
    ax.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    save(fig, "boundary")


# ---------------------------------------------------------------- Figure: prior


def _z_profile(rho: float, depth: int) -> np.ndarray:
    gamma, z, prev = rho / (1 - rho), [], 0.0
    for _ in range(depth + 1):
        lo, hi = prev, prev + 2.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mid**gamma * (mid - prev) < 1 else (lo, mid)
        z.append(lo)
        prev = lo
    return np.diff(np.concatenate([[0.0], z]))


PRIOR_N = 2048
PRIOR_COLORS = {0.25: "#1b9e77", 0.5: "#d95f02", 0.75: "#7570b3"}


def _fixed_point_row(rho: float, n: int) -> np.ndarray:
    """Last row of the constant-score fixed point; entry j-1 is P_{n,j}."""
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    p = UnbalancedSinkhorn(rho, n_iters=60)(torch.zeros(n, n, dtype=torch.float64), causal)
    return p[-1].numpy()


def _prior_shape(ax: Axes, rows: dict[float, np.ndarray]) -> None:
    """Left panel: the limit shape f(s) on top of a sparse sample of the fixed point."""
    n = PRIOR_N
    s = np.linspace(0.005, 0.995, 400)
    j = np.arange(1, n + 1)[49::100]  # about 20 markers per curve; the curves carry the rest
    for rho, row in rows.items():
        color = PRIOR_COLORS[rho]
        ax.plot(j / n, n * row[j - 1], "o", ms=3.2, mfc="none", mew=0.8, color=color,
                alpha=0.8, zorder=2)  # fmt: skip
        ax.plot(s, np.log(1 / s) ** -rho / math.gamma(1 - rho), color=color, lw=1.5,
                zorder=3, label=f"$\\rho={rho}$")  # fmt: skip
    ax.set(xlabel="$s=j/t$", ylabel="$t\\,P_{t,j}$", xlim=(0, 1), ylim=(0, 3),
           title="limit shape $f(s)$")  # fmt: skip
    first = ax.legend(frameon=False, loc="upper left")
    ax.add_artist(first)
    ax.legend(handles=[
        Line2D([], [], color="0.3", ls="none", marker="o", ms=3.2, mfc="none", mew=0.8,
               label=f"fixed point, $t={n}$"),
        Line2D([], [], color="0.3", lw=1.5, label="$f(s)$ (Thm. 7)"),
    ], frameon=False, loc="center left", fontsize=6)  # fmt: skip


def _prior_near(ax: Axes, row: np.ndarray, rho: float) -> None:
    """Right panel: the three scales of Theorem near, compensated by the power law.

    Multiplying by Gamma(1-rho) d^rho makes the power law the line y = 1, so the discrete
    profile (fixed d) and the limit shape f (d ~ t) show up as departures from it.
    """
    n, g = PRIOR_N, math.gamma(1 - rho)
    a_star = ((1 + rho / (1 - rho)) ** rho / g) ** (1 / (1 - rho))
    d = np.unique(np.round(np.geomspace(1, n - 1, 20)).astype(int))
    ax.semilogx(d, g * d**rho * n ** (1 - rho) * row[n - 1 - d], "o", ms=3.2, mfc="none",
                mew=0.8, color="#2166ac", zorder=4, label=f"fixed point, $t={n}$")  # fmt: skip
    depth = 1000
    dd = np.arange(1, depth + 1)
    prof = a_star ** (1 - rho) * _z_profile(rho, depth)[1:]
    ax.semilogx(dd, g * dd**rho * prof, color="black", lw=1.3, zorder=3,
                label="discrete profile, fixed $d$")  # fmt: skip
    ax.axhline(1.0, color="#b2182b", lw=1.3, ls="--", zorder=2,
               label="power law, $1\\ll d\\ll t$")  # fmt: skip
    x = np.geomspace(1, n - 1, 400) / n
    ax.semilogx(x * n, x**rho * np.log(1 / (1 - x)) ** -rho, color="#d95f02", lw=1.3,
                ls=":", zorder=2, label="limit shape $f$, $d\\sim t$")  # fmt: skip
    ylabel = "$\\Gamma(1-\\rho)\\,d^{\\rho}\\,t^{1-\\rho}P_{t,t-d}$"
    ax.set(xlabel="distance $d=t-j$", ylabel=ylabel, ylim=(0.55, 1.1),
           title=f"near the diagonal, $\\rho={rho}$")  # fmt: skip
    ax.legend(frameon=False, fontsize=6, loc="lower left")


def fig_prior(_: Path) -> None:
    """Theorems shape and near: the constant-score fixed point against its limits."""
    rows = {rho: _fixed_point_row(rho, PRIOR_N) for rho in PRIOR_COLORS}
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    _prior_shape(a, rows)
    _prior_near(b, rows[0.5], 0.5)
    fig.tight_layout()
    save(fig, "prior")


FIGURES: dict[str, Callable[[Path], None]] = {
    "farmass": fig_farmass, "interventions": fig_interventions,
    "intervention_examples": fig_intervention_examples, "boundary": fig_boundary,
    "prior": fig_prior,
}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("figures", nargs="*", help=f"any of {', '.join(FIGURES)} (default: all)")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    args = p.parse_args()
    unknown = [f for f in args.figures if f not in FIGURES]
    if unknown:
        p.error(f"unknown figures {unknown}")
    setup()
    for name in args.figures or FIGURES:
        FIGURES[name](args.runs)


if __name__ == "__main__":
    main()
