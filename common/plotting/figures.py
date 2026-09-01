"""
plotting/figures.py — one named, independently-callable function per figure type.

Each function takes a processed-results DataFrame (or its CSV path) and an output
path, and writes both PNG and SVG (plan §5). Keeping each figure as its own function
means a figure can be regenerated after a data fix without re-running the experiment
that produced the underlying data.

STATUS: plot_convergence_curves is real (implemented 2026-08-31 for Experiment 0).
The rest remain stubs — implement each as the experiment that needs it is reached.
Every plot that aggregates across seeds MUST include error bars (std or 95% CI) —
plan §5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from common.plotting.style import DATAFLOW_COLORS, OPTIMIZER_COLORS, apply_style, save_figure


def plot_convergence_curves(df: pd.DataFrame, out_path: Path, *, value_col: str = "value", group_col: str = "group") -> None:
    """Multi-seed convergence-curve overlay: mean trajectory + 95% CI band per group, shared axes.

    Used by Experiment 0 (one line per optimizer) and Experiment 2 (one line — the
    selected optimizer — with individual seed traces also relevant there).

    Args:
        df: Long-format DataFrame with columns [group_col, "seed", "iteration", value_col].
            One row per (group, seed, iteration) — typically the running-best value
            up to that iteration, computed by the caller (this function only plots
            and aggregates across seeds, per plan §3 "one function = one responsibility").
        out_path: Base output path; writes `<out_path>.png` and `<out_path>.svg`.
        value_col: Column to plot on the y-axis (e.g. "best_reward_so_far").
        group_col: Column identifying each line (e.g. "optimizer").
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for group_name, group_df in df.groupby(group_col):
        color = OPTIMIZER_COLORS.get(group_name, DATAFLOW_COLORS.get(group_name))
        agg = group_df.groupby("iteration")[value_col].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"] / agg["count"].clip(lower=1) ** 0.5
        # t-based 95% CI per iteration; NaN (n=1) collapses the band to the mean line.
        t_crit = agg["count"].apply(lambda n: stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan)
        ci = t_crit * agg["sem"]

        ax.plot(agg["iteration"], agg["mean"], label=str(group_name), color=color, linewidth=2)
        ax.fill_between(
            agg["iteration"], agg["mean"] - ci.fillna(0), agg["mean"] + ci.fillna(0),
            color=color, alpha=0.2, linewidth=0,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(value_col.replace("_", " "))
    ax.legend()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_utilization_vs_array_size(df: pd.DataFrame, out_path: Path) -> None:
    """Line plot of utilization vs. array size, one line per method (Experiment 1).

    Args:
        df: Columns ["array_size", "method", "pe_utilization_pct"] — one row per
            (array_size, method); array_size is a single int (e.g. 8 for "8x8",
            assumes square arrays, true for every config/hardware/*.yaml so far).
        out_path: Base output path; writes `<out_path>.png` and `<out_path>.svg`.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method_name, method_df in df.groupby("method"):
        method_df = method_df.sort_values("array_size")
        color = DATAFLOW_COLORS.get(method_name.upper() if isinstance(method_name, str) else method_name)
        ax.plot(
            method_df["array_size"], method_df["pe_utilization_pct"],
            label=str(method_name), color=color, marker="o", linewidth=2,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["array_size"].unique()))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Array size (N×N)")
    ax.set_ylabel("PE utilization (%)")
    ax.legend()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_utilization_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of utilization by workload x array x method (Experiment 3).

    Args:
        df: Columns ["cell", "method", "pe_utilization_pct"] — one row per
            (cell, method); `cell` is a pre-formatted row label combining
            workload and array (e.g. "seq128 @ 16x16"), since a heatmap needs a
            2D grid and (workload, array) is naturally a compound row dimension.
        out_path: Base output path; writes `<out_path>.png` and `<out_path>.svg`.
    """
    import matplotlib.pyplot as plt

    apply_style()
    pivot = df.pivot(index="cell", columns="method", values="pe_utilization_pct")
    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 2, 0.6 * len(pivot.index) + 1.5))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.to_numpy()[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color="white" if val < 50 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="PE utilization (%)")
    save_figure(fig, out_path)
    plt.close(fig)


def plot_pct_improvement_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of %-improvement over best baseline per workload (Experiment 3).

    Args:
        df: Columns ["cell", "pct_improvement_mean", "pct_improvement_ci"] — one
            row per (workload, array) cell; `cell` a pre-formatted label as in
            plot_utilization_heatmap. Error bars (95% CI) are mandatory here since
            this aggregates the optimizer across seeds (plan §5).
        out_path: Base output path; writes `<out_path>.png` and `<out_path>.svg`.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df)), 4.5))
    colors = [OPTIMIZER_COLORS["bayesian_opt"] if v >= 0 else DATAFLOW_COLORS["WS"] for v in df["pct_improvement_mean"]]
    ax.bar(df["cell"], df["pct_improvement_mean"], yerr=df["pct_improvement_ci"], color=colors, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("% improvement over best static baseline")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_figure(fig, out_path)
    plt.close(fig)


def plot_scalability_panels(df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel line plot: (a) %-improvement vs. scale, (b) optimizer cost vs. scale (Experiment 4).

    Args:
        df: Columns ["seq_len", "array_size", "pct_improvement_mean", "pct_improvement_ci",
            "cost_mean", "cost_ci"] — one row per (seq_len, array_size) point.
            `cost_mean`/`cost_ci` are whatever cost unit the caller chose (e.g.
            convergence iteration, or wall-clock seconds) — labeled via `cost_label`.
        out_path: Base output path; writes `<out_path>.png` and `<out_path>.svg`.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for array_size, g in df.groupby("array_size"):
        g = g.sort_values("seq_len")
        color = DATAFLOW_COLORS.get(str(array_size))  # falls back to matplotlib default cycling if unmapped
        ax1.errorbar(g["seq_len"], g["pct_improvement_mean"], yerr=g["pct_improvement_ci"].fillna(0),
                     label=f"{array_size}x{array_size}", color=color, marker="o", capsize=4, linewidth=2)
        ax2.errorbar(g["seq_len"], g["cost_mean"], yerr=g["cost_ci"].fillna(0),
                     label=f"{array_size}x{array_size}", color=color, marker="o", capsize=4, linewidth=2)

    ax1.set_xlabel("seq_len")
    ax1.set_ylabel("% improvement over best-single-dataflow baseline")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.legend()

    ax2.set_xlabel("seq_len")
    ax2.set_ylabel("optimizer cost")
    ax2.legend()

    save_figure(fig, out_path)
    plt.close(fig)


def plot_ablation_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart comparing dataflow-only / tiling-only / joint variants per metric, with error bars (Experiment 5).

    Args:
        df: Columns ["cell", "variant", "value_mean", "value_ci"] — one row per
            (cell, variant); `cell` a pre-formatted label (as in
            plot_utilization_heatmap), `variant` one of
            "dataflow_only"/"tiling_only"/"joint".
        out_path: Base output path; writes `<out_path>.png` and `<out_path>.svg`.
    """
    import matplotlib.pyplot as plt

    apply_style()
    variants = ["dataflow_only", "tiling_only", "joint"]
    variant_colors = {"dataflow_only": DATAFLOW_COLORS["OS"], "tiling_only": DATAFLOW_COLORS["WS"],
                       "joint": OPTIMIZER_COLORS["bayesian_opt"]}
    cells = list(df["cell"].unique())
    x = np.arange(len(cells))
    width = 0.8 / len(variants)

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(cells)), 4.5))
    for i, variant in enumerate(variants):
        sub = df[df["variant"] == variant].set_index("cell").reindex(cells)
        ax.bar(x + i * width, sub["value_mean"], width, yerr=sub["value_ci"].fillna(0),
               label=variant, color=variant_colors[variant], capsize=4)

    # Zoomed y-axis: these variants are typically within a fraction of a percentage
    # point of each other (the ablation's whole point is to isolate a small effect),
    # so a 0-100% axis would render every bar as visually identical. Explicitly
    # labeled as zoomed (not silently misleading) since it does not start at 0 —
    # exact values belong in the accompanying table, this figure is for relative
    # comparison only.
    data_min = (df["value_mean"] - df["value_ci"].fillna(0)).min()
    data_max = (df["value_mean"] + df["value_ci"].fillna(0)).max()
    pad = max((data_max - data_min) * 0.5, 0.5)
    ax.set_ylim(max(0, data_min - pad), data_max + pad)
    ax.text(0.01, 0.99, "y-axis zoomed to the data range (not from 0) — exact values in the accompanying table",
            transform=ax.transAxes, fontsize=8, va="top", ha="left", style="italic", alpha=0.7)

    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(cells)
    ax.set_ylabel("PE utilization (%)")
    ax.legend()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_generalization_comparison(df: Any, out_path: Path) -> None:
    """Bar/line chart comparing in-distribution vs. held-out performance relative to baseline (Experiment 6)."""
    raise NotImplementedError
