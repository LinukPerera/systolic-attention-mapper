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


def plot_utilization_vs_array_size(df: Any, out_path: Path) -> None:
    """Line plot of utilization vs. array size, one line per dataflow mode (Experiment 1)."""
    raise NotImplementedError


def plot_utilization_heatmap(df: Any, out_path: Path) -> None:
    """Heatmap of utilization by workload x array x method (Experiment 3)."""
    raise NotImplementedError


def plot_pct_improvement_bars(df: Any, out_path: Path) -> None:
    """Grouped bar chart of %-improvement over best baseline per workload (Experiment 3)."""
    raise NotImplementedError


def plot_scalability_panels(df: Any, out_path: Path) -> None:
    """Two-panel line plot: (a) %-improvement vs. scale, (b) optimizer cost vs. scale (Experiment 4)."""
    raise NotImplementedError


def plot_ablation_bars(df: Any, out_path: Path) -> None:
    """Bar chart comparing dataflow-only / tiling-only / joint variants per metric, with error bars (Experiment 5)."""
    raise NotImplementedError


def plot_generalization_comparison(df: Any, out_path: Path) -> None:
    """Bar/line chart comparing in-distribution vs. held-out performance relative to baseline (Experiment 6)."""
    raise NotImplementedError
