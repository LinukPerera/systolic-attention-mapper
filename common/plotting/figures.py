"""
plotting/figures.py — one named, independently-callable function per figure type.

Each function takes a processed-results DataFrame (or its CSV path) and an output
path, and writes both PNG and SVG (plan §5). Keeping each figure as its own function
means a figure can be regenerated after a data fix without re-running the experiment
that produced the underlying data.

STATUS: stub — implement one function per figure required by CODING_AGENT_PLAN.md §6
as each experiment reaches the plotting stage. Every plot that aggregates across
seeds MUST include error bars (std or 95% CI) — plan §5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_convergence_curves(df: Any, out_path: Path) -> None:
    """Multi-seed convergence-curve overlay: mean trajectory + CI band, all seeds/optimizers on shared axes.

    Used by Experiment 0 (optimizer overlay) and Experiment 2 (per-seed traces + mean).

    Args:
        df: Processed results with columns including iteration, reward/utilization, seed, (optimizer).
        out_path: Base output path; writes `<out_path>.png` and `<out_path>.svg`.
    """
    raise NotImplementedError


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
