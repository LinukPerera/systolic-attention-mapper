"""
Dedicated comparison figure: best_fixed_per_stage utilization vs. array size,
one line per run (original head_dim=16, representative head_dim=32, validation
head_dim=64) — shows the knee visibly shifting right as head_dim doubles,
directly supporting exp1_stats.md §8's causal (head_dim-tracks-knee) explanation.
Separate from plot_utilization_vs_array_size (which compares methods within one
run) since this compares *runs* for one method — a different axis of comparison,
not a variant of the same figure. Per CODING_AGENT_PLAN.md §5, independently
regenerable from results/processed/ without re-running any experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from common.plotting.style import apply_style, save_figure

THIS_DIR = Path(__file__).parent
PROCESSED_DIR = THIS_DIR / "results" / "processed"
FIGURES_DIR = THIS_DIR / "results" / "figures"

# (label for legend, processed-summary filename, head_dim of that run's workload)
RUNS = [
    ("original (head_dim=16)", "exp1_summary.csv", 16),
    ("representative (head_dim=32)", "exp1_representative_summary.csv", 32),
    ("head64_validation (head_dim=64)", "exp1_head64_validation_summary.csv", 64),
]
# Distinct from DATAFLOW_COLORS/OPTIMIZER_COLORS (this compares runs, not
# dataflows/optimizers) — a separate small sequential palette, light to dark,
# so "bigger head_dim" reads as "darker line".
RUN_COLORS = ["#9ecae1", "#3182bd", "#08306b"]


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (label, filename, head_dim), color in zip(RUNS, RUN_COLORS):
        df = pd.read_csv(PROCESSED_DIR / filename)
        bfps = df[df["method"] == "best_fixed_per_stage"].sort_values("array_size")
        ax.plot(bfps["array_size"], bfps["pe_utilization_pct"], label=label, color=color, marker="o", linewidth=2)
        ax.axvline(head_dim, color=color, linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Array size (N×N)")
    ax.set_ylabel("best_fixed_per_stage PE utilization (%)")
    ax.set_title("Knee location tracks head_dim (dashed lines mark each run's head_dim)")
    ax.legend()

    out_path = FIGURES_DIR / "exp1_knee_vs_head_dim_comparison"
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"[plot_knee_comparison] wrote {out_path}.png/.svg")


if __name__ == "__main__":
    main()
