"""
Analyze Experiment 4's scalability trend: merges Experiment 3's real results (the
4 already-run (seq_len, array) points) with Experiment 4's own new point
(seq_len=512 @ 32x32) — see run_exp4.py's docstring for why the 4 overlapping
points are reused rather than re-simulated. Computes, per point: %-improvement of
`bayesian_opt` over the best single-dataflow baseline (the same FAIR comparison
established in Experiment 3 — best_fixed_per_stage is deliberately excluded here
too, for the same action-space-asymmetry reason), and optimizer cost (both
convergence-iteration, where defined, and total wall-clock for the fixed budget,
which is always defined). Produces the required two-panel scalability plot.

Every number here traces back to results/raw/exp3.csv or results/raw/exp4.csv —
nothing is computed from anything else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from scipy import stats

from common.metrics import pct_diff_vs_baseline
from common.optimizers.convergence import best_reward_so_far, convergence_iteration
from common.plotting.figures import plot_scalability_panels

THIS_DIR = Path(__file__).parent
EXP3_RAW_CSV = _PROJECT_ROOT / "experiments" / "exp3_main_sweep" / "results" / "raw" / "exp3.csv"
EXP4_RAW_CSV = THIS_DIR / "results" / "raw" / "exp4.csv"
PROCESSED_DIR = THIS_DIR / "results" / "processed"
FIGURES_DIR = THIS_DIR / "results" / "figures"

SINGLE_DATAFLOW_METHODS = ["ws_only", "os_only", "is_only"]


def main() -> None:
    exp3 = pd.read_csv(EXP3_RAW_CSV)
    exp4 = pd.read_csv(EXP4_RAW_CSV)
    df = pd.concat([exp3, exp4], ignore_index=True)
    df["cell"] = df.apply(lambda r: f"seq{r['seq_len']} @ {r['array_rows']}x{r['array_rows']}", axis=1)

    baselines = df[df["optimizer"].isin(SINGLE_DATAFLOW_METHODS) & (df["status"] == "OK")]
    opt = df[(df["optimizer"] == "bayesian_opt") & (df["status"] == "OK")].copy()
    opt["reward"] = opt["notes"].apply(lambda s: json.loads(s)["reward"])

    rows = []
    conv_rows = []
    for (seq_len, array_rows), cell_baselines in baselines.groupby(["seq_len", "array_rows"]):
        cell = f"seq{seq_len} @ {array_rows}x{array_rows}"
        best_row = cell_baselines.loc[cell_baselines["pe_utilization_pct"].idxmax()]
        best_util, best_method = best_row["pe_utilization_pct"], best_row["optimizer"]

        cell_opt = opt[(opt["seq_len"] == seq_len) & (opt["array_rows"] == array_rows)]
        if cell_opt.empty:
            continue

        best_per_seed = cell_opt.loc[cell_opt.groupby("seed")["reward"].idxmax()]
        util_vals = best_per_seed["pe_utilization_pct"].to_numpy()
        n = len(util_vals)
        mean_util = util_vals.mean()
        std_util = util_vals.std(ddof=1) if n > 1 else float("nan")
        ci = (stats.t.ppf(0.975, df=n - 1) * std_util / n ** 0.5) if n > 1 else float("nan")
        pct_improvement = pct_diff_vs_baseline(mean_util, best_util)
        pct_improvement_ci = (
            pct_diff_vs_baseline(mean_util + ci, best_util) - pct_improvement if n > 1 else float("nan")
        )

        # Cost: total wall-clock for the fixed budget (always defined) per seed, meaned across seeds.
        wall_clock_per_seed = cell_opt.groupby("seed")["wall_clock_s"].sum()
        cost_mean = wall_clock_per_seed.mean()
        cost_ci = (
            stats.t.ppf(0.975, df=len(wall_clock_per_seed) - 1) * wall_clock_per_seed.std(ddof=1) / len(wall_clock_per_seed) ** 0.5
            if len(wall_clock_per_seed) > 1 else float("nan")
        )

        rows.append({
            "cell": cell, "seq_len": seq_len, "array_size": array_rows, "n_seeds": n,
            "best_single_dataflow_method": best_method, "best_single_dataflow_utilization_pct": best_util,
            "optimizer_utilization_pct_mean": mean_util, "optimizer_utilization_pct_std": std_util,
            "pct_improvement_mean": pct_improvement, "pct_improvement_ci": pct_improvement_ci,
            "cost_mean": cost_mean, "cost_ci": cost_ci,
        })

        # Convergence iteration per seed (may be undefined -- reported separately, not silently dropped).
        for seed, g in cell_opt.groupby("seed"):
            g = g.sort_values("iteration")
            conv_iter = convergence_iteration(best_reward_so_far(g["reward"].to_numpy()))
            conv_rows.append({"cell": cell, "seed": seed, "n_iterations": len(g), "convergence_iteration": conv_iter})

    summary = pd.DataFrame(rows).sort_values(["array_size", "seq_len"])
    convergence = pd.DataFrame(conv_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROCESSED_DIR / "exp4_scalability_summary.csv", index=False)
    convergence.to_csv(PROCESSED_DIR / "exp4_convergence_iterations.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_scalability_panels(summary, FIGURES_DIR / "exp4_scalability_panels")

    print(f"[analyze_exp4] wrote {PROCESSED_DIR}/exp4_{{scalability_summary,convergence_iterations}}.csv")
    print(f"[analyze_exp4] wrote {FIGURES_DIR}/exp4_scalability_panels.png/.svg")
    print()
    print(summary.to_string(index=False))
    print()
    n_converged = convergence["convergence_iteration"].notna().sum()
    print(f"Convergence: {n_converged}/{len(convergence)} seeds across all points converged within budget.")
    print(convergence.groupby("cell")["convergence_iteration"].agg(["count", "mean", "std"]).to_string())


if __name__ == "__main__":
    main()
