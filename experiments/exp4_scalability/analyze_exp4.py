"""
Analyze Experiment 4's scalability trend: merges Experiment 3's real results (the
4 already-run (seq_len, array) points) with Experiment 4's own new point
(seq_len=512 @ 32x32) — see run_exp4.py's docstring for why the 4 overlapping
points are reused rather than re-simulated. Computes, per point: %-improvement of
the selected optimizer over the best single-dataflow baseline (the same FAIR
comparison established in Experiment 3 — best_fixed_per_stage is deliberately
excluded here too, for direct comparability with the original 3-dim trend, even
though Experiment 3's per-stage re-run found the optimizer now statistically
matches best_fixed_per_stage — see exp4_stats.md for how this is handled), and
optimizer cost (both convergence-iteration, where defined, and total wall-clock
for the fixed budget, which is always defined). Produces the required two-panel
scalability plot.

Every number here traces back to the Experiment 3 raw CSV or this experiment's
own raw CSV — nothing is computed from anything else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml
from scipy import stats

from common.metrics import pct_diff_vs_baseline
from common.optimizers.convergence import best_reward_so_far, convergence_iteration
from common.plotting.figures import plot_scalability_panels

THIS_DIR = Path(__file__).parent
DEFAULT_EXP3_RAW_CSV = (
    _PROJECT_ROOT / "experiments" / "exp3_main_sweep" / "results" / "raw" / "exp3_perstage.csv"
)
DEFAULT_EXP4_RAW_CSV = THIS_DIR / "results" / "raw" / "exp4.csv"
PROCESSED_DIR = THIS_DIR / "results" / "processed"
FIGURES_DIR = THIS_DIR / "results" / "figures"

SINGLE_DATAFLOW_METHODS = ["ws_only", "os_only", "is_only"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp3-raw-csv", type=str, default=str(DEFAULT_EXP3_RAW_CSV))
    parser.add_argument("--exp4-raw-csv", type=str, default=str(DEFAULT_EXP4_RAW_CSV))
    # See analyze_exp0.py's --out-prefix for why: keeps this re-run's outputs
    # from colliding with the original (3-dim) exp4 run's.
    parser.add_argument("--out-prefix", type=str, default="exp4")
    parser.add_argument("--optimizer-name", type=str, default=None,
                         help="Which optimizer's rows count as 'the optimizer' -- defaults to whatever "
                              "config/optimizers/selected.yaml currently names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimizer_name = args.optimizer_name
    if optimizer_name is None:
        with open(_PROJECT_ROOT / "config" / "optimizers" / "selected.yaml") as f:
            optimizer_name = yaml.safe_load(f)["selected_optimizer"]

    exp3 = pd.read_csv(args.exp3_raw_csv)
    exp4 = pd.read_csv(args.exp4_raw_csv)
    df = pd.concat([exp3, exp4], ignore_index=True)
    df["cell"] = df.apply(lambda r: f"seq{r['seq_len']} @ {r['array_rows']}x{r['array_rows']}", axis=1)

    baselines = df[df["optimizer"].isin(SINGLE_DATAFLOW_METHODS) & (df["status"] == "OK")]
    opt = df[(df["optimizer"] == optimizer_name) & (df["status"] == "OK")].copy()
    if opt.empty:
        raise ValueError(
            f"No rows with optimizer=={optimizer_name!r} across {args.exp3_raw_csv} + {args.exp4_raw_csv} -- "
            "pass --optimizer-name explicitly if these CSVs used a different optimizer than "
            "config/optimizers/selected.yaml currently names."
        )
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
    summary_path = PROCESSED_DIR / f"{args.out_prefix}_scalability_summary.csv"
    conv_path = PROCESSED_DIR / f"{args.out_prefix}_convergence_iterations.csv"
    summary.to_csv(summary_path, index=False)
    convergence.to_csv(conv_path, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"{args.out_prefix}_scalability_panels"
    plot_scalability_panels(summary, fig_path)

    print(f"[analyze_exp4] optimizer_name={optimizer_name!r}")
    print(f"[analyze_exp4] wrote {summary_path} and {conv_path}")
    print(f"[analyze_exp4] wrote {fig_path}.png/.svg")
    print()
    print(summary.to_string(index=False))
    print()
    n_converged = convergence["convergence_iteration"].notna().sum()
    print(f"Convergence: {n_converged}/{len(convergence)} seeds across all points converged within budget.")
    print(convergence.groupby("cell")["convergence_iteration"].agg(["count", "mean", "std"]).to_string())


if __name__ == "__main__":
    main()
