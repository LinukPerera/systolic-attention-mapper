"""
Analyze Experiment 3's raw results: per (seq_len, array) cell, build the baseline
comparison table, aggregate the optimizer across seeds, compute %-improvement over
the best static baseline, and run a significance test — required per
CODING_AGENT_PLAN.md §6 Exp 3 (unlike Experiments 0/1, this is not optional here).
Also produces the required heatmap and %-improvement bar chart. Separate from
run_exp3.py per plan §5.

Every number here traces back to results/raw/exp3.csv — nothing is computed from
anything else. Literature-reported figures (FLAT/COSA/ReDas/Gemmini) are NOT
computed here — they're added as a clearly separate column directly in
exp3_stats.md, never blended into these numbers (plan §6).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from common.metrics import pct_diff_vs_baseline
from common.plotting.figures import plot_pct_improvement_bars, plot_utilization_heatmap

THIS_DIR = Path(__file__).parent
RAW_CSV = THIS_DIR / "results" / "raw" / "exp3.csv"
PROCESSED_DIR = THIS_DIR / "results" / "processed"
FIGURES_DIR = THIS_DIR / "results" / "figures"

BASELINE_METHODS = ["ws_only", "os_only", "is_only", "best_fixed_per_stage"]


def _cell_label(seq_len: int, array_rows: int) -> str:
    return f"seq{seq_len} @ {array_rows}x{array_rows}"


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    df["cell"] = df.apply(lambda r: _cell_label(r["seq_len"], r["array_rows"]), axis=1)

    # --- Baseline table: one row per (cell, method), status included even if FAILED ---
    baselines = df[df["optimizer"].isin(BASELINE_METHODS)].copy()
    baseline_summary = baselines[["cell", "seq_len", "array_rows", "optimizer", "status",
                                   "pe_utilization_pct", "exec_cycles", "sram_dram_traffic_bytes", "edp"]]
    baseline_summary = baseline_summary.rename(columns={"optimizer": "method"})
    baseline_summary.to_csv(PROCESSED_DIR / "exp3_baseline_summary.csv", index=False)

    # --- Optimizer: per (cell, seed), the iteration with max reward is that seed's "result" ---
    opt = df[df["optimizer"] == "bayesian_opt"].copy()
    opt = opt[opt["status"] == "OK"]
    opt["reward"] = opt["notes"].apply(lambda s: json.loads(s)["reward"])
    best_per_seed = opt.loc[opt.groupby(["cell", "seed"])["reward"].idxmax()]

    # Two distinct baselines, NEVER blended (mirroring the plan's own (a)/(b) literature
    # split, but here both are in-harness): (1) "best_overall" = the strongest of all 4
    # static methods, including best_fixed_per_stage, which can assign a *different*
    # dataflow per GEMM stage; (2) "best_single_dataflow" = the strongest of
    # {ws_only, os_only, is_only} only, which — like this optimizer's action space
    # (common/optimizers/environment.py: one dataflow_mode for the WHOLE workload per
    # run, since SCALE-Sim's Dataflow setting applies per-run, not per-layer) — cannot
    # specialize per stage. Comparing the optimizer only against best_overall would
    # silently compare it to a structurally more expressive alternative it has no way
    # to reach; both comparisons are computed and reported, clearly labeled.
    rows = []
    sig_rows = []
    for cell, cell_baselines in baseline_summary[baseline_summary["status"] == "OK"].groupby("cell"):
        best_overall_row = cell_baselines.loc[cell_baselines["pe_utilization_pct"].idxmax()]
        single_df_rows = cell_baselines[cell_baselines["method"].isin(["ws_only", "os_only", "is_only"])]
        best_single_row = single_df_rows.loc[single_df_rows["pe_utilization_pct"].idxmax()]

        cell_opt = best_per_seed[best_per_seed["cell"] == cell]
        if cell_opt.empty:
            continue
        util_vals = cell_opt["pe_utilization_pct"].to_numpy()
        n = len(util_vals)
        mean_util, std_util = util_vals.mean(), (util_vals.std(ddof=1) if n > 1 else float("nan"))
        sem = std_util / n ** 0.5 if n > 1 else float("nan")
        t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
        ci = t_crit * sem if n > 1 else float("nan")

        row = {"cell": cell, "n_seeds": n, "optimizer_utilization_pct_mean": mean_util,
               "optimizer_utilization_pct_std": std_util, "optimizer_utilization_pct_ci95": ci}
        for label, baseline_row in [("best_overall", best_overall_row), ("best_single_dataflow", best_single_row)]:
            baseline_util = baseline_row["pe_utilization_pct"]
            pct_improvement = pct_diff_vs_baseline(mean_util, baseline_util)
            row[f"{label}_method"] = baseline_row["method"]
            row[f"{label}_utilization_pct"] = baseline_util
            row[f"{label}_pct_improvement_mean"] = pct_improvement
            row[f"{label}_pct_improvement_ci"] = (
                pct_diff_vs_baseline(mean_util + ci, baseline_util) - pct_improvement if n > 1 else float("nan")
            )

            if n > 1 and np.std(util_vals) > 0:
                t_stat, t_p = stats.ttest_1samp(util_vals, popmean=baseline_util)
                try:
                    w_stat, w_p = stats.wilcoxon(util_vals - baseline_util)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
                cohens_d = (mean_util - baseline_util) / std_util
            else:
                t_stat = t_p = w_stat = w_p = cohens_d = float("nan")
            sig_rows.append({
                "cell": cell, "n_seeds": n, "comparison": f"bayesian_opt vs. {label} ({baseline_row['method']})",
                "t_statistic": t_stat, "t_pvalue": t_p, "wilcoxon_statistic": w_stat, "wilcoxon_pvalue": w_p,
                "cohens_d": cohens_d, "significant_at_0.05": bool(t_p < 0.05) if not np.isnan(t_p) else None,
                "optimizer_wins": bool(mean_util > baseline_util),
            })
        rows.append(row)

    summary = pd.DataFrame(rows)
    significance = pd.DataFrame(sig_rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROCESSED_DIR / "exp3_optimizer_vs_baseline.csv", index=False)
    significance.to_csv(PROCESSED_DIR / "exp3_significance.csv", index=False)

    # --- Figures ---
    heatmap_df = pd.concat([
        baseline_summary[baseline_summary["status"] == "OK"][["cell", "method", "pe_utilization_pct"]],
        best_per_seed.assign(method="bayesian_opt").groupby(["cell", "method"])["pe_utilization_pct"].mean().reset_index(),
    ])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_utilization_heatmap(heatmap_df, FIGURES_DIR / "exp3_utilization_heatmap")
    for label in ("best_overall", "best_single_dataflow"):
        bars_df = summary.rename(columns={
            f"{label}_pct_improvement_mean": "pct_improvement_mean", f"{label}_pct_improvement_ci": "pct_improvement_ci",
        })[["cell", "pct_improvement_mean", "pct_improvement_ci"]].fillna(0)
        plot_pct_improvement_bars(bars_df, FIGURES_DIR / f"exp3_pct_improvement_bars_vs_{label}")

    print(f"[analyze_exp3] wrote {PROCESSED_DIR}/exp3_{{baseline_summary,optimizer_vs_baseline,significance}}.csv")
    print(f"[analyze_exp3] wrote {FIGURES_DIR}/exp3_utilization_heatmap.png/.svg and "
          f"exp3_pct_improvement_bars_vs_{{best_overall,best_single_dataflow}}.png/.svg")
    print()
    print("=== Baseline summary (status==OK only) ===")
    print(baseline_summary[baseline_summary["status"] == "OK"].pivot(index="cell", columns="method", values="pe_utilization_pct").to_string())
    failed = baseline_summary[baseline_summary["status"] != "OK"]
    if not failed.empty:
        print()
        print("=== FAILED baseline rows ===")
        print(failed[["cell", "method", "status"]].to_string(index=False))
    print()
    print("=== Optimizer vs. best baseline ===")
    print(summary.to_string(index=False))
    print()
    print("=== Significance ===")
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()
