"""
Analyze Experiment 5's ablation results: aggregates dataflow_only/tiling_only (this
experiment's own results/raw/exp5*.csv) alongside joint (the selected optimizer,
reused from Experiment 3's raw CSV -- see run_exp5.py's docstring for why not
re-run), computes per-variant mean/std/95% CI utilization per cell, and states
plainly how much of the total gain each factor contributes (plan §6 Exp 5's
acceptance criteria). Produces the required bar chart with error bars.

Every number here traces back to results/raw/exp5*.csv or the Experiment 3 raw
CSV it's pointed at -- nothing is computed from anything else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from common.plotting.figures import plot_ablation_bars

THIS_DIR = Path(__file__).parent
DEFAULT_EXP3_RAW_CSV = (
    _PROJECT_ROOT / "experiments" / "exp3_main_sweep" / "results" / "raw" / "exp3_perstage.csv"
)
DEFAULT_EXP5_RAW_CSV = THIS_DIR / "results" / "raw" / "exp5.csv"
PROCESSED_DIR = THIS_DIR / "results" / "processed"
FIGURES_DIR = THIS_DIR / "results" / "figures"


def _cell_label(seq_len: int, array_rows: int) -> str:
    return f"seq{seq_len} @ {array_rows}x{array_rows}"


def _best_per_seed(df: pd.DataFrame, variant: str, seq_len: int, array_rows: int) -> pd.Series:
    sub = df[(df["optimizer"] == variant) & (df["seq_len"] == seq_len) & (df["array_rows"] == array_rows)
             & (df["status"] == "OK")].copy()
    sub["reward"] = sub["notes"].apply(lambda s: json.loads(s)["reward"])
    best = sub.loc[sub.groupby("seed")["reward"].idxmax()]
    return best["pe_utilization_pct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp5-raw-csv", type=str, default=str(DEFAULT_EXP5_RAW_CSV))
    parser.add_argument("--exp3-raw-csv", type=str, default=str(DEFAULT_EXP3_RAW_CSV))
    # See analyze_exp0.py's --out-prefix for why: keeps this re-run's outputs
    # from colliding with the original (3-dim) exp5 run's.
    parser.add_argument("--out-prefix", type=str, default="exp5")
    parser.add_argument("--joint-optimizer-name", type=str, default=None,
                         help="Which optimizer's rows in --exp3-raw-csv count as 'joint' -- defaults to "
                              "whatever config/optimizers/selected.yaml currently names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joint_optimizer_name = args.joint_optimizer_name
    if joint_optimizer_name is None:
        with open(_PROJECT_ROOT / "config" / "optimizers" / "selected.yaml") as f:
            joint_optimizer_name = yaml.safe_load(f)["selected_optimizer"]

    exp3 = pd.read_csv(args.exp3_raw_csv)
    exp5 = pd.read_csv(args.exp5_raw_csv)

    cells = exp5[["seq_len", "array_rows"]].drop_duplicates().to_records(index=False)

    rows = []
    bars_rows = []
    sig_rows = []
    for seq_len, array_rows in cells:
        cell = _cell_label(seq_len, array_rows)

        # Gain-attribution baseline: best_single_dataflow (os_only) -- SAME reference as the
        # original (3-dim) exp5 run, kept unchanged here deliberately. best_fixed_per_stage is
        # reported alongside as separate context (below), not used as the gain-attribution
        # denominator: on the per-stage action space, `joint` (a2c) no longer reliably beats
        # best_fixed_per_stage (exp3_stats.md §11 -- 0/4 cells significant, 2/4 numerically
        # below it) so `joint - best_fixed_per_stage` is near-zero or negative in these 2 cells,
        # making a "% of joint's gain" ratio against it degenerate/sign-flipping rather than
        # informative -- verified by computing it and inspecting the result before choosing not
        # to use it as the primary framing (real evidence, not assumed).
        os_baseline_row = exp3[(exp3["seq_len"] == seq_len) & (exp3["array_rows"] == array_rows)
                                & (exp3["optimizer"] == "os_only") & (exp3["status"] == "OK")]
        best_static_util = os_baseline_row.iloc[0]["pe_utilization_pct"]
        bfps_row = exp3[(exp3["seq_len"] == seq_len) & (exp3["array_rows"] == array_rows)
                         & (exp3["optimizer"] == "best_fixed_per_stage") & (exp3["status"] == "OK")]
        best_fixed_per_stage_util = bfps_row.iloc[0]["pe_utilization_pct"]

        variant_stats = {}
        variant_vals = {}
        for variant, source in [("dataflow_only", exp5), ("tiling_only", exp5), ("joint", exp3)]:
            vals = _best_per_seed(source, joint_optimizer_name if variant == "joint" else variant, seq_len, array_rows)
            variant_vals[variant] = vals
            n = len(vals)
            mean, std = vals.mean(), (vals.std(ddof=1) if n > 1 else float("nan"))
            ci = (stats.t.ppf(0.975, df=n - 1) * std / n ** 0.5) if n > 1 and std > 0 else 0.0
            variant_stats[variant] = (mean, std, ci, n)
            rows.append({
                "cell": cell, "variant": variant, "n_seeds": n,
                "utilization_pct_mean": mean, "utilization_pct_std": std, "utilization_pct_ci95": ci,
                "best_single_dataflow_utilization_pct": best_static_util,
                "best_fixed_per_stage_utilization_pct": best_fixed_per_stage_util,
            })
            bars_rows.append({"cell": cell, "variant": variant, "value_mean": mean, "value_ci": ci})

        # How much of the total gain (joint - best_single_dataflow) is captured by each factor alone?
        joint_gain = variant_stats["joint"][0] - best_static_util
        dataflow_gain = variant_stats["dataflow_only"][0] - best_static_util
        tiling_gain = variant_stats["tiling_only"][0] - best_static_util
        rows.append({
            "cell": cell, "variant": "_gain_attribution", "n_seeds": None,
            "utilization_pct_mean": None, "utilization_pct_std": None, "utilization_pct_ci95": None,
            "best_single_dataflow_utilization_pct": best_static_util,
            "best_fixed_per_stage_utilization_pct": best_fixed_per_stage_util,
            "joint_gain_pct_points": joint_gain,
            "dataflow_only_gain_pct_points": dataflow_gain,
            "dataflow_only_pct_of_joint_gain": (dataflow_gain / joint_gain * 100) if joint_gain != 0 else float("nan"),
            "tiling_only_gain_pct_points": tiling_gain,
            "tiling_only_pct_of_joint_gain": (tiling_gain / joint_gain * 100) if joint_gain != 0 else float("nan"),
        })

        # Paired-by-seed significance: tiling_only vs. joint (same convention as the original
        # 3-dim run's §5), plus dataflow_only vs. joint (new -- worth checking directly given
        # dataflow_only is no longer trivially 0% of a well-defined gain here).
        for variant in ("tiling_only", "dataflow_only"):
            a, b = variant_vals[variant].to_numpy(), variant_vals["joint"].to_numpy()
            if len(a) > 1 and len(b) > 1 and np.std(a - b) > 0:
                t_stat, t_p = stats.ttest_rel(a, b)
                try:
                    w_stat, w_p = stats.wilcoxon(a - b)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
            else:
                t_stat = t_p = w_stat = w_p = float("nan")
            sig_rows.append({
                "cell": cell, "comparison": f"{variant} vs. joint (paired by seed)",
                "t_statistic": t_stat, "t_pvalue": t_p, "wilcoxon_statistic": w_stat, "wilcoxon_pvalue": w_p,
                "significant_at_0.05": bool(t_p < 0.05) if not np.isnan(t_p) else None,
                f"{variant}_wins": bool(a.mean() > b.mean()),
            })

    summary = pd.DataFrame(rows)
    significance = pd.DataFrame(sig_rows)
    bars_df = pd.DataFrame(bars_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = PROCESSED_DIR / f"{args.out_prefix}_ablation_summary.csv"
    sig_path = PROCESSED_DIR / f"{args.out_prefix}_significance.csv"
    summary.to_csv(summary_path, index=False)
    significance.to_csv(sig_path, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"{args.out_prefix}_ablation_bars"
    plot_ablation_bars(bars_df, fig_path)

    print(f"[analyze_exp5] joint_optimizer_name={joint_optimizer_name!r}")
    print(f"[analyze_exp5] wrote {summary_path} and {sig_path}")
    print(f"[analyze_exp5] wrote {fig_path}.png/.svg")
    print()
    print(summary[summary["variant"] != "_gain_attribution"].to_string(index=False))
    print()
    print("=== Gain attribution (vs. best_single_dataflow = os_only, same reference as the original run) ===")
    print(summary[summary["variant"] == "_gain_attribution"].drop(
        columns=["variant", "n_seeds", "utilization_pct_mean", "utilization_pct_std", "utilization_pct_ci95"]
    ).to_string(index=False))
    print()
    print("=== Paired significance (tiling_only/dataflow_only vs. joint) ===")
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()
