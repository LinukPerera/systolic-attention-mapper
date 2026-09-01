"""
Analyze Experiment 5's ablation results: aggregates dataflow_only/tiling_only (this
experiment's own results/raw/exp5.csv) alongside joint (bayesian_opt, reused from
Experiment 3's results/raw/exp3.csv — see run_exp5.py's docstring for why not
re-run), computes per-variant mean/std/95% CI utilization per cell, and states
plainly how much of the total gain each factor contributes (plan §6 Exp 5's
acceptance criteria). Produces the required bar chart with error bars.

Every number here traces back to results/raw/exp5.csv or results/raw/exp3.csv —
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

from common.plotting.figures import plot_ablation_bars

THIS_DIR = Path(__file__).parent
EXP3_RAW_CSV = _PROJECT_ROOT / "experiments" / "exp3_main_sweep" / "results" / "raw" / "exp3.csv"
EXP5_RAW_CSV = THIS_DIR / "results" / "raw" / "exp5.csv"
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


def main() -> None:
    exp3 = pd.read_csv(EXP3_RAW_CSV)
    exp5 = pd.read_csv(EXP5_RAW_CSV)

    cells = exp5[["seq_len", "array_rows"]].drop_duplicates().to_records(index=False)

    rows = []
    bars_rows = []
    for seq_len, array_rows in cells:
        cell = _cell_label(seq_len, array_rows)

        # Best static baseline for this cell (from Experiment 3's real data) -- context, not a 4th bar.
        baseline_row = exp3[(exp3["seq_len"] == seq_len) & (exp3["array_rows"] == array_rows)
                             & (exp3["optimizer"] == "os_only") & (exp3["status"] == "OK")]
        best_static_util = baseline_row.iloc[0]["pe_utilization_pct"]

        variant_stats = {}
        for variant, source in [("dataflow_only", exp5), ("tiling_only", exp5), ("joint", exp3)]:
            vals = _best_per_seed(source, "bayesian_opt" if variant == "joint" else variant, seq_len, array_rows)
            n = len(vals)
            mean, std = vals.mean(), (vals.std(ddof=1) if n > 1 else float("nan"))
            ci = (stats.t.ppf(0.975, df=n - 1) * std / n ** 0.5) if n > 1 and std > 0 else 0.0
            variant_stats[variant] = (mean, std, ci, n)
            rows.append({
                "cell": cell, "variant": variant, "n_seeds": n,
                "utilization_pct_mean": mean, "utilization_pct_std": std, "utilization_pct_ci95": ci,
                "best_static_baseline_utilization_pct": best_static_util,
            })
            bars_rows.append({"cell": cell, "variant": variant, "value_mean": mean, "value_ci": ci})

        # How much of the total gain (joint - best_static) is captured by each factor alone?
        joint_gain = variant_stats["joint"][0] - best_static_util
        dataflow_gain = variant_stats["dataflow_only"][0] - best_static_util
        tiling_gain = variant_stats["tiling_only"][0] - best_static_util
        rows.append({
            "cell": cell, "variant": "_gain_attribution", "n_seeds": None,
            "utilization_pct_mean": None, "utilization_pct_std": None, "utilization_pct_ci95": None,
            "best_static_baseline_utilization_pct": best_static_util,
            "joint_gain_pct_points": joint_gain,
            "dataflow_only_gain_pct_points": dataflow_gain,
            "dataflow_only_pct_of_joint_gain": (dataflow_gain / joint_gain * 100) if joint_gain != 0 else float("nan"),
            "tiling_only_gain_pct_points": tiling_gain,
            "tiling_only_pct_of_joint_gain": (tiling_gain / joint_gain * 100) if joint_gain != 0 else float("nan"),
        })

    summary = pd.DataFrame(rows)
    bars_df = pd.DataFrame(bars_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROCESSED_DIR / "exp5_ablation_summary.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_ablation_bars(bars_df, FIGURES_DIR / "exp5_ablation_bars")

    print(f"[analyze_exp5] wrote {PROCESSED_DIR}/exp5_ablation_summary.csv")
    print(f"[analyze_exp5] wrote {FIGURES_DIR}/exp5_ablation_bars.png/.svg")
    print()
    print(summary[summary["variant"] != "_gain_attribution"].to_string(index=False))
    print()
    print("=== Gain attribution (vs. best_static baseline) ===")
    print(summary[summary["variant"] == "_gain_attribution"].drop(
        columns=["variant", "n_seeds", "utilization_pct_mean", "utilization_pct_std", "utilization_pct_ci95"]
    ).to_string(index=False))


if __name__ == "__main__":
    main()
