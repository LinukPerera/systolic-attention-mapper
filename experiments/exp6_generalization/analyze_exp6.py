"""
Analyze Experiment 6's zero-shot generalization/transfer results: per held-out
cell, aggregates zero-shot transfer (n=5, one per source seed) and fresh-ceiling
training (n=5 seeds), computes the generalization-capture ratio against
`best_fixed_per_stage`, and runs both required significance tests (see
`docs/NEXT_INSTRUCTIONS.md` §4):
  - zero-shot transfer vs. best_fixed_per_stage (one-sample, `common.stats_utils`)
  - zero-shot transfer vs. fresh ceiling (independent two-sample, `common.stats_utils`)
Bonferroni-corrects across all 2 cells x 2 comparisons (m=4).

Every number here traces back to results/raw/exp6.csv — nothing is computed from
anything else.
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

from common.plotting.figures import plot_generalization_comparison
from common.stats_utils import one_sample_vs_baseline, two_sample_independent

THIS_DIR = Path(__file__).parent
DEFAULT_RAW_CSV = THIS_DIR / "results" / "raw" / "exp6.csv"
PROCESSED_DIR = THIS_DIR / "results" / "processed"
FIGURES_DIR = THIS_DIR / "results" / "figures"

BONFERRONI_M = 4  # 2 held-out cells x 2 comparisons each (docs/NEXT_INSTRUCTIONS.md §4)


def _cell_label(seq_len: int, array_rows: int, head_dim: int) -> str:
    return f"seq{seq_len} @ {array_rows}x{array_rows} (head_dim={head_dim})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=str, default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--out-prefix", type=str, default="exp6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.raw_csv)
    df["head_dim"] = df["hidden_dim"] // df["num_heads"]
    df["cell"] = df.apply(lambda r: _cell_label(r["seq_len"], r["array_rows"], r["head_dim"]), axis=1)

    agg_rows = []
    sig_rows = []
    for cell, cell_df in df.groupby("cell"):
        bfps_row = cell_df[(cell_df["optimizer"] == "best_fixed_per_stage") & (cell_df["status"] == "OK")]
        if bfps_row.empty:
            print(f"[analyze_exp6] {cell}: no OK best_fixed_per_stage row -- skipping (real FAILED evidence, not hidden).")
            continue
        baseline_util = bfps_row.iloc[0]["pe_utilization_pct"]

        transfer = cell_df[(cell_df["optimizer"] == "zero_shot_transfer") & (cell_df["status"] == "OK")]
        ceiling = cell_df[(cell_df["optimizer"] == "fresh_ceiling") & (cell_df["status"] == "OK")].copy()
        if transfer.empty or ceiling.empty:
            print(f"[analyze_exp6] {cell}: missing transfer or ceiling rows -- skipping.")
            continue
        ceiling["reward"] = ceiling["notes"].apply(lambda s: json.loads(s)["reward"])
        ceiling_best_per_seed = ceiling.loc[ceiling.groupby("seed")["reward"].idxmax()]

        transfer_vals = transfer["pe_utilization_pct"].to_numpy()
        ceiling_vals = ceiling_best_per_seed["pe_utilization_pct"].to_numpy()
        transfer_mean, ceiling_mean = transfer_vals.mean(), ceiling_vals.mean()

        capture_denominator = ceiling_mean - baseline_util
        capture_pct = (
            (transfer_mean - baseline_util) / capture_denominator * 100 if capture_denominator != 0 else float("nan")
        )

        agg_rows.append({
            "cell": cell, "best_fixed_per_stage_utilization_pct": baseline_util,
            "zero_shot_transfer_n": len(transfer_vals), "zero_shot_transfer_mean": transfer_mean,
            "zero_shot_transfer_std": transfer_vals.std(ddof=1) if len(transfer_vals) > 1 else float("nan"),
            "fresh_ceiling_n": len(ceiling_vals), "fresh_ceiling_mean": ceiling_mean,
            "fresh_ceiling_std": ceiling_vals.std(ddof=1) if len(ceiling_vals) > 1 else float("nan"),
            "generalization_capture_pct": capture_pct,
        })

        vs_baseline = one_sample_vs_baseline(transfer_vals, baseline_util)
        vs_ceiling = two_sample_independent(transfer_vals, ceiling_vals)
        sig_rows.append({
            "cell": cell, "comparison": "zero_shot_transfer vs. best_fixed_per_stage (one-sample)",
            "n": vs_baseline.n, "statistic": vs_baseline.t_statistic, "pvalue": vs_baseline.t_pvalue,
            "wilcoxon_pvalue": vs_baseline.wilcoxon_pvalue, "cohens_d": vs_baseline.cohens_d,
        })
        sig_rows.append({
            "cell": cell, "comparison": "zero_shot_transfer vs. fresh_ceiling (independent two-sample, Welch)",
            "n": f"{vs_ceiling.n_a}v{vs_ceiling.n_b}", "statistic": vs_ceiling.t_statistic, "pvalue": vs_ceiling.t_pvalue,
            "wilcoxon_pvalue": vs_ceiling.u_pvalue, "cohens_d": float("nan"),
        })

    agg = pd.DataFrame(agg_rows)
    significance = pd.DataFrame(sig_rows)
    significance["pvalue_bonferroni"] = (significance["pvalue"] * BONFERRONI_M).clip(upper=1.0)
    significance["significant_at_0.05_uncorrected"] = significance["pvalue"] < 0.05
    significance["significant_at_0.05_bonferroni"] = significance["pvalue_bonferroni"] < 0.05

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    agg_path = PROCESSED_DIR / f"{args.out_prefix}_agg.csv"
    sig_path = PROCESSED_DIR / f"{args.out_prefix}_significance.csv"
    agg.to_csv(agg_path, index=False)
    significance.to_csv(sig_path, index=False)

    bars_rows = []
    for _, row in agg.iterrows():
        bars_rows.append({"cell": row["cell"], "variant": "baseline", "value_mean": row["best_fixed_per_stage_utilization_pct"], "value_ci": 0.0})
        bars_rows.append({"cell": row["cell"], "variant": "zero_shot_transfer", "value_mean": row["zero_shot_transfer_mean"], "value_ci": row["zero_shot_transfer_std"]})
        bars_rows.append({"cell": row["cell"], "variant": "fresh_ceiling", "value_mean": row["fresh_ceiling_mean"], "value_ci": row["fresh_ceiling_std"]})
    bars_df = pd.DataFrame(bars_rows)
    capture_labels = dict(zip(agg["cell"], agg["generalization_capture_pct"]))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"{args.out_prefix}_generalization_bars"
    plot_generalization_comparison(bars_df, capture_labels, fig_path)

    print(f"[analyze_exp6] wrote {agg_path} and {sig_path}")
    print(f"[analyze_exp6] wrote {fig_path}.png/.svg")
    print()
    print(agg.to_string(index=False))
    print()
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()
