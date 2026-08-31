"""
Analyze Experiment 1's raw results into results/processed/ tables and
results/figures/ plots. Separate from run_exp1.py per CODING_AGENT_PLAN.md §5.

Takes --raw-csv/--label so both the original seq_len=64 pilot run and the
seq_len=256 representative re-run (config/workloads/exp1_representative.yaml)
can be analyzed independently into distinctly-named outputs — the original run
is kept as a documented sub-finding (workload-scale dependency of array
utilization), not overwritten, per plan ground rule 6.

Every number here traces back to the given raw CSV — nothing is computed from
anything else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from common.plotting.figures import plot_utilization_vs_array_size

THIS_DIR = Path(__file__).parent
PROCESSED_DIR = THIS_DIR / "results" / "processed"
FIGURES_DIR = THIS_DIR / "results" / "figures"

HEADLINE_METHODS = ["ws_only", "os_only", "is_only", "best_fixed_per_stage"]
SUFFICIENCY_THRESHOLD_PCT = 2.0  # "<2% utilization change per doubling" — plan §6 Exp1's suggested criterion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=str, default=str(THIS_DIR / "results" / "raw" / "exp1.csv"))
    parser.add_argument("--label", type=str, default="exp1",
                        help="Prefix for output filenames, e.g. 'exp1' or 'exp1_representative'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_csv = Path(args.raw_csv)
    label = args.label

    df = pd.read_csv(raw_csv)
    ok = df[df["status"] == "OK"].copy()
    headline = ok[ok["optimizer"].isin(HEADLINE_METHODS)].copy()
    headline["array_size"] = headline["array_rows"]  # square arrays throughout this project so far
    headline = headline.rename(columns={"optimizer": "method"})

    summary = headline[["method", "array_size", "pe_utilization_pct", "exec_cycles", "sram_dram_traffic_bytes", "edp"]].sort_values(["method", "array_size"])

    # --- %-change-per-doubling, per method ---
    pct_change_rows = []
    for method, g in summary.groupby("method"):
        g = g.sort_values("array_size").reset_index(drop=True)
        for i in range(1, len(g)):
            prev_util, cur_util = g.loc[i - 1, "pe_utilization_pct"], g.loc[i, "pe_utilization_pct"]
            pct_change = (cur_util - prev_util) / prev_util * 100.0
            pct_change_rows.append({
                "method": method,
                "from_array_size": int(g.loc[i - 1, "array_size"]), "to_array_size": int(g.loc[i, "array_size"]),
                "utilization_pct_change": pct_change,
                "meets_sufficiency_criterion": abs(pct_change) < SUFFICIENCY_THRESHOLD_PCT,
            })
    pct_change_df = pd.DataFrame(pct_change_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROCESSED_DIR / f"{label}_summary.csv", index=False)
    pct_change_df.to_csv(PROCESSED_DIR / f"{label}_pct_change_per_doubling.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_utilization_vs_array_size(summary, FIGURES_DIR / f"{label}_utilization_vs_array_size")

    print(f"[analyze_exp1] source: {raw_csv}")
    print(f"[analyze_exp1] wrote {PROCESSED_DIR / f'{label}_summary.csv'} and {label}_pct_change_per_doubling.csv")
    print(f"[analyze_exp1] wrote {FIGURES_DIR / f'{label}_utilization_vs_array_size'}.png/.svg")
    print()
    print("Utilization by array size and method:")
    print(summary.pivot(index="array_size", columns="method", values="pe_utilization_pct").to_string())
    print()
    print(f"%-change per doubling (sufficiency criterion: |change| < {SUFFICIENCY_THRESHOLD_PCT}%):")
    print(pct_change_df.to_string(index=False))
    print()
    n_meeting = pct_change_df["meets_sufficiency_criterion"].sum()
    print(f"Transitions meeting the <{SUFFICIENCY_THRESHOLD_PCT}% criterion: {n_meeting} / {len(pct_change_df)}")


if __name__ == "__main__":
    main()
