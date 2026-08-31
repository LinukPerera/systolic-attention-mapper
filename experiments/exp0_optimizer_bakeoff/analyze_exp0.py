"""
Analyze Experiment 0's raw results into results/processed/ tables and
results/figures/ plots. Separate from run_exp0.py per CODING_AGENT_PLAN.md §5: a
figure/table can be regenerated after a data fix without re-running the experiment.

Every number here traces back to results/raw/exp0.csv — nothing is computed from
anything else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from common.plotting.figures import plot_convergence_curves

THIS_DIR = Path(__file__).parent
RAW_CSV = THIS_DIR / "results" / "raw" / "exp0.csv"
PROCESSED_DIR = THIS_DIR / "results" / "processed"
FIGURES_DIR = THIS_DIR / "results" / "figures"

CONVERGENCE_THRESHOLD_FRAC = 0.05  # "within 5% of brute-force best" — see exp0_stats.md §3 for why this value


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    df["reward"] = df["notes"].apply(lambda s: json.loads(s)["reward"])
    ok = df[df["status"] == "OK"].copy()

    brute_force = ok[ok["optimizer"] == "brute_force"]
    if brute_force.empty:
        raise ValueError("No OK brute_force rows in exp0.csv — cannot establish a ground-truth best.")
    best_reward = brute_force["reward"].max()
    print(f"[analyze_exp0] brute-force best reward (ground truth): {best_reward:.4f}")

    candidates = ok[ok["optimizer"] != "brute_force"].copy()
    candidates = candidates.sort_values(["optimizer", "seed", "iteration"])
    candidates["best_reward_so_far"] = candidates.groupby(["optimizer", "seed"])["reward"].cummax()

    # --- Convergence-curve overlay (Exp0's required plot) ---
    plot_df = candidates.rename(columns={"optimizer": "group"})[["group", "seed", "iteration", "best_reward_so_far"]]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_convergence_curves(
        plot_df, FIGURES_DIR / "exp0_convergence_curves", value_col="best_reward_so_far", group_col="group",
    )
    print(f"[analyze_exp0] wrote {FIGURES_DIR / 'exp0_convergence_curves'}.png/.svg")

    # --- Summary table: per optimizer, iterations-to-within-5%-of-best, wall-clock, final utilization ---
    threshold = best_reward * (1 - CONVERGENCE_THRESHOLD_FRAC) if best_reward > 0 else best_reward * (1 + CONVERGENCE_THRESHOLD_FRAC)
    rows = []
    for (optimizer, seed), g in candidates.groupby(["optimizer", "seed"]):
        g = g.sort_values("iteration")
        reached = g[g["best_reward_so_far"] >= threshold]
        iters_to_converge = int(reached["iteration"].iloc[0]) if not reached.empty else None
        rows.append({
            "optimizer": optimizer, "seed": seed,
            "n_evals": len(g), "iters_to_converge": iters_to_converge,
            "final_best_reward": g["best_reward_so_far"].iloc[-1],
            "final_utilization_pct": g.loc[g["reward"] == g["best_reward_so_far"].iloc[-1], "pe_utilization_pct"].iloc[0],
            "mean_wall_clock_s": g["wall_clock_s"].mean(),
        })
    per_seed = pd.DataFrame(rows)

    summary = per_seed.groupby("optimizer").agg(
        n_seeds=("seed", "count"),
        converged_seeds=("iters_to_converge", lambda s: s.notna().sum()),
        iters_to_converge_mean=("iters_to_converge", "mean"),
        iters_to_converge_std=("iters_to_converge", "std"),
        final_best_reward_mean=("final_best_reward", "mean"),
        final_best_reward_std=("final_best_reward", "std"),
        final_utilization_pct_mean=("final_utilization_pct", "mean"),
        final_utilization_pct_std=("final_utilization_pct", "std"),
        mean_wall_clock_s_mean=("mean_wall_clock_s", "mean"),
    ).reset_index()
    summary["distance_from_brute_force_best"] = best_reward - summary["final_best_reward_mean"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(PROCESSED_DIR / "exp0_per_seed.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "exp0_summary.csv", index=False)
    print(f"[analyze_exp0] wrote {PROCESSED_DIR / 'exp0_per_seed.csv'} and exp0_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
