"""
Analyze Experiment 2's raw results: compute the best-reward-so-far trajectory per
seed, apply the convergence criterion stated in run_exp2.py's docstring, and
produce the required outputs (plan §6 Exp 2): a multi-seed convergence-curve plot
(individual traces + mean ± CI band) and a table of convergence iteration per
seed, mean ± std. Separate from run_exp2.py per CODING_AGENT_PLAN.md §5.

Every number here traces back to results/raw/exp2.csv — nothing is computed from
anything else.
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

from common.plotting.figures import plot_convergence_curves

THIS_DIR = Path(__file__).parent
RAW_CSV = THIS_DIR / "results" / "raw" / "exp2.csv"
PROCESSED_DIR = THIS_DIR / "results" / "processed"
FIGURES_DIR = THIS_DIR / "results" / "figures"

WINDOW = 5      # must match run_exp2.py's --window default (documented in exp2_stats.md §3)
EPSILON = 1.0   # must match run_exp2.py's --epsilon default


def convergence_iteration(best_so_far: np.ndarray, *, window: int, epsilon: float) -> int | None:
    """First iteration i (0-indexed) such that Var(best_so_far[i-window+1 : i+1]) < epsilon.

    Returns None if the criterion is never met within the given trajectory —
    a real "did not converge" result, not silently treated as 0 or discarded.
    """
    for i in range(window - 1, len(best_so_far)):
        window_vals = best_so_far[i - window + 1: i + 1]
        if np.var(window_vals) < epsilon:
            return i
    return None


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    df["reward"] = df["notes"].apply(lambda s: json.loads(s)["reward"])
    ok = df[df["status"] == "OK"].copy().sort_values(["seed", "iteration"])
    ok["best_reward_so_far"] = ok.groupby("seed")["reward"].cummax()

    # --- Convergence-curve plot (required output) ---
    plot_df = ok.rename(columns={"seed": "seed"}).copy()
    plot_df["group"] = "bayesian_opt"  # single optimizer — one line's mean±CI, seed traces shown separately below
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_convergence_curves(
        plot_df[["group", "seed", "iteration", "best_reward_so_far"]],
        FIGURES_DIR / "exp2_convergence_curves_mean_ci",
        value_col="best_reward_so_far", group_col="group",
    )

    # Individual seed traces (plan §6: "individual traces + mean ± CI band" — both required).
    # Distinct colors per seed (not one shared color at alpha — that made every seed's
    # line visually indistinguishable despite separate legend entries, a real bug caught
    # by looking at the rendered figure, not just checking the code ran without error).
    import matplotlib.pyplot as plt
    from common.plotting.style import apply_style, save_figure
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seed_colors = plt.cm.tab10.colors
    for i, (seed, g) in enumerate(ok.groupby("seed")):
        ax.plot(g["iteration"], g["best_reward_so_far"], color=seed_colors[i % len(seed_colors)],
                alpha=0.85, linewidth=1.5, label=f"seed {seed}")
    mean_traj = ok.groupby("iteration")["best_reward_so_far"].mean()
    ax.plot(mean_traj.index, mean_traj.values, color="black", linewidth=2.5, linestyle="--", label="mean")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("best reward so far")
    ax.legend(fontsize=9, ncol=2)
    save_figure(fig, FIGURES_DIR / "exp2_convergence_curves_individual_seeds")
    plt.close(fig)

    # --- Convergence iteration per seed (required output) ---
    rows = []
    for seed, g in ok.groupby("seed"):
        g = g.sort_values("iteration")
        best_so_far = g["best_reward_so_far"].to_numpy()
        conv_iter = convergence_iteration(best_so_far, window=WINDOW, epsilon=EPSILON)
        rows.append({
            "seed": seed, "n_iterations": len(g),
            "convergence_iteration": conv_iter, "converged": conv_iter is not None,
            "final_best_reward": best_so_far[-1],
        })
    per_seed = pd.DataFrame(rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(PROCESSED_DIR / "exp2_convergence_iterations.csv", index=False)
    ok[["seed", "iteration", "reward", "best_reward_so_far", "pe_utilization_pct"]].to_csv(
        PROCESSED_DIR / "exp2_trajectories.csv", index=False,
    )

    n_converged = per_seed["converged"].sum()
    converged_iters = per_seed.loc[per_seed["converged"], "convergence_iteration"]

    print(f"[analyze_exp2] wrote {PROCESSED_DIR / 'exp2_convergence_iterations.csv'} and exp2_trajectories.csv")
    print(f"[analyze_exp2] wrote {FIGURES_DIR / 'exp2_convergence_curves_mean_ci'}.png/.svg and "
          f"exp2_convergence_curves_individual_seeds.png/.svg")
    print()
    print(per_seed.to_string(index=False))
    print()
    print(f"Converged: {n_converged}/{len(per_seed)} seeds")
    if not converged_iters.empty:
        print(f"Convergence iteration (converged seeds only): mean={converged_iters.mean():.2f}, std={converged_iters.std():.2f}")


if __name__ == "__main__":
    main()
