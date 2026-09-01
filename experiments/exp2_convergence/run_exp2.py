"""
Experiment 2 -- Search/training convergence study

Prove the *selected* optimizer (bayesian_opt, from config/optimizers/selected.yaml
-- Experiment 0's choice, not re-decided here) reliably converges, with an
explicit numeric stopping criterion, per CODING_AGENT_PLAN.md §6 Experiment 2.

**Convergence criterion (stated as a formula, not "it looked flat" -- plan §6
acceptance criteria):** track each seed's *best-reward-so-far* trajectory
b_1, b_2, ..., b_T (a running max over that seed's raw per-iteration rewards --
monotonically non-decreasing by construction, since best-so-far only ever holds
steady or improves). A seed is declared CONVERGED at the first iteration i >=
WINDOW such that:

    Var(b_{i-WINDOW+1}, ..., b_i) < EPSILON

i.e. the population variance of the best-so-far value over the trailing WINDOW
iterations drops below EPSILON. Because b is monotonically non-decreasing, a
low windowed variance means b has stopped changing meaningfully within that
window -- the optimizer isn't finding anything better any more, not just that
it "looks flat" by eye. WINDOW and EPSILON are real hyperparameters (CLI flags
below, not hardcoded -- plan §3 "no magic numbers"), chosen and documented in
exp2_stats.md §3, not tuned after seeing the data.

A seed that never meets the criterion within the iteration budget is reported
as NOT converged, not discarded (plan §6 acceptance criteria).

See ../../docs/CODING_AGENT_PLAN.md §6 (Experiment 2) for the full spec and
exp2_stats.md for the filled-in report.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from common import logging_utils
from common.metrics import compute_edp
from common.optimizers.bayesian_opt import BayesianOptOptimizer
from common.optimizers.environment import AttentionMappingProblem, EvalResult
from common.simulator_interface import load_hardware_config
from common.workload_gen import load_workload_yaml

EXPERIMENT_ID = "exp2"
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw" / f"{EXPERIMENT_ID}.csv"

SELECTED_OPTIMIZER_YAML = _PROJECT_ROOT / "config" / "optimizers" / "selected.yaml"


def _load_selected_optimizer() -> BayesianOptOptimizer:
    """Load the optimizer Experiment 0 selected -- never re-decided here (plan §6 Exp 2)."""
    with open(SELECTED_OPTIMIZER_YAML) as f:
        selected = yaml.safe_load(f)
    if selected["selected_optimizer"] != "bayesian_opt":
        raise ValueError(
            f"config/optimizers/selected.yaml names {selected['selected_optimizer']!r}, but "
            "run_exp2.py only knows how to instantiate bayesian_opt -- update this function "
            "(not the selection) if Experiment 0's choice changes."
        )
    hp_path = _PROJECT_ROOT / selected["hyperparameters_file"]
    with open(hp_path) as f:
        hp = yaml.safe_load(f) or {}
    return BayesianOptOptimizer(**hp)


def _log_eval_result(
    *, csv_path: Path, git_commit: str, hardware, workload, seed: int, iteration: int, eval_result: EvalResult,
) -> None:
    sim = eval_result.simulation
    action = eval_result.action
    notes = json.dumps({
        "ifmap_frac": round(action.ifmap_frac, 4), "filter_frac": round(action.filter_frac, 4),
        "ofmap_frac": round(action.ofmap_frac, 4), "reward": round(eval_result.reward, 6),
        "sim_notes": sim.notes,
    })
    row = {
        "run_id": f"{EXPERIMENT_ID}-bayesian_opt-seed{seed}-iter{iteration}",
        "experiment": EXPERIMENT_ID, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "optimizer": "bayesian_opt",
        "array_rows": hardware.array_rows, "array_cols": hardware.array_cols,
        "sram_kb": sum(hardware.sram_kb.values()), "dataflow_mode": action.dataflow_mode,
        "seq_len": workload.seq_len, "hidden_dim": workload.hidden_dim, "num_heads": workload.num_heads,
        "seed": seed, "iteration": iteration,
        "pe_utilization_pct": sim.pe_utilization_pct if sim.status == "OK" else None,
        "exec_cycles": sim.exec_cycles if sim.status == "OK" else None,
        "sram_dram_traffic_bytes": sim.sram_dram_traffic_bytes if sim.status == "OK" else None,
        "edp": compute_edp(sim, energy_per_cycle_pj=hardware.energy_per_cycle_pj) if sim.status == "OK" else None,
        "wall_clock_s": sim.wall_clock_s, "status": sim.status, "notes": notes,
    }
    logging_utils.append_run_row(csv_path, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=str, default="config/workloads/exp2_convergence.yaml")
    parser.add_argument("--hardware", type=str, default="config/hardware/16x16.yaml",
                        help="16x16 -- the smaller of Experiment 1's two justified primary sizes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-iterations", type=int, default=40, help="Generous iteration budget per plan §6 Exp 2.")
    parser.add_argument("--window", type=int, default=5, help="Convergence criterion trailing window size.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Convergence criterion variance threshold.")
    parser.add_argument("--out", type=str, default=str(RAW_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = load_workload_yaml(str(_PROJECT_ROOT / args.workload) if not Path(args.workload).is_absolute() else args.workload)
    hardware = load_hardware_config(str(_PROJECT_ROOT / args.hardware) if not Path(args.hardware).is_absolute() else args.hardware)
    csv_path = Path(args.out) if Path(args.out).is_absolute() else (THIS_DIR / args.out)
    git_commit = logging_utils.current_git_commit(_PROJECT_ROOT)
    scratch_root = RESULTS_DIR / "raw" / "_scratch"

    for seed in args.seeds:
        print(f"[exp2] seed={seed} (n_iterations={args.n_iterations}) ...")
        problem = AttentionMappingProblem(
            workload, hardware, traffic_penalty_weight=0.01, failed_run_penalty_reward=-1000.0,
            scratch_dir=scratch_root / f"seed{seed}",
        )
        optimizer = _load_selected_optimizer()
        results = optimizer.run(problem, n_iterations=args.n_iterations, seed=seed)
        for i, r in enumerate(results):
            _log_eval_result(
                csv_path=csv_path, git_commit=git_commit, hardware=hardware,
                workload=workload, seed=seed, iteration=i, eval_result=r,
            )
        best = max(results, key=lambda r: r.reward)
        print(f"[exp2]   -> {len(results)} evals, best reward={best.reward:.3f}")

    print(f"[exp2] done. Raw results: {csv_path}")


if __name__ == "__main__":
    main()
