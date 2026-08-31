"""
Experiment 0 -- Optimizer bake-off

Choose which optimizer (A2C [substituting for REINFORCE -- see
common/optimizers/a2c.py], PPO, evolutionary/GA, Bayesian optimization) is used for
all subsequent experiments, on stability + scalability grounds, not assumption.

See ../../docs/CODING_AGENT_PLAN.md §6 (Experiment 0) for the full spec and
exp0_stats.md for the filled-in report.

Every EvalResult logged here comes from a real, individually-run containerized
SCALE-Sim simulation via common.optimizers.environment.AttentionMappingProblem --
there is no synthetic/interpolated reward anywhere in this script (plan ground
rule 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Bootstrap: allow running this file directly (`python experiments/exp0_.../run_exp0.py`)
# without the project root already on sys.path — common/ is a plain (non-installed)
# package, and a directly-invoked script only gets its own directory on sys.path by
# default. Every run_expN.py needs this same three-line header.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from common import logging_utils
from common.metrics import compute_edp
from common.optimizers.a2c import A2COptimizer
from common.optimizers.bayesian_opt import BayesianOptOptimizer
from common.optimizers.brute_force import brute_force_search
from common.optimizers.environment import AttentionMappingProblem, EvalResult
from common.optimizers.evolutionary import EvolutionaryOptimizer
from common.optimizers.ppo import PPOOptimizer
from common.simulator_interface import load_hardware_config
from common.workload_gen import load_workload_yaml

EXPERIMENT_ID = "exp0"
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw" / f"{EXPERIMENT_ID}.csv"

OPTIMIZER_YAML_DIR = _PROJECT_ROOT / "config" / "optimizers"

OPTIMIZER_BUILDERS = {
    "a2c": lambda cfg: A2COptimizer(**cfg),
    "ppo": lambda cfg: PPOOptimizer(**cfg),
    "evolutionary": lambda cfg: EvolutionaryOptimizer(**cfg),
    "bayesian_opt": lambda cfg: BayesianOptOptimizer(**(cfg or {})),
}


def _load_optimizer_config(name: str) -> dict:
    path = OPTIMIZER_YAML_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _log_eval_result(
    *, csv_path: Path, experiment: str, git_commit: str, optimizer: str,
    hardware, workload, seed: int, iteration: int, eval_result: EvalResult,
) -> None:
    """Append one EvalResult as one raw CSV row, per CODING_AGENT_PLAN.md §4's schema.

    The actual SRAM partition fractions searched (ifmap/filter/ofmap_frac) aren't
    columns in the fixed schema -- they're recorded in `notes` (machine-parseable
    JSON) rather than dropped, so every action's full detail stays traceable back
    from the CSV (plan §3: no silent loss of information).
    """
    sim = eval_result.simulation
    action = eval_result.action
    notes = json.dumps({
        "ifmap_frac": round(action.ifmap_frac, 4), "filter_frac": round(action.filter_frac, 4),
        "ofmap_frac": round(action.ofmap_frac, 4), "reward": round(eval_result.reward, 6),
        "sim_notes": sim.notes,
    })
    row = {
        "run_id": f"{experiment}-{optimizer}-seed{seed}-iter{iteration}",
        "experiment": experiment, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "optimizer": optimizer,
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
    parser.add_argument("--workload", type=str, default="config/workloads/exp0_pilot.yaml")
    parser.add_argument("--hardware", type=str, default="config/hardware/16x16.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-iterations", type=int, default=15, help="Iteration budget per optimizer per seed.")
    parser.add_argument("--sram-grid-divisions", type=int, default=3, help="Brute-force ground-truth grid coarseness.")
    parser.add_argument("--out", type=str, default=str(RAW_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = load_workload_yaml(str(_PROJECT_ROOT / args.workload) if not Path(args.workload).is_absolute() else args.workload)
    hardware = load_hardware_config(str(_PROJECT_ROOT / args.hardware) if not Path(args.hardware).is_absolute() else args.hardware)
    csv_path = Path(args.out)
    git_commit = logging_utils.current_git_commit(_PROJECT_ROOT)

    scratch_root = RESULTS_DIR / "raw" / "_scratch"

    # --- Brute-force ground truth (deterministic, no seed) ---
    print(f"[exp0] brute-force ground truth (sram_grid_divisions={args.sram_grid_divisions}) ...")
    bf_problem = AttentionMappingProblem(
        workload, hardware, traffic_penalty_weight=0.01, failed_run_penalty_reward=-1000.0,
        scratch_dir=scratch_root / "brute_force",
    )
    bf_results = brute_force_search(bf_problem, sram_grid_divisions=args.sram_grid_divisions)
    for i, r in enumerate(bf_results):
        _log_eval_result(
            csv_path=csv_path, experiment=EXPERIMENT_ID, git_commit=git_commit, optimizer="brute_force",
            hardware=hardware, workload=workload, seed=-1, iteration=i, eval_result=r,
        )
    best_bf = max(bf_results, key=lambda r: r.reward)
    print(f"[exp0] brute-force: {len(bf_results)} evals, best reward={best_bf.reward:.3f} "
          f"(dataflow={best_bf.action.dataflow_mode})")

    # --- Each candidate optimizer, each seed ---
    for opt_name, builder in OPTIMIZER_BUILDERS.items():
        cfg = _load_optimizer_config(opt_name)
        for seed in args.seeds:
            print(f"[exp0] {opt_name} seed={seed} (n_iterations={args.n_iterations}) ...")
            problem = AttentionMappingProblem(
                workload, hardware, traffic_penalty_weight=0.01, failed_run_penalty_reward=-1000.0,
                scratch_dir=scratch_root / opt_name / f"seed{seed}",
            )
            optimizer = builder(cfg)
            results = optimizer.run(problem, n_iterations=args.n_iterations, seed=seed)
            for i, r in enumerate(results):
                _log_eval_result(
                    csv_path=csv_path, experiment=EXPERIMENT_ID, git_commit=git_commit, optimizer=opt_name,
                    hardware=hardware, workload=workload, seed=seed, iteration=i, eval_result=r,
                )
            best = max(results, key=lambda r: r.reward)
            print(f"[exp0]   -> {len(results)} evals, best reward={best.reward:.3f}")

    print(f"[exp0] done. Raw results: {csv_path}")


if __name__ == "__main__":
    main()
