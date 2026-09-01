"""
Experiment 3 -- Main sweep vs. baselines (core evidence chapter)

The central empirical claim -- learned optimizer vs. static heuristics, across
realistic workload diversity. See ../../docs/CODING_AGENT_PLAN.md §6 (Experiment 3)
for the full spec and exp3_stats.md for the filled-in report.

Grid: {seq_len in [128, 256]} x {array in [16x16, 32x32]} -- reduced from the
plan's example (seq_len up to 4096) after real, hand-timed evidence that
seq_len=512/1024 at head_dim=64 are infeasible at 16x16 (one timed out entirely,
one OOM-killed at 73% progress after 134s) -- see exp3_stats.md §3 and the
config/workloads/exp3_seq*.yaml files' own comments for the exact measurements.

For each (workload, array) cell: all four static baselines (via
common.baselines.static_heuristics.evaluate_all_baselines, shared with
run_exp1.py) run once each (deterministic), and the selected optimizer
(bayesian_opt, from config/optimizers/selected.yaml -- not re-decided) runs for
N_ITERATIONS across >=5 seeds. Per Experiment 2's finding
(exp2_stats.md §8a: the convergence criterion is fragile to early-trial
clustering and unsuitable for dynamically stopping a run), every optimizer seed
runs the SAME fixed, pre-timed iteration budget -- common.optimizers.convergence
is applied post-hoc to report each seed's convergence iteration as a descriptive
statistic, never to control how many real simulations actually execute.

Every row logged here comes from a real, individually-run containerized SCALE-Sim
simulation -- there is no synthetic/interpolated metric anywhere in this script
(plan ground rule 1). A cell where best_fixed_per_stage has no combinable result
(every dataflow candidate FAILED for some stage -- real infeasibility, precedent:
exp1_head64_validation.yaml's 8x8) is logged as FAILED and the sweep continues
(same fix applied in run_exp1.py).
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
from common.baselines.static_heuristics import evaluate_all_baselines
from common.metrics import compute_edp
from common.optimizers.bayesian_opt import BayesianOptOptimizer
from common.optimizers.convergence import best_reward_so_far, convergence_iteration
from common.optimizers.environment import AttentionMappingProblem
from common.simulator_interface import HardwareConfig, SimulationResult, load_hardware_config
from common.workload_gen import AttentionWorkload, load_workload_yaml

EXPERIMENT_ID = "exp3"
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw" / f"{EXPERIMENT_ID}.csv"

DEFAULT_WORKLOAD_NAMES = ["exp3_seq128", "exp3_seq256"]
DEFAULT_HARDWARE_NAMES = ["16x16", "32x32"]  # Experiment 1's two mechanistically-justified primary sizes

SELECTED_OPTIMIZER_YAML = _PROJECT_ROOT / "config" / "optimizers" / "selected.yaml"


def _load_selected_optimizer() -> BayesianOptOptimizer:
    with open(SELECTED_OPTIMIZER_YAML) as f:
        selected = yaml.safe_load(f)
    if selected["selected_optimizer"] != "bayesian_opt":
        raise ValueError(
            f"config/optimizers/selected.yaml names {selected['selected_optimizer']!r}, but "
            "run_exp3.py only knows how to instantiate bayesian_opt -- update this function "
            "(not the selection) if Experiment 0's choice changes."
        )
    hp_path = _PROJECT_ROOT / selected["hyperparameters_file"]
    with open(hp_path) as f:
        hp = yaml.safe_load(f) or {}
    return BayesianOptOptimizer(**hp)


def _base_row(
    *, method: str, hardware: HardwareConfig, workload: AttentionWorkload,
    dataflow_mode: str, seed: int, iteration: int, result: SimulationResult,
    git_commit: str, notes: dict,
) -> dict:
    return {
        "run_id": f"{EXPERIMENT_ID}-{method}-{workload.seq_len}-{hardware.name}-{dataflow_mode}-seed{seed}-iter{iteration}",
        "experiment": EXPERIMENT_ID, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "optimizer": method,
        "array_rows": hardware.array_rows, "array_cols": hardware.array_cols,
        "sram_kb": sum(hardware.sram_kb.values()), "dataflow_mode": dataflow_mode,
        "seq_len": workload.seq_len, "hidden_dim": workload.hidden_dim, "num_heads": workload.num_heads,
        "seed": seed, "iteration": iteration,
        "pe_utilization_pct": result.pe_utilization_pct if result.status == "OK" else None,
        "exec_cycles": result.exec_cycles if result.status == "OK" else None,
        "sram_dram_traffic_bytes": result.sram_dram_traffic_bytes if result.status == "OK" else None,
        "edp": compute_edp(result, energy_per_cycle_pj=hardware.energy_per_cycle_pj) if result.status == "OK" else None,
        "wall_clock_s": result.wall_clock_s, "status": result.status, "notes": json.dumps(notes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-names", type=str, nargs="+", default=DEFAULT_WORKLOAD_NAMES)
    parser.add_argument("--hardware-names", type=str, nargs="+", default=DEFAULT_HARDWARE_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-iterations", type=int, default=6,
                        help="Fixed optimizer iteration budget per seed -- reduced from Exp 0's 15 "
                             "given the {workload}x{array} grid multiplies total real simulator calls.")
    parser.add_argument("--out", type=str, default=str(RAW_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.out) if Path(args.out).is_absolute() else (THIS_DIR / args.out)
    git_commit = logging_utils.current_git_commit(_PROJECT_ROOT)
    scratch_root = RESULTS_DIR / "raw" / "_scratch"

    for workload_name in args.workload_names:
        workload = load_workload_yaml(str(_PROJECT_ROOT / "config" / "workloads" / f"{workload_name}.yaml"))
        for hw_name in args.hardware_names:
            hardware = load_hardware_config(str(_PROJECT_ROOT / "config" / "hardware" / f"{hw_name}.yaml"))
            cell_scratch = scratch_root / workload_name / hw_name
            print(f"[exp3] {workload_name} x {hw_name} ...")

            try:
                # --- Baselines ---
                baselines = evaluate_all_baselines(workload, hardware, scratch_dir=cell_scratch / "baselines")
                for method, result, dataflow_mode in [
                    ("ws_only", baselines.ws_only, "WS"),
                    ("os_only", baselines.os_only, "OS"),
                    ("is_only", baselines.is_only, "IS"),
                ]:
                    logging_utils.append_run_row(csv_path, _base_row(
                        method=method, hardware=hardware, workload=workload, dataflow_mode=dataflow_mode,
                        seed=-1, iteration=0, result=result, git_commit=git_commit, notes={"sim_notes": result.notes},
                    ))
                    print(f"[exp3]   {method}: status={result.status}" + (
                        f" util={result.pe_utilization_pct:.2f}%" if result.status == "OK" else ""))

                for attempt in baselines.best_fixed_per_stage.all_attempts:
                    logging_utils.append_run_row(csv_path, _base_row(
                        method="best_fixed_per_stage_search", hardware=hardware, workload=workload,
                        dataflow_mode=attempt.dataflow_mode, seed=-1, iteration=0, result=attempt.result,
                        git_commit=git_commit, notes={"stage": attempt.stage_name, "sim_notes": attempt.result.notes},
                    ))

                if baselines.best_fixed_per_stage_combined is not None:
                    per_stage_choice = {s: m.dataflow_mode for s, m in baselines.best_fixed_per_stage.winners.items()}
                    logging_utils.append_run_row(csv_path, _base_row(
                        method="best_fixed_per_stage", hardware=hardware, workload=workload, dataflow_mode="mixed",
                        seed=-1, iteration=0, result=baselines.best_fixed_per_stage_combined, git_commit=git_commit,
                        notes={"per_stage_dataflow": per_stage_choice},
                    ))
                    print(f"[exp3]   best_fixed_per_stage: status=OK util={baselines.best_fixed_per_stage_combined.pe_utilization_pct:.2f}%")
                else:
                    missing = sorted({a.stage_name for a in baselines.best_fixed_per_stage.all_attempts}
                                      - set(baselines.best_fixed_per_stage.winners))
                    failed = SimulationResult(
                        exec_cycles=0, pe_utilization_pct=0.0, sram_dram_traffic_bytes=0,
                        wall_clock_s=sum(a.result.wall_clock_s for a in baselines.best_fixed_per_stage.all_attempts),
                        status="FAILED", notes=f"stage(s) with zero successful dataflow candidates: {missing}",
                    )
                    logging_utils.append_run_row(csv_path, _base_row(
                        method="best_fixed_per_stage", hardware=hardware, workload=workload, dataflow_mode="mixed",
                        seed=-1, iteration=0, result=failed, git_commit=git_commit, notes={"missing_stages": missing},
                    ))
                    print(f"[exp3]   best_fixed_per_stage: status=FAILED (no winner for {missing})")

                # --- Optimizer: fixed budget per seed (see module docstring for why not dynamic) ---
                for seed in args.seeds:
                    problem = AttentionMappingProblem(
                        workload, hardware, traffic_penalty_weight=0.01, failed_run_penalty_reward=-1000.0,
                        scratch_dir=cell_scratch / "bayesian_opt" / f"seed{seed}",
                    )
                    optimizer = _load_selected_optimizer()
                    results = optimizer.run(problem, n_iterations=args.n_iterations, seed=seed)
                    for i, r in enumerate(results):
                        notes = {
                            "ifmap_frac": round(r.action.ifmap_frac, 4), "filter_frac": round(r.action.filter_frac, 4),
                            "ofmap_frac": round(r.action.ofmap_frac, 4), "reward": round(r.reward, 6),
                            "sim_notes": r.simulation.notes,
                        }
                        logging_utils.append_run_row(csv_path, _base_row(
                            method="bayesian_opt", hardware=hardware, workload=workload,
                            dataflow_mode=r.action.dataflow_mode, seed=seed, iteration=i,
                            result=r.simulation, git_commit=git_commit, notes=notes,
                        ))
                    rewards = [r.reward for r in results]
                    conv_iter = convergence_iteration(best_reward_so_far(rewards))  # uses shared defaults (window=5, epsilon=0.1)
                    print(f"[exp3]   bayesian_opt seed={seed}: {len(results)} evals, "
                          f"best reward={max(rewards):.3f}, convergence_iteration={conv_iter}")

            except Exception as e:
                print(f"[exp3]   {workload_name} x {hw_name}: UNEXPECTED ERROR, skipping rest of this cell: {e}")

    print(f"[exp3] done. Raw results: {csv_path}")


if __name__ == "__main__":
    main()
