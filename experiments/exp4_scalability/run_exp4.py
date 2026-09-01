"""
Experiment 4 -- Scalability study (answers RQ3 directly)

Show how the optimizer's advantage over baselines, and its own overhead, scale
with sequence length and array size. See ../../docs/CODING_AGENT_PLAN.md §6
(Experiment 4) for the full spec and exp4_stats.md for the filled-in report.

**Data provenance -- read before assuming this script re-runs everything:**
Experiment 4's scalability trend spans 5 (seq_len, array) points: the 4 already
run for real in Experiment 3 (seq_len in {128,256} x array in {16x16,32x32}) plus
ONE new point (seq_len=512 x array=32x32). bayesian_opt and every baseline here
are fully deterministic given (workload, hardware, seed) -- re-running the 4
already-covered points would reproduce Experiment 3's numbers bit-for-bit at the
cost of real wall-clock time for zero new information (the same reasoning applied
when Experiment 2's epsilon was re-checked without re-simulating). So this script
only runs the ONE new cell by default; analyze_exp4.py merges this experiment's
own results/raw/exp4.csv with experiments/exp3_main_sweep/results/raw/exp3.csv
for the full 5-point trend, with each row's `experiment` column making the
provenance explicit either way.

**Two further (seq_len, array) points were probed and found infeasible before
this script was written** (config/workloads/exp4_seq512.yaml's own comment has
the exact measurements): seq_len=512 @ 16x16 times out entirely; seq_len=1024 @
32x32 FAILS. Neither is re-attempted as a full cell here -- a single-call probe
already establishes infeasibility, and running the full ~42-call baseline+
optimizer procedure against a config already shown infeasible would waste real
time for ~zero new information. This ceiling (max feasible seq_len roughly
doubles as array size doubles: 256 -> 512) is itself part of Experiment 4's
answer to RQ3, not a gap in the sweep.

Same fixed-budget-not-dynamic-early-stopping design as Experiment 3
(exp2_stats.md §8a): the optimizer runs a fixed iteration budget per seed;
common.optimizers.convergence is applied post-hoc to report convergence
iteration/wall-clock cost as a descriptive statistic, matching plan §6 Exp 4's
"optimizer wall-clock/sample count needed to converge" dependent variable.
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

EXPERIMENT_ID = "exp4"
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw" / f"{EXPERIMENT_ID}.csv"

# The one genuinely new (workload, hardware) cell -- see module docstring for why
# the other 4 trend points are reused from Experiment 3 rather than re-run.
DEFAULT_WORKLOAD_NAMES = ["exp4_seq512"]
DEFAULT_HARDWARE_NAMES = ["32x32"]

SELECTED_OPTIMIZER_YAML = _PROJECT_ROOT / "config" / "optimizers" / "selected.yaml"


def _load_selected_optimizer() -> BayesianOptOptimizer:
    with open(SELECTED_OPTIMIZER_YAML) as f:
        selected = yaml.safe_load(f)
    if selected["selected_optimizer"] != "bayesian_opt":
        raise ValueError(
            f"config/optimizers/selected.yaml names {selected['selected_optimizer']!r}, but "
            "run_exp4.py only knows how to instantiate bayesian_opt -- update this function "
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
                        help="Fixed optimizer iteration budget per seed -- matches Experiment 3's "
                             "budget so cost/quality are comparable across the merged trend.")
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
            print(f"[exp4] {workload_name} x {hw_name} ...")

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
                    print(f"[exp4]   {method}: status={result.status}" + (
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
                    print(f"[exp4]   best_fixed_per_stage: status=OK util={baselines.best_fixed_per_stage_combined.pe_utilization_pct:.2f}%")
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
                    print(f"[exp4]   best_fixed_per_stage: status=FAILED (no winner for {missing})")

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
                    conv_iter = convergence_iteration(best_reward_so_far(rewards))  # shared defaults (window=5, epsilon=0.1)
                    print(f"[exp4]   bayesian_opt seed={seed}: {len(results)} evals, "
                          f"best reward={max(rewards):.3f}, convergence_iteration={conv_iter}")

            except Exception as e:
                print(f"[exp4]   {workload_name} x {hw_name}: UNEXPECTED ERROR, skipping rest of this cell: {e}")

    print(f"[exp4] done. Raw results: {csv_path}")


if __name__ == "__main__":
    main()
