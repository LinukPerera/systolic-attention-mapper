"""
Experiment 1 -- Array-size sensitivity study

Justify the primary working array sizes (16x16, 32x32) the way the sample thesis
justifies its mesh choice, by actually running the static-heuristic baselines
(WS-only, OS-only, IS-only, best-fixed-per-stage) across the array-size sweep and
seeing where utilization stops improving.

See ../../docs/CODING_AGENT_PLAN.md §6 (Experiment 1) for the full spec and
exp1_stats.md for the filled-in report.

Every row logged here comes from a real, individually-run containerized SCALE-Sim
simulation -- there is no synthetic/interpolated metric anywhere in this script
(plan ground rule 1). All four baselines are deterministic (no seed), per plan
ground rule 3 -- logged with seed=-1, matching exp0's brute-force convention.
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

from common import logging_utils
from common.baselines.static_heuristics import (
    best_fixed_per_stage_mapping,
    is_only_mapping,
    os_only_mapping,
    ws_only_mapping,
)
from common.metrics import compute_edp
from common.simulator_interface import (
    HardwareConfig,
    SimulationResult,
    combine_simulation_results,
    load_hardware_config,
    run_scale_sim,
    write_scale_sim_inputs,
)
from common.workload_gen import AttentionWorkload, gemm_shapes_to_topology_rows, load_workload_yaml, workload_to_gemm_shapes

EXPERIMENT_ID = "exp1"
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw" / f"{EXPERIMENT_ID}.csv"

# Sweep order matches CODING_AGENT_PLAN.md §6 Experiment 1's variables list.
HARDWARE_NAMES = ["8x8", "16x16", "32x32", "64x64", "128x128"]

SINGLE_DATAFLOW_BASELINES = {
    "ws_only": ws_only_mapping,
    "os_only": os_only_mapping,
    "is_only": is_only_mapping,
}


def _log_row(
    *, csv_path: Path, git_commit: str, method: str, hardware: HardwareConfig,
    workload: AttentionWorkload, dataflow_mode: str, result: SimulationResult,
    energy_per_cycle_pj: float, notes_extra: dict | None = None,
) -> None:
    notes = {"sim_notes": result.notes}
    if notes_extra:
        notes.update(notes_extra)
    row = {
        "run_id": f"{EXPERIMENT_ID}-{method}-{hardware.name}-{dataflow_mode}",
        "experiment": EXPERIMENT_ID, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "optimizer": method,
        "array_rows": hardware.array_rows, "array_cols": hardware.array_cols,
        "sram_kb": sum(hardware.sram_kb.values()), "dataflow_mode": dataflow_mode,
        "seq_len": workload.seq_len, "hidden_dim": workload.hidden_dim, "num_heads": workload.num_heads,
        "seed": -1, "iteration": 0,
        "pe_utilization_pct": result.pe_utilization_pct if result.status == "OK" else None,
        "exec_cycles": result.exec_cycles if result.status == "OK" else None,
        "sram_dram_traffic_bytes": result.sram_dram_traffic_bytes if result.status == "OK" else None,
        "edp": compute_edp(result, energy_per_cycle_pj=energy_per_cycle_pj) if result.status == "OK" else None,
        "wall_clock_s": result.wall_clock_s, "status": result.status, "notes": json.dumps(notes),
    }
    logging_utils.append_run_row(csv_path, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=str, default="config/workloads/exp0_pilot.yaml",
                        help="Reused from Experiment 0: already verified at ~2s/simulation call "
                             "(a literal seq_len=512 workload would cost ~90 calls x ~178s, "
                             "4+ hours -- see exp0_stats.md §3).")
    parser.add_argument("--hardware-names", type=str, nargs="+", default=HARDWARE_NAMES)
    parser.add_argument("--out", type=str, default=str(RAW_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = load_workload_yaml(str(_PROJECT_ROOT / args.workload) if not Path(args.workload).is_absolute() else args.workload)
    csv_path = Path(args.out)
    git_commit = logging_utils.current_git_commit(_PROJECT_ROOT)
    scratch_root = RESULTS_DIR / "raw" / "_scratch"

    shapes = workload_to_gemm_shapes(workload)
    topology_rows = gemm_shapes_to_topology_rows(shapes)

    for hw_name in args.hardware_names:
        hardware = load_hardware_config(str(_PROJECT_ROOT / "config" / "hardware" / f"{hw_name}.yaml"))
        print(f"[exp1] {hw_name} ...")

        # --- Single-dataflow baselines: one whole-workload simulation each ---
        for method, mapping_fn in SINGLE_DATAFLOW_BASELINES.items():
            mapping = mapping_fn(workload, str(_PROJECT_ROOT / "config" / "hardware" / f"{hw_name}.yaml"))
            run_dir = scratch_root / hw_name / method
            topology_path, layout_path, config_path = write_scale_sim_inputs(
                topology_rows, hardware, mapping.dataflow_mode, run_dir,
            )
            result = run_scale_sim(
                topology_path, layout_path, config_path,
                run_name=f"{hardware.name}_{mapping.dataflow_mode}", hardware=hardware, output_dir=run_dir / "output",
            )
            _log_row(
                csv_path=csv_path, git_commit=git_commit, method=method, hardware=hardware,
                workload=workload, dataflow_mode=mapping.dataflow_mode, result=result,
                energy_per_cycle_pj=hardware.energy_per_cycle_pj,
            )
            print(f"[exp1]   {method}: status={result.status} "
                  f"util={result.pe_utilization_pct:.2f}%" if result.status == "OK" else f"[exp1]   {method}: FAILED")

        # --- best_fixed_per_stage: per-stage search (logged in full) + combined aggregate ---
        bfps = best_fixed_per_stage_mapping(workload, hardware, scratch_dir=scratch_root / hw_name / "best_fixed_per_stage_search")
        for attempt in bfps.all_attempts:
            _log_row(
                csv_path=csv_path, git_commit=git_commit, method="best_fixed_per_stage_search", hardware=hardware,
                workload=workload, dataflow_mode=attempt.dataflow_mode, result=attempt.result,
                energy_per_cycle_pj=hardware.energy_per_cycle_pj, notes_extra={"stage": attempt.stage_name},
            )
        combined = combine_simulation_results(list(bfps.winner_results.values()))
        per_stage_choice = {stage: m.dataflow_mode for stage, m in bfps.winners.items()}
        _log_row(
            csv_path=csv_path, git_commit=git_commit, method="best_fixed_per_stage", hardware=hardware,
            workload=workload, dataflow_mode="mixed", result=combined,
            energy_per_cycle_pj=hardware.energy_per_cycle_pj, notes_extra={"per_stage_dataflow": per_stage_choice},
        )
        print(f"[exp1]   best_fixed_per_stage: status={combined.status} "
              f"util={combined.pe_utilization_pct:.2f}% per_stage={per_stage_choice}" if combined.status == "OK"
              else "[exp1]   best_fixed_per_stage: FAILED")

    print(f"[exp1] done. Raw results: {csv_path}")


if __name__ == "__main__":
    main()
