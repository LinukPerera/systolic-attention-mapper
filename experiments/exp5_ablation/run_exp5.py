"""
Experiment 5 -- Ablation study

Isolate the contribution of joint dataflow+tiling learning (the specific gap
claimed in the lit review) from tiling alone or dataflow alone. See
../../docs/CODING_AGENT_PLAN.md §6 (Experiment 5) for the full spec and
exp5_stats.md for the filled-in report.

**Three variants, plan §6 Exp 5:**
  - dataflow_only: common.optimizers.ablation.DataflowOnlyOptimizer -- learns only
    the dataflow mode, SRAM partition fixed at an equal three-way split.
  - tiling_only: common.optimizers.ablation.TilingOnlyOptimizer -- learns only the
    SRAM partition, dataflow fixed to "best-static" -- read directly from
    Experiment 3's real results (results/processed/exp3_optimizer_vs_baseline.csv's
    best_single_dataflow_method column) for the cell being ablated, NOT assumed or
    hardcoded (it happens to be `os_only` in every Experiment 3 cell, but this
    script looks it up rather than encoding that as a constant).
  - joint: this is exactly what common.optimizers.bayesian_opt.BayesianOptOptimizer
    (the selected optimizer) already does -- its Experiment 3 results are REUSED
    here, not re-run (bayesian_opt is fully deterministic given (workload,
    hardware, seed), so re-running would reproduce Experiment 3's numbers
    bit-for-bit at real wall-clock cost for zero new information — same reasoning
    as Experiment 4's data-provenance choice).

**Cells: 2 of Experiment 3's 4 real cells, not all 4** -- (seq128, 16x16) and
(seq256, 32x32), a deliberate, wall-clock-driven scope reduction (this doubles
Experiment 3's per-cell optimizer cost: 2 new variants x 5 seeds x 6 iterations
= 60 new real simulations per cell, on top of the ~72s/call cost of the seq256
cell). Documented, not silent, per this project's established practice.
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

import pandas as pd

from common import logging_utils
from common.metrics import compute_edp
from common.optimizers.ablation import DataflowOnlyOptimizer, TilingOnlyOptimizer
from common.optimizers.environment import AttentionMappingProblem
from common.simulator_interface import HardwareConfig, load_hardware_config
from common.workload_gen import AttentionWorkload, load_workload_yaml

EXPERIMENT_ID = "exp5"
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw" / f"{EXPERIMENT_ID}.csv"

EXP3_OPTIMIZER_VS_BASELINE_CSV = (
    _PROJECT_ROOT / "experiments" / "exp3_main_sweep" / "results" / "processed" / "exp3_optimizer_vs_baseline.csv"
)

# (workload YAML name, hardware name) -- 2 of Experiment 3's 4 real cells, see module docstring.
DEFAULT_CELLS = [("exp3_seq128", "16x16"), ("exp3_seq256", "32x32")]


def _cell_label(seq_len: int, array_rows: int) -> str:
    return f"seq{seq_len} @ {array_rows}x{array_rows}"


def _lookup_best_static_dataflow(cell_label: str) -> str:
    """Real lookup, not a hardcoded guess -- see module docstring."""
    df = pd.read_csv(EXP3_OPTIMIZER_VS_BASELINE_CSV)
    row = df[df["cell"] == cell_label]
    if row.empty:
        raise ValueError(
            f"No Experiment 3 data for cell {cell_label!r} in {EXP3_OPTIMIZER_VS_BASELINE_CSV} -- "
            "tiling_only needs a real best-static dataflow to fix, not an assumed one."
        )
    return row.iloc[0]["best_single_dataflow_method"].removesuffix("_only").upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=str, nargs="+", default=[f"{w}:{h}" for w, h in DEFAULT_CELLS],
                        help="workload_name:hardware_name pairs, e.g. exp3_seq128:16x16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-iterations", type=int, default=6, help="Matches Experiment 3's budget.")
    parser.add_argument("--out", type=str, default=str(RAW_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.out) if Path(args.out).is_absolute() else (THIS_DIR / args.out)
    git_commit = logging_utils.current_git_commit(_PROJECT_ROOT)
    scratch_root = RESULTS_DIR / "raw" / "_scratch"

    cells = [tuple(c.split(":")) for c in args.cells]

    for workload_name, hw_name in cells:
        workload = load_workload_yaml(str(_PROJECT_ROOT / "config" / "workloads" / f"{workload_name}.yaml"))
        hardware = load_hardware_config(str(_PROJECT_ROOT / "config" / "hardware" / f"{hw_name}.yaml"))
        cell_label = _cell_label(workload.seq_len, hardware.array_rows)
        fixed_dataflow = _lookup_best_static_dataflow(cell_label)
        print(f"[exp5] {cell_label} (tiling_only fixed dataflow: {fixed_dataflow}) ...")

        variants = {
            "dataflow_only": DataflowOnlyOptimizer(),
            "tiling_only": TilingOnlyOptimizer(fixed_dataflow_mode=fixed_dataflow),
        }

        for variant_name, optimizer in variants.items():
            try:
                for seed in args.seeds:
                    problem = AttentionMappingProblem(
                        workload, hardware, traffic_penalty_weight=0.01, failed_run_penalty_reward=-1000.0,
                        scratch_dir=scratch_root / workload_name / hw_name / variant_name / f"seed{seed}",
                    )
                    results = optimizer.run(problem, n_iterations=args.n_iterations, seed=seed)
                    for i, r in enumerate(results):
                        notes = {
                            "ifmap_frac": round(r.action.ifmap_frac, 4), "filter_frac": round(r.action.filter_frac, 4),
                            "ofmap_frac": round(r.action.ofmap_frac, 4), "reward": round(r.reward, 6),
                            "sim_notes": r.simulation.notes,
                        }
                        row = {
                            "run_id": f"{EXPERIMENT_ID}-{variant_name}-{workload.seq_len}-{hardware.name}-seed{seed}-iter{i}",
                            "experiment": EXPERIMENT_ID, "timestamp": datetime.now(timezone.utc).isoformat(),
                            "git_commit": git_commit, "optimizer": variant_name,
                            "array_rows": hardware.array_rows, "array_cols": hardware.array_cols,
                            "sram_kb": sum(hardware.sram_kb.values()), "dataflow_mode": r.action.dataflow_mode,
                            "seq_len": workload.seq_len, "hidden_dim": workload.hidden_dim, "num_heads": workload.num_heads,
                            "seed": seed, "iteration": i,
                            "pe_utilization_pct": r.simulation.pe_utilization_pct if r.simulation.status == "OK" else None,
                            "exec_cycles": r.simulation.exec_cycles if r.simulation.status == "OK" else None,
                            "sram_dram_traffic_bytes": r.simulation.sram_dram_traffic_bytes if r.simulation.status == "OK" else None,
                            "edp": compute_edp(r.simulation, energy_per_cycle_pj=hardware.energy_per_cycle_pj) if r.simulation.status == "OK" else None,
                            "wall_clock_s": r.simulation.wall_clock_s, "status": r.simulation.status, "notes": json.dumps(notes),
                        }
                        logging_utils.append_run_row(csv_path, row)
                    rewards = [r.reward for r in results]
                    print(f"[exp5]   {variant_name} seed={seed}: {len(results)} evals, best reward={max(rewards):.3f}")
            except Exception as e:
                print(f"[exp5]   {cell_label} x {variant_name}: UNEXPECTED ERROR, skipping rest: {e}")

    print(f"[exp5] done. Raw results: {csv_path}")


if __name__ == "__main__":
    main()
