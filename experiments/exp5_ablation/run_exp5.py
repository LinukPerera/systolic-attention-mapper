"""
Experiment 5 -- Ablation study

Isolate the contribution of joint dataflow+tiling learning (the specific gap
claimed in the lit review) from tiling alone or dataflow alone. See
../../docs/CODING_AGENT_PLAN.md §6 (Experiment 5) for the full spec and
exp5_stats.md for the filled-in report.

**Per-stage, revised 2026-09-03** (was one global dataflow/tiling choice -- see
common/optimizers/environment.py's module docstring, and exp3_stats.md §11 for
why this re-run matters: Experiment 3's per-stage re-run closed the gap between
the optimizer and `best_fixed_per_stage`, which changes what this experiment's
`dataflow_only` question is actually testing). Both ablation variants now act
per GEMM stage, matching `common.optimizers.environment`'s STAGE_NAMES:

  - dataflow_only: common.optimizers.ablation.DataflowOnlyOptimizer -- learns
    each stage's dataflow mode independently, SRAM partition fixed at an equal
    three-way split per stage.
  - tiling_only: common.optimizers.ablation.TilingOnlyOptimizer -- learns each
    stage's SRAM partition independently, dataflow fixed **per stage** to that
    stage's own real best-static choice -- read directly from this project's
    already-run per-stage Experiment 3 (`results/raw/exp3_perstage.csv`'s
    `best_fixed_per_stage` row's `per_stage_dataflow` note for the matching
    cell), NOT assumed or hardcoded. This is a real change from the original
    (3-dim) exp5 run, which fixed one global dataflow (`best_single_dataflow`,
    always os_only) for tiling_only -- now that dataflow can vary per stage,
    "best-static dataflow" is more meaningfully `best_fixed_per_stage`'s own
    per-stage answer, not the single-global-dataflow baseline.
  - joint: this is exactly what the selected optimizer
    (config/optimizers/selected.yaml, `a2c` as of 2026-09-02) already does --
    its Experiment 3 per-stage results are REUSED here from
    results/raw/exp3_perstage.csv, not re-run. **Caveat, real and worth stating
    explicitly (unlike bayesian_opt, which the original exp5 run reused under a
    strict-determinism argument):** stable-baselines3's A2C is not guaranteed
    bit-for-bit deterministic even given a fixed seed (same caveat already
    documented in exp0_stats.md/exp3_stats.md's reproduction sections) -- reused
    here anyway because (a) it's the same machine/library-versions/environment
    that produced exp3_perstage.csv moments earlier, well within where that
    caveat is a real but minor risk, not a cross-machine reproduction claim, and
    (b) re-running would cost another full 2-cell x 5-seed real sweep for a
    number this project's own data-provenance principle says should already be
    available -- documented as a [LIMITATION] in exp5_stats.md, not hidden.

**Cells: 2 of Experiment 3's 4 real cells, not all 4** -- (seq128, 16x16) and
(seq256, 32x32), same deliberate, wall-clock-driven scope reduction as the
original run (unchanged: this doubles Experiment 3's per-cell optimizer cost
for the 2 variants actually run here).
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
from common.optimizers.environment import STAGE_NAMES, AttentionMappingProblem
from common.simulator_interface import HardwareConfig, SimulationResult, load_hardware_config
from common.workload_gen import AttentionWorkload, load_workload_yaml

EXPERIMENT_ID = "exp5"
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw" / f"{EXPERIMENT_ID}.csv"

# Points at the per-stage Experiment 3 re-run by default (this script's own real
# purpose, 2026-09-03) -- override with --exp3-raw-csv to point at the original
# 3-dim exp3.csv instead (not expected to be needed).
DEFAULT_EXP3_RAW_CSV = (
    _PROJECT_ROOT / "experiments" / "exp3_main_sweep" / "results" / "raw" / "exp3_perstage.csv"
)

# (workload YAML name, hardware name) -- 2 of Experiment 3's 4 real cells, see module docstring.
DEFAULT_CELLS = [("exp3_seq128", "16x16"), ("exp3_seq256", "32x32")]


def _cell_label(seq_len: int, array_rows: int) -> str:
    return f"seq{seq_len} @ {array_rows}x{array_rows}"


def _lookup_best_static_per_stage_dataflow(exp3_df: pd.DataFrame, cell_label: str) -> dict[str, str]:
    """Real lookup of best_fixed_per_stage's own per-stage dataflow choice for
    this cell, not an assumed/hardcoded one -- see module docstring."""
    exp3_df = exp3_df.copy()
    exp3_df["cell"] = exp3_df.apply(lambda r: _cell_label(r["seq_len"], r["array_rows"]), axis=1)
    row = exp3_df[(exp3_df["cell"] == cell_label) & (exp3_df["optimizer"] == "best_fixed_per_stage")]
    if row.empty:
        raise ValueError(
            f"No best_fixed_per_stage row for cell {cell_label!r} in the Experiment 3 CSV -- "
            "tiling_only needs a real per-stage best-static dataflow to fix, not an assumed one."
        )
    notes = json.loads(row.iloc[0]["notes"])
    per_stage_dataflow = notes.get("per_stage_dataflow")
    if not per_stage_dataflow or set(per_stage_dataflow) != set(STAGE_NAMES):
        raise ValueError(
            f"best_fixed_per_stage row for cell {cell_label!r} has no usable per_stage_dataflow "
            f"note (got {notes!r}) -- cannot fix tiling_only's dataflow per stage."
        )
    return per_stage_dataflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=str, nargs="+", default=[f"{w}:{h}" for w, h in DEFAULT_CELLS],
                        help="workload_name:hardware_name pairs, e.g. exp3_seq128:16x16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-iterations", type=int, default=6, help="Matches Experiment 3's budget.")
    parser.add_argument("--out", type=str, default=str(RAW_CSV))
    parser.add_argument("--exp3-raw-csv", type=str, default=str(DEFAULT_EXP3_RAW_CSV))
    return parser.parse_args()


def _base_row(
    *, method: str, hardware: HardwareConfig, workload: AttentionWorkload,
    dataflow_summary: str, seed: int, iteration: int, result: SimulationResult,
    git_commit: str, notes: dict,
) -> dict:
    return {
        "run_id": f"{EXPERIMENT_ID}-{method}-{workload.seq_len}-{hardware.name}-{dataflow_summary}-seed{seed}-iter{iteration}",
        "experiment": EXPERIMENT_ID, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "optimizer": method,
        "array_rows": hardware.array_rows, "array_cols": hardware.array_cols,
        "sram_kb": sum(hardware.sram_kb.values()), "dataflow_mode": dataflow_summary,
        "seq_len": workload.seq_len, "hidden_dim": workload.hidden_dim, "num_heads": workload.num_heads,
        "seed": seed, "iteration": iteration,
        "pe_utilization_pct": result.pe_utilization_pct if result.status == "OK" else None,
        "exec_cycles": result.exec_cycles if result.status == "OK" else None,
        "sram_dram_traffic_bytes": result.sram_dram_traffic_bytes if result.status == "OK" else None,
        "edp": compute_edp(result, energy_per_cycle_pj=hardware.energy_per_cycle_pj) if result.status == "OK" else None,
        "wall_clock_s": result.wall_clock_s, "status": result.status, "notes": json.dumps(notes),
    }


def main() -> None:
    args = parse_args()
    csv_path = Path(args.out) if Path(args.out).is_absolute() else (THIS_DIR / args.out)
    git_commit = logging_utils.current_git_commit(_PROJECT_ROOT)
    scratch_root = RESULTS_DIR / "raw" / "_scratch"
    exp3_df = pd.read_csv(args.exp3_raw_csv)

    cells = [tuple(c.split(":")) for c in args.cells]

    for workload_name, hw_name in cells:
        workload = load_workload_yaml(str(_PROJECT_ROOT / "config" / "workloads" / f"{workload_name}.yaml"))
        hardware = load_hardware_config(str(_PROJECT_ROOT / "config" / "hardware" / f"{hw_name}.yaml"))
        cell_label = _cell_label(workload.seq_len, hardware.array_rows)
        fixed_dataflow_modes = _lookup_best_static_per_stage_dataflow(exp3_df, cell_label)
        print(f"[exp5] {cell_label} (tiling_only fixed per-stage dataflow: {fixed_dataflow_modes}) ...")

        variants = {
            "dataflow_only": DataflowOnlyOptimizer(),
            "tiling_only": TilingOnlyOptimizer(fixed_dataflow_modes=fixed_dataflow_modes),
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
                        per_stage = {
                            name: {
                                "dataflow_mode": sa.dataflow_mode,
                                "ifmap_frac": round(sa.ifmap_frac, 4), "filter_frac": round(sa.filter_frac, 4),
                                "ofmap_frac": round(sa.ofmap_frac, 4),
                            }
                            for name, sa in r.action.stages.items()
                        }
                        dataflow_summary = "-".join(r.action.stages[name].dataflow_mode for name in STAGE_NAMES)
                        notes = {"stages": per_stage, "reward": round(r.reward, 6), "sim_notes": r.simulation.notes}
                        logging_utils.append_run_row(csv_path, _base_row(
                            method=variant_name, hardware=hardware, workload=workload,
                            dataflow_summary=dataflow_summary, seed=seed, iteration=i,
                            result=r.simulation, git_commit=git_commit, notes=notes,
                        ))
                    rewards = [r.reward for r in results]
                    print(f"[exp5]   {variant_name} seed={seed}: {len(results)} evals, best reward={max(rewards):.3f}")
            except Exception as e:
                print(f"[exp5]   {cell_label} x {variant_name}: UNEXPECTED ERROR, skipping rest: {e}")

    print(f"[exp5] done. Raw results: {csv_path}")


if __name__ == "__main__":
    main()
