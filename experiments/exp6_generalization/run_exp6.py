"""
Experiment 6 -- Generalization / transfer test (zero-shot action transfer)

**Scope narrowing from `CODING_AGENT_PLAN.md` §Experiment 6, deliberate and
disclosed** (see `docs/NEXT_INSTRUCTIONS.md` §0 for the full rationale, 2026-09-03):
the plan's literal spec ("train on a subset of (seq_len, head_dim), evaluate
zero-shot on held-out combinations") requires a policy conditioned on workload
parameters -- a new observation-space/training-loop change. This experiment
instead answers the same scientific question (does a learned mapping generalize
to unseen workload shapes?) via **zero-shot action transfer**: take an
already-trained per-stage `Action` (unmodified) and apply it to a new workload,
measuring the real simulated cost of that mismatch against (a) the static
per-stage baseline and (b) a freshly-optimized ceiling for that new workload.
Zero architecture changes; reuses `AttentionMappingProblem.evaluate()` and
`common.baselines.static_heuristics` exactly as every prior experiment does.

**Hypothesis (not a generic train/test split) -- see `docs/NEXT_INSTRUCTIONS.md`
§1 for the full derivation:** Experiment 1 established that PE utilization is
governed by `head_dim` relative to `array_size` (the "knee"). This experiment
asks whether a per-stage mapping learned in one utilization regime (source:
`seq256@32x32`, `head_dim=64` -- above-knee, `array_size=32 < head_dim=64`)
transfers to a workload in the **same** regime (held-out cell A, `head_dim=48`,
still above-knee) better than to one in the **opposite** regime (held-out cell
B, `head_dim=24`, below-knee, `array_size=32 > head_dim=24`).

**Source actions (already computed, zero new training cost):** the 5 real
per-stage `a2c` actions already logged for `seq256@32x32` in
`experiments/exp3_main_sweep/results/raw/exp3_perstage.csv` (one per seed, the
best-reward iteration of each seed's run) -- pulled directly from that CSV, not
retrained.

**Role tagging, adapted from `docs/NEXT_INSTRUCTIONS.md`'s spec:** that doc
asks for a `role` column (`baseline_fixed_per_stage`/`baseline_single_dataflow`/
`zero_shot_transfer`/`fresh_ceiling`). `common/logging_utils.py`'s `RAW_COLUMNS`
schema is fixed and shared by every experiment's raw CSV (adding a column would
mean changing that schema project-wide for one experiment's convenience). This
script instead encodes role via the existing `optimizer` column, exactly the
convention every prior experiment already uses for non-learned methods
(`ws_only`/`os_only`/`is_only`/`best_fixed_per_stage`/`best_fixed_per_stage_search`
unchanged; two new values added, `zero_shot_transfer` and `fresh_ceiling`) --
the same distinguishing information, no schema change. `analyze_exp6.py` groups
`{ws_only, os_only, is_only}` as "baseline_single_dataflow" exactly as
`analyze_exp3.py`/`analyze_exp4.py` already do via their own `SINGLE_DATAFLOW_METHODS`
constant.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml

from common import logging_utils
from common.baselines.static_heuristics import evaluate_all_baselines
from common.metrics import compute_edp
from common.optimizers.base import DataflowTilingOptimizer
from common.optimizers.environment import STAGE_NAMES, Action, AttentionMappingProblem, StageAction
from common.simulator_interface import HardwareConfig, SimulationResult, load_hardware_config
from common.workload_gen import AttentionWorkload, load_workload_yaml

EXPERIMENT_ID = "exp6"
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw" / f"{EXPERIMENT_ID}.csv"

# Source: Experiment 3's per-stage re-run, seq256@32x32 (head_dim=64, above-knee --
# array_size=32 < head_dim=64). See module docstring.
SOURCE_EXP3_CSV = (
    _PROJECT_ROOT / "experiments" / "exp3_main_sweep" / "results" / "raw" / "exp3_perstage.csv"
)
SOURCE_SEQ_LEN = 256
SOURCE_ARRAY_ROWS = 32
SOURCE_OPTIMIZER = "a2c"  # config/optimizers/selected.yaml's selection as of this experiment's design (exp0_stats.md §11)

# (workload YAML name, hardware name) -- both held-out cells, same array/seq_len as
# the source, head_dim the sole manipulated variable (docs/NEXT_INSTRUCTIONS.md §1).
DEFAULT_CELLS = [("exp6_heldout_a_hd48", "32x32"), ("exp6_heldout_b_hd24", "32x32")]

SELECTED_OPTIMIZER_YAML = _PROJECT_ROOT / "config" / "optimizers" / "selected.yaml"


def _load_selected_optimizer() -> tuple[str, DataflowTilingOptimizer]:
    """Same as run_exp3.py/run_exp4.py's own version -- kept in sync deliberately,
    not shared, per this project's established self-contained-script convention."""
    with open(SELECTED_OPTIMIZER_YAML) as f:
        selected = yaml.safe_load(f)
    module = importlib.import_module(selected["module"])
    cls = getattr(module, selected["class"])
    hp_path = _PROJECT_ROOT / selected["hyperparameters_file"]
    with open(hp_path) as f:
        hp = yaml.safe_load(f) or {}
    return selected["selected_optimizer"], cls(**hp)


def _load_source_actions(exp3_csv_path: str) -> dict[int, Action]:
    """Pull the 5 real best-per-seed per-stage actions for the source cell
    (seq256@32x32, SOURCE_OPTIMIZER) directly from Experiment 3's raw CSV --
    the exact actions already trained and logged there, not retrained here.

    Returns:
        {seed: Action} for each of the 5 source seeds.

    Raises:
        ValueError: If the source CSV has no usable rows for the expected cell/optimizer.
    """
    df = pd.read_csv(exp3_csv_path)
    sub = df[(df["seq_len"] == SOURCE_SEQ_LEN) & (df["array_rows"] == SOURCE_ARRAY_ROWS)
             & (df["optimizer"] == SOURCE_OPTIMIZER) & (df["status"] == "OK")].copy()
    if sub.empty:
        raise ValueError(
            f"No OK {SOURCE_OPTIMIZER!r} rows for seq{SOURCE_SEQ_LEN}@{SOURCE_ARRAY_ROWS}x{SOURCE_ARRAY_ROWS} "
            f"in {exp3_csv_path} -- Experiment 6 needs real source actions, not assumed ones."
        )
    sub["reward"] = sub["notes"].apply(lambda s: json.loads(s)["reward"])
    best = sub.loc[sub.groupby("seed")["reward"].idxmax()]
    actions: dict[int, Action] = {}
    for _, row in best.iterrows():
        stages = json.loads(row["notes"])["stages"]
        actions[int(row["seed"])] = Action(stages={name: StageAction(**vals) for name, vals in stages.items()})
    return actions


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
    parser.add_argument("--cells", type=str, nargs="+", default=[f"{w}:{h}" for w, h in DEFAULT_CELLS],
                        help="workload_name:hardware_name pairs, e.g. exp6_heldout_a_hd48:32x32")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seeds for the fresh-ceiling training run (also selects which of the "
                             "5 source-action seeds get zero-shot-transferred).")
    parser.add_argument("--n-iterations", type=int, default=6,
                        help="Fresh-ceiling optimizer iteration budget -- matches Experiment 3's "
                             "per-stage re-run budget exactly (not re-tuned), per "
                             "docs/NEXT_INSTRUCTIONS.md §3.3.")
    parser.add_argument("--out", type=str, default=str(RAW_CSV))
    parser.add_argument("--source-exp3-csv", type=str, default=str(SOURCE_EXP3_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.out) if Path(args.out).is_absolute() else (THIS_DIR / args.out)
    git_commit = logging_utils.current_git_commit(_PROJECT_ROOT)
    scratch_root = RESULTS_DIR / "raw" / "_scratch"

    source_actions = _load_source_actions(args.source_exp3_csv)
    if set(source_actions) != set(args.seeds):
        print(f"[exp6] WARNING: source actions cover seeds {sorted(source_actions)}, "
              f"but --seeds asked for {sorted(args.seeds)} -- using the intersection.")
    transfer_seeds = sorted(set(source_actions) & set(args.seeds))

    cells = [tuple(c.split(":")) for c in args.cells]

    for workload_name, hw_name in cells:
        workload = load_workload_yaml(str(_PROJECT_ROOT / "config" / "workloads" / f"{workload_name}.yaml"))
        hardware = load_hardware_config(str(_PROJECT_ROOT / "config" / "hardware" / f"{hw_name}.yaml"))
        cell_scratch = scratch_root / workload_name / hw_name
        print(f"[exp6] {workload_name} x {hw_name} (head_dim={workload.head_dim}) ...")

        try:
            # --- Baselines (role: baseline_single_dataflow / baseline_fixed_per_stage) ---
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
                print(f"[exp6]   {method}: status={result.status}" + (
                    f" util={result.pe_utilization_pct:.2f}%" if result.status == "OK" else ""))

            for attempt in baselines.best_fixed_per_stage.all_attempts:
                logging_utils.append_run_row(csv_path, _base_row(
                    method="best_fixed_per_stage_search", hardware=hardware, workload=workload,
                    dataflow_mode=attempt.dataflow_mode, seed=-1, iteration=0, result=attempt.result,
                    git_commit=git_commit, notes={"stage": attempt.stage_name, "sim_notes": attempt.result.notes},
                ))

            best_fixed_per_stage_util = None
            if baselines.best_fixed_per_stage_combined is not None:
                per_stage_choice = {s: m.dataflow_mode for s, m in baselines.best_fixed_per_stage.winners.items()}
                logging_utils.append_run_row(csv_path, _base_row(
                    method="best_fixed_per_stage", hardware=hardware, workload=workload, dataflow_mode="mixed",
                    seed=-1, iteration=0, result=baselines.best_fixed_per_stage_combined, git_commit=git_commit,
                    notes={"per_stage_dataflow": per_stage_choice},
                ))
                best_fixed_per_stage_util = baselines.best_fixed_per_stage_combined.pe_utilization_pct
                print(f"[exp6]   best_fixed_per_stage: status=OK util={best_fixed_per_stage_util:.2f}%")
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
                print(f"[exp6]   best_fixed_per_stage: status=FAILED (no winner for {missing})")

            # --- Zero-shot transfer (role: zero_shot_transfer) ---
            problem = AttentionMappingProblem(
                workload, hardware, traffic_penalty_weight=0.01, failed_run_penalty_reward=-1000.0,
                scratch_dir=cell_scratch / "zero_shot_transfer",
            )
            for source_seed in transfer_seeds:
                action = source_actions[source_seed]
                result = problem.evaluate(action)
                dataflow_summary = "-".join(action.stages[name].dataflow_mode for name in STAGE_NAMES)
                per_stage = {
                    name: {
                        "dataflow_mode": sa.dataflow_mode,
                        "ifmap_frac": round(sa.ifmap_frac, 4), "filter_frac": round(sa.filter_frac, 4),
                        "ofmap_frac": round(sa.ofmap_frac, 4),
                    }
                    for name, sa in action.stages.items()
                }
                notes = {
                    "source": f"exp3_perstage seq{SOURCE_SEQ_LEN}@{SOURCE_ARRAY_ROWS}x{SOURCE_ARRAY_ROWS} "
                              f"{SOURCE_OPTIMIZER} seed={source_seed}",
                    "stages": per_stage, "reward": round(result.reward, 6), "sim_notes": result.simulation.notes,
                }
                logging_utils.append_run_row(csv_path, _base_row(
                    method="zero_shot_transfer", hardware=hardware, workload=workload,
                    dataflow_mode=dataflow_summary, seed=source_seed, iteration=0,
                    result=result.simulation, git_commit=git_commit, notes=notes,
                ))
            print(f"[exp6]   zero_shot_transfer: {len(transfer_seeds)} source actions evaluated")

            # --- Fresh ceiling (role: fresh_ceiling) ---
            optimizer_name, _ = _load_selected_optimizer()
            for seed in args.seeds:
                fresh_problem = AttentionMappingProblem(
                    workload, hardware, traffic_penalty_weight=0.01, failed_run_penalty_reward=-1000.0,
                    scratch_dir=cell_scratch / "fresh_ceiling" / f"seed{seed}",
                )
                _, optimizer = _load_selected_optimizer()  # fresh instance per seed (some hold internal state)
                results = optimizer.run(fresh_problem, n_iterations=args.n_iterations, seed=seed)
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
                        method="fresh_ceiling", hardware=hardware, workload=workload,
                        dataflow_mode=dataflow_summary, seed=seed, iteration=i,
                        result=r.simulation, git_commit=git_commit, notes=notes,
                    ))
                rewards = [r.reward for r in results]
                print(f"[exp6]   fresh_ceiling seed={seed}: {len(results)} evals, best reward={max(rewards):.3f}")

        except Exception as e:
            print(f"[exp6]   {workload_name} x {hw_name}: UNEXPECTED ERROR, skipping rest of this cell: {e}")

    print(f"[exp6] done. Raw results: {csv_path}")


if __name__ == "__main__":
    main()
