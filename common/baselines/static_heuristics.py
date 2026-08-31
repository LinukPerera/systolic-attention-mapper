"""
static_heuristics.py — fixed-dataflow baseline mappers (WS-only, OS-only, IS-only, best-fixed-per-stage).

These are the deterministic baselines Experiment 3 compares the learned optimizer
against, and the array-size study in Experiment 1 sweeps over. Deterministic per
plan ground rule 3: run once per condition, not across seeds (unlike the learned
optimizer) — none of these functions take a `seed` argument.

**No RS-only mapper**: CODING_AGENT_PLAN.md §1's module comment for this file lists
"WS-only, OS-only, IS-only, RS-only, best-fixed-per-stage". Verified 2026-08-31
against the actual installed `scalesim==3.0.0` package (`scale_config.py`'s
`valid_df_list = ['os', 'ws', 'is']`): SCALE-Sim does not support a row-stationary
dataflow at all. This is a real finding, not an oversight — see
common.simulator_interface.SUPPORTED_DATAFLOW_MODES and its docstring, and flag it
in the thesis's methodology text if RS was meant to appear there.

A "mapping" here is just a dataflow choice (SCALE-Sim itself, given a dataflow and
a topology, determines its own internal tiling — this project does not hand-tune
tile sizes for the fixed baselines, only for the learned optimizer's action space).
best_fixed_per_stage_mapping is the one non-trivial function: it actually runs the
simulator once per (stage, dataflow) pair to find the true per-stage best, since
that choice is empirical, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.metrics import compute_edp
from common.simulator_interface import (
    SUPPORTED_DATAFLOW_MODES,
    HardwareConfig,
    SimulationResult,
    run_scale_sim,
    write_scale_sim_inputs,
)
from common.workload_gen import AttentionWorkload, gemm_shapes_to_topology_rows, workload_to_gemm_shapes


@dataclass(frozen=True)
class Mapping:
    """A fixed-dataflow baseline's choice for one workload: just the dataflow mode.

    (No tile sizes: see module docstring — fixed baselines rely on SCALE-Sim's own
    tiling for the given dataflow, they don't search a tiling space.)
    """
    dataflow_mode: str


@dataclass(frozen=True)
class StageAttempt:
    """One (stage, dataflow) candidate evaluated while searching for the best-fixed-per-stage mapping."""
    stage_name: str
    dataflow_mode: str
    result: SimulationResult


@dataclass(frozen=True)
class BestFixedPerStageResult:
    """Full output of best_fixed_per_stage_mapping: the winning choice per stage,
    its already-computed SimulationResult (no re-simulation needed to use it), and
    every (stage, dataflow) candidate actually evaluated along the way — logged as
    evidence by the caller, per plan §3, not discarded once the winner is known."""
    winners: dict[str, Mapping]
    winner_results: dict[str, SimulationResult]
    all_attempts: list[StageAttempt]


def ws_only_mapping(workload: AttentionWorkload, hardware_config_path: str) -> Mapping:
    """Weight-stationary-only mapping for one workload/hardware pair. Trivial and deterministic."""
    return Mapping(dataflow_mode="WS")


def os_only_mapping(workload: AttentionWorkload, hardware_config_path: str) -> Mapping:
    """Output-stationary-only mapping for one workload/hardware pair. Trivial and deterministic."""
    return Mapping(dataflow_mode="OS")


def is_only_mapping(workload: AttentionWorkload, hardware_config_path: str) -> Mapping:
    """Input-stationary-only mapping for one workload/hardware pair. Trivial and deterministic."""
    return Mapping(dataflow_mode="IS")


def best_fixed_per_stage_mapping(
    workload: AttentionWorkload,
    hardware: HardwareConfig,
    *,
    scratch_dir: Path,
) -> BestFixedPerStageResult:
    """Best single dataflow chosen independently per GEMM stage (QKV / QK^T / score*V),
    determined empirically by actually simulating every (stage, dataflow) pair and
    picking the one with the lowest EDP — not assumed from intuition.

    Still a *fixed* (non-learned) baseline: the choice is made once via exhaustive
    per-stage evaluation, not learned via search, and is deterministic (no seed).

    Note: SCALE-Sim's `Dataflow` setting applies to a whole simulation run, not
    per-layer, so there is no single simulation that directly measures "stage A
    with its best dataflow, stage B with its (different) best dataflow" together —
    use common.simulator_interface.combine_simulation_results on this function's
    winner_results.values() to get one whole-workload SimulationResult comparable
    to ws_only/os_only/is_only, as run_exp1.py does.

    Args:
        workload: The attention workload to evaluate.
        hardware: The hardware config to evaluate on (a loaded HardwareConfig, not a
            path, since it's used directly by simulator_interface).
        scratch_dir: Directory to write each candidate's SCALE-Sim inputs/outputs into.

    Returns:
        A BestFixedPerStageResult with the winning Mapping and SimulationResult per
        stage, plus every (stage, dataflow) candidate actually evaluated.

    Raises:
        RuntimeError: Propagated from run_scale_sim if Docker isn't available/reachable
            — this is a real simulator call, not a stub (plan ground rule 1: no result
            without a real run).
    """
    shapes = workload_to_gemm_shapes(workload)
    winners: dict[str, Mapping] = {}
    winner_results: dict[str, SimulationResult] = {}
    all_attempts: list[StageAttempt] = []

    for shape in shapes:
        best_edp: float | None = None
        best_mode: str | None = None
        best_result: SimulationResult | None = None
        for mode in SUPPORTED_DATAFLOW_MODES:
            rows = gemm_shapes_to_topology_rows([shape])
            run_dir = Path(scratch_dir) / f"{shape.name}_{mode}"
            topology_path, layout_path, config_path = write_scale_sim_inputs(rows, hardware, mode, run_dir)
            result = run_scale_sim(
                topology_path, layout_path, config_path,
                run_name=f"{hardware.name}_{mode}", hardware=hardware, output_dir=run_dir / "output",
            )
            all_attempts.append(StageAttempt(stage_name=shape.name, dataflow_mode=mode, result=result))
            if result.status != "OK":
                # A candidate that fails to simulate is not silently skipped from the
                # comparison — it simply can't win it. The caller's experiment script
                # is responsible for logging this raw result as evidence (plan §3);
                # it's already captured in all_attempts above.
                continue
            edp = compute_edp(result, energy_per_cycle_pj=hardware.energy_per_cycle_pj)
            if best_edp is None or edp < best_edp:
                best_edp, best_mode, best_result = edp, mode, result

        if best_mode is None or best_result is None:
            raise RuntimeError(
                f"best_fixed_per_stage_mapping: every dataflow candidate FAILED for stage "
                f"'{shape.name}'; cannot pick a best-fixed-per-stage mapping without at least "
                "one successful run. Check all_attempts for the individual FAILED results."
            )
        winners[shape.name] = Mapping(dataflow_mode=best_mode)
        winner_results[shape.name] = best_result

    return BestFixedPerStageResult(winners=winners, winner_results=winner_results, all_attempts=all_attempts)
