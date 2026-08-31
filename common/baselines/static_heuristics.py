"""
static_heuristics.py — fixed-dataflow baseline mappers (WS-only, OS-only, IS-only, RS-only, best-fixed-per-stage).

These are the deterministic baselines Experiment 3 compares the learned optimizer
against, and the array-size study in Experiment 1 sweeps over. Deterministic per
plan ground rule 3: run once per condition, not across seeds (unlike the learned
optimizer) — none of these functions take a `seed` argument.

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
from common.simulator_interface import HardwareConfig, run_scale_sim, write_scale_sim_inputs
from common.workload_gen import AttentionWorkload, gemm_shapes_to_topology_rows, workload_to_gemm_shapes

DATAFLOW_MODES: tuple[str, ...] = ("WS", "OS", "IS", "RS")


@dataclass(frozen=True)
class Mapping:
    """A fixed-dataflow baseline's choice for one workload: just the dataflow mode.

    (No tile sizes: see module docstring — fixed baselines rely on SCALE-Sim's own
    tiling for the given dataflow, they don't search a tiling space.)
    """
    dataflow_mode: str


def ws_only_mapping(workload: AttentionWorkload, hardware_config_path: str) -> Mapping:
    """Weight-stationary-only mapping for one workload/hardware pair. Trivial and deterministic."""
    return Mapping(dataflow_mode="WS")


def os_only_mapping(workload: AttentionWorkload, hardware_config_path: str) -> Mapping:
    """Output-stationary-only mapping for one workload/hardware pair. Trivial and deterministic."""
    return Mapping(dataflow_mode="OS")


def is_only_mapping(workload: AttentionWorkload, hardware_config_path: str) -> Mapping:
    """Input-stationary-only mapping for one workload/hardware pair. Trivial and deterministic."""
    return Mapping(dataflow_mode="IS")


def rs_only_mapping(workload: AttentionWorkload, hardware_config_path: str) -> Mapping:
    """Row-stationary-only mapping for one workload/hardware pair. Trivial and deterministic."""
    return Mapping(dataflow_mode="RS")


def best_fixed_per_stage_mapping(
    workload: AttentionWorkload,
    hardware: HardwareConfig,
    *,
    scratch_dir: Path,
) -> dict[str, Mapping]:
    """Best single dataflow chosen independently per GEMM stage (QKV / QK^T / score*V),
    determined empirically by actually simulating every (stage, dataflow) pair and
    picking the one with the lowest EDP — not assumed from intuition.

    Still a *fixed* (non-learned) baseline: the choice is made once via exhaustive
    per-stage evaluation, not learned via search, and is deterministic (no seed).

    Args:
        workload: The attention workload to evaluate.
        hardware: The hardware config to evaluate on (note: takes a loaded
            HardwareConfig, not a path, since it's used directly by simulator_interface).
        scratch_dir: Directory to write each candidate's SCALE-Sim inputs/outputs into.

    Returns:
        Dict from stage name ("qkv_projection", "qk_transpose", "score_times_v") to
        the winning Mapping for that stage.

    Raises:
        RuntimeError: Propagated from run_scale_sim if Docker isn't available/reachable
            — this is a real simulator call, not a stub (plan ground rule 1: no result
            without a real run).
    """
    shapes = workload_to_gemm_shapes(workload)
    best: dict[str, Mapping] = {}

    for shape in shapes:
        best_edp: float | None = None
        best_mode: str | None = None
        for mode in DATAFLOW_MODES:
            rows = gemm_shapes_to_topology_rows([shape])
            run_dir = Path(scratch_dir) / f"{shape.name}_{mode}"
            topology_path, config_path = write_scale_sim_inputs(rows, hardware, mode, run_dir)
            result = run_scale_sim(
                topology_path, config_path,
                docker_image=hardware.docker_image, output_dir=run_dir / "output",
            )
            if result.status != "OK":
                # A candidate that fails to simulate is not silently skipped from the
                # comparison — it simply can't win it. The caller's experiment script
                # is responsible for logging this raw result as evidence (plan §3).
                continue
            edp = compute_edp(result, energy_per_cycle_pj=hardware.energy_per_cycle_pj)
            if best_edp is None or edp < best_edp:
                best_edp, best_mode = edp, mode

        if best_mode is None:
            raise RuntimeError(
                f"best_fixed_per_stage_mapping: every dataflow candidate FAILED for stage "
                f"'{shape.name}'; cannot pick a best-fixed-per-stage mapping without at least "
                "one successful run. Check the individual FAILED results under scratch_dir."
            )
        best[shape.name] = Mapping(dataflow_mode=best_mode)

    return best
