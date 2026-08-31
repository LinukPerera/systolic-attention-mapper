"""
metrics.py — derived performance metrics from raw SCALE-Sim output.

PE utilization %, memory traffic, and execution cycles pass through mostly as-is
from simulator_interface.SimulationResult; this module owns the metrics that are
*computed* — in particular EDP (Energy-Delay Product) and %-difference-vs-baseline —
so the formula is written once and every experiment/plot trusts the same definition.

Unit tests at tests/test_metrics.py — these formulas are one of the two places
(with workload_gen.py) a silent bug would invisibly corrupt every downstream number
(plan §3).
"""
from __future__ import annotations

from common.simulator_interface import SimulationResult


def compute_edp(
    result: SimulationResult,
    *,
    energy_per_cycle_pj: float,
    energy_per_byte_pj: float = 0.0,
) -> float:
    """Compute Energy-Delay Product for one simulation result.

    Delay is taken directly as exec_cycles (clock period is not modeled — this is a
    fixed multiplicative factor that cancels out in the %-difference-vs-baseline
    comparisons the plan actually reports, but means EDP here is in pJ·cycles, not
    pJ·seconds; note this unit explicitly wherever EDP is reported, e.g. in a
    stats.md §9 limitation).

    Energy is compute energy (exec_cycles * energy_per_cycle_pj) plus, if
    energy_per_byte_pj is supplied, memory-traffic energy
    (sram_dram_traffic_bytes * energy_per_byte_pj). Leaving energy_per_byte_pj at its
    default of 0.0 means memory energy is *not* modeled — this is a real
    simplification (SRAM vs. DRAM access energy differ substantially) that must be
    stated as a [LIMITATION] in any stats.md reporting EDP until a per-byte figure
    is added to config/hardware/*.yaml.

    Args:
        result: A completed (status == "OK") SimulationResult.
        energy_per_cycle_pj: Per-cycle energy figure from config/hardware/*.yaml —
            never hardcoded (plan §3: "no magic numbers").
        energy_per_byte_pj: Optional per-byte memory-traffic energy figure.

    Returns:
        EDP in pJ·cycles (energy in pJ times delay in cycles).

    Raises:
        ValueError: If result.status != "OK" — EDP is undefined for a failed run.
    """
    if result.status != "OK":
        raise ValueError(
            f"compute_edp requires a completed run (status='OK'); got status={result.status!r} "
            f"(notes: {result.notes!r}). A FAILED run has no defined EDP — log it as FAILED, "
            "don't substitute a placeholder EDP value (plan ground rule 1)."
        )
    energy_pj = (
        result.exec_cycles * energy_per_cycle_pj
        + result.sram_dram_traffic_bytes * energy_per_byte_pj
    )
    return energy_pj * result.exec_cycles


def pct_diff_vs_baseline(value: float, baseline_value: float) -> float:
    """Percentage difference of `value` relative to `baseline_value`.

    Args:
        value: The candidate (e.g. learned-optimizer) metric value.
        baseline_value: The reference (e.g. best static-heuristic) metric value.

    Returns:
        (value - baseline_value) / baseline_value * 100.

    Raises:
        ValueError: If baseline_value is 0 — the percentage difference is undefined,
            not "infinite" or "0"; this must be surfaced, not silently produced.
    """
    if baseline_value == 0:
        raise ValueError("pct_diff_vs_baseline: baseline_value is 0; percentage difference is undefined.")
    return (value - baseline_value) / baseline_value * 100.0
