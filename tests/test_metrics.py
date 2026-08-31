"""
Unit tests for common/metrics.py.

Required at minimum per CODING_AGENT_PLAN.md §3: utilization/EDP formulas are one of
the two places a silent bug would invisibly corrupt every downstream number (the
other is workload_gen.py, see test_workload_gen.py).
"""
from __future__ import annotations

import pytest

from common.metrics import compute_edp, pct_diff_vs_baseline
from common.simulator_interface import SimulationResult


def _ok_result(**overrides) -> SimulationResult:
    defaults = dict(
        exec_cycles=1000, pe_utilization_pct=80.0,
        sram_dram_traffic_bytes=2048, wall_clock_s=0.5, status="OK",
    )
    defaults.update(overrides)
    return SimulationResult(**defaults)


def test_compute_edp_compute_only() -> None:
    """With energy_per_byte_pj left at 0, EDP = (cycles * energy_per_cycle_pj) * cycles."""
    result = _ok_result(exec_cycles=1000)
    edp = compute_edp(result, energy_per_cycle_pj=2.0)
    assert edp == pytest.approx((1000 * 2.0) * 1000)


def test_compute_edp_includes_memory_energy_when_given() -> None:
    result = _ok_result(exec_cycles=1000, sram_dram_traffic_bytes=500)
    edp = compute_edp(result, energy_per_cycle_pj=2.0, energy_per_byte_pj=0.1)
    expected_energy = 1000 * 2.0 + 500 * 0.1
    assert edp == pytest.approx(expected_energy * 1000)


def test_compute_edp_rejects_failed_run() -> None:
    """EDP is undefined for a FAILED run and must raise, not silently return a number."""
    failed = SimulationResult(
        exec_cycles=0, pe_utilization_pct=0.0,
        sram_dram_traffic_bytes=0, wall_clock_s=0.1, status="FAILED", notes="simulator crash",
    )
    with pytest.raises(ValueError):
        compute_edp(failed, energy_per_cycle_pj=1.0)


def test_pct_diff_vs_baseline_positive_case() -> None:
    """10% above baseline must report +10.0, not 1.1 or -10.0."""
    assert pct_diff_vs_baseline(value=110.0, baseline_value=100.0) == pytest.approx(10.0)


def test_pct_diff_vs_baseline_negative_case() -> None:
    assert pct_diff_vs_baseline(value=90.0, baseline_value=100.0) == pytest.approx(-10.0)


def test_pct_diff_vs_baseline_zero_baseline_raises() -> None:
    with pytest.raises(ValueError):
        pct_diff_vs_baseline(value=10.0, baseline_value=0.0)


def test_simulation_result_rejects_invalid_status() -> None:
    with pytest.raises(ValueError):
        SimulationResult(
            exec_cycles=1, pe_utilization_pct=1.0,
            sram_dram_traffic_bytes=1, wall_clock_s=0.1, status="MAYBE",
        )
