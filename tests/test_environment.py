"""
Unit tests for common/optimizers/environment.py's per-stage action space
(StageAction, Action, decode_raw_action, AttentionMappingProblem.evaluate/
evaluate_stage) — added 2026-09-02 alongside the redesign from one global action
to one action per GEMM stage (see that module's docstring for why).

`evaluate`/`evaluate_stage` are exercised with `run_scale_sim`/`write_scale_sim_inputs`
monkeypatched to fixed SimulationResults — these are unit tests of the *combining*
and *reward* logic around the real simulator calls, not a substitute for the real
Docker-backed sweeps every experiment script actually runs (plan ground rule 1:
never fabricate an experimental *result* — these fixed numbers are never reported
as one).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from common.optimizers.environment import (
    STAGE_NAMES,
    Action,
    AttentionMappingProblem,
    StageAction,
    decode_raw_action,
)
from common.simulator_interface import HardwareConfig, SimulationResult
from common.workload_gen import AttentionWorkload


def _make_hardware() -> HardwareConfig:
    return HardwareConfig(
        name="test_hw", array_rows=16, array_cols=16,
        sram_kb={"ifmap": 64.0, "filter": 64.0, "ofmap": 64.0},
        dram_bandwidth_bytes_per_cycle=16.0, energy_per_cycle_pj=1.0,
        bytes_per_element=4.0, docker_image="scalesim:latest",
    )


def _make_workload() -> AttentionWorkload:
    return AttentionWorkload(seq_len=128, hidden_dim=64, num_heads=4)


# --- StageAction ---

def test_stage_action_valid_fracs_ok() -> None:
    StageAction(dataflow_mode="WS", ifmap_frac=0.5, filter_frac=0.3, ofmap_frac=0.2)


def test_stage_action_fracs_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        StageAction(dataflow_mode="WS", ifmap_frac=0.5, filter_frac=0.5, ofmap_frac=0.5)


def test_stage_action_fracs_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        StageAction(dataflow_mode="WS", ifmap_frac=-0.1, filter_frac=0.6, ofmap_frac=0.5)


# --- Action ---

def _stage_action() -> StageAction:
    return StageAction(dataflow_mode="OS", ifmap_frac=1 / 3, filter_frac=1 / 3, ofmap_frac=1 / 3)


def test_action_requires_exactly_stage_names_keys() -> None:
    Action(stages={name: _stage_action() for name in STAGE_NAMES})


def test_action_rejects_missing_stage() -> None:
    incomplete = {name: _stage_action() for name in STAGE_NAMES if name != STAGE_NAMES[0]}
    with pytest.raises(ValueError, match="must have exactly the keys"):
        Action(stages=incomplete)


def test_action_rejects_extra_stage() -> None:
    extra = {name: _stage_action() for name in STAGE_NAMES}
    extra["not_a_real_stage"] = _stage_action()
    with pytest.raises(ValueError):
        Action(stages=extra)


def test_action_get_returns_that_stages_action() -> None:
    per_stage = {name: _stage_action() for name in STAGE_NAMES}
    action = Action(stages=per_stage)
    assert action.get(STAGE_NAMES[0]) is per_stage[STAGE_NAMES[0]]


# --- decode_raw_action ---

def test_decode_raw_action_requires_3_per_stage() -> None:
    expected_len = 3 * len(STAGE_NAMES)
    with pytest.raises(ValueError, match=str(expected_len)):
        decode_raw_action(tuple([0.5] * (expected_len - 1)))


def test_decode_raw_action_produces_one_stageaction_per_stage() -> None:
    raw = tuple([0.5] * (3 * len(STAGE_NAMES)))
    action = decode_raw_action(raw)
    assert set(action.stages) == set(STAGE_NAMES)
    for stage_action in action.stages.values():
        assert stage_action.dataflow_mode in ("WS", "OS", "IS")
        total = stage_action.ifmap_frac + stage_action.filter_frac + stage_action.ofmap_frac
        assert total == pytest.approx(1.0)


def test_decode_raw_action_stages_independent() -> None:
    """Different raw slices per stage decode to different StageActions — stages
    aren't accidentally sharing/aliasing one decoded value."""
    n = 3 * len(STAGE_NAMES)
    raw = tuple(i / (n - 1) for i in range(n))  # strictly increasing across the whole vector
    action = decode_raw_action(raw)
    stage_actions = [action.get(name) for name in STAGE_NAMES]
    # At least the ifmap_frac (raw index 1 within each stage's 3-slice) must differ
    # stage to stage given strictly increasing raw input.
    ifmap_fracs = [sa.ifmap_frac for sa in stage_actions]
    assert len(set(ifmap_fracs)) == len(STAGE_NAMES)


def test_decode_raw_action_clips_out_of_range_values() -> None:
    n = 3 * len(STAGE_NAMES)
    raw = tuple([-5.0, 5.0, 0.5] * len(STAGE_NAMES))[:n]
    action = decode_raw_action(raw)  # must not raise
    for stage_action in action.stages.values():
        total = stage_action.ifmap_frac + stage_action.filter_frac + stage_action.ofmap_frac
        assert total == pytest.approx(1.0)


def test_decode_raw_action_dataflow_bins_cover_all_modes() -> None:
    """Regression check for the exact binning bug documented in brute_force.py's
    module docstring: with r0 in {0, 1/3, 2/3, 1}, decode_raw_action must still be
    *capable* of producing every dataflow mode somewhere in [0, 1] (this test picks
    r0 values known to hit each bin, rather than asserting the 4-point grid itself
    is safe — brute_force.py deliberately does NOT rely on this encoding for that
    reason)."""
    seen_modes = set()
    for r0 in (0.0, 0.5, 0.99):
        raw = (r0, 0.5, 0.5) * len(STAGE_NAMES)
        action = decode_raw_action(raw)
        seen_modes.add(action.get(STAGE_NAMES[0]).dataflow_mode)
    assert seen_modes == {"WS", "OS", "IS"}


# --- AttentionMappingProblem.evaluate / evaluate_stage ---

def _fixed_result(status: str = "OK") -> SimulationResult:
    return SimulationResult(
        exec_cycles=1000, pe_utilization_pct=50.0, sram_dram_traffic_bytes=2000,
        wall_clock_s=0.01, status=status,
    )


@pytest.fixture
def problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AttentionMappingProblem:
    monkeypatch.setattr(
        "common.optimizers.environment.write_scale_sim_inputs",
        lambda *a, **k: (tmp_path / "topology.csv", tmp_path / "layout.csv", tmp_path / "config.cfg"),
    )
    monkeypatch.setattr(
        "common.optimizers.environment.run_scale_sim",
        lambda *a, **k: _fixed_result(),
    )
    return AttentionMappingProblem(
        _make_workload(), _make_hardware(),
        traffic_penalty_weight=0.01, failed_run_penalty_reward=-1000.0,
        scratch_dir=tmp_path / "scratch",
    )


def test_evaluate_runs_once_per_stage(problem: AttentionMappingProblem, monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def counting_run_scale_sim(*a, **k):
        call_count["n"] += 1
        return _fixed_result()

    monkeypatch.setattr("common.optimizers.environment.run_scale_sim", counting_run_scale_sim)

    action = decode_raw_action(tuple([0.5] * (3 * len(STAGE_NAMES))))
    problem.evaluate(action)
    assert call_count["n"] == len(STAGE_NAMES)


def test_evaluate_combines_and_computes_reward(problem: AttentionMappingProblem) -> None:
    action = decode_raw_action(tuple([0.5] * (3 * len(STAGE_NAMES))))
    result = problem.evaluate(action)
    assert result.simulation.status == "OK"
    # 3 stages combined: exec_cycles sums, pe_utilization_pct is unweighted-equal
    # here since all 3 fixed results are identical (cycle-weighted mean of equal
    # values equals that value).
    assert result.simulation.exec_cycles == 3000
    assert result.simulation.pe_utilization_pct == pytest.approx(50.0)
    assert result.simulation.sram_dram_traffic_bytes == 6000
    assert isinstance(result.reward, float)


def test_evaluate_any_failed_stage_yields_failed_combined_and_penalty_reward(
    problem: AttentionMappingProblem, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def flaky_run_scale_sim(*a, **k):
        calls["n"] += 1
        return _fixed_result(status="FAILED" if calls["n"] == 1 else "OK")

    monkeypatch.setattr("common.optimizers.environment.run_scale_sim", flaky_run_scale_sim)

    action = decode_raw_action(tuple([0.5] * (3 * len(STAGE_NAMES))))
    result = problem.evaluate(action)
    assert result.simulation.status == "FAILED"
    assert result.reward == problem.failed_run_penalty_reward


def test_evaluate_stage_runs_exactly_one_simulation(
    problem: AttentionMappingProblem, monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}

    def counting_run_scale_sim(*a, **k):
        call_count["n"] += 1
        return _fixed_result()

    monkeypatch.setattr("common.optimizers.environment.run_scale_sim", counting_run_scale_sim)

    result = problem.evaluate_stage(STAGE_NAMES[0], _stage_action(), run_tag="t0")
    assert call_count["n"] == 1
    assert result.status == "OK"


def test_evaluate_stage_rejects_unknown_stage_name(problem: AttentionMappingProblem) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        problem.evaluate_stage("not_a_real_stage", _stage_action(), run_tag="t0")
