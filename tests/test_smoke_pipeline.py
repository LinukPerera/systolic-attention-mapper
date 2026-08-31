"""
End-to-end smoke test: one tiny fixed config through the REAL pipeline —
workload -> simulator inputs -> (simulator) -> metrics -> one logged CSV row.

Per Phase A Stage 0 item 5: this proves the plumbing works before any experiment
logic is built on top of it. It is not a substitute for a real experiment run —
config/workloads/smoke_tiny.yaml and config/hardware/smoke_8x8.yaml exist only for
this test (see the comments in those files).

The workload-derivation and input-writing stages run for real and are asserted on
unconditionally — they need no Docker. The actual SCALE-Sim invocation needs a
reachable Docker daemon (docs/CODING_AGENT_PLAN.md §2); when it isn't reachable this
test skips that one step with a clear reason rather than failing, since "Docker
isn't set up on this machine yet" is an environment fact, not a plumbing bug. When
Docker *is* reachable, the test exercises simulator_interface.run_scale_sim for
real and — regardless of whether the placeholder `scalesim:latest` image actually
exists — proves the FAILED-run path logs a real evidence row too (plan ground rule
1: a failure is recorded, never silently dropped or replaced with a fabricated number).
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone

import pytest

from common.logging_utils import append_run_row, current_git_commit
from common.metrics import compute_edp
from common.simulator_interface import load_hardware_config, run_scale_sim, write_scale_sim_inputs
from common.workload_gen import gemm_shapes_to_topology_rows, load_workload_yaml, workload_to_gemm_shapes

WORKLOAD_YAML = "config/workloads/smoke_tiny.yaml"
HARDWARE_YAML = "config/hardware/smoke_8x8.yaml"


def _docker_daemon_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0


def test_smoke_pipeline_workload_to_simulator_inputs(tmp_path) -> None:
    """Stage 1 (no Docker needed): workload YAML -> GEMM shapes -> topology rows -> files on disk."""
    workload = load_workload_yaml(WORKLOAD_YAML)
    hardware = load_hardware_config(HARDWARE_YAML)

    shapes = workload_to_gemm_shapes(workload)
    assert len(shapes) == 3
    rows = gemm_shapes_to_topology_rows(shapes)
    assert len(rows) == 1 + 2 * workload.num_heads  # qkv_projection + (qk_transpose, score_times_v) per head

    topology_path, config_path = write_scale_sim_inputs(rows, hardware, "WS", tmp_path / "run0")
    assert topology_path.exists() and topology_path.stat().st_size > 0
    assert config_path.exists()
    assert "ArrayHeight = 8" in config_path.read_text()
    assert "Dataflow = ws" in config_path.read_text()


def test_smoke_pipeline_end_to_end_through_simulator_and_logging(tmp_path) -> None:
    """Stage 2: the same tiny config through run_scale_sim -> metrics -> one CSV row.

    Skips the Docker-dependent portion with a clear reason if no daemon is reachable
    in this environment; otherwise runs it for real and logs whatever SimulationResult
    (OK or FAILED) actually comes back — never a fabricated one.
    """
    workload = load_workload_yaml(WORKLOAD_YAML)
    hardware = load_hardware_config(HARDWARE_YAML)
    shapes = workload_to_gemm_shapes(workload)
    rows = gemm_shapes_to_topology_rows(shapes)
    run_dir = tmp_path / "run0"
    topology_path, config_path = write_scale_sim_inputs(rows, hardware, "WS", run_dir)

    if not _docker_daemon_reachable():
        pytest.skip(
            "Docker daemon not reachable on this machine — this is the one stage of the "
            "smoke pipeline that genuinely needs it (docs/CODING_AGENT_PLAN.md §2). "
            "Stages before it (this test's sibling, test_smoke_pipeline_workload_to_simulator_inputs) "
            "already ran for real and passed. Re-run this test once Docker is set up."
        )

    result = run_scale_sim(
        topology_path, config_path,
        docker_image=hardware.docker_image, output_dir=run_dir / "output", timeout_s=120,
    )
    assert result.status in ("OK", "FAILED")

    row = {
        "run_id": "smoke-run0", "experiment": "smoke_test",
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": current_git_commit(),
        "optimizer": "baseline", "array_rows": hardware.array_rows, "array_cols": hardware.array_cols,
        "sram_kb": sum(hardware.sram_kb.values()), "dataflow_mode": "WS",
        "seq_len": workload.seq_len, "hidden_dim": workload.hidden_dim, "num_heads": workload.num_heads,
        "seed": 0, "iteration": 0,
        "pe_utilization_pct": result.pe_utilization_pct, "exec_cycles": result.exec_cycles,
        "sram_dram_traffic_bytes": result.sram_dram_traffic_bytes,
        "edp": (
            compute_edp(result, energy_per_cycle_pj=hardware.energy_per_cycle_pj)
            if result.status == "OK" else None
        ),
        "wall_clock_s": result.wall_clock_s, "status": result.status, "notes": result.notes,
    }
    csv_path = tmp_path / "smoke_test.csv"
    append_run_row(csv_path, row)
    assert csv_path.exists()
    assert csv_path.read_text().count("\n") == 2  # header + one data row
