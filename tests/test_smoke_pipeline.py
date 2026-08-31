"""
End-to-end smoke test: one tiny fixed config through the REAL pipeline —
workload -> simulator inputs -> real containerized SCALE-Sim -> metrics -> one
logged CSV row.

Per Phase A Stage 0 item 5 / Phase B's real-run requirement: this proves the
plumbing works, with real numbers, not a stub. config/workloads/smoke_tiny.yaml
and config/hardware/smoke_8x8.yaml exist only for this test (see the comments in
those files). The image referenced by smoke_8x8.yaml's docker_image field
(scalesim:latest) must be built first — see docker/scalesim/Dockerfile and
README.md's Setup section.

If Docker isn't reachable in whatever environment runs this test, the
Docker-dependent stage skips cleanly with a clear reason rather than failing —
"Docker isn't running here" is an environment fact, not a plumbing bug — but the
non-Docker stage (workload -> GEMM shapes -> topology/layout/config files) always
runs for real and is asserted on unconditionally.
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
    """Stage 1 (no Docker needed): workload YAML -> GEMM shapes -> topology/layout/config files on disk."""
    workload = load_workload_yaml(WORKLOAD_YAML)
    hardware = load_hardware_config(HARDWARE_YAML)

    shapes = workload_to_gemm_shapes(workload)
    assert len(shapes) == 3
    rows = gemm_shapes_to_topology_rows(shapes)
    assert len(rows) == 1 + 2 * workload.num_heads  # qkv_projection + (qk_transpose, score_times_v) per head

    topology_path, layout_path, config_path = write_scale_sim_inputs(rows, hardware, "WS", tmp_path / "run0")
    assert topology_path.exists() and topology_path.stat().st_size > 0
    assert layout_path.exists()
    assert config_path.exists()
    assert "ArrayHeight = 8" in config_path.read_text()
    assert "Dataflow = ws" in config_path.read_text()
    # topology row count == layout row count (one layout row per topology layer)
    assert layout_path.read_text().count("\n") == topology_path.read_text().count("\n")


def test_smoke_pipeline_end_to_end_through_simulator_and_logging(tmp_path) -> None:
    """Stage 2: the same tiny config through run_scale_sim -> metrics -> one CSV row,
    with real SCALE-Sim numbers (not fabricated) when Docker is reachable."""
    workload = load_workload_yaml(WORKLOAD_YAML)
    hardware = load_hardware_config(HARDWARE_YAML)
    shapes = workload_to_gemm_shapes(workload)
    rows = gemm_shapes_to_topology_rows(shapes)
    run_dir = tmp_path / "run0"
    topology_path, layout_path, config_path = write_scale_sim_inputs(rows, hardware, "WS", run_dir)

    if not _docker_daemon_reachable():
        pytest.skip(
            "Docker daemon not reachable in this environment — this is the one stage of the "
            "smoke pipeline that genuinely needs it (docs/CODING_AGENT_PLAN.md §2). "
            "Stages before it (test_smoke_pipeline_workload_to_simulator_inputs) already ran "
            "for real and passed. Re-run this test once Docker is up."
        )

    result = run_scale_sim(
        topology_path, layout_path, config_path,
        run_name=f"{hardware.name}_WS", hardware=hardware, output_dir=run_dir / "output", timeout_s=120,
    )
    assert result.status == "OK", f"expected a real OK run (Docker was reachable); got: {result}"
    assert result.exec_cycles > 0
    assert 0.0 <= result.pe_utilization_pct <= 100.0
    assert result.sram_dram_traffic_bytes > 0

    row = {
        "run_id": "smoke-run0", "experiment": "smoke_test",
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": current_git_commit(),
        "optimizer": "baseline", "array_rows": hardware.array_rows, "array_cols": hardware.array_cols,
        "sram_kb": sum(hardware.sram_kb.values()), "dataflow_mode": "WS",
        "seq_len": workload.seq_len, "hidden_dim": workload.hidden_dim, "num_heads": workload.num_heads,
        "seed": 0, "iteration": 0,
        "pe_utilization_pct": result.pe_utilization_pct, "exec_cycles": result.exec_cycles,
        "sram_dram_traffic_bytes": result.sram_dram_traffic_bytes,
        "edp": compute_edp(result, energy_per_cycle_pj=hardware.energy_per_cycle_pj),
        "wall_clock_s": result.wall_clock_s, "status": result.status, "notes": result.notes,
    }
    csv_path = tmp_path / "smoke_test.csv"
    append_run_row(csv_path, row)
    assert csv_path.exists()
    assert csv_path.read_text().count("\n") == 2  # header + one data row
