"""
simulator_interface.py — thin wrapper around (containerized) SCALE-Sim.

Responsible for: loading a hardware config, writing SCALE-Sim's input files
(topology CSV + INI config), invoking the simulator via Docker, and parsing its
output report into the raw metrics (cycles, utilization, traffic) that metrics.py
then turns into the project's reported metrics (EDP etc.).

This module never contains optimizer logic and never decides *what* to simulate —
callers (baselines/, optimizers/, experiments/) hand it a workload's topology rows
plus a hardware config and dataflow choice, and get raw numbers back (plan §3:
"one function = one responsibility").

Docker: `docs/CODING_AGENT_PLAN.md` §2 specifies SCALE-Sim runs containerized.
run_scale_sim() checks for a reachable Docker daemon and fails loudly with a clear,
actionable error if one isn't available, rather than silently producing no result
(plan ground rule 1). It does not fall back to a non-Docker SCALE-Sim install.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SimulationResult:
    """Raw output of one SCALE-Sim run, before any derived-metric computation."""
    exec_cycles: int
    pe_utilization_pct: float
    sram_dram_traffic_bytes: int
    wall_clock_s: float
    status: str  # "OK" or "FAILED", per the data contract in CODING_AGENT_PLAN.md §4
    notes: str = field(default="")

    def __post_init__(self) -> None:
        if self.status not in ("OK", "FAILED"):
            raise ValueError(f"SimulationResult.status must be 'OK' or 'FAILED', got {self.status!r}.")


@dataclass(frozen=True)
class HardwareConfig:
    """One systolic-array hardware configuration, loaded from config/hardware/*.yaml.

    Attributes:
        name: Identifier used in filenames/logs (e.g. "16x16").
        array_rows: PE array rows.
        array_cols: PE array columns.
        sram_kb: Per-buffer SRAM sizes in KB, keys "ifmap", "filter", "ofmap".
        dram_bandwidth_bytes_per_cycle: DRAM interface bandwidth.
        energy_per_cycle_pj: Per-cycle energy figure, used by metrics.compute_edp.
        docker_image: Docker image (name:tag) that runs SCALE-Sim for this config.
    """
    name: str
    array_rows: int
    array_cols: int
    sram_kb: dict[str, float]
    dram_bandwidth_bytes_per_cycle: float
    energy_per_cycle_pj: float
    docker_image: str


_HARDWARE_REQUIRED_KEYS = (
    "name", "array_rows", "array_cols", "sram_kb",
    "dram_bandwidth_bytes_per_cycle", "energy_per_cycle_pj", "docker_image",
)


def load_hardware_config(path: str) -> HardwareConfig:
    """Load a HardwareConfig from a config/hardware/*.yaml file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed HardwareConfig.

    Raises:
        ValueError: If a required key is missing.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    missing = [k for k in _HARDWARE_REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}; hardware YAML needs {_HARDWARE_REQUIRED_KEYS}.")
    return HardwareConfig(
        name=data["name"], array_rows=data["array_rows"], array_cols=data["array_cols"],
        sram_kb=data["sram_kb"], dram_bandwidth_bytes_per_cycle=data["dram_bandwidth_bytes_per_cycle"],
        energy_per_cycle_pj=data["energy_per_cycle_pj"], docker_image=data["docker_image"],
    )


_TOPOLOGY_FIELDNAMES = (
    "LayerName", "IFMAP Height", "IFMAP Width", "Filter Height",
    "Filter Width", "Channels", "Num Filter", "Strides",
)


def write_scale_sim_inputs(
    topology_rows: list[dict],
    hardware: HardwareConfig,
    dataflow_mode: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write SCALE-Sim's two input files (topology.csv, config.cfg) into out_dir.

    Args:
        topology_rows: Output of workload_gen.gemm_shapes_to_topology_rows.
        hardware: The hardware config to simulate on.
        dataflow_mode: One of "WS", "OS", "IS", "RS" (SCALE-Sim's dataflow setting).
        out_dir: Directory to write into; created if it doesn't exist. Conventionally
            <exp>/results/raw/<run_id>/.

    Returns:
        (topology_path, config_path).

    Raises:
        ValueError: If dataflow_mode isn't a recognised SCALE-Sim dataflow string.
    """
    if dataflow_mode not in ("WS", "OS", "IS", "RS"):
        raise ValueError(f"dataflow_mode must be one of WS/OS/IS/RS, got {dataflow_mode!r}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topology_path = out_dir / "topology.csv"
    with open(topology_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_TOPOLOGY_FIELDNAMES))
        writer.writeheader()
        for row in topology_rows:
            writer.writerow(row)

    # INI-style config, per SCALE-Sim v2's [general]/[architecture_presets] sections.
    # NOTE (same caveat as gemm_shapes_to_topology_rows): confirm this section/key
    # layout against the actual installed SCALE-Sim version before trusting output.
    config_path = out_dir / "config.cfg"
    config_text = (
        "[general]\n"
        f"run_name = {hardware.name}_{dataflow_mode}\n"
        "\n"
        "[architecture_presets]\n"
        f"ArrayHeight = {hardware.array_rows}\n"
        f"ArrayWidth = {hardware.array_cols}\n"
        f"IfmapSramSzkB = {hardware.sram_kb.get('ifmap', 0)}\n"
        f"FilterSramSzkB = {hardware.sram_kb.get('filter', 0)}\n"
        f"OfmapSramSzkB = {hardware.sram_kb.get('ofmap', 0)}\n"
        f"Bandwidth = {hardware.dram_bandwidth_bytes_per_cycle}\n"
        f"Dataflow = {dataflow_mode.lower()}\n"
    )
    config_path.write_text(config_text)

    return topology_path, config_path


def run_scale_sim(
    topology_path: Path,
    config_path: Path,
    *,
    docker_image: str,
    output_dir: Path,
    timeout_s: float | None = None,
) -> SimulationResult:
    """Invoke the containerized SCALE-Sim on the given inputs and parse its output.

    Fails loudly: no Docker daemon is a raised RuntimeError (this is a setup problem,
    not a run outcome); a SCALE-Sim process that errors or times out is a returned
    SimulationResult with status="FAILED" and a descriptive `notes` — it is logged as
    evidence, never silently dropped (plan §3, ground rule 1).

    Args:
        topology_path: Path from write_scale_sim_inputs.
        config_path: Path from write_scale_sim_inputs.
        docker_image: Image (name:tag) to run — typically hardware.docker_image.
        output_dir: Directory SCALE-Sim should write its report into; created if needed.
        timeout_s: Optional wall-clock timeout; a timeout is recorded as FAILED, not a crash.

    Returns:
        The parsed simulation result.

    Raises:
        RuntimeError: If the `docker` CLI is missing or the daemon isn't reachable —
            this is treated as an environment problem to fix, not a failed experiment run.
    """
    if shutil.which("docker") is None:
        raise RuntimeError(
            "docker executable not found on PATH. SCALE-Sim runs containerized per "
            "docs/CODING_AGENT_PLAN.md §2 (\"Simulator: SCALE-Sim (containerized per the "
            "proposal — Docker)\"). Install Docker before running any real experiment."
        )
    daemon_check = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if daemon_check.returncode != 0:
        raise RuntimeError(
            "docker CLI is present but the Docker daemon is not reachable "
            f"(docker info exited {daemon_check.returncode}: "
            f"{daemon_check.stderr.strip()[-500:]}). Start Docker Desktop / the docker "
            "daemon before running any real experiment — this is an environment problem, "
            "not a simulation result, so it is not logged as a FAILED run."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    host_dir = topology_path.parent.resolve()

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_dir}:/workspace",
        docker_image,
        "python", "scale.py",
        "-c", f"/workspace/{config_path.name}",
        "-t", f"/workspace/{topology_path.name}",
        "-p", "/workspace/output",
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return SimulationResult(
            exec_cycles=0, pe_utilization_pct=0.0, sram_dram_traffic_bytes=0,
            wall_clock_s=(time.monotonic() - start), status="FAILED",
            notes=f"SCALE-Sim run timed out after {timeout_s}s (docker image {docker_image}).",
        )
    wall_clock_s = time.monotonic() - start

    if proc.returncode != 0:
        return SimulationResult(
            exec_cycles=0, pe_utilization_pct=0.0, sram_dram_traffic_bytes=0,
            wall_clock_s=wall_clock_s, status="FAILED",
            notes=f"docker run exited {proc.returncode}: {proc.stderr.strip()[-2000:]}",
        )

    try:
        return _parse_scale_sim_report(host_dir / "output", wall_clock_s=wall_clock_s)
    except (FileNotFoundError, KeyError, ValueError) as e:
        # Report layout didn't match what this parser expects — fail loudly with the
        # raw stdout attached, rather than guessing a number (plan ground rule 1).
        return SimulationResult(
            exec_cycles=0, pe_utilization_pct=0.0, sram_dram_traffic_bytes=0,
            wall_clock_s=wall_clock_s, status="FAILED",
            notes=(
                f"SCALE-Sim ran (exit 0) but its report could not be parsed: {e}. "
                f"stdout tail: {proc.stdout.strip()[-1000:]}"
            ),
        )


def _parse_scale_sim_report(output_dir: Path, *, wall_clock_s: float) -> SimulationResult:
    """Parse SCALE-Sim's output report into a SimulationResult.

    NOTE: the filename/columns below reflect SCALE-Sim v2's typical output (an
    `*_avg_stats.csv` per-run summary with per-layer cycles/utilization/DRAM-traffic
    columns). **Confirm this against the actual output your installed SCALE-Sim
    version produces** — if it raises here, that is the first thing to fix, not a
    signal to fall back to a guessed number (plan ground rule 1). This function is
    the single place that needs updating if the report format differs.

    Args:
        output_dir: The directory SCALE-Sim was asked to write its report into.
        wall_clock_s: Wall-clock time measured by the caller around the docker run.

    Returns:
        A SimulationResult with status="OK".

    Raises:
        FileNotFoundError: If no matching report file is found.
        KeyError: If an expected column is missing from the report.
        ValueError: If a value can't be parsed as the expected type.
    """
    candidates = sorted(Path(output_dir).glob("*avg_stats.csv"))
    if not candidates:
        raise FileNotFoundError(f"No '*avg_stats.csv' report found under {output_dir}.")
    report_path = candidates[0]

    with open(report_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{report_path} has a header but no data rows.")

    # Aggregate across topology rows (one per GEMM/head, per
    # workload_gen.gemm_shapes_to_topology_rows) into one whole-attention-block result:
    # cycles sum (layers ran sequentially), utilization is a cycle-weighted mean,
    # traffic sums.
    total_cycles = 0
    total_traffic = 0
    util_x_cycles = 0.0
    for row in rows:
        cycles = int(float(row["Compute time(cycles)"]))
        util_pct = float(row["Overall utilization(%)"])
        traffic = int(float(row.get("DRAM traffic(bytes)", 0)))
        total_cycles += cycles
        total_traffic += traffic
        util_x_cycles += util_pct * cycles

    mean_util = (util_x_cycles / total_cycles) if total_cycles > 0 else 0.0

    return SimulationResult(
        exec_cycles=total_cycles,
        pe_utilization_pct=mean_util,
        sram_dram_traffic_bytes=total_traffic,
        wall_clock_s=wall_clock_s,
        status="OK",
    )
