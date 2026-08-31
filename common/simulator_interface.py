"""
simulator_interface.py — thin wrapper around (containerized) SCALE-Sim.

Responsible for: loading a hardware config, writing SCALE-Sim's three input files
(GEMM-mode topology CSV, layout CSV, INI config), invoking the simulator via Docker,
and parsing its output reports into the raw metrics (cycles, utilization, traffic)
that metrics.py then turns into the project's reported metrics (EDP etc.).

This module never contains optimizer logic and never decides *what* to simulate —
callers (baselines/, optimizers/, experiments/) hand it a workload's topology rows
plus a hardware config and dataflow choice, and get raw numbers back (plan §3:
"one function = one responsibility").

Docker: `docs/CODING_AGENT_PLAN.md` §2 specifies SCALE-Sim runs containerized —
see `docker/scalesim/Dockerfile` (built as the `scalesim:latest` image referenced
by config/hardware/*.yaml's `docker_image` field). Containerizing isn't just
following the proposal: SCALE-Sim 3.0.0 requires numpy<1.25 (verified 2026-08-31 —
it calls `int()` on a non-0-d numpy array, which numpy>=1.25 rejects), which is
directly incompatible with this project's own numpy>=1.25/1.26 requirement
(pandas/matplotlib in `common/logging_utils.py`, `common/plotting/`) — the two
cannot coexist in one Python environment, so the container isn't optional.

run_scale_sim() checks for a reachable Docker daemon and fails loudly with a clear,
actionable error if one isn't available, rather than silently producing no result
(plan ground rule 1).

Every input-file format, config section/key name, and output-report column name in
this module was verified 2026-08-31 against the actual installed `scalesim==3.0.0`
package (source read directly, then confirmed with real end-to-end runs both via a
local venv and via `docker run` against the built image) — not assumed from
documentation or general SCALE-Sim familiarity. If you upgrade the pinned scalesim
version, re-verify every NOTE in this file against the new version's source before
trusting any number it produces.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

# SCALE-Sim (verified 2026-08-31, scalesim==3.0.0) supports exactly these three
# dataflows — `scale_config.py`'s `valid_df_list = ['os', 'ws', 'is']`. There is no
# "RS" (row-stationary) mode, despite CODING_AGENT_PLAN.md §1's module comment on
# static_heuristics.py listing "RS-only" — that comment does not match what the
# actual simulator supports and should be treated as superseded by this finding.
SUPPORTED_DATAFLOW_MODES: tuple[str, ...] = ("WS", "OS", "IS")

# A single trivial (no custom SRAM banking) layout row applies to every layer,
# regardless of its shape — see write_scale_sim_inputs. Verified against
# scalesim/layout_utils.py's 20-field format (LayerName + 20 ints), 2026-08-31.
_TRIVIAL_LAYOUT_FIELDS: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2)


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


def combine_simulation_results(results: list[SimulationResult]) -> SimulationResult:
    """Combine several separately-run SimulationResults into one whole-workload result.

    Needed because SCALE-Sim's `Dataflow` setting applies to an entire simulation
    run, not per-layer — there is no way to simulate "stage A with WS, stage B with
    OS" in one call. `common.baselines.static_heuristics.best_fixed_per_stage_mapping`
    finds each stage's best dataflow by running it separately, and this function
    aggregates those separate results into one figure comparable to the
    single-dataflow baselines (ws_only/os_only/is_only), using exactly the same
    combination rule `_parse_scale_sim_report` already uses across layers *within*
    one run: exec_cycles sums, pe_utilization_pct is the cycle-weighted mean,
    sram_dram_traffic_bytes and wall_clock_s sum.

    Args:
        results: One SimulationResult per stage, in any order. Must be non-empty.

    Returns:
        A single combined SimulationResult. status is "FAILED" (with a `notes`
        explaining which stage failed) if any input result is FAILED — a workload
        with one infeasible stage has no defined whole-workload metrics, so this is
        not silently computed from only the OK stages (plan ground rule 1).
    """
    if not results:
        raise ValueError("combine_simulation_results: results must be non-empty.")

    failed = [r for r in results if r.status != "OK"]
    if failed:
        return SimulationResult(
            exec_cycles=0, pe_utilization_pct=0.0, sram_dram_traffic_bytes=0,
            wall_clock_s=sum(r.wall_clock_s for r in results), status="FAILED",
            notes=f"{len(failed)}/{len(results)} stage(s) FAILED: {'; '.join(r.notes for r in failed if r.notes)}",
        )

    total_cycles = sum(r.exec_cycles for r in results)
    util_x_cycles = sum(r.pe_utilization_pct * r.exec_cycles for r in results)
    mean_util = (util_x_cycles / total_cycles) if total_cycles > 0 else 0.0
    return SimulationResult(
        exec_cycles=total_cycles,
        pe_utilization_pct=mean_util,
        sram_dram_traffic_bytes=sum(r.sram_dram_traffic_bytes for r in results),
        wall_clock_s=sum(r.wall_clock_s for r in results),
        status="OK",
    )


@dataclass(frozen=True)
class HardwareConfig:
    """One systolic-array hardware configuration, loaded from config/hardware/*.yaml.

    Attributes:
        name: Identifier used in filenames/logs (e.g. "16x16").
        array_rows: PE array rows.
        array_cols: PE array columns.
        sram_kb: Per-buffer SRAM sizes in KB, keys "ifmap", "filter", "ofmap".
        dram_bandwidth_bytes_per_cycle: DRAM interface bandwidth. **Not currently
            wired into the simulation** — see write_scale_sim_inputs's docstring;
            SCALE-Sim's CALC (auto-estimate) bandwidth mode is used instead until
            the USER-mode Bandwidth field's exact units are independently verified.
        energy_per_cycle_pj: Per-cycle energy figure, used by metrics.compute_edp.
        bytes_per_element: Bytes per matrix element, used to convert SCALE-Sim's
            reported DRAM word-traffic into bytes for sram_dram_traffic_bytes. No
            default — an explicit, documented choice per hardware config (e.g. 4 for
            fp32, 2 for fp16/bf16), per plan §3 ("no magic numbers").
        docker_image: Docker image (name:tag) that runs SCALE-Sim for this config.
    """
    name: str
    array_rows: int
    array_cols: int
    sram_kb: dict[str, float]
    dram_bandwidth_bytes_per_cycle: float
    energy_per_cycle_pj: float
    bytes_per_element: float
    docker_image: str


_HARDWARE_REQUIRED_KEYS = (
    "name", "array_rows", "array_cols", "sram_kb", "dram_bandwidth_bytes_per_cycle",
    "energy_per_cycle_pj", "bytes_per_element", "docker_image",
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
        energy_per_cycle_pj=data["energy_per_cycle_pj"], bytes_per_element=data["bytes_per_element"],
        docker_image=data["docker_image"],
    )


def write_scale_sim_inputs(
    topology_rows: list[dict],
    hardware: HardwareConfig,
    dataflow_mode: str,
    out_dir: Path,
    *,
    sram_kb: dict[str, float] | None = None,
) -> tuple[Path, Path, Path]:
    """Write SCALE-Sim's three input files (topology.csv, layout.csv, config.cfg) into out_dir.

    Bandwidth: uses SCALE-Sim's CALC (auto-estimate) mode rather than wiring
    hardware.dram_bandwidth_bytes_per_cycle through as a USER-mode value — the
    USER-mode `Bandwidth` field's exact units (words/cycle per which buffer?) were
    not independently verified against real output the way every other format
    choice in this module was, and guessing would risk silently corrupting every
    downstream traffic/EDP number (plan ground rule 1). This means
    hardware.dram_bandwidth_bytes_per_cycle is presently a *documented*, not yet
    *simulated*, hardware axis — a real gap to close (with the same real-run
    verification approach used for everything else here) before an experiment
    treats DRAM bandwidth as an independent variable.

    Layout: every layer gets the same trivial (no custom SRAM banking) layout row —
    _TRIVIAL_LAYOUT_FIELDS — since none of this project's experiments vary SRAM
    interleaving; SCALE-Sim requires a layout file to exist regardless.

    Args:
        topology_rows: Output of workload_gen.gemm_shapes_to_topology_rows.
        hardware: The hardware config to simulate on.
        dataflow_mode: One of SUPPORTED_DATAFLOW_MODES ("WS", "OS", "IS").
        out_dir: Directory to write into; created if it doesn't exist. Conventionally
            <exp>/results/raw/<run_id>/.
        sram_kb: Optional override for the ifmap/filter/ofmap SRAM split, keyed like
            hardware.sram_kb. Defaults to hardware.sram_kb itself. This is how
            common.optimizers.environment.AttentionMappingProblem searches the SRAM
            partition ("tiling") axis without needing a different HardwareConfig per
            candidate — the total budget stays hardware's, only the split changes.

    Returns:
        (topology_path, layout_path, config_path).

    Raises:
        ValueError: If dataflow_mode isn't a SCALE-Sim-supported dataflow string.
    """
    if dataflow_mode not in SUPPORTED_DATAFLOW_MODES:
        raise ValueError(
            f"dataflow_mode must be one of {SUPPORTED_DATAFLOW_MODES} (SCALE-Sim does not support "
            f"'RS' — see SUPPORTED_DATAFLOW_MODES docstring); got {dataflow_mode!r}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_sram_kb = sram_kb if sram_kb is not None else hardware.sram_kb

    # --- topology.csv: GEMM mode, "LayerName, M, N, K," (trailing comma required —
    # scalesim's parser does `row.split(',')[:-1]`, i.e. it expects and discards a
    # trailing empty field). Verified against a real run, 2026-08-31.
    topology_path = out_dir / "topology.csv"
    with open(topology_path, "w", newline="") as f:
        f.write("Layer name, M, N, K,\n")
        for row in topology_rows:
            f.write(f"{row['LayerName']}, {row['M']}, {row['N']}, {row['K']},\n")

    # --- layout.csv: one trivial row per layer, same trailing-comma convention.
    layout_path = out_dir / "layout.csv"
    layout_header = (
        "Layer name, IFMAP Height Intraline Factor, IFMAP Width Intraline Factor, "
        "Filter Height Intraline Factor, Filter Width Intraline Factor, Channel Intraline Factor, "
        "Num Filter Intraline Factor, IFMAP Height Intraline Order, IFMAP Width Intraline Order, "
        "Channel Intraline Order, IFMAP Height Interline Order, IFMAP Width Interline Order, "
        "Channel Interline Order, Num Filter Intraline Order, Channel Intraline Order, "
        "Filter Height Intraline Order, Filter Width Intraline Order, Num Filter Interline Order, "
        "Channel Interline Order, Filter Height Interline Order, Filter Width Interline Order,\n"
    )
    with open(layout_path, "w", newline="") as f:
        f.write(layout_header)
        for row in topology_rows:
            fields = ", ".join(str(v) for v in _TRIVIAL_LAYOUT_FIELDS)
            f.write(f"{row['LayerName']}, {fields},\n")

    # --- config.cfg: INI sections/keys as scalesim==3.0.0 actually reads them.
    # NOTE (verified 2026-08-31 by reading scale_config.py::read_conf_file and
    # confirming with a real run): `TopologyCsvLoc` must live under
    # [architecture_presets], NOT [network_presets] as its own header comment implies
    # — read_conf_file has a bug where the `section` variable is left at
    # 'architecture_presets' when it reads TopologyCsvLoc. A `[network_presets]`
    # section must still exist (config.has_section(...) is checked) but its content
    # is irrelevant. If you upgrade scalesim, re-verify this — it looks exactly like
    # the kind of thing an upstream fix would move.
    run_name = f"{hardware.name}_{dataflow_mode}"
    config_path = out_dir / "config.cfg"
    config_text = (
        "[general]\n"
        f"run_name = {run_name}\n"
        "\n"
        "[run_presets]\n"
        "InterfaceBandwidth = CALC\n"
        "UseRamulatorTrace = False\n"
        "\n"
        "[architecture_presets]\n"
        f"ArrayHeight = {hardware.array_rows}\n"
        f"ArrayWidth = {hardware.array_cols}\n"
        # floored at 1 KB: a 0 KB buffer is not a meaningful "very small SRAM" point,
        # it's a config SCALE-Sim isn't designed to accept — a 1 KB buffer still
        # legitimately stresses/fails a candidate action that starves that buffer.
        f"ifmapsramszkB = {max(1, int(effective_sram_kb.get('ifmap', 0)))}\n"
        f"filtersramszkB = {max(1, int(effective_sram_kb.get('filter', 0)))}\n"
        f"ofmapsramszkB = {max(1, int(effective_sram_kb.get('ofmap', 0)))}\n"
        "IfmapOffset = 0\n"
        "FilterOffset = 10000000\n"
        "OfmapOffset = 20000000\n"
        f"Dataflow = {dataflow_mode.lower()}\n"
        "ReadRequestBuffer = 60\n"
        "WriteRequestBuffer = 60\n"
        f"TopologyCsvLoc = \"{topology_path.name}\"\n"
        "\n"
        "[layout]\n"
        "IfmapCustomLayout = False\n"
        "FilterCustomLayout = False\n"
        "IfmapSRAMBankBandwidth = 10\n"
        "IfmapSRAMBankNum = 10\n"
        "IfmapSRAMBankPort = 2\n"
        "FilterSRAMBankBandwidth = 10\n"
        "FilterSRAMBankNum = 10\n"
        "FilterSRAMBankPort = 2\n"
        "\n"
        "[network_presets]\n"
        "\n"
        "[sparsity]\n"
        "SparsitySupport = False\n"
    )
    config_path.write_text(config_text)

    return topology_path, layout_path, config_path


def run_scale_sim(
    topology_path: Path,
    layout_path: Path,
    config_path: Path,
    *,
    run_name: str,
    hardware: HardwareConfig,
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
        layout_path: Path from write_scale_sim_inputs.
        config_path: Path from write_scale_sim_inputs.
        run_name: Must match the `run_name` baked into config_path's [general]
            section (write_scale_sim_inputs uses "<hardware.name>_<dataflow_mode>") —
            SCALE-Sim writes its report to `<output_dir>/<run_name>/`, verified
            2026-08-31 (`simulator.py::run`: `report_path = self.top_path + '/' + self.conf.get_run_name()`).
        hardware: Supplies both the Docker image to run (hardware.docker_image) and
            hardware.bytes_per_element (needed to convert the parsed report's
            DRAM word-traffic into bytes).
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

    # docker/scalesim/Dockerfile sets ENTRYPOINT ["python", "-m", "scalesim.scale"],
    # so the container is run with just scale.py's own CLI args.
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_dir}:/workspace",
        hardware.docker_image,
        "-t", f"/workspace/{topology_path.name}",
        "-l", f"/workspace/{layout_path.name}",
        "-c", f"/workspace/{config_path.name}",
        "-p", "/workspace/output",
        "-i", "gemm",
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
            notes=f"docker run exited {proc.returncode}: {proc.stderr.strip()[-2000:] or proc.stdout.strip()[-2000:]}",
        )

    try:
        report_dir = host_dir / "output" / run_name
        return _parse_scale_sim_report(
            report_dir, wall_clock_s=wall_clock_s, hardware_bytes_per_element=hardware.bytes_per_element,
        )
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


def _parse_scale_sim_report(
    report_dir: Path,
    *,
    wall_clock_s: float,
    hardware_bytes_per_element: float | None,
) -> SimulationResult:
    """Parse SCALE-Sim's COMPUTE_REPORT.csv + DETAILED_ACCESS_REPORT.csv into a SimulationResult.

    Column names and file layout verified 2026-08-31 against a real run of the
    installed scalesim==3.0.0 package (both via a local venv and via `docker run`
    against the built image — identical output). Specifically:

    - `COMPUTE_REPORT.csv`: one row per topology layer, columns include
      "Total Cycles (incl. prefetch)" and "Overall Util %" (note the real space
      before '%', and values are comma-separated with a trailing comma per row).
      exec_cycles is summed across layers ("Total Cycles (incl. prefetch)" —
      SCALE-Sim simulates each layer independently with no cross-layer pipelining
      modeled, so summing is the correct total for running all layers serially,
      which is how this project uses SCALE-Sim: one "layer" per attention GEMM stage).
      pe_utilization_pct is the cycle-weighted mean of "Overall Util %" across layers.
    - `DETAILED_ACCESS_REPORT.csv`: "DRAM IFMAP Reads", "DRAM Filter Reads",
      "DRAM OFMAP Writes" columns give DRAM traffic in *words*, not bytes — summed
      and multiplied by hardware_bytes_per_element to get sram_dram_traffic_bytes.

    If you upgrade the installed scalesim version, re-verify these column names
    against a real run before trusting this parser's output.

    Args:
        report_dir: `<output_dir>/<run_name>/`, per run_scale_sim's docstring.
        wall_clock_s: Wall-clock time measured by the caller around the docker run.
        hardware_bytes_per_element: hardware.bytes_per_element; required to convert
            word counts to bytes — pass None only from a caller that doesn't need
            sram_dram_traffic_bytes (there is none today; kept explicit rather than
            defaulted, per plan §3 "no magic numbers").

    Returns:
        A SimulationResult with status="OK".

    Raises:
        FileNotFoundError: If the expected report files aren't found.
        KeyError: If an expected column is missing.
        ValueError: If hardware_bytes_per_element is None, or a value can't be parsed.
    """
    if hardware_bytes_per_element is None:
        raise ValueError(
            "_parse_scale_sim_report requires hardware_bytes_per_element to convert "
            "DRAM word-traffic into bytes — no default, per plan §3 'no magic numbers'."
        )

    compute_path = Path(report_dir) / "COMPUTE_REPORT.csv"
    detail_path = Path(report_dir) / "DETAILED_ACCESS_REPORT.csv"
    if not compute_path.exists():
        raise FileNotFoundError(f"No COMPUTE_REPORT.csv found under {report_dir}.")
    if not detail_path.exists():
        raise FileNotFoundError(f"No DETAILED_ACCESS_REPORT.csv found under {report_dir}.")

    compute_df = pd.read_csv(compute_path, sep=r"\s*,\s*", engine="python").rename(columns=str.strip)
    detail_df = pd.read_csv(detail_path, sep=r"\s*,\s*", engine="python").rename(columns=str.strip)

    cycles_col = "Total Cycles (incl. prefetch)"
    util_col = "Overall Util %"
    for col in (cycles_col, util_col):
        if col not in compute_df.columns:
            raise KeyError(f"{compute_path}: expected column {col!r} not found; columns are {list(compute_df.columns)}.")

    total_cycles = int(compute_df[cycles_col].sum())
    util_x_cycles = float((compute_df[util_col] * compute_df[cycles_col]).sum())
    mean_util = (util_x_cycles / total_cycles) if total_cycles > 0 else 0.0

    traffic_cols = ("DRAM IFMAP Reads", "DRAM Filter Reads", "DRAM OFMAP Writes")
    for col in traffic_cols:
        if col not in detail_df.columns:
            raise KeyError(f"{detail_path}: expected column {col!r} not found; columns are {list(detail_df.columns)}.")
    total_words = int(sum(detail_df[col].sum() for col in traffic_cols))
    total_traffic_bytes = int(total_words * hardware_bytes_per_element)

    return SimulationResult(
        exec_cycles=total_cycles,
        pe_utilization_pct=mean_util,
        sram_dram_traffic_bytes=total_traffic_bytes,
        wall_clock_s=wall_clock_s,
        status="OK",
    )
