"""
environment.py — the real dataflow+SRAM-partition search problem every optimizer
in this package searches over.

**Action space, and why**: SCALE-Sim's config (verified 2026-08-31 by reading
scale_config.py, common.simulator_interface) exposes exactly two run-to-run
controllable levers: the dataflow mode (WS/OS/IS) and the ifmap/filter/ofmap SRAM
buffer split. There is no separate, explicit "tile size" parameter. So in this
project, "tiling" search means searching that SRAM partition — Action below is
(dataflow_mode, ifmap_frac, filter_frac, ofmap_frac) with the fractions summing to
1 over the hardware's total SRAM budget (sum of hardware.sram_kb.values()). This is
a real constraint of the actual tool being used, not an arbitrary simplification.

**Reward**: pe_utilization_pct minus a per-MAC memory-traffic penalty —
`reward = utilization - traffic_penalty_weight * (traffic_bytes / total_macs)` —
matching CODING_AGENT_PLAN.md §3's own worked example ("reward is utilization minus
a memory-traffic penalty term ... because raw utilization alone lets the agent
ignore DRAM stalls"). Normalizing traffic by total_macs (rather than raw bytes)
keeps the penalty's scale comparable across workloads of different size.
traffic_penalty_weight and failed_run_penalty_reward are real hyperparameters,
read from config/optimizers/*.yaml by each optimizer — never hardcoded here
(plan §3: "no magic numbers").

Every optimizer (reinforce.py, ppo.py, evolutionary.py, bayesian_opt.py) evaluates
candidates through this one class's `evaluate()`, so simulator-calling and
reward-computation logic is written once and shared (plan §3: "one function = one
responsibility") — no optimizer file calls run_scale_sim directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.simulator_interface import (
    SUPPORTED_DATAFLOW_MODES,
    HardwareConfig,
    SimulationResult,
    run_scale_sim,
    write_scale_sim_inputs,
)
from common.workload_gen import AttentionWorkload, gemm_shapes_to_topology_rows, total_macs, workload_to_gemm_shapes

_FRAC_SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class Action:
    """One candidate mapping: a dataflow choice plus an SRAM partition.

    Attributes:
        dataflow_mode: One of common.simulator_interface.SUPPORTED_DATAFLOW_MODES.
        ifmap_frac: Fraction of the hardware's total SRAM budget given to the ifmap buffer.
        filter_frac: Fraction given to the filter buffer.
        ofmap_frac: Fraction given to the ofmap buffer. ifmap_frac + filter_frac +
            ofmap_frac must sum to 1.0 (within _FRAC_SUM_TOLERANCE).
    """
    dataflow_mode: str
    ifmap_frac: float
    filter_frac: float
    ofmap_frac: float

    def __post_init__(self) -> None:
        fracs = (self.ifmap_frac, self.filter_frac, self.ofmap_frac)
        if any(f < 0 for f in fracs):
            raise ValueError(f"Action: SRAM fractions must be non-negative; got {fracs}.")
        total = sum(fracs)
        if abs(total - 1.0) > _FRAC_SUM_TOLERANCE:
            raise ValueError(f"Action: SRAM fractions must sum to 1.0 (+/- {_FRAC_SUM_TOLERANCE}); got {fracs} summing to {total}.")


def decode_raw_action(raw: tuple[float, float, float]) -> Action:
    """Decode 3 raw values, each in [0, 1], into an Action.

    Shared by every optimizer's search-space encoding (A2C/PPO's continuous Box
    action, the evolutionary candidate's genome, Bayesian optimization's suggested
    floats) so all four candidates search *literally the same encoded 3-real-valued
    space* — required for Experiment 0's bake-off to be a fair comparison of search
    strategy, not an artifact of four different encodings.

    raw[0]: binned into len(SUPPORTED_DATAFLOW_MODES) equal-width ranges -> dataflow_mode.
    raw[1], raw[2]: stick-breaking parameters for the SRAM simplex —
        ifmap_frac = raw[1]
        filter_frac = (1 - ifmap_frac) * raw[2]
        ofmap_frac = 1 - ifmap_frac - filter_frac
    This guarantees the three fractions are non-negative and sum to exactly 1 for
    any raw value in [0,1]^3, with no rejection sampling or clipping needed.

    Args:
        raw: Three values; values outside [0, 1] are clipped, not rejected, so every
            optimizer's native output range (some produce values slightly outside
            [0,1]) always decodes to a valid Action.

    Returns:
        A valid Action.
    """
    r0, r1, r2 = (min(1.0, max(0.0, v)) for v in raw)
    bin_idx = min(int(r0 * len(SUPPORTED_DATAFLOW_MODES)), len(SUPPORTED_DATAFLOW_MODES) - 1)
    dataflow_mode = SUPPORTED_DATAFLOW_MODES[bin_idx]
    ifmap_frac = r1
    filter_frac = (1.0 - ifmap_frac) * r2
    ofmap_frac = 1.0 - ifmap_frac - filter_frac
    return Action(dataflow_mode=dataflow_mode, ifmap_frac=ifmap_frac, filter_frac=filter_frac, ofmap_frac=ofmap_frac)


@dataclass(frozen=True)
class EvalResult:
    """One evaluated Action's real simulation result and derived reward."""
    action: Action
    simulation: SimulationResult
    reward: float


class AttentionMappingProblem:
    """One (workload, hardware) pair's real dataflow+SRAM-partition search problem."""

    def __init__(
        self,
        workload: AttentionWorkload,
        hardware: HardwareConfig,
        *,
        traffic_penalty_weight: float,
        failed_run_penalty_reward: float,
        scratch_dir: Path,
    ) -> None:
        """
        Args:
            workload: The attention workload being optimized for.
            hardware: The hardware config being optimized for.
            traffic_penalty_weight: Reward's per-MAC traffic penalty coefficient —
                from config/optimizers/*.yaml, not hardcoded by any optimizer.
            failed_run_penalty_reward: Reward assigned when a candidate action's
                simulation FAILS (e.g. an SRAM split too small to fit the working
                set) — a real, fixed, config-supplied penalty so the optimizer
                learns to avoid infeasible actions from real feedback, rather than
                the failure being silently hidden (plan ground rule 1).
            scratch_dir: Directory each evaluate() call writes its SCALE-Sim
                inputs/outputs into (one subdirectory per call).
        """
        self.workload = workload
        self.hardware = hardware
        self.traffic_penalty_weight = traffic_penalty_weight
        self.failed_run_penalty_reward = failed_run_penalty_reward
        self.scratch_dir = Path(scratch_dir)
        self._shapes = workload_to_gemm_shapes(workload)
        self._topology_rows = gemm_shapes_to_topology_rows(self._shapes)
        self._total_macs = total_macs(self._shapes)
        self._total_sram_kb = sum(hardware.sram_kb.values())
        self._eval_count = 0

    def evaluate(self, action: Action) -> EvalResult:
        """Run the real simulator on one candidate action and compute its reward.

        Every call is a real, separate SCALE-Sim invocation — there is no caching
        or interpolation here that could let an optimizer's reported result diverge
        from an actual simulation (plan ground rule 1).

        Args:
            action: The candidate mapping to evaluate.

        Returns:
            The real simulation result and its derived reward.

        Raises:
            RuntimeError: Propagated from run_scale_sim if Docker isn't reachable —
                an environment problem, not a candidate's outcome.
        """
        self._eval_count += 1
        sram_kb = {
            "ifmap": action.ifmap_frac * self._total_sram_kb,
            "filter": action.filter_frac * self._total_sram_kb,
            "ofmap": action.ofmap_frac * self._total_sram_kb,
        }
        run_dir = self.scratch_dir / f"eval{self._eval_count:05d}"
        topology_path, layout_path, config_path = write_scale_sim_inputs(
            self._topology_rows, self.hardware, action.dataflow_mode, run_dir, sram_kb=sram_kb,
        )
        result = run_scale_sim(
            topology_path, layout_path, config_path,
            run_name=f"{self.hardware.name}_{action.dataflow_mode}", hardware=self.hardware,
            output_dir=run_dir / "output",
        )
        if result.status != "OK":
            return EvalResult(action=action, simulation=result, reward=self.failed_run_penalty_reward)

        traffic_per_mac = result.sram_dram_traffic_bytes / self._total_macs
        reward = result.pe_utilization_pct - self.traffic_penalty_weight * traffic_per_mac
        return EvalResult(action=action, simulation=result, reward=reward)
