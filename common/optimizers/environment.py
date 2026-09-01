"""
environment.py — the real dataflow+SRAM-partition search problem every optimizer
in this package searches over.

**Action space, and why**: SCALE-Sim's config (verified 2026-08-31 by reading
scale_config.py, common.simulator_interface) exposes exactly two run-to-run
controllable levers: the dataflow mode (WS/OS/IS) and the ifmap/filter/ofmap SRAM
buffer split. There is no separate, explicit "tile size" parameter. So in this
project, "tiling" search means searching that SRAM partition.

**Per-stage, revised 2026-09-02** (was one global `(dataflow_mode, sram_fracs)` for
the whole workload — see git history / `exp3_stats.md` §8, `exp5_stats.md` §8 for
why that was a real, measured limitation: `common.baselines.static_heuristics`'s
`best_fixed_per_stage_mapping` already proved a per-stage choice does better, and
the single-global-dataflow optimizer structurally couldn't compete with or express
that). `Action` now holds one `StageAction` — its own `(dataflow_mode, sram_fracs)`
— **per GEMM stage** (`STAGE_NAMES`: `qkv_projection`, `qk_transpose`,
`score_times_v` — the same 3-stage granularity `best_fixed_per_stage_mapping`
already uses; a per-attention-head action would multiply the space by `num_heads`
for no benefit, since every head within a stage shares the same GEMM shape).
`evaluate()` now runs one real simulation *per stage* (mirroring
`best_fixed_per_stage_mapping`'s own pattern) and combines them via
`common.simulator_interface.combine_simulation_results` — so every real
`evaluate()` call costs ~3x the simulator calls it used to; this was accounted for
by re-timing before any experiment re-run committed to a new budget (see each
experiment's own stats.md for the specific measurement).

**Reward**: pe_utilization_pct minus a per-MAC memory-traffic penalty —
`reward = utilization - traffic_penalty_weight * (traffic_bytes / total_macs)` —
matching CODING_AGENT_PLAN.md §3's own worked example ("reward is utilization minus
a memory-traffic penalty term ... because raw utilization alone lets the agent
ignore DRAM stalls"). Normalizing traffic by total_macs (rather than raw bytes)
keeps the penalty's scale comparable across workloads of different size. Computed
on the *combined* (all 3 stages) result, exactly as it was on the single
whole-workload result before — the reward's definition didn't change, only how
many real simulations produce the SimulationResult it's computed from.
traffic_penalty_weight and failed_run_penalty_reward are real hyperparameters,
read from config/optimizers/*.yaml by each optimizer — never hardcoded here
(plan §3: "no magic numbers").

Every optimizer (a2c.py, ppo.py, evolutionary.py, bayesian_opt.py) evaluates
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
    combine_simulation_results,
    run_scale_sim,
    write_scale_sim_inputs,
)
from common.workload_gen import AttentionWorkload, gemm_shapes_to_topology_rows, total_macs, workload_to_gemm_shapes

_FRAC_SUM_TOLERANCE = 0.01

# The 3-stage granularity common.baselines.static_heuristics.best_fixed_per_stage_mapping
# already uses — every GEMM shape sharing a stage name (e.g. all num_heads copies of
# qk_transpose) gets the SAME StageAction, since they share the same shape.
STAGE_NAMES: tuple[str, ...] = ("qkv_projection", "qk_transpose", "score_times_v")


@dataclass(frozen=True)
class StageAction:
    """One GEMM stage's dataflow choice plus SRAM partition.

    Attributes:
        dataflow_mode: One of common.simulator_interface.SUPPORTED_DATAFLOW_MODES.
        ifmap_frac: Fraction of the hardware's total SRAM budget given to the ifmap buffer.
        filter_frac: Fraction given to the filter buffer.
        ofmap_frac: Fraction given to the ofmap buffer. The three fractions must
            sum to 1.0 (within _FRAC_SUM_TOLERANCE).
    """
    dataflow_mode: str
    ifmap_frac: float
    filter_frac: float
    ofmap_frac: float

    def __post_init__(self) -> None:
        fracs = (self.ifmap_frac, self.filter_frac, self.ofmap_frac)
        if any(f < 0 for f in fracs):
            raise ValueError(f"StageAction: SRAM fractions must be non-negative; got {fracs}.")
        total = sum(fracs)
        if abs(total - 1.0) > _FRAC_SUM_TOLERANCE:
            raise ValueError(f"StageAction: SRAM fractions must sum to 1.0 (+/- {_FRAC_SUM_TOLERANCE}); got {fracs} summing to {total}.")


@dataclass(frozen=True)
class Action:
    """A full candidate mapping: one StageAction per GEMM stage (STAGE_NAMES).

    Attributes:
        stages: dict keyed by stage name, exactly `set(STAGE_NAMES)` — one
            StageAction per stage. Use `Action.get(stage_name)` rather than
            indexing `stages` directly if you want a clearer error on a typo'd name.
    """
    stages: dict[str, StageAction]

    def __post_init__(self) -> None:
        if set(self.stages) != set(STAGE_NAMES):
            raise ValueError(f"Action.stages must have exactly the keys {STAGE_NAMES}; got {sorted(self.stages)}.")

    def get(self, stage_name: str) -> StageAction:
        return self.stages[stage_name]


def _decode_stage(raw: tuple[float, float, float]) -> StageAction:
    """Decode 3 raw values, each in [0, 1], into one StageAction.

    raw[0]: binned into len(SUPPORTED_DATAFLOW_MODES) equal-width ranges -> dataflow_mode.
    raw[1], raw[2]: stick-breaking parameters for the SRAM simplex —
        ifmap_frac = raw[1]
        filter_frac = (1 - ifmap_frac) * raw[2]
        ofmap_frac = 1 - ifmap_frac - filter_frac
    This guarantees the three fractions are non-negative and sum to exactly 1 for
    any raw value in [0,1]^3, with no rejection sampling or clipping needed.
    Values outside [0, 1] are clipped, not rejected.
    """
    r0, r1, r2 = (min(1.0, max(0.0, v)) for v in raw)
    bin_idx = min(int(r0 * len(SUPPORTED_DATAFLOW_MODES)), len(SUPPORTED_DATAFLOW_MODES) - 1)
    dataflow_mode = SUPPORTED_DATAFLOW_MODES[bin_idx]
    ifmap_frac = r1
    filter_frac = (1.0 - ifmap_frac) * r2
    ofmap_frac = 1.0 - ifmap_frac - filter_frac
    return StageAction(dataflow_mode=dataflow_mode, ifmap_frac=ifmap_frac, filter_frac=filter_frac, ofmap_frac=ofmap_frac)


def decode_raw_action(raw: tuple[float, ...]) -> Action:
    """Decode `3 * len(STAGE_NAMES)` (= 9) raw values, each in [0, 1], into an Action.

    Shared by every optimizer's search-space encoding (A2C/PPO's continuous Box
    action, the evolutionary candidate's genome, Bayesian optimization's suggested
    floats) so all four candidates search *literally the same encoded 9-real-valued
    space* — required for Experiment 0's bake-off to be a fair comparison of search
    strategy, not an artifact of four different encodings.

    `raw` is 3 values per stage, concatenated in `STAGE_NAMES` order — i.e.
    `raw[0:3]` decodes `qkv_projection`'s StageAction (via `_decode_stage`),
    `raw[3:6]` decodes `qk_transpose`'s, `raw[6:9]` decodes `score_times_v`'s.

    Args:
        raw: `3 * len(STAGE_NAMES)` values; values outside [0, 1] are clipped, not
            rejected, so every optimizer's native output range always decodes to a
            valid Action.

    Returns:
        A valid Action.

    Raises:
        ValueError: If `len(raw) != 3 * len(STAGE_NAMES)`.
    """
    expected_len = 3 * len(STAGE_NAMES)
    if len(raw) != expected_len:
        raise ValueError(f"decode_raw_action: expected {expected_len} raw values (3 per stage), got {len(raw)}.")
    stages = {
        stage_name: _decode_stage(tuple(raw[3 * i: 3 * i + 3]))
        for i, stage_name in enumerate(STAGE_NAMES)
    }
    return Action(stages=stages)


@dataclass(frozen=True)
class EvalResult:
    """One evaluated Action's real (combined-across-stages) simulation result and derived reward."""
    action: Action
    simulation: SimulationResult
    reward: float


class AttentionMappingProblem:
    """One (workload, hardware) pair's real per-stage dataflow+SRAM-partition search problem."""

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
                simulation FAILS in any stage (e.g. an SRAM split too small to fit
                that stage's working set) — a real, fixed, config-supplied penalty
                so the optimizer learns to avoid infeasible actions from real
                feedback, rather than the failure being silently hidden (plan
                ground rule 1).
            scratch_dir: Directory each evaluate() call writes its SCALE-Sim
                inputs/outputs into (one subdirectory per call, with one
                sub-subdirectory per stage).
        """
        self.workload = workload
        self.hardware = hardware
        self.traffic_penalty_weight = traffic_penalty_weight
        self.failed_run_penalty_reward = failed_run_penalty_reward
        self.scratch_dir = Path(scratch_dir)
        self._shapes_by_name = {s.name: s for s in workload_to_gemm_shapes(workload)}
        self._total_macs = total_macs(list(self._shapes_by_name.values()))
        self._total_sram_kb = sum(hardware.sram_kb.values())
        self._eval_count = 0

    def evaluate(self, action: Action) -> EvalResult:
        """Run the real simulator once per GEMM stage and combine into one reward.

        Every stage's simulation is a real, separate SCALE-Sim invocation — there
        is no caching or interpolation here that could let an optimizer's reported
        result diverge from an actual simulation (plan ground rule 1).

        Args:
            action: The candidate per-stage mapping to evaluate.

        Returns:
            The combined simulation result and its derived reward.

        Raises:
            RuntimeError: Propagated from run_scale_sim if Docker isn't reachable —
                an environment problem, not a candidate's outcome.
        """
        self._eval_count += 1
        stage_results: dict[str, SimulationResult] = {}
        for stage_name in STAGE_NAMES:
            stage_action = action.get(stage_name)
            shape = self._shapes_by_name[stage_name]
            rows = gemm_shapes_to_topology_rows([shape])
            sram_kb = {
                "ifmap": stage_action.ifmap_frac * self._total_sram_kb,
                "filter": stage_action.filter_frac * self._total_sram_kb,
                "ofmap": stage_action.ofmap_frac * self._total_sram_kb,
            }
            run_dir = self.scratch_dir / f"eval{self._eval_count:05d}" / stage_name
            topology_path, layout_path, config_path = write_scale_sim_inputs(
                rows, self.hardware, stage_action.dataflow_mode, run_dir, sram_kb=sram_kb,
            )
            stage_results[stage_name] = run_scale_sim(
                topology_path, layout_path, config_path,
                run_name=f"{self.hardware.name}_{stage_action.dataflow_mode}", hardware=self.hardware,
                output_dir=run_dir / "output",
            )

        failed_stages = [s for s, r in stage_results.items() if r.status != "OK"]
        if failed_stages:
            failed = SimulationResult(
                exec_cycles=0, pe_utilization_pct=0.0, sram_dram_traffic_bytes=0,
                wall_clock_s=sum(r.wall_clock_s for r in stage_results.values()),
                status="FAILED", notes=f"stage(s) FAILED: {failed_stages}",
            )
            return EvalResult(action=action, simulation=failed, reward=self.failed_run_penalty_reward)

        combined = combine_simulation_results(list(stage_results.values()))
        traffic_per_mac = combined.sram_dram_traffic_bytes / self._total_macs
        reward = combined.pe_utilization_pct - self.traffic_penalty_weight * traffic_per_mac
        return EvalResult(action=action, simulation=combined, reward=reward)

    def evaluate_stage(self, stage_name: str, stage_action: StageAction, *, run_tag: str) -> SimulationResult:
        """Run one real simulation for a single GEMM stage's StageAction, without
        combining with the other stages.

        For callers that need to evaluate stages independently rather than
        through the fully-combined `evaluate()` — e.g.
        `common.optimizers.brute_force.brute_force_search`'s per-stage-independent
        grid search (a true full joint grid over all `len(STAGE_NAMES)` stages
        together is combinatorially infeasible — see that module's docstring).

        Args:
            stage_name: One of STAGE_NAMES.
            stage_action: The candidate StageAction for this stage.
            run_tag: Caller-supplied unique subdirectory name under scratch_dir
                (evaluate_stage does not share evaluate()'s auto-incrementing
                eval_count, so callers making many stage-only calls must supply
                their own distinct tags to avoid overwriting each other's inputs).

        Returns:
            The real, single-stage SimulationResult.

        Raises:
            RuntimeError: Propagated from run_scale_sim if Docker isn't reachable.
        """
        if stage_name not in STAGE_NAMES:
            raise ValueError(f"evaluate_stage: stage_name must be one of {STAGE_NAMES}, got {stage_name!r}.")
        shape = self._shapes_by_name[stage_name]
        rows = gemm_shapes_to_topology_rows([shape])
        sram_kb = {
            "ifmap": stage_action.ifmap_frac * self._total_sram_kb,
            "filter": stage_action.filter_frac * self._total_sram_kb,
            "ofmap": stage_action.ofmap_frac * self._total_sram_kb,
        }
        run_dir = self.scratch_dir / "_stage_evals" / run_tag / stage_name
        topology_path, layout_path, config_path = write_scale_sim_inputs(
            rows, self.hardware, stage_action.dataflow_mode, run_dir, sram_kb=sram_kb,
        )
        return run_scale_sim(
            topology_path, layout_path, config_path,
            run_name=f"{self.hardware.name}_{stage_action.dataflow_mode}", hardware=self.hardware,
            output_dir=run_dir / "output",
        )
