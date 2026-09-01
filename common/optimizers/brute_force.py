"""
optimizers/brute_force.py — exhaustive grid search, used only to establish a
ground-truth optimum for Experiment 0's "distance from best" comparison (plan §6,
Experiment 0: "also run brute-force/exhaustive search on this one small case to
establish a ground-truth optimum"). Deterministic — no seed, not a
DataflowTilingOptimizer candidate (it isn't budget-constrained the same way; it
runs one full grid instead of taking an iteration budget).

**Does not use decode_raw_action / the shared 9-real-valued encoding.** A coarse
grid over that [0,1]^9 encoded space can silently skip a dataflow mode entirely
depending on grid alignment — verified by hand: with 4 grid points on a stage's
r0, the values [0, 0.333, 0.667, 1.0] bin to WS, WS, IS, IS under
decode_raw_action's binning (`int(r0 * 3)`), never landing on the OS bin
([1/3, 2/3)). That's a real, confirmed bug avoided here, not a hypothetical one —
ground truth must exhaustively cover the actual decision space (every dataflow
mode, explicitly, per stage), so this module builds StageAction/Action objects
directly instead of decoding a proxy encoding.

**Per-stage, revised 2026-09-02** (was one global dataflow/tiling grid over the
whole workload — see environment.py's module docstring for why the action space
moved to per-stage). A true full *joint* grid over all `len(STAGE_NAMES)` (=3)
stages together — i.e. every combination of (dataflow x SRAM-grid-point) for
stage A times every combination for stage B times every combination for stage C
— is combinatorially infeasible: with `sram_grid_divisions=3` (10 simplex points)
that's `3 dataflows * 10 points = 30` candidates per stage, and
`30 ** 3 = 27,000` real Docker/SCALE-Sim calls for one ground-truth data point.
Measured single-stage call costs (seconds-to-minutes each, see exp*_stats.md
timing notes) make that unrunnable.

Instead this module grids each stage **independently** — exactly the philosophy
`common.baselines.static_heuristics.best_fixed_per_stage_mapping` already uses
for its own (dataflow-only, non-learned) baseline — selecting each stage's
lowest-EDP (dataflow, SRAM-partition) grid point via
`problem.evaluate_stage()`, then combining the per-stage winners into one
`Action` and evaluating it once more through the real `problem.evaluate()` to get
the properly-computed combined reward. This costs `3 * 30 = 90` real per-stage
calls plus 1 combining call for divisions=3 — the same order of magnitude
`best_fixed_per_stage_mapping` itself already costs (`3 * 3 = 9` calls,
dataflow-only, no SRAM grid).

**This is an approximation of the true global joint optimum, not a guarantee of
it** — stage choices interact through the combined result (exec_cycles sum,
pe_utilization_pct becomes a weighted mean, traffic sums), so a per-stage-optimal
combination is not provably the joint-optimal one in general. It is the same
approximation `best_fixed_per_stage_mapping` already makes and reports as a
baseline, applied here with an added SRAM-partition grid for genuine
exhaustiveness within each stage. Document this explicitly as a [LIMITATION]
wherever Experiment 0's "distance from best" uses this module's output as ground
truth.

**Return contract, and why it isn't a flat `list[EvalResult]` any more**: the
per-stage grid candidates (`stage_grid_attempts`) are real evidence (ground rule
1: never hide a real run, including FAILED ones) but their `SimulationResult`s
are single-stage-only — not run through the combined reward formula, and NOT
comparable to `best`'s reward or to any other optimizer's combined reward. Mixing
them into one list a caller `max()`s over (as the pre-per-stage version of this
module's caller did) would silently let an inflated single-stage utilization
figure masquerade as the combined ground truth. `BruteForceResult` keeps them
separate, mirroring `BestFixedPerStageResult.winners` vs. `.all_attempts`.
"""
from __future__ import annotations

from dataclasses import dataclass

from common.metrics import compute_edp
from common.optimizers.environment import (
    STAGE_NAMES,
    Action,
    AttentionMappingProblem,
    EvalResult,
    StageAction,
)
from common.simulator_interface import SUPPORTED_DATAFLOW_MODES, SimulationResult


def _simplex_grid_points(divisions: int) -> list[tuple[float, float, float]]:
    """All (ifmap_frac, filter_frac, ofmap_frac) triples on a `divisions`-step grid
    over the 2-simplex (non-negative, summing to exactly 1). (divisions+1)*(divisions+2)/2 points."""
    points = []
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            k = divisions - i - j
            points.append((i / divisions, j / divisions, k / divisions))
    return points


@dataclass(frozen=True)
class StageGridAttempt:
    """One real (stage, dataflow, SRAM-grid-point) candidate actually evaluated.
    `result.status` may be FAILED — that's real evidence, not filtered out here."""
    stage_name: str
    stage_action: StageAction
    result: SimulationResult


@dataclass(frozen=True)
class BruteForceResult:
    """See module docstring's "Return contract" section for why this isn't a flat
    `list[EvalResult]`.

    Attributes:
        stage_grid_attempts: Every real per-stage grid candidate evaluated —
            evidence only, single-stage-only rewards, NOT comparable across
            stages or to `best`. Log these under a distinct tag if logging to a
            results CSV (not mixed into the same ground-truth-max column).
        best: The one real EvalResult from evaluating the combined
            best-per-stage Action through problem.evaluate() — the actual
            ground-truth data point to use. None if at least one stage had zero
            successful (dataflow, grid-point) candidates (real, total per-stage
            failure — mirrors best_fixed_per_stage_mapping's own partial-result
            contract; check this before using `best`).
    """
    stage_grid_attempts: list[StageGridAttempt]
    best: EvalResult | None


def brute_force_search(problem: AttentionMappingProblem, *, sram_grid_divisions: int) -> BruteForceResult:
    """Per-stage-independent exhaustive grid search + one combining evaluate() call.

    See module docstring for why this is per-stage-independent rather than a true
    joint grid, and for the resulting [LIMITATION].

    Args:
        problem: The real (workload, hardware) problem to search.
        sram_grid_divisions: Number of steps per SRAM-simplex edge (see
            _simplex_grid_points) — `(divisions+1)*(divisions+2)/2` grid points
            per dataflow mode, per stage.

    Returns:
        A BruteForceResult (see its docstring).

    Raises:
        RuntimeError: Propagated from run_scale_sim only if Docker itself isn't
            available/reachable.
    """
    stage_grid_attempts: list[StageGridAttempt] = []
    best_stage_action: dict[str, StageAction] = {}
    best_stage_edp: dict[str, float] = {}

    for stage_name in STAGE_NAMES:
        for dataflow_mode in SUPPORTED_DATAFLOW_MODES:
            for i, (ifmap_frac, filter_frac, ofmap_frac) in enumerate(_simplex_grid_points(sram_grid_divisions)):
                stage_action = StageAction(
                    dataflow_mode=dataflow_mode,
                    ifmap_frac=ifmap_frac, filter_frac=filter_frac, ofmap_frac=ofmap_frac,
                )
                run_tag = f"{stage_name}_{dataflow_mode}_{i:04d}"
                result = problem.evaluate_stage(stage_name, stage_action, run_tag=run_tag)
                stage_grid_attempts.append(StageGridAttempt(
                    stage_name=stage_name, stage_action=stage_action, result=result,
                ))

                if result.status != "OK":
                    # A candidate that fails to simulate can't win it, but stays in
                    # stage_grid_attempts as real evidence (ground rule 1).
                    continue
                edp = compute_edp(result, energy_per_cycle_pj=problem.hardware.energy_per_cycle_pj)
                if stage_name not in best_stage_edp or edp < best_stage_edp[stage_name]:
                    best_stage_edp[stage_name] = edp
                    best_stage_action[stage_name] = stage_action

    best: EvalResult | None = None
    if len(best_stage_action) == len(STAGE_NAMES):
        # Every stage had at least one successful candidate — combine the winners
        # and get the real, properly-computed combined reward via evaluate()
        # (never hand-computed here — same reward formula every other optimizer's
        # results go through).
        best = problem.evaluate(Action(stages=best_stage_action))

    return BruteForceResult(stage_grid_attempts=stage_grid_attempts, best=best)
