"""
optimizers/brute_force.py — exhaustive grid search, used only to establish a
ground-truth optimum for Experiment 0's "distance from best" comparison (plan §6,
Experiment 0: "also run brute-force/exhaustive search on this one small case to
establish a ground-truth optimum"). Deterministic — no seed, not a
DataflowTilingOptimizer candidate (it isn't budget-constrained the same way; it
runs one full grid instead of taking an iteration budget).

**Does not use decode_raw_action / the shared 3-real-valued encoding.** A coarse
grid over that [0,1]^3 encoded space can silently skip a dataflow mode entirely
depending on grid alignment — verified by hand: with 4 grid points on r0, the
values [0, 0.333, 0.667, 1.0] bin to WS, WS, IS, IS under decode_raw_action's
binning (`int(r0 * 3)`), never landing on the OS bin ([1/3, 2/3)). That's a real,
confirmed bug avoided here, not a hypothetical one — ground truth must exhaustively
cover the actual decision space (every dataflow mode, explicitly), so this module
builds Action objects directly instead of decoding a proxy encoding.
"""
from __future__ import annotations

from common.optimizers.environment import Action, AttentionMappingProblem, EvalResult
from common.simulator_interface import SUPPORTED_DATAFLOW_MODES


def _simplex_grid_points(divisions: int) -> list[tuple[float, float, float]]:
    """All (ifmap_frac, filter_frac, ofmap_frac) triples on a `divisions`-step grid
    over the 2-simplex (non-negative, summing to exactly 1). (divisions+1)*(divisions+2)/2 points."""
    points = []
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            k = divisions - i - j
            points.append((i / divisions, j / divisions, k / divisions))
    return points


def brute_force_search(problem: AttentionMappingProblem, *, sram_grid_divisions: int) -> list[EvalResult]:
    """Exhaustively evaluate every dataflow mode x SRAM-simplex-grid-point combination.

    Total real simulator calls: len(SUPPORTED_DATAFLOW_MODES) *
    (sram_grid_divisions+1)*(sram_grid_divisions+2)/2. This is a *coarse-grid*
    exhaustive search, not a true continuous optimum — sram_grid_divisions controls
    the coarseness/wall-clock trade-off explicitly; document the chosen value (and
    that this is a discretized, not continuous, ground truth) in the calling
    experiment's stats.md per plan ground rule 4 (facts vs. limitations kept separate).

    Args:
        problem: The (workload, hardware) search problem.
        sram_grid_divisions: Number of steps per SRAM-fraction grid axis.

    Returns:
        One EvalResult per (dataflow, grid point) combination evaluated, in
        evaluation order.
    """
    eval_results: list[EvalResult] = []
    for dataflow_mode in SUPPORTED_DATAFLOW_MODES:
        for ifmap_frac, filter_frac, ofmap_frac in _simplex_grid_points(sram_grid_divisions):
            action = Action(
                dataflow_mode=dataflow_mode, ifmap_frac=ifmap_frac,
                filter_frac=filter_frac, ofmap_frac=ofmap_frac,
            )
            eval_results.append(problem.evaluate(action))
    return eval_results
