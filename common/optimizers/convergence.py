"""
convergence.py — the convergence criterion Experiment 2 defines and Experiments
3/4 reuse (plan §6: Exp 4 "reuse[s] Exp 2's convergence criterion to decide when
each run is 'done'"; Exp 3 needs the same to decide each optimizer seed's stopping
point). Shared here so it is written once and trusted everywhere (plan §3), not
reimplemented slightly differently per experiment script.

**Criterion:** a search is CONVERGED at the first iteration i >= WINDOW such that

    Var(best_reward_so_far[i-WINDOW+1 : i+1]) < EPSILON

where best_reward_so_far is the running max of the raw per-iteration reward
sequence (monotonically non-decreasing by construction). First stated in
experiments/exp2_convergence/run_exp2.py.

**EPSILON = 0.1 (revised from an initial 1.0):** Experiment 2's original run
(`experiments/exp2_convergence/exp2_stats.md` §4/§8, EPSILON=1.0) found the
criterion triggered "converged" at iteration 4 while every seed's trajectory kept
improving until ~iteration 25 — EPSILON=1.0 was ≈3.6% of the observed reward
range, loose enough that an early plateau within a 1.0-unit band satisfied it.
EPSILON=0.1 (≈0.36% of that same range) was chosen once, deliberately, before
re-analyzing Experiment 2's existing raw data with it (not tuned after looking at
what a "nicer-looking" convergence point would be) — see
`experiments/exp2_convergence/exp2_stats.md`'s epsilon-revision addendum for the
full before/after comparison. WINDOW is left unchanged at 5 so exactly one
variable changes between the two analyses.
"""
from __future__ import annotations

import numpy as np

DEFAULT_WINDOW = 5
DEFAULT_EPSILON = 0.1


def best_reward_so_far(rewards) -> np.ndarray:
    """Running max of a reward sequence (monotonically non-decreasing).

    Args:
        rewards: Sequence of raw per-iteration rewards, in iteration order.

    Returns:
        The running-max array, same length as `rewards`.
    """
    return np.maximum.accumulate(np.asarray(rewards, dtype=float))


def convergence_iteration(
    best_so_far,
    *,
    window: int = DEFAULT_WINDOW,
    epsilon: float = DEFAULT_EPSILON,
) -> int | None:
    """First iteration i (0-indexed) such that Var(best_so_far[i-window+1:i+1]) < epsilon.

    Args:
        best_so_far: A best-reward-so-far trajectory (see best_reward_so_far()).
        window: Trailing window size.
        epsilon: Variance threshold.

    Returns:
        The 0-indexed iteration at which the criterion is first met, or None if
        it is never met within the given trajectory — a real "did not converge"
        result, never silently treated as 0 or discarded by callers (plan
        ground rule 1 / §6 Exp 2 acceptance criteria).
    """
    best_so_far = np.asarray(best_so_far, dtype=float)
    for i in range(window - 1, len(best_so_far)):
        window_vals = best_so_far[i - window + 1: i + 1]
        if np.var(window_vals) < epsilon:
            return i
    return None
