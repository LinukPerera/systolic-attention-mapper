"""
optimizers/base.py — common interface every candidate optimizer implements.

**Revised 2026-08-31** from Stage 0's speculative reset()/step()/is_converged()
design: that design assumed each optimizer exposes its own iteration loop, but the
two library-backed candidates (A2C/PPO via stable-baselines3) own their training
loop internally (`.learn(total_timesteps=...)`) and only expose per-step results via
a callback — there is no clean way to single-step them from outside without
reimplementing their update logic. Rather than force an awkward fit (and risk a
subtly wrong per-step accounting for a library that manages its own state), every
optimizer instead exposes one `run()` that performs its whole budgeted search and
returns every real, individually-simulated candidate it evaluated along the way, in
evaluation order — this is what run_expN.py scripts actually need to log (one raw
CSV row per (exp, config, seed, iteration), per CODING_AGENT_PLAN.md §4).
is_converged() is dropped here too: it belonged to Experiment 2's not-yet-defined
convergence criterion and doesn't need to live on this interface — Experiment 2 can
compute it directly from the logged reward trajectory in results/raw/, the same way
every other aggregate statistic in this project is computed (in logging_utils /
figures, from the raw CSV, not inside the optimizer that produced it).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from common.optimizers.environment import AttentionMappingProblem, EvalResult


class DataflowTilingOptimizer(ABC):
    """Common interface for a dataflow+SRAM-partition search/learning method.

    Concrete subclasses (A2C, PPO, evolutionary, Bayesian optimization) each define
    their own internal state/policy representation but expose this one `run()`
    contract so experiment scripts don't need to know which one is in use.
    """

    @abstractmethod
    def run(self, problem: AttentionMappingProblem, *, n_iterations: int, seed: int) -> list[EvalResult]:
        """Run this optimizer's full budgeted search once against a real problem.

        Every EvalResult returned corresponds to one real, individually-run
        SCALE-Sim simulation via problem.evaluate() — there is no interpolation,
        caching across calls, or synthetic reward standing in for a real one
        (plan ground rule 1).

        Args:
            problem: The (workload, hardware) search problem to optimize.
            n_iterations: Number of candidate actions to evaluate (the "iteration
                budget" plan §6 Experiment 0 refers to). What one "iteration" means
                internally is candidate-specific (one env step for A2C/PPO, one
                generation's population for the evolutionary candidate, one trial
                for Bayesian optimization) but each must correspond to exactly one
                real problem.evaluate() call, so n_iterations means the same thing
                (one real simulation) across every candidate.
            seed: Random seed — must seed this optimizer's own RNG (and, where
                applicable, its underlying library's RNG) completely, so re-running
                with the same seed reproduces the same sequence of evaluated actions.

        Returns:
            One EvalResult per evaluated candidate, in evaluation order.

        Raises:
            RuntimeError: Propagated from problem.evaluate() if Docker isn't
                reachable — an environment problem, not a run outcome.
        """
        raise NotImplementedError
