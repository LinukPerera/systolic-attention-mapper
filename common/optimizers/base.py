"""
optimizers/base.py — common interface every candidate optimizer implements.

Defines the state/action/reward contract so reinforce.py, ppo.py, evolutionary.py,
and bayesian_opt.py are interchangeable from the experiment scripts' point of view
(plan §1). The state/action/reward design itself belongs in each concrete
optimizer's module and must be documented there per plan §3.

STATUS: stub — interface only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataflowTilingOptimizer(ABC):
    """Common interface for a dataflow+tiling search/learning method.

    Concrete subclasses (REINFORCE, PPO, evolutionary, Bayesian optimization) each
    define their own state/action representation internally but expose this same
    reset/step contract so experiment scripts don't need to know which one is in use.
    """

    @abstractmethod
    def reset(self, *, seed: int) -> Any:
        """Reset the optimizer for a fresh run with the given seed. Returns initial state."""
        raise NotImplementedError

    @abstractmethod
    def step(self) -> dict:
        """Run one iteration (one candidate mapping proposed + evaluated + optimizer updated).

        Returns:
            A dict with at least {"iteration", "reward", "mapping", "metrics"} —
            logged verbatim via common.logging_utils.append_run_row by the caller.
        """
        raise NotImplementedError

    @abstractmethod
    def is_converged(self) -> bool:
        """Whether this run has met Experiment 2's stated convergence criterion.

        Must return False until Experiment 2 has defined and configured a concrete
        criterion (plan §6, Exp 2 acceptance criteria: "stated as a formula, not
        just 'it looked flat'").
        """
        raise NotImplementedError
