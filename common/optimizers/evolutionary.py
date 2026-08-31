"""
optimizers/evolutionary.py — evolutionary/genetic-algorithm candidate for dataflow+tiling search.

Uses `deap` (plan §2). Individual/genome encoding, mutation/crossover operators, and
fitness function must be documented here once implemented, per plan §3.

STATUS: stub — one of four candidates evaluated in Experiment 0; only the winner is
carried forward into Experiments 2–6 (plan §6, Exp 0).
"""
from __future__ import annotations

from typing import Any

from common.optimizers.base import DataflowTilingOptimizer


class EvolutionaryOptimizer(DataflowTilingOptimizer):
    """GA candidate. See module docstring for genome/operator/fitness design (TODO)."""

    def reset(self, *, seed: int) -> Any:
        raise NotImplementedError

    def step(self) -> dict:
        raise NotImplementedError

    def is_converged(self) -> bool:
        raise NotImplementedError
