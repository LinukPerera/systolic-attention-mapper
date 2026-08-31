"""
optimizers/bayesian_opt.py — Bayesian-optimization candidate for dataflow+tiling search.

Uses `optuna` or `scikit-optimize` (plan §2 — pick one and note the choice here once
implemented). Search-space definition and acquisition function must be documented
here once implemented, per plan §3.

STATUS: stub — one of four candidates evaluated in Experiment 0; only the winner is
carried forward into Experiments 2–6 (plan §6, Exp 0).
"""
from __future__ import annotations

from typing import Any

from common.optimizers.base import DataflowTilingOptimizer


class BayesianOptOptimizer(DataflowTilingOptimizer):
    """Bayesian-optimization candidate. See module docstring for search-space design (TODO)."""

    def reset(self, *, seed: int) -> Any:
        raise NotImplementedError

    def step(self) -> dict:
        raise NotImplementedError

    def is_converged(self) -> bool:
        raise NotImplementedError
