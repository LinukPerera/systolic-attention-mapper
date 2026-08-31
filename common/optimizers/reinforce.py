"""
optimizers/reinforce.py — REINFORCE policy-gradient candidate for dataflow+tiling search.

Uses stable-baselines3's REINFORCE-family implementation (plan §2: "use established,
citable implementations rather than reimplementing from scratch"). State/action/
reward design must be documented here once implemented, per plan §3 — e.g. state =
[workload features, hardware features], action = (dataflow_mode, tile sizes),
reward = utilization minus a memory-traffic penalty term (see plan §3's example).

STATUS: stub — one of four candidates evaluated in Experiment 0; only the winner is
carried forward into Experiments 2–6 (plan §6, Exp 0).
"""
from __future__ import annotations

from typing import Any

from common.optimizers.base import DataflowTilingOptimizer


class ReinforceOptimizer(DataflowTilingOptimizer):
    """REINFORCE candidate. See module docstring for state/action/reward design (TODO)."""

    def reset(self, *, seed: int) -> Any:
        raise NotImplementedError

    def step(self) -> dict:
        raise NotImplementedError

    def is_converged(self) -> bool:
        raise NotImplementedError
