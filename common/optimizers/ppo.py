"""
optimizers/ppo.py — PPO candidate for dataflow+tiling search.

Uses stable-baselines3's PPO implementation (plan §2). State/action/reward design
must be documented here once implemented, per plan §3.

STATUS: stub — one of four candidates evaluated in Experiment 0; only the winner is
carried forward into Experiments 2–6 (plan §6, Exp 0).
"""
from __future__ import annotations

from typing import Any

from common.optimizers.base import DataflowTilingOptimizer


class PPOOptimizer(DataflowTilingOptimizer):
    """PPO candidate. See module docstring for state/action/reward design (TODO)."""

    def reset(self, *, seed: int) -> Any:
        raise NotImplementedError

    def step(self) -> dict:
        raise NotImplementedError

    def is_converged(self) -> bool:
        raise NotImplementedError
