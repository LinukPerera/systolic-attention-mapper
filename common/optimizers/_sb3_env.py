"""
_sb3_env.py — shared Gymnasium environment wrapping AttentionMappingProblem for the
stable-baselines3-backed candidates (a2c.py, ppo.py).

Modeled as a stateless, one-step-per-episode bandit: every step() call evaluates one
real candidate Action via problem.evaluate() and immediately terminates the episode
— this project's search problem has no meaningful state to carry between
evaluations (the same fixed workload/hardware is optimized throughout), so each SB3
"timestep" corresponds to exactly one real simulator call, and total_timesteps
passed to .learn() is exactly the iteration budget in problem.evaluate() calls.

Action encoding: Box(shape=(3,)), decoded via
common.optimizers.environment.decode_raw_action — see that function's docstring for
why this exact encoding is shared across all four optimizer candidates.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from common.optimizers.environment import AttentionMappingProblem, EvalResult, decode_raw_action


class AttentionMappingGymEnv(gym.Env):
    """One-step-per-episode Gymnasium env around a real AttentionMappingProblem."""

    metadata: dict = {"render_modes": []}

    def __init__(self, problem: AttentionMappingProblem) -> None:
        super().__init__()
        self.problem = problem
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        # Constant dummy observation — there is no state to observe (see module docstring).
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.eval_results: list[EvalResult] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, raw_action: np.ndarray):
        action = decode_raw_action(tuple(float(x) for x in raw_action))
        result = self.problem.evaluate(action)
        self.eval_results.append(result)
        obs = np.zeros(1, dtype=np.float32)
        terminated = True
        truncated = False
        return obs, float(result.reward), terminated, truncated, {"eval_result": result}
