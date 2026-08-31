"""
optimizers/a2c.py — A2C candidate for dataflow+SRAM-partition search.

**Substitutes for "REINFORCE"** in CODING_AGENT_PLAN.md §2's optimizer list:
verified 2026-08-31 that stable-baselines3 (2.9.0) does not ship a plain REINFORCE
implementation — its algorithms are A2C, DDPG, DQN, HER, PPO, SAC, TD3. A2C
(Advantage Actor-Critic) is the closest established, citable, SB3-native
policy-gradient method — a REINFORCE-with-baseline variant — and stands in for it
here. If the methodology chapter names "REINFORCE" specifically, either rename it to
A2C or replace this file with a hand-rolled vanilla REINFORCE update loop.

Action/state/reward design: see common/optimizers/_sb3_env.py and
common/optimizers/environment.py. One real problem.evaluate() call per env step,
so n_iterations passed to run() is exactly the number of real SCALE-Sim invocations.
"""
from __future__ import annotations

from stable_baselines3 import A2C as SB3_A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from common.optimizers._sb3_env import AttentionMappingGymEnv
from common.optimizers.base import DataflowTilingOptimizer
from common.optimizers.environment import AttentionMappingProblem, EvalResult


class A2COptimizer(DataflowTilingOptimizer):
    """A2C candidate (see module docstring for why A2C, not REINFORCE)."""

    def __init__(self, *, n_steps: int = 5, learning_rate: float = 7e-4) -> None:
        """
        Args:
            n_steps: A2C's rollout length before each policy update — from
                config/optimizers/a2c.yaml, kept small so at least a few real policy
                updates happen within a small pilot iteration budget.
            learning_rate: SB3 A2C's learning rate — from config/optimizers/a2c.yaml.
        """
        self.n_steps = n_steps
        self.learning_rate = learning_rate

    def run(self, problem: AttentionMappingProblem, *, n_iterations: int, seed: int) -> list[EvalResult]:
        env = AttentionMappingGymEnv(problem)
        vec_env = DummyVecEnv([lambda: env])
        model = SB3_A2C(
            "MlpPolicy", vec_env, n_steps=self.n_steps, learning_rate=self.learning_rate,
            seed=seed, verbose=0,
        )
        model.learn(total_timesteps=n_iterations)
        return env.eval_results
