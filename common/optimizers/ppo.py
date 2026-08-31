"""
optimizers/ppo.py — PPO candidate for dataflow+SRAM-partition search.

Uses stable-baselines3's PPO (plan §2). Action/state/reward design: see
common/optimizers/_sb3_env.py and common/optimizers/environment.py. One real
problem.evaluate() call per env step, so n_iterations passed to run() is exactly
the number of real SCALE-Sim invocations.

n_steps/batch_size/n_epochs are kept small relative to PPO's usual defaults
(n_steps=2048) specifically because this project's pilot iteration budgets are tiny
(tens, not thousands, of real simulator calls per run — each is an actual Docker
invocation) — PPO needs at least one full rollout of n_steps before its first
update, so n_steps must not exceed n_iterations or PPO would never update at all.
"""
from __future__ import annotations

from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from common.optimizers._sb3_env import AttentionMappingGymEnv
from common.optimizers.base import DataflowTilingOptimizer
from common.optimizers.environment import AttentionMappingProblem, EvalResult


class PPOOptimizer(DataflowTilingOptimizer):
    """PPO candidate."""

    def __init__(self, *, n_steps: int = 5, batch_size: int = 5, n_epochs: int = 2, learning_rate: float = 3e-4) -> None:
        """
        Args:
            n_steps: PPO's rollout length before each update — from config/optimizers/ppo.yaml.
            batch_size: PPO's minibatch size — from config/optimizers/ppo.yaml; must
                divide n_steps evenly (SB3 requirement).
            n_epochs: PPO's optimization epochs per rollout — from config/optimizers/ppo.yaml.
            learning_rate: SB3 PPO's learning rate — from config/optimizers/ppo.yaml.
        """
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    def run(self, problem: AttentionMappingProblem, *, n_iterations: int, seed: int) -> list[EvalResult]:
        env = AttentionMappingGymEnv(problem)
        vec_env = DummyVecEnv([lambda: env])
        model = SB3_PPO(
            "MlpPolicy", vec_env, n_steps=self.n_steps, batch_size=self.batch_size,
            n_epochs=self.n_epochs, learning_rate=self.learning_rate, seed=seed, verbose=0,
        )
        model.learn(total_timesteps=n_iterations)
        return env.eval_results
