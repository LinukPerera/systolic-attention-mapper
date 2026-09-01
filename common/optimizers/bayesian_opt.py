"""
optimizers/bayesian_opt.py — Bayesian-optimization candidate for dataflow+SRAM-partition search.

Uses `optuna` (plan §2 offered optuna or scikit-optimize; optuna chosen since its
Tree-structured Parzen Estimator sampler needs no separate scikit-learn surrogate
setup and its `study.optimize(objective, n_trials=...)` loop maps directly onto
this project's "one real simulator call per iteration" contract).

Search space: `3 * len(STAGE_NAMES)` (= 9) floats, each `trial.suggest_float(name,
0.0, 1.0)`, decoded via common.optimizers.environment.decode_raw_action — the same
encoding every other candidate searches, for a fair bake-off. (Revised 2026-09-02
from a 3-float search space to the per-stage action space — see environment.py's
module docstring.)

Objective: each trial's real EvalResult.reward, from one real problem.evaluate()
call — optuna's TPE sampler fits its surrogate model to these real rewards, it
never sees or optimizes a synthetic stand-in.
"""
from __future__ import annotations

import optuna

from common.optimizers.base import DataflowTilingOptimizer
from common.optimizers.environment import STAGE_NAMES, AttentionMappingProblem, EvalResult, decode_raw_action

optuna.logging.set_verbosity(optuna.logging.WARNING)  # one line of INFO log per real trial is too noisy over 100s of trials

_ACTION_DIM = 3 * len(STAGE_NAMES)


class BayesianOptOptimizer(DataflowTilingOptimizer):
    """Bayesian-optimization candidate (TPE sampler)."""

    def run(self, problem: AttentionMappingProblem, *, n_iterations: int, seed: int) -> list[EvalResult]:
        eval_results: list[EvalResult] = []

        def objective(trial: optuna.Trial) -> float:
            raw = tuple(trial.suggest_float(f"r{i}", 0.0, 1.0) for i in range(_ACTION_DIM))
            result = problem.evaluate(decode_raw_action(raw))
            eval_results.append(result)
            return result.reward

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_iterations)
        return eval_results
