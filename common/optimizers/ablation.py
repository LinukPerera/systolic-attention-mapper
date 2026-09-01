"""
ablation.py — restricted-action-space optimizer variants for Experiment 5's
ablation study (plan §6 Exp 5): isolate the contribution of joint dataflow+tiling
learning from tiling alone or dataflow alone.

Both variants use the same library/sampler as the selected `bayesian_opt`
candidate (optuna's TPE) and the same
`common.optimizers.environment.AttentionMappingProblem`/`Action` — only which
dimensions of the action are searched vs. held fixed differs from
`common.optimizers.bayesian_opt.BayesianOptOptimizer`'s full joint search. This
isolates the effect of the *action space*, not a different search algorithm.

**"Tiling" here means the SRAM ifmap/filter/ofmap partition**, per this project's
established terminology since `common/optimizers/environment.py` — see that
module's docstring for why (SCALE-Sim exposes no separate explicit tile-size knob).
"""
from __future__ import annotations

import optuna

from common.optimizers.base import DataflowTilingOptimizer
from common.optimizers.environment import Action, AttentionMappingProblem, EvalResult
from common.simulator_interface import SUPPORTED_DATAFLOW_MODES

optuna.logging.set_verbosity(optuna.logging.WARNING)

# "tiling fixed to a reasonable default" (plan §6 Exp 5) — an equal three-way SRAM
# split, the most neutral choice absent any other information (no buffer favored).
DEFAULT_TILING_FRACS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


class DataflowOnlyOptimizer(DataflowTilingOptimizer):
    """Learns ONLY the dataflow mode (WS/OS/IS); SRAM partition fixed at DEFAULT_TILING_FRACS."""

    def run(self, problem: AttentionMappingProblem, *, n_iterations: int, seed: int) -> list[EvalResult]:
        eval_results: list[EvalResult] = []

        def objective(trial: optuna.Trial) -> float:
            dataflow_mode = trial.suggest_categorical("dataflow_mode", list(SUPPORTED_DATAFLOW_MODES))
            action = Action(
                dataflow_mode=dataflow_mode,
                ifmap_frac=DEFAULT_TILING_FRACS[0], filter_frac=DEFAULT_TILING_FRACS[1],
                ofmap_frac=DEFAULT_TILING_FRACS[2],
            )
            result = problem.evaluate(action)
            eval_results.append(result)
            return result.reward

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_iterations)
        return eval_results


class TilingOnlyOptimizer(DataflowTilingOptimizer):
    """Learns ONLY the SRAM partition; dataflow mode fixed to a given "best-static" choice."""

    def __init__(self, *, fixed_dataflow_mode: str) -> None:
        """
        Args:
            fixed_dataflow_mode: The dataflow mode to hold fixed — plan §6 Exp 5:
                "dataflow fixed to best-static", i.e. whichever single-dataflow
                baseline was empirically best for the cell being ablated
                (determined by the caller from real Experiment 3 data, not
                assumed or hardcoded here).
        """
        if fixed_dataflow_mode not in SUPPORTED_DATAFLOW_MODES:
            raise ValueError(
                f"fixed_dataflow_mode must be one of {SUPPORTED_DATAFLOW_MODES}, got {fixed_dataflow_mode!r}."
            )
        self.fixed_dataflow_mode = fixed_dataflow_mode

    def run(self, problem: AttentionMappingProblem, *, n_iterations: int, seed: int) -> list[EvalResult]:
        eval_results: list[EvalResult] = []

        def objective(trial: optuna.Trial) -> float:
            # Same stick-breaking parameterization as decode_raw_action's SRAM
            # dimensions, applied directly (not via decode_raw_action itself, since
            # the dataflow-bin dimension isn't part of this variant's search space).
            ifmap_frac = trial.suggest_float("ifmap_frac", 0.0, 1.0)
            filter_frac = (1.0 - ifmap_frac) * trial.suggest_float("filter_split", 0.0, 1.0)
            ofmap_frac = 1.0 - ifmap_frac - filter_frac
            action = Action(
                dataflow_mode=self.fixed_dataflow_mode,
                ifmap_frac=ifmap_frac, filter_frac=filter_frac, ofmap_frac=ofmap_frac,
            )
            result = problem.evaluate(action)
            eval_results.append(result)
            return result.reward

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_iterations)
        return eval_results
