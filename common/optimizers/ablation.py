"""
ablation.py — restricted-action-space optimizer variants for Experiment 5's
ablation study (plan §6 Exp 5): isolate the contribution of joint dataflow+tiling
learning from tiling alone or dataflow alone.

Both variants use the same library/sampler as the selected `bayesian_opt`
candidate (optuna's TPE) and the same
`common.optimizers.environment.AttentionMappingProblem`/`Action`/`StageAction` —
only which dimensions of the action are searched vs. held fixed differs from
`common.optimizers.bayesian_opt.BayesianOptOptimizer`'s full joint search.

**Per-stage, revised 2026-09-02** (was one global dataflow/tiling choice — see
environment.py's module docstring for why). Both variants now act **per GEMM
stage** (`STAGE_NAMES`), matching the joint optimizer's own per-stage action
space:
  - `DataflowOnlyOptimizer` learns each stage's dataflow mode independently;
    every stage's SRAM partition is fixed at `DEFAULT_TILING_FRACS`.
  - `TilingOnlyOptimizer` learns each stage's SRAM partition independently;
    every stage's dataflow is fixed to that stage's own real best-static choice
    (`fixed_dataflow_modes: dict[stage_name, dataflow_mode]` — the caller passes
    in `common.baselines.static_heuristics.best_fixed_per_stage_mapping`'s real
    per-stage winners, not a single global "best-static" string as before, since
    that mapping already IS the per-stage best-static answer).

**"Tiling" here means the SRAM ifmap/filter/ofmap partition**, per this project's
established terminology since `common/optimizers/environment.py` — see that
module's docstring for why (SCALE-Sim exposes no separate explicit tile-size knob).
"""
from __future__ import annotations

import optuna

from common.optimizers.base import DataflowTilingOptimizer
from common.optimizers.environment import (
    STAGE_NAMES,
    Action,
    AttentionMappingProblem,
    EvalResult,
    StageAction,
)
from common.simulator_interface import SUPPORTED_DATAFLOW_MODES

optuna.logging.set_verbosity(optuna.logging.WARNING)

# "tiling fixed to a reasonable default" (plan §6 Exp 5) — an equal three-way SRAM
# split, the most neutral choice absent any other information (no buffer favored).
DEFAULT_TILING_FRACS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


class DataflowOnlyOptimizer(DataflowTilingOptimizer):
    """Learns each stage's dataflow mode (WS/OS/IS) independently; every stage's
    SRAM partition fixed at DEFAULT_TILING_FRACS."""

    def run(self, problem: AttentionMappingProblem, *, n_iterations: int, seed: int) -> list[EvalResult]:
        eval_results: list[EvalResult] = []

        def objective(trial: optuna.Trial) -> float:
            stages = {}
            for stage_name in STAGE_NAMES:
                dataflow_mode = trial.suggest_categorical(f"{stage_name}_dataflow", list(SUPPORTED_DATAFLOW_MODES))
                stages[stage_name] = StageAction(
                    dataflow_mode=dataflow_mode,
                    ifmap_frac=DEFAULT_TILING_FRACS[0], filter_frac=DEFAULT_TILING_FRACS[1],
                    ofmap_frac=DEFAULT_TILING_FRACS[2],
                )
            result = problem.evaluate(Action(stages=stages))
            eval_results.append(result)
            return result.reward

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_iterations)
        return eval_results


class TilingOnlyOptimizer(DataflowTilingOptimizer):
    """Learns each stage's SRAM partition independently; every stage's dataflow
    fixed to that stage's own real best-static choice."""

    def __init__(self, *, fixed_dataflow_modes: dict[str, str]) -> None:
        """
        Args:
            fixed_dataflow_modes: One dataflow mode per stage to hold fixed — plan
                §6 Exp 5: "dataflow fixed to best-static", now applied per stage
                (each stage's own empirically best static choice, e.g. from
                common.baselines.static_heuristics.best_fixed_per_stage_mapping's
                real winners for the cell being ablated — determined by the
                caller from real data, not assumed or hardcoded here).
        """
        if set(fixed_dataflow_modes) != set(STAGE_NAMES):
            raise ValueError(f"fixed_dataflow_modes must have exactly the keys {STAGE_NAMES}, got {sorted(fixed_dataflow_modes)}.")
        for stage_name, mode in fixed_dataflow_modes.items():
            if mode not in SUPPORTED_DATAFLOW_MODES:
                raise ValueError(f"fixed_dataflow_modes[{stage_name!r}] must be one of {SUPPORTED_DATAFLOW_MODES}, got {mode!r}.")
        self.fixed_dataflow_modes = dict(fixed_dataflow_modes)

    def run(self, problem: AttentionMappingProblem, *, n_iterations: int, seed: int) -> list[EvalResult]:
        eval_results: list[EvalResult] = []

        def objective(trial: optuna.Trial) -> float:
            stages = {}
            for stage_name in STAGE_NAMES:
                # Same stick-breaking parameterization as decode_raw_action's SRAM
                # dimensions, applied directly (not via decode_raw_action itself,
                # since the dataflow-bin dimension isn't part of this variant's
                # search space).
                ifmap_frac = trial.suggest_float(f"{stage_name}_ifmap_frac", 0.0, 1.0)
                filter_frac = (1.0 - ifmap_frac) * trial.suggest_float(f"{stage_name}_filter_split", 0.0, 1.0)
                ofmap_frac = 1.0 - ifmap_frac - filter_frac
                stages[stage_name] = StageAction(
                    dataflow_mode=self.fixed_dataflow_modes[stage_name],
                    ifmap_frac=ifmap_frac, filter_frac=filter_frac, ofmap_frac=ofmap_frac,
                )
            result = problem.evaluate(Action(stages=stages))
            eval_results.append(result)
            return result.reward

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_iterations)
        return eval_results
