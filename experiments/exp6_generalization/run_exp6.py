"""
Experiment 6 -- Generalization test (optional/stretch)

Test whether the learned policy generalizes to unseen workload shapes, or has effectively memorized per-shape optima.

See ../../docs/CODING_AGENT_PLAN.md §6 (Experiment 6) for the full spec
(variables, procedure, required outputs, acceptance criteria) and this folder's
exp6_stats.md for the filled-in report once this experiment has been run.

STATUS: stub -- not yet implemented. Do not run until the procedure below is filled
in; a stub run must not produce numbers (plan ground rule 1: never fabricate a result).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import logging_utils
# from common.optimizers.<selected> import <SelectedOptimizer>  # TODO: fill in after Exp 0

EXPERIMENT_ID = "exp6"
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw" / f"{EXPERIMENT_ID}.csv"


def parse_args() -> argparse.Namespace:
    """CLI arguments for this experiment. Extend per the variables listed in exp6_stats.md §2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to this experiment's config YAML.")
    parser.add_argument(
        "--seed", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Random seed(s) to run (>=5 for any stochastic result, per plan ground rule 3).",
    )
    parser.add_argument("--out", type=str, default=str(RAW_CSV), help="Raw results CSV to append to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"{EXPERIMENT_ID}: implement the procedure in exp6_stats.md §3 / "
        "../../docs/CODING_AGENT_PLAN.md §6 before running. "
        "Never substitute fabricated numbers for a real run (plan ground rule 1)."
    )


if __name__ == "__main__":
    main()
