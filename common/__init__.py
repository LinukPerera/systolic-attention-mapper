"""
common/ — shared, experiment-independent library.

Every experiment under experiments/<expN>/ imports from here rather than
reimplementing simulator plumbing, metrics formulas, baselines, optimizers, or
plotting style. This keeps aggregation/formula logic written once and trusted
everywhere (see CODING_AGENT_PLAN.md §3, "one function = one responsibility").

Nothing in this package runs an experiment on import — it is only ever imported.
"""
