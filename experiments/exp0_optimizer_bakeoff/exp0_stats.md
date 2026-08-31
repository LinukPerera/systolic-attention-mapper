# Experiment 0 Stats Report -- Optimizer bake-off

> Copied from `../../docs/EXPERIMENT_STATS_TEMPLATE.md`. **Not yet run** -- every
> results/statistics/figures/interpretation section below is a placeholder, not a
> final claim. Per the template's own rule, this file may stay in this state until
> the experiment actually runs, but must not be treated as final ("TBD" is not
> acceptable in a *final* version) until every section is filled in from
> `results/raw/exp0.csv` and `results/processed/`.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp0` |
| Experiment name | Optimizer bake-off |
| Date run | _not yet run_ |
| Git commit hash | |
| SCALE-Sim version/commit | |
| Python / library versions | (link to `requirements.txt` lock or paste key versions) |
| Random seeds used | e.g. `[0, 1, 2, 3, 4]` |
| Raw data file(s) | `results/raw/exp0.csv` |
| Processed data file(s) | `results/processed/exp0_agg.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Choose which optimizer (REINFORCE, PPO, evolutionary/GA, Bayesian optimization) is used for all subsequent experiments, on stability + scalability grounds, not assumption.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 0, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** Optimizer type (REINFORCE, PPO, evolutionary, Bayesian optimization).
- **Dependent variable(s)/metrics:** Iterations-to-within-X%-of-best, seed variance, wall-clock per iteration.
- **Controlled/fixed variables:** One small workload (seq_len=512, 8 heads) on a 16x16 array.

## 3. Method

- **Procedure:** Run each candidate optimizer for a fixed iteration budget across >=5 seeds; also run brute-force/exhaustive search on this one small case to establish a ground-truth optimum for "distance from best".
- **Exact command used to run this experiment:** `python run_exp0.py --config ../../config/....yaml --seed 0 1 2 3 4`
- **Any deviations from the plan in `CODING_AGENT_PLAN.md` and why:** _none yet -- not run._

## 4. Raw Results Summary

_Not yet run -- table(s) to be filled in from `results/processed/exp0_agg.csv` (mean, std, 95% CI per condition, one table per metric)._

| Condition | n (seeds) | Metric | Mean | Std | 95% CI | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## 5. Statistical Analysis

- **Comparisons made:**
- **Test used:**
- **Result:**
- **Multiple-comparisons correction:**
- _Not yet run._

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| | | | |

## 7. Figures

**Required outputs (per the plan):** Table (per optimizer: mean iterations-to-converge +/- std, mean wall-clock, mean final utilization +/- std); convergence-curve overlay plot (all optimizers, mean +/- CI band, same axes); written justification of the winning optimizer in this folder's stats.md.

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| | `results/figures/fig1.png` | | `results/processed/exp0_agg.csv`, filtered on ... |

## 8. Interpretation

- **[RESULT]** _not yet run_
- **[INTERPRETATION]**
- **[LIMITATION]**

Compare explicitly against the two independent baselines where relevant: (a) the
in-harness static-heuristic baseline (reproducible, same evaluation environment),
and (b) literature-reported figures (clearly labelled as external, not re-run).
Never combine (a) and (b) into a single number.

## 9. Limitations & Reasons for Discrepancy

-

## 10. Reproduction

```bash
cd experiments/exp0_optimizer_bakeoff
python run_exp0.py --config ../../config/....yaml --seed 0 1 2 3 4
```

**Acceptance criteria for this experiment (per the plan):** A single optimizer is selected with a documented, data-backed reason; that choice is then hardcoded in config/optimizers/selected.yaml and referenced (not re-decided) by Experiments 2-6.
