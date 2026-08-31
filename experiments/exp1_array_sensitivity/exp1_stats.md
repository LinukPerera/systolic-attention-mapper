# Experiment 1 Stats Report -- Array-size sensitivity study

> Copied from `../../docs/EXPERIMENT_STATS_TEMPLATE.md`. **Not yet run** -- every
> results/statistics/figures/interpretation section below is a placeholder, not a
> final claim. Per the template's own rule, this file may stay in this state until
> the experiment actually runs, but must not be treated as final ("TBD" is not
> acceptable in a *final* version) until every section is filled in from
> `results/raw/exp1.csv` and `results/processed/`.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp1` |
| Experiment name | Array-size sensitivity study |
| Date run | _not yet run_ |
| Git commit hash | |
| SCALE-Sim version/commit | |
| Python / library versions | (link to `requirements.txt` lock or paste key versions) |
| Random seeds used | e.g. `[0, 1, 2, 3, 4]` |
| Raw data file(s) | `results/raw/exp1.csv` |
| Processed data file(s) | `results/processed/exp1_agg.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Justify the primary working array sizes (16x16, 32x32) the way the sample thesis justifies its mesh choice.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 1, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** Array size (8x8, 16x16, 32x32, 64x64, 128x128).
- **Dependent variable(s)/metrics:** Achievable utilization per dataflow, ranking stability of dataflows.
- **Controlled/fixed variables:** One representative workload.

## 3. Method

- **Procedure:** Run the static-heuristic baselines (not yet the learned optimizer) at each array size; identify the point beyond which larger arrays stop improving utilization or become degenerate (per the lit review's fragmentation argument).
- **Exact command used to run this experiment:** `python run_exp1.py --config ../../config/....yaml --seed 0 1 2 3 4`
- **Any deviations from the plan in `CODING_AGENT_PLAN.md` and why:** _none yet -- not run._

## 4. Raw Results Summary

_Not yet run -- table(s) to be filled in from `results/processed/exp1_agg.csv` (mean, std, 95% CI per condition, one table per metric)._

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

**Required outputs (per the plan):** Line plot of utilization vs. array size per dataflow mode; a stated criterion for "sufficient" array size (e.g. "<2% utilization change per doubling").

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| | `results/figures/fig1.png` | | `results/processed/exp1_agg.csv`, filtered on ... |

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
cd experiments/exp1_array_sensitivity
python run_exp1.py --config ../../config/....yaml --seed 0 1 2 3 4
```

**Acceptance criteria for this experiment (per the plan):** Written justification, grounded in this plot, for why 16x16/32x32 are the primary sizes used in Experiments 3-6.
