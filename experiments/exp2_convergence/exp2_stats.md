# Experiment 2 Stats Report -- Search/training convergence study

> Copied from `../../docs/EXPERIMENT_STATS_TEMPLATE.md`. **Not yet run** -- every
> results/statistics/figures/interpretation section below is a placeholder, not a
> final claim. Per the template's own rule, this file may stay in this state until
> the experiment actually runs, but must not be treated as final ("TBD" is not
> acceptable in a *final* version) until every section is filled in from
> `results/raw/exp2.csv` and `results/processed/`.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp2` |
| Experiment name | Search/training convergence study |
| Date run | _not yet run_ |
| Git commit hash | |
| SCALE-Sim version/commit | |
| Python / library versions | (link to `requirements.txt` lock or paste key versions) |
| Random seeds used | e.g. `[0, 1, 2, 3, 4]` |
| Raw data file(s) | `results/raw/exp2.csv` |
| Processed data file(s) | `results/processed/exp2_agg.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Prove the chosen optimizer reliably converges, with an explicit numeric stopping criterion.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 2, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** Seed (>=5).
- **Dependent variable(s)/metrics:** Reward/utilization trajectory per iteration.
- **Controlled/fixed variables:** Configuration selected in Experiment 0 / justified in Experiment 1.

## 3. Method

- **Procedure:** Run the selected optimizer (config/optimizers/selected.yaml) to a generous iteration budget; define and state the convergence criterion (e.g. moving-average reward variance < epsilon over a window of N iterations); record the iteration at which each seed meets it.
- **Exact command used to run this experiment:** `python run_exp2.py --config ../../config/....yaml --seed 0 1 2 3 4`
- **Any deviations from the plan in `CODING_AGENT_PLAN.md` and why:** _none yet -- not run._

## 4. Raw Results Summary

_Not yet run -- table(s) to be filled in from `results/processed/exp2_agg.csv` (mean, std, 95% CI per condition, one table per metric)._

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

**Required outputs (per the plan):** Multi-seed convergence-curve plot (individual traces + mean +/- CI band); table of convergence iteration per seed, mean +/- std.

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| | `results/figures/fig1.png` | | `results/processed/exp2_agg.csv`, filtered on ... |

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
cd experiments/exp2_convergence
python run_exp2.py --config ../../config/....yaml --seed 0 1 2 3 4
```

**Acceptance criteria for this experiment (per the plan):** Convergence criterion is stated as a formula (not just "it looked flat"), and holds for all/most seeds -- seeds that fail to converge are reported, not discarded.
