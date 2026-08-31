# Experiment 4 Stats Report -- Scalability study (answers RQ3 directly)

> Copied from `../../docs/EXPERIMENT_STATS_TEMPLATE.md`. **Not yet run** -- every
> results/statistics/figures/interpretation section below is a placeholder, not a
> final claim. Per the template's own rule, this file may stay in this state until
> the experiment actually runs, but must not be treated as final ("TBD" is not
> acceptable in a *final* version) until every section is filled in from
> `results/raw/exp4.csv` and `results/processed/`.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp4` |
| Experiment name | Scalability study (answers RQ3 directly) |
| Date run | _not yet run_ |
| Git commit hash | |
| SCALE-Sim version/commit | |
| Python / library versions | (link to `requirements.txt` lock or paste key versions) |
| Random seeds used | e.g. `[0, 1, 2, 3, 4]` |
| Raw data file(s) | `results/raw/exp4.csv` |
| Processed data file(s) | `results/processed/exp4_agg.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Show how the optimizer's advantage over baselines, and its own overhead, scale with sequence length and array size.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 4, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** seq_len (extended beyond Exp 3, e.g. up to 8192) and array size.
- **Dependent variable(s)/metrics:** %-improvement over baseline, optimizer wall-clock/sample count needed to converge.
- **Controlled/fixed variables:** Everything not swept above.

## 3. Method

- **Procedure:** Reuse Experiment 2's convergence criterion to decide when each run is "done"; log both performance and cost.
- **Exact command used to run this experiment:** `python run_exp4.py --config ../../config/....yaml --seed 0 1 2 3 4`
- **Any deviations from the plan in `CODING_AGENT_PLAN.md` and why:** _none yet -- not run._

## 4. Raw Results Summary

_Not yet run -- table(s) to be filled in from `results/processed/exp4_agg.csv` (mean, std, 95% CI per condition, one table per metric)._

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

**Required outputs (per the plan):** Two-panel line plot -- (a) %-improvement vs. seq_len/array size, (b) optimizer cost vs. seq_len/array size.

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| | `results/figures/fig1.png` | | `results/processed/exp4_agg.csv`, filtered on ... |

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
cd experiments/exp4_scalability
python run_exp4.py --config ../../config/....yaml --seed 0 1 2 3 4
```

**Acceptance criteria for this experiment (per the plan):** A direct, evidence-based answer to RQ3 is written into the stats.md (not just plots -- a sentence stating what happens to performance and cost as scale increases).
