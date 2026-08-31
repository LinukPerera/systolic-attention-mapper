# Experiment 6 Stats Report -- Generalization test (optional/stretch)

> Copied from `../../docs/EXPERIMENT_STATS_TEMPLATE.md`. **Not yet run** -- every
> results/statistics/figures/interpretation section below is a placeholder, not a
> final claim. Per the template's own rule, this file may stay in this state until
> the experiment actually runs, but must not be treated as final ("TBD" is not
> acceptable in a *final* version) until every section is filled in from
> `results/raw/exp6.csv` and `results/processed/`.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp6` |
| Experiment name | Generalization test (optional/stretch) |
| Date run | _not yet run_ |
| Git commit hash | |
| SCALE-Sim version/commit | |
| Python / library versions | (link to `requirements.txt` lock or paste key versions) |
| Random seeds used | e.g. `[0, 1, 2, 3, 4]` |
| Raw data file(s) | `results/raw/exp6.csv` |
| Processed data file(s) | `results/processed/exp6_agg.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Test whether the learned policy generalizes to unseen workload shapes, or has effectively memorized per-shape optima.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 6, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** Train/fit subset of (seq_len, head_dim) combinations vs. held-out combinations not seen during training.
- **Dependent variable(s)/metrics:** In-distribution vs. held-out performance relative to baseline.
- **Controlled/fixed variables:** Everything else as in Experiment 3.

## 3. Method

- **Procedure:** Train/fit on the training subset; evaluate zero-shot on the held-out combinations.
- **Exact command used to run this experiment:** `python run_exp6.py --config ../../config/....yaml --seed 0 1 2 3 4`
- **Any deviations from the plan in `CODING_AGENT_PLAN.md` and why:** _none yet -- not run._
- **Note:** Implement only after Experiments 0-5 are complete and documented (plan §6).

## 4. Raw Results Summary

_Not yet run -- table(s) to be filled in from `results/processed/exp6_agg.csv` (mean, std, 95% CI per condition, one table per metric)._

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

**Required outputs (per the plan):** Bar/line chart comparing in-distribution vs. held-out performance relative to baseline.

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| | `results/figures/fig1.png` | | `results/processed/exp6_agg.csv`, filtered on ... |

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
cd experiments/exp6_generalization
python run_exp6.py --config ../../config/....yaml --seed 0 1 2 3 4
```

**Acceptance criteria for this experiment (per the plan):** An honest statement of the generalization gap, even if it's large -- this belongs in limitations either way.
