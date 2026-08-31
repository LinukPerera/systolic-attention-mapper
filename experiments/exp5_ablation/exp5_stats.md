# Experiment 5 Stats Report -- Ablation study

> Copied from `../../docs/EXPERIMENT_STATS_TEMPLATE.md`. **Not yet run** -- every
> results/statistics/figures/interpretation section below is a placeholder, not a
> final claim. Per the template's own rule, this file may stay in this state until
> the experiment actually runs, but must not be treated as final ("TBD" is not
> acceptable in a *final* version) until every section is filled in from
> `results/raw/exp5.csv` and `results/processed/`.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp5` |
| Experiment name | Ablation study |
| Date run | _not yet run_ |
| Git commit hash | |
| SCALE-Sim version/commit | |
| Python / library versions | (link to `requirements.txt` lock or paste key versions) |
| Random seeds used | e.g. `[0, 1, 2, 3, 4]` |
| Raw data file(s) | `results/raw/exp5.csv` |
| Processed data file(s) | `results/processed/exp5_agg.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Isolate the contribution of joint dataflow+tiling learning (the specific gap claimed in the lit review) from tiling alone or dataflow alone.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 5, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** {learned dataflow only (tiling fixed to a reasonable default), learned tiling only (dataflow fixed to best-static), learned joint dataflow+tiling}.
- **Dependent variable(s)/metrics:** Same core metrics as Experiment 3.
- **Controlled/fixed variables:** Same workload/array subset as Experiment 3's representative cases.

## 3. Method

- **Procedure:** Run each of the three variants on the same representative cases used in Experiment 3, across >=5 seeds where stochastic.
- **Exact command used to run this experiment:** `python run_exp5.py --config ../../config/....yaml --seed 0 1 2 3 4`
- **Any deviations from the plan in `CODING_AGENT_PLAN.md` and why:** _none yet -- not run._

## 4. Raw Results Summary

_Not yet run -- table(s) to be filled in from `results/processed/exp5_agg.csv` (mean, std, 95% CI per condition, one table per metric)._

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

**Required outputs (per the plan):** Bar chart comparing the three variants per metric, with error bars.

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| | `results/figures/fig1.png` | | `results/processed/exp5_agg.csv`, filtered on ... |

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
cd experiments/exp5_ablation
python run_exp5.py --config ../../config/....yaml --seed 0 1 2 3 4
```

**Acceptance criteria for this experiment (per the plan):** The stats.md states plainly how much of the total gain is attributable to joint optimization vs. either factor alone.
