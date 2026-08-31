# Experiment Stats Report — Template

> **How to use this file:** copy it to `docs/stats/expN_<short-name>_stats.md` for each experiment and fill in every section from the actual logged data in `results/raw/` and `results/processed/`. Do not leave a section as "TBD" in a version considered final — an incomplete run is recorded as `status: FAILED` rows (per the data contract in `CODING_AGENT_PLAN.md` §4), not as a blank section here.
>
> Tag every claim in §7 and §8 as one of: **[RESULT]** (a number read directly off the processed data), **[INTERPRETATION]** (the author's reading of what the result means), or **[LIMITATION]** (an acknowledged gap or caveat). Never mix these without the tag — this mirrors the FACT/RECOMMENDATION/ASSUMPTION convention used across this project's other documents.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `expN` |
| Experiment name | |
| Date run | |
| Git commit hash | |
| SCALE-Sim version/commit | |
| Python / library versions | (link to `requirements.txt` lock or paste key versions) |
| Random seeds used | e.g. `[0, 1, 2, 3, 4]` |
| Raw data file(s) | `results/raw/expN.csv` |
| Processed data file(s) | `results/processed/expN_agg.csv` |
| Figures directory | `results/figures/expN/` |

---

## 1. Objective & Hypothesis

State the single question this experiment answers, and (if applicable) the hypothesis being tested, stated *before* looking at results-as-if-blind — copy this from `CODING_AGENT_PLAN.md`'s spec for this experiment rather than rewriting it differently after the fact.

## 2. Variables

- **Independent variable(s):**
- **Dependent variable(s)/metrics:**
- **Controlled/fixed variables:** (everything held constant — array size, workload, hyperparameters not under study)

## 3. Method

- Procedure (numbered steps, enough detail that a third party could rerun this exactly).
- Exact command used to run this experiment: `python experiments/expN_....py --config config/....yaml --seed ...`
- Any deviations from the plan in `CODING_AGENT_PLAN.md` and why.

## 4. Raw Results Summary

One row per condition (not per seed) — aggregate across seeds with mean, standard deviation, and 95% confidence interval. Do not report a bare mean without a spread measure.

| Condition | n (seeds) | Metric | Mean | Std | 95% CI | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

(Add one table per metric if multiple metrics are reported, e.g. separate tables for PE utilization, exec cycles, memory traffic, EDP.)

## 5. Statistical Analysis

- **Comparisons made:** (e.g. "learned optimizer vs. best static baseline, per workload")
- **Test used:** (e.g. paired t-test across seeds, Wilcoxon signed-rank if normality is doubtful — state which and why)
- **Result:** test statistic, p-value, and effect size (e.g. Cohen's d) for each comparison claimed as significant in §7/§8.
- **Multiple-comparisons correction** (if many workloads/conditions are compared simultaneously — e.g. Bonferroni or Holm — state whether applied and why/why not).
- If a claimed improvement is **not** statistically significant, say so explicitly rather than omitting the test.

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| | | | |

(This section exists precisely so failures are evidence, not hidden — leave it empty only if `status == FAILED` truly has zero rows in the raw CSV for this experiment.)

## 7. Figures

For each figure: number, file path (PNG+SVG), one-sentence caption, and the exact processed-data file/query it was generated from.

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| | `results/figures/expN/fig1.png` | | `results/processed/expN_agg.csv`, filtered on ... |

## 8. Interpretation

Write the findings as tagged statements:

- **[RESULT]** ...
- **[RESULT]** ...
- **[INTERPRETATION]** ...
- **[LIMITATION]** ...

Compare explicitly against the two independent baselines where relevant: (a) the in-harness static-heuristic baseline (reproducible, same evaluation environment), and (b) literature-reported figures (clearly labelled as external, not re-run — e.g. FLAT ≈1.9×, COSA ≈94% utilization). Never combine (a) and (b) into a single number.

## 9. Limitations & Reasons for Discrepancy

- What this experiment does **not** show / was not tested here, and why (scope boundary, not a flaw to hide).
- If results deviate from what the literature or theory would predict, attribute the discrepancy to a specific, citable mechanism where possible (e.g. simulator abstraction of softmax, a specific hyperparameter choice) rather than leaving it unexplained.

## 10. Reproduction

Exact steps/commands to regenerate every number and figure in this report from a clean checkout, including config file paths and seed list. If regenerating produces different numbers, that is a reproducibility bug to fix before this report is treated as final.
