# Experiment 3 Stats Report -- Main sweep vs. baselines (core evidence chapter)

> Copied from `../../docs/EXPERIMENT_STATS_TEMPLATE.md`. **Not yet run** -- every
> results/statistics/figures/interpretation section below is a placeholder, not a
> final claim. Per the template's own rule, this file may stay in this state until
> the experiment actually runs, but must not be treated as final ("TBD" is not
> acceptable in a *final* version) until every section is filled in from
> `results/raw/exp3.csv` and `results/processed/`.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp3` |
| Experiment name | Main sweep vs. baselines (core evidence chapter) |
| Date run | _not yet run_ |
| Git commit hash | |
| SCALE-Sim version/commit | |
| Python / library versions | (link to `requirements.txt` lock or paste key versions) |
| Random seeds used | e.g. `[0, 1, 2, 3, 4]` |
| Raw data file(s) | `results/raw/exp3.csv` |
| Processed data file(s) | `results/processed/exp3_agg.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

The central empirical claim -- learned optimizer vs. static heuristics, across realistic workload diversity.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 3, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** {optimizer output, WS-only, OS-only, IS-only, best-fixed-per-stage} x {seq_len in [128, 512, 1024, 2048, 4096]} x {array in [16x16, 32x32]}.
- **Dependent variable(s)/metrics:** PE utilization, exec cycles, memory traffic, EDP.
- **Controlled/fixed variables:** Everything not swept above (hardware config beyond array size, optimizer hyperparameters).

## 3. Method

- **Procedure:** For every (workload, array) cell, run every baseline once (deterministic) and the learned optimizer across >=5 seeds; report both absolute values and %-difference from the best static baseline, in paired tables. Add a clearly separate column citing the relevant literature figure (FLAT ~1.9x, COSA ~94% utilization, ReDas ~4.6x, Gemmini WS/OS ~3x) as a directional reference only -- never averaged into your own numbers.
- **Exact command used to run this experiment:** `python run_exp3.py --config ../../config/....yaml --seed 0 1 2 3 4`
- **Any deviations from the plan in `CODING_AGENT_PLAN.md` and why:** _none yet -- not run._

## 4. Raw Results Summary

_Not yet run -- table(s) to be filled in from `results/processed/exp3_agg.csv` (mean, std, 95% CI per condition, one table per metric)._

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

**Required outputs (per the plan):** Heatmap (utilization by workload x array x method); grouped bar chart of %-improvement over best baseline per workload; full results table (abs + %-diff) for the appendix; 2-3 representative workload cases promoted into the main body.

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| | `results/figures/fig1.png` | | `results/processed/exp3_agg.csv`, filtered on ... |

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
cd experiments/exp3_main_sweep
python run_exp3.py --config ../../config/....yaml --seed 0 1 2 3 4
```

**Acceptance criteria for this experiment (per the plan):** Every reported "X% improvement" claim has a seed-level std/CI attached and, ideally, a significance test (paired t-test or Wilcoxon signed-rank across seeds/workloads) reported in the stats.md.
