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

- **Independent variable(s):** {optimizer output (`bayesian_opt`), WS-only, OS-only, IS-only, best-fixed-per-stage} × {seq_len ∈ [128, 256] — reduced from the plan's example [128,512,1024,2048,4096], see §3 deviations} × {array ∈ [16×16, 32×32] — Experiment 1's mechanistically justified primary sizes}.
- **Dependent variable(s)/metrics:** PE utilization, exec cycles, memory traffic, EDP.
- **Controlled/fixed variables:** `head_dim=64` (fixed, per `config/workloads/README.md`'s standard) across every workload point — only `seq_len` varies between them. Optimizer iteration budget (6/seed, fixed, not dynamic — see §3). Hardware SRAM/bandwidth/energy fields held identical across array sizes (matching Experiment 1's convention).

## 3. Method

- **Procedure:** For every (workload, array) cell, run every baseline once (deterministic, via `common.baselines.static_heuristics.evaluate_all_baselines`, shared with Experiment 1) and `bayesian_opt` (Experiment 0's selection, not re-decided) across 5 seeds × a fixed 6-iteration budget. Report both absolute values and %-difference from the best static baseline, in paired tables. Literature figures (FLAT ≈1.9×, COSA ≈94% utilization, ReDas ≈4.6×, Gemmini WS/OS ≈3×) are cited in §8 as a directional reference only — never averaged into these numbers.
- **Exact commands used:**
  ```bash
  python experiments/exp3_main_sweep/run_exp3.py \
    --workload-names exp3_seq128 exp3_seq256 --hardware-names 16x16 32x32 \
    --seeds 0 1 2 3 4 --n-iterations 6 --out results/raw/exp3.csv
  python experiments/exp3_main_sweep/analyze_exp3.py
  ```
- **Deviations from the plan, and why:**
  - **Grid reduced from `seq_len ∈ [128,512,1024,2048,4096]` to `seq_len ∈ [128,256]`.** Real per-call cost was timed by hand at `head_dim=64` (the standard adopted after Experiment 1) before committing to any grid, per `NEXT_INSTRUCTIONS.md` §7.1's explicit instruction: `seq_len=512` timed out entirely (>180s, no result) and `seq_len=1024` was Docker-OOM-killed (`exit 137`) after 134s at only 73% progress — both at **16×16**, the smaller of Experiment 1's two justified array sizes and therefore mandatory in this grid, not a size that could be dropped instead. `seq_len=128` (33.0s/call @16×16, 14.8s/call @32×32) and `seq_len=256` (already measured in `exp1_head64_validation.csv`: ~69–77s/call @16×16, ~28–30s/call @32×32) are both real and feasible.
  - **A real, counterintuitive finding underlying the above:** smaller arrays are *more* expensive to simulate for a fixed workload, not less (16×16 costs 2–2.5× what 32×32 costs at the same `seq_len`) — fewer PEs means more sequential simulation steps for the same total work, independent of dataflow. This compounds with `seq_len` growth, which is why 16×16 is the binding constraint on how large `seq_len` can go in this grid.
  - **Optimizer iteration budget fixed at 6/seed, and NOT used to dynamically stop a run.** Reduced from Experiment 0's pilot budget of 15, since this experiment's 2×2 grid multiplies total real simulator calls (4 cells × (12 baseline calls + 5×6 optimizer calls) ≈ 168 real simulations, at per-call costs up to ~77s). Per Experiment 2's finding (`exp2_stats.md` §8a): the convergence criterion is fragile to early-trial clustering and unsuitable for controlling how many real simulations a run executes, so every seed here runs the identical fixed budget; `convergence_iteration` is computed post-hoc and reported as a descriptive statistic only.

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
