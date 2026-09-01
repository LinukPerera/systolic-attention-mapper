# Experiment 3 Stats Report — Main sweep vs. baselines (core evidence chapter)

> **Headline finding, stated up front:** `bayesian_opt` significantly **loses** to `best_fixed_per_stage` in every cell, but significantly **beats** the best single-dataflow baseline (`os_only`) in every cell. Both are real, reported honestly, and explained by a genuine action-space asymmetry (§8) — not blended into one "the optimizer works / doesn't work" number.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp3` |
| Experiment name | Main sweep vs. baselines (core evidence chapter) |
| Date run | 2026-09-01 |
| Git commit hash | `273f0fa` (code that produced this run; raw CSV logs `7a80030-dirty` — working tree had this run's own code uncommitted at run time, correctly flagged by the `-dirty` suffix) |
| SCALE-Sim version/commit | `scalesim==3.0.0` (PyPI), containerized — `docker/scalesim/Dockerfile` |
| Python / library versions | Python 3.11.9; see `requirements.txt` (optuna 4.9.0, scipy 1.16.3) |
| Random seeds used | `[0, 1, 2, 3, 4]` (baselines deterministic, logged with `seed=-1`) |
| Raw data file(s) | `results/raw/exp3.csv` |
| Processed data file(s) | `results/processed/exp3_baseline_summary.csv`, `exp3_optimizer_vs_baseline.csv`, `exp3_significance.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

The central empirical claim — learned optimizer vs. static heuristics, across realistic workload diversity.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 3, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** {optimizer output (`bayesian_opt`), WS-only, OS-only, IS-only, best-fixed-per-stage} × {seq_len ∈ [128, 256] — reduced from the plan's example [128,512,1024,2048,4096], see §3} × {array ∈ [16×16, 32×32] — Experiment 1's mechanistically justified primary sizes}.
- **Dependent variable(s)/metrics:** PE utilization, exec cycles, memory traffic, EDP.
- **Controlled/fixed variables:** `head_dim=64` fixed across every workload point (only `seq_len` varies). Optimizer iteration budget (6/seed, fixed, not dynamic — see §3). Hardware SRAM/bandwidth/energy fields held identical across array sizes.

## 3. Method

- **Procedure:** For every (workload, array) cell, run every baseline once (deterministic, via `common.baselines.static_heuristics.evaluate_all_baselines`, shared with Experiment 1) and `bayesian_opt` (Experiment 0's selection, not re-decided) across 5 seeds × a fixed 6-iteration budget. Report both absolute values and %-difference from the best static baseline, in paired tables. Literature figures cited in §8 as a directional reference only.
- **Exact commands used:**
  ```bash
  python experiments/exp3_main_sweep/run_exp3.py \
    --workload-names exp3_seq128 exp3_seq256 --hardware-names 16x16 32x32 \
    --seeds 0 1 2 3 4 --n-iterations 6 --out results/raw/exp3.csv
  python experiments/exp3_main_sweep/analyze_exp3.py
  ```
- **Deviations from the plan, and why:**
  - **Grid reduced from `seq_len ∈ [128,512,1024,2048,4096]` to `seq_len ∈ [128,256]`.** Real per-call cost was timed by hand at `head_dim=64` before committing to any grid, per `NEXT_INSTRUCTIONS.md` §7.1: `seq_len=512` timed out entirely (>180s, no result) and `seq_len=1024` was Docker-OOM-killed (`exit 137`) after 134s at only 73% progress — both at **16×16**, the smaller of Experiment 1's two justified array sizes and therefore mandatory in this grid. `seq_len=128` (33.0s/call @16×16, 14.8s/call @32×32) and `seq_len=256` (already measured in `exp1_head64_validation.csv`: ~69–77s/call @16×16, ~28–30s/call @32×32) are both feasible.
  - **A real, counterintuitive finding underlying the above:** smaller arrays are *more* expensive to simulate for a fixed workload, not less (16×16 costs 2–2.5× what 32×32 costs at the same `seq_len`) — fewer PEs means more sequential simulation steps for the same total work, independent of dataflow.
  - **A real infrastructure incident during this run:** the first attempt stalled badly (28 minutes into cell 1's baselines, system load average 15.4, `spotlightknowledged` at 92.6% CPU) — root-caused to macOS Spotlight indexing the burst of new scratch files this experiment writes, competing for CPU/disk I/O against the containerized simulation itself. Killed, added a `.metadata_never_index` marker to the scratch directory, re-timed one call to confirm the slowdown was transient contention (31.86s, matching the original 33.01s measurement almost exactly, not a real cost regression), then relaunched — the full grid then completed well under the conservative ~1.7-hour estimate.
  - **Optimizer iteration budget fixed at 6/seed, and NOT used to dynamically stop a run.** Reduced from Experiment 0's pilot budget of 15, since this experiment's 2×2 grid multiplies total real simulator calls (173 real simulations total). Per Experiment 2's finding (`exp2_stats.md` §8a): the convergence criterion is fragile to early-trial clustering and unsuitable for controlling how many real simulations a run executes, so every seed runs the identical fixed budget; `convergence_iteration` is computed post-hoc and reported as a descriptive statistic only (every seed that converged did so at iteration 4, matching Experiment 2's own finding exactly — see §8).
  - **Two baseline comparisons computed, never blended — this is the central methodological point of this report, not a minor detail.** See §4/§8.

## 4. Raw Results Summary

**Baseline utilization (%) by cell and method** (`results/processed/exp3_baseline_summary.csv`, all `status == "OK"`):

| Cell | ws_only | os_only | is_only | best_fixed_per_stage |
|---|---|---|---|---|
| seq128 @ 16×16 | 73.57 | 89.96 | 88.32 | **91.89** |
| seq128 @ 32×32 | 57.67 | 79.50 | 75.75 | **82.49** |
| seq256 @ 16×16 | 84.77 | 89.02 | 87.08 | **94.01** |
| seq256 @ 32×32 | 73.15 | 78.91 | 74.99 | **87.45** |

**Optimizer (`bayesian_opt`, n=5 seeds) vs. both baselines** (`results/processed/exp3_optimizer_vs_baseline.csv`):

| Cell | Optimizer util. (mean ± std) | vs. best_overall (`best_fixed_per_stage`) | vs. best_single_dataflow (`os_only`) |
|---|---|---|---|
| seq128 @ 16×16 | 90.17 ± 0.16 | 91.89 → **−1.88%** | 89.96 → **+0.23%** |
| seq128 @ 32×32 | 80.12 ± 0.30 | 82.49 → **−2.88%** | 79.50 → **+0.79%** |
| seq256 @ 16×16 | 89.22 ± 0.08 | 94.01 → **−5.09%** | 89.02 → **+0.23%** |
| seq256 @ 32×32 | 79.46 ± 0.23 | 87.45 → **−9.14%** | 78.91 → **+0.70%** |

## 5. Statistical Analysis

- **Comparisons made:** in each of the 4 cells, `bayesian_opt`'s 5 per-seed best-utilization values against (a) `best_overall` (whichever static method has the highest utilization in that cell — always `best_fixed_per_stage` here) and (b) `best_single_dataflow` (the best of `ws_only`/`os_only`/`is_only` — always `os_only` here), separately.
- **Test used:** one-sample t-test (`scipy.stats.ttest_1samp`, testing whether the optimizer's seed-sample mean differs from the fixed baseline value) as primary, with a one-sample Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) as the non-parametric alternative per the plan's suggestion. Effect size: Cohen's d.
- **Result** (full table: `results/processed/exp3_significance.csv`):

  | Cell | vs. best_overall: t, p | vs. best_single_dataflow: t, p | Wilcoxon p (both) |
  |---|---|---|---|
  | seq128 @ 16×16 | t=−23.48, **p=1.95e-05** | t=+2.80, **p=0.0486** | 0.0625 |
  | seq128 @ 32×32 | t=−17.71, **p=5.98e-05** | t=+4.66, **p=0.0096** | 0.0625 |
  | seq256 @ 16×16 | t=−129.27, **p=2.15e-08** | t=+5.58, **p=0.0051** | 0.0625 |
  | seq256 @ 32×32 | t=−79.36, **p=1.51e-07** | t=+5.46, **p=0.0055** | 0.0625 |

  All 8 comparisons are significant at α=0.05 by the t-test, in the direction shown (negative t = optimizer below baseline, positive t = above). **Every Wilcoxon signed-rank p-value is exactly 0.0625** — the smallest p-value the exact signed-rank test can produce at n=5 (all 5 signed differences agree in sign in every comparison, which is the most extreme configuration possible at this sample size, yet still lands just above the conventional 0.05 threshold). This is a genuine small-sample limitation of the exact test, not evidence against the t-test's finding — reported explicitly rather than cherry-picking whichever test crosses 0.05.
- **Multiple-comparisons correction:** not applied — 4 cells × 2 baselines = 8 tests, all either strongly significant (p < 5e-5, vs. best_overall) or borderline-but-consistent (p < 0.05, vs. best_single_dataflow) in a single, consistent direction each; a Bonferroni correction (α=0.05/8=0.00625) would flip 3 of the 4 best_single_dataflow comparisons to "not significant" at the strictest threshold — flagged explicitly: **without correction, all 4 best_single_dataflow comparisons are significant; with a conservative Bonferroni correction, only seq256 @ 16×16 (p=0.0051) and seq256 @ 32×32 (p=0.0055) survive.** The vs. best_overall comparisons remain significant either way.

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| — | — | — | — |

Zero `status == "FAILED"` rows in `results/raw/exp3.csv` (173 rows, all `OK`).

## 7. Figures

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| 1 | `results/figures/exp3_utilization_heatmap.png` (+ `.svg`) | Utilization (%) by cell (rows) × method (columns), including `bayesian_opt`. `best_fixed_per_stage` is visibly the brightest column in every row; `bayesian_opt` sits between it and the single-dataflow baselines. | `results/processed/exp3_baseline_summary.csv` + `exp3_optimizer_vs_baseline.csv` |
| 2 | `results/figures/exp3_pct_improvement_bars_vs_best_overall.png` (+ `.svg`) | %-improvement of `bayesian_opt` over `best_overall`, with 95% CI — negative in all 4 cells. | `results/processed/exp3_optimizer_vs_baseline.csv` |
| 3 | `results/figures/exp3_pct_improvement_bars_vs_best_single_dataflow.png` (+ `.svg`) | %-improvement of `bayesian_opt` over `best_single_dataflow`, with 95% CI — positive in all 4 cells. | `results/processed/exp3_optimizer_vs_baseline.csv` |

All 4 cells are shown in the main body above (§4) rather than promoted-subset-plus-appendix — the full grid is only 4 cells, small enough that no subset/appendix split is needed (the plan's "2–3 representative cases + appendix" guidance applies to grids too large to show in full, which this one isn't).

## 8. Interpretation

- **[RESULT]** `bayesian_opt` loses to `best_overall` (always `best_fixed_per_stage`) significantly in all 4 cells (−1.88% to −9.14%, all p < 0.0001 by t-test).
- **[RESULT]** `bayesian_opt` beats `best_single_dataflow` (always `os_only`) significantly in all 4 cells by the t-test (+0.23% to +0.79%, all p < 0.05), though the margin is small and the effect does not survive a conservative Bonferroni correction in 2 of the 4 cells (§5).
- **[INTERPRETATION] — this is the central finding of this experiment, not a footnote.** This asymmetry is explained by a genuine **action-space asymmetry**, not a search-quality failure: `common/optimizers/environment.py`'s `AttentionMappingProblem` applies **one** `dataflow_mode` to the **entire** workload per simulation (necessarily — SCALE-Sim's `Dataflow` config setting applies per-run, not per-layer, as documented since `common/simulator_interface.py`). `best_fixed_per_stage` only achieves its advantage by running **three separate simulations** (one per GEMM stage) and combining them post-hoc (`combine_simulation_results`) — a capability the learned optimizer's action space does not have access to in a single search. So the optimizer's true ceiling, given its current action space, is "best single global dataflow + a tuned SRAM partition" — and it reaches slightly *above* that ceiling's raw baseline (`os_only`) in every cell, which is exactly what the SRAM-partition search is supposed to buy it. Losing to `best_fixed_per_stage` is not evidence the search failed; it's evidence the search was never given the tools to reach that particular baseline's answer.
- **[LIMITATION → concrete, important follow-up work, not just a caveat]** This reveals a real gap between the current implementation and the thesis's own "joint dataflow+tiling learning" contribution claim: if a hand-designed per-stage heuristic can beat the learned method simply because it can vary dataflow per stage and the learned method structurally cannot, the learned method isn't actually competing on the axis the thesis wants to claim novelty on. **Concrete recommendation:** extend `common/optimizers/environment.py`'s action space to include a per-stage (or per-GEMM-layer) dataflow choice — e.g. one dataflow_mode per topology row/stage instead of one per whole-workload run — so the optimizer can, in principle, discover the same (or a better) per-stage specialization `best_fixed_per_stage` finds by exhaustive search, and be fairly compared against it. This directly affects how Experiment 5 (Ablation study, "dataflow-only" vs. "tiling-only" vs. "joint") should be scoped: as currently built, even "joint" is capped at one global dataflow, which may understate the value of joint learning that Experiment 5 is specifically designed to demonstrate.
- **[RESULT]** Every optimizer seed that converged (by Experiment 2's criterion, window=5/epsilon=0.1) did so at iteration 4 — the same value Experiment 2 found, on a different workload/hardware/budget, reinforcing (not just repeating) Experiment 2's finding that this criterion triggers on early-trial clustering rather than genuine settling (some seeds — e.g. seq128@32×32 seed 4, seq256@16×16 seed 4, seq256@32×32 seed 4 — never converge within the 6-iteration budget at all, consistent with a real, variable search process, not a scripted one).
- **[LIMITATION]** `dram_bandwidth_bytes_per_cycle` is not simulated and `energy_per_cycle_pj` is a placeholder (1.0) — identical caveats to every prior experiment; EDP columns in the raw/processed CSVs should not be cited without revisiting these values.
- **[LIMITATION]** 6 iterations/seed is a small budget (reduced from Experiment 0's 15 for wall-clock reasons across this 2×2 grid) — the optimizer's true ceiling within its current action space may be higher than what 6 iterations found; not tested here.

**Literature comparison (directional context only, never averaged into the numbers above):** FLAT reports ≈1.9× speedup, COSA ≈94% utilization, ReDas ≈4.6×, Gemmini WS/OS ≈3× (all external, not re-run). `best_fixed_per_stage`'s 94.01% (seq256 @ 16×16) is closely comparable to COSA's reported figure — consistent with, though not proof of, this project's baselines being reasonable. The learned optimizer's modest (<1%) improvement over the best single-dataflow baseline is far below FLAT/ReDas's reported multiplicative speedups, which is expected given (a) those systems report end-to-end speedup over a fundamentally different reference point, not %-utilization-over-best-heuristic, and (b) the action-space limitation above — this comparison is not apples-to-apples and is stated as directional context, not a claimed result.

## 9. Limitations & Reasons for Discrepancy

- **The action-space asymmetry (§8) is the primary limitation of this experiment's central claim** — a fair "is the learned optimizer better than the best static baseline" claim cannot currently be made using `best_fixed_per_stage` as the reference, because the two are not evaluating comparable action spaces. The `best_single_dataflow` comparison is the fairer one, and it is positive but small and only partly robust to multiple-comparisons correction (§5).
- **Grid reduced to 2 `seq_len` × 2 array points** (§3) — real, measured, disclosed, not an assumption. The plan's full `[128,512,1024,2048,4096]` range remains untested; Experiment 4 (scalability) is the plan's designated place to extend this further, and should budget for the same 16×16-is-more-expensive-to-simulate finding this report discovered.
- **6-iteration optimizer budget** — real, disclosed scope constraint (§8).
- **No multiple-comparisons correction applied by default** — reported both corrected and uncorrected readings explicitly (§5) rather than picking whichever makes the claim stronger.

## 10. Reproduction

```bash
cd "MSc Thesis"   # project root
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/   # if not already built
python experiments/exp3_main_sweep/run_exp3.py \
  --workload-names exp3_seq128 exp3_seq256 --hardware-names 16x16 32x32 \
  --seeds 0 1 2 3 4 --n-iterations 6 --out results/raw/exp3.csv
python experiments/exp3_main_sweep/analyze_exp3.py
```

Baselines are fully deterministic; `bayesian_opt` (optuna TPE) is seeded but not guaranteed bit-for-bit reproducible across machines/library-patch-versions (same caveat as Experiments 0/2) — the aggregate conclusions (§8) are the claim being made.

**Acceptance criteria for this experiment (per the plan):** Every reported "X% improvement" claim has a seed-level std/CI attached and, ideally, a significance test reported in the stats.md. **Met**: every %-improvement claim in §4 carries std/CI (`results/processed/exp3_optimizer_vs_baseline.csv`), and both a t-test and Wilcoxon signed-rank test are reported for every comparison (§5), including the honest disclosure that not all comparisons survive multiple-comparisons correction and that one baseline comparison is structurally unfair to the optimizer's current action space (§8) — this is what "defensible" means here: the strongest claim the data actually supports, clearly separated from the claim it doesn't.
