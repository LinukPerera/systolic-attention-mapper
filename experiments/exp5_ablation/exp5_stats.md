# Experiment 5 Stats Report — Ablation study

> **Headline finding, stated up front:** of `joint`'s gain over the best static baseline, `dataflow_only` captures **exactly 0%** (it's deterministic — always rediscovers the already-known best static dataflow) and `tiling_only` captures **all of it, and slightly more** (117%/117%) — though that "slightly more" is **not** statistically significant at n=5 (paired test). The safest, fully-supported claim: **essentially all of the joint optimizer's real gain over the best static heuristic is attributable to the tiling (SRAM-partition) search, not the dataflow search** — dataflow search alone adds nothing once the best static dataflow is already known, and may even cost the joint search a little of its budget.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp5` |
| Experiment name | Ablation study |
| Date run | 2026-09-01 |
| Git commit hash | `175ef47` (code that produced this run; raw CSV logs `c84a22a-dirty` — working tree had this run's own code uncommitted at run time, correctly flagged by the `-dirty` suffix) |
| SCALE-Sim version/commit | `scalesim==3.0.0` (PyPI), containerized — `docker/scalesim/Dockerfile` |
| Python / library versions | Python 3.11.9; see `requirements.txt` (optuna 4.9.0, scipy 1.16.3) |
| Random seeds used | `[0, 1, 2, 3, 4]`, same set for all three variants (enables a paired per-seed comparison — see §5) |
| Raw data file(s) | `results/raw/exp5.csv` (`dataflow_only`/`tiling_only`) **+ reused** `experiments/exp3_main_sweep/results/raw/exp3.csv` (`joint` = `bayesian_opt` — see §3 for why reused, not re-run) |
| Processed data file(s) | `results/processed/exp5_ablation_summary.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Isolate the contribution of joint dataflow+tiling learning (the specific gap claimed in the lit review) from tiling alone or dataflow alone.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 5, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** {`dataflow_only` (`common.optimizers.ablation.DataflowOnlyOptimizer` — SRAM partition fixed at an equal three-way split), `tiling_only` (`TilingOnlyOptimizer` — dataflow fixed to the cell's real best-static choice), `joint` (`bayesian_opt`, the full search — Experiment 3's data, reused)}.
- **Dependent variable(s)/metrics:** PE utilization (same core metric as Experiment 3).
- **Controlled/fixed variables:** Same 2 (of Experiment 3's 4) representative cells: seq128 @ 16×16 and seq256 @ 32×32. Iteration budget fixed at 6/seed for all three variants (matching Experiment 3, for comparability).

## 3. Method

- **Procedure:** For each of 2 representative cells, run `dataflow_only` and `tiling_only` for 5 seeds × 6 iterations each (real, new simulations). `joint`'s data for the same cells/seeds is reused directly from Experiment 3 (see deviations below). Compute each variant's mean best-utilization across seeds, then gain attribution: `(variant_mean − best_static) / (joint_mean − best_static) × 100`, where `best_static` is the cell's real best single-dataflow baseline value from Experiment 3.
- **Exact commands used:**
  ```bash
  python experiments/exp5_ablation/run_exp5.py \
    --cells exp3_seq128:16x16 exp3_seq256:32x32 \
    --seeds 0 1 2 3 4 --n-iterations 6 --out results/raw/exp5.csv
  python experiments/exp5_ablation/analyze_exp5.py
  ```
- **Deviations from the plan, and why:**
  - **`joint`'s results are reused from Experiment 3, not re-run.** `bayesian_opt` is fully deterministic given `(workload, hardware, seed)` — re-running it for these exact cells/seeds would reproduce Experiment 3's numbers bit-for-bit at real wall-clock cost for zero new information (same reasoning as `run_exp4.py`'s data-provenance choice, `exp2_stats.md` §8a).
  - **2 of Experiment 3's 4 cells, not all 4.** A deliberate, wall-clock-driven scope reduction (this doubles Experiment 3's per-cell optimizer cost: 2 new variants × 5 seeds × 6 iterations = 60 new real simulations per cell) — documented, not silent, per this project's established practice. The 2 chosen span both array sizes and both tested `seq_len` values.
  - **`tiling_only`'s fixed dataflow is looked up from Experiment 3's real data** (`exp3_optimizer_vs_baseline.csv`'s `best_single_dataflow_method` column), not assumed — it happens to be `os_only` (→ `OS`) in both cells here, matching every Experiment 3 cell at `seq_len` ≤ 256 (though not at `seq_len`=512, per Experiment 4's finding — irrelevant here since neither ablated cell reaches that scale).

## 4. Raw Results Summary

**Per-variant utilization (%), n=5 seeds** (`results/processed/exp5_ablation_summary.csv`):

| Cell | Best static (`os_only`) | `dataflow_only` (mean ± std) | `tiling_only` (mean ± std) | `joint` (mean ± std) |
|---|---|---|---|---|
| seq128 @ 16×16 | 89.960 | 89.960 ± 0.000 | 90.201 ± 0.161 | 90.166 ± 0.164 |
| seq256 @ 32×32 | 78.910 | 78.910 ± 0.000 | 79.551 ± 0.165 | 79.459 ± 0.225 |

`dataflow_only`'s std is exactly 0 in both cells — every one of its 5 seeds' 6-trial searches found the *same* best action (the categorical dataflow space has only 3 options; a 6-trial budget reliably samples all 3 regardless of seed, so the "best found" is deterministic here, not by construction of the search but as an empirical consequence of a small discrete space and an adequate budget for it).

**Gain attribution** (relative to `best_static`, `results/processed/exp5_ablation_summary.csv`'s `_gain_attribution` rows):

| Cell | `joint` gain (pct. points) | `dataflow_only` gain | `dataflow_only` % of joint gain | `tiling_only` gain | `tiling_only` % of joint gain |
|---|---|---|---|---|---|
| seq128 @ 16×16 | 0.206 | 0.000 | **0.0%** | 0.242 | **117.2%** |
| seq256 @ 32×32 | 0.549 | 0.000 | **0.0%** | 0.641 | **116.7%** |

## 5. Statistical Analysis

- **Comparisons made:** `tiling_only` vs. `joint`, paired by seed (both variants share the same seed set `[0,1,2,3,4]`, so a per-seed paired comparison is meaningful — the same seed drives both variants' TPE sampler initialization, differing only in which dimensions are searched). `dataflow_only` vs. `best_static` is not tested statistically — `dataflow_only`'s std is exactly 0 (§4), so there is no variance for a test to act on; the 0% gain is a deterministic fact, not a sample estimate.
- **Test used:** paired t-test (`scipy.stats.ttest_rel`) as primary, Wilcoxon signed-rank as the non-parametric alternative.
- **Result:**

  | Cell | Per-seed (tiling−joint) diffs | Paired t | p | Wilcoxon p |
  |---|---|---|---|---|
  | seq128 @ 16×16 | −0.062, +0.118, −0.122, 0.000, +0.243 | t=0.543 | p=0.616 | p=0.875 |
  | seq256 @ 32×32 | +0.017, −0.121, +0.293, 0.000, +0.269 | t=1.134 | p=0.320 | p=0.375 |

  **Neither comparison is significant** — `tiling_only`'s slightly higher mean (§4) is not distinguishable from `joint`'s at n=5; the per-seed differences are small and don't consistently favor one variant (2 of 5 seeds in cell 1, 1 of 5 in cell 2 actually favor `joint`). This is reported plainly rather than treating the raw mean-gain-attribution percentages (117%) as a confident "tiling beats joint" claim.
- **Multiple-comparisons correction:** not applicable (only 2 independent comparisons, both already non-significant).

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| — | — | — | — |

Zero `status == "FAILED"` rows in `results/raw/exp5.csv` (121 rows, all `OK`).

## 7. Figures

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| 1 | `results/figures/exp5_ablation_bars.png` (+ `.svg`) | Utilization by variant per cell, y-axis zoomed to the data range (labeled in-figure — differences are fractions of a percentage point on an ~80–90% baseline, invisible on a 0–100% axis) — `dataflow_only` (orange) visibly lower and flat (zero error bar) in both cells; `tiling_only` (blue) and `joint` (black) nearly identical with overlapping error bars. | `results/processed/exp5_ablation_summary.csv` |

## 8. Interpretation

- **[RESULT]** `dataflow_only` contributes **exactly 0%** of `joint`'s gain over the best static baseline, in both cells, deterministically (zero variance across seeds).
- **[RESULT]** `tiling_only`'s mean gain exceeds `joint`'s mean gain in both cells (117.2%, 116.7% "of joint's gain") — but this difference is **not statistically significant** (paired t-test p=0.616 and p=0.320; Wilcoxon p=0.875 and p=0.375).
- **[INTERPRETATION] — the answer the plan's acceptance criteria asks for, stated as precisely as the data supports.** Essentially all of `joint`'s real gain over the best static heuristic is attributable to the **tiling (SRAM-partition) search component** — `dataflow_only` demonstrably contributes nothing once the best static dataflow is already known (a clean, unambiguous, deterministic result), and `joint`'s gain is statistically indistinguishable from `tiling_only`'s alone. There is **no measurable evidence that jointly searching both dimensions together outperforms searching tiling alone** (with dataflow correctly fixed) at this budget — if anything, the (non-significant) point estimate suggests joint search may cost a little of its fixed 6-trial budget to the now-superfluous dataflow dimension, since `os_only`/`ws_only`-type choices only take one dataflow call to identify but still consume trials in the joint space.
- **[INTERPRETATION]** This directly qualifies Experiment 3's own finding (`exp3_stats.md` §8): the optimizer's real advantage over fair static baselines is real but small, and this experiment shows that advantage comes almost entirely from tiling, not from the optimizer "discovering" a good dataflow the static heuristics missed — the static heuristics already identify the best dataflow correctly (confirmed exactly by `dataflow_only`'s 0% result). The thesis's "joint dataflow+tiling learning" contribution, as currently implemented and at this budget, is not adding value beyond what a tiling-only search (given a correct static dataflow prior) would already capture.
- **[LIMITATION]** Only 2 of Experiment 3's 4 cells tested, both with `os_only` as the best static dataflow — the one Experiment 4 cell where the best static dataflow *changes* (`ws_only` at seq_len=512) was not included in this ablation; whether `dataflow_only` still contributes 0% there (where the "obviously best" dataflow itself shifts) is untested and would be a natural, informative follow-up.
- **[LIMITATION]** n=5 seeds is underpowered to distinguish the small `tiling_only`-vs-`joint` gap even if it were a genuine, reproducible effect — a larger seed count would be needed to confirm or rule out the (non-significant) direction seen here.
- **[LIMITATION]** `dram_bandwidth_bytes_per_cycle` not simulated, `energy_per_cycle_pj` a placeholder — identical caveats to every prior experiment.

Comparison against literature-reported figures is not directly applicable — the lit review's "joint dataflow+tiling" gap claim this experiment tests is a methodological point about this project's own optimizer design, not a number to compare against external systems.

---

## 11. Per-stage action-space re-run (2026-09-03) — reverses §8's headline finding

`docs/NEXT_INSTRUCTIONS.md` item 1, directly following §9's own second flagged
follow-up: once Experiment 3's action space moved to per-stage dataflow choice
(`exp3_stats.md` §11), this ablation should be re-run to check whether
`dataflow_only` — now able to vary dataflow **per stage** instead of one global
choice — contributes a real, non-zero share of the gain. **It does.**
Everything in §1–10 above is left as-is, unmodified (plan ground rule 6) — this
section is the new, authoritative comparison.

> **Headline finding:** `dataflow_only` no longer contributes 0% — it now
> captures **over 100%** of `joint`'s gain over the best single-dataflow
> baseline (130.2%, 104.4%), essentially matching `tiling_only` and `joint`
> themselves. All three variants are now statistically indistinguishable from
> each other (paired tests, all p > 0.23). The thesis's "joint dataflow+tiling
> learning" question, at this budget, now has a different answer than §8's: not
> "tiling alone explains everything," but "either factor alone, on the
> per-stage action space, already captures essentially all of the achievable
> gain — the two factors are no longer distinguishable in their contribution."

**Method:** identical to §3 (same 2 cells: seq128@16×16, seq256@32×32; same 5
seeds; same 6-iteration budget) — only the action space differs, and
`tiling_only`'s fixed dataflow is now **per-stage** (looked up from Experiment
3's real `best_fixed_per_stage` per-stage winners for each cell, not one global
`os_only` string — see `run_exp5.py`'s docstring for why this is the more
meaningful reference now that dataflow varies per stage). `joint` is `a2c`
(the current selection, `exp0_stats.md` §11), reused from
`results/raw/exp3_perstage.csv`, not `bayesian_opt`.

**A real invocation mistake caught and fixed, not hidden:** the first attempt
at this re-run defaulted to the same output filename (`results/raw/exp5.csv`)
as the original run and was launched without an explicit `--out`, appending
120 new rows onto the 120 original (3-dim) rows in the same file. Caught before
analysis by inspecting `git_commit` values (two distinct commits present in one
file); the two batches were cleanly split by `git_commit` back into the
original `exp5.csv` (restored via `git checkout`, verified byte-identical to
its last commit) and a new `results/raw/exp5_perstage.csv` (120 rows, the real
per-stage data). No data was lost or corrupted — this was a filename-collision
bookkeeping mistake, not a data-quality problem.

```bash
python experiments/exp5_ablation/run_exp5.py --out results/raw/exp5_perstage.csv
python experiments/exp5_ablation/analyze_exp5.py \
  --exp5-raw-csv results/raw/exp5_perstage.csv --exp3-raw-csv ../exp3_main_sweep/results/raw/exp3_perstage.csv \
  --out-prefix exp5_perstage --joint-optimizer-name a2c
```

### 11a. Results

**Per-variant utilization (%), n=5 seeds** (`results/processed/exp5_perstage_ablation_summary.csv`):

| Cell | Best single-dataflow (`os_only`) | `best_fixed_per_stage` (context) | `dataflow_only` (mean ± std) | `tiling_only` (mean ± std) | `joint` (mean ± std) |
|---|---|---|---|---|---|
| seq128 @ 16×16 | 89.960 | 91.892 | 91.843 ± 0.735 | 92.222 ± 0.153 | 91.406 ± 1.716 |
| seq256 @ 32×32 | 78.910 | 87.451 | 86.741 ± 0.972 | 87.863 ± 0.209 | 86.408 ± 2.244 |

`dataflow_only`'s std is **no longer zero** (was exactly 0 in §4) — real,
mechanistic explanation: the per-stage dataflow space has `3^3 = 27` discrete
combinations (was 3), and a 6-trial budget no longer exhaustively covers it, so
different seeds land on different local optima. `tiling_only` has the lowest
variance and highest mean of the three variants in both cells — it numerically
**beats** `best_fixed_per_stage` itself (92.22 > 91.89; 87.86 > 87.45).

**Gain attribution** (relative to `best_single_dataflow` — **kept as the same
reference as §4** for direct comparability, not switched to
`best_fixed_per_stage`; see 11a's own note below for why):

| Cell | `joint` gain (pct. points) | `dataflow_only` gain | `dataflow_only` % of joint gain | `tiling_only` gain | `tiling_only` % of joint gain |
|---|---|---|---|---|---|
| seq128 @ 16×16 | 1.446 | 1.883 | **130.2%** (was 0.0%) | 2.262 | **156.4%** (was 117.2%) |
| seq256 @ 32×32 | 7.499 | 7.832 | **104.4%** (was 0.0%) | 8.954 | **119.4%** (was 116.7%) |

**[NOTE on baseline choice]** `best_fixed_per_stage` was tried as the
gain-attribution reference first (matching `tiling_only`'s own now-per-stage
fixed-dataflow prior) but produces a **degenerate result**: since `joint` no
longer reliably beats `best_fixed_per_stage` (`exp3_stats.md` §11 — 0/4 cells
significant, numerically below it in both of these 2 cells), `joint_gain` is
*negative* against that reference, making "% of joint's gain" sign-flip into
confusing negative percentages for a plainly *positive* `tiling_only` gain.
Verified by computing it before deciding not to use it (real evidence, not
assumed) — kept `os_only` as the reference instead, exactly preserving §4's
methodology so the 0%→130%/104% comparison is apples-to-apples.
`best_fixed_per_stage`'s utilization is still reported alongside (11a's table)
as informative context.

**Significance** (`results/processed/exp5_perstage_significance.csv`; paired
by seed, same convention as §5):

| Cell | Comparison | t, p | Wilcoxon p |
|---|---|---|---|
| seq128 @ 16×16 | tiling_only vs. joint | t=1.05, p=0.355 | p=0.4375 |
| seq128 @ 16×16 | dataflow_only vs. joint | t=0.51, p=0.634 | p=1.0000 |
| seq256 @ 32×32 | tiling_only vs. joint | t=1.38, p=0.239 | p=0.4375 |
| seq256 @ 32×32 | dataflow_only vs. joint | t=0.32, p=0.767 | p=1.0000 |

**Neither `tiling_only` nor `dataflow_only` differs significantly from
`joint`, in either cell** (all p > 0.23) — despite `dataflow_only` now having
real variance to test (unlike §5, where its zero variance made a test
inapplicable).

**Figure:** `results/figures/exp5_perstage_ablation_bars.png` (+`.svg`) — same
plot as §7, new-action-space data.

### 11b. Interpretation

- **[RESULT]** `dataflow_only` captures over 100% of `joint`'s gain in both
  cells (130.2%, 104.4%) — a complete reversal of §8's headline "exactly 0%"
  finding, directly answering `docs/NEXT_INSTRUCTIONS.md` item 1's first
  question.
- **[RESULT]** `joint` does not significantly beat `tiling_only` (unchanged
  from §5 — still not significant, p=0.24–0.36 now vs. p=0.32–0.62 before) —
  directly answering item 1's second question: no, it still doesn't, now
  extended to `dataflow_only` too (also not significant, p=0.63–0.77).
- **[INTERPRETATION] — the central finding of this re-run.** On the per-stage
  action space, `dataflow_only`, `tiling_only`, and `joint` are all
  **statistically indistinguishable** from each other. This is a genuinely
  different conclusion from §8's, not a stronger or weaker version of it: §8
  found tiling explains *all* of the gain and dataflow explains *none*; this
  re-run finds *either* factor alone, searched with the per-stage granularity
  `best_fixed_per_stage` already had, already captures essentially all of the
  achievable gain at this budget — the two factors are no longer separable by
  their contribution. Combined with `exp3_stats.md` §11's finding that the
  optimizer now matches (not beats) `best_fixed_per_stage`, the coherent
  picture is: **per-stage dataflow choice was the real missing capability**
  (§8's own `exp3_stats.md`-referenced concrete recommendation) — once any
  search method has access to it (whether searching dataflow alone, tiling
  alone with dataflow correctly fixed per-stage, or both jointly), it reaches
  essentially the same ceiling.
- **[RESULT]** `tiling_only` numerically *beats* `best_fixed_per_stage` itself
  in both cells (92.22% > 91.89%; 87.86% > 87.45%) — a real, if small,
  additional finding: a per-stage-correct dataflow prior plus a *learned*
  (rather than default equal-split) SRAM partition can exceed the exhaustive
  per-stage-EDP-optimal static baseline. Not tested for significance here
  (not one of item 1's explicit questions) — flagged as a real observation
  worth a dedicated significance check if this becomes a thesis claim.
- **[LIMITATION]** Same 2-of-4-cells scope as §9 — the seq_len=512 cell where
  the best static dataflow itself changes (`exp4_stats.md`) remains untested
  for this ablation; still a natural, informative follow-up, now doubly so
  given how much the per-stage redesign changed this experiment's answer.
- **[LIMITATION]** n=5 seeds — same underpowered-for-small-effects caveat as
  §9's own limitation.
- **[LIMITATION]** `dram_bandwidth_bytes_per_cycle` unsimulated,
  `energy_per_cycle_pj` a placeholder — identical caveat to §9.

## 9. Limitations & Reasons for Discrepancy

- **The central, disclosed limitation is that "joint captures more gain than either factor alone" — the plan's implicit expectation for this kind of ablation — is NOT what this data shows.** `dataflow_only` captures none of the gain and `tiling_only` alone statistically matches `joint`. This is reported as the real finding, not softened or reframed as a partial success.
- **Recommended next step:** test whether this holds at the seq_len=512 scale where the best static dataflow itself changes (Experiment 4's finding) — if `dataflow_only` no longer contributes 0% there, that would show joint search's value is scale-dependent, a materially different and more nuanced conclusion than this experiment alone supports.
- **A second natural follow-up**, given Experiment 3's own action-space-asymmetry finding (`exp3_stats.md` §8): once the action space is extended to per-stage dataflow choice (Experiment 3's recommended follow-up), this ablation should be re-run — a per-stage-capable joint search might show a real, non-zero dataflow contribution that the current single-global-dataflow action space cannot express.

## 10. Reproduction

```bash
cd "MSc Thesis"   # project root
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/   # if not already built
python experiments/exp5_ablation/run_exp5.py \
  --cells exp3_seq128:16x16 exp3_seq256:32x32 \
  --seeds 0 1 2 3 4 --n-iterations 6 --out results/raw/exp5.csv
python experiments/exp5_ablation/analyze_exp5.py
```

`dataflow_only`/`tiling_only` (optuna TPE) are seeded but not guaranteed bit-for-bit reproducible across machines/library-patch-versions (same caveat as every prior optimizer-involving experiment); `dataflow_only`'s specific *result* (0% gain) is robust to this, since it's a deterministic consequence of a 3-option space and a 6-trial budget, not a close numerical estimate.

**Acceptance criteria for this experiment (per the plan):** The stats.md states plainly how much of the total gain is attributable to joint optimization vs. either factor alone. **Met**: §4/§8 state it exactly (`dataflow_only`: 0%; `tiling_only`: ~117%, not significantly different from `joint`'s 100%) — including the honest, plan-relevant conclusion that joint optimization is not shown to add value beyond tiling alone at this budget, rather than reporting only the flattering half of that finding.
