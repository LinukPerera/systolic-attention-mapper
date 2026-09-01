# Results Tracker

Single top-level summary of where every experiment stands and what it found.
This file is a **rollup**, not a source of truth: every number here must trace
back to the experiment's own `experiments/expN_*/expN_stats.md` (which in turn
traces back to `results/raw/`, per `docs/CODING_AGENT_PLAN.md` §0 ground rule 2).
If a number here and the number in the experiment's `stats.md` ever disagree,
the `stats.md` is correct and this file has a bug — fix this file, never the
other way round.

**How to update this file:** only after an experiment's `expN_stats.md` is
filled in from real logged data (not while it's still a placeholder). Copy the
headline number(s) and `[RESULT]`/`[INTERPRETATION]`/`[LIMITATION]` tags
across verbatim — do not rephrase a `[RESULT]` into something stronger than
what the data showed. An experiment with `status: FAILED` runs in its raw CSV
is still summarized here, honestly, not omitted.

---

## Build status (infrastructure, not an experiment result)

Phase A Stage 0 (repo scaffold as a git repo + real `common/` plumbing) and the
start of Phase B (real, verified Docker+SCALE-Sim integration — see
[README.md](README.md#status)) are complete as of 2026-08-31. Docker is running
and a real `scalesim:latest` image is built; `common/simulator_interface.py` and
`common/workload_gen.py` were rewritten against the actual installed SCALE-Sim's
verified I/O formats (not guessed ones). **Real finding for the methodology
chapter:** SCALE-Sim supports only WS/OS/IS dataflows — there is no RS
(row-stationary) mode, contradicting `CODING_AGENT_PLAN.md` §1's module comment.
**Real finding:** `stable-baselines3` has no plain REINFORCE; Experiment 0 uses
A2C in its place (documented in `common/optimizers/a2c.py`).

---

## Status at a glance

| # | Experiment | Status | Date run | Headline result | Report |
|---|---|---|---|---|---|
| 0 | Optimizer bake-off | 🟢 Complete | 2026-08-31 | Bayesian opt (optuna TPE) selected — smallest distance from ground truth (1.06), fastest convergence (0.6 iters), lowest variance (std 0.83) | [exp0_stats.md](experiments/exp0_optimizer_bakeoff/exp0_stats.md) |
| 1 | Array-size sensitivity | 🟢 Complete | 2026-09-01 | 16×16/32×32 justified mechanistically: the utilization "knee" tracks `head_dim`, not `seq_len` — a stated prediction (doubling head_dim 32→64 shifts the knee one array-size doubling higher) confirmed by a dedicated run (see [Justified array sizes](#justified-array-sizes) below) | [exp1_stats.md](experiments/exp1_array_sensitivity/exp1_stats.md) |
| 2 | Search/training convergence | 🟢 Complete | 2026-09-01 | 5/5 seeds converged (mean iter 4.00, std 0.00) by a pre-stated criterion; a 10× stricter ε changed nothing — a structural criterion limitation (early-trial clustering), not a tuning gap. Exp 3 uses fixed budgets, not early-stopping, as a result | [exp2_stats.md](experiments/exp2_convergence/exp2_stats.md) |
| 3 | Main sweep vs. baselines | 🟢 Complete | 2026-09-01 | `bayesian_opt` loses to `best_fixed_per_stage` significantly in all 4 cells (−1.9% to −9.1%) but beats the best single-dataflow baseline significantly in all 4 (+0.2% to +0.8%) — a real action-space asymmetry, not a search failure: the optimizer can't express per-stage dataflow choice (see §8 of the report) | [exp3_stats.md](experiments/exp3_main_sweep/exp3_stats.md) |
| 4 | Scalability study | 🔴 Not started | — | — | [exp4_stats.md](experiments/exp4_scalability/exp4_stats.md) |
| 5 | Ablation study | 🔴 Not started | — | — | [exp5_stats.md](experiments/exp5_ablation/exp5_stats.md) |
| 6 | Generalization test (stretch) | 🔴 Not started | — | — | [exp6_stats.md](experiments/exp6_generalization/exp6_stats.md) |

Status legend: 🔴 Not started · 🟡 In progress · 🟢 Complete (stats.md finalized) · ⚫ Failed/blocked (see notes)

---

## Selected optimizer

`config/optimizers/selected.yaml` — set by Experiment 0, 2026-08-31.

- **Winner:** `bayesian_opt` (optuna TPE sampler)
- **Reason:** Across 5 seeds on the exp0 pilot workload/16×16 array, smallest mean distance from the brute-force ground-truth best reward (1.06, vs. 2.08 for a2c/ppo, 3.70 for evolutionary), fastest convergence (mean 0.6 iterations to within 5% of ground truth, vs. 5.2 for a2c/ppo), and lowest variance in final reward (std 0.83, vs. evolutionary's 2.56). All 5 seeds converged. See `exp0_stats.md` §4/§8 for the full table and caveats (small pilot scale, no significance test at n=5 — flagged as limitations, not hidden).

## Justified array sizes

Set by Experiment 1 — mechanistic explanation confirmed 2026-09-01.

- **Primary sizes:** `16×16` and `32×32`.
- **Mechanistic basis (primary argument):** Experiment 1's utilization "knee" (a sharp acceleration in decline rate as array size grows) tracks `head_dim = hidden_dim/num_heads`, not `seq_len` or array size directly. Traced to `common/workload_gen.py`: `qk_transpose`'s K dimension and `score_times_v`'s N dimension are fixed at `head_dim` regardless of `seq_len` — only `qkv_projection` scales enough to use an array bigger than `head_dim`, so two of the three GEMM stages structurally cannot fill any array larger than `head_dim`. This directly reproduces the literature review's per-stage GEMM heterogeneity argument (§2.2, Ivanov et al./Kao et al.). **Tested, not just observed:** the falsifiable prediction that doubling `head_dim` (32→64, `seq_len`/`num_heads` held fixed) would shift the knee one array-size doubling higher (~32×32 → ~64×64) was stated before running the validation, then confirmed by a dedicated run (`config/workloads/exp1_head64_validation.yaml`) — the knee moved from between 32×32–64×64 to between 64×64–128×128, exactly as predicted. 16×16/32×32 sit below every tested `head_dim` (16, 32, 64), including the `head_dim=64` now standardized for Experiments 2+ (below) — this is why they're justified as primary sizes, not merely that they performed well at one specific workload scale.
- **Supporting detail (decline-rate numbers, not the main argument):** at `head_dim=32`, mean |%-change| is 13.2% across 8→16→32 vs. 49.7% across 32→64→128 (~3.8× acceleration); at `head_dim=64`, 15.8% across 16→32→64 vs. 48.9% across 64→128 (~3.1× acceleration). The plan's literal `|utilization %-change| < 2%` per-doubling criterion is never satisfied at any workload scale tested (0/44 transitions across all three runs) — a real result, reported as such, not silently redefined; the mechanistic explanation above is what actually grounds the array-size choice.
- **Standard adopted for Experiments 2+:** `config/workloads/README.md` now mandates `head_dim=64` (fixed, matching BERT-base/GPT-2) for every Experiment 2+ workload config — vary `seq_len`/`num_heads`, not `hidden_dim` directly. `head_dim=16`/`32` (Exp 1's original and representative runs) both understated how much of a larger array the two head_dim-capped stages could use.
- **A real infeasibility, handled correctly:** the head64-validation run's 8×8 array config genuinely could not be simulated (Docker OOM, `exit 137`, on every dataflow) — logged as real `FAILED` evidence, not hidden (see "Failed / excluded runs" below). Fixed a real bug along the way: `best_fixed_per_stage_mapping` used to raise and crash the whole sweep when every candidate failed for one stage; it now logs the failure and lets the sweep continue to the next array size.
- **Caveat:** the mechanism is confirmed at exactly two `head_dim` values (32, 64) and one `seq_len` (256) — real, tested evidence, not a full characterization of the whole (`seq_len`, `head_dim`) space. Experiment 3's own multi-`seq_len` × multi-array sweep (at the now-standard `head_dim=64`) will provide much broader evidence on whether 16×16/32×32 hold across `seq_len`; if it disagrees, Experiment 3's data is authoritative, not this section.

Full diagnosis and confirmation: `exp1_stats.md` §4a–c/§8–9. Original handoff doc: [`docs/NEXT_INSTRUCTIONS.md`](docs/NEXT_INSTRUCTIONS.md).

---

## Per-experiment summary

Fill in each block only once the corresponding `expN_stats.md` is finalized.
Copy tagged claims verbatim from that file's §8 (Interpretation); do not add
new interpretation here that isn't already in the source report.

### Experiment 0 — Optimizer bake-off
**Status:** 🟢 Complete (2026-08-31)

- **[RESULT]** Across 5 seeds, `bayesian_opt` (optuna TPE) had the smallest mean distance from the brute-force ground-truth best reward (1.06), the fastest mean convergence to within 5% of that ground truth (0.60 iterations), the lowest final-reward standard deviation (0.83), and 5/5 seeds converging.
- **[RESULT]** `a2c` and `ppo` produced near-identical final-reward distributions (66.68 ± 0.61 for both).
- **[RESULT]** `evolutionary` had the worst mean final reward (65.06), the highest variance (std 2.56), and one non-converged seed.
- **[INTERPRETATION]** `a2c`/`ppo`'s near-identical trajectories reflect this experiment's very small iteration budget (15 evals/seed) dominating both algorithms' behavior with near-random early exploration — not evidence the two methods are equivalent in general.
- **[LIMITATION]** Pilot workload was reduced 5–15x below the plan's example scale, and evaluation budgets were small, specifically to finish in one session (real per-evaluation cost at the plan's literal scale was measured at ~178s — see `exp0_stats.md` §3). No significance test was run at n=5 seeds. Both are real, disclosed scope limits, not hidden ones — full detail in `exp0_stats.md` §8–9.

### Experiment 1 — Array-size sensitivity study
**Status:** 🟢 Complete (2026-09-01, includes representative-scale and head_dim-validation re-runs)

- **[RESULT]** At `head_dim=16` (original run), all 16 doubling transitions decline more than 2% with no knee anywhere. At `head_dim=32` (representative run), a knee appears between 32×32–64×64. At `head_dim=64` (validation run), the knee appears one doubling higher, between 64×64–128×128 — exactly matching a stated prediction made before the validation run.
- **[RESULT]** The knee tracks `head_dim` exactly (it sits at `head_dim` in both the head_dim=32 and head_dim=64 runs) — a mechanistic cause traced to `common/workload_gen.py`'s fixed-at-head_dim GEMM dimensions, not a coincidental correlation.
- **[RESULT]** `best_fixed_per_stage` is the top performer at every array size in all three runs; its advantage over single-dataflow baselines is largest below each run's own knee and shrinks sharply above it.
- **[RESULT]** The head_dim=64 validation run's 8×8 array was genuinely infeasible to simulate (Docker OOM, `exit 137`, all dataflows) — real evidence, logged as `FAILED`, not hidden; see "Failed / excluded runs" below.
- **[INTERPRETATION]** 16×16/32×32 are justified because they sit below every tested `head_dim`, including the `head_dim=64` now standardized for Experiments 2+ — a mechanistic argument, not just an empirical decline-rate observation at one workload scale (see [Justified array sizes](#justified-array-sizes) above).
- **[LIMITATION]** Mechanism confirmed at exactly two `head_dim` values and one `seq_len` — Experiment 3's own data (at the now-standard `head_dim=64`) is authoritative if it disagrees with this provisional choice.

### Experiment 2 — Search/training convergence study
**Status:** 🟢 Complete (2026-09-01)

- **[RESULT]** 5/5 seeds converged by the stated criterion (Var of best-reward-so-far over a trailing 5-iteration window < 1.0), all at exactly iteration 4 (mean 4.00, std 0.00).
- **[RESULT]** Final best-reward spread across seeds is small: 85.72 ± 0.47 (range 84.81–86.02).
- **[RESULT] — the criterion, taken literally, triggers well before trajectories visually stabilize.** Every seed keeps improving substantially past iteration 4; the mean trajectory doesn't flatten until ~iteration 25. EPSILON=1.0 is ≈3.6% of the observed reward range — loose enough that an early plateau within a 1.0-point band satisfies the formula despite later iterations finding meaningfully better regions.
- **[INTERPRETATION]** This is a real property of the chosen (WINDOW, EPSILON), not evidence the optimizer stops improving at iteration 4 — the criterion answers a narrower question ("has best-so-far been locally stable") than "has search found its final answer."
- **[RESULT — checked, not left open]** A 10× stricter EPSILON (0.1, re-analyzed on the same raw data, no re-simulation needed) changed nothing — all 5 seeds still converge at iteration 4. Root cause: the first triggering window's actual variance is already 0.0000–0.0137, far below either threshold, because TPE's initial exploration trials happen to cluster tightly by chance. A structural limitation of a trailing-window variance test, not an epsilon-tuning gap.
- **[DECISION, adopted]** Given this, Experiment 3 does not use this criterion to dynamically early-stop optimizer runs — every seed runs a fixed, real-timed budget, and `convergence_iteration` (now shared via `common.optimizers.convergence`) is reported per seed as a descriptive statistic only. Full detail: `exp2_stats.md` §8a.

### Experiment 3 — Main sweep vs. baselines
**Status:** 🟢 Complete (2026-09-01)

- **[RESULT]** `bayesian_opt` loses to `best_overall` (always `best_fixed_per_stage`) significantly in all 4 cells (−1.88% to −9.14%, all p<0.0001 by t-test).
- **[RESULT]** `bayesian_opt` beats `best_single_dataflow` (always `os_only`) significantly in all 4 cells by t-test (+0.23% to +0.79%, all p<0.05) — though only 2/4 survive a conservative Bonferroni correction.
- **[INTERPRETATION] — central finding, not a footnote.** Explained by a genuine action-space asymmetry, not a search failure: `common/optimizers/environment.py` applies one dataflow to the WHOLE workload per run (SCALE-Sim's Dataflow setting is per-run, not per-layer), while `best_fixed_per_stage` achieves its edge by running 3 separate simulations (one per GEMM stage) and combining them — a capability the learned optimizer's action space doesn't have. The optimizer reaches slightly above its true ceiling's raw baseline (`os_only`) in every cell, which is exactly what its SRAM-partition search should buy it.
- **[LIMITATION → concrete follow-up work]** This is a real gap versus the thesis's "joint dataflow+tiling learning" claim — recommend extending the action space to a per-stage dataflow choice so the optimizer can compete with `best_fixed_per_stage` on equal footing. Directly relevant to how Experiment 5's ablation is scoped (even "joint" is currently capped at one global dataflow).
- **[LIMITATION]** Grid reduced to seq_len∈{128,256} (real, measured infeasibility of 512/1024 at 16×16 — see [Justified array sizes](#justified-array-sizes)); 6-iteration optimizer budget; no multiple-comparisons correction applied by default (reported both ways).

### Experiment 4 — Scalability study
**Status:** 🔴 Not started

- **[RESULT]** —
- **[INTERPRETATION]** —
- **[LIMITATION]** —

### Experiment 5 — Ablation study
**Status:** 🔴 Not started

- **[RESULT]** —
- **[INTERPRETATION]** —
- **[LIMITATION]** —

### Experiment 6 — Generalization test (optional/stretch)
**Status:** 🔴 Not started

- **[RESULT]** —
- **[INTERPRETATION]** —
- **[LIMITATION]** —

---

## Failed / excluded runs (cross-experiment)

Only entries whose experiment-level `expN_stats.md` §6 already documents them —
this section exists so a reader doesn't have to open all seven reports to see
whether anything failed.

| Experiment | Run ID | Condition | Failure reason |
|---|---|---|---|
| exp1 (head64 validation) | `exp1-{ws,os,is}_only-8x8-*`, `exp1-best_fixed_per_stage_search-8x8-*` (qkv_projection stage), `exp1-best_fixed_per_stage-8x8-mixed` (8 rows total) | 8×8 array, `head_dim=64` workload | Docker OOM-killed (`exit 137`), 88–98% through each simulation — real infeasibility (largest GEMM in any Exp 1 run vs. smallest array), not a code fault. Full detail: `exp1_stats.md` §6. |

---

_Last updated: 2026-09-01 (Experiments 0–3 complete; Exp 3 found a real action-space asymmetry — the optimizer beats fair single-dataflow baselines but loses to the per-stage-specialized best_fixed_per_stage, flagged as concrete follow-up work; Experiments 4–6 not started)._
