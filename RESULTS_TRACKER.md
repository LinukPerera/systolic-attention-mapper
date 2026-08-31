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
| 1 | Array-size sensitivity | 🟢 Complete | 2026-08-31 | 0/16 doublings meet the <2% sufficiency criterion — utilization declines monotonically 8×8→128×128 at this pilot's small workload scale; does not, on its own, justify 16×16/32×32 (see [Justified array sizes](#justified-array-sizes) below) | [exp1_stats.md](experiments/exp1_array_sensitivity/exp1_stats.md) |
| 2 | Search/training convergence | 🔴 Not started | — | — | [exp2_stats.md](experiments/exp2_convergence/exp2_stats.md) |
| 3 | Main sweep vs. baselines | 🔴 Not started | — | — | [exp3_stats.md](experiments/exp3_main_sweep/exp3_stats.md) |
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

**Not justified by Experiment 1's data as run** — set by Experiment 1, 2026-08-31, but the result was negative, not a size choice. See `exp1_stats.md` §8 for full detail.

- **Primary sizes:** Not determined. Experiment 1 reused Experiment 0's small pilot workload (seq_len=64) for wall-clock reasons; at that scale, utilization declines monotonically from 8×8 (the smallest tested) all the way to 128×128 with zero doubling transitions meeting the sufficiency criterion below — i.e. the data says *smaller is better* within the tested range, not that 16×16/32×32 are a plateau point.
- **Criterion used:** `|utilization %-change| < 2%` per array-size doubling (the plan's own suggested threshold) — applied to real data, 0/16 transitions across 4 methods met it.
- **Why this doesn't settle the question:** the workload tested (seq_len=64) is much smaller than Experiment 3+'s planned seq_len range (128–4096); a fixed small workload will underutilize any array size tested, so this result reflects workload-scale mismatch, not evidence against 16×16/32×32 specifically. **Next step:** either re-run Experiment 1 with a workload near Experiment 3's actual scale (flagged as follow-up, same deferral pattern as Exp 0's own scale reduction), or treat Experiment 3's own seq_len × array-size sweep as the real evidence base once it exists, and revisit this section then.

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
**Status:** 🟢 Complete (2026-08-31)

- **[RESULT]** Every one of the 16 array-size-doubling transitions tested (4 methods × 4 transitions each, 8×8 through 128×128) shows a utilization decrease larger than 2% (range: −11.1% to −75.3%). Zero transitions meet the plan's stated sufficiency criterion anywhere in the range tested.
- **[RESULT]** `best_fixed_per_stage` has the highest utilization at every array size (88.3% at 8×8 down to 8.4% at 128×128), consistently above all three single-dataflow baselines.
- **[INTERPRETATION]** The monotonic decline (rather than a plateau) is consistent with the pilot workload (seq_len=64) being small relative to every array size tested, including the smallest — larger arrays only make an existing under-utilization gap worse, not a new phenomenon that appears past some threshold.
- **[LIMITATION] — central to this experiment's outcome, not a footnote.** Because Exp 1 reused Exp 0's small pilot workload for wall-clock reasons (not because it's representative of Experiment 3+'s planned seq_len range), this run cannot justify or reject 16×16/32×32 as primary sizes — see [Justified array sizes](#justified-array-sizes) above for the honest scope of what this data does and doesn't show, and the recommended next step.

### Experiment 2 — Search/training convergence study
**Status:** 🔴 Not started

- **[RESULT]** —
- **[INTERPRETATION]** —
- **[LIMITATION]** —

### Experiment 3 — Main sweep vs. baselines
**Status:** 🔴 Not started

- **[RESULT]** —
- **[INTERPRETATION]** —
- **[LIMITATION]** —

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
| | | | |

---

_Last updated: 2026-08-31 (Experiments 0 and 1 complete; Experiments 2–6 not started)._
