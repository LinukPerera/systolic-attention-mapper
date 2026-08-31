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

Phase A Stage 0 (repo scaffold as a git repo + real `common/` plumbing —
`workload_gen.py`, `simulator_interface.py`, `metrics.py`, `logging_utils.py`,
`plotting/style.py`, `baselines/static_heuristics.py`, all with passing unit
tests) is complete as of 2026-08-31; see [README.md](README.md#status) for detail.
This is prerequisite plumbing, not a result — it does not move any experiment
below off 🔴, and none of the `[RESULT]`/`[INTERPRETATION]` blocks change until an
experiment actually runs. Docker's CLI is installed but its daemon is not
currently reachable on this machine — the one thing blocking any real
`run_scale_sim` call, and therefore Phase B / Experiment 0.

---

## Status at a glance

| # | Experiment | Status | Date run | Headline result | Report |
|---|---|---|---|---|---|
| 0 | Optimizer bake-off | 🔴 Not started | — | — | [exp0_stats.md](experiments/exp0_optimizer_bakeoff/exp0_stats.md) |
| 1 | Array-size sensitivity | 🔴 Not started | — | — | [exp1_stats.md](experiments/exp1_array_sensitivity/exp1_stats.md) |
| 2 | Search/training convergence | 🔴 Not started | — | — | [exp2_stats.md](experiments/exp2_convergence/exp2_stats.md) |
| 3 | Main sweep vs. baselines | 🔴 Not started | — | — | [exp3_stats.md](experiments/exp3_main_sweep/exp3_stats.md) |
| 4 | Scalability study | 🔴 Not started | — | — | [exp4_stats.md](experiments/exp4_scalability/exp4_stats.md) |
| 5 | Ablation study | 🔴 Not started | — | — | [exp5_stats.md](experiments/exp5_ablation/exp5_stats.md) |
| 6 | Generalization test (stretch) | 🔴 Not started | — | — | [exp6_stats.md](experiments/exp6_generalization/exp6_stats.md) |

Status legend: 🔴 Not started · 🟡 In progress · 🟢 Complete (stats.md finalized) · ⚫ Failed/blocked (see notes)

---

## Selected optimizer

_Not yet decided — set by Experiment 0. Once decided:_ `config/optimizers/selected.yaml`
_names the winner; record it here with the one-line data-backed reason from
`exp0_stats.md` §8._

- **Winner:** —
- **Reason:** —

## Justified array sizes

_Not yet decided — set by Experiment 1._

- **Primary sizes:** —
- **Criterion used:** —

---

## Per-experiment summary

Fill in each block only once the corresponding `expN_stats.md` is finalized.
Copy tagged claims verbatim from that file's §8 (Interpretation); do not add
new interpretation here that isn't already in the source report.

### Experiment 0 — Optimizer bake-off
**Status:** 🔴 Not started

- **[RESULT]** —
- **[INTERPRETATION]** —
- **[LIMITATION]** —

### Experiment 1 — Array-size sensitivity study
**Status:** 🔴 Not started

- **[RESULT]** —
- **[INTERPRETATION]** —
- **[LIMITATION]** —

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

_Last updated: 2026-08-31 (Stage 0 build-out complete; no experiments run yet)._
