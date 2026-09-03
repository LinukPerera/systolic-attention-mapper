# Next Instructions for the Coding Agent

Context lives in `RESULTS_TRACKER.md` (read that first — especially "Per-stage
action space", the "Infrastructure incident" section, and Experiments 3-5's
summaries). This file lists only what to do next, triaged by yield-per-compute
given real time/CPU constraints (single core, ~108% load, slow). When an item
is done, delete it from this file rather than marking it complete in place.

---

## 1. Experiment 5 per-stage re-run (highest yield, do first)

Smallest remaining job (2 cells, same budget as the original run — no new
iteration budget needed) and the most directly informative: it retests whether
`dataflow_only` still captures 0% of the optimizer's gain now that Experiment
3's per-stage re-run closed the gap with `best_fixed_per_stage`. This is
likely the single most important remaining number for the thesis's "joint
dataflow+tiling learning" claim. Log to a new CSV (don't overwrite the
original), update `exp5_stats.md` with old vs. new results clearly labelled.

## 2. Experiment 4 per-stage re-run — scoped down

Re-run at seq_len=512 using the **same 6-iteration budget as the original run**
first (not a larger one) — this isolates whether the per-stage action space
alone changes the seq_len=512 reversal finding, at the same cost as before.
Only attempt a larger-budget follow-up afterward if time remains; do not
combine both changes (action space + budget) in one run, or it becomes
impossible to tell which change caused any difference. Log to a new CSV,
update `exp4_stats.md`.

## 3. After 1-2 land

Update `RESULTS_TRACKER.md`'s status table and per-experiment summaries from
the finalized `expN_stats.md` files, same handoff pattern as every prior step.
At that point every experiment the thesis's core claims depend on (0, 1, 2, 3,
5, and 4's seq_len=512 point) is on the per-stage action space, and the
Results & Evaluation chapter can be written directly from `docs/stats/*.md` /
each experiment's own `expN_stats.md` and figures, per `CODING_AGENT_PLAN.md`
§7's definition of done.
