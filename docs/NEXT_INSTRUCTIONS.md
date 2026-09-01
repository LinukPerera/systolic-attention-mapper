# Next Instructions for the Coding Agent

Context for every item below lives in `RESULTS_TRACKER.md` (read that first,
especially Experiments 3 and 5's summaries) — this file lists only what to do
next, not why. Do not add narrative/history here; when a step is done, delete
it from this file rather than marking it complete in place.

---

## 2. Decide whether Experiment 0's optimizer selection still holds

`bayesian_opt` was selected on the *old*, smaller action space. A larger
per-stage action space may change which search strategy is most stable/scalable
(e.g. evolutionary or RL may handle a bigger joint discrete+continuous space
differently than TPE). Either re-run Experiment 0's bake-off on the new action
space (small pilot scale, same as before) to confirm the selection still holds,
or explicitly document a reasoned decision not to and proceed with
`bayesian_opt` regardless — don't silently carry the old selection forward
unexamined.

## 3. Re-run Experiment 3 (main sweep) on the new action space

Same grid as before (seq_len∈{128,256} × array∈{16×16,32×32}, real, measured
infeasibility of 512/1024 at 16×16 stands). Log to a new CSV (don't overwrite
the existing `results/raw/exp3.csv`), re-run the significance tests, and check
whether `bayesian_opt` now closes the gap with `best_fixed_per_stage`. Update
`exp3_stats.md` with both the old and new action-space results, clearly labelled.

## 4. Re-run Experiment 5 (ablation) on the new action space

Same 2 cells as before. Check whether `dataflow_only` now captures a nonzero
share of `joint`'s gain, and whether `joint` significantly beats `tiling_only`
now that dataflow is meaningfully searchable per-stage. Update `exp5_stats.md`
accordingly, same old/new labelling convention as Experiment 3.

## 5. Revisit Experiment 4 (scalability) after 3–4 land

Experiment 4 already flagged its own follow-up (larger iteration budget at
seq_len=512, where the fixed 6-iteration budget missed the relocated optimum
in one of five seeds) — fold that into this re-run rather than doing a third
separate pass. Re-run at seq_len=512 on the new action space with a larger,
justified iteration budget; update `exp4_stats.md`.

## 6. Experiment 6 (generalization, optional/stretch)

Only after 1–5 above are complete and documented, per the plan's own ordering.
