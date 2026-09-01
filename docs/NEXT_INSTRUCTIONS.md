# Next Instructions for the Coding Agent

Context for every item below lives in `RESULTS_TRACKER.md` (read that first,
especially Experiments 3 and 5's summaries) — this file lists only what to do
next, not why. Do not add narrative/history here; when a step is done, delete
it from this file rather than marking it complete in place.

---

## 3. Re-run Experiment 3 (main sweep) on the new action space

`config/optimizers/selected.yaml` now points to `a2c`, not `bayesian_opt` (Exp0
was re-examined on the new action space — see `RESULTS_TRACKER.md`'s "Selected
optimizer" section — use whatever it currently says, don't hardcode `a2c` either
in case this changes again). Same grid as before (seq_len∈{128,256} ×
array∈{16×16,32×32}, real, measured infeasibility of 512/1024 at 16×16 stands).
Log to a new CSV (don't overwrite the existing `results/raw/exp3.csv`), re-run
the significance tests, and check whether the selected optimizer now closes the
gap with `best_fixed_per_stage`. Update `exp3_stats.md` with both the old and
new action-space results, clearly labelled.

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
