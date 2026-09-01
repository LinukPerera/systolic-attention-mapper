# Next Instructions for the Coding Agent — Head-Dim Fix + Resume Phase B

**Context (read first):** Experiment 1's array-size "knee" was traced to a real
root cause, not a bug: `qk_transpose` and `score_times_v`'s GEMM dimensions are
capped at `head_dim = hidden_dim/num_heads` regardless of `seq_len`
(`common/workload_gen.py`). `exp1_representative.yaml` used `head_dim=32`, well
below the `head_dim=64` typical of real Transformer configs, which understated
how much of a larger array those two stages could use. Full diagnosis:
`experiments/exp1_array_sensitivity/exp1_stats.md` §8, `RESULTS_TRACKER.md`'s
"Justified array sizes" section. **Do not re-litigate this diagnosis — execute
against it.**

Everything below assumes a Docker-capable shell (this doc was prepared without
one; nothing in it has been run).

---

## Step 1 — Run the validation workload (already prepared)

`config/workloads/exp1_head64_validation.yaml` exists (hidden_dim=512,
num_heads=8, head_dim=64; seq_len=256 unchanged from `exp1_representative.yaml`
for a clean, single-variable comparison). Run the same sweep pattern as the
representative re-run, into its own CSV (plan ground rule 6 — never overwrite):

```bash
python experiments/exp1_array_sensitivity/run_exp1.py \
    --workload config/workloads/exp1_head64_validation.yaml \
    --out results/raw/exp1_head64_validation.csv
```

Time one call first on 64x64 if wall-clock is a concern (this workload is larger
than `exp1_representative.yaml`, so per-call cost will be somewhat higher — verify,
don't assume).

## Step 2 — Check the prediction

Recompute the decline-rate-acceleration analysis (same method as `exp1_stats.md`
§4b/§8) on the new CSV. The stated, falsifiable prediction: the knee shifts from
~32×32 to **~64×64**.

- **If confirmed:** the head_dim causal explanation is validated with real evidence
  — this becomes the strongest, most citable finding in Experiment 1.
- **If not confirmed:** say so plainly in `exp1_stats.md` — revise §8's "root
  cause found" language to reflect that the mechanism is not fully explanatory,
  and reopen the array-size question rather than forcing a fit. Do not adjust the
  prediction after seeing the data to make it "work."

## Step 3 — Update `exp1_stats.md`

Add a `## 4c. Raw Results Summary — head_dim=64 validation run` section (same
format as §4a/§4b), then extend §8's Interpretation with the confirmed/disconfirmed
verdict from Step 2. Regenerate `plot_utilization_vs_array_size` to include all
three runs (or a dedicated comparison figure showing the knee-location shift) —
new figure files, don't overwrite the existing ones.

## Step 4 — Finalize the array-size justification

Once Step 2/3 land, rewrite `RESULTS_TRACKER.md`'s "Justified array sizes" section
to cite the mechanistic (head_dim) explanation as primary, with the decline-rate
numbers as supporting detail rather than the main argument. Update Exp 1's status
row/summary block to match, per the tracker's own "copy from stats.md, don't
rephrase stronger" rule.

## Step 5 — Move to Experiment 2, using the new standard

`config/workloads/README.md` now mandates `head_dim=64` fixed for every
Experiment 2+ workload (vary `seq_len`/`num_heads`, not `hidden_dim` directly).
Create Experiment 2's workload config under this rule (e.g. `num_heads=8,
hidden_dim=512` for head_dim=64, `seq_len` chosen for a tractable per-call cost —
verify by hand as every prior pilot workload did, don't assume).

Implement `experiments/exp2_convergence/run_exp2.py` per
`docs/CODING_AGENT_PLAN.md` §6 Experiment 2: the *selected* optimizer only
(`bayesian_opt`, from `config/optimizers/selected.yaml` — do not re-decide this),
≥5 seeds, on the fixed configuration from this step. Define and apply an explicit
numeric convergence criterion (e.g. moving-average reward variance < ε over N
iterations) — state the formula in `exp2_stats.md`, not just "it looked flat."

## Step 6 — Note for whoever designs Experiment 3's workload sweep

When Experiment 3's seq_len × array-size grid is defined, hold `head_dim=64`
fixed throughout (per the new standard) and treat its own results as the
authoritative check on whether 16×16/32×32 still hold as primary sizes at
Experiment 3's larger seq_len range — this was already flagged as a caveat in
`exp1_stats.md` and remains true regardless of Step 2's outcome.

---

**Acceptance criteria for this doc:** Steps 1–4 done means Experiment 1 is
finalized on a mechanistic, tested basis rather than a provisional one. Step 5
done means Experiment 2 is complete and `RESULTS_TRACKER.md` reflects it. Step 6
is a standing note, not a checklist item — revisit it when Experiment 3 starts.

---

## Step 7 — Experiment 3: Main sweep vs. baselines (added 2026-09-01, after Exp 0-2 landed)

Exp 0 (optimizer=`bayesian_opt`), Exp 1 (sizes=16×16/32×32, mechanistically
justified), and Exp 2 (convergence criterion defined, though loosely-tuned —
see the note at the end of this step) are all complete. `run_exp3.py` is still
a stub. This is the core evidence chapter — budget real care here.

### 7.1 — Cost check FIRST, before committing to a grid

Every workload used so far has needed its real per-call cost measured by hand
before committing to a sweep (`exp0_pilot.yaml`, `exp1_representative.yaml`,
`exp1_head64_validation.yaml` all documented this). Exp 3's workloads are
`head_dim=64` (the new standard) at multiple `seq_len` values — likely the most
expensive calls yet. Time ONE call at the largest candidate `seq_len` before
deciding the final grid:

```bash
python experiments/exp3_main_sweep/run_exp3.py --config config/workloads/exp3_seq512.yaml \
    --hardware-names 16x16 --dry-run-single-call   # or equivalent single-call probe
```

The plan's example grid is `seq_len ∈ {128, 512, 1024, 2048, 4096} × array ∈
{16×16, 32×32}`, 4 static baselines + the optimizer at ≥5 seeds per cell. At
Exp 0's ~178s/call (measured at seq_len=512/hidden_dim=512) that grid is
hundreds of calls per cell — likely infeasible in one sitting. **Scale down
deliberately and document it**, the same way every prior experiment did:
prefer fewer `seq_len` points (e.g. {128, 512, 1024}) and/or a smaller
optimizer eval budget (Exp 0 used 15 evals/seed) over silently shrinking
`head_dim` again — `head_dim=64` must stay fixed per the standard.

### 7.2 — Build the workload grid

Create `config/workloads/exp3_seq{N}.yaml` per chosen `seq_len`, all with
`head_dim=64` (`hidden_dim = 64 * num_heads`, matching `config/workloads/README.md`).

### 7.3 — Implement `run_exp3.py`

For each (workload, array) cell: run the 4 deterministic baselines (as Exp 1's
script does) AND the selected optimizer (`common.optimizers.bayesian_opt`, per
`config/optimizers/selected.yaml`) across ≥5 seeds, using Exp 2's convergence
criterion to decide when each optimizer run stops. Log every run via
`logging_utils.append_run_row`, FAILED runs included (Exp 1's head64/8×8 OOM
is the precedent — log it, don't crash the sweep; `best_fixed_per_stage_mapping`
was already fixed to do this in Exp 1, reuse that fix).

### 7.4 — Report both baselines, never blended

Per `CODING_AGENT_PLAN.md` §6: report every metric as absolute value AND
%-difference from the best in-harness static baseline, in paired tables. Add
literature-reported figures (FLAT ≈1.9×, COSA ≈94% utilization, ReDas ≈4.6×,
Gemmini WS/OS ≈3×) as a clearly separate column in `exp3_stats.md` — directional
context only, never averaged into your own numbers.

### 7.5 — Statistical significance is required here, not optional

Exp 0/1 didn't run significance tests (small-n, disclosed limitation). Exp 3 is
the thesis's central claim ("learned optimizer beats heuristics") — run a
paired test (t-test or Wilcoxon signed-rank) across seeds for each workload/array
cell's optimizer-vs-best-baseline comparison, report the statistic and p-value
in `exp3_stats.md` §5, and say plainly if any comparison is NOT significant
rather than omitting it.

### 7.6 — Acceptance criteria

Every reported "X% improvement" claim has a seed-level std/CI AND, where claimed
significant, a test result backing it. Representative cases (2-3 workloads)
promoted to the main-body figures; the rest of the grid to an appendix table,
per the body/appendix split every prior experiment plan has followed.

---

## Note: Experiment 2's convergence criterion is loosely tuned (not blocking, but worth deciding now)

`exp2_stats.md` found EPSILON=1.0 triggers "converged" at iteration 4 while
trajectories keep improving until ~iteration 25 — a real, disclosed looseness,
not an error. Per the plan, Experiment 4 reuses this same criterion. **Decide
now, once, rather than per-experiment:** either (a) accept it as-is (it's honestly
documented, and re-tuning after seeing the data risks looking like fitting the
criterion to a desired answer), or (b) re-run Exp 2 with a stricter EPSILON
chosen blind (before looking at the new results) and update `exp2_stats.md` +
`RESULTS_TRACKER.md` accordingly. Either choice is defensible; not deciding and
letting Experiment 4 inherit an unexamined criterion is the one bad option.
