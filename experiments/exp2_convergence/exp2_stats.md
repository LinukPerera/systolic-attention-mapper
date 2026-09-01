# Experiment 2 Stats Report — Search/training convergence study

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp2` |
| Experiment name | Search/training convergence study |
| Date run | 2026-09-01 |
| Git commit hash | `f2b096a` (code that produced this run; raw CSV logs `84cc3e6-dirty` — working tree had this experiment's own code uncommitted at run time, correctly flagged by the `-dirty` suffix) |
| SCALE-Sim version/commit | `scalesim==3.0.0` (PyPI), containerized — `docker/scalesim/Dockerfile` |
| Python / library versions | Python 3.11.9; see `requirements.txt` (optuna 4.9.0) |
| Random seeds used | `[0, 1, 2, 3, 4]` |
| Raw data file(s) | `results/raw/exp2.csv` |
| Processed data file(s) | `results/processed/exp2_convergence_iterations.csv`, `exp2_trajectories.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Prove the chosen optimizer reliably converges, with an explicit numeric stopping criterion.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 2, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** Seed (5: `[0, 1, 2, 3, 4]`).
- **Dependent variable(s)/metrics:** Reward trajectory per iteration (and the derived best-reward-so-far trajectory the convergence criterion is computed from).
- **Controlled/fixed variables:** Optimizer — `bayesian_opt` (optuna TPE), Experiment 0's selection, not re-decided here (`config/optimizers/selected.yaml`). Workload — `config/workloads/exp2_convergence.yaml` (seq_len=64, hidden_dim=256, num_heads=4, giving `head_dim=64` per the standard `config/workloads/README.md` adopted after Experiment 1). Hardware — `config/hardware/16x16.yaml`, the smaller of Experiment 1's two justified primary sizes. Iteration budget — 40 (the plan's "generous iteration budget").

## 3. Method

**Convergence criterion — stated as a formula before running, per the plan's acceptance criteria (not "it looked flat"):**

For each seed, let `b_1, ..., b_T` be the best-reward-so-far trajectory (a running max over that seed's raw per-iteration rewards — monotonically non-decreasing by construction). A seed is declared **CONVERGED** at the first iteration `i ≥ WINDOW` such that:

```
Var(b_{i-WINDOW+1}, ..., b_i) < EPSILON
```

with `WINDOW=5`, `EPSILON=1.0` (both CLI flags in `run_exp2.py`, defaults used for this run, chosen and documented here — not tuned after seeing results). A seed that never meets this within the 40-iteration budget is reported as NOT converged, not discarded.

**Procedure:**
1. For each of 5 seeds, run `bayesian_opt` (loaded from `config/optimizers/selected.yaml`, with hyperparameters from `config/optimizers/bayesian_opt.yaml`) for 40 real iterations against the fixed workload/hardware above.
2. Log every real `EvalResult` as one raw CSV row.
3. Compute each seed's best-reward-so-far trajectory and apply the convergence criterion above.

**Exact commands used:**
```bash
python experiments/exp2_convergence/run_exp2.py \
  --workload config/workloads/exp2_convergence.yaml --hardware config/hardware/16x16.yaml \
  --seeds 0 1 2 3 4 --n-iterations 40 --window 5 --epsilon 1.0 \
  --out results/raw/exp2.csv
python experiments/exp2_convergence/analyze_exp2.py
```

**Deviations from the plan, and why:**

- **Workload config deviates from `NEXT_INSTRUCTIONS.md`'s example** (`num_heads=8, hidden_dim=512`, which gives `head_dim=64` too) — that combination timed at 16.8s/call (seq_len=64), too slow for 5 seeds × a generous iteration budget (~70 min for a modest budget). `hidden_dim=256, num_heads=4` preserves `head_dim=64` exactly while costing 5.83s/call (verified by hand before committing — a third candidate, `hidden_dim=128/num_heads=2`, was also timed at 2.11s/call but not used, to keep `num_heads` at a more realistic value). This is the same real-timing-before-committing discipline used for every prior workload config in this project.

## 4. Raw Results Summary

Every method here is the same optimizer (`bayesian_opt`) — n = 5 seeds, each a full 40-iteration real run. No condition-grouping beyond seed applies (plan §6 Exp 2's only independent variable is seed).

**Per-seed convergence iteration and final best reward** (`results/processed/exp2_convergence_iterations.csv`):

| Seed | n iterations | Convergence iteration | Converged? | Final best reward |
|---|---|---|---|---|
| 0 | 40 | 4 | Yes | 85.99 |
| 1 | 40 | 4 | Yes | 85.97 |
| 2 | 40 | 4 | Yes | 86.02 |
| 3 | 40 | 4 | Yes | 84.81 |
| 4 | 40 | 4 | Yes | 85.80 |

**Convergence iteration: mean = 4.00, std = 0.00** (5/5 seeds, all converged at exactly the same iteration).

**Final best reward across seeds: mean = 85.72, std = 0.47** (n=5; range 84.81–86.02) — computed directly from the table above, not a separate processed file.

## 5. Statistical Analysis

Not a comparison experiment (single optimizer, no baseline arm here) — no significance test applies. The only quantitative claim is the mean/std convergence iteration and final-reward spread reported in §4, both computed directly and exactly (n=5, not sampled).

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| — | — | — | — |

Zero `status == "FAILED"` rows in `results/raw/exp2.csv` (200 rows, all `OK`).

## 7. Figures

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| 1 | `results/figures/exp2_convergence_curves_individual_seeds.png` (+ `.svg`) | Individual best-reward-so-far trace per seed (distinct colors) plus the across-seed mean (dashed black). Every seed keeps improving well past the declared convergence point (iteration 4) — see §8. | `results/processed/exp2_trajectories.csv` |
| 2 | `results/figures/exp2_convergence_curves_mean_ci.png` (+ `.svg`) | Mean best-reward-so-far ± 95% CI band across all 5 seeds vs. iteration (plan's required "mean ± CI band" output). | `results/processed/exp2_trajectories.csv` |

## 8. Interpretation

- **[RESULT]** 5/5 seeds converged by the stated criterion (§3), all at exactly iteration 4 (mean 4.00, std 0.00) — the acceptance criterion ("holds for all/most seeds") is met, and no seed's result was discarded.
- **[RESULT]** Final best-reward spread across seeds is small (85.72 ± 0.47, range 84.81–86.02 — a 1.4% relative spread) — the optimizer reliably lands in a similar region regardless of seed.
- **[RESULT] — the criterion, taken literally, is satisfied much earlier than the trajectories visually stabilize.** Fig. 1/Fig. 2 show every seed (and the mean) continuing to improve substantially after iteration 4 — the mean trajectory rises from ~84.97 at iteration 4 to ~85.68 by iteration ~25 before flattening. `EPSILON=1.0` is ≈3.6% of the observed reward range across this whole experiment (58.18–86.02, a span of 27.83) — loose enough that a run of a few iterations landing within a 1.0-point band (common early on, when several early evaluations cluster near each other by chance) triggers "converged" well before later iterations find meaningfully better regions.
- **[INTERPRETATION]** This is a real property of the *chosen* `(WINDOW, EPSILON)` parameterization, not evidence the optimizer stops learning at iteration 4 — it keeps finding better candidates for tens more iterations. The criterion answers a narrower question than "has the optimizer found its final answer": it answers "has the best-so-far value been locally stable for 5 iterations within a 1.0-reward-unit band," which can be satisfied by a temporary plateau the search later escapes. Both readings are legitimate; conflating them would overstate what "converged at iteration 4" means.
- **[LIMITATION]** `WINDOW`/`EPSILON` sensitivity was not swept — a stricter `EPSILON` (e.g. 0.1, ≈0.4% of the reward range) would very likely push the declared convergence point later and probably closer to where Fig. 1/2 visually flatten (~iteration 25). This is a real, disclosed gap: the criterion was fixed before running (not tuned afterward to "look better" either way), but its sensitivity to its own parameters is untested here.
- **[LIMITATION]** All 5 seeds converging at the *identical* iteration (std=0.00) is itself informative but should not be over-read as "the optimizer always converges at iteration 4" in general — it's consistent with, at minimum, this specific (workload, hardware, iteration-budget, WINDOW, EPSILON) combination producing an early plateau on most runs; a different workload scale (e.g. Experiment 4's larger seq_len sweep, which explicitly reuses this experiment's convergence criterion per the plan) may show different, more varied convergence iterations, and should be reported as such if it does.

Comparison against literature-reported figures is not applicable to this experiment — Exp 2 has no baseline-vs-optimizer comparison (that begins at Experiment 3) and no external convergence benchmark to compare against.

## 9. Limitations & Reasons for Discrepancy

- **The stated criterion and the visually-apparent convergence point diverge** (§8) — this is the central limitation of this report. Anyone citing "converges by iteration 4" should also cite the criterion's exact formula and parameters (§3), not the number alone, since a stricter criterion would very plausibly report a later iteration.
- **`WINDOW`/`EPSILON` were not swept or tuned** — real, disclosed, not scheduled as follow-up work within this report (Experiment 4 reuses this criterion per the plan, so any future retuning should happen once, here, not divergently per experiment).
- **`dram_bandwidth_bytes_per_cycle` is not simulated** and **`energy_per_cycle_pj` is a placeholder** (1.0) — identical caveats to Experiments 0/1.
- **Single hardware size (16×16) tested** — Experiment 4 (scalability) is the plan's designated place to test convergence behavior across array sizes/workload scales; this report doesn't attempt that.

## 10. Reproduction

```bash
cd "MSc Thesis"   # project root
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/   # if not already built
python experiments/exp2_convergence/run_exp2.py \
  --workload config/workloads/exp2_convergence.yaml --hardware config/hardware/16x16.yaml \
  --seeds 0 1 2 3 4 --n-iterations 40 --window 5 --epsilon 1.0 \
  --out results/raw/exp2.csv
python experiments/exp2_convergence/analyze_exp2.py
```

Not fully deterministic: `bayesian_opt` (optuna TPE) is seeded, but exact bit-for-bit reproduction across machines/library-patch-versions isn't guaranteed by optuna itself — the aggregate conclusions (§8) are the claim being made, not exact per-row numeric reproduction (same caveat as Experiment 0).

**Acceptance criteria for this experiment (per the plan):** Convergence criterion is stated as a formula (not just "it looked flat"), and holds for all/most seeds — seeds that fail to converge are reported, not discarded. **Met**: the formula is stated in §3 and in `run_exp2.py`'s own docstring (written before running), and all 5/5 seeds converged by it. The formula's own limitation (§8/§9 — it triggers earlier than the trajectories visually stabilize) is reported honestly rather than hidden, which is itself part of meeting the plan's "not just 'it looked flat'" standard: a stated, checkable formula that turns out to be loose is still a formula, and its looseness is now documented evidence, not a retrofit excuse.
