# Experiment 0 Stats Report — Optimizer bake-off

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp0` |
| Experiment name | Optimizer bake-off |
| Date run | 2026-08-31 |
| Git commit hash | `7d96fa2` (code that produced this run's results; see §9 for a caveat about the raw CSV's own `git_commit` column) |
| SCALE-Sim version/commit | `scalesim==3.0.0` (PyPI), containerized — `docker/scalesim/Dockerfile` |
| Python / library versions | Python 3.11.9; see `requirements.txt` for exact pins (numpy 1.26.4, stable-baselines3 2.9.0, gymnasium 1.3.0, deap 1.4, optuna 4.9.0, pandas 2.3.3, matplotlib 3.10.7) |
| Random seeds used | `[0, 1, 2, 3, 4]` (brute-force ground truth is deterministic, logged with `seed=-1`) |
| Raw data file(s) | `results/raw/exp0.csv` |
| Processed data file(s) | `results/processed/exp0_summary.csv`, `results/processed/exp0_per_seed.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Choose which optimizer (REINFORCE, PPO, evolutionary/GA, Bayesian optimization) is used for all subsequent experiments, on stability + scalability grounds, not assumption.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 0, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** Optimizer type — A2C (see §3 deviations re: "REINFORCE"), PPO, evolutionary (GA), Bayesian optimization (TPE).
- **Dependent variable(s)/metrics:** Iterations-to-within-5%-of-brute-force-best, seed variance (std of final reward), mean wall-clock per iteration.
- **Controlled/fixed variables:** One small pilot workload (`config/workloads/exp0_pilot.yaml`) on the 16×16 array (`config/hardware/16x16.yaml`); the same real search problem (`common.optimizers.environment.AttentionMappingProblem`) and reward function for every candidate; the same 3-real-valued action encoding (`decode_raw_action`) for every candidate.

## 3. Method

**Procedure:**
1. Load the pilot workload and hardware config.
2. Run an exhaustive brute-force grid search (`common/optimizers/brute_force.py`) over all 3 dataflow modes × a 15-point SRAM-partition simplex grid (`sram_grid_divisions=3`) — 30 real simulations — to establish the ground-truth best reward.
3. For each of the 4 candidate optimizers, run 5 seeds (0–4), each a fresh 15-real-simulation budget (`n_iterations=15`), logging every individual evaluated candidate.
4. Aggregate per-seed "best reward so far" trajectories; compute, per optimizer: mean/std iterations to reach within 5% of the brute-force best, mean final best reward ± std, mean final utilization ± std, mean wall-clock per evaluation, and mean distance from the brute-force best.

**Exact command used to run this experiment:**
```bash
python experiments/exp0_optimizer_bakeoff/run_exp0.py \
  --workload config/workloads/exp0_pilot.yaml --hardware config/hardware/16x16.yaml \
  --seeds 0 1 2 3 4 --n-iterations 15 --sram-grid-divisions 3
python experiments/exp0_optimizer_bakeoff/analyze_exp0.py
```

**Deviations from the plan, and why:**

- **"REINFORCE" → A2C.** Verified 2026-08-31: `stable-baselines3==2.9.0` does not ship a plain REINFORCE implementation — its algorithms are A2C, DDPG, DQN, HER, PPO, SAC, TD3. A2C (Advantage Actor-Critic, a REINFORCE-with-baseline variant) is the closest established, citable, SB3-native policy-gradient method and stands in for it (`common/optimizers/a2c.py`). If the thesis names "REINFORCE" specifically, this needs correcting to "A2C" or a hand-rolled REINFORCE loop substituted.
- **"Tiling" search space = SRAM partition, not an explicit tile-size parameter.** SCALE-Sim's config exposes exactly two run-to-run controllable levers (verified by reading `scale_config.py`): dataflow mode and the ifmap/filter/ofmap SRAM buffer split. There is no separate explicit tile-size knob. Every optimizer here therefore searches (dataflow, SRAM partition) — a real constraint of the tool being used, applying to every experiment downstream, not just this one.
- **No RS (row-stationary) dataflow.** `scale_config.py`'s `valid_df_list = ['os', 'ws', 'is']` — verified by reading source and confirmed with real runs. `CODING_AGENT_PLAN.md` §1's module comment for `static_heuristics.py` lists "RS-only"; that does not match what the actual simulator supports. `common.simulator_interface.SUPPORTED_DATAFLOW_MODES = ("WS", "OS", "IS")` is the source of truth from here on.
- **Pilot workload scale reduced from the plan's example.** The plan gives "e.g. seq_len=512, 8 heads" as the pilot size. A real, timed containerized SCALE-Sim call at seq_len=512, hidden_dim=512 on the 16×16 array took **~178 seconds**; even seq_len=64 at hidden_dim=512 took ~16s (hidden_dim drives the QKV-projection GEMM's cost roughly quadratically, dominating even more than seq_len at this scale). A full bake-off (330 real simulator calls) at either scale would take multiple hours to tens of hours. `config/workloads/exp0_pilot.yaml` uses **seq_len=64, hidden_dim=128, num_heads=8** (num_heads kept at the plan's example value), verified at ~2.0s/call — the full run completed in the session. Re-running at the plan's literal seq_len=512/hidden_dim=512 scale is flagged as follow-up work once a larger wall-clock/compute budget is available (e.g. an overnight background run).
- **Iteration budget (15) and evolutionary population size (5) are small** relative to what any of these methods would normally be given, specifically to make a same-session real run of all 4 candidates × 5 seeds feasible. This is expected to limit how much any candidate "learns" within budget (see §8/§9) — the bake-off's job is to compare candidates fairly under a shared, small budget, not to find each one's best possible asymptotic performance.

## 4. Raw Results Summary

n = 5 seeds per optimizer. All values from `results/processed/exp0_summary.csv` (mean/std across seeds).

| Optimizer | n (seeds) | Converged seeds (≤5% of best) | Iters-to-converge (mean ± std) | Final best reward (mean ± std) | Final utilization % (mean ± std) | Mean wall-clock/eval (s) | Distance from brute-force best |
|---|---|---|---|---|---|---|---|
| bayesian_opt | 5 | 5/5 | 0.60 ± 1.34 | 67.70 ± 0.83 | 67.71 ± 0.83 | 1.38 | **1.06** |
| a2c | 5 | 5/5 | 5.20 ± 4.09 | 66.68 ± 0.61 | 66.68 ± 0.61 | 1.59 | 2.08 |
| ppo | 5 | 5/5 | 5.20 ± 4.09 | 66.68 ± 0.61 | 66.68 ± 0.61 | 1.55 | 2.09 |
| evolutionary | 5 | 4/5 | 2.75 ± 1.71 (converged seeds only) | 65.06 ± 2.56 | 65.06 ± 2.56 | 1.38 | 3.70 |

Brute-force ground truth: **68.7629** (best reward), achieved at dataflow=OS with an SRAM partition of roughly (0.33, 0.33, 0.33) — `results/raw/exp0.csv`, `optimizer=="brute_force"`.

## 5. Statistical Analysis

- **Comparisons made:** each optimizer's mean final best reward vs. the brute-force ground truth (distance, above); pairwise comparison of the four optimizers' final-reward distributions.
- **Test used:** none run — n=5 seeds per optimizer is underpowered for a meaningful parametric or non-parametric significance test (a Wilcoxon signed-rank test needs ≥6 pairs to reach even p<0.05 in principle), and CODING_AGENT_PLAN.md §0 ground rule 3 sets 5 seeds as the *minimum*, not a target. **[LIMITATION]** — this is flagged, not silently skipped: the summary table's std columns are the honest evidence available at this seed count, and a formal test is deferred to Experiment 3 (the plan's own core statistical-evidence experiment) where the workload/condition count is much larger.
- **Result:** N/A (no test run — see above).
- **Multiple-comparisons correction:** N/A (no test run).

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| — | — | — | — |

Zero `status == "FAILED"` rows in `results/raw/exp0.csv` (350 total rows, all `OK`) — every candidate SRAM partition sampled across all four optimizers and the brute-force grid was a feasible SCALE-Sim configuration at this workload/hardware scale.

## 7. Figures

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| 1 | `results/figures/exp0_convergence_curves.png` (+ `.svg`) | Best-reward-so-far vs. iteration, one line per optimizer, mean ± 95% CI band across 5 seeds. `bayesian_opt` (black) starts and stays near the ground truth; `a2c`/`ppo` (near-identical, see §8) climb steadily from a lower start; `evolutionary` (yellow) has visibly higher variance and a shorter mean trajectory (some seeds evaluate fewer than 15 candidates — see §9). | `results/raw/exp0.csv`, filtered to `optimizer != "brute_force"`, `status == "OK"`, grouped by (optimizer, seed), plotted as running-max reward vs. iteration |

## 8. Interpretation

- **[RESULT]** Across 5 seeds, `bayesian_opt` (optuna TPE) had the smallest mean distance from the brute-force ground-truth best reward (1.06), the fastest mean convergence to within 5% of that ground truth (0.60 iterations), the lowest final-reward standard deviation (0.83), and 5/5 seeds converging.
- **[RESULT]** `a2c` and `ppo` produced near-identical final-reward distributions (66.68 ± 0.61 for both; per-seed values match to 2–3 decimal places for 3 of 5 seeds — see `results/processed/exp0_per_seed.csv`).
- **[RESULT]** `evolutionary` had the worst mean final reward (65.06), the highest variance (std 2.56), and the only non-converged seed (seed 4, final reward 60.58, well below the other three candidates' worst seeds).
- **[INTERPRETATION]** `a2c` and `ppo`'s near-identical trajectories are consistent with both being dominated by their (similarly-initialized, similarly small-batch-updated) policy network's near-random early exploration under this experiment's very small iteration budget (n_steps=5, 2–3 rollouts total per seed) — 15 real evaluations is not enough budget for either algorithm's policy-gradient updates to diverge meaningfully from each other's behavior, not evidence that the two methods are equivalent in general.
- **[INTERPRETATION]** `bayesian_opt`'s advantage here plausibly reflects that a TPE sampler's default behavior already resembles quasi-random/structured exploration from the first few trials, which happens to suit a 15-evaluation budget on a 3-real-valued search space well — this is a property of the *budget regime tested*, not necessarily a general claim that Bayesian optimization dominates policy-gradient/evolutionary methods on this problem class at larger budgets.
- **[INTERPRETATION]** `evolutionary`'s worse, higher-variance result is plausibly explained by its small population (5) and the fact that a substantial fraction of its "iteration budget" is spent on individuals that survive selection unchanged and are correctly *not* re-evaluated (see §9) — meaning it effectively receives less real search signal than the 15-iteration budget suggests, an artifact of population-based search under a small shared budget, not necessarily evidence it is a weaker method in general.
- **[LIMITATION]** This bake-off used a 5–15x smaller pilot workload than the plan's own example (see §3), a 15-evaluation budget, and no significance test (see §5). The selection below is a real, data-backed decision at this scale, but should be treated as provisional pending a re-run at larger scale/budget if wall-clock/compute budget allows — flagged as follow-up work, not hidden.

Comparison against literature-reported figures (FLAT/COSA/ReDas/Gemmini) is not applicable to this experiment — Exp 0 has no baseline-vs-optimizer comparison; that begins at Experiment 3.

## 9. Limitations & Reasons for Discrepancy

- **Small iteration budget and workload scale** (§3, §8) — this experiment answers "which of these four search strategies works best under a small, shared, same-session-feasible budget on a small workload," not "which works best at the scale the thesis's main results (Experiment 3+) will actually run at." Re-running Experiment 0 at the plan's literal seq_len=512/hidden_dim=512 scale, with a larger iteration budget, is real follow-up work, not something this report claims to have already covered.
- **`evolutionary`'s real evaluation count varies per seed** (7, 8, 8, 11, 12 out of a 15-iteration nominal budget — see `results/processed/exp0_per_seed.csv`'s `n_evals` column) because DEAP correctly skips re-evaluating individuals whose genome is unchanged by crossover/mutation in a given generation (their existing fitness is still valid — re-evaluating would be a wasted, redundant real simulator call). This is documented in `common/optimizers/evolutionary.py`'s docstring as expected GA behavior, not a bug, but it does mean `evolutionary` received *fewer real evaluations* than the other three candidates on average (9.2 vs. 15) — part of, not separate from, why it under-performed here.
- **The raw CSV's own `git_commit` column is stale for this run.** `common.logging_utils.current_git_commit()` was called against a working tree that still had this experiment's own code (environment.py, a2c.py, ppo.py, evolutionary.py, bayesian_opt.py, brute_force.py, run_exp0.py) uncommitted — so every row in `results/raw/exp0.csv` logs the *prior* commit (`b44c137`), not the commit that actually produced these results (`7d96fa2`, this report's §0). This is a real reproducibility gap in the tooling: `current_git_commit()` should ideally warn (or refuse) on a dirty working tree. Not fixed in this run to avoid re-running the whole bake-off a third time over a metadata-only issue — flagged here as a concrete follow-up (`common/logging_utils.py`), and the correct commit for reproducing this exact run is this report's §0 hash, not the CSV's own column.
- **No significance test** (§5) — a real, disclosed gap, not a hidden one.
- **`dram_bandwidth_bytes_per_cycle` is not simulated** (see `common/simulator_interface.py`'s `write_scale_sim_inputs` docstring) — SCALE-Sim's CALC (auto-estimate) bandwidth mode was used for every run in this experiment instead. This does not affect the optimizer comparison (identical for every candidate) but does mean the DRAM-traffic term in every candidate's reward reflects SCALE-Sim's auto-estimated bandwidth, not the hardware config's stated value.

## 10. Reproduction

```bash
cd "MSc Thesis"   # project root
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/   # if not already built
python experiments/exp0_optimizer_bakeoff/run_exp0.py \
  --workload config/workloads/exp0_pilot.yaml --hardware config/hardware/16x16.yaml \
  --seeds 0 1 2 3 4 --n-iterations 15 --sram-grid-divisions 3
python experiments/exp0_optimizer_bakeoff/analyze_exp0.py
```

Note: `bayesian_opt` and the brute-force ground truth are deterministic given a seed / grid; `a2c`/`ppo` (stable-baselines3) and `evolutionary` (deap) may not reproduce bit-identically across machines/library-patch-versions even with the same seed, per those libraries' own reproducibility guarantees (or lack thereof) — the aggregate conclusions (§8) are the claim being made, not exact per-row numeric reproduction.

**Acceptance criteria for this experiment (per the plan):** A single optimizer is selected with a documented, data-backed reason (§8 above); that choice is now recorded in `config/optimizers/selected.yaml` and referenced (not re-decided) by Experiments 2–6.
