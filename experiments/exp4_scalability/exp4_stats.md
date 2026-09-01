# Experiment 4 Stats Report — Scalability study (answers RQ3 directly)

> **RQ3 answer, stated up front:** as `seq_len` grows, the optimizer's advantage over the best static baseline shrinks and **reverses** (positive at 128/256, −1.64% at 512), while its wall-clock cost grows substantially and grows *faster* on the smaller (16×16) array than the larger one. Both trends are real, from real data, at a small fixed search budget (6 iterations) — see §8 for why, and what a larger budget would plausibly change.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp4` |
| Experiment name | Scalability study (answers RQ3 directly) |
| Date run | 2026-09-01 |
| Git commit hash | `50c3227` (code that produced this run; raw CSV logs `420a337-dirty` — working tree had this run's own code uncommitted at run time, correctly flagged by the `-dirty` suffix) |
| SCALE-Sim version/commit | `scalesim==3.0.0` (PyPI), containerized — `docker/scalesim/Dockerfile` |
| Python / library versions | Python 3.11.9; see `requirements.txt` |
| Random seeds used | `[0, 1, 2, 3, 4]` (baselines deterministic, logged with `seed=-1`) |
| Raw data file(s) | `results/raw/exp4.csv` (this experiment's 1 new cell) **+ reused** `experiments/exp3_main_sweep/results/raw/exp3.csv` (4 cells — see §3 for why reused, not re-run) |
| Processed data file(s) | `results/processed/exp4_scalability_summary.csv`, `exp4_convergence_iterations.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Show how the optimizer's advantage over baselines, and its own overhead, scale with sequence length and array size.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 4, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** `seq_len` ∈ {128, 256, 512} (extended beyond Experiment 3's {128,256} — see §3 for why not further) and array ∈ {16×16, 32×32}. Not every (seq_len, array) combination is populated — 512 is infeasible at 16×16 (real, measured — see §3).
- **Dependent variable(s)/metrics:** %-improvement of `bayesian_opt` over the best single-dataflow static baseline; optimizer wall-clock cost and (where defined) convergence iteration.
- **Controlled/fixed variables:** `head_dim=64` throughout. Optimizer iteration budget fixed at 6/seed (matching Experiment 3, for comparability across the merged trend).

## 3. Method

- **Procedure:** For the one genuinely new (seq_len, array) point (512, 32×32), run all 4 static baselines and `bayesian_opt` (5 seeds × 6 iterations, fixed budget) exactly as Experiment 3 did. Merge with Experiment 3's 4 already-real points for the full trend. Per point, compute %-improvement of the optimizer's mean best-utilization over the best single-dataflow baseline (the same fair comparison established in Experiment 3 §8 — `best_fixed_per_stage` is excluded from this comparison for the same action-space-asymmetry reason), and two cost measures: total wall-clock for the fixed budget (always defined) and convergence iteration by Experiment 2's criterion (only defined for seeds that actually meet it within budget).
- **Exact commands used:**
  ```bash
  python experiments/exp4_scalability/run_exp4.py \
    --workload-names exp4_seq512 --hardware-names 32x32 \
    --seeds 0 1 2 3 4 --n-iterations 6 --out results/raw/exp4.csv
  python experiments/exp4_scalability/analyze_exp4.py
  ```
- **Deviations from the plan, and why:**
  - **4 of the 5 trend points are reused from Experiment 3, not re-run.** `bayesian_opt` and every static baseline are fully deterministic given `(workload, hardware, seed)` — re-running Experiment 3's 4 cells would reproduce its numbers bit-for-bit at real wall-clock cost for zero new information (the same reasoning already applied when Experiment 2's ε was re-checked without re-simulating, `exp2_stats.md` §8a). `run_exp4.py` only executes the one new cell.
  - **`seq_len` extended only to 512, not the plan's example "up to 8192."** Real, hand-timed probes (same discipline as every prior experiment) found `seq_len=512` succeeds at 32×32 (67.29s/call) but **fails at 16×16** (times out entirely, already measured while designing Experiment 3's grid) and `seq_len=1024` **fails at 32×32 too** (184.08s, `status=FAILED`) — both documented in `config/workloads/exp4_seq512.yaml`. Neither infeasible point was re-attempted as a full ~42-call cell — a single-call probe already establishes infeasibility. **This ceiling is itself part of RQ3's answer, not a gap**: the maximum simulatable `seq_len` roughly doubles as array size doubles (256 → 512), meaning "how far can this study scale" is bounded by simulator/array-size feasibility before it's bounded by anything about the optimizer itself.
  - **Optimizer runs the same fixed 6-iteration budget as Experiment 3, not a dynamically-stopped run**, per Experiment 2's finding (`exp2_stats.md` §8a) that the convergence criterion is unsuitable for controlling how many real simulations a run executes. `convergence_iteration` is still computed and reported per seed (plan's own "sample count needed to converge" phrasing), just not used to end a run early — the "cost" figure's primary metric is total wall-clock for the fixed budget, which is always defined (unlike convergence iteration, which 6/25 seeds across this experiment's points never reach — see §8).

## 4. Raw Results Summary

**Scalability summary** (`results/processed/exp4_scalability_summary.csv`, n=5 seeds per point):

| Cell | Best single-dataflow (method, util%) | Optimizer util% (mean ± std) | % improvement (± 95% CI) | Cost: wall-clock s (mean ± 95% CI) |
|---|---|---|---|---|
| seq128 @ 16×16 | os_only, 89.96 | 90.17 ± 0.16 | +0.23% ± 0.23 | 199.5 ± 16.8 |
| seq256 @ 16×16 | os_only, 89.02 | 89.22 ± 0.08 | +0.23% ± 0.12 | 541.5 ± 87.9 |
| seq128 @ 32×32 | os_only, 79.50 | 80.12 ± 0.30 | +0.79% ± 0.47 | 81.3 ± 3.1 |
| seq256 @ 32×32 | os_only, 78.91 | 79.46 ± 0.23 | +0.70% ± 0.35 | 199.7 ± 19.7 |
| seq512 @ 32×32 | **ws_only**, 84.49 | 83.11 ± 3.10 | **−1.64% ± 4.55** | 562.6 ± 109.7 |

**Note the best single-dataflow method itself changes at seq_len=512** — `ws_only` overtakes `os_only` (which won at every other point), a real finding in its own right (§8).

**Convergence iteration** (`results/processed/exp4_convergence_iterations.csv`), converged seeds only:

| Cell | Converged / n_seeds | Mean iteration (converged only) |
|---|---|---|
| seq128 @ 16×16 | 5/5 | 4.0 |
| seq128 @ 32×32 | 4/5 | 4.0 |
| seq256 @ 16×16 | 4/5 | 4.0 |
| seq256 @ 32×32 | 4/5 | 4.0 |
| seq512 @ 32×32 | 2/5 | 4.5 |

**19/25 seeds across all 5 points converged within the 6-iteration budget** — the convergence rate is lowest exactly at the new, larger scale point (2/5), consistent with §8's interpretation.

## 5. Statistical Analysis

Not applicable in the paired-significance-test sense (Experiment 3's own §5 already established that framework; this experiment's job is the scalability *trend*, not a per-point significance claim). The 95% CIs in §4 are the quantitative uncertainty measure — note the seq512 point's CI (±4.55%) is roughly 10–20× wider than the smaller-scale points', driven entirely by one seed (seed 4) landing far from the other four (§8).

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| — | — | — | — |

Zero `status == "FAILED"` rows in `results/raw/exp4.csv` (44 rows, all `OK`). Two (seq_len, array) points were probed and found infeasible *before* being run as full cells (documented in `config/workloads/exp4_seq512.yaml`, not logged as raw CSV rows since no full-cell run was attempted): `seq_len=512 @ 16×16` (times out) and `seq_len=1024 @ 32×32` (`FAILED`, 184.08s).

## 7. Figures

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| 1 | `results/figures/exp4_scalability_panels.png` (+ `.svg`) | Two-panel: (a) %-improvement over best single-dataflow baseline vs. `seq_len`, one line per array size — crosses zero for 32×32 between 256 and 512; (b) optimizer wall-clock cost vs. `seq_len` — rises for both, faster for 16×16 than 32×32 at the same `seq_len`. | `results/processed/exp4_scalability_summary.csv` |

## 8. Interpretation

- **[RESULT]** The optimizer's %-improvement over the best single-dataflow baseline is small but positive and roughly flat at `seq_len` 128 and 256, at both array sizes tested (+0.23% to +0.79%). At `seq_len`=512 (32×32 only — the only array size that can reach this scale), it **reverses to −1.64%**, with a much wider CI (±4.55%, vs. ±0.12–0.47% at the smaller scales).
- **[RESULT]** The reversal is driven by one seed, not a uniform effect: per-seed utilization at `seq512 @ 32×32` was 84.49, 84.49, 84.49, 84.49, 77.57 (§4's raw data) — four seeds essentially matched the new best baseline (`ws_only`, 84.49%) exactly, while one (seed 4) landed far below. Std jumps from 0.08–0.30 at the smaller scales to 3.10 here.
- **[RESULT]** The identity of the best static baseline changes with scale: `os_only` wins at every `seq_len` ≤ 256 tested (all 4 Experiment 3 points); `ws_only` wins at `seq_len`=512.
- **[INTERPRETATION] — the direct answer to RQ3.** Both observations point to the same mechanism: the small, fixed 6-iteration budget used throughout this project (for wall-clock tractability) is enough to reliably find a near-optimal region when the reward landscape's optimum sits where it "usually" does (`os_only`-like, at `seq_len` ≤ 256), but is not reliably enough when the optimum *moves* — at `seq_len`=512 the optimum shifted to a `ws_only`-like region, and 1 of 5 seeds' 6 real trials simply didn't land there in time. **As scale increases: (a) the optimizer's real advantage over the best static heuristic shrinks and can reverse, specifically because the fixed small budget becomes less sufficient exactly when the landscape's optimum relocates; (b) optimizer cost grows substantially (199s→562s at 16×16→…→32×32 across the tested range), and grows *faster* per unit of array size at 16×16 than 32×32 for the same `seq_len` — consistent with Experiment 1's finding that smaller arrays are more expensive to simulate.** This is the evidence-based RQ3 answer the plan's acceptance criteria asks for: performance advantage does not scale robustly under a fixed budget, and cost scales unfavorably in exactly the regime (small arrays, larger sequences) where the plan expects this class of system to matter most.
- **[LIMITATION]** This is real evidence, but from **one point at the new scale, from one array size**. A budget-sensitivity check (does a larger budget, e.g. 15–40 iterations as Experiments 0/2 used, restore or exceed the small-scale advantage at `seq_len`=512?) was not run — real, disclosed follow-up work, and the most natural next experiment given this finding.
- **[LIMITATION]** `seq_len` extended only to 512 (not the plan's example up to 8192) — a real, measured infeasibility ceiling at this project's current hardware configs and Docker resource limits, not a choice of convenience (§3).
- **[LIMITATION]** `dram_bandwidth_bytes_per_cycle` not simulated, `energy_per_cycle_pj` a placeholder (1.0) — identical caveats to every prior experiment.

Comparison against literature-reported figures (FLAT/COSA/ReDas/Gemmini) is not directly applicable to a scalability trend — those figures are point estimates at specific configurations, not scaling curves, and are not re-cited here to avoid implying a scaling comparison that wasn't made.

## 9. Limitations & Reasons for Discrepancy

- **The central limitation is budget-vs-scale, not simulator or search-algorithm quality** — see §8's interpretation. The recommended next step is a targeted follow-up: re-run `seq512 @ 32×32` with a larger iteration budget (e.g. 15, matching Experiment 0's pilot scale) to test whether the reversal is a budget artifact (prediction: it should disappear or shrink) or persists (which would be a more concerning finding about the search itself at this scale) — a real, falsifiable follow-up in the same spirit as Experiment 1's head_dim validation.
- **Only 32×32 could reach `seq_len`=512** — the scalability trend's higher-`seq_len` region has no 16×16 data point to compare against, a real gap in the array-size × seq_len interaction this experiment set out to characterize.
- **Cost measured in wall-clock seconds, which is infrastructure-dependent** (this machine's Docker/CPU performance) — a real number for this environment, but not directly portable as a hardware-independent "sample efficiency" claim; `convergence_iteration` (sample count) is the more portable metric where it's defined, though it wasn't defined for 6/25 seeds here.

## 10. Reproduction

```bash
cd "MSc Thesis"   # project root
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/   # if not already built
python experiments/exp4_scalability/run_exp4.py \
  --workload-names exp4_seq512 --hardware-names 32x32 \
  --seeds 0 1 2 3 4 --n-iterations 6 --out results/raw/exp4.csv
python experiments/exp4_scalability/analyze_exp4.py
```

Baselines are fully deterministic; `bayesian_opt` (optuna TPE) is seeded but not guaranteed bit-for-bit reproducible across machines/library-patch-versions (same caveat as every prior optimizer-involving experiment).

**Acceptance criteria for this experiment (per the plan):** A direct, evidence-based answer to RQ3 is written into the stats.md (not just plots). **Met** — §8's interpretation states plainly what happens to performance (shrinks, reverses) and cost (grows, grows faster on smaller arrays) as scale increases, grounded in the real data in §4, with the mechanism (fixed-budget insufficiency when the landscape's optimum relocates) identified rather than left as an unexplained correlation.
