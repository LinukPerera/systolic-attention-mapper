# Experiment 1 Stats Report — Array-size sensitivity study

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp1` |
| Experiment name | Array-size sensitivity study |
| Date run | 2026-08-31 |
| Git commit hash | `46f218c` (code that produced this run; the raw CSV's own `git_commit` column logs `75a7499-dirty` — the working tree had this experiment's own code uncommitted at run time, correctly flagged by the `-dirty` suffix added after Experiment 0 hit the same gap) |
| SCALE-Sim version/commit | `scalesim==3.0.0` (PyPI), containerized — `docker/scalesim/Dockerfile` |
| Python / library versions | Python 3.11.9; see `requirements.txt` |
| Random seeds used | N/A — every method here is a deterministic static heuristic (plan ground rule 3); logged with `seed=-1` |
| Raw data file(s) | `results/raw/exp1.csv` |
| Processed data file(s) | `results/processed/exp1_summary.csv`, `results/processed/exp1_pct_change_per_doubling.csv` |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Justify the primary working array sizes (16x16, 32x32) the way the sample thesis justifies its mesh choice.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 1, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** Array size — 8×8, 16×16, 32×32, 64×64, 128×128.
- **Dependent variable(s)/metrics:** PE utilization %, exec cycles, DRAM traffic, EDP, per method; ranking stability of the four methods across array sizes.
- **Controlled/fixed variables:** One workload (`config/workloads/exp0_pilot.yaml` — see §3 for why), all five hardware configs' SRAM/bandwidth/energy fields held identical (only array size varies), the four methods compared: `ws_only`, `os_only`, `is_only`, `best_fixed_per_stage` (all from `common/baselines/static_heuristics.py`).

## 3. Method

**Procedure:**
1. For each of the 5 array sizes, run `ws_only_mapping`/`os_only_mapping`/`is_only_mapping` — one whole-workload simulation each with that fixed dataflow.
2. Also run `best_fixed_per_stage_mapping` — 9 simulations (3 GEMM stages × 3 dataflows) to find each stage's empirically-best dataflow, then `combine_simulation_results` to aggregate the 3 winning per-stage results into one whole-workload figure comparable to the single-dataflow baselines (SCALE-Sim's `Dataflow` setting applies per-run, not per-layer, so there's no single simulation that measures a per-stage-mixed dataflow directly).
3. Compute the %-change in utilization between consecutive array-size doublings (8→16, 16→32, 32→64, 64→128), per method, and apply the plan's own suggested sufficiency criterion: a transition "qualifies" if `|% change| < 2%`.

**Exact command used to run this experiment:**
```bash
python experiments/exp1_array_sensitivity/run_exp1.py \
  --workload config/workloads/exp0_pilot.yaml
python experiments/exp1_array_sensitivity/analyze_exp1.py
```

**Deviations from the plan, and why:**

- **Workload reused from Experiment 0, not a newly-defined "representative workload."** The plan says "one representative workload" without specifying it. `config/workloads/exp0_pilot.yaml` (seq_len=64, hidden_dim=128, num_heads=8) was reused for the pragmatic reason already established in Exp 0: it's already verified at ~2s/simulation call, whereas a literal seq_len=512 workload (closer to what Experiment 3+ will actually sweep) would cost roughly 90 real simulation calls (5 array sizes × (3 single-dataflow + 9 per-stage-search + implicit combine) ≈ 65 calls, plus re-runs) at ~178s each — 4+ hours. **This choice has real consequences for what this experiment can conclude — see §8/§9: this is not a minor implementation detail, it directly explains the result.**

## 4. Raw Results Summary

n = 1 per (method, array_size) — every method here is deterministic (plan ground rule 3), so no seed-based mean/std/CI applies; each cell below is one real simulation (or, for `best_fixed_per_stage`, one `combine_simulation_results` aggregate of 3 real simulations).

**PE utilization (%) by array size and method** (`results/processed/exp1_summary.csv`):

| Array size | ws_only | os_only | is_only | best_fixed_per_stage |
|---|---|---|---|---|
| 8×8 | 74.44 | 81.07 | 76.10 | **88.33** |
| 16×16 | 58.23 | 67.02 | 62.08 | **78.48** |
| 32×32 | 30.68 | 43.20 | 42.00 | **48.19** |
| 64×64 | 14.10 | 25.57 | 27.45 | **30.50** |
| 128×128 | 4.71 | 6.31 | 7.21 | **8.40** |

**%-change in utilization per doubling** (`results/processed/exp1_pct_change_per_doubling.csv`) — sufficiency criterion: `|change| < 2%`:

| Transition | ws_only | os_only | is_only | best_fixed_per_stage |
|---|---|---|---|---|
| 8→16 | −21.77% | −17.33% | −18.43% | −11.14% |
| 16→32 | −47.32% | −35.53% | −32.33% | −38.60% |
| 32→64 | −54.03% | −40.83% | −34.64% | −36.71% |
| 64→128 | −66.57% | −75.33% | −73.75% | −72.47% |

**Transitions meeting the <2% sufficiency criterion: 0 / 16.**

`best_fixed_per_stage` has the highest utilization at every array size, consistent with it being the empirically-optimized combination of the other three. Its per-stage dataflow choice is stable at 8×8/16×16 (`qkv_projection`→IS, `qk_transpose`→WS, `score_times_v`→OS) and shifts once at 32×32 and above (`qk_transpose`→OS instead of WS) — see `results/raw/exp1.csv`'s `notes` column (`optimizer=="best_fixed_per_stage"` rows) for the exact per-array-size choice.

## 5. Statistical Analysis

Not applicable — every method compared here is a deterministic static heuristic (plan ground rule 3 explicitly permits single-run reporting only for deterministic baselines, which is exactly what this experiment is). No seeds, no distribution, no significance test to run.

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| — | — | — | — |

Zero `status == "FAILED"` rows in `results/raw/exp1.csv` (65 total rows, all `OK`).

## 7. Figures

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| 1 | `results/figures/exp1_utilization_vs_array_size.png` (+ `.svg`) | PE utilization (%) vs. array size, log₂ x-axis, one line per method. All four methods decline monotonically across the entire 8×8–128×128 range with no plateau; `best_fixed_per_stage` (blue) stays above the three single-dataflow baselines throughout. | `results/raw/exp1.csv`, filtered to `optimizer in {ws_only, os_only, is_only, best_fixed_per_stage}`, `status == "OK"` — `results/processed/exp1_summary.csv` |

## 8. Interpretation

- **[RESULT]** Every one of the 16 array-size-doubling transitions tested, across all 4 methods, shows a utilization *decrease* larger than 2% (range: −11.1% to −75.3%). Zero transitions meet the plan's stated sufficiency criterion anywhere in the 8×8–128×128 range tested.
- **[RESULT]** `best_fixed_per_stage` has the highest utilization at every array size tested (88.3% at 8×8 down to 8.4% at 128×128), consistently above all three single-dataflow baselines by 7–18 percentage points at the smaller sizes.
- **[RESULT]** The single-dataflow ranking is not perfectly stable: `os_only` > `is_only` at 8×8/16×16/32×32, but `is_only` > `os_only` at 64×64/128×128 (25.57% vs. 27.45%, then 6.31% vs. 7.21%). `ws_only` is the worst performer at every array size tested.
- **[INTERPRETATION]** The monotonic decline (rather than the plateau the plan's framing anticipated) is consistent with the workload used (`exp0_pilot.yaml`, seq_len=64) being small relative to every array size tested — even the smallest, 8×8, already only reaches 74–88% utilization, meaning the workload doesn't offer enough parallelism to keep even a small array consistently busy, and larger arrays only make that gap worse. This is the literature's "fragmentation" mechanism operating continuously across this entire size range, not a phenomenon that only appears past some threshold.
- **[LIMITATION] — this is the central caveat of this experiment, not a minor footnote.** Because Experiment 1 reused Experiment 0's small pilot workload (§3 — a pragmatic, wall-clock-driven choice, not a workload chosen to be representative of Experiment 3+'s actual seq_len range of 128–4096), **this run cannot, by itself, justify or reject 16×16/32×32 as Experiment 3–6's primary array sizes** the way the plan's acceptance criterion asks for. The honest reading of this data is narrower: *for a seq_len=64 workload*, smaller arrays (8×8, the smallest tested) are consistently more efficient than any larger size tested, with no evidence of a plateau anywhere in [8, 128]. Whether a plateau exists for the larger, more realistic workloads Experiment 3+ will actually run is a separate, untested question — answering it properly requires either (a) re-running this experiment with a workload sized closer to Experiment 3's planned range (flagged as follow-up work, same wall-clock-driven deferral pattern as Exp 0's own scale reduction), or (b) deferring the array-size justification to Experiment 3's own multi-workload × multi-array sweep, which will contain the relevant evidence as a byproduct of its own design.

Comparison against literature-reported figures (FLAT/COSA/ReDas/Gemmini) is noted here only as context, not a re-run: several of those report high utilization (COSA ≈94%) at array sizes in this project's tested range, but on workloads at realistic production scale — consistent with, not contradicting, the interpretation above that utilization here is workload-scale-limited, not array-size-limited in some absolute sense.

## 9. Limitations & Reasons for Discrepancy

- **This experiment does not, on its own, satisfy its own acceptance criterion** ("written justification, grounded in this plot, for why 16×16/32×32 are the primary sizes") — see §8's central limitation. The data collected is real and correctly analyzed; it simply doesn't support that specific conclusion at this workload scale. Per plan ground rule 1, this is reported as "additional experimentation required," not forced into an unsupported justification.
- **`dram_bandwidth_bytes_per_cycle` is not simulated** (see `common/simulator_interface.py`'s `write_scale_sim_inputs` docstring) — SCALE-Sim's CALC auto-estimate mode was used for every run here, identically to Experiment 0.
- **`energy_per_cycle_pj` is a placeholder** (1.0, not sourced from any citable hardware figure) — the EDP column in `results/processed/exp1_summary.csv` should not be cited in the thesis without revisiting this value first.
- **Recommended next step**, in order of preference: (1) re-run this experiment with a workload at or near Experiment 3's planned scale once wall-clock/compute budget allows (a background/overnight run, as flagged for Exp 0), or (2) treat Experiment 3's own results (which sweep seq_len × array size together) as the actual evidence base for the array-size choice, and revisit this report's conclusion once that data exists.

## 10. Reproduction

```bash
cd "MSc Thesis"   # project root
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/   # if not already built
python experiments/exp1_array_sensitivity/run_exp1.py --workload config/workloads/exp0_pilot.yaml
python experiments/exp1_array_sensitivity/analyze_exp1.py
```

Fully deterministic — every method is a fixed (non-learned) heuristic, so re-running should reproduce these numbers exactly (modulo any change to the installed `scalesim` version — re-verify §0's version pin first if it differs).

**Acceptance criteria for this experiment (per the plan):** Written justification, grounded in this plot, for why 16×16/32×32 are the primary sizes used in Experiments 3–6. **Not met by this run** — see §8/§9. The plot and data are real and complete; the specific justification the plan anticipated is not supported by them at this workload scale, and a concrete path to actually meeting this criterion (re-run at larger scale, or defer to Experiment 3's own data) is given above rather than the criterion being silently declared satisfied.
