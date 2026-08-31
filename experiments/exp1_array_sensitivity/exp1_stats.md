# Experiment 1 Stats Report — Array-size sensitivity study

> **This report covers two runs.** The original run (§4a) used Experiment 0's small
> pilot workload (seq_len=64) and was inconclusive about array-size sufficiency —
> kept here as a documented sub-finding (workload-scale dependency of array
> utilization), per plan ground rule 6 ("nothing is deleted, only superseded"). The
> re-run (§4b) uses a more representative workload (seq_len=256) and is the primary
> basis for §8's array-size interpretation and `RESULTS_TRACKER.md`'s "Justified
> array sizes" section.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp1` |
| Experiment name | Array-size sensitivity study |
| Date run | 2026-08-31 (original run); 2026-08-31 (representative re-run) |
| Git commit hash | Original run: `46f218c` (raw CSV logs `75a7499-dirty`). Representative re-run: `93cda16` (code that produced it; raw CSV logs `85d5b77-dirty` — working tree had this re-run's own code uncommitted at run time, correctly flagged by the `-dirty` suffix). |
| SCALE-Sim version/commit | `scalesim==3.0.0` (PyPI), containerized — `docker/scalesim/Dockerfile` |
| Python / library versions | Python 3.11.9; see `requirements.txt` |
| Random seeds used | N/A — every method is a deterministic static heuristic (plan ground rule 3); logged with `seed=-1` |
| Raw data file(s) | `results/raw/exp1.csv` (original, seq_len=64); `results/raw/exp1_representative.csv` (re-run, seq_len=256) |
| Processed data file(s) | `results/processed/exp1_summary.csv` + `exp1_pct_change_per_doubling.csv` (original); `exp1_representative_summary.csv` + `exp1_representative_pct_change_per_doubling.csv` (re-run) |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Justify the primary working array sizes (16x16, 32x32) the way the sample thesis justifies its mesh choice.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 1, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** Array size — 8×8, 16×16, 32×32, 64×64, 128×128.
- **Dependent variable(s)/metrics:** PE utilization %, exec cycles, DRAM traffic, EDP, per method; ranking stability of the four methods across array sizes; %-change in utilization per doubling.
- **Controlled/fixed variables:** All five hardware configs' SRAM/bandwidth/energy fields held identical (only array size varies); the four methods compared: `ws_only`, `os_only`, `is_only`, `best_fixed_per_stage`. Workload differs between the two runs — see §3.

## 3. Method

**Procedure** (both runs, identical except for the workload):
1. For each of the 5 array sizes, run `ws_only_mapping`/`os_only_mapping`/`is_only_mapping` — one whole-workload simulation each.
2. Also run `best_fixed_per_stage_mapping` — 9 simulations (3 GEMM stages × 3 dataflows) to find each stage's empirically-best dataflow, then `combine_simulation_results` to aggregate the 3 winners into one whole-workload figure.
3. Compute the %-change in utilization between consecutive array-size doublings, per method, and apply the plan's suggested sufficiency criterion (`|% change| < 2%`).

**Exact commands used:**
```bash
# Original run (seq_len=64)
python experiments/exp1_array_sensitivity/run_exp1.py --workload config/workloads/exp0_pilot.yaml
python experiments/exp1_array_sensitivity/analyze_exp1.py --raw-csv results/raw/exp1.csv --label exp1

# Representative re-run (seq_len=256) — timed on one hardware config first (~11.4s/call
# mean, ~127s per array size, verified real before committing to the full sweep):
python experiments/exp1_array_sensitivity/run_exp1.py \
  --workload config/workloads/exp1_representative.yaml \
  --hardware-names 16x16 --out results/raw/_timing_probe.csv   # then discarded, superseded by the full sweep below
python experiments/exp1_array_sensitivity/run_exp1.py \
  --workload config/workloads/exp1_representative.yaml \
  --out results/raw/exp1_representative.csv
python experiments/exp1_array_sensitivity/analyze_exp1.py \
  --raw-csv results/raw/exp1_representative.csv --label exp1_representative
```

**Deviations from the plan, and why:**

- **Two workloads, not one.** The original run reused Experiment 0's pilot workload (seq_len=64) for pragmatic wall-clock reasons; it turned out too small to answer this experiment's actual question (§4a/§8). The re-run uses `config/workloads/exp1_representative.yaml` (seq_len=256, hidden_dim=256, num_heads=8) — a real, timed midpoint between the too-small pilot and the plan's literal seq_len=512 example (measured at ~178s/call in `exp0_stats.md` §3, making a 90-call sweep at that scale 4+ hours). This is exactly the same verify-before-committing discipline used for `exp0_pilot.yaml`, applied again here.

## 4a. Raw Results Summary — Original run (seq_len=64) — sub-finding

**PE utilization (%) by array size and method** (`results/processed/exp1_summary.csv`):

| Array size | ws_only | os_only | is_only | best_fixed_per_stage |
|---|---|---|---|---|
| 8×8 | 74.44 | 81.07 | 76.10 | 88.33 |
| 16×16 | 58.23 | 67.02 | 62.08 | 78.48 |
| 32×32 | 30.68 | 43.20 | 42.00 | 48.19 |
| 64×64 | 14.10 | 25.57 | 27.45 | 30.50 |
| 128×128 | 4.71 | 6.31 | 7.21 | 8.40 |

**0 of 16 doubling transitions meet the <2% criterion** (range: −11.1% to −75.3%) — utilization declines monotonically across the *entire* tested range, with no plateau anywhere.

## 4b. Raw Results Summary — Representative re-run (seq_len=256) — primary

**PE utilization (%) by array size and method** (`results/processed/exp1_representative_summary.csv`):

| Array size | ws_only | os_only | is_only | best_fixed_per_stage |
|---|---|---|---|---|
| 8×8 | 92.09 | 88.09 | 84.61 | **95.56** |
| 16×16 | 84.77 | 77.62 | 71.60 | **90.85** |
| 32×32 | 73.16 | 63.95 | 53.88 | **82.14** |
| 64×64 | 36.37 | 37.07 | 30.12 | **42.73** |
| 128×128 | 15.09 | 17.98 | 15.48 | **19.39** |

**%-change per doubling** (`results/processed/exp1_representative_pct_change_per_doubling.csv`):

| Transition | ws_only | os_only | is_only | best_fixed_per_stage |
|---|---|---|---|---|
| 8→16 | −7.94% | −11.88% | −15.38% | −4.93% |
| 16→32 | −13.70% | −17.62% | −24.76% | −9.59% |
| 32→64 | −50.28% | −42.04% | −44.10% | −47.98% |
| 64→128 | −58.51% | −51.49% | −48.60% | −54.63% |

**Still 0 of 16 transitions meet the strict <2% criterion.** But the pattern is qualitatively different from §4a: decline is *gentle* through 8→16→32 (mean |change| across those two transitions: **13.2%**, range 4.9–24.8% per method) and then *steepens sharply* at 32→64→128 (mean |change|: **49.7%**, range 42.0–58.5%) — a **~3.8× steeper decline rate** after the 32×32 point than before it (per-method ratio 2.3×–7.1×; `best_fixed_per_stage` shows the sharpest change, 7.1×). This is a real, quantified "knee" between 32×32 and 64×64, not a plateau in the plan's exact <2% sense.

`best_fixed_per_stage`'s per-stage dataflow choice is stable (`qkv_projection`→IS, `qk_transpose`→WS, `score_times_v`→OS) at every array size except 128×128, where `qk_transpose` switches to OS.

## 5. Statistical Analysis

Not applicable — every method is a deterministic static heuristic (plan ground rule 3 explicitly permits single-run reporting only for deterministic baselines).

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| — | — | — | — |

Zero `status == "FAILED"` rows in either raw CSV (65 rows each, both `OK`).

## 7. Figures

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| 1 | `results/figures/exp1_utilization_vs_array_size.png` (+ `.svg`) | Original run (seq_len=64): utilization declines monotonically across the entire range, no plateau. | `results/processed/exp1_summary.csv` |
| 2 | `results/figures/exp1_representative_utilization_vs_array_size.png` (+ `.svg`) | Representative re-run (seq_len=256): a visibly gentler decline through 32×32, then a sharp drop at 64×64+ — the "knee" described in §4b. | `results/processed/exp1_representative_summary.csv` |

## 8. Interpretation

- **[RESULT]** Neither run produces a transition meeting the plan's literal <2%-per-doubling sufficiency criterion — 0/16 in both §4a and §4b.
- **[RESULT]** The two runs' *shapes* differ substantially: §4a (seq_len=64) declines monotonically and roughly proportionally across the whole 8×8–128×128 range; §4b (seq_len=256) shows a distinct two-regime pattern — mean |%-change| 13.2% for 8→16→32 vs. 49.7% for 32→64→128, a ~3.8× acceleration.
- **[RESULT]** `best_fixed_per_stage` is the top performer at every array size in both runs, and shows the *largest* before/after-knee ratio in §4b (7.1×) — meaning its advantage over the single-dataflow baselines is largest specifically in the 8–32 range, and largely disappears by 64×64+ (42.73% vs. 36.37–37.07% for the single-dataflow methods at 64×64 — a much smaller gap than at 8×8's 95.56% vs. 84.61–92.09%).
- **[INTERPRETATION]** The knee in §4b is consistent with the array beginning to outgrow what a seq_len=256 workload's GEMM shapes can keep fed once the array exceeds roughly 32×32 in this dimension — a genuine hardware/workload-scale interaction, not a criterion-satisfying plateau (utilization is still falling meaningfully even in the "gentle" 8–32 region, just far more slowly than after it).
- **[INTERPRETATION] — this is the array-size justification the plan's acceptance criterion asks for, stated honestly rather than as an unsupported "plateau" claim:** 16×16 and 32×32 sit within the gentler-decline regime (mean 13.2% loss per doubling) in the representative-scale run, while 64×64 and 128×128 fall in the steep-decline regime (mean 49.7% loss per doubling) — a ~3.8× difference in how much utilization is sacrificed per doubling. This is real, data-backed support for treating 16×16/32×32 as the primary sizes for Experiments 3–6, on the grounds that they capture most of the achievable utilization before the steep drop-off begins — **not** on the grounds that utilization has plateaued in the plan's literal sense, which it has not at either workload scale tested.
- **[LIMITATION]** The knee's exact location (between 32×32 and 64×64) is workload-scale-dependent by construction — §4a's smaller workload shows no comparable knee at all in the tested range. A workload larger than seq_len=256 (e.g. the plan's literal seq_len=512, or Experiment 3's largest planned seq_len=4096) would plausibly push the knee to a larger array size; this has not been tested here (wall-clock cost — see §3, §9) and is a real open question, not a settled one.
- **[LIMITATION]** Two workloads is a small basis for a general "knee location" claim. Experiment 3's own multi-workload × multi-array sweep will provide much stronger evidence on this exact question as a byproduct of its own design; if its results contradict this experiment's provisional 16×16/32×32 recommendation, Experiment 3's data should be treated as authoritative, not this report's.

Comparison against literature-reported figures (FLAT/COSA/ReDas/Gemmini) is noted here only as context, not a re-run: COSA's reported ≈94% utilization is closely matched by `best_fixed_per_stage`'s 95.56% (8×8) and 90.85% (16×16) in the representative re-run — consistent with, though not proof of, this project's array-size range being reasonable at that workload scale.

## 9. Limitations & Reasons for Discrepancy

- **The strict <2% criterion, applied literally, is never satisfied at either workload scale tested.** §8's array-size recommendation rests on a *relative* decline-rate-acceleration reading of the data instead, explicitly flagged as an adapted criterion, not the plan's literal one — this is a real deviation from the plan's exact wording, made because the literal criterion produced no usable signal at all (0/32 transitions across both runs), and reported as such rather than silently redefining "sufficiency" without saying so.
- **`dram_bandwidth_bytes_per_cycle` is not simulated** (SCALE-Sim's CALC auto-estimate mode is used) and **`energy_per_cycle_pj` is a placeholder** (1.0) in both runs — identical caveats to Experiment 0 and the original Exp 1 run; the EDP columns in the processed CSVs should not be cited without revisiting these values.
- **Only two workload scales tested** (seq_len 64 and 256) — see §8's limitations. A third point (e.g. seq_len=512 or 1024) would substantially strengthen confidence in where the knee actually sits and whether it moves with workload scale as expected; flagged as follow-up work, not scheduled.
- **Recommended next step:** treat 16×16/32×32 as the working primary sizes for Experiments 3–6 on the basis above, but revisit this section once Experiment 3's own seq_len × array-size data exists — that sweep spans a much larger workload range and will either corroborate or override this provisional choice.

## 10. Reproduction

```bash
cd "MSc Thesis"   # project root
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/   # if not already built

# Original run
python experiments/exp1_array_sensitivity/run_exp1.py --workload config/workloads/exp0_pilot.yaml
python experiments/exp1_array_sensitivity/analyze_exp1.py --raw-csv results/raw/exp1.csv --label exp1

# Representative re-run
python experiments/exp1_array_sensitivity/run_exp1.py \
  --workload config/workloads/exp1_representative.yaml \
  --out results/raw/exp1_representative.csv
python experiments/exp1_array_sensitivity/analyze_exp1.py \
  --raw-csv results/raw/exp1_representative.csv --label exp1_representative
```

Fully deterministic — every method is a fixed (non-learned) heuristic; re-running should reproduce these numbers exactly (modulo the installed `scalesim` version — re-verify §0's pin first if it differs).

**Acceptance criteria for this experiment (per the plan):** Written justification, grounded in this plot, for why 16×16/32×32 are the primary sizes used in Experiments 3–6. **Met by §4b/§8, with an explicitly adapted criterion** (relative decline-rate acceleration, not the plan's literal <2% plateau, which neither run satisfied) and an honestly-flagged limitation (knee location tested at only two workload scales, to be corroborated or overridden by Experiment 3's own data).
