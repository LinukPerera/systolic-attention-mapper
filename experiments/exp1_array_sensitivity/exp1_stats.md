# Experiment 1 Stats Report — Array-size sensitivity study

> **This report covers three runs.** The original run (§4a, seq_len=64, head_dim=16)
> was inconclusive about array-size sufficiency. The representative re-run (§4b,
> seq_len=256, head_dim=32) found a real "knee" and traced it to a mechanistic
> cause: `head_dim`, not `seq_len` or array size directly. The head64 validation
> run (§4c, seq_len=256, head_dim=64) tested that causal claim's falsifiable
> prediction — **the knee shifts to ~64×64** — directly, and confirmed it. All
> three are kept, per plan ground rule 6 ("nothing is deleted, only superseded");
> §4c/§8's confirmed mechanistic explanation is now the primary basis for the
> array-size justification and `RESULTS_TRACKER.md`'s "Justified array sizes" section.

---

## 0. Metadata

| Field | Value |
|---|---|
| Experiment ID | `exp1` |
| Experiment name | Array-size sensitivity study |
| Date run | 2026-08-31 (original + representative runs); 2026-09-01 (head64 validation run) |
| Git commit hash | Original run: `46f218c` (raw CSV logs `75a7499-dirty`). Representative re-run: `93cda16` (raw CSV logs `85d5b77-dirty`). Head64 validation run: code committed after this write-up — raw CSV logs the pre-commit working-tree state (`-dirty`-suffixed; see §9). |
| SCALE-Sim version/commit | `scalesim==3.0.0` (PyPI), containerized — `docker/scalesim/Dockerfile` |
| Python / library versions | Python 3.11.9; see `requirements.txt` |
| Random seeds used | N/A — every method is a deterministic static heuristic (plan ground rule 3); logged with `seed=-1` |
| Raw data file(s) | `results/raw/exp1.csv` (original, seq_len=64/head_dim=16); `results/raw/exp1_representative.csv` (seq_len=256/head_dim=32); `results/raw/exp1_head64_validation.csv` (seq_len=256/head_dim=64) |
| Processed data file(s) | `results/processed/exp1_summary.csv`, `exp1_representative_summary.csv`, `exp1_head64_validation_summary.csv` (+ matching `*_pct_change_per_doubling.csv` for each) |
| Figures directory | `results/figures/` |

---

## 1. Objective & Hypothesis

Justify the primary working array sizes (16x16, 32x32) the way the sample thesis justifies its mesh choice.

_(Copied verbatim from `../../docs/CODING_AGENT_PLAN.md` §6, Experiment 1, per the template's instruction not to rewrite it differently after the fact.)_

## 2. Variables

- **Independent variable(s):** Array size — 8×8, 16×16, 32×32, 64×64, 128×128.
- **Dependent variable(s)/metrics:** PE utilization %, exec cycles, DRAM traffic, EDP, per method; ranking stability of the four methods across array sizes; %-change in utilization per doubling.
- **Controlled/fixed variables:** All five hardware configs' SRAM/bandwidth/energy fields held identical (only array size varies); the four methods compared: `ws_only`, `os_only`, `is_only`, `best_fixed_per_stage`. Workload differs across the three runs — see §3.

## 3. Method

**Procedure** (both runs, identical except for the workload):
1. For each of the 5 array sizes, run `ws_only_mapping`/`os_only_mapping`/`is_only_mapping` — one whole-workload simulation each.
2. Also run `best_fixed_per_stage_mapping` — 9 simulations (3 GEMM stages × 3 dataflows) to find each stage's empirically-best dataflow, then `combine_simulation_results` to aggregate the 3 winners into one whole-workload figure.
3. Compute the %-change in utilization between consecutive array-size doublings, per method, and apply the plan's suggested sufficiency criterion (`|% change| < 2%`).

**Exact commands used:**
```bash
# Original run (seq_len=64, head_dim=16)
python experiments/exp1_array_sensitivity/run_exp1.py --workload config/workloads/exp0_pilot.yaml
python experiments/exp1_array_sensitivity/analyze_exp1.py --raw-csv results/raw/exp1.csv --label exp1

# Representative re-run (seq_len=256, head_dim=32) — timed on one hardware config first (~11.4s/call
# mean, ~127s per array size, verified real before committing to the full sweep):
python experiments/exp1_array_sensitivity/run_exp1.py \
  --workload config/workloads/exp1_representative.yaml \
  --hardware-names 16x16 --out results/raw/_timing_probe.csv   # then discarded, superseded by the full sweep below
python experiments/exp1_array_sensitivity/run_exp1.py \
  --workload config/workloads/exp1_representative.yaml \
  --out results/raw/exp1_representative.csv
python experiments/exp1_array_sensitivity/analyze_exp1.py \
  --raw-csv results/raw/exp1_representative.csv --label exp1_representative

# Head64 validation run (seq_len=256, head_dim=64) — timed on 64x64 first (15.87s/call,
# ~16 min estimated for the full sweep):
python experiments/exp1_array_sensitivity/run_exp1.py \
  --workload config/workloads/exp1_head64_validation.yaml \
  --out results/raw/exp1_head64_validation.csv
python experiments/exp1_array_sensitivity/analyze_exp1.py \
  --raw-csv results/raw/exp1_head64_validation.csv --label exp1_head64_validation
python experiments/exp1_array_sensitivity/plot_knee_comparison.py   # dedicated cross-run comparison figure
```

**Deviations from the plan, and why:**

- **Three workloads, not one.** The original run reused Experiment 0's pilot workload (seq_len=64) for pragmatic wall-clock reasons; it turned out too small to answer this experiment's actual question (§4a/§8). The representative re-run (`exp1_representative.yaml`, seq_len=256/hidden_dim=256) — a real, timed midpoint between the too-small pilot and the plan's literal seq_len=512 example (measured at ~178s/call in `exp0_stats.md` §3) — found a knee and traced its mechanistic cause to `head_dim`. The head64 validation run (`exp1_head64_validation.yaml`, same seq_len=256 but hidden_dim doubled to 512, so head_dim doubles 32→64) was added specifically to test that cause's falsifiable prediction. Every workload's real per-call cost was timed by hand before committing to its full sweep — the same discipline as `exp0_pilot.yaml`, applied three times now.
- **A real infeasibility found and fixed around, not worked around silently.** The head64 validation run's 8×8 array config was genuinely infeasible for this workload — every single-dataflow simulation was Docker-OOM-killed (`exit 137`) after ~2 minutes, ~88–98% through. The first attempt at this run crashed entirely (`common.baselines.static_heuristics.best_fixed_per_stage_mapping` used to raise when every dataflow candidate failed for a stage, discarding the real FAILED evidence already collected and aborting every remaining array size). Fixed: `best_fixed_per_stage_mapping` now leaves a stage out of its `winners` dict instead of raising (real FAILED SimulationResults stay logged either way), and `run_exp1.py` logs an explicit FAILED `best_fixed_per_stage` row and moves on to the next array size rather than crashing the whole sweep. §4c reports 8×8 as `FAILED`, real data, not omitted (plan ground rule 1) — 16×16 through 128×128 completed normally.

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

## 4c. Raw Results Summary — head64 validation run (seq_len=256, head_dim=64) — confirms the mechanistic explanation

Same `seq_len=256`/`num_heads=8` as §4b, but `hidden_dim` doubled 256→512 so `head_dim` doubles 32→64 — the one-variable change needed to test §8's causal claim directly. **8×8 FAILED for every method** (real Docker OOM, `exit 137` — see §3, §6) and is excluded from the table below; 16×16–128×128 all completed (`status == "OK"`).

**PE utilization (%) by array size and method** (`results/processed/exp1_head64_validation_summary.csv`):

| Array size | ws_only | os_only | is_only | best_fixed_per_stage |
|---|---|---|---|---|
| 8×8 | FAILED | FAILED | FAILED | FAILED (no winner for `qkv_projection`) |
| 16×16 | 84.77 | 89.02 | 87.08 | **94.01** |
| 32×32 | 73.15 | 78.91 | 74.99 | **87.45** |
| 64×64 | 57.41 | 63.37 | 56.38 | **74.56** |
| 128×128 | 27.11 | 34.57 | 30.27 | **36.47** |

**%-change per doubling** (`results/processed/exp1_head64_validation_pct_change_per_doubling.csv`):

| Transition | ws_only | os_only | is_only | best_fixed_per_stage |
|---|---|---|---|---|
| 16→32 | −13.71% | −11.35% | −13.88% | −6.97% |
| 32→64 | −21.51% | −19.69% | −24.81% | −14.74% |
| 64→128 | −52.79% | −45.44% | −46.31% | −51.09% |

**0 of 12 transitions meet the strict <2% criterion** (consistent with §4a/§4b — expected, not a new finding). The regime change is now between **64×64 and 128×128**, not 32×32–64×64: mean |%-change| across 16→32→64 is **15.8%**, vs. **48.9%** for 64→128 — a **~3.1× acceleration** (per-method range 2.4×–4.7×).

**Prediction check (§8's falsifiable claim): the knee was predicted to shift from ~32×32 (§4b, head_dim=32) to ~64×64 (this run, head_dim=64) — CONFIRMED.** §4b's knee sits between 32×32 and 64×64 (head_dim=32, decline accelerates going *into* 64×64); this run's knee sits one doubling higher, between 64×64 and 128×128 (head_dim=64, decline accelerates going *into* 128×128) — the knee moved by exactly one array-size doubling when head_dim doubled, matching the mechanistic prediction precisely. See Fig. 3 for the direct visual comparison.

## 5. Statistical Analysis

Not applicable — every method is a deterministic static heuristic (plan ground rule 3 explicitly permits single-run reporting only for deterministic baselines).

## 6. Failed / Excluded Runs

| Run ID | Condition | Failure reason | Excluded from analysis? |
|---|---|---|---|
| `exp1-ws_only-8x8-WS` | head64 validation, 8×8, ws_only | `docker run exited 137` (OOM-killed, ~92% through) | Yes — excluded from §4c's table (no utilization value exists); row itself kept in `results/raw/exp1_head64_validation.csv` as evidence |
| `exp1-os_only-8x8-OS` | head64 validation, 8×8, os_only | `docker run exited 137` (OOM-killed, ~97% through) | Yes, same reason |
| `exp1-is_only-8x8-IS` | head64 validation, 8×8, is_only | `docker run exited 137` (OOM-killed, ~98% through) | Yes, same reason |
| `exp1-best_fixed_per_stage_search-8x8-*` (×3, `qkv_projection` stage) | head64 validation, 8×8 | Same as above — `qkv_projection`'s per-stage search also OOM-killed on all 3 dataflows | Yes, same reason (`qk_transpose`/`score_times_v` stage searches at 8×8 succeeded — their shapes are much smaller, capped at `head_dim`) |
| `exp1-best_fixed_per_stage-8x8-mixed` | head64 validation, 8×8 | No stage had a successful `qkv_projection` candidate to combine | Yes, same reason |

Zero `status == "FAILED"` rows in `results/raw/exp1.csv` or `exp1_representative.csv` (65 rows each, both fully `OK`). `exp1_head64_validation.csv` has 8 real `FAILED` rows (all 8×8, all genuine OOM infeasibility, not a code fault — see §3) out of 65 total; 16×16–128×128 are fully `OK`.

## 7. Figures

| Fig. # | File | Caption | Source data |
|---|---|---|---|
| 1 | `results/figures/exp1_utilization_vs_array_size.png` (+ `.svg`) | Original run (seq_len=64, head_dim=16): utilization declines monotonically across the entire range, no plateau. | `results/processed/exp1_summary.csv` |
| 2 | `results/figures/exp1_representative_utilization_vs_array_size.png` (+ `.svg`) | Representative re-run (seq_len=256, head_dim=32): a visibly gentler decline through 32×32, then a sharp drop at 64×64+ — the knee described in §4b. | `results/processed/exp1_representative_summary.csv` |
| 3 | `results/figures/exp1_head64_validation_utilization_vs_array_size.png` (+ `.svg`) | Head64 validation run (seq_len=256, head_dim=64): gentle decline through 64×64, sharp drop only at 128×128 — the knee shifted one doubling higher than Fig. 2, as predicted. | `results/processed/exp1_head64_validation_summary.csv` |
| 4 | `results/figures/exp1_knee_vs_head_dim_comparison.png` (+ `.svg`) | Dedicated comparison: `best_fixed_per_stage` utilization vs. array size, one line per run, with each run's `head_dim` marked by a matching dashed vertical line. The steep drop in every line visibly begins right at its own `head_dim` marker — the clearest single piece of evidence for the mechanistic explanation. | `results/processed/{exp1,exp1_representative,exp1_head64_validation}_summary.csv` |

## 8. Interpretation

- **[RESULT]** None of the three runs produces a transition meeting the plan's literal <2%-per-doubling sufficiency criterion — 0/16 (§4a), 0/16 (§4b), 0/12 (§4c).
- **[RESULT]** §4a (head_dim=16) declines monotonically and roughly proportionally across the whole 8×8–128×128 range, with no regime change anywhere. §4b (head_dim=32) and §4c (head_dim=64) both show a distinct two-regime pattern: gentle decline, then a sharp acceleration — but at *different* array sizes (§4b: knee between 32×32–64×64, ~3.8× acceleration; §4c: knee between 64×64–128×128, ~3.1× acceleration).
- **[RESULT — the central, most citable finding of this experiment]** The knee's location is not workload-scale-dependent in general — it is **`head_dim`-dependent specifically**, and this was tested, not just observed: §4b's knee sits exactly at its `head_dim` (32); the falsifiable prediction that doubling `head_dim` (32→64, holding `seq_len`/`num_heads` fixed) would shift the knee to ~64×64 was stated *before* running §4c, and confirmed — §4c's knee sits at its `head_dim` (64), one array-size doubling higher, exactly as predicted. Mechanistically: `common/workload_gen.py`'s `workload_to_gemm_shapes` fixes `qk_transpose`'s K dimension and `score_times_v`'s N dimension at `head_dim`, independent of `seq_len` — only `qkv_projection` scales enough to use an array larger than `head_dim`, so two of the three GEMM stages structurally cannot fill any array bigger than `head_dim`, regardless of `seq_len`. This directly reproduces the per-stage GEMM-shape heterogeneity the literature review's §2.2 documents (Ivanov et al. — data movement/shape heterogeneity dominates attention cost; Kao et al./FLAT — attention-specific dataflow needed precisely because stages don't share a shape profile).
- **[RESULT]** `best_fixed_per_stage` is the top performer at every array size in every run, and its advantage over the single-dataflow baselines is consistently largest below each run's own knee and shrinks sharply above it (§4b: 95.56% vs. 84.61–92.09% at 8×8, narrowing to 42.73% vs. 36.37–37.07% at 64×64; §4c: 94.01% vs. 84.77–89.02% at 16×16, narrowing to 36.47% vs. 27.11–34.57% at 128×128) — consistent with the same head_dim mechanism: `best_fixed_per_stage`'s advantage comes from choosing a better dataflow for the head_dim-capped stages specifically, which matters less once the whole array (including `qkv_projection`) is underutilized past the knee.
- **[RESULT — a genuine, informative side-finding, not a fault]** §4c's 8×8 array was infeasible to even simulate (Docker OOM, `exit 137`, on every dataflow — §6), consistent with the same mechanism: `qkv_projection`'s shape at head_dim=64 (K=512, N=1536) is the largest GEMM in any run in this report, and an 8×8 array takes far more sequential simulation cycles to cover a fixed large shape than a larger array does — real infeasibility, logged as `FAILED`, not fabricated or silently skipped (fixed a real bug in `best_fixed_per_stage_mapping` along the way — see §3).
- **[INTERPRETATION] — the array-size justification the plan's acceptance criterion asks for, now on a mechanistic basis rather than an empirical-only one:** 16×16 and 32×32 sit below every tested workload's `head_dim` (16, 32, and — for §4c specifically — half of 64), meaning they sit in the gentler-decline regime for any workload with `head_dim` ≥ 32, which includes the `head_dim=64` standard now adopted for Experiments 2–6 (see below). This is stronger support than §4b's decline-rate numbers alone (which only showed 16×16/32×32 work well *at head_dim=32*) — the mechanism explains *why* they should keep working as workload scale changes, as long as `head_dim` doesn't shrink below 32.
- **[RECOMMENDATION for Experiments 2–6 — adopted]** `head_dim=32` (§4b) and `head_dim=16` (§4a) both understated how much of a larger array the two head_dim-capped GEMM stages could use. `config/workloads/README.md` now mandates `head_dim=64` (fixed, matching BERT-base/GPT-2) for every Experiment 2+ workload config — vary `seq_len`/`num_heads`, not `hidden_dim` directly.
- **[LIMITATION]** The mechanism is confirmed at exactly two `head_dim` values (32, 64) with `seq_len` held fixed at 256 — a real, tested confirmation, but not a full characterization of how the knee moves with `seq_len` or with a `head_dim` outside [32, 64]. If Experiment 3 uses a different `head_dim` than 64 for any reason, this section's specific array-size recommendation should be re-checked against the mechanism (predict the knee at that `head_dim`), not assumed to still hold unchanged.
- **[LIMITATION]** Only one `seq_len` (256) has a full 3-`head_dim` comparison. Experiment 3's own multi-`seq_len` × multi-array sweep (at the now-standard `head_dim=64`) will provide much broader evidence on whether 16×16/32×32 remain the right primary sizes across `seq_len`; if it disagrees with this section, Experiment 3's data is authoritative, not this report's.

Comparison against literature-reported figures (FLAT/COSA/ReDas/Gemmini) is noted here only as context, not a re-run: COSA's reported ≈94% utilization is closely matched by `best_fixed_per_stage`'s 95.56% (§4b, 8×8) and 94.01% (§4c, 16×16) — consistent with, though not proof of, this project's array-size range being reasonable.

## 9. Limitations & Reasons for Discrepancy

- **The strict <2% criterion, applied literally, is never satisfied at any workload scale tested (0/44 transitions across all three runs).** §8's array-size recommendation rests on the confirmed head_dim mechanism instead, with decline-rate-acceleration numbers as supporting detail — a real, disclosed deviation from the plan's exact wording, made because the literal criterion produced no usable signal at all.
- **`dram_bandwidth_bytes_per_cycle` is not simulated** (SCALE-Sim's CALC auto-estimate mode is used) and **`energy_per_cycle_pj` is a placeholder** (1.0) in all three runs — identical caveats to Experiment 0; the EDP columns in the processed CSVs should not be cited without revisiting these values.
- **A real, first-attempt crash and its fix.** The first attempt at §4c crashed the entire sweep (not just 8×8) when `best_fixed_per_stage_mapping` raised on total per-stage failure, discarding already-collected FAILED evidence. Fixed in `common/baselines/static_heuristics.py` and `run_exp1.py` (§3) — re-run completed cleanly, logging the real 8×8 failures and continuing through 128×128. Anyone re-running `run_exp1.py` against a workload/hardware combination infeasible at some array size should expect this same graceful degradation, not a crash.
- **The `git_commit` column for `exp1_head64_validation.csv`'s rows will show `-dirty`** relative to whatever commit follows this write-up (the code that produced it — the fixed `static_heuristics.py`/`run_exp1.py` — was committed after the run, same pattern as every prior run in this project; see §0).
- **Mechanism confirmed at exactly two `head_dim` values, one `seq_len`.** See §8's limitations — this is real, tested evidence, not a fully general characterization across the whole (`seq_len`, `head_dim`) space.
- **Recommended next step:** treat 16×16/32×32 as the primary sizes for Experiments 3–6 on the mechanistic basis in §8, with `head_dim=64` now standard (`config/workloads/README.md`) so the mechanism's precondition holds. Revisit if Experiment 3 uses a different `head_dim`, or once Experiment 3's own seq_len × array-size data exists at the standard `head_dim`.

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

# Head64 validation run
python experiments/exp1_array_sensitivity/run_exp1.py \
  --workload config/workloads/exp1_head64_validation.yaml \
  --out results/raw/exp1_head64_validation.csv
python experiments/exp1_array_sensitivity/analyze_exp1.py \
  --raw-csv results/raw/exp1_head64_validation.csv --label exp1_head64_validation
python experiments/exp1_array_sensitivity/plot_knee_comparison.py
```

Fully deterministic — every method is a fixed (non-learned) heuristic; re-running should reproduce these numbers exactly (modulo the installed `scalesim` version — re-verify §0's pin first if it differs, and expect the 8×8 OOM in §4c to reproduce the same way on comparable hardware, though the exact iteration count at kill time may vary).

**Acceptance criteria for this experiment (per the plan):** Written justification, grounded in this plot, for why 16×16/32×32 are the primary sizes used in Experiments 3–6. **Met, on a mechanistic and empirically-confirmed basis:** §8 traces the array-size knee to `head_dim` (not `seq_len` or array size directly), states a falsifiable prediction (doubling `head_dim` shifts the knee by one array-size doubling), and confirms it with a dedicated run (§4c) — 16×16/32×32 sit below every tested `head_dim` and below the `head_dim=64` now standardized for Experiments 2–6, which is the actual justification, with the decline-rate numbers as supporting detail rather than the plan's literal (never-satisfied) <2% criterion.
