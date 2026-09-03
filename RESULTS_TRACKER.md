# Results Tracker

Single top-level summary of where every experiment stands and what it found.
This file is a **rollup**, not a source of truth: every number here must trace
back to the experiment's own `experiments/expN_*/expN_stats.md` (which in turn
traces back to `results/raw/`, per `docs/CODING_AGENT_PLAN.md` §0 ground rule 2).
If a number here and the number in the experiment's `stats.md` ever disagree,
the `stats.md` is correct and this file has a bug — fix this file, never the
other way round.

**How to update this file:** only after an experiment's `expN_stats.md` is
filled in from real logged data (not while it's still a placeholder). Copy the
headline number(s) and `[RESULT]`/`[INTERPRETATION]`/`[LIMITATION]` tags
across verbatim — do not rephrase a `[RESULT]` into something stronger than
what the data showed. An experiment with `status: FAILED` runs in its raw CSV
is still summarized here, honestly, not omitted.

---

## Build status (infrastructure, not an experiment result)

Phase A Stage 0 (repo scaffold as a git repo + real `common/` plumbing) and the
start of Phase B (real, verified Docker+SCALE-Sim integration — see
[README.md](README.md#status)) are complete as of 2026-08-31. Docker is running
and a real `scalesim:latest` image is built; `common/simulator_interface.py` and
`common/workload_gen.py` were rewritten against the actual installed SCALE-Sim's
verified I/O formats (not guessed ones). **Real finding for the methodology
chapter:** SCALE-Sim supports only WS/OS/IS dataflows — there is no RS
(row-stationary) mode, contradicting `CODING_AGENT_PLAN.md` §1's module comment.
**Real finding:** `stable-baselines3` has no plain REINFORCE; Experiment 0 uses
A2C in its place (documented in `common/optimizers/a2c.py`).

---

## Status at a glance

| # | Experiment | Status | Date run | Headline result | Report |
|---|---|---|---|---|---|
| 0 | Optimizer bake-off | 🟢 Complete | 2026-08-31 (re-examined 2026-09-02) | **Re-examined on the per-stage action space; selection changed to A2C.** On the new 9-dim action space, `bayesian_opt` had the *worst* mean final reward of the 4 candidates (reversed from the original 3-dim finding); `a2c`/`ppo` now tie for best. See `exp0_stats.md` §11 | [exp0_stats.md](experiments/exp0_optimizer_bakeoff/exp0_stats.md) |
| 1 | Array-size sensitivity | 🟢 Complete | 2026-09-01 | 16×16/32×32 justified mechanistically: the utilization "knee" tracks `head_dim`, not `seq_len` — a stated prediction (doubling head_dim 32→64 shifts the knee one array-size doubling higher) confirmed by a dedicated run (see [Justified array sizes](#justified-array-sizes) below) | [exp1_stats.md](experiments/exp1_array_sensitivity/exp1_stats.md) |
| 2 | Search/training convergence | 🟢 Complete | 2026-09-01 | 5/5 seeds converged (mean iter 4.00, std 0.00) by a pre-stated criterion; a 10× stricter ε changed nothing — a structural criterion limitation (early-trial clustering), not a tuning gap. Exp 3 uses fixed budgets, not early-stopping, as a result | [exp2_stats.md](experiments/exp2_convergence/exp2_stats.md) |
| 3 | Main sweep vs. baselines | 🟡 In progress | 2026-09-01 (3-dim, below); **re-run on per-stage action space started 2026-09-02, not yet finished** | 3-dim (superseded) finding: `bayesian_opt` loses to `best_fixed_per_stage` significantly in all 4 cells (−1.9% to −9.1%) but beats the best single-dataflow baseline significantly in all 4 (+0.2% to +0.8%) — the real action-space asymmetry that motivated the per-stage redesign (see §8 of the report). **Per-stage re-run in progress** — see [Infrastructure incident](#infrastructure-incident-disk-exhaustion-pipeline-slowness-and-process-interruptions-2026-09-02) below for why it's taking multiple sessions | [exp3_stats.md](experiments/exp3_main_sweep/exp3_stats.md) |
| 4 | Scalability study | 🟡 Blocked on Exp 3 re-run | 2026-09-01 (3-dim, below) | 3-dim (superseded) finding: RQ3 answered: optimizer's advantage over best baseline shrinks and reverses at seq_len=512 (+0.2–0.8% → −1.6%, one seed missed the relocated optimum within the fixed 6-iter budget); cost grows substantially and faster on 16×16 than 32×32. **Per-stage re-run not started** (`docs/NEXT_INSTRUCTIONS.md` item 5 — waits on Exp 3/5) | [exp4_stats.md](experiments/exp4_scalability/exp4_stats.md) |
| 5 | Ablation study | 🔴 Not started (per-stage) | 2026-09-01 (3-dim, below) | 3-dim (superseded) finding: `dataflow_only` captures 0% of the joint optimizer's gain over best-static (deterministic); `tiling_only` alone statistically matches `joint` (paired test not significant, p=0.32–0.62) — the thesis's real gain is attributable to tiling search, not joint learning, at this budget/action-space. **Per-stage re-run not started** (`docs/NEXT_INSTRUCTIONS.md` item 4) | [exp5_stats.md](experiments/exp5_ablation/exp5_stats.md) |
| 6 | Generalization test (stretch) | 🔴 Not started | — | — | [exp6_stats.md](experiments/exp6_generalization/exp6_stats.md) |

Status legend: 🔴 Not started · 🟡 In progress · 🟢 Complete (stats.md finalized) · ⚫ Failed/blocked (see notes)

---

## Per-stage action space

Set 2026-09-02, per `docs/NEXT_INSTRUCTIONS.md` item 1 — `common/optimizers/environment.py`'s `AttentionMappingProblem.evaluate()` was rewritten to apply one `(dataflow_mode, SRAM-split)` choice **per GEMM stage** (`qkv_projection`, `qk_transpose`, `score_times_v`) instead of one global choice for the whole workload, matching the granularity `common.baselines.static_heuristics.best_fixed_per_stage_mapping` already used. Real cause of Experiment 3/5's central findings below (optimizer structurally couldn't express or compete with a per-stage baseline). Every optimizer's search space widened from 3 to 9 real values (`decode_raw_action`); `evaluate()` now runs 3 real simulator calls per candidate instead of 1. Full detail, including the new `brute_force.py` per-stage-independent grid-search design (and its [LIMITATION] as an approximation, not a guaranteed global optimum) in the commit message and `common/optimizers/environment.py`'s module docstring. This is why Experiment 0's selection was re-examined (below) and why Experiments 3–5 are being re-run (see `docs/NEXT_INSTRUCTIONS.md`).

## Infrastructure incident: disk exhaustion, pipeline slowness, and process interruptions (2026-09-02)

Real, technical account of why the per-stage Experiment 3 re-run has taken
multiple sessions/restarts instead of one continuous run — kept here rather than
in a commit message because it directly explains the unusual wall-clock time and
the gaps in `results/raw/exp3_perstage.csv`'s timestamps.

**1. Root cause of a real system crash: SCALE-Sim was writing megabytes of unused
trace files per call.** A first per-stage Exp 3 re-run attempt filled the disk —
`results/raw/_scratch/` reached **59GB** partway through one sweep, exhausted
available disk space (down to ~120MB free), and took Docker Desktop's daemon down
with it (`ENOSPC` during container churn). Diagnosed by reading the installed
`scalesim==3.0.0` package source directly inside the built image: `scale.py`'s own
`-s`/save-trace CLI flag (default `"Y"`) is parsed into a local `save_space`
variable but **never passed to the `scalesim` object it constructs** —
`save_disk_space=False` is hardcoded on that call (a real upstream bug, not a usage
error on this project's side). Every real run was therefore writing full per-cycle
`layer*/{IFMAP,FILTER,OFMAP}_{SRAM,DRAM}_TRACE.csv` files — measured up to **~430MB
for a single (stage, dataflow) candidate** at `seq_len=256`/`head_dim=64` — that
`common/simulator_interface.py::_parse_scale_sim_report` never reads (only
`COMPUTE_REPORT.csv` and `DETAILED_ACCESS_REPORT.csv`, both a few hundred bytes).
**Fix:** `docker/scalesim/Dockerfile` now patches the installed `scale.py` with a
one-line `sed` (`save_disk_space=False` → `save_disk_space=save_space`) — a
verified, minimal fix to a real vendored bug, not a fork of the package.
**Measured, real, before/after** on the identical workload/action: scratch output
per `evaluate()` call dropped **550MB → 144KB (~3800×)**; wall-clock dropped
**~81s → ~67-68s (~16-18%)** — trace writing was overwhelmingly a *storage*
problem and only secondarily a *speed* one. Full detail and the exact patch:
commit `08e8176` ("Fix real disk-space/slowness bug...").

**2. A smaller, also-real per-call cost, fixed alongside it.**
`run_scale_sim` ran a full `docker info` daemon healthcheck (measured ~0.6s)
before *every single* real simulation call — with the per-stage action space
already tripling calls per `evaluate()` (1 → 3), this was paid 3× as often as
before the redesign. Memoized per-process instead (checked once, not once per
call); a genuine mid-sweep daemon loss is still caught distinctly from a real
simulation failure via a stderr-pattern check that resets the memoized flag and
re-raises loudly. Same commit as above; unit-tested in `tests/test_simulator_interface.py`.

**3. Why the per-stage action space is *inherently* ~3× more expensive per
`evaluate()` call — a real, accepted methodological cost, not a bug.** The
redesign (`common/optimizers/environment.py`, 2026-09-02) runs one real SCALE-Sim
invocation per GEMM stage (3 stages) instead of one for the whole workload, to let
the optimizer actually express per-stage dataflow choice (the real cause of
Experiment 3/5's original central finding — see
[Per-stage action space](#per-stage-action-space) above). This roughly triples
real Docker/SCALE-Sim calls per candidate evaluated, which is the dominant,
irreducible driver of Experiment 3's per-seed wall-clock now — accepted
deliberately (re-timed before committing to a budget, per this project's standing
discipline), not something further pipeline tuning should try to remove.

**4. The dominant driver of *calendar* time, distinct from per-call speed:
long background sweeps did not survive session boundaries.** The corrected Exp 3
re-run was interrupted **twice** after the disk-space fix landed — once when
Docker Desktop's daemon was lost again (unrelated to disk this time — a session
continuity event killed the backend process, confirmed by `docker info` timing
out with no disk-space signal present) and once when the `nohup`'d sweep process
itself (and its accompanying disk-space safety `Monitor`) had no completion
record after a session compaction/restart, despite `disown`ing it. Each
interruption required manually restarting Docker Desktop and re-launching the
sweep. **Real, honest consequence:** ~10 rows of one (workload, hardware,
optimizer, seed) cell were logged mid-flight before an interruption and had to be
discarded and redone (not corrupted data, just an incomplete attempt that would
have skewed per-seed aggregation if kept alongside a clean re-run of the same
seed) — `run_exp3.py`'s existing per-row incremental logging meant only that one
partial cell's progress was lost, not the whole sweep (seq_len=128's two cells,
~40 minutes of real compute, survived both interruptions intact in
`results/raw/exp3_perstage.csv`). **Lesson for future large sweeps in this
environment:** do not assume a multi-hour background process survives
uninterrupted across a session; either check on long sweeps periodically with an
active monitor (as done for the corrected re-run) rather than a single long wait,
or split a sweep into smaller per-cell invocations against the same `--out` CSV
so an interruption only costs one cell's partial progress, never more.

**Net effect on results already reported:** none of the above changes any
*computed* metric (trace-file suppression removes only unused byproduct files;
the daemon-check memoization and process interruptions are pipeline/operational
concerns, not simulation logic) — Experiments 0–2's finalized numbers are
unaffected. It solely explains why the per-stage Experiment 3/4/5 re-run
(`docs/NEXT_INSTRUCTIONS.md` items 3–5) has taken substantially longer in
wall-clock/calendar time than the original 3-dim runs did.

**Going forward: standing storage/memory discipline for every remaining
experiment (items 2–6), not just a one-time cleanup.**
- **Check disk headroom before committing to any sweep's full budget**, the same
  way per-call *timing* is already re-measured before every new grid (established
  practice since Experiment 1) — a quick `df -h /` plus a real one-cell scratch-size
  sample, not an assumption. With the trace-file fix, per-`evaluate()` scratch is
  now ~144KB (was ~550MB) — three orders of magnitude of headroom recovered, but
  headroom is not infinite, and the daemon-crash mode above shows the failure isn't
  graceful when it's exceeded.
- **`results/raw/_scratch/` is regenerable and `.gitignore`'d by design** — safe to
  delete between sweeps once a sweep's raw CSV rows are safely logged (the CSV, not
  the scratch dir, is the real result — established since this project's first
  Docker integration). Delete it proactively after a large sweep completes, rather
  than letting it accumulate across multiple experiments' runs unnoticed.
- **Docker's own image/container footprint should stay minimal**: only
  `scalesim:latest` (currently ~884MB) is needed for this project. Two unrelated
  images from an old, separate "moodle-test" project (`mariadb:10.11`,
  `moodlehq/moodle-php-apache:8.2`, ~1.73GB combined, last used 6 months prior)
  were found alongside it on 2026-09-02 and removed with the user's confirmation —
  worth a periodic `docker images -a` / `docker system df` check so unrelated
  images don't quietly accumulate again. Build cache (`docker builder prune`) is
  always safe to clear between rebuilds — it's fully reclaimable by design.
- **The ~430GB of non-project disk usage found on 2026-09-02** (this project +
  Docker together account for only a few GB) is outside this project's or Docker's
  footprint entirely and was left untouched — not this project's data to manage,
  and out of scope for these sweeps' own storage discipline, but worth the user's
  own separate attention if disk headroom becomes tight again.
- **Prefer resumable, checkpointed sweeps over one long unattended run**: this
  incident's point 4 above (background processes not reliably surviving session
  boundaries) means a very long single invocation risks losing more progress than
  a sweep split into smaller `--workload-names`/`--hardware-names` slices against
  the same `--out` CSV, or one actively monitored (as the corrected Exp 3 re-run
  now is) rather than a single long blind wait.

---

## Selected optimizer

`config/optimizers/selected.yaml` — set by Experiment 0, 2026-08-31; **changed 2026-09-02** after the optimizer action space was extended to per-stage dataflow (see [Per-stage action space](#per-stage-action-space) below).

- **Winner:** `a2c` (was `bayesian_opt`, 2026-08-31–2026-09-02)
- **Reason:** On the new 9-dim per-stage action space, a re-run of the same 5-seed/15-iteration pilot bake-off found `bayesian_opt` no longer competitive — worst mean final reward (77.32) of the 4 candidates, fewest converged seeds (3/5), slowest convergence (mean 8.0 iters). A real follow-up test lowering optuna's `n_startup_trials` (default 10, plausible mechanism: fewer random-startup samples cover a 3-dim space much better than a 9-dim one within the same budget) partially recovered performance (77.72) but did not close the gap with `a2c`/`ppo` (78.39–78.40) — the reversal isn't fully a hyperparameter artifact. `a2c` and `ppo` are statistically indistinguishable at n=5; `a2c` is chosen as the plan's originally-intended primary method (before the REINFORCE→A2C substitution), not because any measured axis favors it over `ppo`. See `exp0_stats.md` §11 for the full data, the `n_startup_trials` test, and caveats (same small-pilot-scale limitations as the original selection).
- **Original (2026-08-31, now superseded) reason, kept for history:** Across 5 seeds on the 3-dim action space, `bayesian_opt` had smallest mean distance from the brute-force ground-truth best reward (1.06, vs. 2.08 for a2c/ppo, 3.70 for evolutionary), fastest convergence (mean 0.6 iterations to within 5% of ground truth, vs. 5.2 for a2c/ppo), and lowest variance in final reward (std 0.83, vs. evolutionary's 2.56). See `exp0_stats.md` §4/§8 for that data.

## Justified array sizes

Set by Experiment 1 — mechanistic explanation confirmed 2026-09-01.

- **Primary sizes:** `16×16` and `32×32`.
- **Mechanistic basis (primary argument):** Experiment 1's utilization "knee" (a sharp acceleration in decline rate as array size grows) tracks `head_dim = hidden_dim/num_heads`, not `seq_len` or array size directly. Traced to `common/workload_gen.py`: `qk_transpose`'s K dimension and `score_times_v`'s N dimension are fixed at `head_dim` regardless of `seq_len` — only `qkv_projection` scales enough to use an array bigger than `head_dim`, so two of the three GEMM stages structurally cannot fill any array larger than `head_dim`. This directly reproduces the literature review's per-stage GEMM heterogeneity argument (§2.2, Ivanov et al./Kao et al.). **Tested, not just observed:** the falsifiable prediction that doubling `head_dim` (32→64, `seq_len`/`num_heads` held fixed) would shift the knee one array-size doubling higher (~32×32 → ~64×64) was stated before running the validation, then confirmed by a dedicated run (`config/workloads/exp1_head64_validation.yaml`) — the knee moved from between 32×32–64×64 to between 64×64–128×128, exactly as predicted. 16×16/32×32 sit below every tested `head_dim` (16, 32, 64), including the `head_dim=64` now standardized for Experiments 2+ (below) — this is why they're justified as primary sizes, not merely that they performed well at one specific workload scale.
- **Supporting detail (decline-rate numbers, not the main argument):** at `head_dim=32`, mean |%-change| is 13.2% across 8→16→32 vs. 49.7% across 32→64→128 (~3.8× acceleration); at `head_dim=64`, 15.8% across 16→32→64 vs. 48.9% across 64→128 (~3.1× acceleration). The plan's literal `|utilization %-change| < 2%` per-doubling criterion is never satisfied at any workload scale tested (0/44 transitions across all three runs) — a real result, reported as such, not silently redefined; the mechanistic explanation above is what actually grounds the array-size choice.
- **Standard adopted for Experiments 2+:** `config/workloads/README.md` now mandates `head_dim=64` (fixed, matching BERT-base/GPT-2) for every Experiment 2+ workload config — vary `seq_len`/`num_heads`, not `hidden_dim` directly. `head_dim=16`/`32` (Exp 1's original and representative runs) both understated how much of a larger array the two head_dim-capped stages could use.
- **A real infeasibility, handled correctly:** the head64-validation run's 8×8 array config genuinely could not be simulated (Docker OOM, `exit 137`, on every dataflow) — logged as real `FAILED` evidence, not hidden (see "Failed / excluded runs" below). Fixed a real bug along the way: `best_fixed_per_stage_mapping` used to raise and crash the whole sweep when every candidate failed for one stage; it now logs the failure and lets the sweep continue to the next array size.
- **Caveat:** the mechanism is confirmed at exactly two `head_dim` values (32, 64) and one `seq_len` (256) — real, tested evidence, not a full characterization of the whole (`seq_len`, `head_dim`) space. Experiment 3's own multi-`seq_len` × multi-array sweep (at the now-standard `head_dim=64`) will provide much broader evidence on whether 16×16/32×32 hold across `seq_len`; if it disagrees, Experiment 3's data is authoritative, not this section.

Full diagnosis and confirmation: `exp1_stats.md` §4a–c/§8–9. Original handoff doc: [`docs/NEXT_INSTRUCTIONS.md`](docs/NEXT_INSTRUCTIONS.md).

---

## Per-experiment summary

Fill in each block only once the corresponding `expN_stats.md` is finalized.
Copy tagged claims verbatim from that file's §8 (Interpretation); do not add
new interpretation here that isn't already in the source report.

### Experiment 0 — Optimizer bake-off
**Status:** 🟢 Complete (2026-08-31; re-examined on the per-stage action space 2026-09-02, selection changed)

**Original (3-dim action space) run, 2026-08-31:**
- **[RESULT]** Across 5 seeds, `bayesian_opt` (optuna TPE) had the smallest mean distance from the brute-force ground-truth best reward (1.06), the fastest mean convergence to within 5% of that ground truth (0.60 iterations), the lowest final-reward standard deviation (0.83), and 5/5 seeds converging.
- **[RESULT]** `a2c` and `ppo` produced near-identical final-reward distributions (66.68 ± 0.61 for both).
- **[RESULT]** `evolutionary` had the worst mean final reward (65.06), the highest variance (std 2.56), and one non-converged seed.
- **[INTERPRETATION]** `a2c`/`ppo`'s near-identical trajectories reflect this experiment's very small iteration budget (15 evals/seed) dominating both algorithms' behavior with near-random early exploration — not evidence the two methods are equivalent in general.
- **[LIMITATION]** Pilot workload was reduced 5–15x below the plan's example scale, and evaluation budgets were small, specifically to finish in one session (real per-evaluation cost at the plan's literal scale was measured at ~178s — see `exp0_stats.md` §3). No significance test was run at n=5 seeds. Both are real, disclosed scope limits, not hidden ones — full detail in `exp0_stats.md` §8–9.

**Re-run on the per-stage action space (9-dim, see [Per-stage action space](#per-stage-action-space) above), 2026-09-02:**
- **[RESULT]** `bayesian_opt` had the *worst* mean final reward (77.32) of the 4 candidates — reversed from the original finding above — the fewest converged seeds (3/5), and the slowest mean convergence among converged seeds (8.0 iterations).
- **[RESULT]** `a2c` and `ppo` tied for best (78.391 ± 1.843 vs. 78.393 ± 1.841; per-seed values match to 2-3 decimal places for 3 of 5 seeds), both with full 5/5 convergence and lowest distance from the brute-force ground truth.
- **[RESULT]** A real re-test lowering `bayesian_opt`'s `n_startup_trials` from optuna's default 10 to 5 (plausible mechanism: fewer random-startup samples cover a 9-dim space far worse than a 3-dim one within the same 15-trial budget) partially recovered performance (mean 77.72, vs. 77.32 at the default) but did not close the gap with `a2c`/`ppo`.
- **[LIMITATION]** The brute-force "ground truth" used for this re-run is a per-stage-independent-EDP approximation, not a guaranteed global optimum (a true joint grid over 3 stages is combinatorially infeasible — ~27,000 real Docker calls). Real evidence this matters: `a2c`/`ppo` seed 1 both reached a real combined reward (80.4316) *above* the "ground truth" (79.4537).
- **[DECISION]** Selection changed to `a2c` (`config/optimizers/selected.yaml`). `a2c`/`ppo` are tied on the data; `a2c` is chosen as the plan's originally-intended primary method, not because any measured axis favors it over `ppo`.
- Full data, tables, and the `n_startup_trials` test: `exp0_stats.md` §11.

### Experiment 1 — Array-size sensitivity study
**Status:** 🟢 Complete (2026-09-01, includes representative-scale and head_dim-validation re-runs)

- **[RESULT]** At `head_dim=16` (original run), all 16 doubling transitions decline more than 2% with no knee anywhere. At `head_dim=32` (representative run), a knee appears between 32×32–64×64. At `head_dim=64` (validation run), the knee appears one doubling higher, between 64×64–128×128 — exactly matching a stated prediction made before the validation run.
- **[RESULT]** The knee tracks `head_dim` exactly (it sits at `head_dim` in both the head_dim=32 and head_dim=64 runs) — a mechanistic cause traced to `common/workload_gen.py`'s fixed-at-head_dim GEMM dimensions, not a coincidental correlation.
- **[RESULT]** `best_fixed_per_stage` is the top performer at every array size in all three runs; its advantage over single-dataflow baselines is largest below each run's own knee and shrinks sharply above it.
- **[RESULT]** The head_dim=64 validation run's 8×8 array was genuinely infeasible to simulate (Docker OOM, `exit 137`, all dataflows) — real evidence, logged as `FAILED`, not hidden; see "Failed / excluded runs" below.
- **[INTERPRETATION]** 16×16/32×32 are justified because they sit below every tested `head_dim`, including the `head_dim=64` now standardized for Experiments 2+ — a mechanistic argument, not just an empirical decline-rate observation at one workload scale (see [Justified array sizes](#justified-array-sizes) above).
- **[LIMITATION]** Mechanism confirmed at exactly two `head_dim` values and one `seq_len` — Experiment 3's own data (at the now-standard `head_dim=64`) is authoritative if it disagrees with this provisional choice.

### Experiment 2 — Search/training convergence study
**Status:** 🟢 Complete (2026-09-01)

- **[RESULT]** 5/5 seeds converged by the stated criterion (Var of best-reward-so-far over a trailing 5-iteration window < 1.0), all at exactly iteration 4 (mean 4.00, std 0.00).
- **[RESULT]** Final best-reward spread across seeds is small: 85.72 ± 0.47 (range 84.81–86.02).
- **[RESULT] — the criterion, taken literally, triggers well before trajectories visually stabilize.** Every seed keeps improving substantially past iteration 4; the mean trajectory doesn't flatten until ~iteration 25. EPSILON=1.0 is ≈3.6% of the observed reward range — loose enough that an early plateau within a 1.0-point band satisfies the formula despite later iterations finding meaningfully better regions.
- **[INTERPRETATION]** This is a real property of the chosen (WINDOW, EPSILON), not evidence the optimizer stops improving at iteration 4 — the criterion answers a narrower question ("has best-so-far been locally stable") than "has search found its final answer."
- **[RESULT — checked, not left open]** A 10× stricter EPSILON (0.1, re-analyzed on the same raw data, no re-simulation needed) changed nothing — all 5 seeds still converge at iteration 4. Root cause: the first triggering window's actual variance is already 0.0000–0.0137, far below either threshold, because TPE's initial exploration trials happen to cluster tightly by chance. A structural limitation of a trailing-window variance test, not an epsilon-tuning gap.
- **[DECISION, adopted]** Given this, Experiment 3 does not use this criterion to dynamically early-stop optimizer runs — every seed runs a fixed, real-timed budget, and `convergence_iteration` (now shared via `common.optimizers.convergence`) is reported per seed as a descriptive statistic only. Full detail: `exp2_stats.md` §8a.

### Experiment 3 — Main sweep vs. baselines
**Status:** 🟢 Complete (2026-09-01)

- **[RESULT]** `bayesian_opt` loses to `best_overall` (always `best_fixed_per_stage`) significantly in all 4 cells (−1.88% to −9.14%, all p<0.0001 by t-test).
- **[RESULT]** `bayesian_opt` beats `best_single_dataflow` (always `os_only`) significantly in all 4 cells by t-test (+0.23% to +0.79%, all p<0.05) — though only 2/4 survive a conservative Bonferroni correction.
- **[INTERPRETATION] — central finding, not a footnote.** Explained by a genuine action-space asymmetry, not a search failure: `common/optimizers/environment.py` applies one dataflow to the WHOLE workload per run (SCALE-Sim's Dataflow setting is per-run, not per-layer), while `best_fixed_per_stage` achieves its edge by running 3 separate simulations (one per GEMM stage) and combining them — a capability the learned optimizer's action space doesn't have. The optimizer reaches slightly above its true ceiling's raw baseline (`os_only`) in every cell, which is exactly what its SRAM-partition search should buy it.
- **[LIMITATION → concrete follow-up work]** This is a real gap versus the thesis's "joint dataflow+tiling learning" claim — recommend extending the action space to a per-stage dataflow choice so the optimizer can compete with `best_fixed_per_stage` on equal footing. Directly relevant to how Experiment 5's ablation is scoped (even "joint" is currently capped at one global dataflow).
- **[LIMITATION]** Grid reduced to seq_len∈{128,256} (real, measured infeasibility of 512/1024 at 16×16 — see [Justified array sizes](#justified-array-sizes)); 6-iteration optimizer budget; no multiple-comparisons correction applied by default (reported both ways).

### Experiment 4 — Scalability study
**Status:** 🟢 Complete (2026-09-01)

- **[RESULT]** %-improvement over the best single-dataflow baseline is small but positive and flat at seq_len 128/256 (+0.23% to +0.79%, both array sizes), then reverses to −1.64% (CI ±4.55%) at seq_len=512 @ 32×32 — the only array size that could reach that scale (16×16 fails, real infeasibility).
- **[RESULT]** The reversal is driven by one seed of five landing far below the other four, which matched the new best baseline almost exactly — std jumps from 0.08–0.30 to 3.10 at this point.
- **[RESULT]** The best static baseline itself changes with scale: `os_only` wins at seq_len≤256 (every Experiment 3 point); `ws_only` wins at seq_len=512.
- **[INTERPRETATION] — direct RQ3 answer.** The fixed small (6-iteration) budget is enough when the landscape's optimum sits where it usually does, but not reliably enough when the optimum relocates at larger scale. Cost grows substantially with seq_len and grows faster on 16×16 than 32×32 for the same seq_len (consistent with Experiment 1's smaller-arrays-cost-more-to-simulate finding).
- **[LIMITATION]** One point at the new scale, one array size; a budget-sensitivity re-run (larger iteration budget at seq_len=512) is the natural, falsifiable follow-up — flagged, not run.

### Experiment 5 — Ablation study
**Status:** 🟢 Complete (2026-09-01)

- **[RESULT]** `dataflow_only` captures exactly 0% of `joint`'s gain over the best static baseline in both tested cells, deterministically (zero variance across 5 seeds — a 3-option categorical space is reliably fully explored in 6 trials).
- **[RESULT]** `tiling_only`'s mean gain exceeds `joint`'s (117.2%, 116.7% "of joint's gain") but the difference is **not statistically significant** (paired t-test p=0.616, p=0.320; Wilcoxon p=0.875, p=0.375).
- **[INTERPRETATION]** Essentially all of the joint optimizer's real gain over best-static is attributable to the tiling (SRAM-partition) search — dataflow search adds no measurable value once the best static dataflow is known, and there is no evidence joint search outperforms tiling-only at this budget/action-space. Directly qualifies Experiment 3's finding: the optimizer's advantage comes from tiling, not from discovering a better dataflow than the static heuristics already found.
- **[LIMITATION]** Only 2 of Experiment 3's 4 cells tested, both where `os_only` is best-static; untested whether `dataflow_only` still contributes 0% at Experiment 4's seq_len=512 point, where the best static dataflow itself changes to `ws_only` — flagged as a natural, informative follow-up, along with re-running once the action space supports per-stage dataflow choice (Experiment 3's own recommended fix).

### Experiment 6 — Generalization test (optional/stretch)
**Status:** 🔴 Not started

- **[RESULT]** —
- **[INTERPRETATION]** —
- **[LIMITATION]** —

---

## Failed / excluded runs (cross-experiment)

Only entries whose experiment-level `expN_stats.md` §6 already documents them —
this section exists so a reader doesn't have to open all seven reports to see
whether anything failed.

| Experiment | Run ID | Condition | Failure reason |
|---|---|---|---|
| exp1 (head64 validation) | `exp1-{ws,os,is}_only-8x8-*`, `exp1-best_fixed_per_stage_search-8x8-*` (qkv_projection stage), `exp1-best_fixed_per_stage-8x8-mixed` (8 rows total) | 8×8 array, `head_dim=64` workload | Docker OOM-killed (`exit 137`), 88–98% through each simulation — real infeasibility (largest GEMM in any Exp 1 run vs. smallest array), not a code fault. Full detail: `exp1_stats.md` §6. |

---

_Last updated: 2026-09-02. Experiments 0–2 complete and unaffected by the per-stage redesign. Experiment 0 re-examined on the new 9-dim per-stage action space and selection changed to `a2c` (§ above). Experiments 3–5's 3-dim-action-space findings (below, in "Per-experiment summary") are superseded-pending-re-run, not wrong — they're exactly what motivated the per-stage redesign. Experiment 3's per-stage re-run is in progress, delayed by a real infrastructure incident (disk exhaustion from an unused-trace-file bug in SCALE-Sim itself, now fixed; two background-process interruptions from session boundaries) — see [Infrastructure incident](#infrastructure-incident-disk-exhaustion-pipeline-slowness-and-process-interruptions-2026-09-02) above for the full technical account and the storage/memory discipline adopted going forward. Experiments 4–5's per-stage re-runs have not started (blocked on Experiment 3 per `docs/NEXT_INSTRUCTIONS.md`'s stated ordering). Experiment 6 (optional/stretch) not started._
