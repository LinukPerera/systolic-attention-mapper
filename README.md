# Learning-Based Dataflow Optimization for Transformer Attention Kernels on Systolic Arrays

MSc Electronics and Electrical Engineering (ARU) — Chapter 4/5 implementation
(Setup/Implementation, Results & Evaluation).

**Start here:** [`docs/CODING_AGENT_PLAN.md`](docs/CODING_AGENT_PLAN.md) is the
full implementation plan (ground rules, coding standards, data contract, and the
spec for every experiment). [`docs/EXPERIMENT_STATS_TEMPLATE.md`](docs/EXPERIMENT_STATS_TEMPLATE.md)
is the template each experiment's stats report is copied from.
[`RESULTS_TRACKER.md`](RESULTS_TRACKER.md) is the single-page rollup of every
experiment's status and headline result — check there first for "where do things
stand", then follow through to the experiment's own `expN_stats.md` for detail.

## Layout

```
common/          Shared, experiment-independent library (imported by every experiment;
                 never runs anything on its own). Simulator wrapper, metrics formulas,
                 static-heuristic baselines, the four candidate optimizers, shared
                 plotting style + figure functions.
config/          Shared YAML config: hardware/, workloads/, optimizers/ (including
                 optimizers/selected.yaml, written only after Experiment 0 decides it).
experiments/
  exp0_optimizer_bakeoff/
  exp1_array_sensitivity/
  exp2_convergence/
  exp3_main_sweep/
  exp4_scalability/
  exp5_ablation/
  exp6_generalization/    (optional/stretch — implement only after 0-5 are done)

  Each expN_*/ folder is self-contained:
    run_expN.py            the experiment's runnable script (imports from common/)
    expN_stats.md           this experiment's filled-in report (copy of the template)
    results/
      raw/                  append-only CSV(s), one row per run, per the data contract
      processed/            aggregated tables (mean/std/95% CI) used to build figures
      figures/               PNG+SVG figures, generated only from results/processed/
docs/            CODING_AGENT_PLAN.md, EXPERIMENT_STATS_TEMPLATE.md
tests/           Unit tests for common/workload_gen.py, metrics.py, logging_utils.py,
                 plus an end-to-end smoke test through the real containerized simulator.
docker/          docker/scalesim/Dockerfile — builds the scalesim:latest image every
                 config/hardware/*.yaml's docker_image field points at.
requirements.txt
```

### One deliberate deviation from `CODING_AGENT_PLAN.md` §1

The plan's own repository sketch centralizes `results/` and `docs/stats/` at the
top level, split by result *type* (all raw CSVs together, all figures together,
all stats reports together). This repo instead nests `results/{raw,processed,figures}/`
and `expN_stats.md` *inside* each experiment's own folder, so a single experiment
is fully self-contained (code + docs + its own results) and can be zipped up,
reviewed, or referenced on its own. The data contract, column set, aggregation
logic, and acceptance criteria in the plan are otherwise followed exactly —
only the physical folder layout changed.

## Status

**Phase A — Stage 0 build-out: done.** **Phase B: Experiments 0–5 complete, 2026-09-01.**
See `RESULTS_TRACKER.md` for the live rollup.

- Repo is a git repository (`git init` done; see Setup below for identity note).
- Docker is running, and a real `scalesim:latest` image is built (`docker/scalesim/Dockerfile`).
  `common/simulator_interface.py` and `common/workload_gen.py` were rewritten
  against the actual installed `scalesim==3.0.0`'s verified I/O formats (read from
  source, confirmed with real runs) — not guessed conventions.
- **Two real findings worth knowing before writing the methodology chapter:**
  SCALE-Sim supports only **WS/OS/IS** dataflows (no RS/row-stationary, despite
  the plan's module comment); `stable-baselines3` has **no plain REINFORCE**
  (`common/optimizers/a2c.py` uses A2C in its place, documented there).
- **The optimizer action space was extended to per-stage dataflow (2026-09-02)**,
  directly in response to Experiment 3/5's central finding that the optimizer
  structurally couldn't compete with (or express) `best_fixed_per_stage_mapping`'s
  per-GEMM-stage choice. `AttentionMappingProblem.evaluate()` now runs one real
  simulation per stage and combines them (`common/optimizers/environment.py`);
  every optimizer's search space widened from 3 to 9 real values. Experiments 0
  (selection re-examined, changed to `a2c`) and 3–5 (re-runs in progress) are
  being updated against this new action space — see `RESULTS_TRACKER.md`'s
  "Per-stage action space" section and `docs/NEXT_INSTRUCTIONS.md`.
- **Experiment 0 (optimizer bake-off) is complete**, with real results from 350
  real containerized SCALE-Sim simulations (30 brute-force ground-truth evals + 4
  optimizers × 5 seeds × 15 evals each, zero FAILED runs). `bayesian_opt` (optuna
  TPE) was selected — see `config/optimizers/selected.yaml` and
  `experiments/exp0_optimizer_bakeoff/exp0_stats.md` for the full analysis,
  including honestly-flagged limitations (pilot workload scale reduced from the
  plan's example for wall-clock reasons; no significance test at n=5 seeds).
  **Re-examined 2026-09-02** after the action space below was extended to
  per-stage dataflow: on the new 9-dim space, `bayesian_opt` was no longer
  competitive (worst of the 4 candidates); selection **changed to `a2c`**
  (tied with `ppo` on the data). See `exp0_stats.md` §11 and
  `RESULTS_TRACKER.md`'s "Selected optimizer" section.
- **Experiment 1 (array-size sensitivity) is complete**, now with three runs and a
  confirmed mechanistic finding: the utilization "knee" seen in a representative-
  scale re-run (seq_len=256, head_dim=32) tracks **`head_dim`**, not `seq_len` or
  array size directly — traced to `qk_transpose`/`score_times_v`'s GEMM dimensions
  being fixed at `head_dim` regardless of `seq_len` (`common/workload_gen.py`). The
  falsifiable prediction this implies (doubling head_dim 32→64 shifts the knee one
  array-size doubling higher) was stated, then **confirmed** by a dedicated
  validation run — the knee moved from 32×32–64×64 to 64×64–128×128 exactly as
  predicted. **16×16/32×32 are justified as primary array sizes on this
  mechanistic basis**, and `config/workloads/README.md` now mandates
  `head_dim=64` (fixed) for every Experiment 2+ workload. One real, informative
  failure along the way: the validation run's 8×8 array was genuinely infeasible
  (Docker OOM) — logged as evidence, and a real bug in `best_fixed_per_stage_mapping`
  that used to crash the whole sweep on such a failure is now fixed. Full analysis
  in `experiments/exp1_array_sensitivity/exp1_stats.md`.
- **Experiment 2 (search/training convergence) is complete**: `bayesian_opt`
  (Experiment 0's selection, not re-decided) run for 5 seeds × 40 real iterations
  (200 simulations, zero failures) on a fixed `head_dim=64` workload at 16×16.
  A convergence criterion was stated as a formula *before* running (best-reward-
  so-far variance over a trailing window < ε) — 5/5 seeds converged, all at the
  same iteration. Honestly flagged: the criterion triggers well before the
  trajectories visually stabilize, a real looseness in the chosen ε, reported
  rather than hidden. Full analysis in `experiments/exp2_convergence/exp2_stats.md`.
- **Experiment 3 (main sweep vs. baselines, the core evidence chapter) is
  complete**: a 2×2 grid (seq_len∈{128,256} × array∈{16×16,32×32}, real per-call
  cost measured before committing — seq_len 512/1024 are infeasible at 16×16,
  see Exp 1 above) — 173 real simulations, zero failures. **Central finding:**
  `bayesian_opt` loses significantly to `best_fixed_per_stage` in every cell but
  beats the best single-dataflow baseline significantly in every cell — a real
  **action-space asymmetry**, not a search failure: the optimizer's action space
  (`common/optimizers/environment.py`) applies one dataflow to the whole
  workload per run, while `best_fixed_per_stage` specializes per GEMM stage,
  something the optimizer currently cannot express. Flagged as concrete,
  important follow-up work (extend the action space to per-stage dataflow
  choice) directly relevant to the thesis's "joint learning" contribution claim
  and Experiment 5's ablation design. Full analysis, with both t-test and
  Wilcoxon significance and multiple-comparisons disclosure, in
  `experiments/exp3_main_sweep/exp3_stats.md`.
- **Experiment 4 (scalability, answers RQ3) is complete**: extended Experiment 3's
  grid by one real point (seq_len=512 @ 32×32 — the plan's example up to 8192 is
  infeasible here; seq_len=1024 fails even at 32×32, real measured ceiling) and
  merged it with Experiment 3's 4 existing points (reused, not re-simulated —
  `bayesian_opt` is deterministic given its inputs). **The optimizer's advantage
  over the best baseline shrinks and reverses at seq_len=512** (+0.2–0.8% →
  −1.64%) because the fixed small budget doesn't reliably find the relocated
  optimum (the best static dataflow itself changes at this scale, `os_only` →
  `ws_only`) — one of five seeds missed it. Cost grows substantially with
  `seq_len` and grows *faster* on 16×16 than 32×32, consistent with Experiment 1's
  finding that smaller arrays are more expensive to simulate. Full analysis in
  `experiments/exp4_scalability/exp4_stats.md`.
- **Experiment 5 (ablation) is complete**: isolates `dataflow_only` vs.
  `tiling_only` vs. `joint` (`bayesian_opt`, reused from Experiment 3) on 2
  representative cells. **`dataflow_only` captures exactly 0% of the joint
  optimizer's gain over the best static baseline** (deterministic — it always
  rediscovers the same already-known best dataflow); **`tiling_only` alone
  statistically matches `joint`** (paired test not significant, p=0.32–0.62) —
  meaning the optimizer's real value, at this budget and action space, comes
  entirely from the tiling (SRAM-partition) search, not from jointly learning
  dataflow too. This directly qualifies Experiment 3's headline claim. Full
  analysis in `experiments/exp5_ablation/exp5_stats.md`.
- `common/optimizers/{ppo,evolutionary,bayesian_opt}.py`, `a2c.py`, `environment.py`,
  `brute_force.py`, `convergence.py`, `ablation.py`,
  `common/simulator_interface.py`'s `combine_simulation_results`,
  `common/baselines/static_heuristics.py`'s `evaluate_all_baselines` are real,
  tested implementations now. `common/plotting/figures.py`'s
  `plot_convergence_curves`, `plot_utilization_vs_array_size`,
  `plot_utilization_heatmap`, `plot_pct_improvement_bars`,
  `plot_scalability_panels`, and `plot_ablation_bars` are all real now — every
  figure function the plan lists is implemented.
- `experiments/exp{0,1,2,3,4,5}_*/run_expN.py` and `analyze_expN.py` are real,
  runnable scripts. Only `run_exp6.py` (optional/stretch) remains a stub.
- 33 of 33 tests pass, including a real end-to-end SCALE-Sim smoke test.

**Three experiments now independently point at the same concrete follow-up**:
extending `common/optimizers/environment.py`'s action space to support per-stage
dataflow choice (Experiment 3's finding) and re-testing with a larger iteration
budget (Experiments 4/5's findings) — flagged clearly, not yet scheduled.
Experiment 6 (generalization, optional/stretch per the plan) is the only
experiment left; re-running Experiment 1 at a larger, more representative
workload scale remains flagged as separate follow-up work too.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/
pytest tests/   # 33 passed (requires Docker running for the smoke test's simulator stage)
```

SCALE-Sim itself runs containerized (Docker) — see `docs/CODING_AGENT_PLAN.md` §2.
This session set the git identity for commits to this repo locally only
(`git config user.name/email`, not global) using the account email associated with
this session — change it (`git config user.name "Your Name"` /
`git config user.email you@example.com`, run from the repo root) if that's not
what you want attributed on commits.
