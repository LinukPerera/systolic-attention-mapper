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
tests/           Unit tests for common/workload_gen.py and common/metrics.py (required
                 minimum per the plan — these are the two places a silent bug would
                 invisibly corrupt every downstream number).
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

**Phase A — Stage 0 build-out: done, 2026-08-31.** This is prerequisite plumbing,
not an experiment result — `RESULTS_TRACKER.md` still shows every experiment as
🔴 Not started, correctly.

- Repo is now a git repository (`git init` done; see Setup below for identity note).
- `common/workload_gen.py`, `common/simulator_interface.py`, `common/metrics.py`,
  `common/logging_utils.py`, `common/plotting/style.py`, and
  `common/baselines/static_heuristics.py` (WS/OS/IS/RS + best-fixed-per-stage) are
  **real implementations**, not stubs — GEMM shape derivation, SCALE-Sim topology/config
  writing, Docker invocation + report parsing, EDP/%-diff formulas, the append-only
  CSV schema + mean/std/95%-CI aggregation, and the shared Okabe-Ito plotting style.
- `common/optimizers/*.py` and `common/plotting/figures.py` remain stubs — out of
  Stage 0's scope (they're Phase B, experiment-specific work), still raise
  `NotImplementedError` with the design docstrings from the original scaffold.
- Every `run_expN.py` is still a stub — running an experiment is Phase B.
- **Docker:** the CLI is installed (`/usr/local/bin/docker`, v29.2.0) but **the
  daemon is not currently running/reachable** on this machine. `run_scale_sim()`
  checks for this and raises a clear `RuntimeError` rather than hanging or
  fabricating a result — confirmed by hand (see git log). Start Docker Desktop (or
  the daemon) before running anything that touches the simulator, and build/pull
  whatever real SCALE-Sim image you intend to use — `docker_image: "scalesim:latest"`
  in the two `config/*/smoke_*.yaml` files is a placeholder, not a real image.
- **Smoke test** (`tests/test_smoke_pipeline.py`): the workload → GEMM shapes →
  topology/config-file stage runs for real and passes, proving that plumbing.
  The Docker-dependent stage (→ simulator → metrics → one logged CSV row) is written
  for real but **skips cleanly** in this environment ("Docker daemon not reachable")
  rather than faking a pass — re-run `pytest tests/` once Docker is up to exercise it.
- 21 of 22 tests pass, 1 skipped (the Docker-dependent smoke stage, for the reason above).

Phase B (running Experiments 0–6 in dependency order, updating `RESULTS_TRACKER.md`
after each) has not started.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/   # 21 passed, 1 skipped until Docker is running (see Status above)
```

SCALE-Sim itself runs containerized (Docker) — see `docs/CODING_AGENT_PLAN.md` §2.
This session set the git identity for commits to this repo locally only
(`git config user.name/email`, not global) using the account email associated with
this session — change it (`git config user.name "Your Name"` /
`git config user.email you@example.com`, run from the repo root) if that's not
what you want attributed on commits.
