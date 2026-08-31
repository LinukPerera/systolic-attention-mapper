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

**Phase A — Stage 0 build-out: done.** **Phase B: Experiment 0 complete, 2026-08-31.**
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
- **Experiment 0 (optimizer bake-off) is complete**, with real results from 350
  real containerized SCALE-Sim simulations (30 brute-force ground-truth evals + 4
  optimizers × 5 seeds × 15 evals each, zero FAILED runs). `bayesian_opt` (optuna
  TPE) was selected — see `config/optimizers/selected.yaml` and
  `experiments/exp0_optimizer_bakeoff/exp0_stats.md` for the full analysis,
  including honestly-flagged limitations (pilot workload scale reduced from the
  plan's example for wall-clock reasons; no significance test at n=5 seeds).
- `common/optimizers/{ppo,evolutionary,bayesian_opt}.py`, `a2c.py`, `environment.py`,
  `brute_force.py` are real, tested implementations now. `common/plotting/figures.py`'s
  `plot_convergence_curves` is real; the rest of that file remains stubbed until
  the experiment that needs each figure is reached.
- `experiments/exp0_optimizer_bakeoff/run_exp0.py` and `analyze_exp0.py` are real,
  runnable scripts. `run_exp1.py` … `run_exp6.py` are still stubs.
- 25 of 25 tests pass, including a real end-to-end SCALE-Sim smoke test.

Phase B continues with Experiment 1 (array-size sensitivity — independent of
Exp 0's outcome), then 2 → 3 → 4/5 → 6, updating `RESULTS_TRACKER.md` after each.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/
pytest tests/   # 25 passed (requires Docker running for the smoke test's simulator stage)
```

SCALE-Sim itself runs containerized (Docker) — see `docs/CODING_AGENT_PLAN.md` §2.
This session set the git identity for commits to this repo locally only
(`git config user.name/email`, not global) using the account email associated with
this session — change it (`git config user.name "Your Name"` /
`git config user.email you@example.com`, run from the repo root) if that's not
what you want attributed on commits.
