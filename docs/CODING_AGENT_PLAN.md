# Coding Agent Implementation Plan
**Project:** Learning-Based Dataflow Optimization for Transformer Attention Kernels on Systolic Arrays
**Degree:** MSc Electronics and Electrical Engineering (ARU)
**Scope of this document:** everything needed to implement, run, and document Chapter 4 (Setup/Implementation) and Chapter 5 (Results & Evaluation) of the dissertation.
**Companion document:** `EXPERIMENT_STATS_TEMPLATE.md` — one copy of this template must be filled in per experiment and saved alongside that experiment's results. This plan and the template are not optional scaffolding; **the volume and rigor of documented evidence they produce is the primary deliverable of this phase of the project**, on par with the code itself.

---

## 0. Non-negotiable ground rules

1. **Never fabricate a result.** If SCALE-Sim fails, a config is infeasible, a run times out, or an optimizer diverges, log it as such (`status: FAILED`, with the error) — do not substitute a plausible-looking number. A missing data point plus a note ("additional experimentation required") is always acceptable; an invented one is never acceptable, per the project's academic-integrity rules.
2. **Every number in every plot and table must trace back to a logged raw-data row.** No hand-entered or "illustrative" figures. If a plot cannot be produced from `results/raw/*.csv`, it should not exist yet.
3. **Every stochastic result (anything touching the optimizer) is reported across ≥5 random seeds**, as mean ± standard deviation (or 95% CI), never as a single run. Single-run numbers may appear only for deterministic baselines (fixed-heuristic mappers).
4. **Facts, hypotheses, and interpretation are kept separate in all written output.** When the stats-template asks for interpretation, tag each claim `[RESULT]` (directly read from data), `[INTERPRETATION]` (the agent's/author's reading of the result), or `[LIMITATION]` (an acknowledged gap). This mirrors the FACT/RECOMMENDATION/ASSUMPTION convention already used elsewhere in this project's documents.
5. **Code must be logically modular and heavily commented** — see §3. A reviewer (supervisor, examiner, or future-you) should be able to read `optimizer/reinforce.py` in isolation and understand the state/action/reward design without cross-referencing five other files.
6. **Nothing is deleted, only superseded.** Every experiment run gets a unique, timestamped run ID; re-running an experiment creates a new run directory rather than overwriting the old one, so failed/early attempts stay available for the "reasons for error" discussion later.

---

## 1. Repository structure to create

```
project-root/
├── config/
│   ├── hardware/              # YAML: array sizes, SRAM capacities, bandwidths (per Table 2.4 of the lit review)
│   ├── workloads/              # YAML: seq_len, hidden_dim, num_heads combinations to test
│   └── optimizers/             # YAML: hyperparameters per candidate optimizer (Exp 0)
├── src/
│   ├── workload_gen.py         # Transformer hyperparams -> QKV/QK^T/score*V GEMM shapes -> SCALE-Sim input
│   ├── simulator_interface.py  # wraps SCALE-Sim: writes config, triggers run, parses cycles/utilization/traffic
│   ├── baselines/
│   │   ├── static_heuristics.py   # WS-only, OS-only, IS-only, RS-only, best-fixed-per-stage
│   ├── optimizers/
│   │   ├── base.py             # common interface: state, action, reward, step(), reset()
│   │   ├── reinforce.py
│   │   ├── ppo.py
│   │   ├── evolutionary.py
│   │   └── bayesian_opt.py
│   ├── metrics.py              # PE utilization %, memory traffic, execution cycles, EDP; %-diff vs baseline
│   ├── logging_utils.py        # structured run logging -> results/raw/*.csv, one row per (exp, config, seed, iter)
│   └── plotting/
│       ├── style.py            # one shared matplotlib/seaborn style (fonts, colour palette, dpi=300) used by every figure
│       └── figures.py          # one function per figure type, each takes a results CSV and writes a PNG+SVG
├── experiments/
│   ├── exp0_optimizer_bakeoff.py
│   ├── exp1_array_sensitivity.py
│   ├── exp2_convergence.py
│   ├── exp3_main_sweep.py
│   ├── exp4_scalability.py
│   ├── exp5_ablation.py
│   └── exp6_generalization.py   # optional/stretch — implement last
├── results/
│   ├── raw/                     # append-only CSVs, one per experiment, one row per run
│   ├── processed/                # aggregated tables (mean/std/CI) used to generate plots and the stats MD
│   └── figures/                  # PNG+SVG output, one subfolder per experiment
├── docs/
│   ├── CODING_AGENT_PLAN.md      # this file
│   ├── EXPERIMENT_STATS_TEMPLATE.md
│   └── stats/                    # one filled-in copy of the template per experiment: exp0_stats.md ... exp6_stats.md
└── requirements.txt / environment.yml
```

Rationale: `src/` is only ever imported, never contains a `__main__` that runs an experiment — every experiment is a thin script in `experiments/` that composes `src/` building blocks. This keeps each experiment reproducible from a single command and keeps the reusable logic testable in isolation.

---

## 2. Environment & tooling

- **Simulator:** SCALE-Sim (containerized per the proposal — Docker), used exactly as characterised in the literature review: 2D PE grid, WS/OS/IS dataflows, double-buffered SRAM, DRAM interface.
- **Language:** Python 3.10+.
- **Core libraries:** `numpy`, `pandas` (results tables), `scipy.stats` (significance tests, CIs), `matplotlib` + `seaborn` (all plotting — one shared style module, §5), `pyyaml` (config), `tqdm` (progress, not a substitute for logging).
- **Optimizer libraries** (use established, citable implementations rather than reimplementing from scratch, so the methodology chapter can cite them): `stable-baselines3` (REINFORCE/PPO), `deap` (evolutionary), `scikit-optimize` or `optuna` (Bayesian).
- **Reproducibility:** every run script accepts `--seed`; the global seed also seeds `numpy`, the optimizer library's own RNG, and (if used) `torch`. Every run writes its resolved config (all hyperparameters, not just the ones changed) to `results/raw/<exp>/<run_id>/config.yaml`.
- **Version pinning:** `requirements.txt` with exact versions; log the SCALE-Sim commit/version hash used, since simulator version is a stated methodological detail the sample thesis records.

---

## 3. Coding standards

- **Docstrings on every module, class, and function**: purpose, inputs, outputs, and — critically — *why* a design choice was made where it isn't obvious (e.g., "reward is utilization minus a memory-traffic penalty term, weighted per §3.3 of the methodology chapter, because raw utilization alone lets the agent ignore DRAM stalls").
- **Type hints throughout.**
- **No magic numbers.** Array sizes, SRAM capacities, reward weights, convergence thresholds all come from `config/*.yaml`, never hardcoded in a script.
- **Inline comments explain intent, not mechanics** — don't comment `i += 1  # increment i`; do comment `# discard the first 10% of iterations as warm-up before computing the convergence-stability window (see Exp 2 criterion)`.
- **Fail loudly.** A SCALE-Sim run that errors should raise/log clearly, not be silently caught and skipped.
- **One function = one responsibility.** `simulator_interface.py` never contains optimizer logic; `optimizers/*.py` never parses SCALE-Sim output directly — they call `metrics.py`.
- **Unit tests** for `workload_gen.py` (shape arithmetic) and `metrics.py` (utilization/EDP formulas) at minimum — these are the two places a silent bug would invisibly corrupt every downstream number.

---

## 4. Logging & data contract (read before writing any experiment script)

Every experiment appends rows to `results/raw/<exp_name>.csv` with **at minimum** these columns:

`run_id, experiment, timestamp, git_commit, optimizer (or "baseline"), array_rows, array_cols, sram_kb, dataflow_mode, seq_len, hidden_dim, num_heads, seed, iteration, pe_utilization_pct, exec_cycles, sram_dram_traffic_bytes, edp, wall_clock_s, status, notes`

- `status` is `OK` or `FAILED` (never omit failed runs from the log — they are evidence for the limitations section).
- Aggregation into `results/processed/` (mean, std, 95% CI grouped by the non-seed columns) happens in a shared helper in `logging_utils.py`, used identically by every experiment, so aggregation logic is written once and trusted everywhere.

---

## 5. Plotting requirements (applies to every experiment)

- One shared style (`src/plotting/style.py`): consistent colour-per-dataflow-mode and colour-per-optimizer mapping used across *all* figures in the thesis, colourblind-safe palette, 300 dpi, PNG for the document + SVG kept for later touch-ups.
- Every figure is generated by a named function in `figures.py` (e.g. `plot_convergence_curves(df, out_path)`), callable independently — so a figure can be regenerated after a data fix without re-running the whole experiment.
- Every figure ships with a **caption stub** written into the corresponding stats MD file (see template) — figure number, one-sentence description, and the source CSV/processed-table path it was built from, so any figure can be traced back to raw data during viva prep.
- Error bars (std or 95% CI) are mandatory on every plot that aggregates across seeds. A bar/line chart with no error bars is not acceptable for a stochastic result.

---

## 6. Experiment specifications

For each experiment: **objective**, **variables**, **procedure**, **required outputs**, **acceptance criteria** (what "this experiment is complete" means). Implement in this order — later experiments depend on earlier ones' outputs.

### Experiment 0 — Optimizer bake-off (pilot; selects the method used everywhere after)
- **Objective:** choose which optimizer (REINFORCE, PPO, evolutionary/GA, Bayesian optimization) is used for all subsequent experiments, on stability + scalability grounds, not assumption.
- **Variables:** independent = optimizer type; fixed = one small workload (e.g. seq_len=512, 8 heads) on a 16×16 array; dependent = iterations-to-within-X%-of-best, seed variance, wall-clock per iteration.
- **Procedure:** run each candidate for a fixed iteration budget × 5 seeds; also run brute-force/exhaustive search on this one small case to establish a ground-truth optimum for "distance from best."
- **Required outputs:** table (per optimizer: mean iterations-to-converge ± std, mean wall-clock, mean final utilization ± std); convergence-curve overlay plot (all optimizers, mean ± CI band, same axes); a short written justification of the winning optimizer in `docs/stats/exp0_stats.md`.
- **Acceptance criteria:** a single optimizer is selected with a documented, data-backed reason; that choice is then hardcoded as the default in `config/optimizers/selected.yaml` and referenced (not re-decided) by Experiments 2–6.

### Experiment 1 — Array-size sensitivity study
- **Objective:** justify the primary working array sizes (16×16, 32×32) the way the sample thesis justifies its mesh choice.
- **Variables:** independent = array size (8×8, 16×16, 32×32, 64×64, 128×128); fixed = one representative workload; dependent = achievable utilization per dataflow, ranking stability of dataflows.
- **Procedure:** run the static-heuristic baselines (not yet the learned optimizer) at each array size; identify the point beyond which larger arrays stop improving utilization or become degenerate (per the lit review's fragmentation argument).
- **Required outputs:** line plot of utilization vs. array size per dataflow mode; a stated criterion for "sufficient" array size (mirroring the sample's explicit convergence threshold, e.g. "<2% utilization change per doubling").
- **Acceptance criteria:** written justification, grounded in this plot, for why 16×16/32×32 are the primary sizes used in Experiments 3–6.

### Experiment 2 — Search/training convergence study
- **Objective:** prove the chosen optimizer reliably converges, with an explicit numeric stopping criterion.
- **Variables:** fixed configuration (from Exp 0/1); independent = seed (≥5); dependent = reward/utilization trajectory per iteration.
- **Procedure:** run the selected optimizer to a generous iteration budget; define and state the convergence criterion (e.g. moving-average reward variance < ε over a window of N iterations); record the iteration at which each seed meets it.
- **Required outputs:** multi-seed convergence-curve plot (individual traces + mean ± CI band); table of convergence iteration per seed, mean ± std.
- **Acceptance criteria:** convergence criterion is stated as a formula (not just "it looked flat"), and holds for all/most seeds — seeds that fail to converge are reported, not discarded.

### Experiment 3 — Main sweep vs. baselines (core evidence chapter)
- **Objective:** the central empirical claim — learned optimizer vs. static heuristics, across realistic workload diversity.
- **Variables:** independent = {optimizer output, WS-only, OS-only, IS-only, best-fixed-per-stage} × {seq_len ∈ [128,512,1024,2048,4096]} × {array ∈ [16×16, 32×32]}; dependent = PE utilization, exec cycles, memory traffic, EDP.
- **Procedure:** for every (workload, array) cell, run every baseline once (deterministic) and the learned optimizer across ≥5 seeds; report both absolute values and %-difference from the best static baseline, in paired tables (mirrors the sample's Tables 5–8 pattern). Add a clearly separate column citing the relevant literature figure (FLAT ≈1.9×, COSA ≈94% utilization, ReDas ≈4.6×, Gemmini WS/OS ≈3×) as a directional reference only — never averaged into your own numbers.
- **Required outputs:** heatmap (utilization by workload × array × method); grouped bar chart of %-improvement over best baseline per workload; full results table (abs + %-diff) for the appendix; 2–3 representative workload cases promoted into the main body.
- **Acceptance criteria:** every reported "X% improvement" claim has a seed-level std/CI attached and, ideally, a significance test (paired t-test or Wilcoxon signed-rank across seeds/workloads) reported in the stats MD — this is what turns "our method is better" into a defensible thesis claim.

### Experiment 4 — Scalability study (answers RQ3 directly)
- **Objective:** show how the optimizer's advantage over baselines, and its own overhead, scale with sequence length and array size.
- **Variables:** independent = seq_len (extend range beyond Exp 3, e.g. up to 8192) and array size; dependent = %-improvement over baseline, optimizer wall-clock/sample count needed to converge.
- **Procedure:** reuse Exp 2's convergence criterion to decide when each run is "done"; log both performance and cost.
- **Required outputs:** two-panel line plot — (a) %-improvement vs. seq_len/array size, (b) optimizer cost vs. seq_len/array size — makes the scalability trade-off explicit rather than implied.
- **Acceptance criteria:** a direct, evidence-based answer to RQ3 is written into the stats MD (not just plots — a sentence stating what happens to performance and cost as scale increases).

### Experiment 5 — Ablation study
- **Objective:** isolate the contribution of joint dataflow+tiling learning (the specific gap claimed in the lit review) from tiling alone or dataflow alone.
- **Variables:** independent = {learned dataflow only (tiling fixed to a reasonable default), learned tiling only (dataflow fixed to best-static), learned joint dataflow+tiling}; dependent = same core metrics as Exp 3, on the same workload/array subset as Exp 3's representative cases.
- **Required outputs:** bar chart comparing the three variants per metric, with error bars.
- **Acceptance criteria:** the stats MD states plainly how much of the total gain is attributable to joint optimization vs. either factor alone — this is the number that defends the thesis's specific novelty claim.

### Experiment 6 — Generalization test (optional/stretch — implement only after 0–5 are complete and documented)
- **Objective:** test whether the learned policy generalizes to unseen workload shapes, or has effectively memorized per-shape optima.
- **Variables:** train/fit on a subset of (seq_len, head_dim) combinations, evaluate zero-shot on held-out combinations not seen during training.
- **Required outputs:** bar/line chart comparing in-distribution vs. held-out performance relative to baseline.
- **Acceptance criteria:** an honest statement of the generalization gap, even if it's large — this belongs in limitations either way.

---

## 7. Definition of "done" for the whole plan

For each experiment N:
1. `experiments/expN_*.py` runs end-to-end from a clean checkout with one command and a config file.
2. `results/raw/expN.csv` contains every run (including failures) with the full column set from §4.
3. `results/figures/expN/` contains every required plot as PNG+SVG, generated only from `results/processed/`.
4. `docs/stats/expN_stats.md` is a filled-in copy of `EXPERIMENT_STATS_TEMPLATE.md` — not a stub, not "TBD" in the results sections.
5. Nothing in 1–4 was hand-edited after generation; if a number in the stats MD doesn't match the CSV, that is a bug to fix, not a number to adjust.

The thesis's Results & Evaluation chapter is then written directly from the `docs/stats/*.md` files and `results/figures/*` — the chapter text should not need to introduce any number that isn't already sitting in one of those files.
