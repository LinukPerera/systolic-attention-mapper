<div align="center">

<img src="docs/paper/figures/fig00_hero.svg" alt="Learning-Based Dataflow Optimisation for Transformer Attention Kernels on Systolic Arrays — an animated 8 by 8 systolic array in which operands stream in from the left and top and each processing element lights as they meet." width="100%">

</div>

<br>

> **Abstract.** Transformer attention maps awkwardly onto systolic-array accelerators. Its three chained matrix multiplications differ in shape and in which operands are static, so one dataflow chosen for the whole workload leaves processing elements idle and moves data redundantly. We build a hardware-in-the-loop search around the cycle-accurate simulator SCALE-Sim that chooses a dataflow and an on-chip memory split **for each attention stage**, and evaluate it in seven experiments under validity rules fixed in advance (five or more seeds, Bonferroni-corrected tests, append-only logs). The main comparison exposed a flaw in our own first formulation: a single global action could not express the per-stage choice that a static baseline made, and the learned mapper lost to it significantly in all four cells. Correcting the action space reversed two independent conclusions — the mapper then matched the strongest static baseline (0 of 4 significant losses), kept its advantage at four times the sequence length, and the ablation flipped from "all tiling, no dataflow" to "either alone suffices". A final zero-shot transfer test found a real but statistically unconfirmed gap across utilisation regimes. The lasting contribution is the diagnostic process, and the reproducibility discipline under which it was carried out.
>
> **Keywords** — systolic array · dataflow · transformer attention · design-space exploration · reinforcement learning · SCALE-Sim · reproducibility

<sub>**Reading guide.** Every figure is an animated SVG that loops gently: it opens on the finished picture, then rebuilds itself so you can watch the data move. Motion stops automatically if your system asks for reduced motion. Numbers in the figures are read straight from this repository's result files — each figure prints its source. This page is a guided tour; the engineering record lives in [`RESULTS_TRACKER.md`](RESULTS_TRACKER.md) and each experiment's own report under [`experiments/`](experiments/).</sub>

**Contents**
[1 Introduction](#1-introduction) · [2 How a systolic array moves data](#2-how-a-systolic-array-moves-data) · [3 Method](#3-method) · [4 Results](#4-results) · [5 Discussion](#5-discussion) · [6 Reproducing this work](#6-reproducing-this-work) · [7 Cite, licence, thanks](#7-cite-licence-thanks) · [References](#references)

---

## 1 Introduction

Self-attention — the core of every Transformer — is made almost entirely of matrix multiplications. Hardware designers run them on **systolic arrays**: two-dimensional grids of simple multiply-accumulate units through which operands ripple rhythmically, so that most data movement is a cheap hop to a neighbour instead of a trip to memory (Jouppi et al., 2017). How efficiently an array is used depends on its **dataflow** — which operand stays put inside each processing element (PE) and which streams past — and on how on-chip memory is divided up. Choosing both well is, today, largely manual work, and most heuristics were inherited from convolutional networks, whose layers are regular.

Attention is not regular. One block is three different multiplications in a row:

<p align="center"><img src="docs/paper/figures/fig01_problem.svg" alt="Three columns, one per attention stage — QKV projection, scores Q times K-transpose, and scores times V. An 8 by 8 array under each lights up where it can be used: all of it for the first stage, only half of the columns for the other two, because their inner dimension is the head dimension." width="100%"></p>

<sub>**Figure 1 · One attention block is three matrix multiplications — and one array cannot fill them all.** Stage 1 multiplies by static weights and grows with the hidden size *H*. Stages 2 and 3 are repeated for every head and multiply two quantities produced at run time; one of their dimensions is capped at the head dimension *D = H / n<sub>h</sub>*. On an array wider than *D*, the surplus PEs sit idle. Array fill is illustrative (schematic), not measured.</sub>

> [!NOTE]
> **In plain words.** The same chip has to run three differently-shaped jobs. A setting that is perfect for the first can waste half the chip on the other two. The question this project asks is: *can a program learn, for each stage separately, the setting that wastes least?*

**Research questions.**

| | Question |
|---|---|
| **RQ1** | How can attention's irregular, multi-stage matrix structure be mapped into an optimisable hardware state–action space? |
| **RQ2** | How far can a learning-based mapper outperform rule-based and brute-force mappers on PE utilisation and energy-delay product? |
| **RQ3** | How well does it hold up as sequence length and head dimension scale? |

---

## 2 How a systolic array moves data

A **dataflow** is a rule for which operand each PE keeps. SCALE-Sim, the simulator used throughout, implements three:

<p align="center"><img src="docs/paper/figures/fig02_dataflows.svg" alt="Three 4 by 4 processing-element grids. In weight-stationary, weights stay in each PE while inputs stream in from the left and partial sums move down. In output-stationary, each output stays in its PE while inputs and weights both stream. In input-stationary, inputs stay while weights stream. A PE fires on the clock cycle its operands arrive together." width="100%"></p>

<sub>**Figure 2 · Which operand stays put, and which streams.** Large marks are *held* in the PE; small marks *stream*. Operands are injected with a one-cycle skew per row and column, so that the two a PE needs reach it on the same clock cycle — the diagonal wavefront. (A row-stationary mode exists in the literature but not in SCALE-Sim 3.0.0, so it is outside this study.)</sub>

No dataflow wins everywhere. Holding weights suits stage 1, whose weights are static; holding outputs avoids partial-sum traffic; holding inputs pays off when one matrix is reused heavily. Attention's three stages differ in exactly the properties that decide this — which is why *per-stage* choice is the interesting design space.

---

## 3 Method

### 3.1 Hardware in the loop

Fabricating hardware was out of scope, so the "chip" is **SCALE-Sim 3.0.0**, a cycle-accurate analytical simulator, run inside a pinned Docker image. A search engine proposes a mapping; the simulator measures it; the measurement returns as a reward. Nothing is predicted or interpolated — every reward is a real simulation.

<p align="center"><img src="docs/paper/figures/fig03_loop.svg" alt="A search engine on the left proposes a dataflow for each of the three stages; the proposal crosses a boundary into the simulated hardware on the right, which returns a measured PE utilisation. The figure steps through the ten proposals of one real search." width="100%"></p>

<sub>**Figure 3 · The hardware-in-the-loop search.** The figure cycles through the ten proposals of one real run (A2C, seed 2, sequence 256 on a 32×32 array), showing each proposed per-stage dataflow and the utilisation the simulator measured for it.</sub>

Formally, a candidate is a point $a \in [0,1]^9$: three numbers per stage, decoded into a dataflow $\in \lbrace \mathrm{WS}, \mathrm{OS}, \mathrm{IS} \rbrace$ and a split of the on-chip SRAM between the *ifmap*, *filter* and *ofmap* buffers. The simulator is called once per stage and the three results are combined; the reward penalises memory traffic so that utilisation cannot be bought with DRAM stalls:

$$
r = U - \lambda \frac{T}{M} \qquad\text{with}\qquad U = \frac{\sum_{s} C_s U_s}{\sum_{s} C_s}
$$

where for stage $s$, $C_s$ is the cycle count and $U_s$ the PE utilisation; $T$ is total SRAM↔DRAM traffic, $M$ the total multiply-accumulates, and $\lambda = 0.01$. Four interchangeable search engines sit behind one interface — **A2C** and **PPO** (stable-baselines3; plain REINFORCE is not available there, so A2C substitutes), an **evolutionary** search (DEAP) and **Bayesian optimisation** (Optuna's TPE).

### 3.2 What a search actually does

Here is that same run laid out as a trajectory — each point is one real simulation, labelled with the dataflow it chose for stages 1, 2 and 3:

<p align="center"><img src="docs/paper/figures/fig04_trajectory.svg" alt="Utilisation of ten successive mappings tried by one search, labelled with their dataflow per stage. The run reaches input-stationary, weight-stationary, output-stationary at step four, at 88.3 percent, above the exhaustive per-stage static baseline of 87.45 percent." width="100%"></p>

<sub>**Figure 4 · Watching one search find its mapping.** The mapping reached at step 4 — input-stationary for the projection, weight-stationary for the scores, output-stationary for the aggregation — is exactly what an exhaustive per-stage static search selects, and the learner reaches it in four simulations. Three of the five seeds find the same mapping.</sub>

### 3.3 Seven experiments as a validity chain

A single head-to-head comparison cannot show that the hardware model is sensible, that the search has converged, or that a win generalises. So the programme is a chain in which each experiment validates a precondition of the next:

<p align="center"><img src="docs/paper/figures/fig05_programme.svg" alt="A dependency graph of seven experiments. Experiment 0 selects the optimiser, 1 the array sizes, 2 checks convergence; all feed Experiment 3, the main comparison, which feeds Experiments 4 (scaling), 5 (ablation) and 6 (transfer). A dashed arrow from Experiment 3 back to Experiment 0 marks the redesign the main comparison forced." width="100%"></p>

<sub>**Figure 5 · The programme, and the arrow that sent the work back.** The dashed arc is the story of this project: a problem found in Experiment 3 reopened Experiment 0 and forced Experiments 3, 4 and 5 to be re-run.</sub>

**Rules fixed before any result was seen.**
- Every stochastic result uses **five or more seeds** and is reported with a spread, never a bare mean.
- Pairwise tests are **Bonferroni-corrected** where several comparisons share a grid, and *both* corrected and uncorrected outcomes are reported.
- Every simulator call is **appended** to a raw log with run ID, time and git commit; **nothing is overwritten, only superseded**, and failed runs are logged as failed — never dropped.
- Claims are tagged as a **result** (read off the data), an **interpretation**, or a **limitation**.

---

## 4 Results

### 4.1 Which search engine? — and a reversal

Four engines were compared against a brute-force reference on a pilot workload. On the original action space Bayesian optimisation was best on every axis. After the later redesign the same procedure gave the *opposite* ranking:

<p align="center"><img src="docs/paper/figures/fig06_bakeoff.svg" alt="A slope chart: Bayesian optimisation moves from best (distance 1.06) on the original action space to worst (2.14) on the per-stage space, while A2C and PPO move from 2.08 and 2.09 to a joint-best 1.06." width="100%"></p>

<sub>**Figure 6 · The ranking of search engines reverses when the action space is corrected.** Distance from the brute-force optimum (lower is better; five seeds). A2C and PPO are statistically indistinguishable; A2C — the method originally specified — is retained for everything that follows. Lowering Optuna's random-start count recovered only part of Bayesian optimisation's deficit (mean reward 77.72), so the reversal is not a simple tuning artefact.</sub>

### 4.2 Which array size? — a prediction, tested

Utilisation falls as arrays grow, but not evenly. It falls slowly until the array dimension *A* reaches the head dimension *D*, then steeply:

<p align="center"><img src="docs/paper/figures/fig07_knee.svg" alt="Utilisation against array size for head dimensions 16, 32 and 64. Each curve falls steeply once the array size passes the head dimension; doubling the head dimension moves the knee one doubling to the right. Beneath the axis, small arrays show how much of an A by A array a stage capped at D equals 32 can occupy." width="100%"></p>

<sub>**Figure 7 · The knee sits where the array size reaches the head dimension.** At *D* = 32 the mean per-doubling decline rises from 13.2 % (8→16→32) to 49.7 % (32→64→128), a 3.8-fold acceleration. The inset arrays show why: beyond *A = D* a quarter, then a half, of the PEs have nothing to do. The prediction that doubling *D* would move the knee one doubling right was written down first, then confirmed by a dedicated run. It is the reason 16×16 and 32×32 arrays are used throughout.</sub>

### 4.3 When has the search converged? — a criterion that misled

A numeric convergence rule was fixed in advance. It declared all five seeds converged at iteration 4:

<p align="center"><img src="docs/paper/figures/fig08_convergence.svg" alt="Running-best reward against iteration for five seeds and their mean. A vertical line at iteration 4 marks where the stated criterion declares convergence; the mean curve keeps rising and reaches 95 percent of its total gain only at iteration 25." width="100%"></p>

<sub>**Figure 8 · A criterion that fires too early.** The mean reward keeps improving until about iteration 25, and a ten-fold stricter threshold left the verdict unchanged. The trailing-window test was being satisfied by a calm run of early, seed-determined trials. Every later experiment therefore uses a fixed, pre-declared budget; convergence iteration is reported only descriptively.</sub>

### 4.4 The turning point: a flaw in the formulation

The main comparison pits the learned mapper against two static baselines: the best *single* dataflow for the whole workload, and **best-fixed-per-stage**, which exhaustively picks the best dataflow for each stage separately. The learned mapper beat the first narrowly — and **lost significantly to the second in all four cells** (−1.9 % to −9.1 %, every *p* < 10<sup>−4</sup>).

That could have been reported as "the learner is not good enough". Instead the cause was traced: the static baseline's edge was *per-stage choice*, which the learner's action space could not even express.

<p align="center"><img src="docs/paper/figures/fig09_redesign.svg" alt="Two pipelines through the three attention stages. In the original action space one dataflow, output-stationary, is applied to all three stages and the best reachable utilisation is 78.9 percent. In the per-stage action space each stage has its own dataflow and SRAM split, found by search as input-, weight- and output-stationary, reaching 88.3 percent at the cost of three simulator calls." width="100%"></p>

<sub>**Figure 9 · From one dataflow for the whole workload to one per stage.** The action space grows from 3 numbers to 9, and each candidate now costs three simulator calls instead of one — an accepted cost of roughly 3×. The per-stage values shown are the best action of the real run in Figure 4.</sub>

### 4.5 Does it beat static mappers? (RQ2)

With the corrected action space the identical grid, seeds and tests were re-run:

<p align="center"><img src="docs/paper/figures/fig10_gap.svg" alt="Dumbbell charts for four workload and array cells. Left: relative to the best per-stage static mapping, the original action space (hollow markers) lost in every cell, and the per-stage action space (filled markers) sits level with parity. Right: relative to the best single dataflow, the margin grows from at most 0.8 percent to as much as 9.5 percent." width="100%"></p>

<sub>**Figure 10 · The gap to the per-stage static baseline closes.** (a) The learner no longer loses significantly in any cell (0/4, was 4/4) and is numerically ahead in one; it does **not** significantly *beat* the exhaustive baseline either. (b) Its margin over the best single dataflow grows from +0.2–0.8 % to +1.6–9.5 %, significant in 3/4 cells uncorrected and 2/4 after Bonferroni correction. Bars are 95 % confidence intervals; *n* = 5 seeds.</sub>

| Cell | best-fixed-per-stage | learned mapper | vs. per-stage (Δ %, *p*) | vs. best single dataflow (Δ %, *p*) |
|---|---:|---:|---:|---:|
| seq 128 · 16×16 | 91.89 % | 91.41 ± 1.72 | −0.53 (0.561, n.s.) | +1.61 (0.133, n.s.) |
| seq 128 · 32×32 | 82.49 % | 83.17 ± 2.57 | +0.82 (0.587, n.s.) | +4.62 (0.033) |
| seq 256 · 16×16 | 94.01 % | 93.33 ± 1.17 | −0.72 (0.266, n.s.) | +4.85 (0.0012) |
| seq 256 · 32×32 | 87.45 % | 86.41 ± 2.24 | −1.19 (0.358, n.s.) | +9.50 (0.0017) |

> [!NOTE]
> **An honest reading of "matches".** Absence of a significant difference at *n* = 5 means "not detected at this sample size", not "proven equal". The defensible claim is the one the dissertation makes: a learned mapper reaches, without hand-written per-workload rules, the performance that otherwise needs an exhaustive per-stage static search — it does not exceed that ceiling here.

### 4.6 Does it scale? (RQ3)

The same test at a four-times-longer sequence (L = 512, the longest feasible on a 32×32 array):

<p align="center"><img src="docs/paper/figures/fig11_scale.svg" alt="Left: improvement over the best single dataflow against sequence length. The original action space turns negative at sequence length 512 (minus 1.6 percent); the per-stage action space stays positive (plus 6.3 percent). Right: search wall-clock cost grows with sequence length and is higher for the per-stage action space." width="100%"></p>

<sub>**Figure 11 · The reversal at longer sequences disappears; cost still grows.** The original run's loss at L = 512 had been blamed on too small a search budget. Re-running with the *same* six-iteration budget — deliberately unchanged, to isolate one variable — turned −1.6 % into +6.3 %, with all five seeds converging. The earlier "budget problem" was a symptom of the action space. Quality improved; cost did not, and is higher because of the extra simulator calls.</sub>

### 4.7 Which component does the work?

Restricted searches isolate each part: *dataflow-only* (SRAM split fixed) and *tiling-only* (dataflow fixed to the static best), compared with the full joint search:

<p align="center"><img src="docs/paper/figures/fig12_ablation.svg" alt="Bars showing the share of the joint search's gain captured by dataflow-only and tiling-only variants. On the original action space dataflow-only captured 0 percent; on the per-stage action space it captures 130 and 104 percent in the two cells, and tiling-only 156 and 119 percent." width="100%"></p>

<sub>**Figure 12 · Dataflow-only search goes from capturing none of the gain to all of it.** The original conclusion — "essentially all the gain is tiling" — reversed. On the per-stage space all three variants are statistically indistinguishable (paired tests, every *p* > 0.23): once any method may choose dataflow *per stage*, it reaches about the same ceiling. Per-stage granularity — not the joint formulation — was the missing capability.</sub>

### 4.8 Does a learned mapping transfer? (a deliberately scoped-down test)

The plan's literal generalisation test needs a policy conditioned on workload parameters — a new training loop. With finite compute, the question was instead asked of *actions*: take five already-learned per-stage actions (head dimension 64, above the knee on a 32-wide array) and apply them, **unchanged**, to two unseen workloads — head dimension 48 (same regime) and 24 (opposite regime). Hypothesis, stated beforehand: transfer works within a regime and degrades across it.

<p align="center"><img src="docs/paper/figures/fig13_transfer.svg" alt="For each of five seeds and two held-out workloads, the utilisation of a transferred action (filled marker) and of a freshly searched action (hollow marker). In most seeds they coincide; gaps appear in two seeds of the different-regime workload and one seed of the same-regime workload." width="100%"></p>

<sub>**Figure 13 · Zero-shot transfer, seed by seed.** Aggregate tests find nothing significant (*p* = 0.69 and 0.59 for transfer vs. fresh search). The per-seed picture is more informative: total gap 3.11 pp in the same regime vs. 4.37 pp in the different regime — the predicted direction, but not confirmable at *n* = 5. A larger search budget (6 → 15 iterations) did not close it; sample size, not training budget, is the binding constraint. The figure also shows a third instance of the same seed reproducing the same action on very different workloads — a limitation of this optimiser at small budgets worth stating plainly.</sub>

### 4.9 An engineering detour worth recording

Re-running the sweeps on the per-stage space filled a disk and crashed Docker. The cause was not the experiments:

<p align="center"><img src="docs/paper/figures/fig14_incident.svg" alt="Four steps — a sweep fills the disk with 59 gigabytes of trace files; Docker's daemon dies; the cause is a simulator command-line flag that is parsed but never used; a one-line patch fixes it — and a log-scale chart showing output per evaluation falling from about 550 megabytes to 144 kilobytes." width="100%"></p>

<sub>**Figure 14 · A silent upstream bug, and a one-line patch.** SCALE-Sim's command line parses a save-trace flag and then hard-codes the opposite, so every run wrote hundreds of megabytes of per-cycle traces that nothing reads. The Docker build now corrects the line; no reported metric changed. Diagnosing it, rather than buying a bigger disk, is the same habit as the rest of the project.</sub>

```diff
-    s = scalesim(save_disk_space=False, …)
+    s = scalesim(save_disk_space=save_space, …)     # applied by a one-line sed in docker/scalesim/Dockerfile
```

---

## 5 Discussion

| | Question | Answer, as far as the evidence goes |
|---|---|---|
| **RQ1** | Formalisation | Each attention block decomposes into three GEMM stages; giving each its own (dataflow, SRAM split) is the granularity the search space needed (Sections 3.1, 4.4). |
| **RQ2** | Beat the heuristics? | **Qualified yes.** It matches the strongest exhaustive per-stage baseline (0/4 significant losses) and significantly beats the best single dataflow in 2 of 4 cells after correction — it does not exceed the exhaustive ceiling (4.5, 4.7). |
| **RQ3** | Scale and adapt? | Quality holds at 4× the sequence length at unchanged budget, but search cost rises (4.6). Zero-shot transfer across utilisation regimes is **not yet demonstrated**: a real but unconfirmed gap (4.8). |

**Limitations.** All comparisons are inside one simulator harness; literature baselines were not re-run under identical conditions. Statistical power is capped at five seeds, so the null results above mean "not detected", not "equal". The search budget was fixed and never swept. Finally, the same seed converging to similar actions across very different workloads bounds how much of the optimiser's performance should be credited to landscape search rather than seed-fixed exploration.

**Further work.** The highest-yield next step is not a new experiment but a bigger one: re-run the two held-out cells with more seeds, add the third held-out cell and the reverse-direction transfer check. Beyond that — re-run a literature baseline inside this harness; study the seed-determinism effect across optimisers; extend the per-stage formulation to the feed-forward layers that surround attention.

> The through-line of the work: an unexpected hardware-mapping result was treated as a signal to re-examine the framework itself, tested through a falsifiable prediction, and its correction measurably reversed two independent conclusions — rather than being optimised past as noise.

---

## 6 Reproducing this work

Everything is scripted; every raw result is in the repository, so any figure or table above can be regenerated without re-simulating.

```bash
git clone https://github.com/LinukPerera/systolic-attention-mapper && cd systolic-attention-mapper
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# SCALE-Sim runs in a container (it needs numpy<1.25, which conflicts with the rest of the stack)
docker build -t scalesim:latest -f docker/scalesim/Dockerfile docker/scalesim/

python -m pytest tests -q                 # unit tests; the end-to-end smoke test needs Docker running
```

Re-run one cell of the main comparison with real simulations (slow: every evaluation is a containerised simulation), then analyse it:

```bash
python experiments/exp3_main_sweep/run_exp3.py \
  --workload-names exp3_seq128 --hardware-names 16x16 \
  --seeds 0 1 2 3 4 --n-iterations 6 --out results/raw/exp3_mine.csv
python experiments/exp3_main_sweep/analyze_exp3.py \
  --raw-csv experiments/exp3_main_sweep/results/raw/exp3_mine.csv --out-prefix exp3_mine --optimizer-name a2c
```

Rebuild the animated figures on this page from the logged results (no Docker needed):

```bash
python docs/paper/build_figures.py
```

<details>
<summary><b>Repository map</b></summary>

```
common/            shared library: simulator wrapper, workload → GEMM shapes, metrics, logging,
                   static baselines, the four optimisers, statistics helpers, plotting style
config/             hardware (array sizes), workloads (sequence length, hidden size, heads), optimiser settings
docker/scalesim/    the pinned SCALE-Sim image, including the one-line trace-file fix
experiments/
  exp0_optimizer_bakeoff/   exp1_array_sensitivity/   exp2_convergence/   exp3_main_sweep/
  exp4_scalability/         exp5_ablation/            exp6_generalization/
     each: run script · analysis script · expN_stats.md (full report) · results/{raw,processed,figures}
docs/paper/         this page's figures: build_figures.py, svgkit.py, figures/*.svg
docs/               implementation plan (ground rules, data contract), report template, development log
tests/              unit tests and an end-to-end smoke test through the real simulator
RESULTS_TRACKER.md  one-page rollup of every experiment, including superseded results and the incident log
```

Each experiment folder keeps the **original** (three-number action space) results next to the **per-stage** re-run, clearly labelled — nothing was overwritten.

</details>

---

## 7 Cite, licence, thanks

**Cite.** GitHub's *Cite this repository* button reads [`CITATION.cff`](CITATION.cff); or:

```bibtex
@mastersthesis{perera2026dataflow,
  author = {Perera, Linuk},
  title  = {Learning-Based Dataflow Optimisation for Transformer Attention Kernels on Systolic Arrays},
  school = {Anglia Ruskin University},
  year   = {2026},
  type   = {MSc dissertation},
  url    = {https://github.com/LinukPerera/systolic-attention-mapper}
}
```

**Licence.** Code: [MIT](LICENSE). Text, figures and result reports: [CC BY 4.0](LICENSE-DOCS.md). The figures embed a renamed subset of the STIX fonts under the SIL Open Font Licence ([notice](docs/paper/STIX-FONT-LICENSE.txt)).

**Thanks.** To Dr Keerthi Gunawickrama for supervision, and to the authors of the open tools this work stands on: SCALE-Sim, stable-baselines3, Optuna, DEAP and Docker.

The complete MSc dissertation is not hosted in this repository.

---

## References

Chen, Y.-H., Emer, J. and Sze, V. (2016) 'Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks', *ISCA*.
Genc, H. et al. (2021) 'Gemmini: Enabling systematic deep-learning architecture evaluation via full-stack integration', *DAC*.
Ivanov, A., Dryden, N., Ben-Nun, T., Li, S. and Hoefler, T. (2021) 'Data movement is all you need: A case study on optimizing transformers', *MLSys*.
Jouppi, N.P. et al. (2017) 'In-datacenter performance analysis of a tensor processing unit', *ISCA*.
Kao, S.-C., Subramanian, S., Agrawal, G., Yazdanbakhsh, A. and Krishna, T. (2023) 'FLAT: An optimized dataflow for mitigating attention bottlenecks', *ASPLOS*.
Samajdar, A., Joseph, J.M., Zhu, Y., Whatmough, P., Mattina, M. and Krishna, T. (2020) 'A systematic methodology for characterizing scalability of DNN accelerators using SCALE-Sim', *ISPASS*.
Vaswani, A. et al. (2017) 'Attention is all you need', *NeurIPS*.
