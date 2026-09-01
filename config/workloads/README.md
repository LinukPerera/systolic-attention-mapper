# `config/workloads/`

One YAML file per transformer workload configuration (`seq_len`, `hidden_dim`,
`num_heads`), or one YAML file per experiment listing the sweep of workloads it
uses. Consumed by `common.workload_gen.AttentionWorkload` / `workload_to_gemm_shapes`.

No files exist here yet — add them as each experiment's workload sweep is defined
per `CODING_AGENT_PLAN.md` §6.

## Standard head_dim (adopted 2026-08-31, after Experiment 1's root-cause diagnosis)

Experiment 1 traced its array-size "knee" to `head_dim = hidden_dim/num_heads`:
`qk_transpose`'s K dimension and `score_times_v`'s N dimension are architecturally
capped at `head_dim` regardless of `seq_len` (see `common/workload_gen.py`'s module
docstring). `exp1_representative.yaml`'s `head_dim=32` and the original pilot's
`head_dim=16` are both smaller than the `head_dim=64` typical of real Transformer
configs (BERT-base, GPT-2), which understates how much of the array those two
GEMM stages could otherwise use, and confounds any array-size or dataflow
comparison with an unrealistically tight cap.

**Every workload config for Experiment 2 onward should hold `head_dim=64` fixed**
(i.e. `hidden_dim = 64 * num_heads`) unless a specific experiment has a stated
reason to vary it (e.g. a dedicated head_dim-sensitivity check). Vary `seq_len`
and `num_heads` (not `hidden_dim` directly) to explore workload scale instead.
See `experiments/exp1_array_sensitivity/exp1_stats.md` §8 for the full diagnosis
and `config/workloads/exp1_head64_validation.yaml` for the validation run testing
this causal claim directly.
