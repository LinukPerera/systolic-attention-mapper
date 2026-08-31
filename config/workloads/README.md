# `config/workloads/`

One YAML file per transformer workload configuration (`seq_len`, `hidden_dim`,
`num_heads`), or one YAML file per experiment listing the sweep of workloads it
uses. Consumed by `common.workload_gen.AttentionWorkload` / `workload_to_gemm_shapes`.

No files exist here yet — add them as each experiment's workload sweep is defined
per `CODING_AGENT_PLAN.md` §6.
