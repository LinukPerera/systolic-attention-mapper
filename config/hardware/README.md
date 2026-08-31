# `config/hardware/`

One YAML file per systolic-array hardware configuration, e.g. `16x16.yaml`,
`32x32.yaml`. Shared across all experiments that reference an array size (Exp 1
sweeps over several; Exp 3–6 pin to the ones Exp 1 justifies).

Fields required at minimum (per Table 2.4 of the literature review and
`CODING_AGENT_PLAN.md` §1/§2): array rows, array columns, SRAM capacity (KB),
DRAM bandwidth, and the per-cycle energy figure `metrics.compute_edp` needs.

No files exist here yet — add one per hardware point as Experiment 1 needs it.
Never hand-edit a value used by a completed run; add a new file instead (plan
ground rule 6: nothing is deleted, only superseded).
