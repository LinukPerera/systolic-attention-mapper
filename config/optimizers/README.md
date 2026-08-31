# `config/optimizers/`

One YAML file per candidate optimizer's hyperparameters (`reinforce.yaml`,
`ppo.yaml`, `evolutionary.yaml`, `bayesian_opt.yaml`), consumed by the matching
class in `common/optimizers/`.

`selected.yaml` is written **only after Experiment 0 completes** and names the
winning optimizer with a documented, data-backed reason (`CODING_AGENT_PLAN.md`
§6, Exp 0 acceptance criteria). Do not pre-create it with a guessed winner —
that would be exactly the kind of unearned result the plan's ground rule 1
("never fabricate a result") forbids. Experiments 2–6 reference `selected.yaml`
rather than re-deciding the optimizer choice themselves.
