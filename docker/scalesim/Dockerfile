# SCALE-Sim v3.0.0 (PyPI package `scalesim`), containerized per
# docs/CODING_AGENT_PLAN.md §2 ("Simulator: SCALE-Sim (containerized per the
# proposal — Docker)").
#
# numpy is pinned to <1.25 — verified by hand (2026-08-31): scalesim 3.0.0's
# memory/double_buffered_scratchpad_mem.py calls int(numpy_array) on a
# non-0-dimensional array, which raises TypeError on numpy>=1.25 (the implicit
# scalar conversion for size-1, ndim>0 arrays that pre-1.25 numpy allowed with a
# DeprecationWarning is a hard error from 1.25 onward). This is exactly why
# SCALE-Sim runs in its own container rather than the project's own Python
# environment: common/ (pandas, matplotlib) needs numpy>=1.25/1.26, which is
# directly incompatible with scalesim 3.0.0's numpy<1.25 requirement — the two
# cannot coexist in one environment. If you upgrade the scalesim version here,
# re-verify this pin is still needed (it may be fixed upstream).
FROM python:3.11-slim

RUN pip install --no-cache-dir "numpy<1.25" "scalesim==3.0.0"

# Real upstream bug patch, verified 2026-09-02 by reading scale.py/scale_sim.py
# source in this exact image: scale.py's own `-s`/save-trace CLI flag ("Y"/"N",
# default "Y") is parsed into a local `save_space` variable but then never passed
# to the scalesim() object it constructs -- `save_disk_space=False` is hardcoded
# on that call, so `-s N` silently has NO effect and every real run always writes
# full per-cycle layer*/{IFMAP,FILTER,OFMAP}_{SRAM,DRAM}_TRACE.csv files (measured:
# up to ~430MB for a single (stage, dataflow) candidate at this project's
# seq_len=256/head_dim=64 scale) that common/simulator_interface.py's
# _parse_scale_sim_report NEVER reads (only COMPUTE_REPORT.csv and
# DETAILED_ACCESS_REPORT.csv, both a few hundred bytes). This was the real,
# measured root cause of results/raw/_scratch/ reaching 59GB partway through one
# Experiment 3 re-run, and of real, unnecessary per-call wall-clock I/O (the
# per-stage action space already triples real Docker calls per evaluate()).
# One-line source patch: make save_disk_space actually reflect the CLI flag,
# rather than forking/vendoring the whole package for a single hardcoded
# argument. Re-verify this patch still applies/is still needed if the pinned
# scalesim version above ever changes.
RUN sed -i \
    's/s = scalesim(save_disk_space=False,/s = scalesim(save_disk_space=save_space,/' \
    /usr/local/lib/python3.11/site-packages/scalesim/scale.py && \
    grep -q "save_disk_space=save_space" /usr/local/lib/python3.11/site-packages/scalesim/scale.py

WORKDIR /workspace

ENTRYPOINT ["python", "-m", "scalesim.scale"]
