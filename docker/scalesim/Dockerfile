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

WORKDIR /workspace

ENTRYPOINT ["python", "-m", "scalesim.scale"]
