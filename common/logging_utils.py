"""
logging_utils.py — structured run logging (<exp>/results/raw/*.csv) and shared aggregation.

Every experiment writes rows via `append_run_row` using the exact column set from
CODING_AGENT_PLAN.md §4, and aggregates via `aggregate_results` — implemented once
here and used identically by every experiment, so aggregation logic (mean/std/95% CI
grouped by non-seed columns) is written once and trusted everywhere (plan §4).
"""
from __future__ import annotations

import csv
import subprocess
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

# Column order per the data contract in CODING_AGENT_PLAN.md §4 — do not reorder or
# drop columns without updating the plan and every experiment that reads this CSV.
RAW_COLUMNS: list[str] = [
    "run_id", "experiment", "timestamp", "git_commit", "optimizer",
    "array_rows", "array_cols", "sram_kb", "dataflow_mode",
    "seq_len", "hidden_dim", "num_heads", "seed", "iteration",
    "pe_utilization_pct", "exec_cycles", "sram_dram_traffic_bytes", "edp",
    "wall_clock_s", "status", "notes",
]

_VALID_STATUS = ("OK", "FAILED")


def append_run_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one run's result row to an experiment's raw results CSV.

    Creates the file with a header if it does not yet exist. Never overwrites or
    drops existing rows — always opens in append mode (plan ground rule 6: "nothing
    is deleted, only superseded").

    Args:
        csv_path: Destination CSV, conventionally <exp>/results/raw/expN.csv.
        row: Must contain every key in RAW_COLUMNS (extra keys are rejected, to catch
            typos early rather than silently writing a column that never gets read
            back); `status` must be "OK" or "FAILED".

    Raises:
        ValueError: If `row` is missing a required column, has an unexpected extra
            column, or `status` is not "OK"/"FAILED".
    """
    missing = [c for c in RAW_COLUMNS if c not in row]
    extra = [c for c in row if c not in RAW_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"append_run_row: row does not match RAW_COLUMNS. missing={missing}, extra={extra}. "
            "If this is a deliberate new column, update RAW_COLUMNS (and CODING_AGENT_PLAN.md §4) "
            "first — every experiment's CSV must share the same schema."
        )
    if row["status"] not in _VALID_STATUS:
        raise ValueError(f"append_run_row: status must be one of {_VALID_STATUS}, got {row['status']!r}.")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def aggregate_results(csv_path: Path, *, group_by: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Aggregate a raw results CSV into mean/std/95% CI grouped by the given columns.

    Only status == "OK" rows are aggregated — a FAILED run has no meaningful metric
    values and would silently corrupt a mean if included (plan §3, ground rule 1).
    FAILED rows remain in the raw CSV as evidence; they are handled separately (an
    experiment's stats.md §6, "Failed / Excluded Runs"), not dropped from existence.

    The 95% CI uses Student's t-distribution (appropriate for the plan's small,
    ≥5-seed sample sizes, more conservative than a normal-approximation CI). A group
    with n < 2 gets NaN std/CI (undefined for a single sample) with a warning, rather
    than raising — this is expected and fine for deterministic single-run baselines
    (n=1 by design), but a stochastic result found with n < 5 violates plan ground
    rule 3 and should be investigated, not silently plotted.

    Args:
        csv_path: Raw results CSV (see append_run_row).
        group_by: Columns to group by — typically everything except `seed` and
            `iteration`, so aggregation is across seeds.
        metric_cols: Numeric columns to aggregate (e.g. pe_utilization_pct, edp).

    Returns:
        A DataFrame with one row per group and <metric>_n / _mean / _std / _ci_low /
        _ci_high columns per metric in metric_cols.
    """
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "OK"].copy()
    if ok.empty:
        raise ValueError(f"aggregate_results: {csv_path} has zero status=='OK' rows; nothing to aggregate.")

    def _agg_one_metric(group: pd.DataFrame, col: str) -> pd.Series:
        values = group[col].dropna()
        n = len(values)
        mean = values.mean()
        if n < 2:
            if n == 1:
                warnings.warn(
                    f"aggregate_results: group has n=1 for metric '{col}' — std/CI are undefined "
                    "(NaN). Expected for deterministic single-run baselines; unexpected (and a "
                    "plan ground-rule-3 violation) for a stochastic optimizer result.",
                    stacklevel=2,
                )
            std = float("nan")
            ci_low = ci_high = float("nan")
        else:
            std = values.std(ddof=1)
            sem = std / (n ** 0.5)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_low, ci_high = mean - t_crit * sem, mean + t_crit * sem
        return pd.Series({
            f"{col}_n": n, f"{col}_mean": mean, f"{col}_std": std,
            f"{col}_ci_low": ci_low, f"{col}_ci_high": ci_high,
        })

    grouped = ok.groupby(group_by, dropna=False)
    per_metric = [grouped.apply(_agg_one_metric, col, include_groups=False) for col in metric_cols]
    result = pd.concat(per_metric, axis=1).reset_index()
    return result


def current_git_commit(repo_root: Path | None = None) -> str:
    """Best-effort short git commit hash for the `git_commit` log column.

    This is auxiliary provenance metadata, not a simulated result — unlike a failed
    SCALE-Sim run, a missing/unavailable git commit degrades gracefully to a clear
    "unknown" marker rather than raising, so a run is never blocked on git state.

    Args:
        repo_root: Directory to run `git rev-parse` from; defaults to the current
            working directory.

    Returns:
        The short commit hash, or "unknown" if not in a git repo / git isn't available.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_root,
        )
    except FileNotFoundError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()
