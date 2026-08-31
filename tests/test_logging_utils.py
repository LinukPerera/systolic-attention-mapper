"""
Unit tests for common/logging_utils.py — the append-only CSV schema and aggregation
every experiment shares (plan §4).
"""
from __future__ import annotations

import subprocess
import warnings

import pandas as pd
import pytest

from common.logging_utils import RAW_COLUMNS, aggregate_results, append_run_row, current_git_commit


def _row(**overrides) -> dict:
    base = {c: None for c in RAW_COLUMNS}
    base.update({
        "run_id": "r1", "experiment": "exp_test", "timestamp": "2026-01-01T00:00:00",
        "git_commit": "abc123", "optimizer": "baseline",
        "array_rows": 16, "array_cols": 16, "sram_kb": 64, "dataflow_mode": "WS",
        "seq_len": 512, "hidden_dim": 768, "num_heads": 12, "seed": 0, "iteration": 0,
        "pe_utilization_pct": 80.0, "exec_cycles": 1000, "sram_dram_traffic_bytes": 2048,
        "edp": 2_000_000.0, "wall_clock_s": 0.5, "status": "OK", "notes": "",
    })
    base.update(overrides)
    return base


def test_append_run_row_creates_header_then_appends(tmp_path) -> None:
    csv_path = tmp_path / "exp_test.csv"
    append_run_row(csv_path, _row(run_id="r1", seed=0))
    append_run_row(csv_path, _row(run_id="r2", seed=1))

    df = pd.read_csv(csv_path)
    assert list(df.columns) == RAW_COLUMNS
    assert len(df) == 2
    assert list(df["run_id"]) == ["r1", "r2"]


def test_append_run_row_rejects_missing_column(tmp_path) -> None:
    csv_path = tmp_path / "exp_test.csv"
    bad_row = _row()
    del bad_row["edp"]
    with pytest.raises(ValueError):
        append_run_row(csv_path, bad_row)


def test_append_run_row_rejects_invalid_status(tmp_path) -> None:
    csv_path = tmp_path / "exp_test.csv"
    with pytest.raises(ValueError):
        append_run_row(csv_path, _row(status="MAYBE"))


def test_aggregate_results_computes_mean_std_ci_across_seeds(tmp_path) -> None:
    csv_path = tmp_path / "exp_test.csv"
    for seed, util in enumerate([78.0, 80.0, 82.0, 79.0, 81.0]):
        append_run_row(csv_path, _row(run_id=f"r{seed}", seed=seed, pe_utilization_pct=util))

    agg = aggregate_results(csv_path, group_by=["optimizer", "dataflow_mode"], metric_cols=["pe_utilization_pct"])
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["pe_utilization_pct_n"] == 5
    assert row["pe_utilization_pct_mean"] == pytest.approx(80.0)
    assert row["pe_utilization_pct_std"] > 0
    assert row["pe_utilization_pct_ci_low"] < row["pe_utilization_pct_mean"] < row["pe_utilization_pct_ci_high"]


def test_aggregate_results_excludes_failed_rows(tmp_path) -> None:
    csv_path = tmp_path / "exp_test.csv"
    append_run_row(csv_path, _row(run_id="r_ok", seed=0, status="OK", pe_utilization_pct=80.0))
    append_run_row(csv_path, _row(
        run_id="r_fail", seed=1, status="FAILED",
        pe_utilization_pct=None, exec_cycles=None, sram_dram_traffic_bytes=None, edp=None,
    ))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # n=1 after excluding the FAILED row — expected here
        agg = aggregate_results(csv_path, group_by=["optimizer"], metric_cols=["pe_utilization_pct"])
    assert agg.iloc[0]["pe_utilization_pct_n"] == 1


def test_aggregate_results_n1_group_warns_and_gives_nan_std(tmp_path) -> None:
    csv_path = tmp_path / "exp_test.csv"
    append_run_row(csv_path, _row(run_id="r1", seed=0))

    with pytest.warns(UserWarning, match="n=1"):
        agg = aggregate_results(csv_path, group_by=["optimizer"], metric_cols=["pe_utilization_pct"])
    assert agg.iloc[0]["pe_utilization_pct_n"] == 1
    assert pd.isna(agg.iloc[0]["pe_utilization_pct_std"])


def _run_git(args: list[str], cwd) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_current_git_commit_clean_tree_has_no_dirty_suffix(tmp_path) -> None:
    _run_git(["init", "-q"], tmp_path)
    _run_git(["config", "user.email", "test@example.com"], tmp_path)
    _run_git(["config", "user.name", "Test"], tmp_path)
    (tmp_path / "f.txt").write_text("hello")
    _run_git(["add", "."], tmp_path)
    _run_git(["commit", "-q", "-m", "initial"], tmp_path)

    commit = current_git_commit(tmp_path)
    assert commit != "unknown"
    assert not commit.endswith("-dirty")


def test_current_git_commit_dirty_tree_gets_suffix(tmp_path) -> None:
    _run_git(["init", "-q"], tmp_path)
    _run_git(["config", "user.email", "test@example.com"], tmp_path)
    _run_git(["config", "user.name", "Test"], tmp_path)
    (tmp_path / "f.txt").write_text("hello")
    _run_git(["add", "."], tmp_path)
    _run_git(["commit", "-q", "-m", "initial"], tmp_path)
    (tmp_path / "f.txt").write_text("changed, not committed")

    commit = current_git_commit(tmp_path)
    assert commit.endswith("-dirty")


def test_current_git_commit_not_a_repo_returns_unknown(tmp_path) -> None:
    assert current_git_commit(tmp_path) == "unknown"
