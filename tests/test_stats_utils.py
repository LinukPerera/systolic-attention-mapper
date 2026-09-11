"""Unit tests for common/stats_utils.py — added 2026-09-03 alongside Experiment 6,
which first needed a genuinely reusable one-sample/two-sample significance-test
helper (see that module's docstring for why it's a new extraction, not a literal
import from analyze_exp3.py's inline logic)."""
from __future__ import annotations

import numpy as np

from common.stats_utils import one_sample_vs_baseline, two_sample_independent


def test_one_sample_vs_baseline_identical_to_baseline_has_undefined_stats() -> None:
    # Zero variance -> no test can act on it (mirrors analyze_exp3.py's own handling).
    result = one_sample_vs_baseline(np.array([5.0, 5.0, 5.0, 5.0, 5.0]), baseline_value=5.0)
    assert result.n == 5
    assert result.mean == 5.0
    assert result.std == 0.0
    assert np.isnan(result.t_pvalue)


def test_one_sample_vs_baseline_detects_a_real_shift() -> None:
    values = np.array([10.0, 10.2, 9.8, 10.1, 9.9])
    result = one_sample_vs_baseline(values, baseline_value=5.0)
    assert result.n == 5
    assert result.mean == np.mean(values)
    assert result.t_pvalue < 0.001  # far from baseline, tiny variance -> should be very significant
    assert result.cohens_d > 0  # values are above baseline


def test_one_sample_vs_baseline_n1_is_nan_not_crash() -> None:
    result = one_sample_vs_baseline(np.array([7.0]), baseline_value=5.0)
    assert result.n == 1
    assert np.isnan(result.std)
    assert np.isnan(result.t_pvalue)


def test_two_sample_independent_detects_a_real_difference() -> None:
    a = np.array([10.0, 10.1, 9.9, 10.2, 9.8])
    b = np.array([5.0, 5.1, 4.9, 5.2, 4.8])
    result = two_sample_independent(a, b)
    assert result.n_a == 5 and result.n_b == 5
    assert result.mean_a > result.mean_b
    assert result.t_pvalue < 0.001
    assert result.u_pvalue < 0.05


def test_two_sample_independent_no_difference_is_not_significant() -> None:
    a = np.array([10.0, 9.5, 10.5, 9.8, 10.2])
    b = np.array([10.1, 9.6, 10.4, 9.9, 10.0])
    result = two_sample_independent(a, b)
    assert result.t_pvalue > 0.05
