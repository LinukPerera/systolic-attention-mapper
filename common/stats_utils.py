"""
stats_utils.py — shared significance-test helpers, extracted 2026-09-03 for
Experiment 6 so it can genuinely reuse the same one-sample-vs-fixed-baseline
test pattern `analyze_exp3.py` already established (t-test + Wilcoxon +
Cohen's d), per `docs/NEXT_INSTRUCTIONS.md`'s explicit instruction to "reuse
that function, do not rewrite it." `analyze_exp3.py`'s own version of this
logic is inline within its `main()`, not a standalone function — rather than
literally importing from there (which would mean either duplicating the same
few lines anyway or refactoring already-finalized, cited analysis code for a
downstream experiment's convenience), this module extracts the same tested
pattern once, here, for both `analyze_exp3.py`-style comparisons and the new
independent two-sample comparison Experiment 6 also needs. `analyze_exp3.py`
itself is left unmodified (its own numbers are already verified and cited in
`exp3_stats.md`) — only new code (`analyze_exp6.py`) uses this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class OneSampleResult:
    """One sample's values vs. one fixed (deterministic) baseline value —
    exactly `analyze_exp3.py`'s own comparison pattern (optimizer seeds vs. a
    fixed static-baseline utilization)."""
    n: int
    mean: float
    std: float
    t_statistic: float
    t_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    cohens_d: float


def one_sample_vs_baseline(values: np.ndarray, baseline_value: float) -> OneSampleResult:
    """One-sample t-test + Wilcoxon signed-rank of `values` against a fixed
    `baseline_value` — same test choice and same degenerate-variance handling
    as `analyze_exp3.py`'s inline version (kept identical deliberately, for a
    directly comparable methodology)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean, std = float(values.mean()), (float(values.std(ddof=1)) if n > 1 else float("nan"))
    if n > 1 and np.std(values) > 0:
        t_stat, t_p = stats.ttest_1samp(values, popmean=baseline_value)
        try:
            w_stat, w_p = stats.wilcoxon(values - baseline_value)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        cohens_d = (mean - baseline_value) / std
    else:
        t_stat = t_p = w_stat = w_p = cohens_d = float("nan")
    return OneSampleResult(
        n=n, mean=mean, std=std, t_statistic=float(t_stat), t_pvalue=float(t_p),
        wilcoxon_statistic=float(w_stat), wilcoxon_pvalue=float(w_p), cohens_d=float(cohens_d),
    )


@dataclass(frozen=True)
class TwoSampleResult:
    """Two independent (unpaired) samples — e.g. zero-shot transfer's 5 values
    vs. fresh-ceiling training's 5 values: two different distributions (a
    fixed transferred action's simulation determinism vs. a freshly-trained
    policy's seed variance), not naturally paired by seed the way this
    project's other seed-matched comparisons are (`docs/NEXT_INSTRUCTIONS.md`
    §4's explicit instruction to use an independent, not paired, test here)."""
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    t_statistic: float
    t_pvalue: float
    u_statistic: float
    u_pvalue: float


def two_sample_independent(values_a: np.ndarray, values_b: np.ndarray) -> TwoSampleResult:
    """Welch's t-test (unequal-variance, `equal_var=False`) + Mann-Whitney U —
    the independent-samples analogue of `one_sample_vs_baseline`'s paired-style
    tests, for two genuinely different distributions rather than one sample vs.
    a fixed constant."""
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    t_stat, t_p = stats.ttest_ind(values_a, values_b, equal_var=False)
    try:
        u_stat, u_p = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    except ValueError:
        u_stat, u_p = float("nan"), float("nan")
    return TwoSampleResult(
        n_a=len(values_a), n_b=len(values_b), mean_a=float(values_a.mean()), mean_b=float(values_b.mean()),
        t_statistic=float(t_stat), t_pvalue=float(t_p), u_statistic=float(u_stat), u_pvalue=float(u_p),
    )
