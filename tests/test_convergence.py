"""Unit tests for common/optimizers/convergence.py — the criterion shared by Experiments 2/3/4."""
from __future__ import annotations

import numpy as np
import pytest

from common.optimizers.convergence import best_reward_so_far, convergence_iteration


def test_best_reward_so_far_is_running_max() -> None:
    assert list(best_reward_so_far([1.0, 3.0, 2.0, 5.0, 4.0])) == [1.0, 3.0, 3.0, 5.0, 5.0]


def test_convergence_iteration_detects_flat_window() -> None:
    # Flat from index 2 onward; window=3 first sees 3 identical values at index 4.
    trajectory = [0.0, 1.0, 5.0, 5.0, 5.0, 5.0]
    assert convergence_iteration(trajectory, window=3, epsilon=0.01) == 4


def test_convergence_iteration_returns_none_if_never_met() -> None:
    # Strictly increasing forever -> windowed variance never drops below a tiny epsilon.
    trajectory = list(range(20))
    assert convergence_iteration(trajectory, window=5, epsilon=0.001) is None


def test_convergence_iteration_stricter_epsilon_converges_later_or_never() -> None:
    trajectory = [0.0, 0.5, 0.9, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loose = convergence_iteration(trajectory, window=3, epsilon=1.0)
    strict = convergence_iteration(trajectory, window=3, epsilon=0.0001)
    assert loose is not None
    assert strict is None or strict > loose


def test_convergence_iteration_requires_full_window() -> None:
    # Fewer points than window -> can never converge.
    assert convergence_iteration([1.0, 1.0], window=5, epsilon=1.0) is None
