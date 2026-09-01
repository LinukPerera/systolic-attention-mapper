"""
Unit tests for common/plotting/style.py::save_figure — regression test for a real
bug found 2026-09-01: `Path.with_suffix()` treats everything after the *last* dot
in a filename stem as a replaceable extension, so a stub containing a literal dot
(e.g. "exp2_epsilon0.1_convergence_curves") silently collapsed to the wrong name
and a second save_figure call overwrote the first.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.plotting.style import save_figure


def test_save_figure_stub_without_dots(tmp_path) -> None:
    fig, ax = plt.subplots()
    save_figure(fig, tmp_path / "myfig")
    plt.close(fig)
    assert (tmp_path / "myfig.png").exists()
    assert (tmp_path / "myfig.svg").exists()


def test_save_figure_stub_with_dot_in_name_does_not_mangle(tmp_path) -> None:
    fig, ax = plt.subplots()
    save_figure(fig, tmp_path / "exp2_epsilon0.1_convergence_curves")
    plt.close(fig)
    assert (tmp_path / "exp2_epsilon0.1_convergence_curves.png").exists()
    assert (tmp_path / "exp2_epsilon0.1_convergence_curves.svg").exists()
    # The old with_suffix() bug would have produced this instead:
    assert not (tmp_path / "exp2_epsilon0.png").exists()


def test_save_figure_two_dotted_stubs_do_not_collide(tmp_path) -> None:
    fig1, ax1 = plt.subplots()
    save_figure(fig1, tmp_path / "run_eps0.1_mean_ci")
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    save_figure(fig2, tmp_path / "run_eps0.1_individual_seeds")
    plt.close(fig2)

    assert (tmp_path / "run_eps0.1_mean_ci.png").exists()
    assert (tmp_path / "run_eps0.1_individual_seeds.png").exists()
