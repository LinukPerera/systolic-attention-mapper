"""
Unit tests for common/simulator_interface.py's Docker-daemon-reachability check —
added 2026-09-02 alongside memoizing it (was a real ~0.6s `docker info` subprocess
call paid on every single run_scale_sim call; now checked once per process). These
tests monkeypatch subprocess.run/shutil.which so they run without a real Docker
daemon; they check the memoization/reset behavior itself, not real SCALE-Sim output
(that's covered by the real Docker-backed experiment runs, per plan ground rule 1 —
never substitute a mocked result for a reported one).
"""
from __future__ import annotations

import subprocess

import pytest

import common.simulator_interface as sim_iface


@pytest.fixture(autouse=True)
def _reset_daemon_check_cache():
    """_daemon_checked is process-global state (see its own module comment for
    why) -- reset it before and after every test so tests don't leak state into
    each other or into the rest of the suite."""
    sim_iface._daemon_checked = False
    yield
    sim_iface._daemon_checked = False


def test_ensure_docker_daemon_reachable_raises_if_docker_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sim_iface.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="not found on PATH"):
        sim_iface._ensure_docker_daemon_reachable()


def test_ensure_docker_daemon_reachable_raises_if_daemon_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sim_iface.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        sim_iface.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], returncode=1, stdout="", stderr="daemon down"),
    )
    with pytest.raises(RuntimeError, match="not reachable"):
        sim_iface._ensure_docker_daemon_reachable()


def test_ensure_docker_daemon_reachable_only_calls_subprocess_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The actual point of memoizing: a successful check must not re-invoke
    `docker info` on a second call within the same process."""
    call_count = {"n": 0}

    def fake_run(*a, **k):
        call_count["n"] += 1
        return subprocess.CompletedProcess(a[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sim_iface.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(sim_iface.subprocess, "run", fake_run)

    sim_iface._ensure_docker_daemon_reachable()
    sim_iface._ensure_docker_daemon_reachable()
    sim_iface._ensure_docker_daemon_reachable()

    assert call_count["n"] == 1


def test_ensure_docker_daemon_reachable_rechecks_after_explicit_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors run_scale_sim's own reset-on-daemon-connectivity-failure path
    (see its `_daemon_checked = False` line): after that reset, the next call
    must genuinely re-check, not trust the stale cached success."""
    call_count = {"n": 0}

    def fake_run(*a, **k):
        call_count["n"] += 1
        return subprocess.CompletedProcess(a[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sim_iface.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(sim_iface.subprocess, "run", fake_run)

    sim_iface._ensure_docker_daemon_reachable()
    assert call_count["n"] == 1

    sim_iface._daemon_checked = False  # what run_scale_sim does on a daemon-connectivity failure
    sim_iface._ensure_docker_daemon_reachable()
    assert call_count["n"] == 2
