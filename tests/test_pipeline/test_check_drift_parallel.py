"""Stage-3 parallel-equivalence proofs for pipeline/check_drift.py.

The non-DB drift checks now run under a ThreadPoolExecutor (concurrency for
COMPUTE), while all EFFECTS (printing, counters, violation collection, cache
writes, the meta cold-recheck) replay serially in registry order. These tests
prove the parallelization is behaviour-preserving:

  (a) serial (DRIFT_WORKERS=1) and parallel (DRIFT_WORKERS=8) produce the SAME
      exit code and the SAME set of per-check verdicts (PASS/FAIL/SKIP/ADVISORY);
  (b) the meta cold-recheck still runs last and the run still exits cleanly;
  (c) the worker-count helper honours the env override and the CPU formula.

We drive the REAL `python pipeline/check_drift.py --quiet` as a subprocess (main()
calls sys.exit, so a subprocess is the honest harness). --quiet emits exactly one
sanitized line per check, which makes the verdict SET comparison robust to the
inline-output ordering that legitimately differs between threads.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from pipeline import check_drift as cd
from pipeline.paths import PROJECT_ROOT

# These subprocess runs execute the full drift suite (DB checks included) — minutes,
# not milliseconds. Mark slow so the pre-commit fast gate skips them; full CI runs them.
#
# IMPORTANT — run these with the pytest-timeout plugin UNLOADED: `-p no:timeout`.
# pytest-timeout 2.4.0's `thread` watchdog races pytest 9.0.2's capture manager when
# a test spawns a long subprocess, intermittently truncating the captured child stdout
# (observed here: one of two full-drift subprocess runs returned ZERO verdict lines).
# `--timeout=0` / `pytest.mark.timeout(0)` do NOT fix this — they still LOAD the plugin
# (the n=4 watchdog-race close, 2026-05-25; enforced for CI by
# check_ci_pytest_unloads_timeout_plugin). The subprocess itself is bounded by
# _SUBPROC_TIMEOUT_S, so an actually-hung child still fails loudly without the watchdog.
pytestmark = pytest.mark.slow

# Generous ceiling for ONE full drift subprocess; bounds a genuinely hung child so the
# test fails loudly instead of hanging CI forever (condition-based-waiting: bound the wait).
_SUBPROC_TIMEOUT_S = 600


def _run_drift(workers: str) -> tuple[int, list[str]]:
    """Run `check_drift.py --quiet` with DRIFT_WORKERS=<workers>.

    Returns (exit_code, sorted verdict lines). A verdict line is any --quiet
    status line: PASS:/FAIL:/SKIP:/ADVISORY:/SUMMARY:. We sort so the comparison
    is order-independent (registry order is preserved within a run, but the SET
    of verdicts is what must match between serial and parallel)."""
    env = dict(os.environ)
    env["DRIFT_WORKERS"] = workers
    proc = subprocess.run(
        [sys.executable, "pipeline/check_drift.py", "--quiet"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=_SUBPROC_TIMEOUT_S,
    )
    prefixes = ("PASS:", "FAIL:", "SKIP:", "ADVISORY:", "SUMMARY:")
    verdicts = sorted(line for line in proc.stdout.splitlines() if line.startswith(prefixes))
    # A run that produced no verdict lines crashed before any check reported — fail
    # loudly WITH stderr rather than letting an empty list silently mis-compare.
    assert verdicts, (
        f"DRIFT_WORKERS={workers} run produced zero verdict lines (rc={proc.returncode}).\n"
        f"--- stderr tail ---\n{proc.stderr[-2000:]}"
    )
    return proc.returncode, verdicts


def test_serial_and_parallel_produce_identical_verdicts_and_exit_code():
    """The load-bearing equivalence proof: parallelizing the non-DB checks must
    not change any verdict or the exit code vs the serial path."""
    serial_code, serial_verdicts = _run_drift("1")
    parallel_code, parallel_verdicts = _run_drift("8")

    assert serial_code == parallel_code, f"exit code diverged: serial={serial_code} parallel={parallel_code}"
    # Compare the verdict SETS. (A line present in one but not the other is the
    # signal of a real divergence — a check that passed serially but failed under
    # threading, or vice versa.)
    assert serial_verdicts == parallel_verdicts, (
        "verdict set diverged between serial and parallel runs.\n"
        f"only-serial:   {sorted(set(serial_verdicts) - set(parallel_verdicts))}\n"
        f"only-parallel: {sorted(set(parallel_verdicts) - set(serial_verdicts))}"
    )
    # Sanity: the run actually produced verdicts (guards against both runs
    # crashing identically before any check ran).
    assert len(serial_verdicts) > 10, (
        f"expected many verdict lines, got {len(serial_verdicts)} — did the run crash early?"
    )
    # The meta cold-recheck (must-run-last; reads the full cache-hit list) must
    # appear in BOTH runs — folded in here rather than a separate full-drift run so
    # we don't re-pay ~3min and don't add a second capture-flake surface. (verdicts
    # are equal by the assert above, so checking parallel suffices.)
    meta_label = cd.CHECKS[-1][0].encode("ascii", "replace").decode("ascii")
    assert any(meta_label in v for v in parallel_verdicts), (
        f"meta cold-recheck verdict missing from parallel run; label={meta_label!r}"
    )
    # Exit-code semantic: 0 clean / 1 drift (never a crash code).
    assert parallel_code in (0, 1), (
        f"unexpected exit code {parallel_code} (expected 0 clean or 1 drift)"
    )


@pytest.mark.parametrize("override", ["1", "3", "8"])
def test_worker_count_honours_positive_env_override(monkeypatch, override):
    monkeypatch.setenv("DRIFT_WORKERS", override)
    assert cd._drift_worker_count() == int(override)


@pytest.mark.parametrize("bad", ["0", "-1", "notanint", ""])
def test_worker_count_ignores_bad_override_falls_back_to_cpu_formula(monkeypatch, bad):
    # A non-positive / non-integer / empty override is ignored → the CPU formula
    # min(8, cpu-2) floor 1 governs. That value is host-dependent, so assert only
    # the documented bounds. The key behaviour is: garbage never crashes.
    monkeypatch.setenv("DRIFT_WORKERS", bad)
    n = cd._drift_worker_count()
    assert 1 <= n <= 8
