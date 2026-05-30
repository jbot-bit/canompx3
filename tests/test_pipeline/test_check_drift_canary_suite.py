"""Tests for ``check_canary_suite_green`` (canary harness deliverable #4, BLOCKING).

The drift gate runs every Tier-1 canary against its trap and returns a violation
for each guard that FAILED to fire (``fired=False``) — a contamination class the
pipeline can no longer reject. Empty == all guards caught their traps.

Per institutional-rigor.md §11 ("never trust a check's PASS label — verify by
known-violation injection"), the load-bearing test monkeypatches a guard to a
no-op and asserts the gate fires. A gate that cannot be made to fail proves
nothing.
"""

from __future__ import annotations

import pytest

from pipeline import check_drift
from scripts.tests import canary_suite


def test_clean_tree_passes() -> None:
    """With real guards, every canary fires → the blocking check returns []."""
    assert check_drift.check_canary_suite_green() == []


def test_dead_holdout_guard_makes_gate_fire(monkeypatch) -> None:
    """Known-violation injection: neuter enforce_holdout_date → gate fires.

    A no-op holdout guard never raises on a post-sacred date, so canary 8 can no
    longer catch the 2026-contamination trap. The blocking check MUST surface it.
    """

    def _noop_enforce(holdout_date, override_token=None):
        return holdout_date  # never raises — guard defeated

    monkeypatch.setattr(canary_suite, "enforce_holdout_date", _noop_enforce)
    violations = check_drift.check_canary_suite_green()
    assert violations, "a defeated holdout guard must make the gate fire"
    assert any("holdout_2026_contamination" in v for v in violations), violations


def test_dead_session_guard_makes_gate_fire(monkeypatch) -> None:
    """Known-violation injection: neuter is_feature_safe → gate fires.

    An always-'safe' session guard can no longer catch the look-ahead /
    session-relabel traps (canaries 3, 4, 5), so the gate must surface ≥1 miss.
    """
    monkeypatch.setattr(canary_suite, "is_feature_safe", lambda *a, **k: True)
    violations = check_drift.check_canary_suite_green()
    assert violations, "a defeated session guard must make the gate fire"


def test_restored_guard_passes_again(monkeypatch) -> None:
    """After a temporary defeat, restoring the guard returns the gate to green.

    Proves the gate tracks the guard's live behaviour, not a cached verdict.
    """
    # Defeat, then undo within the same test via monkeypatch context teardown.
    monkeypatch.setattr(canary_suite, "enforce_holdout_date", lambda d, override_token=None: d)
    assert check_drift.check_canary_suite_green()  # fires while defeated
    monkeypatch.undo()
    assert check_drift.check_canary_suite_green() == []  # green once restored


def test_violation_strings_name_the_guard_and_signature() -> None:
    """When the gate fires, each message must name the canary, guard, signature.

    Constructed by injecting a single dead guard and inspecting the message
    shape (so an operator reading a blocked commit knows what regressed).
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(canary_suite, "enforce_holdout_date", lambda d, override_token=None: d)
        violations = check_drift.check_canary_suite_green()
    assert violations
    msg = next(v for v in violations if "holdout_2026_contamination" in v)
    assert "guard FAILED to fire" in msg
    assert "Expected signature" in msg
    assert "canary_suite.py" in msg


def test_check_is_registered_blocking_and_not_requires_db() -> None:
    """The check is wired into CHECKS as BLOCKING (advisory=False), no DB."""
    label = "Canary contamination suite green (every guard catches its trap)"
    entry = next((c for c in check_drift.CHECKS if c[0] == label), None)
    assert entry is not None, "check_canary_suite_green not registered in CHECKS"
    _label, fn, is_advisory, requires_db = entry
    assert fn is check_drift.check_canary_suite_green
    assert is_advisory is False, "must be BLOCKING, not advisory"
    assert requires_db is False, "Tier-1 canaries are pure-function — no DB"


def test_check_dep_entry_present_for_caching() -> None:
    """The check has a complete CHECK_DEPS entry (cache-eligible)."""
    label = "Canary contamination suite green (every guard catches its trap)"
    assert label in check_drift.CHECK_DEPS
    deps = check_drift.CHECK_DEPS[label]
    # The suite + the guard modules it delegates to must all be declared.
    for required in (
        "scripts/tests/canary_suite.py",
        "research/oos_power.py",
        "pipeline/session_guard.py",
        "trading_app/holdout_policy.py",
        "trading_app/config.py",
    ):
        assert required in deps, f"{required} missing from CHECK_DEPS"


def test_meta_recheck_remains_last() -> None:
    """The blocking canary check must NOT displace the cache cold-recheck tail."""
    assert check_drift.CHECKS[-1][1] is check_drift.check_drift_cache_meta_recheck
