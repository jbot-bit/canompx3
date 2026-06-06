"""Tests for ``check_active_profiles_survival_state_current`` (Row 05, ADVISORY).

The check re-couples the Criterion-11 survival GATE to the active-profile
REGISTRY: an ``active=True`` prop profile whose persisted survival state is stale
(DB advanced) or regressed (op_pass below floor / strict DD breach) is surfaced as
an ADVISORY. It is non-blocking by design — the arm-time preflight is the real
fail-closed capital gate; this is the visibility layer (results/05.md: "deploy-
readiness only ... no blind path to live execution").

Per institutional-rigor.md §11 (never trust a PASS label — verify by injection),
the load-bearing tests monkeypatch the gate verdict and assert the advisory fires,
and prove the check NEVER returns a blocking violation regardless of gate state.
"""

from __future__ import annotations

import pytest

from pipeline import check_drift


def test_returns_empty_when_no_active_profiles(monkeypatch, capsys) -> None:
    """No active prop profile (nothing armed) → silent, no violation."""
    monkeypatch.setattr("trading_app.prop_profiles.get_active_profile_ids", lambda *a, **k: [])
    assert check_drift.check_active_profiles_survival_state_current() == []
    assert "ADVISORY" not in capsys.readouterr().out


def test_passing_gate_is_silent(monkeypatch, capsys) -> None:
    """A current, passing survival proof → no advisory, no violation."""
    monkeypatch.setattr("trading_app.prop_profiles.get_active_profile_ids", lambda *a, **k: ["topstep_50k_mnq_auto"])
    monkeypatch.setattr(
        "trading_app.account_survival.check_survival_report_gate",
        lambda profile_id, **k: (True, "Criterion 11 pass: operational 99.8%"),
    )
    assert check_drift.check_active_profiles_survival_state_current() == []
    assert "ADVISORY" not in capsys.readouterr().out


def test_stale_state_fires_advisory_but_never_blocks(monkeypatch, capsys) -> None:
    """A stale (db-identity-mismatch) state → ADVISORY printed, [] returned.

    This is the real-world state observed 2026-06-06: the active profile's
    survival proof went stale when gold.db advanced. The check MUST surface it
    (visibility) WITHOUT blocking the commit (daily ingest must not fail drift).
    """
    monkeypatch.setattr("trading_app.prop_profiles.get_active_profile_ids", lambda *a, **k: ["topstep_50k_mnq_auto"])
    monkeypatch.setattr(
        "trading_app.account_survival.check_survival_report_gate",
        lambda profile_id, **k: (False, "BLOCKED: Criterion 11 state db identity mismatch. Re-run account survival."),
    )
    violations = check_drift.check_active_profiles_survival_state_current()
    assert violations == [], "advisory check must NEVER return a blocking violation"
    out = capsys.readouterr().out
    assert "ADVISORY" in out
    assert "topstep_50k_mnq_auto" in out
    assert "db identity mismatch" in out  # the gate's own discriminating message survives verbatim


def test_regression_fires_advisory_with_verbatim_message(monkeypatch, capsys) -> None:
    """A genuine regression (op-pass below floor) surfaces the gate message verbatim."""
    monkeypatch.setattr("trading_app.prop_profiles.get_active_profile_ids", lambda *a, **k: ["topstep_50k_mnq_auto"])
    monkeypatch.setattr(
        "trading_app.account_survival.check_survival_report_gate",
        lambda profile_id, **k: (False, "BLOCKED: Criterion 11 operational pass 90.0% < 95%"),
    )
    assert check_drift.check_active_profiles_survival_state_current() == []
    out = capsys.readouterr().out
    assert "operational pass 90.0% < 95%" in out


def test_gate_exception_is_advisory_not_silent_pass(monkeypatch, capsys) -> None:
    """If the gate raises, surface an ADVISORY — never a silent pass, never a block."""
    monkeypatch.setattr("trading_app.prop_profiles.get_active_profile_ids", lambda *a, **k: ["topstep_50k_mnq_auto"])

    def _boom(profile_id, **k):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("trading_app.account_survival.check_survival_report_gate", _boom)
    assert check_drift.check_active_profiles_survival_state_current() == []
    out = capsys.readouterr().out
    assert "ADVISORY" in out
    assert "could not be evaluated" in out


def test_import_failure_fails_open(monkeypatch, capsys) -> None:
    """An import failure (e.g. broken env) is an advisory, not a hard block."""
    import builtins

    real_import = builtins.__import__

    def _fail_survival_import(name, *args, **kwargs):
        if name == "trading_app.account_survival":
            raise ImportError("simulated broken environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_survival_import)
    assert check_drift.check_active_profiles_survival_state_current() == []
    assert "ADVISORY" in capsys.readouterr().out


def test_registered_as_advisory_requires_db() -> None:
    """The check must be wired ADVISORY (is_advisory=True) + requires_db=True.

    is_advisory=True so daily DB advance never blocks a commit; requires_db=True
    so it is correctly skipped (not falsely passed) when the DB is unavailable.
    """
    matches = [
        entry for entry in check_drift.CHECKS if entry[1] is check_drift.check_active_profiles_survival_state_current
    ]
    assert len(matches) == 1, "check must be registered exactly once in CHECKS"
    _desc, _fn, is_advisory, requires_db = matches[0]
    assert is_advisory is True, "survival-state check must be ADVISORY (non-blocking)"
    assert requires_db is True, "survival-state check must be requires_db=True"
