"""Preflight wiring of the telemetry maturity gate.

Mutation proofs:
- below-floor + signal_only -> passed=True with informational count/threshold
- below-floor + live (not signal_only) -> passed=False with FAILED message
- above-floor + either mode -> passed=True with "gate clear" message
- the check is registered in PREFLIGHT_CHECKS exactly once and is distinct
  from the copy-trading gate
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.run_live_session import (
    PREFLIGHT_CHECKS,
    PreflightContext,
    _check_copy_trading_accounts,
    _check_telemetry_maturity,
)
from trading_app.live import session_orchestrator
from trading_app.live.telemetry_maturity import MIN_TELEMETRY_TRADING_DAYS


def _write_n_distinct_days(signals_dir: Path, n: int, instrument: str = "MNQ") -> None:
    """Synthesize n distinct trading_day SESSION_START records for instrument."""
    signals_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        day = (datetime(2026, 5, 1, tzinfo=UTC) + timedelta(days=i)).date().isoformat()
        rec = {
            "ts": f"{day}T20:00:00+00:00",
            "instrument": instrument,
            "type": "SESSION_START",
            "contract": f"CON.F.US.{instrument}.M26",
            "mode": "signal_only",
        }
        (signals_dir / f"live_signals_{day}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic_signals_dir(tmp_path, monkeypatch):
    """Redirect SessionOrchestrator.SIGNALS_DIR to tmp_path."""
    monkeypatch.setattr(session_orchestrator.SessionOrchestrator, "SIGNALS_DIR", tmp_path)
    return tmp_path


def _ctx(
    instrument: str = "MNQ",
    signal_only: bool = True,
    demo: bool = False,
    profile_id: str | None = None,
) -> PreflightContext:
    return PreflightContext(
        instrument=instrument,
        broker_name="test",
        demo=demo,
        portfolio=None,
        signal_only=signal_only,
        profile_id=profile_id,
    )


def test_below_floor_signal_only_returns_passed_with_count(synthetic_signals_dir):
    """Signal-only must NEVER block on telemetry below floor -- it's the path that accumulates it."""
    _write_n_distinct_days(synthetic_signals_dir, n=MIN_TELEMETRY_TRADING_DAYS - 1)
    result = _check_telemetry_maturity(_ctx(signal_only=True))
    assert result.passed is True, "signal-only path must surface, not block"
    assert "signal-only" in result.message
    assert "29/30" in result.message
    assert "MNQ" in result.message
    assert "auto-clears at 30" in result.message


def test_below_floor_live_returns_failed(synthetic_signals_dir):
    """Capital-touching mode (not signal_only) must BLOCK below the floor."""
    _write_n_distinct_days(synthetic_signals_dir, n=MIN_TELEMETRY_TRADING_DAYS - 1)
    result = _check_telemetry_maturity(_ctx(signal_only=False))
    assert result.passed is False, "live/demo must refuse to launch below floor"
    assert "FAILED" in result.message
    assert "UNVERIFIED_INSUFFICIENT_TELEMETRY" in result.message
    assert "29/30" in result.message
    assert "--signal-only" in result.message, "message must direct operator to the path that accumulates telemetry"


def test_above_floor_passes_in_both_modes(synthetic_signals_dir):
    """At/above floor, both signal_only and live paths pass with the same gate-clear message."""
    _write_n_distinct_days(synthetic_signals_dir, n=MIN_TELEMETRY_TRADING_DAYS)
    for signal_only in (True, False):
        result = _check_telemetry_maturity(_ctx(signal_only=signal_only))
        assert result.passed is True
        assert "gate clear" in result.message
        assert "30/30" in result.message


def test_check_is_registered_exactly_once_and_distinct_from_copy_gate():
    """Doctrine: maturity check is registered, distinct from copy-trading gate, not duplicated."""
    occurrences = [c for c in PREFLIGHT_CHECKS if c is _check_telemetry_maturity]
    assert len(occurrences) == 1, "_check_telemetry_maturity must be registered exactly once"
    assert _check_telemetry_maturity is not _check_copy_trading_accounts, (
        "maturity gate must NOT be conflated with copy-trading gate"
    )
    # Both must be in the registry; positional adjacency is desired but not asserted (refactor-tolerant).
    assert _check_copy_trading_accounts in PREFLIGHT_CHECKS


def test_unrecognized_instrument_defaults_to_mnq(synthetic_signals_dir):
    """When ctx.instrument is not in ACTIVE_ORB_INSTRUMENTS (e.g. 'ALL'), gate evaluates against MNQ."""
    _write_n_distinct_days(synthetic_signals_dir, n=MIN_TELEMETRY_TRADING_DAYS)
    result = _check_telemetry_maturity(_ctx(instrument="ALL", signal_only=True))
    assert result.passed is True
    assert "MNQ" in result.message, "fallback to MNQ when instrument sentinel is unrecognized"


# ── 2026-05-18 FAIL→WARN doctrine-debt demotion ────────────────────────────
# The 30-day live-uptime floor was never canonical doctrine. Below-floor
# verdicts are advisory WARN for demo / Express-Funded prop live; only
# real-capital live (is_express_funded=False, unknown profile, or no
# profile) preserves the original FAIL. See telemetry_maturity.py module
# docstring "DOCTRINE NOTE" for the full rationale.


def test_below_floor_demo_returns_warn(synthetic_signals_dir):
    """Demo mode below floor must surface as advisory WARN, never FAIL — no real capital at risk."""
    _write_n_distinct_days(synthetic_signals_dir, n=MIN_TELEMETRY_TRADING_DAYS - 1)
    result = _check_telemetry_maturity(_ctx(signal_only=False, demo=True))
    assert result.passed is True, "demo must not block on advisory gate"
    assert result.message.startswith("WARN:"), "must emit WARN status for dashboard parser"
    assert "demo" in result.message
    assert "29/30" in result.message


def test_below_floor_live_xfa_profile_returns_warn(synthetic_signals_dir):
    """--live + Express-Funded prop profile (is_express_funded=True) is advisory WARN, not FAIL."""
    _write_n_distinct_days(synthetic_signals_dir, n=MIN_TELEMETRY_TRADING_DAYS - 1)
    # topstep_50k_mnq_auto is the active deployment profile; per
    # prop_profiles.AccountProfile default at line 107, is_express_funded=True.
    result = _check_telemetry_maturity(
        _ctx(signal_only=False, demo=False, profile_id="topstep_50k_mnq_auto")
    )
    assert result.passed is True, "Express-Funded prop must not block on advisory gate"
    assert result.message.startswith("WARN:"), "must emit WARN status for dashboard parser"
    assert "Express-Funded prop" in result.message
    assert "29/30" in result.message


def test_below_floor_live_unknown_profile_returns_failed(synthetic_signals_dir):
    """--live + unknown profile_id stays FAIL (conservative default; can't verify XFA shape)."""
    _write_n_distinct_days(synthetic_signals_dir, n=MIN_TELEMETRY_TRADING_DAYS - 1)
    result = _check_telemetry_maturity(
        _ctx(signal_only=False, demo=False, profile_id="not_a_real_profile_xyz")
    )
    assert result.passed is False, "unknown profile on live mode must FAIL conservatively"
    assert "FAILED" in result.message
    assert "UNVERIFIED_INSUFFICIENT_TELEMETRY" in result.message


def test_below_floor_live_real_capital_profile_returns_failed(synthetic_signals_dir):
    """--live + known profile with is_express_funded=False stays FAIL (real capital at risk)."""
    # Build a synthetic real-capital profile inline to avoid coupling the test
    # to a specific profile dict entry that may be edited.
    from trading_app.prop_profiles import ACCOUNT_PROFILES, AccountProfile

    fake_id = "_test_real_capital_profile_below_floor"
    ACCOUNT_PROFILES[fake_id] = AccountProfile(
        profile_id=fake_id,
        firm="self",
        account_size=50_000,
        is_express_funded=False,
    )
    try:
        _write_n_distinct_days(synthetic_signals_dir, n=MIN_TELEMETRY_TRADING_DAYS - 1)
        result = _check_telemetry_maturity(
            _ctx(signal_only=False, demo=False, profile_id=fake_id)
        )
        assert result.passed is False, "real-capital live below floor must FAIL"
        assert "FAILED" in result.message
        assert "UNVERIFIED_INSUFFICIENT_TELEMETRY" in result.message
    finally:
        ACCOUNT_PROFILES.pop(fake_id, None)
