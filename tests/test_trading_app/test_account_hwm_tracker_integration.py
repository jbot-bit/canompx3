"""Integration tests for HWM persistence integrity hardening — Stage 2.

Scenarios from design v3 § 10 cross-stage validation. Each scenario exercises
multiple primitives end-to-end (B1-B9) against a realistic state directory
and notify-callback. Lands with Stage 2 commit so CI gets coverage from day
one rather than deferring to Stage 4.

Stage 3 + Stage 4 will append additional scenarios as they land.

Parent design: docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md (v3 § 10)
Stage doc: docs/runtime/stages/hwm-stage2-tracker-integrity.md
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_app.account_hwm_tracker import (
    AccountHWMTracker,
    state_file_age_days,
)


@pytest.fixture
def state_dir(tmp_path):
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed(state_dir: Path, account_id: str, *, age_seconds: float, halted: bool = False) -> Path:
    last_ts = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
    data = {
        "account_id": account_id,
        "firm": "topstep",
        "hwm_dollars": 50000.0,
        "hwm_timestamp": last_ts,
        "last_equity": 50000.0,
        "last_equity_timestamp": last_ts,
        "halt_triggered": halted,
        "halt_timestamp": None,
        "halt_reason": "",
        "consecutive_poll_failures": 0,
        "hwm_frozen": False,
        "session_log": [],
    }
    path = state_dir / f"account_hwm_{account_id}.json"
    path.write_text(json.dumps(data))
    return path


def test_scenario_1_19_day_audit_case_warns_no_raise_callback_visibility(state_dir, caplog):
    """Scenario 1 — mirrors the original audit case (account_hwm_20092334.json, 19 days old).

    Without callback: log.warning emitted, no exception, no notify.
    With callback: log.warning emitted, exactly one notify dispatch.
    """
    _seed(state_dir, "20092334", age_seconds=19 * 86400)

    # 1a: no callback
    with caplog.at_level(logging.WARNING, logger="trading_app.account_hwm_tracker"):
        t = AccountHWMTracker("20092334", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
    assert t._hwm == 50000.0
    stale_warns = [r for r in caplog.records if "old (account" in r.message or "days old" in r.message]
    assert stale_warns, "Expected stale-state log.warning"

    # 1b: with callback
    caplog.clear()
    calls: list[str] = []
    with caplog.at_level(logging.WARNING, logger="trading_app.account_hwm_tracker"):
        AccountHWMTracker(
            "20092334",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
    stale_notifies = [c for c in calls if "STATE STALE" in c]
    assert len(stale_notifies) == 1, f"Expected one stale notify; got {calls!r}"


def test_scenario_2_30_days_plus_1s_raises_with_repair_recipe(state_dir):
    """Scenario 2 — fail-closed boundary with full diagnostic message."""
    _seed(state_dir, "STALE30P", age_seconds=30 * 86400 + 1)
    calls: list[str] = []
    with pytest.raises(RuntimeError, match=r"STALE_STATE_FAIL") as exc_info:
        AccountHWMTracker(
            "STALE30P",
            "topstep",
            dd_limit_dollars=2000.0,
            state_dir=state_dir,
            notify_callback=calls.append,
        )
    msg = str(exc_info.value)
    assert "30" in msg
    assert "archive" in msg.lower() or "delete" in msg.lower()
    assert "@canonical-source" not in msg  # honesty disclaimer
    assert "topstep_xfa_parameters.txt:349" not in msg


def test_scenario_3_30_days_minus_1s_warns_no_raise(state_dir, caplog):
    """Scenario 3 — boundary direction confirmation in the other direction."""
    _seed(state_dir, "STALE30M", age_seconds=30 * 86400 - 1)
    with caplog.at_level(logging.WARNING, logger="trading_app.account_hwm_tracker"):
        t = AccountHWMTracker("STALE30M", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)
    assert t._hwm == 50000.0  # loaded
    stale_warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert stale_warns


def test_scenario_4_corrupt_json_with_callback_dispatches_and_backs_up(state_dir):
    """Scenario 4 — corrupt-state notify in addition to log.error and backup."""
    (state_dir / "account_hwm_CORRUPT_INT.json").write_text("{not valid json")
    calls: list[str] = []
    t = AccountHWMTracker(
        "CORRUPT_INT",
        "topstep",
        dd_limit_dollars=2000.0,
        state_dir=state_dir,
        notify_callback=calls.append,
    )
    # Reinitialised
    assert t._hwm == 0.0
    # Backup created
    backups = list(state_dir.glob("account_hwm_CORRUPT_INT_CORRUPT_*.json"))
    assert len(backups) == 1
    # Exactly one notify dispatch with CORRUPT token
    corrupt_notifies = [c for c in calls if "CORRUPT" in c]
    assert len(corrupt_notifies) == 1, f"Expected one CORRUPT notify; got {calls!r}"


def test_scenario_5_persist_oserror_dispatches_and_reraises(state_dir):
    """Scenario 5 — synthetic OSError on save-state path raises STATE_PERSIST_FAIL notify and re-raises."""
    calls: list[str] = []
    t = AccountHWMTracker(
        "PERSIST_INT",
        "topstep",
        dd_limit_dollars=2000.0,
        state_dir=state_dir,
        notify_callback=calls.append,
    )
    t.update_equity(50000.0)  # primes state

    original_write = Path.write_text

    def _raising_write(self, *args, **kwargs):
        if "tmp" in self.name:
            raise OSError("simulated full-disk")
        return original_write(self, *args, **kwargs)

    with patch.object(Path, "write_text", _raising_write):
        with pytest.raises(OSError, match="simulated full-disk"):
            t.update_equity(50100.0)

    persist_notifies = [c for c in calls if "STATE_PERSIST_FAIL" in c]
    assert len(persist_notifies) == 1, f"Expected one persist-fail notify; got {calls!r}"
    # Counter NOT incremented (mutation guard against the broker-poll signal class)
    assert t._consecutive_poll_failures == 0


def test_state_file_age_days_helper_used_by_load_state(state_dir):
    """Cross-cutting B9 verification: _load_state's stale gate calls state_file_age_days
    (single source of truth shared with Stage 4's pre-session check).

    Verifies the helper is the actual code path, not a copy-paste age computation."""
    _seed(state_dir, "AGE_INT", age_seconds=29 * 86400)
    calls: list[int] = []
    real_helper = state_file_age_days

    def _spy_helper(path: Path):
        calls.append(1)
        return real_helper(path)

    with patch("trading_app.account_hwm_tracker.state_file_age_days", _spy_helper):
        AccountHWMTracker("AGE_INT", "topstep", dd_limit_dollars=2000.0, state_dir=state_dir)

    assert calls, "_load_state must delegate to state_file_age_days; helper was not called"


def test_scenario_6_orchestrator_bound_method_wired_as_notify_callback(state_dir):
    """Scenario 6 (Stage 3 wire-up integration) — design v3 § 12.2 / Stage 3 stage doc.

    Real-object integration: construct a SessionOrchestrator-shaped class with a
    real `_notify` bound method, pass that bound method as `notify_callback` to a
    real `AccountHWMTracker`, then verify:
      (a) tracker._notify_callback compares equal to the bound method
      (mutation guard against passing None or a different attribute);
      (b) when the tracker invokes its callback path (B4 corrupt-state notify),
      the bound method actually receives the message — proving runtime wiring,
      not just static-source pattern matching.

    This complements the static-source assertion in test_session_orchestrator.py
    (which pins the construction call site against grep) by exercising the
    callback at runtime through the tracker's actual dispatch path. A future
    refactor that drops the kwarg silently or rebinds it to a no-op would pass
    the static grep but fail this test.

    Stage 3 of HWM persistence integrity hardening: closes audit-gate finding C-1
    (Scenario 6 not in integration file).
    """

    class _OrchestratorStub:
        """Minimal orchestrator-shaped class with a real bound _notify method.

        Mirrors the SessionOrchestrator._notify contract: takes a string message,
        returns None, never raises. Records calls for assertion.
        """

        def __init__(self):
            self.calls: list[str] = []

        def _notify(self, message: str) -> None:
            self.calls.append(message)

    orch = _OrchestratorStub()

    # Seed a CORRUPT state file so construction triggers the B4 corrupt-state
    # notify path through the bound method (exercises the runtime wiring).
    (state_dir / "account_hwm_SCEN6.json").write_text("{not valid json")

    tracker = AccountHWMTracker(
        "SCEN6",
        "topstep",
        dd_limit_dollars=2000.0,
        state_dir=state_dir,
        notify_callback=orch._notify,
    )

    # Assertion (a): bound-method equality (== not is, per pre-execution audit
    # improvement — Python bound methods are fresh wrappers per access).
    assert tracker._notify_callback is not None, "callback must not be None when explicitly passed at construction"
    assert tracker._notify_callback == orch._notify, (
        "tracker._notify_callback must equal orchestrator's bound _notify method"
    )
    assert tracker._notify_callback.__func__ is type(orch)._notify, (  # type: ignore[attr-defined]
        "callback must wrap the orchestrator's _notify function (not a different attribute)"
    )

    # Assertion (b): runtime dispatch path actually calls the bound method.
    # Corrupt state file → tracker invokes _safe_notify → callback fires → orch
    # receives the message via its real _notify implementation.
    corrupt_notifies = [c for c in orch.calls if "CORRUPT" in c]
    assert len(corrupt_notifies) == 1, (
        f"Expected one CORRUPT notify dispatched through the bound method; got {orch.calls!r}"
    )
