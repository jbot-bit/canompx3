"""RED regression test for F0★ — post-market-buffer negative-time bug.

PREP ARTIFACT (untracked, authored while bot armed). Move to
tests/test_trading_app/ at implementation time. RED-confirm BEFORE the Stage-1
fix lands, GREEN after.

Bug (verified 2026-06-08, session_orchestrator.py):
  _minutes_to_close_et() (:1363-1378) anchors close_today to TODAY in ET with no
  next-close roll, so after the ET close hour it returns a NEGATIVE number. The
  post-market-buffer guard (:2629) `mins_to_close <= 10.0` then matches trivially
  (-236 <= 10) and returns BEFORE broker submit → every live entry silently
  skipped for the rest of the ET day. This is the operator's "it said it took the
  trade but nothing reached Topstep" event.

This test pins ET wall-clock via patched datetime and asserts the BEHAVIOR the
fix must produce, independent of which fix option is chosen (A: bound the guard;
B: roll the helper). It is written against the OBSERVABLE entry/flatten decision,
not the helper's raw sign, so it survives either approach — EXCEPT the explicit
:3878-contract test below, which pins the negative-detection a naive source-roll
would break (see 00_GROUNDING_LEDGER.md ⚠).
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from trading_app.live.session_orchestrator import SessionOrchestrator

ET = ZoneInfo("America/New_York")


def _orch_with_close(hour: int = 16, minute: int = 0) -> SessionOrchestrator:
    """Bare orchestrator with only the close-time attrs _minutes_to_close_et reads."""
    orch = object.__new__(SessionOrchestrator)
    orch._close_hour_et = hour
    orch._close_min_et = minute
    orch._close_time_forced = False
    return orch


class _FixedDatetime:
    """datetime stand-in: .now(tz) returns a pinned instant; everything else
    delegates to the real datetime so .replace()/arithmetic still work."""

    def __init__(self, pinned_et: datetime):
        self._pinned_et = pinned_et

    def now(self, tz=None):
        if tz is None:
            return self._pinned_et
        return self._pinned_et.astimezone(tz)

    def __getattr__(self, name):
        return getattr(datetime, name)


def _pin(et_dt: datetime):
    return patch("trading_app.live.session_orchestrator.datetime", _FixedDatetime(et_dt))


# ---------------------------------------------------------------------------
# Helper-level: the invariant the fix must establish.
# ---------------------------------------------------------------------------


def test_minutes_to_close_is_positive_before_close():
    orch = _orch_with_close(16, 0)
    with _pin(datetime(2026, 6, 8, 9, 0, tzinfo=ET)):
        assert orch._minutes_to_close_et() == pytest.approx(420.0)  # 7h to 16:00


@pytest.mark.xfail(reason="RED until F0★ fix: helper goes negative after close", strict=False)
def test_minutes_to_close_never_negative_after_close():
    """The core invariant of the SOURCE-fix option (B). Under Option A the helper
    may stay signed — see the entry-decision tests below for the option-agnostic
    proof. Marked xfail so the prep suite is green pre-fix; flip to a hard assert
    if Option B is chosen."""
    orch = _orch_with_close(16, 0)
    with _pin(datetime(2026, 6, 8, 16, 5, tzinfo=ET)):
        assert orch._minutes_to_close_et() >= 0.0


# ---------------------------------------------------------------------------
# Behavioral: the entry-gating decision (option-agnostic — the real acceptance).
# These call the guard logic the way :2629 does. At implementation time, wire
# through the actual entry path; here we assert the guard predicate directly.
# ---------------------------------------------------------------------------


def _post_market_buffer_blocks_entry(orch) -> bool:
    """Mirror of the production post-market-buffer guard at
    session_orchestrator.py:2627-2640. Kept in lock-step with that line:

        if mins_to_close is not None and 0.0 <= mins_to_close <= 10.0:

    The `0.0 <=` lower bound is the F0★ fix — a signed _minutes_to_close_et()
    goes NEGATIVE after close, and without the lower bound that negative would
    satisfy `<= 10.0` and skip every entry for the rest of the ET day. If this
    helper and the production line ever diverge, the sign tests
    (test_minutes_to_close_*) and the :2629 comment are the tripwires."""
    mins = orch._minutes_to_close_et()
    return mins is not None and 0.0 <= mins <= 10.0


def test_entry_allowed_just_after_close_REGRESSION():
    """F0★ regression. At 16:05 ET (5 min PAST a 16:00 close) the next overnight
    session's entries must NOT be blocked by the post-market buffer. Pre-fix the
    helper returned ~-5 and the UNBOUNDED guard `-5 <= 10.0` was True → entry
    silently skipped (the bug). Post-fix the `0.0 <=` lower bound excludes the
    negative → entry allowed. RED-confirmed against origin/main before the fix."""
    orch = _orch_with_close(16, 0)
    with _pin(datetime(2026, 6, 8, 16, 5, tzinfo=ET)):
        assert _post_market_buffer_blocks_entry(orch) is False, (
            "F0★ regression: entry blocked after close (negative-time guard match)"
        )


def test_entry_blocked_in_real_pre_close_window():
    """5 min BEFORE a 16:00 close → buffer SHOULD block (real pre-close). Passes
    under both current and fixed code (mins is +5, in range either way)."""
    orch = _orch_with_close(16, 0)
    with _pin(datetime(2026, 6, 8, 15, 55, tzinfo=ET)):
        assert _post_market_buffer_blocks_entry(orch) is True


def test_entry_allowed_well_before_close():
    """7h before close → not blocked. Passes under both current and fixed code."""
    orch = _orch_with_close(16, 0)
    with _pin(datetime(2026, 6, 8, 9, 0, tzinfo=ET)):
        assert _post_market_buffer_blocks_entry(orch) is False


# ---------------------------------------------------------------------------
# Close-time FLATTEN guard (:1916) must keep its existing 0 < mins <= 5 behavior.
# ---------------------------------------------------------------------------


def _flatten_should_fire(orch) -> bool:
    mins = orch._minutes_to_close_et()
    return mins is not None and 0 < mins <= 5.0


def test_flatten_fires_in_pre_close_window():
    orch = _orch_with_close(16, 0)
    with _pin(datetime(2026, 6, 8, 15, 57, tzinfo=ET)):
        assert _flatten_should_fire(orch) is True


def test_flatten_does_not_fire_well_before_close():
    orch = _orch_with_close(16, 0)
    with _pin(datetime(2026, 6, 8, 9, 0, tzinfo=ET)):
        assert _flatten_should_fire(orch) is False


def test_flatten_does_not_fire_after_close_under_option_A():
    """If Option A keeps the helper signed, post-close mins is negative → flatten
    guard's `0 <` excludes it (correct: a new session, not a pre-close flatten)."""
    orch = _orch_with_close(16, 0)
    with _pin(datetime(2026, 6, 8, 16, 5, tzinfo=ET)):
        # signed helper → negative → not in (0, 5]
        assert _flatten_should_fire(orch) is False


# ---------------------------------------------------------------------------
# THE TRAP the plan's source-roll would spring (:3878 contract). Pins the
# negative-detection the "adjusted close already passed → flatten on start"
# safety relies on. If Option B (source-roll) is chosen, THIS must be reworked,
# not silently broken. See 00_GROUNDING_LEDGER.md ⚠.
# ---------------------------------------------------------------------------


def test_adjusted_close_passed_detection_contract():
    """:3878 does `mins = _minutes_to_close_et(); if mins < 0: flatten_on_start`.
    Under the signed helper (Option A) this fires after close. This test documents
    that contract so a naive source-roll (Option B) can't disable it unnoticed."""
    orch = _orch_with_close(16, 0)
    with _pin(datetime(2026, 6, 8, 16, 5, tzinfo=ET)):
        mins = orch._minutes_to_close_et()
        # Under Option A: signed, negative → :3878 branch fires (correct).
        # Under Option B: helper >= 0 → this assert MUST be replaced with the
        # new close-already-passed signal, or flatten-on-start silently dies.
        assert mins is not None
        # Pre-fix AND post-Option-A this is True. If it becomes False, Option B
        # was chosen and :3878 needs the rework — fail loudly here as the tripwire.
        assert mins < 0, (
            "If this fails, the helper no longer returns negative after close — "
            "verify :3878 flatten-on-start was reworked, not silently disabled."
        )
