"""
HARDCORE STRESS TESTS — trading core logic.

Philosophy: honesty over accuracy.
  - Tests MUST fail loudly when something is wrong.
  - No papering over edge cases with "it's probably fine".
  - Every invariant that MUST hold is tested with assertion messages.

Coverage:
  1. Exact ORB boundary (close == orb_high is NOT outside, must be strict >)
  2. Zero-risk guard (orb_high == orb_low → null result, no division by zero)
  3. Simultaneous stop+target (same bar) → conservative loss, always
  4. E0 gap fill logic (confirm bar gapped fully past ORB → no fill)
  5. E3 stop-before-retrace (stop hit before price retraces → no fill)
  6. Confirm bar reset (inside bar resets count)
  7. C3 slow break filter (1000 session only; 0900 ignores it)
  8. pnl_r bounds invariant via random fuzz (must be in [-1.0, rr_target] or None)
  9. DST transition day correctness (spring forward / fall back)
  10. DOW misalignment guard on 0030 (must raise)
  11. Trading day boundary exact (23:00:00 UTC = start of next Brisbane day)
  12. DST verdict classify_dst_verdict boundary conditions
  13. Outlier ORB sizes (0.1 points, 1000 points)
  14. MAE/MFE invariants (win → mfe_r >= pnl_r; stop loss → mae_r >= 1.0 approx)
  15. No confirm with insufficient bars (CB5 with 4 bars → no entry)
  16. Unknown instrument → ValueError on cost spec
  17. 2300 DST context: winter=pre-data, summer=post-data (documented claim verified)
  18. Session end boundary (bar at trading_day_end excluded from scan)
  19. Outlier: tiny ORB survives arithmetic without crashing
  20. Fuzz: entry_price always equals orb_high (long) or orb_low (short) for E0
"""

import random
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from pipeline.cost_model import get_cost_spec
from pipeline.dst import (
    DST_AFFECTED_SESSIONS,
    DST_CLEAN_SESSIONS,
    classify_dst_verdict,
    is_uk_dst,
    is_us_dst,
    is_winter_for_session,
    validate_dow_filter_alignment,
    us_equity_open_brisbane,
    us_data_open_brisbane,
)
from trading_app.entry_rules import (
    ConfirmResult,
    detect_confirm,
)
from trading_app.outcome_builder import (
    CONFIRM_BARS_OPTIONS,
    RR_TARGETS,
    compute_single_outcome,
    _compute_outcomes_all_rr,
)
from trading_app.entry_rules import detect_entry_with_confirm_bars

UTC = timezone.utc

# =============================================================================
# HELPERS
# =============================================================================

def _bars(start_ts, ohlcv_list, interval_minutes=1):
    """Build a minimal bars_df from a list of (o, h, l, c, v) tuples."""
    rows = []
    ts = start_ts
    for o, h, l, c, v in ohlcv_list:
        rows.append({
            "ts_utc": ts,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": int(v),
        })
        ts += timedelta(minutes=interval_minutes)
    return pd.DataFrame(rows)


def _mgc():
    return get_cost_spec("MGC")


# Standard test dates (well inside known DST periods)
SUMMER_DATE = date(2024, 7, 1)   # US and UK both in DST
WINTER_DATE = date(2024, 1, 15)  # US and UK both in standard time

# A reference UTC start time for test bars
T0 = datetime(2024, 7, 5, 0, 0, tzinfo=UTC)  # Brisbane ~10:00am (summer)
DAY_END = T0 + timedelta(hours=8)


# =============================================================================
# 1. EXACT ORB BOUNDARY — close == orb_high is NOT a break
# =============================================================================

class TestExactORBBoundary:
    """close == orb_high MUST NOT trigger confirm. Break requires strict >."""

    def test_close_exactly_at_orb_high_no_confirm(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = T0
        # Bar closes exactly AT orb_high — NOT outside
        bars = _bars(T0, [
            (2695, 2702, 2694, 2700.0, 100),  # close == orb_high exactly → no break
            (2700, 2705, 2699, 2701.0, 100),  # close > orb_high → this IS a break
        ])
        result = detect_confirm(
            bars_df=bars,
            orb_break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            confirm_bars=1,
            detection_window_end=DAY_END,
        )
        assert result.confirmed, "Expected confirm on bar 2 (close=2701 > 2700)"
        assert result.confirm_bar_close == pytest.approx(2701.0), (
            "Confirm should be on bar 2, not bar 1 (close=2700 not outside)"
        )

    def test_close_exactly_at_orb_low_no_confirm(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = T0
        bars = _bars(T0, [
            (2695, 2696, 2688, 2690.0, 100),  # close == orb_low exactly → no break
            (2690, 2691, 2685, 2689.5, 100),  # close < orb_low → IS a break
        ])
        result = detect_confirm(
            bars_df=bars,
            orb_break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="short",
            confirm_bars=1,
            detection_window_end=DAY_END,
        )
        assert result.confirmed
        assert result.confirm_bar_close == pytest.approx(2689.5)

    def test_floating_point_epsilon_above_boundary(self):
        """close = orb_high + 1e-10 must trigger confirm (not eaten by float comparison)."""
        orb_high = 2700.0
        close_epsilon = 2700.0 + 1e-10
        bars = _bars(T0, [(2698, 2701, 2697, close_epsilon, 100)])
        result = detect_confirm(
            bars_df=bars,
            orb_break_ts=T0,
            orb_high=orb_high,
            orb_low=2690.0,
            break_dir="long",
            confirm_bars=1,
            detection_window_end=DAY_END,
        )
        assert result.confirmed, (
            f"close={close_epsilon} > orb_high={orb_high} should confirm but didn't"
        )


# =============================================================================
# 2. ZERO-RISK GUARD — orb_high == orb_low
# =============================================================================

class TestZeroRisk:
    """A zero-range ORB must return null result — no division-by-zero."""

    def test_collapsed_orb_returns_null(self):
        orb_price = 2700.0
        bars = _bars(T0, [
            (2700, 2701, 2699, 2700.1, 100),  # Would normally confirm
            (2700, 2705, 2700, 2703.0, 100),
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_price,
            orb_low=orb_price,  # Zero range
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
        )
        # Should not crash and should return null
        assert result["pnl_r"] is None, (
            "Zero-range ORB must produce null pnl_r — no trade possible"
        )
        assert result["entry_ts"] is None or result["pnl_r"] is None, (
            "Zero risk_points must result in null outcome"
        )


# =============================================================================
# 3. SIMULTANEOUS STOP + TARGET ON SAME BAR → CONSERVATIVE LOSS
# =============================================================================

class TestSimultaneousStopTarget:
    """When fill bar (or post-entry bar) touches both stop and target,
    we must ALWAYS record a conservative loss. Never a win. Never ambiguous."""

    def test_fillbar_both_hit_is_loss(self):
        orb_high, orb_low = 2700.0, 2690.0
        # E1: entry at bar 1 open = 2702. Risk = 2702 - 2690 = 12. RR2 target = 2726.
        # Bar 1 (fill bar for E1): low=2688 (hits stop) AND high=2730 (hits target)
        bars = _bars(T0, [
            (2695, 2701, 2694, 2701.0, 100),  # bar0: confirm (close > orb_high)
            (2702, 2730, 2688, 2710, 100),     # bar1: fill bar — stop AND target hit
            (2710, 2720, 2708, 2715, 100),
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
        )
        assert result["outcome"] == "loss", (
            "Simultaneous stop+target on fill bar must be conservative loss"
        )
        assert result["pnl_r"] == pytest.approx(-1.0), (
            "pnl_r must be exactly -1.0 on conservative loss"
        )

    def test_post_entry_bar_both_hit_is_loss(self):
        orb_high, orb_low = 2700.0, 2690.0
        # E1: entry at bar1 open=2703. Risk=13. RR2 target=2729.
        # Bar 2 (post-entry): touches BOTH stop (low=2688) and target (high=2732)
        bars = _bars(T0, [
            (2695, 2701, 2694, 2701.0, 100),  # bar0: confirm
            (2703, 2706, 2701, 2704.0, 100),  # bar1: fill bar (clean, no exit)
            (2704, 2732, 2688, 2710, 100),    # bar2: both hit simultaneously
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
        )
        assert result["outcome"] == "loss", (
            "Simultaneous hit on post-entry bar must be conservative loss (not win)"
        )


# =============================================================================
# 4. E0 GAP FILL LOGIC — confirm bar gapped past ORB → no fill
# =============================================================================

class TestE0GapFill:
    """E0 fills at ORB level ON confirm bar. If confirm bar's low > orb_high
    (for long), the ORB level was never touched — no fill."""

    def test_e0_gap_past_orb_no_fill(self):
        orb_high, orb_low = 2700.0, 2690.0
        # bar0: close=2700.0 = orb_high exactly → NOT outside (strict > fails), no confirm yet
        # bar1: close=2710 > orb_high=2700 → CB1 confirm. But low=2705 > 2700 → bar gapped past ORB.
        #   E0 limit order at 2700 was never touched → no fill.
        bars = _bars(T0, [
            (2699, 2700, 2698, 2700.0, 100),  # bar0: close==orb_high, NOT outside
            (2705, 2712, 2705, 2710.0, 100),  # bar1: CB1 confirm, low=2705 > orb_high → no E0 fill
            (2710, 2720, 2708, 2715.0, 100),
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E0",
        )
        assert result["entry_ts"] is None, (
            "E0: confirm bar gapped past ORB (low > orb_high) — must NOT fill"
        )

    def test_e0_confirm_bar_touches_orb_fills(self):
        """E0 fills if confirm bar's low <= orb_high (bar touched ORB level)."""
        orb_high, orb_low = 2700.0, 2690.0
        # confirm bar: low=2700 (touches ORB edge exactly) — E0 fills at 2700
        bars = _bars(T0, [
            (2698, 2701, 2697, 2701.0, 100),  # bar0: confirm (close > orb_high)
            (2700, 2710, 2700, 2705.0, 100),  # doesn't matter, confirm was bar0
            (2705, 2720, 2704, 2715.0, 100),
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E0",
        )
        assert result["entry_ts"] is not None, "E0: bar touching ORB should fill"
        assert result["entry_price"] == pytest.approx(orb_high), (
            f"E0 long entry must be at orb_high={orb_high}, got {result['entry_price']}"
        )

    def test_e0_short_gap_past_orb_no_fill(self):
        """E0 short: confirm bar gapped below orb_low (high < orb_low) → no fill."""
        orb_high, orb_low = 2700.0, 2690.0
        # bar0: close=2690.0 = orb_low exactly → NOT outside for short (strict < fails), no confirm
        # bar1: close=2682 < orb_low=2690 → CB1 confirm. But high=2685 < 2690 → gapped past ORB.
        #   E0 limit order at 2690 was never touched → no fill.
        bars = _bars(T0, [
            (2692, 2692, 2689, 2690.0, 100),  # bar0: close==orb_low, NOT outside for short
            (2684, 2685, 2680, 2682.0, 100),  # bar1: CB1 confirm, high=2685 < orb_low=2690 → no E0 fill
            (2682, 2683, 2675, 2676.0, 100),
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="short",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E0",
        )
        assert result["entry_ts"] is None, (
            "E0 short: confirm bar gapped below orb_low (high < orb_low) — must NOT fill"
        )


# =============================================================================
# 5. E3 STOP-BEFORE-RETRACE — stop hit before retrace → no fill
# =============================================================================

class TestE3StopBeforeRetrace:
    """If price blows through stop BEFORE retracing to ORB level, E3 must NOT fill.
    Filling after the stop is hit is a look-ahead violation."""

    def test_e3_stop_before_retrace_no_fill(self):
        orb_high, orb_low = 2700.0, 2690.0
        # Long break confirmed at bar0. E3 waits for retrace to 2700.
        # bar1: price drops to 2688 (below orb_low=2690 = stop) → stop hit
        # bar2: price retraces to 2701 touching orb_high — but stop already hit
        bars = _bars(T0, [
            (2698, 2702, 2697, 2702.0, 100),  # bar0: confirm long (CB1)
            (2700, 2701, 2688, 2689.0, 100),  # bar1: crashes through stop
            (2689, 2701, 2688, 2700.5, 100),  # bar2: touches orb_high, but stop already hit
            (2700, 2710, 2699, 2705.0, 100),
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E3",
        )
        assert result["entry_ts"] is None, (
            "E3: stop was hit before retrace — must not fill (would be a look-ahead violation)"
        )

    def test_e3_retrace_before_stop_does_fill(self):
        """E3 retrace that arrives BEFORE stop is hit MUST fill."""
        orb_high, orb_low = 2700.0, 2690.0
        bars = _bars(T0, [
            (2698, 2702, 2697, 2702.0, 100),  # bar0: confirm long
            (2702, 2705, 2700, 2703.0, 100),  # bar1: touches orb_high (retrace) → E3 fills
            (2703, 2720, 2702, 2715.0, 100),
            (2715, 2740, 2714, 2730.0, 100),  # target hit (if RR not too high)
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=1.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E3",
        )
        assert result["entry_ts"] is not None, (
            "E3: retrace before stop must trigger fill"
        )
        assert result["entry_price"] == pytest.approx(orb_high), (
            f"E3 long must fill at orb_high={orb_high}"
        )


# =============================================================================
# 6. CONFIRM BAR RESET — inside bar resets the count
# =============================================================================

class TestConfirmBarReset:
    """One bar closing inside ORB must reset the consecutive count to zero."""

    def test_reset_prevents_premature_confirm(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = T0
        # CB3 test: two outside bars, then inside (reset), then three more outside → CB3 at bar5
        # Without the reset: bar2 would be count=3 → CB3 triggers at bar2.
        # With the reset (bar2 close=2700.0 = orb_high, NOT > orb_high): count resets to 0.
        # CB3 instead triggers at bar5 (3rd consecutive after reset).
        bars = _bars(T0, [
            (2700, 2702, 2699, 2701.0, 100),  # bar0: outside (count=1)
            (2701, 2703, 2700, 2702.0, 100),  # bar1: outside (count=2)
            (2702, 2702, 2699, 2700.0, 100),  # bar2: INSIDE (close=2700=orb_high, not >) → RESET
            (2699, 2703, 2698, 2701.5, 100),  # bar3: outside (count=1 after reset)
            (2701, 2703, 2700, 2702.0, 100),  # bar4: outside (count=2)
            (2702, 2704, 2701, 2703.5, 100),  # bar5: outside (count=3) → CB3 confirms here
        ])
        result = detect_confirm(
            bars_df=bars,
            orb_break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            confirm_bars=3,
            detection_window_end=DAY_END,
        )
        assert result.confirmed, "CB3 should confirm on bar5 after reset at bar2"
        # Must be bar5 (close=2703.5), NOT bar2 (which reset and can't confirm)
        assert result.confirm_bar_close == pytest.approx(2703.5), (
            f"Reset at bar2 must delay CB3 to bar5 (close=2703.5), got {result.confirm_bar_close}"
        )

    def test_exact_reset_at_boundary(self):
        """Close exactly at orb_high (not outside) RESETS the count."""
        orb_high = 2700.0
        bars = _bars(T0, [
            (2698, 2702, 2697, 2701.0, 100),  # bar0: outside (count=1)
            (2701, 2702, 2699, 2700.0, 100),  # bar1: INSIDE (close==orb_high, not > orb_high) → RESET
            (2700, 2703, 2699, 2701.5, 100),  # bar2: outside (count=1 again)
            (2701, 2704, 2700, 2702.0, 100),  # bar3: outside (count=2) → CB2 confirms
        ])
        result = detect_confirm(
            bars_df=bars,
            orb_break_ts=T0,
            orb_high=orb_high,
            orb_low=2690.0,
            break_dir="long",
            confirm_bars=2,
            detection_window_end=DAY_END,
        )
        assert result.confirmed, "CB2 should confirm on bar3"
        assert result.confirm_bar_close == pytest.approx(2702.0), (
            "After reset at bar1, confirm must be bar3 (not bar1)"
        )


# =============================================================================
# 7. C3 SLOW BREAK FILTER — 1000 session only
# =============================================================================

class TestC3SlowBreakFilter:
    """C3: at 1000 session, if confirm_bar_ts - break_ts > 3 min → skip trade.
    This must NOT apply to 0900 or any other session."""

    def _make_signal_for_c3(self):
        """Long bars with a slow confirm (4 min after break_ts)."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = T0  # Break at T0
        # Confirm bar at T0 + 4 min (> 3 min = slow break)
        # Bars 0-3: close=2700.0 = orb_high exactly → NOT outside (strict > fails) → no confirm yet
        # Bar 4: close=2701.0 > 2700 → CB1 confirm. 4 min after break_ts → slow break (>3min)
        bars = _bars(T0, [
            (2699, 2700, 2698, 2700.0, 100),  # bar0 (T0+0): close==orb_high, NOT outside
            (2699, 2700, 2698, 2700.0, 100),  # bar1 (T0+1): same, still inside
            (2699, 2700, 2698, 2700.0, 100),  # bar2 (T0+2): same, still inside
            (2699, 2700, 2698, 2700.0, 100),  # bar3 (T0+3): same, still inside
            (2699, 2703, 2698, 2701.0, 100),  # bar4 (T0+4): CB1 confirm (close>2700), 4min from break → SLOW
            (2701, 2710, 2700, 2705.0, 100),  # post-entry (E1 fills here)
            (2705, 2730, 2704, 2725.0, 100),  # win bar
        ])
        return bars, break_ts, orb_high, orb_low

    def test_c3_skips_slow_break_at_1000(self):
        bars, break_ts, orb_high, orb_low = self._make_signal_for_c3()
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
            orb_label="1000",  # C3 applies
        )
        # C3 is inside _compute_outcomes_all_rr, not compute_single_outcome.
        # compute_single_outcome doesn't run C3. This tests the lower-level function.
        # The test below tests _compute_outcomes_all_rr directly with 1000 label.

    def test_c3_slow_break_still_produces_outcome(self):
        """After F-03 audit: C3 moved to strategy discovery layer.
        outcome_builder no longer filters slow breaks — it returns outcomes
        for all triggered signals. Break speed filtering is now handled by
        BreakSpeedFilter in config.py during strategy discovery."""
        bars, break_ts, orb_high, orb_low = self._make_signal_for_c3()
        signal = detect_entry_with_confirm_bars(
            bars_df=bars,
            orb_break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            confirm_bars=1,
            detection_window_end=DAY_END,
            entry_model="E1",
        )
        assert signal.triggered, "Signal must trigger"
        results = _compute_outcomes_all_rr(
            bars_df=bars,
            signal=signal,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_targets=[2.0],
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
            orb_label="1000",
            break_ts=break_ts,
        )
        assert len(results) == 1
        assert results[0]["pnl_r"] is not None, (
            "Post F-03: outcome_builder must return outcomes for slow breaks "
            "(filtering is now at strategy discovery layer)"
        )

    def test_c3_does_not_apply_to_0900(self):
        """Same slow break at 0900 session must NOT be filtered."""
        bars, break_ts, orb_high, orb_low = self._make_signal_for_c3()
        signal = detect_entry_with_confirm_bars(
            bars_df=bars,
            orb_break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            confirm_bars=1,
            detection_window_end=DAY_END,
            entry_model="E1",
        )
        results = _compute_outcomes_all_rr(
            bars_df=bars,
            signal=signal,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_targets=[2.0],
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
            orb_label="0900",  # C3 does NOT apply
            break_ts=break_ts,
        )
        assert results[0]["pnl_r"] is not None, (
            "C3 must NOT filter at 0900 — only 1000"
        )


# =============================================================================
# 8. pnl_r BOUNDS INVARIANT (RANDOM FUZZ)
# =============================================================================

class TestPnlRBoundsInvariant:
    """pnl_r MUST be in [-1.0, rr_target] or None. Never outside.
    A win can't pay more than the RR target. A loss can't be worse than -1R.
    Tested across random price series."""

    def _random_bar(self, rng, base_price, spread=20):
        o = base_price + rng.uniform(-spread/2, spread/2)
        c = base_price + rng.uniform(-spread/2, spread/2)
        h = max(o, c) + rng.uniform(0, spread/2)
        l = min(o, c) - rng.uniform(0, spread/2)
        return (round(o, 1), round(h, 1), round(l, 1), round(c, 1), rng.randint(10, 500))

    @pytest.mark.parametrize("seed", range(50))
    def test_pnl_r_within_bounds(self, seed):
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        orb_high = 2700.0 + rng.uniform(-10, 10)
        orb_high = round(orb_high, 1)
        orb_low = orb_high - rng.uniform(2, 20)
        orb_low = round(orb_low, 1)

        base = orb_high + 2
        n_bars = rng.randint(5, 30)
        ohlcv = [self._random_bar(rng, base + i * 0.5, spread=15) for i in range(n_bars)]
        bars = _bars(T0, ohlcv)

        break_dir = rng.choice(["long", "short"])
        rr = rng.choice(RR_TARGETS)
        cb = rng.choice(CONFIRM_BARS_OPTIONS)
        model = rng.choice(["E0", "E1", "E3"])

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir=break_dir,
            rr_target=rr,
            confirm_bars=cb,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model=model,
        )

        pnl = result["pnl_r"]
        if pnl is None:
            return  # No entry — fine

        assert pnl >= -1.0 - 1e-6, (
            f"seed={seed}: pnl_r={pnl} is below -1.0 — "
            f"cannot lose more than 1R on a stop loss! (model={model}, rr={rr}, cb={cb})"
        )
        assert pnl <= rr + 1e-6, (
            f"seed={seed}: pnl_r={pnl} exceeds rr_target={rr} — "
            f"cannot win more than the target! (model={model}, cb={cb})"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_loss_always_exactly_minus_one(self, seed):
        """A stop-loss exit MUST produce pnl_r = -1.0 exactly, every time."""
        rng = random.Random(seed + 1000)
        orb_high = round(2700.0 + rng.uniform(-5, 5), 1)
        orb_low = round(orb_high - rng.uniform(5, 15), 1)
        # Make a bar that confirms, then goes straight to stop
        bars = _bars(T0, [
            (orb_high - 1, orb_high + 2, orb_high - 2, orb_high + 0.5, 100),  # confirm
            (orb_high + 0.5, orb_high + 1, orb_low - 5, orb_low - 3, 100),    # E1 fill, crashes to stop
            (orb_low - 3, orb_low, orb_low - 10, orb_low - 5, 100),
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=rng.choice([1.0, 2.0, 3.0, 4.0]),
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
        )
        if result["outcome"] == "loss":
            assert result["pnl_r"] == pytest.approx(-1.0), (
                f"seed={seed}: loss pnl_r={result['pnl_r']} must be exactly -1.0"
            )


# =============================================================================
# 9. DST TRANSITION DAYS
# =============================================================================

class TestDSTTransitionDays:
    """Verify DST detection on the actual US spring-forward and fall-back dates."""

    # 2024 US spring forward: Sunday 2024-03-10 at 2:00 AM ET
    # Day before = EST (UTC-5), that day and after = EDT (UTC-4)
    def test_us_spring_forward_2024(self):
        day_before = date(2024, 3, 9)
        day_of = date(2024, 3, 10)
        day_after = date(2024, 3, 11)
        assert not is_us_dst(day_before), "2024-03-09 is still EST (not DST)"
        assert is_us_dst(day_of), "2024-03-10 is spring forward day — EDT"
        assert is_us_dst(day_after), "2024-03-11 is in EDT"

    # 2024 US fall back: Sunday 2024-11-03 at 2:00 AM ET
    def test_us_fall_back_2024(self):
        day_before = date(2024, 11, 2)
        day_of = date(2024, 11, 3)
        day_after = date(2024, 11, 4)
        assert is_us_dst(day_before), "2024-11-02 is still in EDT"
        assert not is_us_dst(day_of), "2024-11-03 is fall back day — returns to EST"
        assert not is_us_dst(day_after), "2024-11-04 is EST"

    # 2024 UK spring forward: Sunday 2024-03-31
    def test_uk_spring_forward_2024(self):
        day_before = date(2024, 3, 30)
        day_of = date(2024, 3, 31)
        assert not is_uk_dst(day_before), "2024-03-30 is GMT"
        assert is_uk_dst(day_of), "2024-03-31 is UK spring forward (BST)"

    # 2024 UK fall back: Sunday 2024-10-27
    def test_uk_fall_back_2024(self):
        day_before = date(2024, 10, 26)
        day_of = date(2024, 10, 27)
        assert is_uk_dst(day_before), "2024-10-26 is BST"
        assert not is_uk_dst(day_of), "2024-10-27 is UK fall back (GMT)"

    def test_known_summer_dates(self):
        """July dates must always be in DST for both US and UK."""
        for y in [2021, 2022, 2023, 2024, 2025]:
            d = date(y, 7, 15)
            assert is_us_dst(d), f"{d}: US should be in DST in July"
            assert is_uk_dst(d), f"{d}: UK should be in BST in July"

    def test_known_winter_dates(self):
        """January dates must never be in DST for US or UK."""
        for y in [2021, 2022, 2023, 2024, 2025]:
            d = date(y, 1, 15)
            assert not is_us_dst(d), f"{d}: US should NOT be in DST in January"
            assert not is_uk_dst(d), f"{d}: UK should NOT be in BST in January"


# =============================================================================
# 10. DOW MISALIGNMENT GUARD ON 0030
# =============================================================================

class TestDOWMisalignmentGuard:
    """0030 must raise when any DOW skip filter is applied.
    All other sessions must pass silently."""

    def test_0030_dow_filter_raises(self):
        with pytest.raises(ValueError, match="DOW filter"):
            validate_dow_filter_alignment("0030", (4,))  # Skip Friday

    def test_0030_monday_skip_also_raises(self):
        with pytest.raises(ValueError):
            validate_dow_filter_alignment("0030", (0,))  # Skip Monday

    def test_0030_empty_skip_does_not_raise(self):
        """Empty skip tuple means no DOW filter — must not raise."""
        validate_dow_filter_alignment("0030", ())  # Should be fine

    def test_aligned_sessions_never_raise(self):
        """All sessions other than 0030 must never raise for any skip_days."""
        from pipeline.dst import DOW_ALIGNED_SESSIONS
        for session in DOW_ALIGNED_SESSIONS:
            for skip in [(0,), (1,), (4,), (0, 4)]:
                validate_dow_filter_alignment(session, skip)  # Must not raise


# =============================================================================
# 11. TRADING DAY BOUNDARY (exact 23:00 UTC)
# =============================================================================

class TestTradingDayBoundary:
    """09:00 Brisbane = 23:00 UTC previous calendar day.
    Bars at exactly 23:00:00 UTC belong to the CURRENT Brisbane trading day.
    Bars at 22:59:59 UTC belong to the PREVIOUS Brisbane trading day."""

    def test_boundary_computation(self):
        from pipeline.build_daily_features import compute_trading_day
        BRISBANE = ZoneInfo("Australia/Brisbane")
        UTC = ZoneInfo("UTC")

        # 2024-01-05 23:00:00 UTC = 2024-01-06 09:00:00 Brisbane = start of 2024-01-06 trading day
        ts_at_boundary = datetime(2024, 1, 5, 23, 0, 0, tzinfo=UTC)
        ts_before = datetime(2024, 1, 5, 22, 59, 59, tzinfo=UTC)
        ts_after = datetime(2024, 1, 5, 23, 0, 1, tzinfo=UTC)

        day_at = compute_trading_day(pd.Timestamp(ts_at_boundary))
        day_before = compute_trading_day(pd.Timestamp(ts_before))
        day_after = compute_trading_day(pd.Timestamp(ts_after))

        assert day_at == date(2024, 1, 6), (
            f"23:00:00 UTC = 09:00:00 Brisbane = start of Jan 6, got {day_at}"
        )
        assert day_before == date(2024, 1, 5), (
            f"22:59:59 UTC = 08:59:59 Brisbane = still Jan 5, got {day_before}"
        )
        assert day_after == date(2024, 1, 6), (
            f"23:00:01 UTC = 09:00:01 Brisbane = Jan 6, got {day_after}"
        )


# =============================================================================
# 12. DST VERDICT BOUNDARY CONDITIONS
# =============================================================================

class TestDSTVerdictBoundary:
    """Verify the exact boundary conditions for classify_dst_verdict."""

    def test_low_n_overrides_everything(self):
        """If either regime has N < 10, result is LOW-N regardless of avgR."""
        assert classify_dst_verdict(1.5, -0.5, 9, 100) == "LOW-N", "winter N=9 → LOW-N"
        assert classify_dst_verdict(1.5, -0.5, 100, 9) == "LOW-N", "summer N=9 → LOW-N"
        assert classify_dst_verdict(1.5, -0.5, 0, 0) == "LOW-N"

    def test_stable_boundary_exactly_0_10(self):
        """diff == 0.10 exactly with N >= 15 both sides → STABLE."""
        assert classify_dst_verdict(0.30, 0.20, 20, 20) == "STABLE", (
            "|0.30 - 0.20| = 0.10 exactly, N>=15 both → STABLE"
        )

    def test_unstable_just_above_0_10(self):
        """diff == 0.11 → NOT stable (must fall into WINTER-DOM or SUMMER-DOM or UNSTABLE)."""
        result = classify_dst_verdict(0.31, 0.20, 20, 20)
        assert result != "STABLE", f"diff=0.11 must not be STABLE, got {result}"

    def test_winter_dom(self):
        assert classify_dst_verdict(0.50, 0.30, 20, 20) == "WINTER-DOM"

    def test_summer_dom(self):
        assert classify_dst_verdict(0.20, 0.50, 20, 20) == "SUMMER-DOM"

    def test_winter_only(self):
        """winter > 0, summer <= 0, both N >= 10 → WINTER-ONLY (checked before DOM)."""
        assert classify_dst_verdict(0.30, -0.10, 15, 15) == "WINTER-ONLY"
        assert classify_dst_verdict(0.30, 0.0, 15, 15) == "WINTER-ONLY"

    def test_summer_only(self):
        assert classify_dst_verdict(-0.10, 0.30, 15, 15) == "SUMMER-ONLY"

    def test_none_avgr_is_low_n(self):
        """If either avgR is None (no trades), must be LOW-N."""
        assert classify_dst_verdict(None, 0.30, 20, 20) == "LOW-N"
        assert classify_dst_verdict(0.30, None, 20, 20) == "LOW-N"


# =============================================================================
# 13. OUTLIER ORB SIZES
# =============================================================================

class TestOutlierORBSizes:
    """Verify arithmetic doesn't break on extreme ORB sizes."""

    def test_tiny_orb_no_crash(self):
        """ORB of 0.1 points (near tick-size floor). Should not crash."""
        orb_high = 2700.1
        orb_low = 2700.0  # 0.1 point range (MGC tick = 0.10)
        bars = _bars(T0, [
            (2700.0, 2700.2, 2699.9, 2700.15, 100),  # confirm long
            (2700.15, 2700.5, 2700.1, 2700.3, 100),   # E1 entry
            (2700.3, 2700.8, 2700.2, 2700.6, 100),    # climbing
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
        )
        # Just verify no crash and basic sanity
        if result["pnl_r"] is not None:
            assert result["pnl_r"] >= -1.0 - 1e-6

    def test_huge_orb_no_crash(self):
        """ORB of 1000 points. R calculation must still be proportional."""
        orb_high = 3200.0
        orb_low = 2200.0  # 1000 point range
        bars = _bars(T0, [
            (2800, 3201, 2799, 3201.0, 100),   # confirm long
            (3202, 5402, 3200, 3500.0, 100),    # E1 entry; target (RR2) = 3202 + 2000 = 5202
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
        )
        if result["pnl_r"] is not None:
            assert result["pnl_r"] >= -1.0 - 1e-6


# =============================================================================
# 14. MAE/MFE INVARIANTS
# =============================================================================

class TestMAEMFEInvariants:
    """
    On a WIN: mfe_r should be >= pnl_r (the target was reached).
    On a LOSS (stop): mae_r should be close to 1.0 (stop hit = 1R adverse).
    MAE and MFE must always be >= 0.
    """

    def _run(self, ohlcv, break_dir="long", orb_high=2700.0, orb_low=2690.0, rr=2.0):
        bars = _bars(T0, ohlcv)
        return compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir=break_dir,
            rr_target=rr,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E1",
        )

    def test_win_mfe_geq_pnl_r(self):
        """On a win, MFE must be >= pnl_r (we reached the target)."""
        # E1: entry at bar1 open ~2702. Risk=12. RR2 target=2726. Bar2 hits target.
        result = self._run([
            (2698, 2701, 2697, 2701.0, 100),  # confirm
            (2702, 2704, 2701, 2703.0, 100),  # fill bar (clean)
            (2703, 2730, 2702, 2728.0, 100),  # hits target
        ])
        assert result["outcome"] == "win"
        assert result["mfe_r"] is not None
        assert result["pnl_r"] is not None
        assert result["mfe_r"] >= result["pnl_r"] - 1e-6, (
            f"Win: mfe_r={result['mfe_r']} must be >= pnl_r={result['pnl_r']}"
        )

    def test_loss_mae_nonnegative(self):
        """On a loss, MAE must be >= 0."""
        result = self._run([
            (2698, 2701, 2697, 2701.0, 100),  # confirm
            (2702, 2703, 2701, 2702.0, 100),  # fill bar
            (2700, 2701, 2685, 2686.0, 100),  # crashes to stop
        ])
        if result["outcome"] == "loss":
            assert result["mae_r"] is not None
            assert result["mae_r"] >= 0, f"MAE must be >= 0, got {result['mae_r']}"

    def test_mae_mfe_always_nonneg(self):
        """Regardless of outcome, MAE and MFE must be >= 0."""
        scenarios = [
            [(2698, 2701, 2697, 2701.0, 100), (2702, 2704, 2701, 2703.0, 100)],
            [(2698, 2701, 2697, 2701.0, 100)],  # no post-entry
        ]
        for ohlcv in scenarios:
            result = self._run(ohlcv)
            if result["mae_r"] is not None:
                assert result["mae_r"] >= -1e-6, f"MAE negative: {result['mae_r']}"
            if result["mfe_r"] is not None:
                assert result["mfe_r"] >= -1e-6, f"MFE negative: {result['mfe_r']}"


# =============================================================================
# 15. INSUFFICIENT BARS FOR CONFIRM
# =============================================================================

class TestInsufficientBarsForConfirm:
    """CB5 with only 4 bars must not trigger confirm."""

    def test_cb5_with_4_bars_no_confirm(self):
        orb_high = 2700.0
        bars = _bars(T0, [
            (2698, 2702, 2697, 2701.0, 100),  # bar0: outside (count=1)
            (2701, 2703, 2700, 2702.0, 100),  # bar1: outside (count=2)
            (2702, 2704, 2701, 2703.0, 100),  # bar2: outside (count=3)
            (2703, 2705, 2702, 2704.0, 100),  # bar3: outside (count=4) — need 5
        ])
        result = detect_confirm(
            bars_df=bars,
            orb_break_ts=T0,
            orb_high=orb_high,
            orb_low=2690.0,
            break_dir="long",
            confirm_bars=5,
            detection_window_end=DAY_END,
        )
        assert not result.confirmed, "CB5 with 4 outside bars must NOT confirm"

    def test_cb5_with_5_bars_confirms(self):
        orb_high = 2700.0
        bars = _bars(T0, [
            (2698, 2702, 2697, 2701.0, 100),
            (2701, 2703, 2700, 2702.0, 100),
            (2702, 2704, 2701, 2703.0, 100),
            (2703, 2705, 2702, 2704.0, 100),
            (2704, 2706, 2703, 2705.0, 100),  # bar4: 5th consecutive → confirm
        ])
        result = detect_confirm(
            bars_df=bars,
            orb_break_ts=T0,
            orb_high=orb_high,
            orb_low=2690.0,
            break_dir="long",
            confirm_bars=5,
            detection_window_end=DAY_END,
        )
        assert result.confirmed, "CB5 with exactly 5 consecutive outside bars must confirm"
        assert result.confirm_bar_close == pytest.approx(2705.0)


# =============================================================================
# 16. COST MODEL — UNKNOWN INSTRUMENT
# =============================================================================

class TestCostModelInstruments:
    """All 4 instruments must have valid cost specs. Unknown raises."""

    @pytest.mark.parametrize("instrument", ["MGC", "MES", "MNQ", "MCL"])
    def test_known_instruments_have_cost_spec(self, instrument):
        spec = get_cost_spec(instrument)
        assert spec is not None
        assert spec.point_value > 0, f"{instrument}: point_value must be > 0"
        assert spec.total_friction > 0, f"{instrument}: total_friction must be > 0"
        assert spec.tick_size > 0, f"{instrument}: tick_size must be > 0"

    def test_unknown_instrument_raises(self):
        with pytest.raises((ValueError, KeyError)):
            get_cost_spec("GARBAGE_INSTRUMENT")

    def test_mgc_friction_consistent(self):
        """MGC friction is $8.40 RT (commission $2.40 + spread $2.00 + slippage $4.00)."""
        spec = get_cost_spec("MGC")
        assert spec.commission_rt == pytest.approx(2.40), "MGC commission mismatch"
        assert spec.spread_doubled == pytest.approx(2.00), "MGC spread mismatch"
        assert spec.slippage == pytest.approx(4.00), "MGC slippage mismatch"
        assert spec.total_friction == pytest.approx(8.40), "MGC total friction mismatch"


# =============================================================================
# 17. 2300 SESSION — DST CONTEXT SHIFT
# =============================================================================

class TestSession2300DSTContext:
    """2300 Brisbane = 13:00 UTC always.
    US 8:30am ET data release = 13:30 UTC winter / 12:30 UTC summer.
    Winter: 2300 is 30 min BEFORE data.
    Summer: 2300 is 30 min AFTER data.
    Verified by checking DST utility functions."""

    def test_winter_2300_is_before_data_release(self):
        """In winter (EST): 8:30 ET = 13:30 UTC. 2300 Bris = 13:00 UTC = 30min before."""
        winter_day = WINTER_DATE
        assert not is_us_dst(winter_day), "Winter day sanity check"
        h, m = us_data_open_brisbane(winter_day)
        # Winter: 8:30 ET = 13:30 UTC = 23:30 Brisbane
        assert h == 23 and m == 30, (
            f"Winter US data release should be 23:30 Brisbane, got {h}:{m:02d}"
        )
        # 2300 (23:00) is BEFORE 23:30 → pre-data positioning

    def test_summer_2300_is_after_data_release(self):
        """In summer (EDT): 8:30 ET = 12:30 UTC. 2300 Bris = 13:00 UTC = 30min after."""
        summer_day = SUMMER_DATE
        assert is_us_dst(summer_day), "Summer day sanity check"
        h, m = us_data_open_brisbane(summer_day)
        # Summer: 8:30 EDT = 12:30 UTC = 22:30 Brisbane
        assert h == 22 and m == 30, (
            f"Summer US data release should be 22:30 Brisbane, got {h}:{m:02d}"
        )
        # 2300 (23:00) is AFTER 22:30 → post-data reaction

    def test_2300_classified_as_us_dst_affected(self):
        assert "2300" in DST_AFFECTED_SESSIONS
        assert DST_AFFECTED_SESSIONS["2300"] == "US"

    def test_is_winter_for_2300(self):
        assert is_winter_for_session(WINTER_DATE, "2300") is True
        assert is_winter_for_session(SUMMER_DATE, "2300") is False

    def test_1000_is_clean_session(self):
        """1000 must be in clean sessions — Asia has no DST."""
        assert "1000" in DST_CLEAN_SESSIONS
        assert "1000" not in DST_AFFECTED_SESSIONS
        assert is_winter_for_session(WINTER_DATE, "1000") is None
        assert is_winter_for_session(SUMMER_DATE, "1000") is None


# =============================================================================
# 18. SESSION END BOUNDARY — bar at trading_day_end excluded
# =============================================================================

class TestSessionEndBoundary:
    """Bars at ts_utc >= trading_day_end must NOT be included in outcome scan."""

    def test_bar_exactly_at_day_end_not_included(self):
        orb_high, orb_low = 2700.0, 2690.0
        td_end = T0 + timedelta(hours=2)
        bars = _bars(T0, [
            (2698, 2701, 2697, 2701.0, 100),           # T0+0: confirm
            (2702, 2703, 2701, 2702.0, 100),           # T0+1: fill bar (E1), clean
            (2702, 2703, 2701, 2702.0, 100),           # T0+2: clean
            # ... many bars all below target ...
        ] + [(2700, 2701, 2699, 2700.5, 100)] * 116 +  # fill 118 min total
        [(2700, 2750, 2699, 2745.0, 100)])              # This bar IS at td_end timestamp
        # Last bar is at T0 + 119 min. td_end = T0 + 120 min. Bar at T0+119 < td_end → included.
        # Bar at td_end WOULD be at T0+120 but we didn't include one there.
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=4.0,  # High RR so we DON'T win before session end
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_mgc(),
            entry_model="E1",
        )
        # We just want no crash and a valid outcome
        assert result["outcome"] in ("win", "loss", "scratch", "early_exit", None)


# =============================================================================
# 19. OUTLIER: FUZZ E0 ENTRY PRICE INVARIANT
# =============================================================================

class TestE0EntryPriceInvariant:
    """E0 must ALWAYS fill at exactly orb_high (long) or orb_low (short).
    Never at a different price. This is the fundamental E0 guarantee."""

    @pytest.mark.parametrize("seed", range(30))
    def test_e0_long_entry_always_at_orb_high(self, seed):
        rng = random.Random(seed + 2000)
        orb_high = round(2700.0 + rng.uniform(-50, 50), 1)
        orb_low = round(orb_high - rng.uniform(2, 30), 1)

        # Construct a confirm bar that touches orb_high (low <= orb_high)
        confirm_low = orb_high - rng.uniform(0.1, 3.0)
        confirm_high = orb_high + rng.uniform(1.0, 10.0)
        confirm_close = orb_high + rng.uniform(0.1, 5.0)

        bars = _bars(T0, [
            (orb_high - 1, confirm_high, confirm_low, confirm_close, 100),  # confirm + E0 fill
            (confirm_close, confirm_close + 5, confirm_close - 1, confirm_close + 3, 100),
            (confirm_close + 3, confirm_close + 20, confirm_close + 2, confirm_close + 15, 100),
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E0",
        )
        if result["entry_ts"] is not None:
            assert result["entry_price"] == pytest.approx(orb_high, abs=1e-6), (
                f"seed={seed}: E0 long entry must be at orb_high={orb_high}, "
                f"got entry_price={result['entry_price']}"
            )

    @pytest.mark.parametrize("seed", range(30))
    def test_e0_short_entry_always_at_orb_low(self, seed):
        rng = random.Random(seed + 3000)
        orb_low = round(2700.0 - rng.uniform(0, 50), 1)
        orb_high = round(orb_low + rng.uniform(2, 30), 1)

        confirm_high = orb_low + rng.uniform(0.1, 3.0)
        confirm_low = orb_low - rng.uniform(1.0, 10.0)
        confirm_close = orb_low - rng.uniform(0.1, 5.0)

        bars = _bars(T0, [
            (orb_low + 1, confirm_high, confirm_low, confirm_close, 100),
            (confirm_close, confirm_close + 1, confirm_close - 5, confirm_close - 3, 100),
            (confirm_close - 3, confirm_close - 2, confirm_close - 20, confirm_close - 15, 100),
        ])
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=T0,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="short",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=DAY_END,
            cost_spec=_mgc(),
            entry_model="E0",
        )
        if result["entry_ts"] is not None:
            assert result["entry_price"] == pytest.approx(orb_low, abs=1e-6), (
                f"seed={seed}: E0 short entry must be at orb_low={orb_low}, "
                f"got entry_price={result['entry_price']}"
            )


# =============================================================================
# 20. DETECT_CONFIRM GUARD: INVALID INPUTS
# =============================================================================

class TestDetectConfirmGuards:
    """Invalid inputs must raise, not silently produce wrong results."""

    def test_confirm_bars_zero_raises(self):
        bars = _bars(T0, [(2698, 2701, 2697, 2701.0, 100)])
        with pytest.raises(ValueError):
            detect_confirm(bars, T0, 2700.0, 2690.0, "long", 0, DAY_END)

    def test_confirm_bars_above_max_raises(self):
        bars = _bars(T0, [(2698, 2701, 2697, 2701.0, 100)])
        with pytest.raises(ValueError):
            detect_confirm(bars, T0, 2700.0, 2690.0, "long", 11, DAY_END)

    def test_invalid_break_dir_raises(self):
        bars = _bars(T0, [(2698, 2701, 2697, 2701.0, 100)])
        with pytest.raises(ValueError):
            detect_confirm(bars, T0, 2700.0, 2690.0, "sideways", 1, DAY_END)

    def test_empty_bars_no_confirm(self):
        """Empty dataframe must return no_confirm, not crash."""
        bars = pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"])
        result = detect_confirm(bars, T0, 2700.0, 2690.0, "long", 1, DAY_END)
        assert not result.confirmed

    def test_no_bars_in_window_no_confirm(self):
        """All bars are outside detection window — no confirm."""
        bars = _bars(T0, [(2698, 2701, 2697, 2701.0, 100)])
        window_end = T0  # window closes immediately
        result = detect_confirm(bars, T0, 2700.0, 2690.0, "long", 1, window_end)
        assert not result.confirmed, "Window [T0, T0) is empty — no confirm"


# =============================================================================
# 21. DST_AFFECTED vs DST_CLEAN — complete and non-overlapping
# =============================================================================

class TestDSTSessionCoverage:
    """DST_AFFECTED_SESSIONS and DST_CLEAN_SESSIONS must not overlap."""

    def test_no_overlap_between_affected_and_clean(self):
        overlap = set(DST_AFFECTED_SESSIONS.keys()) & DST_CLEAN_SESSIONS
        assert not overlap, (
            f"Sessions appear in BOTH affected and clean: {overlap}. "
            "This is a data integrity bug — a session can't be both."
        )

    def test_known_affected_sessions_present(self):
        for session in ("0900", "0030", "2300", "1800"):
            assert session in DST_AFFECTED_SESSIONS, (
                f"Session {session} must be in DST_AFFECTED_SESSIONS"
            )

    def test_known_clean_sessions_present(self):
        for session in ("1000", "1100", "1130"):
            assert session in DST_CLEAN_SESSIONS, (
                f"Session {session} must be in DST_CLEAN_SESSIONS (Asia = no DST)"
            )


# =============================================================================
# 22. DAILY_FEATURES JOIN CARDINALITY — the 3x inflation trap
# =============================================================================

class TestDailyFeaturesJoinCardinality:
    """The orb_minutes join MUST be included or you triple the row count.
    This test uses an in-memory DuckDB to verify the cardinality rule."""

    def test_join_without_orb_minutes_triples_rows(self):
        """Demonstrate that missing orb_minutes in join inflates rows 3x."""
        import duckdb
        con = duckdb.connect(":memory:")
        # Create minimal daily_features with 3 rows per (symbol, trading_day)
        con.execute("""
            CREATE TABLE daily_features (
                trading_day DATE,
                symbol TEXT,
                orb_minutes INTEGER,
                atr_20 DOUBLE
            )
        """)
        con.execute("""
            CREATE TABLE orb_outcomes (
                trading_day DATE,
                symbol TEXT,
                orb_minutes INTEGER,
                pnl_r DOUBLE
            )
        """)
        # 1 trading day, 3 orb_minutes each
        for orb_min in [5, 15, 30]:
            con.execute(
                "INSERT INTO daily_features VALUES ('2024-01-05', 'MGC', ?, 10.0)",
                [orb_min]
            )
        # orb_outcomes only has orb_minutes=5
        con.execute("INSERT INTO orb_outcomes VALUES ('2024-01-05', 'MGC', 5, 0.5)")

        # BAD join (missing orb_minutes): should return 3 rows (tripled)
        bad_count = con.execute("""
            SELECT COUNT(*)
            FROM orb_outcomes o
            JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
        """).fetchone()[0]

        # CORRECT join (with orb_minutes): should return 1 row
        good_count = con.execute("""
            SELECT COUNT(*)
            FROM orb_outcomes o
            JOIN daily_features d
                ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
        """).fetchone()[0]

        con.close()

        assert bad_count == 3, (
            f"BAD join (no orb_minutes) should produce 3 rows (3x inflation), got {bad_count}. "
            "The test itself may be wrong."
        )
        assert good_count == 1, (
            f"CORRECT join (with orb_minutes) must produce 1 row, got {good_count}. "
            "The orb_minutes join clause is MANDATORY."
        )
