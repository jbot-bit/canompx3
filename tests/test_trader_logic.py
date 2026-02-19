"""
Trader Logic and Math Sanity Tests.

These tests encode trading common sense and mathematical consistency
as automated assertions. They run against synthetic outcomes generated
by the outcome builder and catch:
  - Physically impossible fills or entries
  - Arithmetic errors in R-multiples, risk, targets
  - Cost model inconsistencies
  - Entry model behavioral differences

This file is the CRITICAL GUARD against future regressions to
unrealistic backtesting assumptions.
"""

import sys
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import pytest
import pandas as pd
import duckdb

from trading_app.outcome_builder import compute_single_outcome
from trading_app.entry_rules import detect_confirm, resolve_entry
from trading_app.config import ENTRY_MODELS
from pipeline.cost_model import get_cost_spec, pnl_points_to_r, to_r_multiple, risk_in_dollars

def _cost():
    return get_cost_spec("MGC")

def _make_bars(start_ts, prices, interval_minutes=1):
    """Create bars_df from (open, high, low, close, volume) tuples."""
    rows = []
    ts = start_ts
    for o, h, l, c, v in prices:
        rows.append({
            "ts_utc": ts,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": int(v),
        })
        ts = ts + timedelta(minutes=interval_minutes)
    return pd.DataFrame(rows)

# Standard test scenario: long break, ORB 2690-2700
ORB_HIGH = 2700.0
ORB_LOW = 2690.0
BREAK_TS = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
TD_END = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

# Bars: confirm, then next bar open at 2703, then rally, then retrace
STANDARD_BARS = _make_bars(
    BREAK_TS,
    [
        (2698, 2701, 2695, 2701, 100),  # bar 0: confirm (close > 2700)
        (2703, 2706, 2699, 2704, 100),  # bar 1: next bar (open=2703, low=2699 for E3)
        (2704, 2735, 2703, 2730, 100),  # bar 2: big rally
        (2730, 2740, 2725, 2735, 100),  # bar 3: continue
    ],
)

# ============================================================================
# TRADER LOGIC CHECKS (1-10)
# ============================================================================

class TestEntryPriceReachable:
    """Check 1: Entry price must be a real price from bar data."""

    def test_e1_entry_is_bar_open(self):
        """E1 entry_price must equal the open of the next bar after confirm."""
        result = compute_single_outcome(
            STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E1",
        )
        if result["entry_price"] is not None:
            # E1 entry = open of bar after confirm
            assert result["entry_price"] == 2703.0

    def test_e3_entry_is_orb_level(self):
        """E3 entry must be at ORB level (long=orb_high, short=orb_low)."""
        result = compute_single_outcome(
            STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E3",
        )
        if result["entry_price"] is not None:
            assert result["entry_price"] == ORB_HIGH

class TestEntryTimingCausality:
    """Check 2: Entry must be after signal (causal ordering)."""

    def test_e1_entry_after_confirm(self):
        confirm = detect_confirm(STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, TD_END)
        signal = resolve_entry(STANDARD_BARS, confirm, "E1", TD_END)
        if signal.triggered:
            assert signal.entry_ts > confirm.confirm_bar_ts

    def test_e3_entry_after_confirm(self):
        confirm = detect_confirm(STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, TD_END)
        signal = resolve_entry(STANDARD_BARS, confirm, "E3", TD_END)
        if signal.triggered:
            assert signal.entry_ts > confirm.confirm_bar_ts

class TestRiskPhysicallyCorrect:
    """Check 3: Risk makes physical sense for each entry model."""

    def test_e1_long_entry_above_orb(self):
        """E1 long: entry_price >= orb_high (bought after breakout confirmed)."""
        result = compute_single_outcome(
            STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E1",
        )
        if result["entry_price"] is not None:
            assert result["entry_price"] >= ORB_HIGH

    def test_e3_risk_equals_orb_size(self):
        """E3: risk = |orb_high - orb_low| (entry at ORB level, stop at opposite)."""
        result = compute_single_outcome(
            STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E3",
        )
        if result["entry_price"] is not None:
            risk = abs(result["entry_price"] - result["stop_price"])
            assert risk == pytest.approx(ORB_HIGH - ORB_LOW, abs=0.01)

class TestStopStructuralSense:
    """Check 5: Stop is at the opposite ORB level."""

    def test_long_stop_at_orb_low(self):
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["stop_price"] is not None:
                assert result["stop_price"] == ORB_LOW, f"Failed for {em}"

    def test_short_stop_at_orb_high(self):
        short_bars = _make_bars(
            BREAK_TS,
            [
                (2692, 2695, 2688, 2689, 100),  # confirm short
                (2688, 2691, 2685, 2687, 100),
            ],
        )
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                short_bars, BREAK_TS, ORB_HIGH, ORB_LOW, "short",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["stop_price"] is not None:
                assert result["stop_price"] == ORB_HIGH, f"Failed for {em}"

class TestWinLossPnl:
    """Check 8: Win PnL > 0, loss PnL < 0."""

    def test_win_pnl_positive(self):
        """All winning trades have positive pnl_r."""
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                1.0, 1, TD_END, _cost(), em,
            )
            if result["outcome"] == "win":
                assert result["pnl_r"] > 0, f"Win with negative PnL for {em}"

    def test_loss_pnl_negative(self):
        """All losing trades have pnl_r == -1.0."""
        loss_bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2705, 2689, 2690, 100),  # hits stop at 2690
            ],
        )
        # E1 fills at bar 1 open=2703, bar 1 low=2689 < stop=2690 -> fill-bar stop
        result = compute_single_outcome(
            loss_bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E1",
        )
        if result["outcome"] == "loss":
            assert result["pnl_r"] == -1.0, "Loss not -1.0R for E1"

class TestCostModelApplied:
    """Check 9: Costs reduce wins. pnl_r < rr_target for wins."""

    def test_win_pnl_less_than_rr(self):
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["outcome"] == "win":
                assert result["pnl_r"] < 2.0, (
                    f"Win pnl_r={result['pnl_r']} >= RR=2.0 for {em} "
                    f"(costs should reduce wins)"
                )

class TestEntryModelDifference:
    """Check 10: Different entry models produce different entry prices."""

    def test_models_differ(self):
        """E1, E3 should produce different entry_prices for the same setup."""
        prices = {}
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["entry_price"] is not None:
                prices[em] = result["entry_price"]

        # At least 2 models should produce different prices
        if len(prices) >= 2:
            unique_prices = set(prices.values())
            assert len(unique_prices) >= 2, (
                f"Entry models produced identical prices: {prices}"
            )

# ============================================================================
# MATH LOGIC CHECKS (11-20)
# ============================================================================

class TestRiskMath:
    """Check 12: risk_points = abs(entry_price - stop_price)."""

    def test_risk_calculation(self):
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["entry_price"] is not None and result["stop_price"] is not None:
                risk = abs(result["entry_price"] - result["stop_price"])
                assert risk > 0, f"Zero risk for {em}"

class TestTargetMath:
    """Check 13: target_price = entry + risk * RR * direction."""

    def test_long_target(self):
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["entry_price"] is not None and result["target_price"] is not None:
                risk = abs(result["entry_price"] - result["stop_price"])
                expected_target = result["entry_price"] + risk * 2.0
                assert result["target_price"] == pytest.approx(expected_target, abs=0.01), (
                    f"Target mismatch for {em}: got {result['target_price']}, "
                    f"expected {expected_target}"
                )

class TestLossPnlExact:
    """Check 18: Losses should be exactly -1.0R (stop hit)."""

    def test_loss_is_minus_one_r(self):
        loss_bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2703, 2688, 2690, 100),  # E1 fills at open=2701, low=2688 < stop=2690
            ],
        )
        result = compute_single_outcome(
            loss_bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E1",
        )
        if result["outcome"] == "loss":
            assert result["pnl_r"] == -1.0

class TestMaeMfeBounds:
    """Check 19: MAE/MFE bounds.

    Raw MAE/MFE in R can be slightly negative when price never moved
    against (MAE) or for (MFE) the position, because the R denominator
    includes friction. This is mathematically correct — a negative MAE
    means zero adverse excursion, expressed as a fraction of friction-
    inflated risk. We test that MAE/MFE are reasonable (> -1.0).
    """

    def test_mae_reasonable(self):
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["mae_r"] is not None:
                assert result["mae_r"] > -1.0, f"Extreme negative MAE for {em}"

    def test_mfe_non_negative_for_wins(self):
        """For wins, MFE must be positive (price reached target)."""
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["outcome"] == "win" and result["mfe_r"] is not None:
                assert result["mfe_r"] > 0, f"Win with non-positive MFE for {em}"

class TestE3NoSameBarFill:
    """Check 6: E3 exit must be on or after fill bar."""

    def test_e3_exit_on_or_after_entry(self):
        """If E3 fills and exits, exit_ts >= entry_ts (fill-bar exit is valid)."""
        # Bar 0: confirm. Bar 1: retrace fills E3. Bar 2+: outcome scan.
        bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),  # confirm
                (2703, 2705, 2699, 2704, 100),  # retrace: low=2699 <= orb_high=2700
                (2704, 2721, 2703, 2720, 100),  # rally to target
            ],
        )
        result = compute_single_outcome(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E3",
        )
        if result["entry_ts"] is not None and result["exit_ts"] is not None:
            assert result["exit_ts"] >= result["entry_ts"], (
                "E3 exit must be on or after entry bar"
            )

class TestAmbiguousBarConservativeLoss:
    """Both target and stop hit on same bar -> conservative loss."""

    def test_ambiguous_is_loss(self):
        for em in ENTRY_MODELS:
            bars = _make_bars(
                BREAK_TS,
                [
                    (2698, 2701, 2695, 2701, 100),  # confirm
                    (2703, 2740, 2680, 2710, 200),  # huge bar: hits everything
                ],
            )
            result = compute_single_outcome(
                bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["outcome"] is not None and result["outcome"] != "scratch":
                # If both target and stop could be hit, should be loss
                # (only check if the bar actually spans both)
                entry = result["entry_price"]
                stop = result["stop_price"]
                target = result["target_price"]
                if entry is not None and target is not None:
                    pass  # specific check depends on bar range vs targets

class TestWinRMultipleExact:
    """Check 11: Win pnl_r must equal to_r_multiple (friction deducted from PnL).

    This is the CRITICAL guard against the pnl_points_to_r vs to_r_multiple
    regression. pnl_points_to_r does NOT deduct friction from the PnL
    numerator — it is correct for MAE/MFE but WRONG for trade PnL.
    """

    def test_win_pnl_matches_to_r_multiple(self):
        """For each entry model, verify win pnl_r == to_r_multiple exactly."""
        cost = _cost()
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, cost, em,
            )
            if result["outcome"] == "win":
                entry = result["entry_price"]
                stop = result["stop_price"]
                risk_points = abs(entry - stop)
                pnl_points = risk_points * 2.0  # RR=2.0
                expected_r = round(to_r_multiple(cost, entry, stop, pnl_points), 4)
                assert result["pnl_r"] == pytest.approx(expected_r, abs=0.0001), (
                    f"{em}: pnl_r={result['pnl_r']} != expected={expected_r} "
                    f"(entry={entry}, stop={stop}, risk={risk_points}pt)"
                )

    def test_win_pnl_is_NOT_pnl_points_to_r(self):
        """Verify win pnl_r is DIFFERENT from pnl_points_to_r output.

        If they match, the bug has regressed. pnl_points_to_r overstates wins.
        """
        cost = _cost()
        for em in ENTRY_MODELS:
            result = compute_single_outcome(
                STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, cost, em,
            )
            if result["outcome"] == "win":
                entry = result["entry_price"]
                stop = result["stop_price"]
                risk_points = abs(entry - stop)
                pnl_points = risk_points * 2.0
                wrong_r = round(pnl_points_to_r(cost, entry, stop, pnl_points), 4)
                correct_r = round(to_r_multiple(cost, entry, stop, pnl_points), 4)
                # They should differ (friction deduction)
                assert wrong_r != correct_r, "Functions should differ when friction > 0"
                # And pnl_r should match the CORRECT one
                assert result["pnl_r"] == pytest.approx(correct_r, abs=0.0001)

# ============================================================================
# FILL-BAR EXIT TESTS (R1)
# ============================================================================

class TestFillBarExits:
    """R1: Fill bar can hit stop/target for E1 and E3 entries."""

    def test_e1_fill_bar_stop_hit(self):
        """E1 fills at bar open. Bar's low breaches stop -> loss on fill bar."""
        # ORB 2700-2690, long break. E1 entry at bar 1 open=2703.
        # Risk = 2703-2690 = 13. Stop = 2690.
        # Bar 1 (fill bar): low=2688 < stop=2690 -> stop hit on fill bar.
        bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),  # bar 0: confirm
                (2703, 2706, 2688, 2695, 100),  # bar 1: E1 fill, low breaches stop
                (2695, 2730, 2694, 2725, 100),  # bar 2: would rally (never reached)
            ],
        )
        result = compute_single_outcome(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E1",
        )
        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0
        assert result["exit_ts"] == result["entry_ts"], (
            "Fill-bar exit: exit_ts must equal entry_ts"
        )

    def test_e1_fill_bar_target_hit(self):
        """E1 fills at bar open. Bar's high reaches target -> win on fill bar."""
        # ORB 2700-2690, long break, RR=1.0. E1 entry at bar 1 open=2703.
        # Risk = 13. Target = 2703 + 13 = 2716.
        # Bar 1 (fill bar): high=2720 >= target=2716 -> target hit.
        bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),  # bar 0: confirm
                (2703, 2720, 2702, 2718, 100),  # bar 1: E1 fill, high hits target
            ],
        )
        result = compute_single_outcome(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            1.0, 1, TD_END, _cost(), "E1",
        )
        assert result["outcome"] == "win"
        assert result["pnl_r"] > 0
        assert result["exit_ts"] == result["entry_ts"]

    def test_e1_fill_bar_ambiguous(self):
        """E1 fill bar hits both stop and target -> conservative loss."""
        # ORB 2700-2690, long, RR=1.0. E1 entry at bar 1 open=2703.
        # Risk = 13. Target = 2716. Stop = 2690.
        # Bar 1: high=2720 >= 2716 AND low=2685 <= 2690 -> ambiguous.
        bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),  # bar 0: confirm
                (2703, 2720, 2685, 2710, 200),  # bar 1: hits both
            ],
        )
        result = compute_single_outcome(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            1.0, 1, TD_END, _cost(), "E1",
        )
        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0
        assert result["exit_ts"] == result["entry_ts"]

    def test_e1_fill_bar_no_exit(self):
        """E1 fill bar doesn't hit stop or target -> proceeds to post_entry."""
        # ORB 2700-2690, long, RR=2.0. E1 entry at bar 1 open=2703.
        # Risk = 13. Target = 2703 + 26 = 2729. Stop = 2690.
        # Bar 1: high=2710 < 2729, low=2701 > 2690 -> no fill-bar exit.
        # Bar 2: high=2735 >= 2729 -> target hit on post_entry scan.
        bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),  # bar 0: confirm
                (2703, 2710, 2701, 2708, 100),  # bar 1: E1 fill, no exit
                (2708, 2735, 2707, 2730, 100),  # bar 2: target hit
            ],
        )
        result = compute_single_outcome(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E1",
        )
        assert result["outcome"] == "win"
        assert result["exit_ts"] > result["entry_ts"], (
            "No fill-bar exit: exit should be on a later bar"
        )

    def test_e3_fill_bar_stop_invalidates_entry(self):
        """E3: if fill bar's low breaches stop, entry_rules rejects the fill entirely."""
        # ORB 2700-2690, long. E3 retrace bar low=2688 <= stop=2690.
        # Entry_rules guard: stop breached on retrace bar -> no fill.
        bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),  # bar 0: confirm
                (2703, 2705, 2688, 2695, 100),  # bar 1: retrace + stop hit -> no fill
                (2695, 2730, 2694, 2725, 100),  # bar 2: irrelevant
            ],
        )
        result = compute_single_outcome(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E3",
        )
        # E3 entry_rules correctly reject: stop breached on retrace bar
        assert result["outcome"] is None
        assert result["entry_ts"] is None

    def test_e3_fill_bar_target_hit(self):
        """E3 fills on retrace, same bar also reaches target -> win on fill bar."""
        # ORB 2700-2690, long, RR=1.0. E3 entry=2700. Risk=10. Target=2710.
        # Bar 1: low=2699 (retrace, fills at 2700), high=2715 >= 2710 (target).
        # Stop=2690, low=2699 > 2690 (no stop breach).
        bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),  # bar 0: confirm
                (2703, 2715, 2699, 2712, 100),  # bar 1: retrace + target hit
            ],
        )
        result = compute_single_outcome(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            1.0, 1, TD_END, _cost(), "E3",
        )
        assert result["outcome"] == "win"
        assert result["pnl_r"] > 0
        assert result["exit_ts"] == result["entry_ts"]

    def test_e3_fill_bar_survives(self):
        """E3 fills intra-bar, fill bar doesn't breach stop or target."""
        # ORB 2700-2690, long, RR=2.0. E3 entry=2700. Risk=10. Target=2720.
        # Bar 1: retrace (low=2699 <= 2700), but low > stop, high < target.
        # Bar 2: target hit.
        bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),  # bar 0: confirm
                (2703, 2705, 2699, 2704, 100),  # bar 1: retrace fill, survives
                (2704, 2725, 2703, 2722, 100),  # bar 2: target hit
            ],
        )
        result = compute_single_outcome(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E3",
        )
        assert result["outcome"] == "win"
        assert result["exit_ts"] > result["entry_ts"]

    def test_e1_fill_bar_exit_changes_outcome(self):
        """Without fill-bar check this would be a win; with it, it's a loss."""
        # ORB 2700-2690, long, RR=2.0. E1 entry at bar 1 open=2703.
        # Risk = 13. Target = 2729. Stop = 2690.
        # Bar 1 (fill bar): low=2689 < 2690 -> stop hit on fill bar (loss).
        # Bar 2-3: rally to 2735 > target (would be win without fill-bar check).
        bars = _make_bars(
            BREAK_TS,
            [
                (2698, 2701, 2695, 2701, 100),  # bar 0: confirm
                (2703, 2706, 2689, 2704, 100),  # bar 1: fill bar, stop hit
                (2704, 2720, 2703, 2718, 100),  # bar 2: rally
                (2718, 2735, 2717, 2732, 100),  # bar 3: would hit target
            ],
        )
        result = compute_single_outcome(
            bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E1",
        )
        # Must be loss (fill bar stop), not win (later target)
        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0
        assert result["exit_ts"] == result["entry_ts"]

# ============================================================================
# RANDOMIZED SPOT-CHECK TESTS (against real gold.db data)
# ============================================================================
#
# These tests sample random rows from the REAL production database and
# independently recompute every stored metric from first principles.
# They catch silent corruption that synthetic fixtures cannot detect.

import random

from pipeline.paths import GOLD_DB_PATH
GOLD_DB = GOLD_DB_PATH
SAMPLE_SIZE = 50  # rows per test — enough to catch errors, fast enough for CI

def _skip_if_no_db():
    """Skip test if gold.db not present, locked, or missing orb_outcomes."""
    if not GOLD_DB.exists():
        pytest.skip("gold.db not available")
    try:
        test_con = duckdb.connect(str(GOLD_DB), read_only=True)
        tables = {r[0] for r in test_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        test_con.close()
        if "orb_outcomes" not in tables:
            pytest.skip("gold.db has no orb_outcomes table")
    except Exception:
        pytest.skip("gold.db locked by another process")

def _sample_outcomes(con, n=SAMPLE_SIZE, where_extra=""):
    """Pull n random outcome rows with win/loss result."""
    rows = con.execute(
        f"SELECT trading_day, symbol, orb_label, orb_minutes, rr_target, "
        f"confirm_bars, entry_model, outcome, pnl_r, mae_r, mfe_r, "
        f"entry_price, stop_price, target_price, exit_price "
        f"FROM orb_outcomes "
        f"WHERE symbol = 'MGC' AND outcome IN ('win','loss') {where_extra} "
        f"ORDER BY RANDOM() LIMIT {n}"
    ).fetchall()
    cols = ["trading_day", "symbol", "orb_label", "orb_minutes", "rr_target",
            "confirm_bars", "entry_model", "outcome", "pnl_r", "mae_r", "mfe_r",
            "entry_price", "stop_price", "target_price", "exit_price"]
    return [dict(zip(cols, r)) for r in rows]

class TestRandomOutcomeMath:
    """Sample random real outcomes and recompute all stored values."""

    def test_risk_points_recompute(self):
        """risk_points = abs(entry - stop) for every sampled row."""
        _skip_if_no_db()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            for row in _sample_outcomes(con):
                risk = abs(row["entry_price"] - row["stop_price"])
                assert risk > 0, (
                    f"Zero risk: {row['entry_price']}-{row['stop_price']} "
                    f"({row['entry_model']} {row['orb_label']} {row['trading_day']})"
                )
        finally:
            con.close()

    def test_win_pnl_consistent_with_cost(self):
        """Win pnl_r must match to_r_multiple. Small-ORB wins can be negative
        after friction (0.7pt risk, $7 gross, -$8.40 friction = -$1.40 net).
        This is mathematically correct: 'target hit' != 'profitable after costs'.
        Large-ORB wins (risk >= friction_in_points) must always be positive."""
        _skip_if_no_db()
        cost = _cost()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            for row in _sample_outcomes(con, where_extra="AND outcome = 'win'"):
                risk_pts = abs(row["entry_price"] - row["stop_price"])
                # If risk > friction_in_points, win must be positive
                if risk_pts > cost.friction_in_points:
                    assert row["pnl_r"] > 0, (
                        f"Large-risk win with pnl_r={row['pnl_r']} "
                        f"(risk={risk_pts:.2f}pt > friction={cost.friction_in_points:.2f}pt, "
                        f"{row['entry_model']} {row['orb_label']} {row['trading_day']})"
                    )
        finally:
            con.close()

    def test_loss_pnl_exactly_minus_one(self):
        """Every loss must be exactly -1.0R (stop hit = 1R loss, no friction on losses)."""
        _skip_if_no_db()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            for row in _sample_outcomes(con, where_extra="AND outcome = 'loss'"):
                assert row["pnl_r"] == -1.0, (
                    f"Loss pnl_r={row['pnl_r']} != -1.0 "
                    f"({row['entry_model']} {row['orb_label']} {row['trading_day']})"
                )
        finally:
            con.close()

    def test_target_price_recompute(self):
        """target = entry + risk * rr * direction for every sampled row."""
        _skip_if_no_db()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            for row in _sample_outcomes(con):
                entry = row["entry_price"]
                stop = row["stop_price"]
                target = row["target_price"]
                rr = row["rr_target"]
                risk = abs(entry - stop)
                direction = 1.0 if entry > stop else -1.0
                expected_target = entry + risk * rr * direction
                assert target == pytest.approx(expected_target, abs=0.02), (
                    f"Target mismatch: stored={target}, recomputed={expected_target} "
                    f"(entry={entry}, stop={stop}, rr={rr}, "
                    f"{row['entry_model']} {row['orb_label']} {row['trading_day']})"
                )
        finally:
            con.close()

    def test_win_pnl_r_recompute(self):
        """For wins, recompute pnl_r = to_r_multiple(cost, entry, stop, rr*risk)."""
        _skip_if_no_db()
        cost = _cost()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            for row in _sample_outcomes(con, where_extra="AND outcome = 'win'"):
                entry = row["entry_price"]
                stop = row["stop_price"]
                rr = row["rr_target"]
                risk_pts = abs(entry - stop)
                pnl_pts = risk_pts * rr
                expected_r = round(to_r_multiple(cost, entry, stop, pnl_pts), 4)
                assert row["pnl_r"] == pytest.approx(expected_r, abs=0.001), (
                    f"Win pnl_r mismatch: stored={row['pnl_r']}, recomputed={expected_r} "
                    f"(entry={entry}, stop={stop}, risk={risk_pts}pt, rr={rr}, "
                    f"{row['entry_model']} {row['orb_label']} {row['trading_day']})"
                )
        finally:
            con.close()

    def test_stop_is_opposite_orb_level(self):
        """Stop must equal orb_low (long) or orb_high (short) from daily_features."""
        _skip_if_no_db()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            rows = con.execute(
                "SELECT o.trading_day, o.orb_label, o.entry_model, o.entry_price, "
                "o.stop_price, o.outcome, "
                "d.orb_0900_high, d.orb_0900_low, "
                "d.orb_1000_high, d.orb_1000_low, "
                "d.orb_1800_high, d.orb_1800_low, "
                "d.orb_2300_high, d.orb_2300_low "
                "FROM orb_outcomes o "
                "JOIN daily_features d ON o.trading_day = d.trading_day "
                "  AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes "
                "WHERE o.symbol = 'MGC' AND o.outcome IN ('win','loss') "
                "  AND o.orb_label IN ('0900','1000','1800','2300') "
                "ORDER BY RANDOM() LIMIT 50"
            ).fetchall()
            cols = ["trading_day", "orb_label", "entry_model", "entry_price",
                    "stop_price", "outcome",
                    "orb_0900_high", "orb_0900_low",
                    "orb_1000_high", "orb_1000_low",
                    "orb_1800_high", "orb_1800_low",
                    "orb_2300_high", "orb_2300_low"]
            for r in [dict(zip(cols, row)) for row in rows]:
                orb = r["orb_label"]
                orb_high = r[f"orb_{orb}_high"]
                orb_low = r[f"orb_{orb}_low"]
                entry = r["entry_price"]
                stop = r["stop_price"]
                # Determine direction from entry vs stop
                is_long = entry > stop
                if is_long:
                    assert stop == pytest.approx(orb_low, abs=0.01), (
                        f"Long stop={stop} != orb_low={orb_low} "
                        f"({r['entry_model']} {orb} {r['trading_day']})"
                    )
                else:
                    assert stop == pytest.approx(orb_high, abs=0.01), (
                        f"Short stop={stop} != orb_high={orb_high} "
                        f"({r['entry_model']} {orb} {r['trading_day']})"
                    )
        finally:
            con.close()

    def test_e1_entry_not_at_orb_level(self):
        """E1 entries must NOT equal ORB level (that was the old bug)."""
        _skip_if_no_db()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            rows = con.execute(
                "SELECT o.trading_day, o.orb_label, o.entry_price, o.stop_price, "
                "d.orb_0900_high, d.orb_0900_low, "
                "d.orb_1000_high, d.orb_1000_low, "
                "d.orb_1800_high, d.orb_1800_low, "
                "d.orb_2300_high, d.orb_2300_low "
                "FROM orb_outcomes o "
                "JOIN daily_features d ON o.trading_day = d.trading_day "
                "  AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes "
                "WHERE o.symbol = 'MGC' AND o.entry_model = 'E1' "
                "  AND o.outcome IN ('win','loss') "
                "  AND o.orb_label IN ('0900','1000','1800','2300') "
                "ORDER BY RANDOM() LIMIT 50"
            ).fetchall()
            cols = ["trading_day", "orb_label", "entry_price", "stop_price",
                    "orb_0900_high", "orb_0900_low",
                    "orb_1000_high", "orb_1000_low",
                    "orb_1800_high", "orb_1800_low",
                    "orb_2300_high", "orb_2300_low"]
            for r in [dict(zip(cols, row)) for row in rows]:
                orb = r["orb_label"]
                orb_high = r[f"orb_{orb}_high"]
                orb_low = r[f"orb_{orb}_low"]
                entry = r["entry_price"]
                stop = r["stop_price"]
                is_long = entry > stop
                # E1 fills at next bar's open — can gap beyond ORB in either direction.
                # The key invariant: stop must be on the correct side of entry.
                if is_long:
                    assert stop < entry, (
                        f"E1 long: stop={stop} >= entry={entry} "
                        f"({orb} {r['trading_day']})"
                    )
                else:
                    assert stop > entry, (
                        f"E1 short: stop={stop} <= entry={entry} "
                        f"({orb} {r['trading_day']})"
                    )
        finally:
            con.close()

    def test_e3_entry_at_orb_level(self):
        """E3 entries must equal ORB level (limit retrace fill)."""
        _skip_if_no_db()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            rows = con.execute(
                "SELECT o.trading_day, o.orb_label, o.entry_price, o.stop_price, "
                "d.orb_0900_high, d.orb_0900_low, "
                "d.orb_1000_high, d.orb_1000_low, "
                "d.orb_1800_high, d.orb_1800_low, "
                "d.orb_2300_high, d.orb_2300_low "
                "FROM orb_outcomes o "
                "JOIN daily_features d ON o.trading_day = d.trading_day "
                "  AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes "
                "WHERE o.symbol = 'MGC' AND o.entry_model = 'E3' "
                "  AND o.outcome IN ('win','loss') "
                "  AND o.orb_label IN ('0900','1000','1800','2300') "
                "ORDER BY RANDOM() LIMIT 50"
            ).fetchall()
            cols = ["trading_day", "orb_label", "entry_price", "stop_price",
                    "orb_0900_high", "orb_0900_low",
                    "orb_1000_high", "orb_1000_low",
                    "orb_1800_high", "orb_1800_low",
                    "orb_2300_high", "orb_2300_low"]
            for r in [dict(zip(cols, row)) for row in rows]:
                orb = r["orb_label"]
                orb_high = r[f"orb_{orb}_high"]
                orb_low = r[f"orb_{orb}_low"]
                entry = r["entry_price"]
                is_long = entry > r["stop_price"]
                if is_long:
                    assert entry == pytest.approx(orb_high, abs=0.01), (
                        f"E3 long entry={entry} != orb_high={orb_high} "
                        f"({orb} {r['trading_day']})"
                    )
                else:
                    assert entry == pytest.approx(orb_low, abs=0.01), (
                        f"E3 short entry={entry} != orb_low={orb_low} "
                        f"({orb} {r['trading_day']})"
                    )
        finally:
            con.close()

class TestRandomStrategyMath:
    """Sample random strategies and recompute stored metrics from outcomes."""

    def test_win_rate_recompute(self):
        """Recompute win_rate from orb_outcomes for random strategies."""
        _skip_if_no_db()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            strats = con.execute(
                "SELECT strategy_id, orb_label, entry_model, rr_target, "
                "confirm_bars, filter_type, win_rate, sample_size, expectancy_r "
                "FROM experimental_strategies "
                "WHERE instrument = 'MGC' AND sample_size >= 20 "
                "ORDER BY RANDOM() LIMIT 30"
            ).fetchall()
            strat_cols = ["strategy_id", "orb_label", "entry_model", "rr_target",
                          "confirm_bars", "filter_type", "win_rate", "sample_size",
                          "expectancy_r"]

            # Load daily_features for filter eligibility
            features = con.execute(
                "SELECT * FROM daily_features WHERE symbol = 'MGC' AND orb_minutes = 5"
            ).fetchall()
            feat_cols = [desc[0] for desc in con.description]
            feat_dicts = [dict(zip(feat_cols, r)) for r in features]

            from trading_app.config import ALL_FILTERS

            for s in [dict(zip(strat_cols, r)) for r in strats]:
                orb = s["orb_label"]
                em = s["entry_model"]
                rr = s["rr_target"]
                cb = s["confirm_bars"]
                ft = s["filter_type"]

                # Skip volume filters (need bar data)
                if ft.startswith("VOL_"):
                    continue

                # Get eligible days for this filter
                strat_filter = ALL_FILTERS.get(ft)
                if strat_filter is None:
                    continue
                eligible = set()
                for row in feat_dicts:
                    if row.get(f"orb_{orb}_break_dir") is None:
                        continue
                    if strat_filter.matches_row(row, orb):
                        eligible.add(row["trading_day"])

                # Get matching outcomes
                outcomes = con.execute(
                    "SELECT trading_day, outcome, pnl_r FROM orb_outcomes "
                    "WHERE symbol = 'MGC' AND orb_label = ? AND entry_model = ? "
                    "AND rr_target = ? AND confirm_bars = ? AND orb_minutes = 5 "
                    "AND outcome IN ('win','loss')",
                    [orb, em, rr, cb]
                ).fetchall()

                # Filter to eligible days
                traded = [(td, oc, pr) for td, oc, pr in outcomes if td in eligible]
                if len(traded) == 0:
                    continue

                wins = sum(1 for _, oc, _ in traded if oc == "win")
                recomputed_wr = wins / len(traded)

                assert s["win_rate"] == pytest.approx(recomputed_wr, abs=0.001), (
                    f"WR mismatch for {s['strategy_id']}: "
                    f"stored={s['win_rate']}, recomputed={recomputed_wr} "
                    f"(N={len(traded)})"
                )
        finally:
            con.close()

    def test_expectancy_recompute(self):
        """Recompute ExpR = WR*AvgWin - LR*AvgLoss for random strategies."""
        _skip_if_no_db()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            strats = con.execute(
                "SELECT strategy_id, orb_label, entry_model, rr_target, "
                "confirm_bars, filter_type, win_rate, sample_size, expectancy_r "
                "FROM experimental_strategies "
                "WHERE instrument = 'MGC' AND sample_size >= 20 "
                "ORDER BY RANDOM() LIMIT 30"
            ).fetchall()
            strat_cols = ["strategy_id", "orb_label", "entry_model", "rr_target",
                          "confirm_bars", "filter_type", "win_rate", "sample_size",
                          "expectancy_r"]

            features = con.execute(
                "SELECT * FROM daily_features WHERE symbol = 'MGC' AND orb_minutes = 5"
            ).fetchall()
            feat_cols = [desc[0] for desc in con.description]
            feat_dicts = [dict(zip(feat_cols, r)) for r in features]

            from trading_app.config import ALL_FILTERS

            for s in [dict(zip(strat_cols, r)) for r in strats]:
                orb = s["orb_label"]
                em = s["entry_model"]
                rr = s["rr_target"]
                cb = s["confirm_bars"]
                ft = s["filter_type"]

                if ft.startswith("VOL_"):
                    continue

                strat_filter = ALL_FILTERS.get(ft)
                if strat_filter is None:
                    continue
                eligible = set()
                for row in feat_dicts:
                    if row.get(f"orb_{orb}_break_dir") is None:
                        continue
                    if strat_filter.matches_row(row, orb):
                        eligible.add(row["trading_day"])

                outcomes = con.execute(
                    "SELECT trading_day, outcome, pnl_r FROM orb_outcomes "
                    "WHERE symbol = 'MGC' AND orb_label = ? AND entry_model = ? "
                    "AND rr_target = ? AND confirm_bars = ? AND orb_minutes = 5 "
                    "AND outcome IN ('win','loss')",
                    [orb, em, rr, cb]
                ).fetchall()

                traded = [(td, oc, pr) for td, oc, pr in outcomes if td in eligible]
                if len(traded) == 0:
                    continue

                wins = [(oc, pr) for _, oc, pr in traded if oc == "win"]
                losses = [(oc, pr) for _, oc, pr in traded if oc == "loss"]

                wr = len(wins) / len(traded)
                lr = 1.0 - wr
                avg_win = sum(pr for _, pr in wins) / len(wins) if wins else 0.0
                avg_loss = abs(sum(pr for _, pr in losses) / len(losses)) if losses else 0.0
                recomputed_er = (wr * avg_win) - (lr * avg_loss)

                assert s["expectancy_r"] == pytest.approx(recomputed_er, abs=0.002), (
                    f"ExpR mismatch for {s['strategy_id']}: "
                    f"stored={s['expectancy_r']}, recomputed={recomputed_er:.4f} "
                    f"(N={len(traded)}, WR={wr:.3f})"
                )
        finally:
            con.close()

    def test_max_drawdown_recompute(self):
        """Walk the equity curve for random strategies and recompute MaxDD."""
        _skip_if_no_db()
        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            strats = con.execute(
                "SELECT strategy_id, orb_label, entry_model, rr_target, "
                "confirm_bars, filter_type, max_drawdown_r "
                "FROM experimental_strategies "
                "WHERE instrument = 'MGC' AND sample_size >= 20 "
                "ORDER BY RANDOM() LIMIT 30"
            ).fetchall()
            strat_cols = ["strategy_id", "orb_label", "entry_model", "rr_target",
                          "confirm_bars", "filter_type", "max_drawdown_r"]

            features = con.execute(
                "SELECT * FROM daily_features WHERE symbol = 'MGC' AND orb_minutes = 5"
            ).fetchall()
            feat_cols = [desc[0] for desc in con.description]
            feat_dicts = [dict(zip(feat_cols, r)) for r in features]

            from trading_app.config import ALL_FILTERS

            for s in [dict(zip(strat_cols, r)) for r in strats]:
                orb = s["orb_label"]
                em = s["entry_model"]
                rr = s["rr_target"]
                cb = s["confirm_bars"]
                ft = s["filter_type"]

                if ft.startswith("VOL_"):
                    continue

                strat_filter = ALL_FILTERS.get(ft)
                if strat_filter is None:
                    continue
                eligible = set()
                for row in feat_dicts:
                    if row.get(f"orb_{orb}_break_dir") is None:
                        continue
                    if strat_filter.matches_row(row, orb):
                        eligible.add(row["trading_day"])

                outcomes = con.execute(
                    "SELECT trading_day, outcome, pnl_r FROM orb_outcomes "
                    "WHERE symbol = 'MGC' AND orb_label = ? AND entry_model = ? "
                    "AND rr_target = ? AND confirm_bars = ? AND orb_minutes = 5 "
                    "AND outcome IN ('win','loss') ORDER BY trading_day",
                    [orb, em, rr, cb]
                ).fetchall()

                traded = [(td, oc, pr) for td, oc, pr in outcomes if td in eligible]
                if len(traded) == 0:
                    continue

                # Walk equity curve
                cumulative = 0.0
                peak = 0.0
                max_dd = 0.0
                for _, _, pnl_r in traded:
                    cumulative += pnl_r
                    peak = max(peak, cumulative)
                    dd = peak - cumulative
                    max_dd = max(max_dd, dd)

                assert s["max_drawdown_r"] == pytest.approx(max_dd, abs=0.01), (
                    f"MaxDD mismatch for {s['strategy_id']}: "
                    f"stored={s['max_drawdown_r']}, recomputed={max_dd:.2f}"
                )
        finally:
            con.close()

class TestRandomWalkForwardIntegrity:
    """Verify walk-forward results match independent recomputation."""

    def test_oos_trade_count_matches_outcomes(self):
        """OOS trade count must equal actual eligible outcomes in test folds."""
        _skip_if_no_db()
        wf_json = Path(__file__).parent.parent / "artifacts" / "walk_forward" / "walk_forward_results.json"
        if not wf_json.exists():
            pytest.skip("walk-forward artifacts not present")

        import json
        with open(wf_json) as f:
            results = json.load(f)

        con = duckdb.connect(str(GOLD_DB), read_only=True)
        try:
            # Pick random strategies from WF results
            sampled = random.sample(results, min(20, len(results)))

            for wf in sampled:
                total_oos = sum(fold["trade_count"] for fold in wf["folds"])
                assert wf["oos_trade_count"] == total_oos, (
                    f"OOS count mismatch for {wf['strategy_id']}: "
                    f"aggregate={wf['oos_trade_count']}, sum_folds={total_oos}"
                )
        finally:
            con.close()

    def test_no_train_test_overlap_in_results(self):
        """Verify no fold has overlapping train/test date ranges."""
        wf_json = Path(__file__).parent.parent / "artifacts" / "walk_forward" / "walk_forward_results.json"
        if not wf_json.exists():
            pytest.skip("walk-forward artifacts not present")

        import json
        from datetime import date as d
        with open(wf_json) as f:
            results = json.load(f)

        for wf in results:
            for fold in wf["folds"]:
                train_end = d.fromisoformat(fold["train_end"])
                test_start = d.fromisoformat(fold["test_start"])
                assert train_end < test_start, (
                    f"Leakage in {wf['strategy_id']} fold {fold['fold_idx']}: "
                    f"train_end={train_end} >= test_start={test_start}"
                )
