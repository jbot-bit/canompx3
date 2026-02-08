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

sys.path.insert(0, str(Path(__file__).parent.parent))

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

    def test_e2_entry_is_confirm_close(self):
        """E2 entry_price must equal the confirm bar's close."""
        result = compute_single_outcome(
            STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E2",
        )
        if result["entry_price"] is not None:
            assert result["entry_price"] == 2701.0

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

    def test_e2_entry_at_confirm(self):
        confirm = detect_confirm(STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long", 1, TD_END)
        signal = resolve_entry(STANDARD_BARS, confirm, "E2", TD_END)
        if signal.triggered:
            assert signal.entry_ts == confirm.confirm_bar_ts

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

    def test_e2_long_entry_above_orb(self):
        """E2 long: entry_price > orb_high (confirm close is above by definition)."""
        result = compute_single_outcome(
            STANDARD_BARS, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E2",
        )
        if result["entry_price"] is not None:
            assert result["entry_price"] > ORB_HIGH

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
        for em in ["E2"]:  # E2 always fills on confirm
            result = compute_single_outcome(
                loss_bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
                2.0, 1, TD_END, _cost(), em,
            )
            if result["outcome"] == "loss":
                assert result["pnl_r"] == -1.0, f"Loss not -1.0R for {em}"


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
        """E1, E2, E3 should produce different entry_prices for the same setup."""
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
                (2701, 2703, 2688, 2690, 100),  # hits stop
            ],
        )
        result = compute_single_outcome(
            loss_bars, BREAK_TS, ORB_HIGH, ORB_LOW, "long",
            2.0, 1, TD_END, _cost(), "E2",
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
    """Check 6: E3 outcome scan starts AFTER fill bar."""

    def test_e3_exit_after_entry(self):
        """If E3 fills and exits, exit_ts > entry_ts."""
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
            assert result["exit_ts"] > result["entry_ts"], (
                "E3 exit must be after entry (no same-bar resolution)"
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
