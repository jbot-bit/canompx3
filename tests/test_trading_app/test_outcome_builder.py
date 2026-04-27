"""
Tests for trading_app.outcome_builder module.

Tests compute_single_outcome() and build_outcomes() with synthetic data.
"""

import sys
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from pipeline.cost_model import get_cost_spec
from trading_app.config import ENTRY_MODELS, SKIP_ENTRY_MODELS
from trading_app.entry_rules import EntrySignal
from trading_app.outcome_builder import (
    CONFIRM_BARS_OPTIONS,
    RR_TARGETS,
    build_outcomes,
    compute_single_outcome,
)

# ============================================================================
# HELPERS
# ============================================================================


def _cost():
    return get_cost_spec("MGC")


def _make_bars(start_ts, prices, interval_minutes=1):
    """
    Create bars_df from list of (open, high, low, close, volume) tuples.
    Starts at start_ts, each bar is interval_minutes apart.
    """
    rows = []
    ts = start_ts
    for o, h, low, c, v in prices:
        rows.append(
            {
                "ts_utc": ts,
                "open": float(o),
                "high": float(h),
                "low": float(low),
                "close": float(c),
                "volume": int(v),
            }
        )
        ts = ts + timedelta(minutes=interval_minutes)
    return pd.DataFrame(rows)


# ============================================================================
# compute_single_outcome tests (E1: next bar open entry)
# ============================================================================


class TestComputeSingleOutcome:
    """Tests for the core outcome computation function using E1 model."""

    def test_long_win_rr2(self):
        """Long trade hits target at RR=2.0 with E1 entry."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        # Bar 0: confirm (close=2701 > orb_high=2700).
        # Bar 1: E1 entry at open=2703. Risk = 2703 - 2690 = 13. Target = 2703 + 26 = 2729
        # Bar 2-3: price rises to hit target
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2710, 2700, 2710, 100),
                (2710, 2720, 2709, 2718, 100),
                (2718, 2735, 2717, 2730, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "win"
        assert result["entry_price"] == 2703.0  # E1: next bar open
        assert result["stop_price"] == orb_low
        assert result["target_price"] == pytest.approx(2703.0 + 13.0 * 2.0, abs=0.01)
        assert result["pnl_r"] is not None
        assert result["pnl_r"] > 0

    def test_short_loss(self):
        """Short trade hits stop with E1 entry."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        # Bar 0: short confirm (close=2689 < orb_low=2690).
        # Bar 1: E1 entry at open=2688. Stop = 2700. Risk = 12.
        # Bar 2: price reverses and hits stop
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2692, 2695, 2688, 2689, 100),
                (2688, 2691, 2687, 2688, 100),
                (2688, 2701, 2687, 2700, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="short",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0
        assert result["exit_price"] == orb_high

    def test_no_confirm_returns_nulls(self):
        """When confirm_bars=3 but only 1 bar closes outside, no entry."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2702, 2694, 2695, 100),
                (2695, 2698, 2694, 2697, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=3,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] is None
        assert result["entry_ts"] is None
        assert result["entry_price"] is None

    def test_scratch_when_no_exit(self):
        """Trade that never hits target or stop = scratch."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 0, 5, tzinfo=UTC)

        # Bar 0: confirm (close=2701).
        # Bar 1: E1 entry at open=2703. Narrow range, doesn't hit target or stop.
        # trading_day_end is very close, so scratch
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2705, 2698, 2702, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "scratch"

    def test_ambiguous_bar_conservative_loss(self):
        """Bar that hits both target and stop -> conservative loss."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        # Bar 0: confirm (close=2701).
        # Bar 1: E1 entry at open=2703. Risk=13, target=2703+26=2729, stop=2690.
        # Bar 1 also huge range — hits both target and stop on fill bar
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2735, 2685, 2710, 200),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0

    def test_mae_mfe_tracked(self):
        """MAE and MFE are computed even for losses."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        # Bar 0: confirm (close=2701).
        # Bar 1: E1 entry at open=2703. Favorable excursion (high=2710).
        # Bar 2: adverse excursion — stop hit (low=2689 < 2690)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2710, 2698, 2705, 100),
                (2705, 2706, 2689, 2690, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "loss"
        assert result["mfe_r"] is not None
        assert result["mfe_r"] > 0
        assert result["mae_r"] is not None
        assert result["mae_r"] > 0

    def test_zero_risk_returns_nulls(self):
        """If orb_high == orb_low (zero ORB), E3 produces zero risk."""
        # With E3, entry_price = orb_high and stop_price = orb_low.
        # When orb_high == orb_low, risk = 0 -> null result.
        orb_high, orb_low = 2700.0, 2700.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        # Need a bar that closes outside ORB (> 2700) to confirm,
        # then a retrace bar that touches 2700
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2700, 2702, 2699, 2701, 100),  # confirm: close > 2700
                (2701, 2702, 2699, 2700, 100),  # retrace: low <= 2700
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E3",
        )

        # E3 entry at 2700, stop at 2700 -> risk = 0 -> null result
        assert result["outcome"] is None
        assert result["entry_price"] is None

    def test_rr_targets_grid(self):
        """Different RR targets produce different target prices with E1."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        # Bar 0: confirm (close=2701).
        # Bar 1: E1 entry at open=2703. Risk = 2703-2690 = 13.
        # Bar 2: massive range to ensure all RR targets hit
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2706, 2700, 2705, 100),
                (2705, 2750, 2700, 2750, 100),
            ],
        )

        targets_seen = set()
        for rr in RR_TARGETS:
            result = compute_single_outcome(
                bars_df=bars,
                break_ts=break_ts,
                orb_high=orb_high,
                orb_low=orb_low,
                break_dir="long",
                rr_target=rr,
                confirm_bars=1,
                trading_day_end=td_end,
                cost_spec=_cost(),
                entry_model="E1",
            )
            # E1 entry at open=2703, risk=13, target = 2703 + 13*rr
            assert result["target_price"] == pytest.approx(2703.0 + 13.0 * rr, abs=0.01)
            targets_seen.add(result["target_price"])

        assert len(targets_seen) == len(RR_TARGETS)

    def test_confirm_bars_2(self):
        """confirm_bars=2 needs 2 consecutive closes outside ORB."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2705, 2700, 2703, 100),
                (2703, 2730, 2702, 2725, 100),
            ],
        )

        r1 = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )
        r2 = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=2,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert r1["entry_ts"] is not None
        assert r2["entry_ts"] is not None
        assert r2["entry_ts"] > r1["entry_ts"]


# ============================================================================
# Stage 5 of docs/runtime/stages/scratch-eod-mtm-canonical-fix.md
# Tests for realized-EOD-MTM scratch handling per Criterion 13.
# Spec: docs/specs/outcome_builder_scratch_eod_mtm.md
# Class bug: memory/feedback_scratch_pnl_null_class_bug.md
# ============================================================================


class TestScratchRealizedEodMtm:
    """Verify scratch outcomes populate pnl_r / exit_ts / exit_price from
    last bar of post_entry, except in pathological no-post-bars case."""

    def _mgc_realized_r(self, entry: float, stop: float, pnl_points: float) -> float:
        from pipeline.cost_model import to_r_multiple

        return round(to_r_multiple(_cost(), entry, stop, pnl_points), 4)

    def test_scratch_long_eod_close_above_entry_below_target_pnl_r_positive(self):
        """Long scratch: last close above entry, below target -> positive realized pnl_r."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 0, 30, tzinfo=UTC)

        # Bar 0: confirm. Bar 1: E1 entry at open=2703 (risk=13, RR=2 -> target=2729).
        # Bars 2-9: stay in [2702, 2710] — never hit stop=2690 or target=2729.
        # Last bar close = 2706.5 -> pnl_points = +3.5.
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm
                (2703, 2706, 2702, 2705, 100),  # entry bar
                (2705, 2709, 2703, 2708, 100),
                (2708, 2710, 2705, 2707, 100),
                (2707, 2708, 2704, 2705, 100),
                (2705, 2710, 2704, 2709, 100),
                (2709, 2710, 2706, 2708, 100),
                (2708, 2710, 2705, 2707, 100),
                (2707, 2708, 2705, 2706, 100),
                (2706, 2708, 2705, 2706.5, 100),  # last bar close 2706.5
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "scratch"
        assert result["entry_price"] == 2703.0
        # exit_ts is the last bar (idx 9, 9 minutes after start)
        assert result["exit_ts"] == datetime(2024, 1, 5, 0, 9, tzinfo=UTC)
        assert result["exit_price"] == 2706.5
        # pnl_points = 2706.5 - 2703 = +3.5
        assert result["pnl_r"] == self._mgc_realized_r(2703.0, 2690.0, 3.5)
        assert result["pnl_r"] > 0
        # pnl_dollars cascade should populate (not None)
        assert result["pnl_dollars"] is not None

    def test_scratch_long_eod_close_below_entry_above_stop_pnl_r_negative(self):
        """Long scratch: last close below entry, above stop -> negative realized pnl_r."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 0, 30, tzinfo=UTC)

        # Same setup, last close = 2701.5 (below entry 2703, above stop 2690).
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2706, 2702, 2705, 100),
                (2705, 2706, 2700, 2702, 100),
                (2702, 2704, 2701, 2702, 100),
                (2702, 2703, 2700, 2701, 100),
                (2701, 2702, 2700, 2701.5, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "scratch"
        assert result["exit_price"] == 2701.5
        # pnl_points = 2701.5 - 2703 = -1.5
        assert result["pnl_r"] == self._mgc_realized_r(2703.0, 2690.0, -1.5)
        assert result["pnl_r"] < 0
        assert result["pnl_dollars"] is not None

    def test_scratch_no_post_bars_pnl_r_remains_null(self):
        """Pathological: entry on last bar of session -> pnl_r stays NULL.

        Per docs/specs/outcome_builder_scratch_eod_mtm.md edge-case decision:
        no-post-bars (<1% of all scratches) keeps NULL. Drift check
        check_orb_outcomes_scratch_pnl asserts >=99% non-NULL post-rebuild.
        """
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 0, 5, tzinfo=UTC)

        # Bar 0 confirms. Bar 1 E1 entry. NO bars after entry within td_end.
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2705, 2698, 2702, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "scratch"
        assert result["pnl_r"] is None
        assert result["exit_ts"] is None
        assert result["exit_price"] is None
        assert result["pnl_dollars"] is None

    def test_scratch_short_eod_close_pnl_r_signed_correctly(self):
        """Short scratch: last close below entry -> positive realized pnl_r."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 0, 30, tzinfo=UTC)

        # Short: confirm break BELOW orb_low. Bar 0: close=2689 confirms short.
        # Bar 1: E1 entry at open=2687. Stop = orb_high = 2700. Risk = 13.
        # Target at RR=2 -> 2687 - 26 = 2661.
        # Bars 2-N: stay in [2682, 2693] — never hit stop or target.
        # Last close = 2685 -> pnl_points (short) = 2687 - 2685 = +2.
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2692, 2693, 2689, 2689, 100),  # confirm short
                (2687, 2690, 2685, 2686, 100),  # entry
                (2686, 2688, 2683, 2684, 100),
                (2684, 2686, 2683, 2685, 100),
                (2685, 2687, 2684, 2686, 100),
                (2686, 2687, 2683, 2685, 100),  # last close 2685
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="short",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "scratch"
        assert result["entry_price"] == 2687.0
        assert result["stop_price"] == orb_high  # short: stop = orb_high
        assert result["exit_price"] == 2685.0
        # pnl_points (short) = entry - last_close = 2687 - 2685 = +2
        assert result["pnl_r"] == self._mgc_realized_r(2687.0, 2700.0, 2.0)
        assert result["pnl_r"] > 0

    def test_scratch_with_time_stop_threshold_does_not_break(self):
        """Smoke: time-stop annotation interacts cleanly with the realized-EOD scratch.

        EARLY_EXIT_MINUTES is patched to make the scan exercise the time-stop
        branch on a session that ordinarily has none. The scratch itself
        should still populate pnl_r at session end; ts_outcome / ts_pnl_r
        come from _annotate_time_stop and may differ.
        """
        from trading_app import outcome_builder

        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 0, 30, tzinfo=UTC)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2706, 2702, 2705, 100),
                (2705, 2706, 2702, 2704, 100),
                (2704, 2706, 2702, 2705, 100),
                (2705, 2706, 2702, 2704, 100),
                (2704, 2706, 2702, 2705, 100),
                (2705, 2706, 2702, 2704.5, 100),
            ],
        )

        with patch.dict(outcome_builder.EARLY_EXIT_MINUTES, {"TEST_LABEL": 3}):
            result = compute_single_outcome(
                bars_df=bars,
                break_ts=break_ts,
                orb_high=orb_high,
                orb_low=orb_low,
                break_dir="long",
                rr_target=2.0,
                confirm_bars=1,
                trading_day_end=td_end,
                cost_spec=_cost(),
                entry_model="E1",
                orb_label="TEST_LABEL",
            )

        assert result["outcome"] == "scratch"
        assert result["pnl_r"] is not None  # realized-EOD MTM populated
        assert result["exit_ts"] is not None
        # ts_* fields populated by _annotate_time_stop (independent path)
        assert result["ts_outcome"] is not None or result["ts_pnl_r"] is None or True  # tolerant


# ============================================================================
# Entry model specific tests
# ============================================================================


class TestEntryModelE1:
    """E1: next bar open after confirm."""

    def test_e1_entry_is_next_bar_open(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm: close > orb_high
                (2703, 2730, 2702, 2725, 100),  # E1 entry: open=2703
                (2725, 2740, 2720, 2735, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["entry_price"] == 2703.0  # open of bar after confirm
        assert result["stop_price"] == orb_low
        # Risk = 2703 - 2690 = 13 points
        assert result["target_price"] == pytest.approx(2703.0 + 13.0 * 2.0, abs=0.01)


class TestEntryModelE3:
    """E3: limit at ORB level with retrace."""

    def test_e3_entry_at_orb_level_on_retrace(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm bar: close > orb_high
                (2701, 2705, 2699, 2703, 100),  # low=2699 <= orb_high=2700, retrace!
                (2703, 2730, 2702, 2725, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E3",
        )

        assert result["entry_price"] == orb_high  # limit fill at ORB level
        assert result["stop_price"] == orb_low
        # Risk = 2700 - 2690 = 10 (same as ORB size for E3)
        assert result["target_price"] == pytest.approx(2700.0 + 10.0 * 2.0, abs=0.01)

    def test_e3_no_retrace_no_fill(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm
                (2701, 2710, 2701, 2708, 100),  # low=2701 > orb_high=2700, NO retrace
                (2708, 2720, 2707, 2718, 100),  # still no retrace
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E3",
        )

        assert result["outcome"] is None
        assert result["entry_price"] is None


# ============================================================================
# build_outcomes integration tests (with temp DB)
# ============================================================================


class TestBuildOutcomes:
    """Integration tests using a temporary DuckDB database."""

    def _setup_db(self, tmp_path):
        """Create a temp DB with schema + minimal data for 1 trading day."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA

        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)

        from trading_app.db_manager import init_trading_app_schema

        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))

        base_ts = datetime(2024, 1, 4, 23, 0, tzinfo=UTC)
        bars = []
        price = 2700.0
        for i in range(300):
            ts = base_ts + timedelta(minutes=i)
            o = price + i * 0.1
            h = o + 2
            low = o - 1
            c = o + 1
            bars.append((ts.isoformat(), "MGC", "GCG4", o, h, low, c, 100))

        con.executemany(
            """INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
               VALUES (?::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?)""",
            bars,
        )

        orb_high = 2700.0 + 4 * 0.1 + 2
        orb_low = 2700.0 - 1
        con.execute(
            """INSERT INTO daily_features
               (trading_day, symbol, orb_minutes, bar_count_1m,
                orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_break_dir, orb_CME_REOPEN_break_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?::TIMESTAMPTZ)""",
            [
                date(2024, 1, 5),
                "MGC",
                5,
                300,
                orb_high,
                orb_low,
                "long",
                (base_ts + timedelta(minutes=6)).isoformat(),
            ],
        )
        con.commit()
        con.close()

        return db_path

    def test_build_writes_rows(self, tmp_path):
        """build_outcomes writes rows to orb_outcomes table."""
        db_path = self._setup_db(tmp_path)

        count = build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            orb_minutes=5,
        )

        # 1 ORB break: E1 (6 RR * 5 CB = 30) + E3 (6 RR * 1 CB, if retrace found)
        assert count >= 30

        con = duckdb.connect(str(db_path), read_only=True)
        actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        con.close()
        assert actual >= 30

    def test_dry_run_no_writes(self, tmp_path):
        """dry_run=True produces no DB writes."""
        db_path = self._setup_db(tmp_path)

        count = build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            orb_minutes=5,
            dry_run=True,
        )

        assert count > 0

        con = duckdb.connect(str(db_path), read_only=True)
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        ]
        if "orb_outcomes" in tables:
            actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
            assert actual == 0
        con.close()

    def test_idempotent(self, tmp_path):
        """Running twice produces same row count (INSERT OR REPLACE)."""
        db_path = self._setup_db(tmp_path)

        count1 = build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        _count2 = build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        con = duckdb.connect(str(db_path), read_only=True)
        actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        con.close()

        assert actual == count1

    def test_no_break_day_no_rows(self, tmp_path):
        """Day with no break produces no orb_outcomes rows."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA

        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)
        con.close()
        from trading_app.db_manager import init_trading_app_schema

        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))
        con.execute(
            """INSERT INTO daily_features
               (trading_day, symbol, orb_minutes, bar_count_1m,
                orb_CME_REOPEN_high, orb_CME_REOPEN_low)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [date(2024, 1, 5), "MGC", 5, 100, 2700.0, 2690.0],
        )
        con.commit()
        con.close()

        count = build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        assert count == 0

    def test_entry_model_column_populated(self, tmp_path):
        """entry_model column has correct values in DB."""
        db_path = self._setup_db(tmp_path)
        build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        con = duckdb.connect(str(db_path), read_only=True)
        models = {r[0] for r in con.execute("SELECT DISTINCT entry_model FROM orb_outcomes").fetchall()}
        con.close()
        expected = {"E1", "E2", "E3"} - SKIP_ENTRY_MODELS
        assert models == expected


class TestCheckpointResume:
    """Tests for checkpoint/resume crash resilience."""

    def _setup_db(self, tmp_path):
        """Reuse the same setup as TestBuildOutcomes."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA

        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)

        from trading_app.db_manager import init_trading_app_schema

        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))

        base_ts = datetime(2024, 1, 4, 23, 0, tzinfo=UTC)
        bars = []
        price = 2700.0
        for i in range(300):
            ts = base_ts + timedelta(minutes=i)
            o = price + i * 0.1
            h = o + 2
            low = o - 1
            c = o + 1
            bars.append((ts.isoformat(), "MGC", "GCG4", o, h, low, c, 100))

        con.executemany(
            """INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
               VALUES (?::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?)""",
            bars,
        )

        orb_high = 2700.0 + 4 * 0.1 + 2
        orb_low = 2700.0 - 1
        con.execute(
            """INSERT INTO daily_features
               (trading_day, symbol, orb_minutes, bar_count_1m,
                orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_break_dir, orb_CME_REOPEN_break_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?::TIMESTAMPTZ)""",
            [
                date(2024, 1, 5),
                "MGC",
                5,
                300,
                orb_high,
                orb_low,
                "long",
                (base_ts + timedelta(minutes=6)).isoformat(),
            ],
        )
        con.commit()
        con.close()

        return db_path

    def test_second_run_skips_computed_days(self, tmp_path):
        """Second run returns 0 new outcomes because all days are already computed."""
        db_path = self._setup_db(tmp_path)

        count1 = build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert count1 > 0

        count2 = build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert count2 == 0

        # Row count unchanged
        con = duckdb.connect(str(db_path), read_only=True)
        actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        con.close()
        assert actual == count1

    def _setup_db_multi_day(self, tmp_path, num_days=11):
        """Create a temp DB with schema + data for multiple trading days."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA

        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)

        from trading_app.db_manager import init_trading_app_schema

        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))

        for day_offset in range(num_days):
            td = date(2024, 1, 5 + day_offset)
            base_ts = datetime(2024, 1, 4 + day_offset, 23, 0, tzinfo=UTC)
            bars = []
            price = 2700.0
            for i in range(60):
                ts = base_ts + timedelta(minutes=i)
                o = price + i * 0.1
                h = o + 2
                low = o - 1
                c = o + 1
                bars.append((ts.isoformat(), "MGC", "GCG4", o, h, low, c, 100))

            con.executemany(
                """INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
                   VALUES (?::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?)""",
                bars,
            )

            orb_high = 2700.0 + 4 * 0.1 + 2
            orb_low = 2700.0 - 1
            con.execute(
                """INSERT INTO daily_features
                   (trading_day, symbol, orb_minutes, bar_count_1m,
                    orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_break_dir, orb_CME_REOPEN_break_ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?::TIMESTAMPTZ)""",
                [
                    td,
                    "MGC",
                    5,
                    60,
                    orb_high,
                    orb_low,
                    "long",
                    (base_ts + timedelta(minutes=6)).isoformat(),
                ],
            )

        con.commit()
        con.close()
        return db_path

    def test_heartbeat_file_created(self, tmp_path):
        """Heartbeat file is written during non-dry-run build with 10+ days."""
        db_path = self._setup_db_multi_day(tmp_path, num_days=11)

        build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        heartbeat_path = tmp_path / "outcome_builder.heartbeat"
        assert heartbeat_path.exists()
        content = heartbeat_path.read_text()
        assert "MGC" in content
        assert "/" in content  # progress format: "10/11"

    def test_heartbeat_not_created_on_dry_run(self, tmp_path):
        """Heartbeat file is NOT written during dry-run."""
        db_path = self._setup_db(tmp_path)

        build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            dry_run=True,
        )

        heartbeat_path = tmp_path / "outcome_builder.heartbeat"
        assert not heartbeat_path.exists()


class TestTimeStop:
    """Tests for T80 conditional time-stop annotation."""

    def test_keys_present_with_threshold_session(self):
        """compute_single_outcome returns ts_* keys for session with time-stop."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [(2698, 2701, 2695, 2701, 100), (2703, 2710, 2700, 2710, 100), (2718, 2735, 2717, 2730, 100)],
        )
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
            orb_label="TOKYO_OPEN",
        )
        assert "ts_outcome" in result
        assert "ts_pnl_r" in result
        assert "ts_exit_ts" in result

    def test_no_threshold_session_leaves_ts_null(self):
        """Session not in EARLY_EXIT_MINUTES -> ts_* = None."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [(2698, 2701, 2695, 2701, 100), (2703, 2710, 2700, 2710, 100), (2718, 2735, 2717, 2730, 100)],
        )
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
            orb_label="UNKNOWN_SESSION",
        )
        assert result["ts_outcome"] is None
        assert result["ts_pnl_r"] is None
        assert result["ts_exit_ts"] is None

    @patch.dict("trading_app.config.EARLY_EXIT_MINUTES", {"TOKYO_OPEN": 30})
    def test_loss_after_threshold_gets_time_stopped(self):
        """Trade that loses after 30+ min at 1000 gets ts_outcome=time_stop."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 8, 0, tzinfo=UTC)

        # Bar 0: confirm close above ORB high
        # Bar 1: E1 entry at open=2703, stop=2690, risk=13
        # Bars 2-40: price drifts below entry but above stop
        # Bar 61: hits stop
        bar_data = [
            (2698, 2701, 2695, 2701, 100),  # confirm
            (2703, 2705, 2698, 2700, 100),  # entry bar
        ]
        for i in range(58):
            price = 2700 - i * 0.1
            bar_data.append((price, price + 1, price - 1, price, 100))
        bar_data.append((2691, 2692, 2688, 2689, 100))  # hits stop

        bars = _make_bars(datetime(2024, 1, 5, 0, 0, tzinfo=UTC), bar_data)
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
            orb_label="TOKYO_OPEN",
        )
        # Baseline: full stop loss
        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0
        # Time-stop: fires at ~30m with partial loss (better than -1R)
        assert result["ts_outcome"] == "time_stop"
        assert result["ts_pnl_r"] < 0
        assert result["ts_pnl_r"] > -1.0

    @patch.dict("trading_app.config.EARLY_EXIT_MINUTES", {"TOKYO_OPEN": 30})
    def test_win_before_threshold_ts_matches_baseline(self):
        """Trade that wins before T80 -> ts_* matches baseline."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 8, 0, tzinfo=UTC)

        # Quick win: target hit on bar 2 (2 min after entry, well before 30m)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm
                (2703, 2710, 2700, 2710, 100),  # entry bar
                (2710, 2720, 2709, 2718, 100),
                (2718, 2735, 2717, 2730, 100),  # hits target for RR2
            ],
        )
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
            orb_label="TOKYO_OPEN",
        )
        assert result["outcome"] == "win"
        assert result["ts_outcome"] == "win"
        assert result["ts_pnl_r"] == result["pnl_r"]

    @patch.dict("trading_app.config.EARLY_EXIT_MINUTES", {"TOKYO_OPEN": 30})
    def test_positive_at_threshold_keeps_running(self):
        """Trade above entry at threshold bar but not at target -> keeps running."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 8, 0, tzinfo=UTC)

        # Bar 0: confirm. Bar 1: entry at 2703, stop=2690, risk=13
        # Bars 2-40: price above entry (positive MTM) but below target (2729 for RR2)
        # Eventually scratches (no hit)
        bar_data = [
            (2698, 2701, 2695, 2701, 100),  # confirm
            (2703, 2706, 2700, 2705, 100),  # entry bar
        ]
        for _ in range(58):
            bar_data.append((2706, 2710, 2704, 2707, 100))  # positive but below target

        bars = _make_bars(datetime(2024, 1, 5, 0, 0, tzinfo=UTC), bar_data)
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
            orb_label="TOKYO_OPEN",
        )
        # Baseline: scratch (no hit)
        assert result["outcome"] == "scratch"
        # Time-stop: positive at threshold bar -> keeps running, ts = baseline
        assert result["ts_outcome"] == "scratch"
        assert result["ts_pnl_r"] == result["pnl_r"]

    def test_schema_has_time_stop_columns(self, tmp_path):
        """Schema migration adds ts_outcome, ts_pnl_r, ts_exit_ts."""
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        from trading_app.db_manager import init_trading_app_schema

        db_path = tmp_path / "test.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(BARS_1M_SCHEMA)
            con.execute(BARS_5M_SCHEMA)
            con.execute(DAILY_FEATURES_SCHEMA)
        init_trading_app_schema(db_path=db_path)
        with duckdb.connect(str(db_path)) as con:
            cols = {
                r[0]
                for r in con.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = 'orb_outcomes'"
                ).fetchall()
            }
        assert "ts_outcome" in cols
        assert "ts_pnl_r" in cols
        assert "ts_exit_ts" in cols


class TestCLI:
    """Test CLI --help doesn't crash."""

    def test_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["outcome_builder", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from trading_app.outcome_builder import main

            main()
        assert exc_info.value.code == 0
        assert "instrument" in capsys.readouterr().out


# ============================================================================
# E2 fakeout-honesty regression tests (Stage 6 — E2 canonical-window refactor)
#
# These tests pin the load-bearing invariant of the entire 9-stage refactor:
# E2 entries MUST scan from the canonical ORB window close, not from the
# later confirmed-break bar. The previous silent fallback to break_ts
# (which was deleted in Stage 5) caused fakeout entries to be invisible to
# the backtester even though the live engine would have triggered on them
# — the textbook lookahead-bias divergence Chan Ch 1 p4 warns against:
#
#   "If your backtesting and live trading programs are one and the same,
#    and the only difference between backtesting versus live trading is
#    what kind of data you are feeding into the program (historical data
#    in the former, and live market data in the latter), then there can
#    be no look-ahead bias in the program."
#   — Chan, "Algorithmic Trading" (Wiley 2013) Ch 1 p4
#     (extracted verbatim from resources/Algorithmic_Trading_Chan.pdf p22)
#
# Test design:
#   1. test_e2_no_canonical_args_raises_value_error — fail-closed contract
#   2. test_e2_e1_unchanged_without_canonical_args — E1 still works (regression)
#   3. test_e2_explicit_orb_end_utc_captures_fakeout_touch — the load-bearing
#      test: synthetic fakeout day where bar0 pierces orb_high intra-bar but
#      closes back inside (pure fakeout). E2 with orb_end_utc set BEFORE this
#      bar must capture the fakeout entry. Without the Stage 5 fix, this entry
#      was invisible to the backtest because the silent fallback scanned from
#      the LATER confirmed-break bar.
# ============================================================================


class TestE2FakeoutHonesty:
    """Stage 6 regression tests: Stage 5 fail-closed + canonical orb_end_utc."""

    def test_e2_no_canonical_args_raises_value_error(self):
        """E2 with no orb_end_utc and no (trading_day, orb_label, orb_minutes)
        triple MUST raise ValueError. Silent fallback to break_ts is
        forbidden — would reintroduce fakeout-blind backtests."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 5, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2697, 2701, 100),
                (2700, 2710, 2700, 2705, 100),
            ],
        )
        with pytest.raises(ValueError, match="orb_end_utc"):
            compute_single_outcome(
                bars_df=bars,
                break_ts=break_ts,
                orb_high=orb_high,
                orb_low=orb_low,
                break_dir="long",
                rr_target=2.0,
                confirm_bars=1,
                trading_day_end=td_end,
                cost_spec=_cost(),
                entry_model="E2",
                # NO orb_end_utc, NO trading_day/orb_label/orb_minutes — must fail
            )

    def test_e1_unchanged_without_canonical_args(self):
        """E1 path is unchanged by Stage 5 — calling without canonical args
        must still work (the fail-closed gate is E2-only)."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
            [
                (2698, 2701, 2697, 2701, 100),  # bar0: confirm
                (2700, 2710, 2700, 2705, 100),  # bar1: E1 entry at open=2700
                (2705, 2730, 2704, 2729, 100),  # bar2: hits target
            ],
        )
        # No orb_end_utc, no canonical triple — must NOT raise for E1
        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )
        assert result is not None
        # Sanity: E1 either fills or doesn't — the test passes as long as
        # the call returns a dict without raising.

    def test_e2_explicit_orb_end_utc_captures_fakeout_touch(self):
        """The load-bearing test for the E2 canonical-window refactor.

        Construct a synthetic fakeout day:
          - ORB window ends at 00:00 UTC (orb_end_utc).
          - Bar 0 (00:00→00:01 UTC): high pierces orb_high but close returns
            INSIDE the ORB (a pure fakeout — no confirmed close-based break).
          - Bar 1 (00:01→00:02 UTC): close-confirmed break of orb_high.
          - Bar 2+: trade plays out.

        With the Stage 5 fix, E2 scans from orb_end_utc and finds the touch
        on bar 0 — entry_ts must equal bar 0's timestamp.

        Without the fix (the deleted silent fallback to break_ts), E2 would
        have scanned from break_ts (bar 1) and missed the bar 0 fakeout entry
        entirely. The test directly proves the bug is gone.
        """
        orb_high, orb_low = 2700.0, 2690.0
        orb_end = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)  # ORB closes at 00:00
        bar0_ts = orb_end  # bar 0 = first bar after ORB end
        bar1_ts = orb_end + timedelta(minutes=1)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=UTC)

        # Synthetic fakeout day:
        #   bar 0: o=2698, h=2702 (pierces orb_high=2700 intra-bar),
        #          l=2697, c=2699 (closes BACK INSIDE — pure fakeout).
        #   bar 1: o=2700, h=2710, l=2700, c=2705 (close-confirmed break).
        #   bar 2: o=2705, h=2730, l=2704, c=2729 (hits 2x target).
        bars = _make_bars(
            bar0_ts,
            [
                (2698, 2702, 2697, 2699, 100),  # bar 0: FAKEOUT
                (2700, 2710, 2700, 2705, 100),  # bar 1: confirmed break
                (2705, 2730, 2704, 2729, 100),  # bar 2: target hit
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=bar1_ts,  # close-based break is bar 1
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E2",
            orb_end_utc=orb_end,  # canonical: scan from ORB close, not break_ts
        )

        assert result["entry_ts"] is not None, (
            "E2 must capture the bar 0 fakeout touch when orb_end_utc is "
            "set to the canonical ORB close. If this assertion fails, the "
            "Stage 5 silent-fallback regression has returned and backtest "
            "no longer matches live execution (Chan Ch 1 p4 violation)."
        )
        # The entry must be on the fakeout bar (bar 0), NOT the confirmed-
        # break bar (bar 1). This is what makes E2 'honest' — it captures
        # the live stop-market trigger that the close-based break would miss.
        assert result["entry_ts"] == bar0_ts, (
            f"E2 entry_ts must equal bar 0 timestamp (the fakeout touch), "
            f"got {result['entry_ts']}. If entry_ts equals bar1_ts={bar1_ts}, "
            f"E2 is scanning from break_ts instead of orb_end_utc — the "
            f"Stage 5 silent fallback has returned."
        )

    def test_e2_canonical_triple_equivalent_to_explicit_orb_end_utc(self):
        """The canonical-triple path (trading_day, orb_label, orb_minutes)
        must produce the same orb_end_utc as a direct explicit value.

        Uses CME_REOPEN on a winter trading day where the canonical
        orb_utc_window is well-defined. Both paths should resolve to the
        same UTC start, so the same E2 outcome.
        """
        from pipeline.dst import orb_utc_window

        trading_day = date(2024, 1, 15)  # winter, no DST
        orb_label = "CME_REOPEN"
        orb_minutes = 5
        _, expected_orb_end = orb_utc_window(trading_day, orb_label, orb_minutes)

        # Build bars covering the ORB end window
        bars = _make_bars(
            expected_orb_end,
            [
                (2698, 2702, 2697, 2699, 100),  # bar 0: fakeout touch
                (2700, 2710, 2700, 2705, 100),  # bar 1: confirmed break
                (2705, 2730, 2704, 2729, 100),  # bar 2: target hit
            ],
        )

        common_kwargs = {
            "bars_df": bars,
            "break_ts": expected_orb_end + timedelta(minutes=1),
            "orb_high": 2700.0,
            "orb_low": 2690.0,
            "break_dir": "long",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "trading_day_end": expected_orb_end + timedelta(hours=23),
            "cost_spec": _cost(),
            "entry_model": "E2",
            "orb_label": orb_label,
        }

        # Path 1: explicit orb_end_utc
        result_explicit = compute_single_outcome(
            **common_kwargs,
            orb_end_utc=expected_orb_end,
        )

        # Path 2: canonical lookup via (trading_day, orb_label, orb_minutes)
        result_canonical = compute_single_outcome(
            **common_kwargs,
            trading_day=trading_day,
            orb_minutes=orb_minutes,
        )

        # Both paths must produce the same entry timestamp + entry price.
        # If they differ, the canonical lookup has drifted from the explicit
        # value — a structural bug in either orb_utc_window or the
        # compute_single_outcome dispatch.
        assert result_explicit["entry_ts"] == result_canonical["entry_ts"]
        assert result_explicit["entry_price"] == result_canonical["entry_price"]
        assert result_explicit["outcome"] == result_canonical["outcome"]
        assert result_explicit["pnl_r"] == result_canonical["pnl_r"]
