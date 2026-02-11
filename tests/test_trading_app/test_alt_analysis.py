"""
Tests for alternative strategy analysis scripts (double break, gap fade).

Tests the shared utilities, core logic functions, and MCP integration.
Does NOT require the live database -- uses synthetic data where possible.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import get_cost_spec
from scripts._alt_strategy_utils import (
    _add_months,
    compute_strategy_metrics,
    compute_walk_forward_windows,
    resolve_bar_outcome,
)
from scripts.analyze_double_break import find_double_break_entry
from scripts.analyze_gap_fade import prepare_gap_data
from trading_app.ai.sql_adapter import QueryTemplate, SQLAdapter


# =============================================================================
# _alt_strategy_utils tests
# =============================================================================

class TestComputeStrategyMetrics:
    def test_empty_returns_none(self):
        assert compute_strategy_metrics(np.array([])) is None

    def test_all_wins(self):
        pnls = np.array([1.0, 2.0, 1.5])
        stats = compute_strategy_metrics(pnls)
        assert stats["n"] == 3
        assert stats["wr"] == 1.0
        assert stats["expr"] == pytest.approx(1.5)
        assert stats["maxdd"] == 0.0  # No drawdown on all-win sequence

    def test_all_losses(self):
        pnls = np.array([-1.0, -1.0, -1.0])
        stats = compute_strategy_metrics(pnls)
        assert stats["n"] == 3
        assert stats["wr"] == 0.0
        assert stats["expr"] < 0
        assert stats["maxdd"] < 0

    def test_mixed(self):
        pnls = np.array([2.0, -1.0, 1.5, -1.0, 0.5])
        stats = compute_strategy_metrics(pnls)
        assert stats["n"] == 5
        assert 0 < stats["wr"] < 1
        assert stats["total"] == pytest.approx(2.0)

    def test_sharpe_positive_for_winning_system(self):
        pnls = np.array([1.0, 0.5, 1.5, -0.5, 2.0, 1.0])
        stats = compute_strategy_metrics(pnls)
        assert stats["sharpe"] > 0


class TestComputeWalkForwardWindows:
    def test_generates_windows(self):
        windows = compute_walk_forward_windows(
            test_start=date(2024, 1, 1),
            test_end=date(2024, 4, 1),
            train_months=12,
            step_months=1,
        )
        assert len(windows) == 3  # Jan, Feb, Mar
        assert windows[0]["test_start"] == date(2024, 1, 1)
        assert windows[0]["train_start"] == date(2023, 1, 1)

    def test_train_before_test(self):
        windows = compute_walk_forward_windows(
            test_start=date(2025, 1, 1),
            test_end=date(2025, 3, 1),
            train_months=6,
        )
        for w in windows:
            assert w["train_end"] < w["test_start"]

    def test_empty_if_start_after_end(self):
        windows = compute_walk_forward_windows(
            test_start=date(2025, 6, 1),
            test_end=date(2025, 1, 1),
            train_months=12,
        )
        assert len(windows) == 0


class TestAddMonths:
    def test_forward(self):
        assert _add_months(date(2024, 1, 15), 3) == date(2024, 4, 15)

    def test_backward(self):
        assert _add_months(date(2024, 3, 15), -3) == date(2023, 12, 15)

    def test_year_wrap_forward(self):
        assert _add_months(date(2024, 11, 1), 3) == date(2025, 2, 1)

    def test_day_clamp(self):
        # Jan 31 + 1 month = Feb 28/29
        result = _add_months(date(2024, 1, 31), 1)
        assert result.month == 2
        assert result.day == 29  # 2024 is leap year


class TestResolveBarOutcome:
    def _make_bars(self, data):
        """Create a bars DataFrame from list of (open, high, low, close) tuples."""
        return pd.DataFrame(data, columns=["open", "high", "low", "close"])

    def test_long_target_hit(self):
        bars = self._make_bars([
            (100, 105, 99, 104),  # Target hit (high >= 105)
        ])
        result = resolve_bar_outcome(bars, 100, 95, 105, "long", 0)
        assert result["outcome"] == "win"
        assert result["pnl_points"] == 5.0

    def test_long_stop_hit(self):
        bars = self._make_bars([
            (100, 101, 94, 96),  # Stop hit (low <= 95)
        ])
        result = resolve_bar_outcome(bars, 100, 95, 105, "long", 0)
        assert result["outcome"] == "loss"
        assert result["pnl_points"] == -5.0

    def test_short_target_hit(self):
        bars = self._make_bars([
            (100, 101, 94, 95),  # Target hit (low <= 95)
        ])
        result = resolve_bar_outcome(bars, 100, 105, 95, "short", 0)
        assert result["outcome"] == "win"
        assert result["pnl_points"] == 5.0

    def test_short_stop_hit(self):
        bars = self._make_bars([
            (100, 106, 99, 105),  # Stop hit (high >= 105)
        ])
        result = resolve_bar_outcome(bars, 100, 105, 95, "short", 0)
        assert result["outcome"] == "loss"
        assert result["pnl_points"] == -5.0

    def test_gate_c_ambiguous_bar_is_loss(self):
        """Gate C: If stop AND target hit on same bar, resolve as LOSS."""
        bars = self._make_bars([
            (100, 110, 90, 100),  # Both stop (90<=95) AND target (110>=105) hit
        ])
        result = resolve_bar_outcome(bars, 100, 95, 105, "long", 0)
        assert result["outcome"] == "loss"
        assert result["pnl_points"] == -5.0

    def test_no_resolution_returns_none(self):
        bars = self._make_bars([
            (100, 102, 98, 101),  # Neither stop nor target hit
            (101, 103, 97, 100),
        ])
        result = resolve_bar_outcome(bars, 100, 95, 105, "long", 0)
        assert result is None

    def test_start_idx_skips_bars(self):
        bars = self._make_bars([
            (100, 110, 90, 100),  # Would trigger if not skipped
            (100, 101, 99, 100),  # Neither hit
        ])
        result = resolve_bar_outcome(bars, 100, 95, 105, "long", 1)
        assert result is None


# =============================================================================
# Double break analysis tests
# =============================================================================

class TestFindDoubleBreakEntry:
    def _make_bars_with_ts(self, data):
        """Create bars DataFrame with timestamps."""
        return pd.DataFrame(data, columns=["ts_utc", "open", "high", "low", "close", "volume"])

    def test_long_break_reversal_to_short(self):
        """Long break fails -> reversal entry is SHORT at orb_low."""
        break_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        bars = self._make_bars_with_ts([
            # Before break_ts - should be skipped (Gate A)
            (pd.Timestamp("2024-12-31 23:55:00", tz="UTC"), 100, 102, 99, 101, 10),
            # After break_ts - fakeout tracking starts
            (pd.Timestamp("2025-01-01 00:05:00", tz="UTC"), 101, 108, 100, 106, 10),  # Fakeout high = 108
            (pd.Timestamp("2025-01-01 00:10:00", tz="UTC"), 106, 107, 104, 105, 10),
            # Reversal: price drops below orb_low (95)
            (pd.Timestamp("2025-01-01 00:15:00", tz="UTC"), 105, 105, 93, 94, 10),
        ])

        result = find_double_break_entry(
            bars, break_ts, "long", orb_high=105, orb_low=95
        )

        assert result is not None
        assert result["direction"] == "short"
        assert result["entry_price"] == 95  # orb_low
        assert result["fakeout_extreme"] == 108  # Highest high during long fakeout
        assert result["stop_price"] == 108

    def test_short_break_reversal_to_long(self):
        """Short break fails -> reversal entry is LONG at orb_high."""
        break_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        bars = self._make_bars_with_ts([
            (pd.Timestamp("2025-01-01 00:05:00", tz="UTC"), 100, 101, 88, 90, 10),  # Fakeout low = 88
            (pd.Timestamp("2025-01-01 00:10:00", tz="UTC"), 90, 92, 89, 91, 10),
            # Reversal: price rises above orb_high (105)
            (pd.Timestamp("2025-01-01 00:15:00", tz="UTC"), 91, 107, 90, 106, 10),
        ])

        result = find_double_break_entry(
            bars, break_ts, "short", orb_high=105, orb_low=95
        )

        assert result is not None
        assert result["direction"] == "long"
        assert result["entry_price"] == 105  # orb_high
        assert result["fakeout_extreme"] == 88  # Lowest low during short fakeout
        assert result["stop_price"] == 88

    def test_gate_a_no_same_bar_fill(self):
        """Gate A: Bars at or before break_ts should be skipped."""
        break_ts = pd.Timestamp("2025-01-01 00:10:00", tz="UTC")
        bars = self._make_bars_with_ts([
            # This bar is AT break_ts -> should be skipped
            (pd.Timestamp("2025-01-01 00:10:00", tz="UTC"), 100, 120, 80, 90, 10),
        ])

        result = find_double_break_entry(
            bars, break_ts, "long", orb_high=105, orb_low=95
        )
        assert result is None

    def test_gate_b_risk_floor(self):
        """Gate B: If fakeout extreme too close to entry, return None."""
        spec = get_cost_spec("MGC")
        break_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        # Fakeout extreme very close to ORB level -> tiny risk
        bars = self._make_bars_with_ts([
            (pd.Timestamp("2025-01-01 00:05:00", tz="UTC"), 100, 105.5, 99, 101, 10),  # Fakeout high = 105.5
            (pd.Timestamp("2025-01-01 00:10:00", tz="UTC"), 101, 101, 94, 94, 10),  # Crosses below 95
        ])

        result = find_double_break_entry(
            bars, break_ts, "long", orb_high=105, orb_low=95
        )

        # Risk = |95 - 105.5| = 10.5 pts, which is above floor
        # This should pass Gate B
        assert result is not None

        # Now test with tiny fakeout: all bars have highs very close to orb_low
        bars2 = self._make_bars_with_ts([
            (pd.Timestamp("2025-01-01 00:05:00", tz="UTC"), 95.2, 95.3, 95.0, 95.1, 10),  # Fakeout high = 95.3
            (pd.Timestamp("2025-01-01 00:10:00", tz="UTC"), 95.1, 95.2, 94.0, 94.0, 10),  # Crosses below 95, high=95.2
        ])

        result2 = find_double_break_entry(
            bars2, break_ts, "long", orb_high=105, orb_low=95
        )
        # Fakeout extreme = max(95.3, 95.2) = 95.3
        # Risk = |95 - 95.3| = 0.3 pts < 1.0 min floor -> should be None
        assert result2 is None

    def test_no_reversal_returns_none(self):
        """No reversal found -> return None."""
        break_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        bars = self._make_bars_with_ts([
            (pd.Timestamp("2025-01-01 00:05:00", tz="UTC"), 100, 110, 99, 108, 10),
            (pd.Timestamp("2025-01-01 00:10:00", tz="UTC"), 108, 112, 107, 111, 10),
        ])

        result = find_double_break_entry(
            bars, break_ts, "long", orb_high=105, orb_low=95
        )
        assert result is None  # Price never drops below orb_low


# =============================================================================
# Gap fade analysis tests
# =============================================================================

class TestPrepareGapData:
    def test_atr_computed(self):
        """ATR_20 should be rolling 20-day mean of true range, shifted by 1 (no lookahead)."""
        n = 30
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]
        df = pd.DataFrame({
            "trading_day": dates,
            "daily_open": [100.0] * n,
            "daily_high": [105.0] * n,  # TR = 5.0 every day
            "daily_low": [100.0] * n,
            "daily_close": [103.0] * n,
            "gap_open_points": [0.5] * n,
        })

        result = prepare_gap_data(df)

        # Rolling(20) fills at index 19, but shift(1) pushes it to index 20
        assert pd.isna(result.iloc[19]["atr_20"])
        # Row 20 onwards should be 5.0 (uses prior 20 days' data)
        assert result.iloc[20]["atr_20"] == pytest.approx(5.0)

    def test_prev_close_shift(self):
        df = pd.DataFrame({
            "trading_day": [date(2025, 1, 1), date(2025, 1, 2)],
            "daily_open": [100.0, 101.0],
            "daily_high": [105.0, 106.0],
            "daily_low": [99.0, 100.0],
            "daily_close": [103.0, 104.0],
            "gap_open_points": [0.0, 1.0],
        })

        result = prepare_gap_data(df)
        assert pd.isna(result.iloc[0]["prev_close"])
        assert result.iloc[1]["prev_close"] == 103.0

    def test_dow_assignment(self):
        # 2025-01-06 is a Monday
        df = pd.DataFrame({
            "trading_day": [date(2025, 1, 6)],
            "daily_open": [100.0],
            "daily_high": [105.0],
            "daily_low": [99.0],
            "daily_close": [103.0],
            "gap_open_points": [1.0],
        })

        result = prepare_gap_data(df)
        assert result.iloc[0]["dow"] == 0  # Monday


# =============================================================================
# MCP template tests
# =============================================================================

class TestMCPTemplates:
    def test_double_break_stats_in_enum(self):
        assert QueryTemplate.DOUBLE_BREAK_STATS.value == "double_break_stats"

    def test_gap_analysis_in_enum(self):
        assert QueryTemplate.GAP_ANALYSIS.value == "gap_analysis"

    def test_templates_listed(self):
        templates = SQLAdapter.available_templates()
        names = [t["template"] for t in templates]
        assert "double_break_stats" in names
        assert "gap_analysis" in names

    def test_template_descriptions_present(self):
        templates = SQLAdapter.available_templates()
        for t in templates:
            if t["template"] in ("double_break_stats", "gap_analysis"):
                assert len(t["description"]) > 0
