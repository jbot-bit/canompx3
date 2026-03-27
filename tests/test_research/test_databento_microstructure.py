"""Tests for databento microstructure repricing logic.

All tests use synthetic DataFrames mimicking tbbo schema — no network calls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.databento_microstructure import (
    RepricedEntry,
    _cache_path,
    analyze_slippage,
    reprice_e2_entry,
)

TICK_SIZE = 0.10  # MGC tick size


def _make_tbbo_df(
    records: list[dict],
    *,
    index_by_ts: bool = True,
) -> pd.DataFrame:
    """Build a minimal tbbo-shaped DataFrame from record dicts.

    Each record needs: ts_event, price, bid_px_00, ask_px_00.
    Optional: side, size, bid_sz_00, ask_sz_00.
    """
    defaults = {"side": "N", "size": 1, "bid_sz_00": 10, "ask_sz_00": 10}
    rows = [{**defaults, **r} for r in records]
    df = pd.DataFrame(rows)
    if index_by_ts and "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        df = df.set_index("ts_event")
    return df


# ── reprice_e2_entry ────────────────────────────────────────────────────


class TestRepriceE2Entry:
    """Test the core repricing logic."""

    def _base_kwargs(self, **overrides):
        defaults = {
            "orb_high": 2350.0,
            "orb_low": 2340.0,
            "model_entry_price": 2350.1,  # ORB high + 1 tick
            "model_entry_ts_utc": "2024-06-05T22:10:00+00:00",
            "trading_day": "2024-06-05",
            "symbol_pulled": "MGC.FUT",
            "tick_size": TICK_SIZE,
            "modeled_slippage_ticks": 1,
        }
        defaults.update(overrides)
        return defaults

    def test_clean_long_fill_1_tick(self):
        """Long break: trade at ORB high, ask is 1 tick above → 1 tick slippage."""
        tbbo = _make_tbbo_df([
            {"ts_event": "2024-06-05T22:08:00", "price": 2349.5, "bid_px_00": 2349.4, "ask_px_00": 2349.6},
            {"ts_event": "2024-06-05T22:09:00", "price": 2349.8, "bid_px_00": 2349.7, "ask_px_00": 2349.9},
            {"ts_event": "2024-06-05T22:10:00", "price": 2350.0, "bid_px_00": 2349.9, "ask_px_00": 2350.1},
            {"ts_event": "2024-06-05T22:10:01", "price": 2350.2, "bid_px_00": 2350.1, "ask_px_00": 2350.3},
        ])
        result = reprice_e2_entry(tbbo_df=tbbo, break_dir="long", **self._base_kwargs())

        assert result.error is None
        assert result.trigger_trade_price == 2350.0
        assert result.estimated_fill_price == 2350.1  # ask at trigger
        assert result.actual_slippage_ticks == 1.0
        assert result.actual_slippage_points == pytest.approx(0.1)

    def test_gapped_long_fill(self):
        """Long break: trade gaps through ORB high, ask is 3 ticks above."""
        tbbo = _make_tbbo_df([
            {"ts_event": "2024-06-05T22:09:00", "price": 2349.5, "bid_px_00": 2349.4, "ask_px_00": 2349.6},
            {"ts_event": "2024-06-05T22:10:00", "price": 2350.3, "bid_px_00": 2350.2, "ask_px_00": 2350.3},
        ])
        result = reprice_e2_entry(tbbo_df=tbbo, break_dir="long", **self._base_kwargs())

        assert result.error is None
        assert result.trigger_trade_price == 2350.3
        assert result.estimated_fill_price == 2350.3  # ask
        assert result.actual_slippage_ticks == 3.0  # (2350.3 - 2350.0) / 0.1

    def test_wide_spread_long(self):
        """Long break with wide spread at trigger — worse fill."""
        tbbo = _make_tbbo_df([
            {"ts_event": "2024-06-05T22:10:00", "price": 2350.0, "bid_px_00": 2349.7, "ask_px_00": 2350.5},
        ])
        result = reprice_e2_entry(tbbo_df=tbbo, break_dir="long", **self._base_kwargs())

        assert result.error is None
        assert result.estimated_fill_price == 2350.5
        assert result.actual_slippage_ticks == 5.0  # (2350.5 - 2350.0) / 0.1
        assert result.bbo_at_trigger_spread == 8.0  # (2350.5 - 2349.7) / 0.1

    def test_clean_short_fill_1_tick(self):
        """Short break: trade at ORB low, bid is 1 tick below → 1 tick slippage."""
        tbbo = _make_tbbo_df([
            {"ts_event": "2024-06-05T22:09:00", "price": 2340.5, "bid_px_00": 2340.4, "ask_px_00": 2340.6},
            {"ts_event": "2024-06-05T22:10:00", "price": 2340.0, "bid_px_00": 2339.9, "ask_px_00": 2340.1},
        ])
        result = reprice_e2_entry(
            tbbo_df=tbbo,
            break_dir="short",
            **self._base_kwargs(model_entry_price=2339.9),
        )

        assert result.error is None
        assert result.trigger_trade_price == 2340.0
        assert result.estimated_fill_price == 2339.9  # bid at trigger
        assert result.actual_slippage_ticks == 1.0  # (2340.0 - 2339.9) / 0.1

    def test_gapped_short_fill(self):
        """Short break: trade gaps below ORB low, bid is 4 ticks below."""
        tbbo = _make_tbbo_df([
            {"ts_event": "2024-06-05T22:09:00", "price": 2340.5, "bid_px_00": 2340.4, "ask_px_00": 2340.6},
            {"ts_event": "2024-06-05T22:10:00", "price": 2339.5, "bid_px_00": 2339.6, "ask_px_00": 2339.8},
        ])
        result = reprice_e2_entry(
            tbbo_df=tbbo,
            break_dir="short",
            **self._base_kwargs(model_entry_price=2339.9),
        )

        assert result.error is None
        assert result.trigger_trade_price == 2339.5
        assert result.estimated_fill_price == 2339.6  # bid
        assert result.actual_slippage_ticks == 4.0  # (2340.0 - 2339.6) / 0.1

    def test_no_trigger_trade(self):
        """No trade crosses the ORB level → no repricing possible."""
        tbbo = _make_tbbo_df([
            {"ts_event": "2024-06-05T22:08:00", "price": 2349.0, "bid_px_00": 2348.9, "ask_px_00": 2349.1},
            {"ts_event": "2024-06-05T22:09:00", "price": 2349.5, "bid_px_00": 2349.4, "ask_px_00": 2349.6},
        ])
        result = reprice_e2_entry(tbbo_df=tbbo, break_dir="long", **self._base_kwargs())

        assert result.error == "no_trigger_trade_found"
        assert result.estimated_fill_price is None
        assert result.actual_slippage_ticks is None

    def test_empty_tbbo(self):
        """Empty DataFrame → error."""
        tbbo = pd.DataFrame()
        result = reprice_e2_entry(tbbo_df=tbbo, break_dir="long", **self._base_kwargs())

        assert result.error == "no_tbbo_records"
        assert result.tbbo_records_in_window == 0

    def test_negative_slippage_long(self):
        """Edge case: ask is AT orb_high (zero slippage) — should work."""
        tbbo = _make_tbbo_df([
            {"ts_event": "2024-06-05T22:10:00", "price": 2350.0, "bid_px_00": 2349.9, "ask_px_00": 2350.0},
        ])
        result = reprice_e2_entry(tbbo_df=tbbo, break_dir="long", **self._base_kwargs())

        assert result.error is None
        assert result.estimated_fill_price == 2350.0
        assert result.actual_slippage_ticks == 0.0

    def test_tbbo_records_counted(self):
        """tbbo_records_in_window should reflect full window, not just trigger."""
        tbbo = _make_tbbo_df([
            {"ts_event": "2024-06-05T22:08:00", "price": 2349.0, "bid_px_00": 2348.9, "ask_px_00": 2349.1},
            {"ts_event": "2024-06-05T22:09:00", "price": 2349.5, "bid_px_00": 2349.4, "ask_px_00": 2349.6},
            {"ts_event": "2024-06-05T22:10:00", "price": 2350.0, "bid_px_00": 2349.9, "ask_px_00": 2350.1},
            {"ts_event": "2024-06-05T22:10:01", "price": 2350.2, "bid_px_00": 2350.1, "ask_px_00": 2350.3},
            {"ts_event": "2024-06-05T22:11:00", "price": 2350.5, "bid_px_00": 2350.4, "ask_px_00": 2350.6},
        ])
        result = reprice_e2_entry(tbbo_df=tbbo, break_dir="long", **self._base_kwargs())
        assert result.tbbo_records_in_window == 5

    def test_pre_orb_trades_filtered_out(self):
        """Trades before orb_end_utc must not trigger — ORB not formed yet."""
        tbbo = _make_tbbo_df([
            # Pre-ORB: price crosses level but should be ignored
            {"ts_event": "2024-06-05T22:03:00", "price": 2351.0, "bid_px_00": 2350.9, "ask_px_00": 2351.1},
            # Post-ORB: this is the real trigger
            {"ts_event": "2024-06-05T22:06:00", "price": 2349.5, "bid_px_00": 2349.4, "ask_px_00": 2349.6},
            {"ts_event": "2024-06-05T22:10:00", "price": 2350.0, "bid_px_00": 2349.9, "ask_px_00": 2350.1},
        ])
        # ORB ends at 22:05 (5 min aperture from 22:00)
        result = reprice_e2_entry(
            tbbo_df=tbbo,
            break_dir="long",
            orb_end_utc="2024-06-05T22:05:00+00:00",
            **self._base_kwargs(),
        )
        assert result.error is None
        # Should match the 22:10 trade, not the 22:03 pre-ORB trade
        assert result.trigger_trade_price == 2350.0
        assert result.estimated_fill_price == 2350.1

    def test_all_trades_pre_orb_returns_error(self):
        """If all trades are before orb_end, return error."""
        tbbo = _make_tbbo_df([
            {"ts_event": "2024-06-05T22:03:00", "price": 2351.0, "bid_px_00": 2350.9, "ask_px_00": 2351.1},
            {"ts_event": "2024-06-05T22:04:00", "price": 2350.5, "bid_px_00": 2350.4, "ask_px_00": 2350.6},
        ])
        result = reprice_e2_entry(
            tbbo_df=tbbo,
            break_dir="long",
            orb_end_utc="2024-06-05T22:05:00+00:00",
            **self._base_kwargs(),
        )
        assert result.error == "no_post_orb_records"


# ── analyze_slippage ────────────────────────────────────────────────────


class TestAnalyzeSlippage:
    """Test the slippage analysis aggregation."""

    def _make_repriced_df(self, records: list[dict]) -> pd.DataFrame:
        defaults = {
            "trading_day": "2024-06-05",
            "symbol_pulled": "MGC.FUT",
            "orb_level": 2350.0,
            "modeled_entry_price": 2350.1,
            "modeled_slippage_ticks": 1,
            "trigger_trade_price": 2350.0,
            "trigger_trade_ts": "2024-06-05T22:10:00",
            "bbo_at_trigger_bid": 2349.9,
            "bbo_at_trigger_ask": 2350.1,
            "bbo_at_trigger_spread": 2.0,
            "estimated_fill_price": 2350.1,
            "actual_slippage_points": 0.1,
            "actual_slippage_ticks": 1.0,
            "tick_size": TICK_SIZE,
            "tbbo_records_in_window": 100,
            "error": None,
        }
        return pd.DataFrame([{**defaults, **r} for r in records])

    def test_basic_analysis(self):
        df = self._make_repriced_df([
            {"break_dir": "long", "actual_slippage_ticks": 1.0},
            {"break_dir": "long", "actual_slippage_ticks": 2.0},
            {"break_dir": "short", "actual_slippage_ticks": 1.5},
        ])
        analysis = analyze_slippage(df)
        mgc = analysis["MGC.FUT"]
        assert mgc["n"] == 3
        assert mgc["slippage_ticks"]["mean"] == 1.5
        assert mgc["by_direction"]["long"]["n"] == 2
        assert mgc["by_direction"]["short"]["n"] == 1

    def test_skips_null_slippage(self):
        df = self._make_repriced_df([
            {"break_dir": "long", "actual_slippage_ticks": 1.0},
            {"break_dir": "long", "actual_slippage_ticks": None, "error": "no_trigger"},
        ])
        analysis = analyze_slippage(df)
        assert analysis["MGC.FUT"]["n"] == 1

    def test_multi_symbol(self):
        df = self._make_repriced_df([
            {"symbol_pulled": "MGC.FUT", "break_dir": "long", "actual_slippage_ticks": 1.0},
            {"symbol_pulled": "GC.FUT", "break_dir": "long", "actual_slippage_ticks": 0.5},
        ])
        analysis = analyze_slippage(df)
        assert "MGC.FUT" in analysis
        assert "GC.FUT" in analysis
        # GC should show lower slippage (deeper book)
        assert analysis["GC.FUT"]["slippage_ticks"]["mean"] < analysis["MGC.FUT"]["slippage_ticks"]["mean"]


# ── _cache_path ─────────────────────────────────────────────────────────


def test_cache_path_deterministic():
    p1 = _cache_path("2024-06-05", "MGC.FUT")
    p2 = _cache_path("2024-06-05", "MGC.FUT")
    assert p1 == p2
    assert "MGC_FUT" in p1.name
    assert p1.suffix == ".zst"


def test_cache_path_differs_by_symbol():
    p1 = _cache_path("2024-06-05", "MGC.FUT")
    p2 = _cache_path("2024-06-05", "GC.FUT")
    assert p1 != p2
