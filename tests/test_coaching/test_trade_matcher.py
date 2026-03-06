"""Tests for trade matcher — fills -> round-trip trades."""

import json

import pytest

from scripts.tools.trade_matcher import (
    classify_trade_type,
    load_fills,
    match_fills_to_trades,
)


def _make_fill(fill_id, account_id, instrument, timestamp, side, size, price, pnl=0.0, fees=0.0):
    return {
        "fill_id": fill_id,
        "broker": "topstepx",
        "account_id": account_id,
        "account_name": "test-account",
        "instrument": instrument,
        "timestamp": timestamp,
        "side": side,
        "size": size,
        "price": price,
        "pnl": pnl,
        "fees": fees,
    }


class TestClassifyTradeType:
    def test_scalp_under_5min(self):
        assert classify_trade_type(120) == "scalp"

    def test_swing_5_to_60min(self):
        assert classify_trade_type(600) == "swing"

    def test_position_over_60min(self):
        assert classify_trade_type(7200) == "position"


class TestSimpleRoundTrip:
    """BUY 4, then SELL 4 -> one LONG round-trip trade."""

    def test_simple_long(self):
        fills = [
            _make_fill("f1", 123, "MNQ", "2026-03-06T13:00:00Z", "BUY", 4, 24800.0, 0, 1.48),
            _make_fill("f2", 123, "MNQ", "2026-03-06T13:02:00Z", "SELL", 4, 24810.0, 40.0, 1.48),
        ]
        trades = match_fills_to_trades(fills)
        assert len(trades) == 1
        t = trades[0]
        assert t["direction"] == "LONG"
        assert t["instrument"] == "MNQ"
        assert t["entry_price_avg"] == 24800.0
        assert t["exit_price_avg"] == 24810.0
        assert t["size"] == 4
        assert t["hold_seconds"] == 120
        assert t["trade_type"] == "scalp"
        assert t["num_fills"] == 2

    def test_simple_short(self):
        fills = [
            _make_fill("f1", 123, "MNQ", "2026-03-06T13:00:00Z", "SELL", 2, 24800.0, 0, 1.0),
            _make_fill("f2", 123, "MNQ", "2026-03-06T13:01:00Z", "BUY", 2, 24790.0, 20.0, 1.0),
        ]
        trades = match_fills_to_trades(fills)
        assert len(trades) == 1
        assert trades[0]["direction"] == "SHORT"


class TestMultiFillRoundTrip:
    """Scale-in: BUY 2 + BUY 2, then SELL 4 -> one trade with VWAP entry."""

    def test_scale_in_vwap(self):
        fills = [
            _make_fill("f1", 123, "MNQ", "2026-03-06T13:00:00Z", "BUY", 2, 24800.0),
            _make_fill("f2", 123, "MNQ", "2026-03-06T13:00:30Z", "BUY", 2, 24805.0),
            _make_fill("f3", 123, "MNQ", "2026-03-06T13:02:00Z", "SELL", 4, 24810.0, 30.0, 2.96),
        ]
        trades = match_fills_to_trades(fills)
        assert len(trades) == 1
        t = trades[0]
        assert t["entry_price_avg"] == pytest.approx(24802.5)  # VWAP: (2*24800 + 2*24805) / 4
        assert t["size"] == 4
        assert t["num_fills"] == 3


class TestPositionFlip:
    """BUY 2, SELL 4 -> close LONG (2), open SHORT (2). Two trades."""

    def test_flip_generates_two_trades(self):
        fills = [
            _make_fill("f1", 123, "MNQ", "2026-03-06T13:00:00Z", "BUY", 2, 24800.0),
            _make_fill("f2", 123, "MNQ", "2026-03-06T13:01:00Z", "SELL", 4, 24810.0),
            _make_fill("f3", 123, "MNQ", "2026-03-06T13:03:00Z", "BUY", 2, 24805.0),
        ]
        trades = match_fills_to_trades(fills)
        assert len(trades) == 2
        assert trades[0]["direction"] == "LONG"
        assert trades[1]["direction"] == "SHORT"


class TestMultiAccountIsolation:
    """Fills from different accounts don't cross-match."""

    def test_accounts_isolated(self):
        fills = [
            _make_fill("f1", 111, "MNQ", "2026-03-06T13:00:00Z", "BUY", 2, 24800.0),
            _make_fill("f2", 222, "MNQ", "2026-03-06T13:00:00Z", "SELL", 2, 24810.0),
        ]
        trades = match_fills_to_trades(fills)
        # No round-trips: each account has an open position, neither is closed
        assert len(trades) == 0


class TestLoadFills:
    def test_load_fills_from_jsonl(self, tmp_path):
        path = tmp_path / "fills.jsonl"
        path.write_text(
            json.dumps({"fill_id": "a", "timestamp": "2026-03-06T13:00:00Z"})
            + "\n"
            + json.dumps({"fill_id": "b", "timestamp": "2026-03-06T13:01:00Z"})
            + "\n"
        )
        fills = load_fills(path=path)
        assert len(fills) == 2
