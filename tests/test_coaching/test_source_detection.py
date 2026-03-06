"""Tests for source detection — matching broker trades to system signals."""

import json
from pathlib import Path

import pytest

from scripts.tools.trade_matcher import detect_source


def _make_trade(entry_time, instrument):
    return {"entry_time": entry_time, "instrument": instrument, "source": "manual", "strategy_id": None}


def _make_signal(ts, instrument, strategy_id, signal_type="SIGNAL_ENTRY"):
    return {"ts": ts, "instrument": instrument, "strategy_id": strategy_id, "type": signal_type}


class TestSourceDetection:
    def test_matches_within_60s(self):
        trade = _make_trade("2026-03-06T13:00:00+00:00", "MNQ")
        signals = [_make_signal("2026-03-06T13:00:30+00:00", "MNQ", "strat-001")]
        detect_source(trade, signals)
        assert trade["source"] == "system"
        assert trade["strategy_id"] == "strat-001"

    def test_no_match_beyond_60s(self):
        trade = _make_trade("2026-03-06T13:00:00+00:00", "MNQ")
        signals = [_make_signal("2026-03-06T13:05:00+00:00", "MNQ", "strat-001")]
        detect_source(trade, signals)
        assert trade["source"] == "manual"
        assert trade["strategy_id"] is None

    def test_no_match_wrong_instrument(self):
        trade = _make_trade("2026-03-06T13:00:00+00:00", "MNQ")
        signals = [_make_signal("2026-03-06T13:00:10+00:00", "MGC", "strat-001")]
        detect_source(trade, signals)
        assert trade["source"] == "manual"

    def test_empty_signals(self):
        trade = _make_trade("2026-03-06T13:00:00+00:00", "MNQ")
        detect_source(trade, [])
        assert trade["source"] == "manual"
