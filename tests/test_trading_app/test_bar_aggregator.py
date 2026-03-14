from datetime import datetime, timezone

from trading_app.live.bar_aggregator import Bar, BarAggregator


def _ts(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 3, 3, 10, minute, second, tzinfo=timezone.utc)


def test_first_tick_opens_bar_returns_none():
    agg = BarAggregator()
    assert agg.on_tick(price=2000.0, volume=1, ts=_ts(0, 5)) is None


def test_tick_crossing_minute_boundary_closes_previous_bar():
    agg = BarAggregator()
    agg.on_tick(price=2000.0, volume=1, ts=_ts(0, 5))
    agg.on_tick(price=2001.0, volume=2, ts=_ts(0, 30))
    agg.on_tick(price=1999.0, volume=1, ts=_ts(0, 59))
    completed = agg.on_tick(price=2002.0, volume=1, ts=_ts(1, 1))
    assert completed is not None
    assert completed.open == 2000.0
    assert completed.high == 2001.0
    assert completed.low == 1999.0
    assert completed.close == 1999.0
    assert completed.volume == 4


def test_bar_ts_utc_is_minute_start():
    agg = BarAggregator()
    agg.on_tick(price=2000.0, volume=1, ts=_ts(5, 3))
    completed = agg.on_tick(price=2001.0, volume=1, ts=_ts(6, 0))
    assert completed.ts_utc.minute == 5
    assert completed.ts_utc.second == 0


def test_bar_as_dict_has_ts_utc_key():
    """ExecutionEngine.on_bar() requires key 'ts_utc'."""
    agg = BarAggregator()
    agg.on_tick(price=2000.0, volume=1, ts=_ts(0, 0))
    bar = agg.on_tick(price=2001.0, volume=1, ts=_ts(1, 0))
    d = bar.as_dict()
    assert "ts_utc" in d
    assert "ts_event" not in d  # wrong key name — engine reads ts_utc


def test_flush_returns_in_progress_bar():
    agg = BarAggregator()
    agg.on_tick(price=2000.0, volume=5, ts=_ts(10, 30))
    bar = agg.flush()
    assert bar is not None
    assert bar.open == 2000.0
    assert bar.volume == 5


def test_flush_on_empty_returns_none():
    agg = BarAggregator()
    assert agg.flush() is None


def test_symbol_propagated_via_setter():
    agg = BarAggregator()
    agg.on_tick(price=2000.0, volume=1, ts=_ts(0, 0))
    bar = agg.on_tick(price=2001.0, volume=1, ts=_ts(1, 0))
    bar.symbol = "MGCM6"
    assert bar.symbol == "MGCM6"


def test_out_of_order_tick_dropped():
    """Ticks older than current bar minute must be silently dropped."""
    agg = BarAggregator()
    agg.on_tick(price=3000.0, volume=1, ts=_ts(1, 0))
    agg.on_tick(price=3001.0, volume=1, ts=_ts(1, 30))
    # Send tick from minute 0 — out of order
    result = agg.on_tick(price=2900.0, volume=1, ts=_ts(0, 15))
    assert result is None  # no bar emitted
    assert agg._current.low == 3000.0  # 2900 NOT incorporated
