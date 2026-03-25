import threading
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


def test_concurrent_ticks_do_not_corrupt_bar():
    """R2-C1: Two threads calling on_tick simultaneously must not corrupt OHLCV.

    Without the threading.Lock in BarAggregator, concurrent read-modify-write
    on high/low/volume can produce wrong values. This test hammers the aggregator
    from multiple threads and verifies the final bar is consistent.
    """
    agg = BarAggregator()
    n_threads = 4
    ticks_per_thread = 500
    barrier = threading.Barrier(n_threads)

    def worker(thread_id: int) -> None:
        barrier.wait()  # synchronize start for maximum contention
        for i in range(ticks_per_thread):
            price = 2000.0 + thread_id * 100 + i * 0.01
            agg.on_tick(price=price, volume=1, ts=_ts(5, 30))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    # Seed the aggregator with one tick to open the bar
    agg.on_tick(price=2000.0, volume=1, ts=_ts(5, 0))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_expected = 1 + n_threads * ticks_per_thread  # 1 seed + all thread ticks
    assert agg._current is not None
    assert agg._current.volume == total_expected
    # High must be the maximum price any thread sent
    max_price = 2000.0 + (n_threads - 1) * 100 + (ticks_per_thread - 1) * 0.01
    assert agg._current.high == max_price
    # Low must be the seed (2000.0 is lowest)
    assert agg._current.low == 2000.0
