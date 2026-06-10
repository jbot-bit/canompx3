"""Tests for SpreadAccumulator (Defect B — live spread capture).

Verifies the crossed/locked/one-sided guard, per-minute stats, minute-roll
boundary, and final-partial-minute flush. The accumulator is diagnostic-only
and never on the capital path; these tests pin its arithmetic and drop logic.
"""

import math
import threading
from datetime import UTC, datetime

from trading_app.live.spread_accumulator import QuoteMinute, SpreadAccumulator


def _ts(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 6, 10, 14, minute, second, tzinfo=UTC)


class TestDropGuard:
    def test_missing_bid_dropped(self):
        acc = SpreadAccumulator()
        assert acc.add(None, 100.5, _ts(30)) is None
        # Nothing accumulated — flush yields no minute.
        assert acc.flush() is None

    def test_missing_ask_dropped(self):
        acc = SpreadAccumulator()
        assert acc.add(100.0, None, _ts(30)) is None
        assert acc.flush() is None

    def test_crossed_quote_dropped(self):
        """ask < bid (crossed book) must be dropped, not counted."""
        acc = SpreadAccumulator()
        assert acc.add(100.5, 100.4, _ts(30)) is None
        assert acc.flush() is None

    def test_locked_quote_dropped(self):
        """ask == bid (locked book, zero spread) must be dropped.

        A zero-spread row would understate measured friction — the dangerous
        direction for validating the cost model — so we drop rather than coerce.
        """
        acc = SpreadAccumulator()
        assert acc.add(100.5, 100.5, _ts(30)) is None
        assert acc.flush() is None

    def test_dropped_ticks_excluded_from_n_quotes(self):
        acc = SpreadAccumulator()
        acc.add(100.0, 100.25, _ts(30, 1))  # valid
        acc.add(100.5, 100.4, _ts(30, 2))  # crossed — dropped
        acc.add(None, 100.5, _ts(30, 3))  # one-sided — dropped
        acc.add(100.0, 100.75, _ts(30, 4))  # valid
        qm = acc.flush()
        assert qm is not None
        assert qm.n_quotes == 2  # only the two valid quotes


class TestStats:
    def test_avg_min_max_close(self):
        acc = SpreadAccumulator()
        acc.add(100.0, 100.25, _ts(30, 1))  # spread 0.25
        acc.add(100.0, 100.75, _ts(30, 2))  # spread 0.75
        acc.add(100.0, 100.50, _ts(30, 3))  # spread 0.50 (last → close)
        qm = acc.flush()
        assert qm is not None
        assert math.isclose(qm.avg_spread, 0.5, abs_tol=1e-9)
        assert math.isclose(qm.min_spread, 0.25, abs_tol=1e-9)
        assert math.isclose(qm.max_spread, 0.75, abs_tol=1e-9)
        assert math.isclose(qm.close_spread, 0.50, abs_tol=1e-9)
        assert qm.n_quotes == 3

    def test_single_quote_minute(self):
        acc = SpreadAccumulator()
        acc.add(100.0, 100.10, _ts(30))
        qm = acc.flush("MNQ")
        assert qm is not None
        assert math.isclose(qm.avg_spread, 0.10, abs_tol=1e-9)
        assert qm.min_spread == qm.max_spread == qm.close_spread
        assert qm.symbol == "MNQ"


class TestMinuteRoll:
    def test_roll_emits_at_boundary(self):
        acc = SpreadAccumulator()
        assert acc.add(100.0, 100.25, _ts(30, 5)) is None  # opens 14:30
        assert acc.add(100.0, 100.75, _ts(30, 50)) is None  # accumulates 14:30
        rolled = acc.add(100.0, 100.10, _ts(31, 2))  # tick in 14:31 rolls 14:30
        assert rolled is not None
        assert rolled.ts_utc == _ts(30)
        assert rolled.n_quotes == 2
        assert math.isclose(rolled.avg_spread, 0.5, abs_tol=1e-9)
        # 14:31 is still in progress — flush yields it.
        final = acc.flush()
        assert final is not None
        assert final.ts_utc == _ts(31)
        assert final.n_quotes == 1

    def test_out_of_order_tick_dropped(self):
        acc = SpreadAccumulator()
        acc.add(100.0, 100.25, _ts(31, 5))  # opens 14:31
        # A tick from an earlier minute must be dropped, not roll backwards.
        assert acc.add(100.0, 100.50, _ts(30, 5)) is None
        qm = acc.flush()
        assert qm is not None
        assert qm.ts_utc == _ts(31)
        assert qm.n_quotes == 1  # the out-of-order tick excluded


class TestFlush:
    def test_flush_empty_returns_none(self):
        assert SpreadAccumulator().flush() is None

    def test_flush_resets_state(self):
        acc = SpreadAccumulator()
        acc.add(100.0, 100.25, _ts(30))
        assert acc.flush() is not None
        # Second flush with no new ticks is empty.
        assert acc.flush() is None


class TestQuoteMinuteValidity:
    def test_zero_quotes_invalid(self):
        qm = QuoteMinute(ts_utc=_ts(30), avg_spread=0.5, close_spread=0.5, min_spread=0.5, max_spread=0.5, n_quotes=0)
        assert qm.is_valid() is False

    def test_nan_spread_invalid(self):
        qm = QuoteMinute(
            ts_utc=_ts(30), avg_spread=float("nan"), close_spread=0.5, min_spread=0.5, max_spread=0.5, n_quotes=1
        )
        assert qm.is_valid() is False

    def test_valid_minute(self):
        qm = QuoteMinute(ts_utc=_ts(30), avg_spread=0.5, close_spread=0.5, min_spread=0.25, max_spread=0.75, n_quotes=3)
        assert qm.is_valid() is True


class TestThreadSafety:
    def test_concurrent_adds_do_not_lose_quotes(self):
        """Two threads adding to the same minute must not corrupt n_quotes.

        signalrcore fires on a foreign thread; the accumulator must be
        race-safe (same reason BarAggregator carries a lock).
        """
        acc = SpreadAccumulator()
        # Seed the minute so all adds land in the same in-progress minute.
        acc.add(100.0, 100.25, _ts(30, 0))
        n_per_thread = 500

        def worker():
            for i in range(n_per_thread):
                # All within minute 30 (seconds 1..59 cycled) → same minute.
                acc.add(100.0, 100.50, _ts(30, 1 + (i % 58)))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        qm = acc.flush()
        assert qm is not None
        assert qm.n_quotes == 1 + 4 * n_per_thread  # seed + every threaded add counted
