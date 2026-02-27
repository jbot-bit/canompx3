"""Tests for research.research_portfolio_assembly — pure computation functions."""

import datetime as dt

import pytest

from research.research_portfolio_assembly import (
    build_daily_equity,
    compute_drawdown,
    compute_honest_sharpe,
    count_trading_days,
)


def _trade(day, pnl_r, outcome="win", instrument="MGC", session="TOKYO_OPEN", sid="S1"):
    return {
        "trading_day": day,
        "pnl_r": pnl_r,
        "outcome": outcome,
        "instrument": instrument,
        "session": session,
        "strategy_id": sid,
    }


# ── build_daily_equity ────────────────────────────────────────────────────


class TestBuildDailyEquity:
    def test_single_slot(self):
        trades = {
            "S1": [
                _trade(dt.date(2024, 1, 2), 1.5),
                _trade(dt.date(2024, 1, 3), -1.0, "loss"),
            ]
        }
        daily, all_t, counts = build_daily_equity(trades)
        assert len(all_t) == 2
        assert dict(daily) == {dt.date(2024, 1, 2): 1.5, dt.date(2024, 1, 3): -1.0}
        assert counts == {dt.date(2024, 1, 2): 1, dt.date(2024, 1, 3): 1}

    def test_multi_slot_same_day(self):
        """Two slots firing on the same day should sum R."""
        trades = {
            "S1": [_trade(dt.date(2024, 1, 2), 1.5)],
            "S2": [_trade(dt.date(2024, 1, 2), -1.0, "loss", sid="S2")],
        }
        daily, all_t, counts = build_daily_equity(trades)
        assert len(all_t) == 2
        assert dict(daily) == {dt.date(2024, 1, 2): pytest.approx(0.5)}
        assert counts[dt.date(2024, 1, 2)] == 2

    def test_empty(self):
        daily, all_t, counts = build_daily_equity({})
        assert daily == []
        assert all_t == []
        assert counts == {}


# ── count_trading_days ────────────────────────────────────────────────────


class TestCountTradingDays:
    def test_one_week(self):
        # Mon Jan 1 2024 to Fri Jan 5 2024 = 5 business days
        assert count_trading_days(dt.date(2024, 1, 1), dt.date(2024, 1, 5)) == 5

    def test_includes_endpoints(self):
        # Single day (Monday)
        assert count_trading_days(dt.date(2024, 1, 1), dt.date(2024, 1, 1)) == 1

    def test_weekend_excluded(self):
        # Sat Jan 6 to Sun Jan 7 = 0 business days
        assert count_trading_days(dt.date(2024, 1, 6), dt.date(2024, 1, 7)) == 0


# ── compute_honest_sharpe ─────────────────────────────────────────────────


class TestComputeHonestSharpe:
    def test_constant_returns_gives_none(self):
        """Constant 1R every day has zero std — Sharpe is undefined."""
        start = dt.date(2024, 1, 1)  # Monday
        end = dt.date(2024, 1, 5)    # Friday
        daily = [(dt.date(2024, 1, d), 1.0) for d in range(1, 6)]
        _, sharpe_ann, n = compute_honest_sharpe(daily, start, end)
        assert n == 5
        assert sharpe_ann is None

    def test_varying_returns(self):
        start = dt.date(2024, 1, 1)
        end = dt.date(2024, 1, 5)
        daily = [
            (dt.date(2024, 1, 1), 2.0),
            (dt.date(2024, 1, 2), -1.0),
            (dt.date(2024, 1, 3), 1.5),
            (dt.date(2024, 1, 4), 0.5),
            (dt.date(2024, 1, 5), 1.0),
        ]
        sharpe_d, sharpe_ann, n = compute_honest_sharpe(daily, start, end)
        assert n == 5
        assert sharpe_d > 0
        assert sharpe_ann > 0
        assert sharpe_ann == pytest.approx(sharpe_d * (252 ** 0.5))

    def test_zero_days_dilute_sharpe(self):
        """Returns on 2 of 5 days — zeros pad the series, lowering Sharpe."""
        start = dt.date(2024, 1, 1)  # Mon
        end = dt.date(2024, 1, 5)    # Fri
        daily = [
            (dt.date(2024, 1, 1), 2.0),
            (dt.date(2024, 1, 3), 2.0),
        ]
        _, sharpe_full, n_full = compute_honest_sharpe(daily, start, end)

        # Same returns but compressed range = higher Sharpe
        _, sharpe_dense, n_dense = compute_honest_sharpe(
            daily, dt.date(2024, 1, 1), dt.date(2024, 1, 3)
        )
        assert n_full == 5
        assert n_dense == 3
        # Dense range should have higher Sharpe (same total R, fewer zero-days)
        assert sharpe_dense > sharpe_full

    def test_single_day_returns_none(self):
        sharpe_d, sharpe_ann, n = compute_honest_sharpe(
            [(dt.date(2024, 1, 1), 1.0)],
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 1),
        )
        assert n == 1
        assert sharpe_d is None
        assert sharpe_ann is None


# ── compute_drawdown ──────────────────────────────────────────────────────


class TestComputeDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing equity has 0 drawdown."""
        start = dt.date(2024, 1, 1)
        end = dt.date(2024, 1, 5)
        daily = [(dt.date(2024, 1, d), 1.0) for d in range(1, 6)]
        dd = compute_drawdown(daily, start, end)
        assert dd["max_dd_r"] == 0.0
        assert dd["longest_losing_streak"] == 0

    def test_simple_drawdown(self):
        """Win then lose then recover."""
        start = dt.date(2024, 1, 1)
        end = dt.date(2024, 1, 5)
        daily = [
            (dt.date(2024, 1, 1), 3.0),
            (dt.date(2024, 1, 2), -2.0),
            (dt.date(2024, 1, 3), -1.0),
            (dt.date(2024, 1, 4), 2.0),
            (dt.date(2024, 1, 5), 1.0),
        ]
        dd = compute_drawdown(daily, start, end)
        assert dd["max_dd_r"] == pytest.approx(3.0)
        assert dd["worst_single_day"] == pytest.approx(-2.0)
        assert dd["longest_losing_streak"] == 2

    def test_worst_day_tracked(self):
        start = dt.date(2024, 1, 1)
        end = dt.date(2024, 1, 3)
        daily = [
            (dt.date(2024, 1, 1), 1.0),
            (dt.date(2024, 1, 2), -5.0),
            (dt.date(2024, 1, 3), 2.0),
        ]
        dd = compute_drawdown(daily, start, end)
        assert dd["worst_single_day"] == pytest.approx(-5.0)
        assert dd["worst_single_day_date"] == dt.date(2024, 1, 2)

    def test_zero_days_break_losing_streak(self):
        """Zero-return days should NOT extend a losing streak."""
        # Mon-Fri: loss, 0, 0, 0, loss — should be streak of 1, not 5
        start = dt.date(2024, 1, 1)
        end = dt.date(2024, 1, 5)
        daily = [
            (dt.date(2024, 1, 1), -1.0),
            # Jan 2-4: zero-return days (no trades)
            (dt.date(2024, 1, 5), -1.0),
        ]
        dd = compute_drawdown(daily, start, end)
        assert dd["longest_losing_streak"] == 1

    def test_recovery_time(self):
        """Recovery time is computed from trough to when equity returns to peak."""
        start = dt.date(2024, 1, 1)
        end = dt.date(2024, 1, 5)
        daily = [
            (dt.date(2024, 1, 1), 3.0),   # cum=3, peak=3
            (dt.date(2024, 1, 2), -2.0),   # cum=1, dd=2
            (dt.date(2024, 1, 3), -1.0),   # cum=0, dd=3 (trough)
            (dt.date(2024, 1, 4), 2.0),    # cum=2, recovering
            (dt.date(2024, 1, 5), 1.0),    # cum=3, recovered!
        ]
        dd = compute_drawdown(daily, start, end)
        assert dd["recovery_days"] == (dt.date(2024, 1, 5) - dt.date(2024, 1, 3)).days

    def test_no_recovery(self):
        """If equity never returns to peak, recovery_days is None."""
        start = dt.date(2024, 1, 1)
        end = dt.date(2024, 1, 3)
        daily = [
            (dt.date(2024, 1, 1), 3.0),
            (dt.date(2024, 1, 2), -2.0),
            (dt.date(2024, 1, 3), -1.0),
        ]
        dd = compute_drawdown(daily, start, end)
        assert dd["recovery_days"] is None

    def test_dd_start_when_equity_starts_negative(self):
        """Portfolio that starts losing should still track drawdown start."""
        start = dt.date(2024, 1, 1)
        end = dt.date(2024, 1, 3)
        daily = [
            (dt.date(2024, 1, 1), -2.0),
            (dt.date(2024, 1, 2), -1.0),
            (dt.date(2024, 1, 3), 1.0),
        ]
        dd = compute_drawdown(daily, start, end)
        assert dd["max_dd_r"] == pytest.approx(3.0)
        assert dd["max_dd_start"] is not None
