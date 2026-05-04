from __future__ import annotations

import datetime as dt

from research.mnq_nyse_close_rr10_role_audit import (
    compute_snapshot,
    daily_return_frame,
    determine_role_verdict,
)


def _trade(day: dt.date, pnl_r: float, sid: str) -> dict:
    return {
        "trading_day": day,
        "outcome": "win" if pnl_r > 0 else "loss",
        "pnl_r": pnl_r,
        "instrument": "MNQ",
        "session": "TEST",
        "strategy_id": sid,
    }


def test_daily_return_frame_zero_fills_missing_business_days() -> None:
    trades = {
        "S1": [_trade(dt.date(2024, 1, 1), 1.0, "S1"), _trade(dt.date(2024, 1, 3), -1.0, "S1")],
        "S2": [_trade(dt.date(2024, 1, 2), 0.5, "S2")],
    }
    frame = daily_return_frame(trades, dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert list(frame.index) == [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    assert frame.loc[dt.date(2024, 1, 2), "S1"] == 0.0
    assert frame.loc[dt.date(2024, 1, 1), "S2"] == 0.0


def test_compute_snapshot_annualizes_over_business_days() -> None:
    trades = {
        "S1": [
            _trade(dt.date(2024, 1, 1), 1.0, "S1"),
            _trade(dt.date(2024, 1, 2), -1.0, "S1"),
            _trade(dt.date(2024, 1, 3), 2.0, "S1"),
        ]
    }
    snap = compute_snapshot("test", trades, dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert snap.total_r == 2.0
    assert snap.business_days == 3
    assert snap.annual_r == 168.0
    assert snap.trade_days == 3
    assert snap.total_trades == 3


def test_determine_role_verdict_prefers_allocator_candidate_when_add_is_clean() -> None:
    base = compute_snapshot(
        "base",
        {"A": [_trade(dt.date(2024, 1, 1), 0.5, "A"), _trade(dt.date(2024, 1, 2), 0.0, "A")]},
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 2),
    )
    plus = compute_snapshot(
        "plus",
        {"A": [_trade(dt.date(2024, 1, 1), 0.5, "A")], "NYC": [_trade(dt.date(2024, 1, 2), 1.0, "NYC")]},
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 2),
    )
    candidate = compute_snapshot(
        "candidate",
        {"NYC": [_trade(dt.date(2024, 1, 2), 1.0, "NYC")]},
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 2),
    )
    verdict, reason = determine_role_verdict(
        base_is=base,
        plus_is=plus,
        candidate_is=candidate,
        corr_summary_is={"corr_to_base_portfolio": 0.0},
        max_slots=7,
        live_lane_count=6,
    )
    assert verdict == "CONTINUE as allocator candidate"
    assert "Sharpe improve" in reason or "Sharpe" in reason
