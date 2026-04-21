from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import duckdb
import pytest

from pipeline.build_daily_features import _load_postpass_seed_rows, build_daily_features
from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA

UTC_TZ = ZoneInfo("UTC")


def _make_db(tmp_path):
    db_path = tmp_path / "incremental_seed.db"
    con = duckdb.connect(str(db_path))
    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    return con


def _seed_prior_daily_features(con, symbol: str, orb_minutes: int, end_day: date, n_rows: int) -> None:
    start_ord = end_day.toordinal() - (n_rows - 1)
    for i in range(n_rows):
        td = date.fromordinal(start_ord + i)
        close = 1000.0 + i
        con.execute(
            """
            INSERT INTO daily_features (
                trading_day, symbol, orb_minutes,
                daily_open, daily_high, daily_low, daily_close,
                overnight_range, atr_20, garch_forecast_vol
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                td,
                symbol,
                orb_minutes,
                close - 1.0,
                close + 2.0,
                close - 2.0,
                close,
                10.0 + (i % 5),
                20.0 + (i % 7),
                0.100 + i / 1000.0,
            ],
        )


def _seed_bars_for_trading_day(con, symbol: str, trading_day: date, base_price: float) -> None:
    td_start = datetime(trading_day.year, trading_day.month, trading_day.day, 23, 0, 0, tzinfo=UTC_TZ) - timedelta(
        days=1
    )
    for i in range(90):
        ts = td_start + timedelta(minutes=i)
        price = base_price + i * 0.25
        con.execute(
            "INSERT INTO bars_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [ts.isoformat(), symbol, f"{symbol}M4", price, price + 1.0, price - 1.0, price + 0.2, 100 + i],
        )

    base_5m = td_start - timedelta(hours=6)
    for i in range(40):
        ts = base_5m + timedelta(minutes=i * 5)
        price = base_price - 5.0 + i * 0.1
        con.execute(
            "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [ts.isoformat(), symbol, f"{symbol}M4", price, price + 0.5, price - 0.5, price + 0.1, 500],
        )


def test_load_postpass_seed_rows_scopes_symbol_and_aperture(tmp_path):
    con = _make_db(tmp_path)
    try:
        _seed_prior_daily_features(con, "MGC", 5, date(2024, 1, 4), 10)
        _seed_prior_daily_features(con, "MGC", 15, date(2024, 1, 4), 10)
        _seed_prior_daily_features(con, "MNQ", 5, date(2024, 1, 4), 10)

        seed = _load_postpass_seed_rows(con, "MGC", 5, date(2024, 1, 5), lookback_rows=20)
        assert len(seed) == 10
        assert all(r["symbol"] == "MGC" for r in seed)
        assert all(r["orb_minutes"] == 5 for r in seed)
        assert seed[0]["trading_day"] == date(2023, 12, 26)
        assert seed[-1]["trading_day"] == date(2024, 1, 4)
    finally:
        con.close()


def test_incremental_build_uses_seed_for_rolling_features_and_prev_day(monkeypatch, tmp_path):
    con = _make_db(tmp_path)
    try:
        # Seed enough prior daily_features rows so a 1-day incremental rebuild
        # should have full rolling context without recomputing history.
        _seed_prior_daily_features(con, "MGC", 5, date(2024, 1, 4), 260)
        _seed_bars_for_trading_day(con, "MGC", date(2024, 1, 5), 2000.0)

        def _fake_garch(closes, min_obs=252):
            return round(len(closes) / 1000.0, 6) if len(closes) >= min_obs else None

        monkeypatch.setattr("pipeline.build_daily_features.compute_garch_forecast", _fake_garch)

        count = build_daily_features(con, "MGC", date(2024, 1, 5), date(2024, 1, 5), 5, False)
        assert count == 1

        row = con.execute(
            """
            SELECT prev_day_close, gap_open_points, atr_20, atr_20_pct,
                   overnight_range_pct, garch_forecast_vol, garch_forecast_vol_pct
            FROM daily_features
            WHERE symbol = 'MGC' AND trading_day = '2024-01-05' AND orb_minutes = 5
            """
        ).fetchone()

        assert row is not None
        prev_day_close, gap_open_points, atr_20, atr_20_pct, overnight_range_pct, garch_vol, garch_pct = row
        assert prev_day_close == pytest.approx(1259.0)
        assert gap_open_points is not None
        assert atr_20 is not None
        assert atr_20_pct is not None
        assert overnight_range_pct is not None
        assert garch_vol == pytest.approx(0.252)
        assert garch_pct == pytest.approx(57.14, abs=0.01)
    finally:
        con.close()
