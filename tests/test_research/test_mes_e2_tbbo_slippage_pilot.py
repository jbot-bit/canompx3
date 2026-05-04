from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd

import research.mes_e2_tbbo_slippage_pilot as mes


def _create_mes_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE daily_features (
                trading_day DATE,
                symbol VARCHAR,
                orb_minutes INTEGER,
                atr_20 DOUBLE,
                orb_CME_PRECLOSE_high DOUBLE,
                orb_CME_PRECLOSE_low DOUBLE,
                orb_CME_PRECLOSE_break_dir VARCHAR,
                orb_COMEX_SETTLE_high DOUBLE,
                orb_COMEX_SETTLE_low DOUBLE,
                orb_COMEX_SETTLE_break_dir VARCHAR,
                orb_SINGAPORE_OPEN_high DOUBLE,
                orb_SINGAPORE_OPEN_low DOUBLE,
                orb_SINGAPORE_OPEN_break_dir VARCHAR,
                orb_US_DATA_830_high DOUBLE,
                orb_US_DATA_830_low DOUBLE,
                orb_US_DATA_830_break_dir VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE orb_outcomes (
                trading_day DATE,
                symbol VARCHAR,
                orb_label VARCHAR,
                entry_model VARCHAR,
                confirm_bars INTEGER,
                orb_minutes INTEGER,
                outcome DOUBLE
            )
            """
        )
        rows = [
            ("2024-01-02", "CME_PRECLOSE", "long", 4800.0, 4798.0, 10.0),
            ("2024-01-03", "CME_PRECLOSE", "short", 4810.0, 4807.0, 30.0),
            ("2024-01-04", "COMEX_SETTLE", "long", 4820.0, 4818.0, 12.0),
            ("2024-01-05", "COMEX_SETTLE", "short", 4830.0, 4827.0, 32.0),
            ("2024-01-06", "SINGAPORE_OPEN", "long", 4840.0, 4838.0, 14.0),
            ("2024-01-07", "SINGAPORE_OPEN", "short", 4850.0, 4847.0, 34.0),
            ("2024-01-08", "US_DATA_830", "long", 4860.0, 4858.0, 16.0),
            ("2024-01-09", "US_DATA_830", "short", 4870.0, 4867.0, 36.0),
        ]
        for trading_day, session, direction, high, low, atr in rows:
            values = {
                "trading_day": trading_day,
                "symbol": "MES",
                "orb_minutes": 5,
                "atr_20": atr,
                "orb_CME_PRECLOSE_high": None,
                "orb_CME_PRECLOSE_low": None,
                "orb_CME_PRECLOSE_break_dir": None,
                "orb_COMEX_SETTLE_high": None,
                "orb_COMEX_SETTLE_low": None,
                "orb_COMEX_SETTLE_break_dir": None,
                "orb_SINGAPORE_OPEN_high": None,
                "orb_SINGAPORE_OPEN_low": None,
                "orb_SINGAPORE_OPEN_break_dir": None,
                "orb_US_DATA_830_high": None,
                "orb_US_DATA_830_low": None,
                "orb_US_DATA_830_break_dir": None,
            }
            values[f"orb_{session}_high"] = high
            values[f"orb_{session}_low"] = low
            values[f"orb_{session}_break_dir"] = direction
            con.execute(
                f"INSERT INTO daily_features VALUES ({','.join(['?'] * len(values))})",
                list(values.values()),
            )
            con.execute(
                "INSERT INTO orb_outcomes VALUES (?, 'MES', ?, 'E2', 1, 5, 1.0)",
                [trading_day, session],
            )
    finally:
        con.close()


def test_parse_cache_filename_accepts_mes_only():
    assert mes.parse_cache_filename("2024-01-02_CME_PRECLOSE_MES.dbn.zst") == ("2024-01-02", "CME_PRECLOSE")
    assert mes.parse_cache_filename("2024-01-02_US_DATA_830_MES.dbn.zst") == ("2024-01-02", "US_DATA_830")
    assert mes.parse_cache_filename("2024-01-02_CME_PRECLOSE_MNQ.dbn.zst") is None
    assert mes.parse_cache_filename("not-a-cache-file") is None


def test_validate_pilot_sessions_rejects_unsupported_session():
    assert mes._validate_pilot_sessions(["CME_PRECLOSE", "US_DATA_830"]) == ["CME_PRECLOSE", "US_DATA_830"]
    try:
        mes._validate_pilot_sessions(["NYSE_OPEN"])
    except ValueError as exc:
        assert "Unsupported MES pilot session" in str(exc)
        assert "CME_PRECLOSE" in str(exc)
    else:
        raise AssertionError("unsupported session was accepted")


def test_build_manifest_from_cache_uses_mes_daily_features(tmp_path, monkeypatch):
    db_path = tmp_path / "gold.db"
    _create_mes_db(db_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "2024-01-02_CME_PRECLOSE_MES.dbn.zst").write_bytes(b"stub")
    monkeypatch.setattr(mes, "GOLD_DB_PATH", db_path)

    manifest = mes.build_manifest_from_cache(cache_dir)

    assert manifest == [
        {
            "trading_day": "2024-01-02",
            "orb_label": "CME_PRECLOSE",
            "cache_path": str(cache_dir / "2024-01-02_CME_PRECLOSE_MES.dbn.zst"),
            "orb_high": 4800.0,
            "orb_low": 4798.0,
            "break_dir": "long",
            "atr_20": 10.0,
            "error": None,
        }
    ]


def test_build_manifest_from_cache_rejects_unsupported_session_filename(tmp_path, monkeypatch):
    db_path = tmp_path / "gold.db"
    _create_mes_db(db_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "2024-01-02_NYSE_OPEN_MES.dbn.zst"
    cache_path.write_bytes(b"stub")
    monkeypatch.setattr(mes, "GOLD_DB_PATH", db_path)

    assert mes.build_manifest_from_cache(cache_dir) == [
        {
            "trading_day": "2024-01-02",
            "orb_label": "NYSE_OPEN",
            "cache_path": str(cache_path),
            "orb_high": None,
            "orb_low": None,
            "break_dir": None,
            "atr_20": None,
            "error": "unsupported_session: NYSE_OPEN",
        }
    ]


def test_build_pilot_manifest_is_bounded_to_deployable_mes_sessions(tmp_path, monkeypatch):
    db_path = tmp_path / "gold.db"
    _create_mes_db(db_path)
    monkeypatch.setattr(mes, "GOLD_DB_PATH", db_path)
    monkeypatch.setattr(
        mes,
        "_orb_utc_window",
        lambda trading_day, session, minutes: (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        ),
    )

    manifest = mes.build_pilot_manifest(seed=7)

    assert {day.orb_label for day in manifest} == set(mes.PILOT_SESSIONS)
    assert len(manifest) == 8
    assert all(day.orb_label in {"CME_PRECLOSE", "COMEX_SETTLE", "SINGAPORE_OPEN", "US_DATA_830"} for day in manifest)
    assert all(day.window_start_utc.endswith("+00:00") for day in manifest)
    assert all(day.window_end_utc.endswith("+00:00") for day in manifest)


def test_reprice_cache_manifest_delegates_with_mes_cost_spec(tmp_path, monkeypatch):
    cache_path = tmp_path / "2024-01-02_CME_PRECLOSE_MES.dbn.zst"
    cache_path.write_bytes(b"stub")
    monkeypatch.setattr(mes, "load_tbbo_df", lambda path: pd.DataFrame({"price": [1.0]}))
    monkeypatch.setattr(
        mes,
        "_orb_utc_window",
        lambda trading_day, session, minutes: (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        ),
    )

    captured = {}

    def fake_reprice_e2_entry(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            trading_day=kwargs["trading_day"],
            break_dir=kwargs["break_dir"],
            orb_level=kwargs["orb_high"],
            trigger_trade_price=kwargs["orb_high"],
            bbo_at_trigger_bid=kwargs["orb_high"] - 0.25,
            bbo_at_trigger_ask=kwargs["orb_high"] + 0.25,
            bbo_at_trigger_spread=2.0,
            estimated_fill_price=kwargs["orb_high"] + 0.25,
            actual_slippage_points=0.25,
            actual_slippage_ticks=1.0,
            tbbo_records_in_window=10,
            error=None,
        )

    monkeypatch.setattr(mes, "reprice_e2_entry", fake_reprice_e2_entry)
    manifest = [
        {
            "trading_day": "2024-01-02",
            "orb_label": "CME_PRECLOSE",
            "cache_path": str(cache_path),
            "orb_high": 4800.0,
            "orb_low": 4798.0,
            "break_dir": "long",
            "atr_20": 10.0,
            "error": None,
        }
    ]

    results = mes.reprice_cache_manifest(manifest)

    assert results[0]["error"] is None
    assert captured["symbol_pulled"] == "MES.FUT"
    assert captured["tick_size"] == 0.25
    assert captured["modeled_slippage_ticks"] == 1
    assert captured["model_entry_price"] == 4800.25


def test_summarize_results_pass_warn_fail():
    base = {
        "trading_day": "2024-01-02",
        "orb_label": "CME_PRECLOSE",
        "break_dir": "long",
        "error": None,
    }
    assert (
        mes.summarize_results(pd.DataFrame([{**base, "slippage_ticks": 0.0}, {**base, "slippage_ticks": 1.0}]))[
            "verdict"
        ]
        == "PASS"
    )
    warn_df = pd.DataFrame(
        [{**base, "slippage_ticks": 0.0}, {**base, "slippage_ticks": 1.0}, {**base, "slippage_ticks": 8.0}]
    )
    assert mes.summarize_results(warn_df)["verdict"] == "WARN"
    assert (
        mes.summarize_results(pd.DataFrame([{**base, "slippage_ticks": 2.0}, {**base, "slippage_ticks": 3.0}]))[
            "verdict"
        ]
        == "FAIL"
    )
