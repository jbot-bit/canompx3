"""HTTP-level tests for /api/trade-book and /api/lane-status routes.

Covers the new endpoints added in d999facc on feat/live-app-ux. The existing
test_bot_dashboard.py covers helper functions only; without these, the routes
pass verify-complete Gate 3 vacuously because no test exercises them.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from trading_app.live import bot_dashboard


@pytest.fixture
def client():
    return TestClient(bot_dashboard.app)


def _seed_live_journal(path: Path) -> None:
    """Create a live_journal.db fixture with one live_trades row."""
    with duckdb.connect(str(path)) as con:
        con.execute(
            """
            CREATE TABLE live_trades (
                trade_id VARCHAR,
                trading_day DATE,
                instrument VARCHAR,
                strategy_id VARCHAR,
                direction VARCHAR,
                entry_model VARCHAR,
                fill_entry DOUBLE,
                fill_exit DOUBLE,
                actual_r DOUBLE,
                expected_r DOUBLE,
                pnl_dollars DOUBLE,
                exit_reason VARCHAR,
                contracts INTEGER,
                session_mode VARCHAR,
                broker VARCHAR,
                created_at TIMESTAMP,
                exited_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            INSERT INTO live_trades VALUES
            ('t1', DATE '2026-05-05', 'MNQ', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
             'long', 'E2', 18000.0, 18020.0, 1.0, 1.0, 100.0, 'tp', 1, 'LIVE',
             'topstep', TIMESTAMP '2026-05-05 14:30:00', TIMESTAMP '2026-05-05 15:00:00')
            """
        )


def _seed_gold_db(path: Path) -> None:
    """Create a gold.db fixture with one paper_trades row."""
    with duckdb.connect(str(path)) as con:
        con.execute(
            """
            CREATE TABLE paper_trades (
                trading_day DATE,
                instrument VARCHAR,
                strategy_id VARCHAR,
                lane_name VARCHAR,
                direction VARCHAR,
                entry_model VARCHAR,
                orb_label VARCHAR,
                orb_minutes INTEGER,
                rr_target DOUBLE,
                filter_type VARCHAR,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price DOUBLE,
                exit_price DOUBLE,
                pnl_r DOUBLE,
                pnl_dollar DOUBLE,
                slippage_ticks DOUBLE,
                exit_reason VARCHAR,
                execution_source VARCHAR
            )
            """
        )
        con.execute(
            """
            INSERT INTO paper_trades VALUES
            (DATE '2026-05-05', 'MNQ', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
             'NYSE_OPEN', 'long', 'E2', 'NYSE_OPEN', 5, 1.0, 'COST_LT12',
             TIMESTAMP '2026-05-05 14:30:00', TIMESTAMP '2026-05-05 15:00:00',
             18000.0, 18020.0, 1.0, 100.0, 0.5, 'tp', 'paper')
            """
        )


def test_trade_book_happy(client, monkeypatch, tmp_path):
    """Both DBs populated → endpoint returns separated arrays + matching counts."""
    journal = tmp_path / "live_journal.db"
    gold = tmp_path / "gold.db"
    _seed_live_journal(journal)
    _seed_gold_db(gold)
    monkeypatch.setattr(bot_dashboard, "JOURNAL_PATH", journal)
    monkeypatch.setattr(bot_dashboard, "GOLD_DB_PATH", gold)

    resp = client.get("/api/trade-book")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) >= {"live_trades", "paper_trades", "counts"}
    assert body["counts"] == {"live": 1, "paper": 1}
    assert len(body["live_trades"]) == 1
    assert len(body["paper_trades"]) == 1
    assert body["live_trades"][0]["instrument"] == "MNQ"
    assert body["paper_trades"][0]["execution_source"] == "paper"


def test_trade_book_empty(client, monkeypatch, tmp_path):
    """Both DBs exist but empty → arrays empty, no exception, counts zero."""
    journal = tmp_path / "live_journal.db"
    gold = tmp_path / "gold.db"
    # Create empty tables (no INSERT) so the files exist and queries return [].
    with duckdb.connect(str(journal)) as con:
        con.execute(
            """
            CREATE TABLE live_trades (
                trade_id VARCHAR, trading_day DATE, instrument VARCHAR,
                strategy_id VARCHAR, direction VARCHAR, entry_model VARCHAR,
                fill_entry DOUBLE, fill_exit DOUBLE, actual_r DOUBLE,
                expected_r DOUBLE, pnl_dollars DOUBLE, exit_reason VARCHAR,
                contracts INTEGER, session_mode VARCHAR, broker VARCHAR,
                created_at TIMESTAMP, exited_at TIMESTAMP
            )
            """
        )
    with duckdb.connect(str(gold)) as con:
        con.execute(
            """
            CREATE TABLE paper_trades (
                trading_day DATE, instrument VARCHAR, strategy_id VARCHAR,
                lane_name VARCHAR, direction VARCHAR, entry_model VARCHAR,
                orb_label VARCHAR, orb_minutes INTEGER, rr_target DOUBLE,
                filter_type VARCHAR, entry_time TIMESTAMP, exit_time TIMESTAMP,
                entry_price DOUBLE, exit_price DOUBLE, pnl_r DOUBLE,
                pnl_dollar DOUBLE, slippage_ticks DOUBLE, exit_reason VARCHAR,
                execution_source VARCHAR
            )
            """
        )
    monkeypatch.setattr(bot_dashboard, "JOURNAL_PATH", journal)
    monkeypatch.setattr(bot_dashboard, "GOLD_DB_PATH", gold)

    resp = client.get("/api/trade-book")
    assert resp.status_code == 200
    body = resp.json()
    assert body["counts"] == {"live": 0, "paper": 0}
    assert body["live_trades"] == []
    assert body["paper_trades"] == []


def test_trade_book_missing_db(client, monkeypatch, tmp_path):
    """Both DB paths absent → empty arrays + non-null *_note explaining absence.

    Mirrors /api/trades contract at bot_dashboard.py:1162 (graceful degradation,
    no HTTP 500). The route emits live_note/paper_note (not error) for the
    missing-file case — see bot_dashboard.py:1222 / :1266.
    """
    monkeypatch.setattr(bot_dashboard, "JOURNAL_PATH", tmp_path / "missing_journal.db")
    monkeypatch.setattr(bot_dashboard, "GOLD_DB_PATH", tmp_path / "missing_gold.db")

    resp = client.get("/api/trade-book")
    assert resp.status_code == 200
    body = resp.json()
    assert body["live_trades"] == []
    assert body["paper_trades"] == []
    assert body["counts"] == {"live": 0, "paper": 0}
    assert body["live_note"] and "live_journal" in body["live_note"]
    assert body["paper_note"] and "gold.db" in body["paper_note"]


def _seed_gold_db_n_paper_rows(path: Path, n: int) -> None:
    """Seed gold.db with N paper_trades rows (entry_time spaced by minutes
    so ORDER BY trading_day DESC, entry_time DESC is well-defined)."""
    with duckdb.connect(str(path)) as con:
        con.execute(
            """
            CREATE TABLE paper_trades (
                trading_day DATE, instrument VARCHAR, strategy_id VARCHAR,
                lane_name VARCHAR, direction VARCHAR, entry_model VARCHAR,
                orb_label VARCHAR, orb_minutes INTEGER, rr_target DOUBLE,
                filter_type VARCHAR, entry_time TIMESTAMP, exit_time TIMESTAMP,
                entry_price DOUBLE, exit_price DOUBLE, pnl_r DOUBLE,
                pnl_dollar DOUBLE, slippage_ticks DOUBLE, exit_reason VARCHAR,
                execution_source VARCHAR
            )
            """
        )
        # Bulk insert via VALUES is fast enough for 5050 rows.
        rows_sql = ",\n".join(
            f"(DATE '2026-05-05', 'MNQ', 's', 'l', 'long', 'E2', 'NYSE_OPEN', 5, 1.0, "
            f"'COST_LT12', TIMESTAMP '2026-05-05 14:30:00' + INTERVAL '{i} minutes', "
            f"TIMESTAMP '2026-05-05 15:00:00', 18000.0, 18020.0, 1.0, 100.0, 0.5, "
            f"'tp', 'paper')"
            for i in range(n)
        )
        con.execute(f"INSERT INTO paper_trades VALUES {rows_sql}")


def test_trade_book_paper_truncation_flag_under_limit(client, monkeypatch, tmp_path):
    """N=1 < limit → paper_truncated=False, paper_total_count=N, single query."""
    journal = tmp_path / "live_journal.db"
    gold = tmp_path / "gold.db"
    _seed_live_journal(journal)
    _seed_gold_db(gold)
    monkeypatch.setattr(bot_dashboard, "JOURNAL_PATH", journal)
    monkeypatch.setattr(bot_dashboard, "GOLD_DB_PATH", gold)

    resp = client.get("/api/trade-book")
    assert resp.status_code == 200
    body = resp.json()
    assert body["paper_truncated"] is False
    assert body["paper_total_count"] == 1
    assert len(body["paper_trades"]) == 1


def test_trade_book_paper_truncation_flag_over_limit(client, monkeypatch, tmp_path):
    """N=5050 > limit → 5000 returned, paper_truncated=True, total_count=5050."""
    journal = tmp_path / "live_journal.db"
    gold = tmp_path / "gold.db"
    _seed_live_journal(journal)
    _seed_gold_db_n_paper_rows(gold, 5050)
    monkeypatch.setattr(bot_dashboard, "JOURNAL_PATH", journal)
    monkeypatch.setattr(bot_dashboard, "GOLD_DB_PATH", gold)

    resp = client.get("/api/trade-book")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["paper_trades"]) == 5000, "must trim to LIMIT"
    assert body["paper_truncated"] is True
    assert body["paper_total_count"] == 5050
    # counts.paper reflects what's RETURNED, total_count reflects what EXISTS.
    assert body["counts"]["paper"] == 5000


def test_lane_status_happy(client, monkeypatch):
    """One paused strategy → response surfaces strategy_id, reason, expires_on."""
    paused_sid = "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12"
    override_payload = {
        "active": False,
        "reason": "SR alarm stat=33.27 ≥ thr=31.96",
        "since": "2026-04-23",
        "expires": "2026-05-23",
        "paused_at": "2026-04-23T12:00:00+00:00",
    }
    import trading_app.lane_ctl as lc

    monkeypatch.setattr(lc, "get_paused_strategy_ids", lambda profile: {paused_sid})
    monkeypatch.setattr(
        lc,
        "get_lane_override",
        lambda profile, sid: override_payload if sid == paused_sid else None,
    )

    resp = client.get("/api/lane-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["paused_count"] == 1
    assert body["profile"] == "topstep_50k_mnq_auto"
    assert len(body["paused"]) == 1
    entry = body["paused"][0]
    assert entry["strategy_id"] == paused_sid
    assert entry["reason"] == "SR alarm stat=33.27 ≥ thr=31.96"
    assert entry["expires_on"] == "2026-05-23"


def test_lane_status_empty(client, monkeypatch):
    """Zero paused → paused_count=0, paused=[], no error field."""
    import trading_app.lane_ctl as lc

    monkeypatch.setattr(lc, "get_paused_strategy_ids", lambda profile: set())

    resp = client.get("/api/lane-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["paused_count"] == 0
    assert body["paused"] == []
    assert "error" not in body


def test_lane_status_bad_profile(client):
    """Unknown profile id → response carries error string, no HTTP 500.

    The route wraps the lane_ctl lookup in try/except and emits an `error`
    field rather than crashing — see bot_dashboard.py:1324.
    """
    resp = client.get("/api/lane-status", params={"profile": "__nonexistent_profile__"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"] == "__nonexistent_profile__"
    # The lookup may succeed (returning empty) or fail; either way there must
    # be no HTTP 500. If it fails, error is non-null; if it succeeds, paused
    # must be empty for an unknown profile.
    if body.get("error"):
        assert isinstance(body["error"], str)
    else:
        assert body["paused"] == []
        assert body["paused_count"] == 0
