"""HTTP-level tests for /api/trade-book and /api/lane-status routes.

Covers the new endpoints added in d999facc on feat/live-app-ux. The existing
test_bot_dashboard.py covers helper functions only; without these, the routes
pass verify-complete Gate 3 vacuously because no test exercises them.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import duckdb
import httpx
import pytest

from trading_app.live import bot_dashboard
from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes


class _ASGIClient:
    """Tiny synchronous ASGI test client that avoids Starlette TestClient's portal thread."""

    def __init__(self, app):
        self._app = app

    def get(self, url: str, **kwargs) -> httpx.Response:
        async def _get() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(url, **kwargs)

        return anyio.run(_get)

    def post(self, url: str, **kwargs) -> httpx.Response:
        async def _post() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(url, **kwargs)

        return anyio.run(_post)


@pytest.fixture
def client():
    return _ASGIClient(bot_dashboard.app)


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


def test_lane_control_surfaces_active_and_parked_lanes(client, monkeypatch, tmp_path):
    import trading_app.lane_ctl as lc

    monkeypatch.setattr(lc, "STATE_DIR", tmp_path)
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {"running": False, "raw_mode": "STOPPED", "profile": None},
    )
    lc.pause_strategy_id(
        "topstep_50k_mnq_auto",
        "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
        reason="SR ALARM live-pilot exclusion",
        source="test",
    )

    resp = client.get("/api/lane-control", params={"profile": "topstep_50k_mnq_auto"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"] == "topstep_50k_mnq_auto"
    assert body["enabled_count"] == body["active_book_count"]
    assert body["parked_count"] == 1
    assert body["counts"] == {
        "active": body["active_book_count"],
        "enabled": body["enabled_count"],
        "disabled": body["disabled_count"],
        "parked": body["parked_count"],
    }
    parked = next(row for row in body["lanes"] if row["strategy_id"] == "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12")
    assert parked["status"] == "PARKED"
    assert parked["active_book"] is False
    assert "SR ALARM" in parked["reason"]


def test_dashboard_live_start_blocks_non_pilot_profile(client):
    resp = client.post("/api/action/start", params={"profile": "self_funded_tradovate", "mode": "live"})

    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "blocked"
    assert body["profile"] == "self_funded_tradovate"
    assert "approved Topstep MNQ pilot" in body["message"]


def test_lane_control_toggle_pauses_and_resumes_active_lane(client, monkeypatch, tmp_path):
    import trading_app.lane_ctl as lc

    monkeypatch.setattr(lc, "STATE_DIR", tmp_path)
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {"running": False, "raw_mode": "STOPPED", "profile": None},
    )
    active_sid = effective_daily_lanes(ACCOUNT_PROFILES["topstep_50k_mnq_auto"])[0].strategy_id

    off = client.post(
        "/api/lane-control/toggle",
        json={"profile": "topstep_50k_mnq_auto", "strategy_id": active_sid, "enabled": False},
    )
    assert off.status_code == 200
    assert off.json()["enabled"] is False
    assert lc.get_lane_override("topstep_50k_mnq_auto", active_sid)["reason"] == "Dashboard lane off"

    on = client.post(
        "/api/lane-control/toggle",
        json={"profile": "topstep_50k_mnq_auto", "strategy_id": active_sid, "enabled": True},
    )
    assert on.status_code == 200
    assert on.json()["enabled"] is True
    assert lc.get_lane_override("topstep_50k_mnq_auto", active_sid) is None


def test_lane_control_toggle_blocks_when_session_running(client, monkeypatch, tmp_path):
    import trading_app.lane_ctl as lc

    monkeypatch.setattr(lc, "STATE_DIR", tmp_path)
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {"running": True, "raw_mode": "SIGNAL", "profile": "topstep_50k_mnq_auto"},
    )
    active_sid = effective_daily_lanes(ACCOUNT_PROFILES["topstep_50k_mnq_auto"])[0].strategy_id

    resp = client.post(
        "/api/lane-control/toggle",
        json={"profile": "topstep_50k_mnq_auto", "strategy_id": active_sid, "enabled": False},
    )

    assert resp.status_code == 409
    assert resp.json()["status"] == "blocked"
    assert lc.get_lane_override("topstep_50k_mnq_auto", active_sid) is None


def test_lane_control_toggle_blocks_parked_lane(client, monkeypatch, tmp_path):
    import trading_app.lane_ctl as lc

    monkeypatch.setattr(lc, "STATE_DIR", tmp_path)
    monkeypatch.setattr(
        bot_dashboard,
        "_session_snapshot",
        lambda: {"running": False, "raw_mode": "STOPPED", "profile": None},
    )

    resp = client.post(
        "/api/lane-control/toggle",
        json={
            "profile": "topstep_50k_mnq_auto",
            "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
            "enabled": True,
        },
    )

    assert resp.status_code == 400
    assert resp.json()["status"] == "blocked"


# ---------- /api/sessions market_open contract (2026-05-30 TZ fix) ----------
#
# Root cause: the endpoint (and START_BOT.bat) keyed off the Brisbane weekday,
# which reads "Saturday" during the entire Friday US session while the CME
# market is open. The fix surfaces market_open from the canonical
# pipeline.market_calendar.is_market_open_at so display agrees with market truth
# rather than the local weekday. These tests lock that contract.


def test_sessions_market_open_true_when_market_open(client, monkeypatch):
    """market_open=True → endpoint surfaces it and still computes next session."""
    monkeypatch.setattr(
        "pipeline.market_calendar.is_market_open_at",
        lambda _utc_now: True,
    )
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("error") is None
    assert body["market_open"] is True
    assert "sessions" in body


def test_sessions_market_open_false_on_weekend_gap(client, monkeypatch):
    """market_open=False (genuine weekend/holiday closure) must be surfaced so
    the frontend shows 'Market closed' instead of a misleading countdown.

    This is the regression guard for the Saturday-Brisbane / Friday-ET defect:
    the value must come from is_market_open_at, NOT the Brisbane weekday."""
    monkeypatch.setattr(
        "pipeline.market_calendar.is_market_open_at",
        lambda _utc_now: False,
    )
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("error") is None
    assert body["market_open"] is False
