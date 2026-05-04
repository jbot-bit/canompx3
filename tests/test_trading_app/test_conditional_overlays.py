from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from math import nan
from pathlib import Path

import duckdb

from trading_app import conditional_overlays


def _write_breakpoint_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "instrument,lane,session,direction,is_n,p20,p40,p60,p80",
                "MGC,TOKYO_OPEN_long,TOKYO_OPEN,long,100,0.5,1.0,1.5,2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _build_daily_features_db(path: Path, *, trading_day: date, rel_vol_tokyo_open: float) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE daily_features (
                symbol VARCHAR,
                orb_minutes INTEGER,
                trading_day DATE,
                rel_vol_TOKYO_OPEN DOUBLE
            )
            """
        )
        con.execute(
            """
            INSERT INTO daily_features(symbol, orb_minutes, trading_day, rel_vol_TOKYO_OPEN)
            VALUES (?, ?, ?, ?)
            """,
            ["MGC", 5, trading_day, rel_vol_tokyo_open],
        )
    finally:
        con.close()


def test_current_trading_day_uses_brisbane_0900_boundary():
    assert conditional_overlays._current_trading_day(
        datetime(2026, 4, 23, 8, 59, tzinfo=conditional_overlays.BRISBANE_TZ)
    ) == date(2026, 4, 22)
    assert conditional_overlays._current_trading_day(
        datetime(2026, 4, 23, 9, 0, tzinfo=conditional_overlays.BRISBANE_TZ)
    ) == date(2026, 4, 23)


def test_read_overlay_states_auto_refreshes_missing_state(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    db_path = tmp_path / "gold.db"
    csv_path = tmp_path / "breakpoints.csv"
    trading_day = date(2026, 4, 23)
    _write_breakpoint_csv(csv_path)
    _build_daily_features_db(db_path, trading_day=trading_day, rel_vol_tokyo_open=3.25)

    spec = replace(
        conditional_overlays.PR48_MGC_CONT_EXEC_V1,
        overlay_id="test_overlay",
        profile_id="topstep_50k",
        sessions=("TOKYO_OPEN",),
        directions=("long",),
        breakpoint_artifact_path=csv_path,
    )

    monkeypatch.setattr(conditional_overlays, "STATE_DIR", state_dir)
    monkeypatch.setattr(conditional_overlays, "CONDITIONAL_OVERLAYS", {spec.overlay_id: spec})
    monkeypatch.setattr(conditional_overlays, "build_code_fingerprint", lambda _paths: "code-identity")
    monkeypatch.setattr(conditional_overlays, "get_git_head", lambda _root=None: "testsha")

    state = conditional_overlays.read_overlay_states("topstep_50k", db_path=db_path, today=trading_day)

    assert state["available"] is True
    assert state["valid"] is True
    assert len(state["overlays"]) == 1
    overlay = state["overlays"][0]
    assert overlay["overlay_id"] == "test_overlay"
    assert overlay["status"] == "ready"
    assert overlay["summary"]["ready_count"] == 1
    assert overlay["summary"]["row_count"] == 1
    assert overlay["rows"][0]["bucket"] == 5
    assert overlay["rows"][0]["size_multiplier"] == 2.0
    assert conditional_overlays.get_overlay_state_path("topstep_50k", "test_overlay").exists()


def test_read_overlay_states_without_today_uses_current_trading_day(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    db_path = tmp_path / "gold.db"
    csv_path = tmp_path / "breakpoints.csv"
    trading_day = date(2026, 4, 22)
    _write_breakpoint_csv(csv_path)
    _build_daily_features_db(db_path, trading_day=trading_day, rel_vol_tokyo_open=3.25)

    spec = replace(
        conditional_overlays.PR48_MGC_CONT_EXEC_V1,
        overlay_id="boundary_overlay",
        profile_id="topstep_50k",
        sessions=("TOKYO_OPEN",),
        directions=("long",),
        breakpoint_artifact_path=csv_path,
    )

    monkeypatch.setattr(conditional_overlays, "STATE_DIR", state_dir)
    monkeypatch.setattr(conditional_overlays, "CONDITIONAL_OVERLAYS", {spec.overlay_id: spec})
    monkeypatch.setattr(conditional_overlays, "build_code_fingerprint", lambda _paths: "code-identity")
    monkeypatch.setattr(conditional_overlays, "get_git_head", lambda _root=None: "testsha")
    monkeypatch.setattr(conditional_overlays, "_current_trading_day", lambda: trading_day)

    state = conditional_overlays.read_overlay_states("topstep_50k", db_path=db_path)

    overlay = state["overlays"][0]
    assert overlay["valid"] is True
    assert overlay["status"] == "ready"
    assert overlay["state_date"] == "2026-04-22"


def test_read_overlay_states_degrades_invalid_artifact(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    db_path = tmp_path / "gold.db"
    trading_day = date(2026, 4, 23)
    _build_daily_features_db(db_path, trading_day=trading_day, rel_vol_tokyo_open=1.25)

    spec = replace(
        conditional_overlays.PR48_MGC_CONT_EXEC_V1,
        overlay_id="bad_overlay",
        profile_id="topstep_50k",
        sessions=("TOKYO_OPEN",),
        directions=("long",),
        breakpoint_artifact_path=tmp_path / "missing.csv",
    )

    monkeypatch.setattr(conditional_overlays, "STATE_DIR", state_dir)
    monkeypatch.setattr(conditional_overlays, "CONDITIONAL_OVERLAYS", {spec.overlay_id: spec})
    monkeypatch.setattr(conditional_overlays, "build_code_fingerprint", lambda _paths: "code-identity")

    state = conditional_overlays.read_overlay_states("topstep_50k", db_path=db_path, today=trading_day)

    assert state["available"] is True
    assert state["valid"] is False
    assert len(state["overlays"]) == 1
    overlay = state["overlays"][0]
    assert overlay["overlay_id"] == "bad_overlay"
    assert overlay["valid"] is False
    assert overlay["status"] == "invalid"
    assert "Missing breakpoint artifact" in str(overlay["reason"])


def test_read_overlay_states_marks_semantic_invalid_status_invalid(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    db_path = tmp_path / "gold.db"
    csv_path = tmp_path / "breakpoints.csv"
    trading_day = date(2026, 4, 23)
    _write_breakpoint_csv(csv_path)
    _build_daily_features_db(db_path, trading_day=trading_day, rel_vol_tokyo_open=1.25)

    spec = replace(
        conditional_overlays.PR48_MGC_CONT_EXEC_V1,
        overlay_id="semantic_bad_overlay",
        profile_id="topstep_50k",
        sessions=("TOKYO_OPEN",),
        directions=("short",),
        breakpoint_artifact_path=csv_path,
    )

    monkeypatch.setattr(conditional_overlays, "STATE_DIR", state_dir)
    monkeypatch.setattr(conditional_overlays, "CONDITIONAL_OVERLAYS", {spec.overlay_id: spec})
    monkeypatch.setattr(conditional_overlays, "build_code_fingerprint", lambda _paths: "code-identity")
    monkeypatch.setattr(conditional_overlays, "get_git_head", lambda _root=None: "testsha")

    state = conditional_overlays.read_overlay_states("topstep_50k", db_path=db_path, today=trading_day)

    overlay = state["overlays"][0]
    assert state["valid"] is False
    assert overlay["valid"] is False
    assert overlay["status"] == "invalid"
    assert overlay["reason"] == "missing breakpoint row"


def test_build_rows_does_not_bucket_non_finite_feature_value():
    spec = replace(
        conditional_overlays.PR48_MGC_CONT_EXEC_V1,
        sessions=("TOKYO_OPEN",),
        directions=("long",),
    )

    rows = conditional_overlays._build_rows(
        spec,
        feature_row={"rel_vol_TOKYO_OPEN": nan},
        breakpoints={("TOKYO_OPEN", "long"): {"p20": 0.5, "p40": 1.0, "p60": 1.5, "p80": 2.0}},
    )

    assert rows[0]["status"] == "unscored"
    assert rows[0]["bucket"] is None
    assert "non-finite" in rows[0]["reason"]
