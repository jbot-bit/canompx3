"""
Tests for the forward-only REGIME shadow accumulator.

Central safety properties (capital-path):
  1. SEPARATION: shadow rows are invisible to every live/monitoring consumer
     (which filter execution_source='live'/'backfill'). Running the runner adds
     ONLY source='shadow' rows; live/backfill counts are untouched.
  2. FORWARD-ONLY: no row earlier than forward_start is ever written; a sourced
     pre-boundary row is a fail-closed ValueError.
  3. IDEMPOTENT: re-running the sync does not double-count shadow rows.
  4. DRY-RUN: writes nothing.

Fixtures use a temp DuckDB with the canonical paper_trades schema plus minimal
orb_outcomes / daily_features rows. build_universe is monkeypatched to inject a
controlled lane set so we test the runner mechanics, not the fitness engine.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb
import pytest

from scripts.tools import regime_shadow_runner as runner
from scripts.tools.regime_shadow_universe import RegimeLane

FWD = datetime.date(2026, 6, 1)
TODAY = datetime.date(2026, 6, 10)


def _lane(strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER_O5", **kw):
    base = dict(
        strategy_id=strategy_id,
        instrument="MNQ",
        orb_label="COMEX_SETTLE",
        orb_minutes=5,
        rr_target=1.0,
        entry_model="E2",
        confirm_bars=1,
        filter_type="NO_FILTER",
        sample_size=60,
        fitness_status="FIT",
        rolling_sample=40,
        rolling_exp_r=0.5,
        included=True,
        reason="FIT — recorded",
    )
    base.update(kw)
    return RegimeLane(**base)


def _make_db(tmp_path: Path, outcome_days: list[datetime.date]) -> Path:
    """Build a temp DB using the CANONICAL trading-app schema, then seed minimal
    orb_outcomes + daily_features so the NO_FILTER lane passes matches_row on
    each day. Named-column INSERTs keep the fixture robust to canonical schema
    column-order/count changes (we never hand-roll the canonical tables)."""
    from trading_app.db_manager import init_trading_app_schema

    db = tmp_path / "shadow_test.db"

    # daily_features is the FK target of orb_outcomes — must exist before
    # init_trading_app_schema. Carry the break_dir column the filter reads.
    with duckdb.connect(str(db)) as con:
        con.execute(
            """CREATE TABLE daily_features (
                 symbol VARCHAR, orb_minutes INTEGER, trading_day DATE,
                 orb_COMEX_SETTLE_break_dir INTEGER,
                 UNIQUE (symbol, trading_day, orb_minutes))"""
        )

    # init creates the canonical orb_outcomes + paper_trades (with the migration).
    init_trading_app_schema(db_path=db)

    with duckdb.connect(str(db)) as con:
        # Seed an existing 'backfill' row to prove separation leaves it untouched.
        con.execute(
            """INSERT INTO paper_trades
               (trading_day, orb_label, strategy_id, instrument, pnl_r, execution_source)
               VALUES (DATE '2026-05-01', 'COMEX_SETTLE', 'OTHER_LANE', 'MNQ', 0.3, 'backfill')"""
        )
        for d in outcome_days:
            # daily_features FIRST — it is the FK target of orb_outcomes.
            con.execute(
                "INSERT INTO daily_features (symbol, orb_minutes, trading_day, orb_COMEX_SETTLE_break_dir) "
                "VALUES ('MNQ', 5, ?, 1)",
                [d],
            )
            con.execute(
                """INSERT INTO orb_outcomes
                   (symbol, orb_label, orb_minutes, rr_target, entry_model,
                    confirm_bars, trading_day, entry_ts, entry_price, stop_price,
                    target_price, exit_price, exit_ts, outcome, pnl_r)
                   VALUES ('MNQ','COMEX_SETTLE',5,1.0,'E2',1,?, ?, 100.0, 99.0,
                           102.0, 102.0, ?, 'win', 1.0)""",
                [
                    d,
                    datetime.datetime(d.year, d.month, d.day, 14, 0, tzinfo=datetime.UTC),
                    datetime.datetime(d.year, d.month, d.day, 15, 0, tzinfo=datetime.UTC),
                ],
            )
    return db


def _source_counts(db: Path) -> dict:
    with duckdb.connect(str(db), read_only=True) as con:
        return dict(con.execute("SELECT execution_source, COUNT(*) FROM paper_trades GROUP BY 1").fetchall())


# ── 1. SEPARATION ────────────────────────────────────────────────────────


def test_sync_writes_only_shadow_source_and_leaves_backfill_untouched(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2), datetime.date(2026, 6, 3)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])

    before = _source_counts(db)
    assert before == {"backfill": 1}

    summary = runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")

    after = _source_counts(db)
    assert after.get("backfill") == 1, "backfill rows must be untouched"
    assert after.get("shadow") == 2, "two forward days -> two shadow rows"
    assert summary.trades_appended == 2


def test_live_query_does_not_see_shadow_rows(tmp_path, monkeypatch):
    """The exact predicate live/monitoring code uses (execution_source='live')
    returns zero shadow rows — the structural invisibility proof."""
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])
    runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")

    with duckdb.connect(str(db), read_only=True) as con:
        live_seen = con.execute("SELECT COUNT(*) FROM paper_trades WHERE execution_source = 'live'").fetchone()[0]
        backfill_seen = con.execute("SELECT COUNT(*) FROM paper_trades WHERE execution_source = 'backfill'").fetchone()[
            0
        ]
        shadow_seen = con.execute("SELECT COUNT(*) FROM paper_trades WHERE execution_source = 'shadow'").fetchone()[0]
    assert live_seen == 0, "live path must never see shadow rows"
    assert backfill_seen == 1
    assert shadow_seen == 1


# ── 2. FORWARD-ONLY ──────────────────────────────────────────────────────


def test_no_pre_boundary_row_written(tmp_path, monkeypatch):
    """Outcomes exist before AND after the boundary; only >= boundary written."""
    db = _make_db(
        tmp_path,
        [datetime.date(2026, 5, 20), datetime.date(2026, 6, 2), datetime.date(2026, 6, 5)],
    )
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])
    runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")

    with duckdb.connect(str(db), read_only=True) as con:
        days = [
            r[0]
            for r in con.execute(
                "SELECT trading_day FROM paper_trades WHERE execution_source='shadow' ORDER BY 1"
            ).fetchall()
        ]
    assert days, "expected forward rows written (test would pass vacuously if empty)"
    assert all(d >= FWD for d in days), f"pre-boundary row leaked: {days}"
    assert datetime.date(2026, 5, 20) not in days
    assert set(days) == {datetime.date(2026, 6, 2), datetime.date(2026, 6, 5)}


def test_forward_boundary_violation_raises(tmp_path, monkeypatch):
    """If the source query were ever to return a pre-boundary row, the tripwire
    fails closed. We force it by pointing _query_outcomes at an early row."""
    db = _make_db(tmp_path, [datetime.date(2026, 5, 1)])
    lane = _lane()

    # Force a pre-boundary row through the source to exercise the tripwire.
    import trading_app.paper_trade_logger as ptl

    monkeypatch.setattr(
        ptl,
        "_query_outcomes",
        lambda con, ld, since=None: [
            (
                datetime.date(2026, 5, 1),
                "COMEX_SETTLE",
                None,
                "long",
                100.0,
                99.0,
                102.0,
                102.0,
                None,
                "win",
                1.0,
            )
        ],
    )
    monkeypatch.setattr(
        ptl,
        "_load_features",
        lambda con, inst, om, since=None: {datetime.date(2026, 5, 1): {"orb_COMEX_SETTLE_break_dir": 1}},
    )

    with duckdb.connect(str(db), read_only=True) as con:
        with pytest.raises(ValueError, match="FORWARD BOUNDARY VIOLATION"):
            runner._shadow_rows_for_lane(con, lane, FWD)


# ── 3. IDEMPOTENT ────────────────────────────────────────────────────────


def test_idempotent_resync_does_not_double_count(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2), datetime.date(2026, 6, 3)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])

    runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")
    runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")

    assert _source_counts(db).get("shadow") == 2, "re-sync must not duplicate shadow rows"


# ── 4. DRY-RUN ───────────────────────────────────────────────────────────


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2), datetime.date(2026, 6, 3)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])

    summary = runner.sync_shadow(db_path=db, dry_run=True, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")

    assert _source_counts(db) == {"backfill": 1}, "dry-run must not write shadow rows"
    assert summary.trades_appended == 2, "dry-run still reports would-append count"


# ── forward_start persistence ────────────────────────────────────────────


def test_forward_start_persists_and_is_reused(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])
    uyaml = tmp_path / "u.yaml"

    # First run at TODAY persists boundary = TODAY.
    runner.sync_shadow(db_path=db, as_of_date=TODAY, universe_yaml=uyaml)
    fs, persisted = runner.resolve_forward_start(uyaml, today=datetime.date(2026, 7, 1))
    assert persisted is True
    assert fs == TODAY, "boundary must be fixed by first run, not drift to a later 'today'"


def test_oos_context_report_writes_nothing(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [datetime.date(2026, 1, 5), datetime.date(2026, 6, 2)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])

    results = runner.oos_context_report(db_path=db, as_of_date=FWD)

    assert _source_counts(db) == {"backfill": 1}, "OOS report must insert nothing"
    # would-record includes the Jan row (OOS boundary), proving the path works.
    assert sum(r.appended for r in results) >= 1


# ── G2: boundary durability ──────────────────────────────────────────────


def test_write_universe_yaml_preserves_persisted_boundary(tmp_path):
    """A universe refresh must NOT move an already-persisted forward_start."""
    from scripts.tools.regime_shadow_universe import write_universe_yaml

    uyaml = tmp_path / "u.yaml"
    write_universe_yaml([_lane()], forward_start=FWD, path=uyaml, as_of_date=TODAY)
    # Refresh with a LATER boundary — must be ignored, original preserved.
    write_universe_yaml(
        [_lane()], forward_start=datetime.date(2026, 7, 1), path=uyaml, as_of_date=datetime.date(2026, 7, 1)
    )

    fs, persisted = runner.resolve_forward_start(uyaml, today=datetime.date(2026, 8, 1))
    assert persisted is True
    assert fs == FWD, "refresh must preserve the original boundary, never advance it"


def test_missing_yaml_with_existing_shadow_rows_rederives_boundary(tmp_path, monkeypatch):
    """G2 fail-closed: YAML deleted but shadow rows exist -> boundary re-derived
    as MIN(trading_day) of shadow rows, NOT reset to today."""
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2), datetime.date(2026, 6, 3)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])
    uyaml = tmp_path / "u.yaml"
    runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=uyaml)  # boundary = FWD
    assert _source_counts(db).get("shadow") == 2

    uyaml.unlink()  # simulate accidental deletion

    fs, persisted = runner.resolve_forward_start(uyaml, today=TODAY, db_path=db)
    assert persisted is True, "re-derived boundary is authoritative, not re-stamped"
    assert fs == datetime.date(2026, 6, 2), "boundary re-derived from MIN(shadow trading_day), not today"


# ── G5: stale-orphan reconcile ───────────────────────────────────────────


def test_orphan_reported_not_pruned_by_default(tmp_path, monkeypatch):
    """A strategy that leaves the universe is reported but its shadow rows are
    NOT removed unless --prune-orphans is set; live/backfill never touched."""
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2)])
    # First sync with TWO lanes.
    lane_a = _lane(strategy_id="LANE_A")
    lane_b = _lane(strategy_id="LANE_B")
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [lane_a, lane_b])
    runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")
    assert _source_counts(db).get("shadow") == 2

    # LANE_B leaves the universe.
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [lane_a])
    summary = runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")

    assert "LANE_B" in summary.orphans_found
    assert summary.orphans_pruned == 0, "default must not prune"
    assert _source_counts(db).get("shadow") == 2, "orphan rows remain when not pruning"
    assert _source_counts(db).get("backfill") == 1


def test_orphan_pruned_only_with_flag_and_only_shadow(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2)])
    lane_a = _lane(strategy_id="LANE_A")
    lane_b = _lane(strategy_id="LANE_B")
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [lane_a, lane_b])
    runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")

    monkeypatch.setattr(runner, "build_universe", lambda **kw: [lane_a])
    summary = runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml", prune_orphans=True)

    assert summary.orphans_pruned == 1
    assert _source_counts(db).get("shadow") == 1, "only LANE_A shadow rows remain"
    assert _source_counts(db).get("backfill") == 1, "backfill untouched by prune"


# ── G1: single-writer live-session guard ─────────────────────────────────


def test_sync_refuses_when_live_session_lock_held(tmp_path, monkeypatch):
    """G1: if a live bot holds an instance lock (live PID), the writing sync
    fails closed rather than contending the gold.db write path."""
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])

    # Simulate a live lock dir with one live-PID lock file.
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    (lock_dir / "bot_MNQ.lock").write_text("12345")
    import trading_app.live.instance_lock as il

    monkeypatch.setattr(il, "_LOCK_DIR", lock_dir)
    monkeypatch.setattr(il, "is_pid_alive", lambda pid: True)

    with pytest.raises(RuntimeError, match="LIVE SESSION ACTIVE"):
        runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")
    # Nothing written.
    assert _source_counts(db) == {"backfill": 1}


def test_sync_proceeds_when_lock_pid_dead(tmp_path, monkeypatch):
    """A stale lock file with a dead PID must NOT block the sync."""
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    (lock_dir / "bot_MNQ.lock").write_text("99999")
    import trading_app.live.instance_lock as il

    monkeypatch.setattr(il, "_LOCK_DIR", lock_dir)
    monkeypatch.setattr(il, "is_pid_alive", lambda pid: False)

    summary = runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")
    assert summary.trades_appended == 1, "dead-PID lock must not block"
