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
        # F1: default first_seen == FWD so existing tests stay a no-op
        # (max(FWD, FWD) == FWD). Late-joiner tests pass a later first_seen.
        first_seen=FWD,
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


def test_oos_context_report_not_clamped_by_first_seen(tmp_path, monkeypatch):
    """REGRESSION: the OOS context report uses the OOS start (2026-01-01), NOT
    the per-lane first_seen. A late-joiner lane (first_seen far in the future)
    must STILL have its full OOS history counted — first_seen is a forward-
    monitoring concept and must not silently clamp the backward OOS analysis.

    Guards the implementation choice to make _shadow_rows_for_lane take an
    EXPLICIT boundary (not derive max(.,first_seen) internally), so the sync and
    OOS paths keep distinct time semantics."""
    db = _make_db(tmp_path, [datetime.date(2026, 1, 5), datetime.date(2026, 6, 2)])
    # Lane joined the universe only on 2026-06-20 (future first_seen).
    late = _lane(strategy_id="LATE_JOINER", first_seen=datetime.date(2026, 6, 20))
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [late])

    results = runner.oos_context_report(db_path=db, as_of_date=FWD)

    assert _source_counts(db) == {"backfill": 1}, "OOS report still writes nothing"
    # BOTH the Jan row AND the Jun row are counted — first_seen does NOT clamp the
    # OOS boundary (would be only 0 rows if it were wrongly clamped to 2026-06-20).
    assert sum(r.appended for r in results) == 2, (
        "OOS report must span from OOS_CONTEXT_START, not be clamped by first_seen"
    )


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


# ── F1: per-lane first_seen boundary ─────────────────────────────────────


def test_lane_boundary_uses_max_of_floor_and_first_seen():
    """_lane_boundary = max(global forward_start floor, lane.first_seen)."""
    floor = datetime.date(2026, 6, 1)
    # Late-joiner: first_seen AFTER the floor -> boundary rises to first_seen.
    late = _lane(first_seen=datetime.date(2026, 6, 20))
    assert runner._lane_boundary(late, floor) == datetime.date(2026, 6, 20)
    # Existing lane: first_seen == floor -> boundary == floor (no-op).
    existing = _lane(first_seen=floor)
    assert runner._lane_boundary(existing, floor) == floor
    # Defensive: a first_seen EARLIER than the floor can never lower the boundary.
    early = _lane(first_seen=datetime.date(2026, 5, 1))
    assert runner._lane_boundary(early, floor) == floor


def test_late_joiner_writes_nothing_before_its_first_seen(tmp_path, monkeypatch):
    """A lane that joins later monitors from ITS first_seen, not the global floor.

    Outcomes exist on 06-02 (>= floor, < first_seen) and 06-25 (>= first_seen).
    The late-joiner must record ONLY the 06-25 row — the 06-02 row is before the
    lane was eligible, so it is NOT its forward-monitoring evidence.
    """
    floor = datetime.date(2026, 6, 1)
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2), datetime.date(2026, 6, 25)])
    late = _lane(strategy_id="LATE_JOINER", first_seen=datetime.date(2026, 6, 20))
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [late])

    runner.sync_shadow(db_path=db, as_of_date=floor, universe_yaml=tmp_path / "u.yaml")

    with duckdb.connect(str(db), read_only=True) as con:
        days = [
            r[0]
            for r in con.execute(
                "SELECT trading_day FROM paper_trades WHERE execution_source='shadow' ORDER BY 1"
            ).fetchall()
        ]
    assert days == [datetime.date(2026, 6, 25)], (
        f"late-joiner must record only >= its first_seen 2026-06-20, got {days}"
    )


def test_existing_lane_row_set_identical_before_and_after_first_seen(tmp_path, monkeypatch):
    """No-op proof: a lane whose first_seen == forward_start records the SAME
    rows it would have under the pre-F1 global-only boundary."""
    floor = datetime.date(2026, 6, 1)
    days = [datetime.date(2026, 6, 2), datetime.date(2026, 6, 3), datetime.date(2026, 6, 5)]
    db = _make_db(tmp_path, days)
    existing = _lane(strategy_id="EXISTING", first_seen=floor)
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [existing])

    runner.sync_shadow(db_path=db, as_of_date=floor, universe_yaml=tmp_path / "u.yaml")

    with duckdb.connect(str(db), read_only=True) as con:
        got = [
            r[0]
            for r in con.execute(
                "SELECT trading_day FROM paper_trades WHERE execution_source='shadow' ORDER BY 1"
            ).fetchall()
        ]
    assert got == days, "first_seen==floor must record every forward day (no-op vs pre-F1)"


def test_per_lane_delete_window_matches_insert_window(tmp_path, monkeypatch):
    """The idempotent per-lane DELETE clears from the SAME per-lane boundary the
    INSERT repopulates — a re-sync of a late-joiner does not double-count and does
    not strand rows."""
    floor = datetime.date(2026, 6, 1)
    db = _make_db(tmp_path, [datetime.date(2026, 6, 25), datetime.date(2026, 6, 26)])
    late = _lane(strategy_id="LATE_JOINER", first_seen=datetime.date(2026, 6, 20))
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [late])

    runner.sync_shadow(db_path=db, as_of_date=floor, universe_yaml=tmp_path / "u.yaml")
    runner.sync_shadow(db_path=db, as_of_date=floor, universe_yaml=tmp_path / "u.yaml")

    assert _source_counts(db).get("shadow") == 2, "re-sync of a late-joiner must not double-count"


# ── F3: write atomicity ──────────────────────────────────────────────────


def test_mid_loop_failure_rolls_back_all_shadow_rows(tmp_path, monkeypatch):
    """A failure partway through the per-lane loop rolls the WHOLE sync back —
    zero shadow rows committed (single-transaction atomicity), so no second
    writer observes a half-written universe. backfill rows are untouched."""
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2)])
    lane_ok = _lane(strategy_id="LANE_OK")
    lane_boom = _lane(strategy_id="LANE_BOOM")
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [lane_ok, lane_boom])

    # Make the SECOND lane explode inside the loop (after LANE_OK wrote rows).
    real = runner._shadow_rows_for_lane

    def explode(con, lane, boundary):
        if lane.strategy_id == "LANE_BOOM":
            raise RuntimeError("injected mid-loop failure")
        return real(con, lane, boundary)

    monkeypatch.setattr(runner, "_shadow_rows_for_lane", explode)

    with pytest.raises(RuntimeError, match="injected mid-loop failure"):
        runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")

    after = _source_counts(db)
    assert after.get("shadow") is None, "LANE_OK rows must be rolled back, not partially committed"
    assert after.get("backfill") == 1, "backfill untouched by the rolled-back sync"


def test_live_lock_reasserted_before_write_open(tmp_path, monkeypatch):
    """F3: the live-session lock is re-checked immediately before the write
    connection opens. A lock that appears AFTER the first (boundary-persist) check
    but BEFORE write-open is still caught — nothing is written."""
    db = _make_db(tmp_path, [datetime.date(2026, 6, 2)])
    monkeypatch.setattr(runner, "build_universe", lambda **kw: [_lane()])

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    import trading_app.live.instance_lock as il

    monkeypatch.setattr(il, "_LOCK_DIR", lock_dir)
    monkeypatch.setattr(il, "is_pid_alive", lambda pid: True)

    # The lock does NOT exist on the first assert_no_live_session() call, but
    # appears just before the second (pre-write-open) one. We emulate that by
    # creating the lock on the first call and letting the second call see it.
    calls = {"n": 0}
    real_assert = runner.assert_no_live_session

    def racy_assert():
        calls["n"] += 1
        if calls["n"] == 1:
            # First check passes (no lock yet); a live bot starts right after.
            (lock_dir / "bot_MNQ.lock").write_text("12345")
            return
        real_assert()  # second check sees the lock -> raises

    monkeypatch.setattr(runner, "assert_no_live_session", racy_assert)

    with pytest.raises(RuntimeError, match="LIVE SESSION ACTIVE"):
        runner.sync_shadow(db_path=db, as_of_date=FWD, universe_yaml=tmp_path / "u.yaml")
    assert _source_counts(db) == {"backfill": 1}, "second-check race must write nothing"
