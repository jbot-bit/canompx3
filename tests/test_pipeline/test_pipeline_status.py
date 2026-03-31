"""Tests for rebuild_manifest table schema and pipeline staleness engine."""

from datetime import date
from unittest.mock import patch

import duckdb

from pipeline.init_db import init_db
from scripts.tools.pipeline_status import (
    _trading_days_between,
    build_step_list,
    get_resume_point,
    is_stale,
    preflight_check,
    read_last_manifest,
    run_rebuild,
    staleness_engine,
    write_manifest,
)


class TestRebuildManifest:
    def test_rebuild_manifest_table_exists(self, tmp_path):
        """rebuild_manifest table is created by init_db."""
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path), read_only=True)
        tables = [
            t[0]
            for t in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        ]
        con.close()

        assert "rebuild_manifest" in tables

    def test_rebuild_manifest_schema(self, tmp_path):
        """rebuild_manifest has all expected columns with correct types."""
        db_path = tmp_path / "test.db"
        init_db(db_path, force=False)

        con = duckdb.connect(str(db_path), read_only=True)
        cols = {
            r[0]: r[1]
            for r in con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'rebuild_manifest'"
            ).fetchall()
        }
        con.close()

        expected_cols = [
            "rebuild_id",
            "instrument",
            "started_at",
            "completed_at",
            "status",
            "failed_step",
            "steps_completed",
            "trigger",
        ]
        for col in expected_cols:
            assert col in cols, f"Missing column: {col}"

        # Verify key type constraints
        assert "TIMESTAMP" in cols["started_at"].upper()
        assert "TIMESTAMP" in cols["completed_at"].upper()
        assert cols["steps_completed"].upper().endswith("[]")


# ---------------------------------------------------------------------------
# Helper: create all tables needed for staleness_engine in an in-memory DB
# ---------------------------------------------------------------------------


def _create_test_db(tmp_path):
    """Create a test DB with all tables the staleness engine queries."""
    db_path = tmp_path / "test.db"
    # Use init_db for pipeline tables (bars_1m, bars_5m, daily_features, etc.)
    init_db(db_path, force=False)

    con = duckdb.connect(str(db_path))
    # Create trading_app tables with minimal schemas
    con.execute("""
        CREATE TABLE IF NOT EXISTS orb_outcomes (
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_label TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            rr_target DOUBLE NOT NULL,
            confirm_bars INTEGER NOT NULL,
            entry_model TEXT NOT NULL,
            outcome TEXT,
            pnl_r DOUBLE,
            PRIMARY KEY (symbol, trading_day, orb_label, orb_minutes, rr_target, confirm_bars, entry_model)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS experimental_strategies (
            strategy_id TEXT PRIMARY KEY,
            instrument TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS validated_setups (
            strategy_id TEXT PRIMARY KEY,
            instrument TEXT NOT NULL,
            promoted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active'
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS edge_families (
            family_hash TEXT PRIMARY KEY,
            instrument TEXT NOT NULL,
            head_strategy_id TEXT NOT NULL,
            member_count INTEGER NOT NULL,
            trade_day_count INTEGER NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    return db_path, con


def _insert_bar_1m(con, symbol, dt_str):
    """Insert a minimal bars_1m row."""
    con.execute(
        "INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume) "
        "VALUES (?::TIMESTAMPTZ, ?, ?, 100, 101, 99, 100, 1000)",
        [dt_str, symbol, symbol],
    )


def _insert_bar_5m(con, symbol, dt_str):
    """Insert a minimal bars_5m row."""
    con.execute(
        "INSERT INTO bars_5m (ts_utc, symbol, open, high, low, close, volume) "
        "VALUES (?::TIMESTAMPTZ, ?, 100, 101, 99, 100, 1000)",
        [dt_str, symbol],
    )


def _insert_daily_features(con, symbol, trading_day, orb_minutes):
    """Insert a minimal daily_features row."""
    con.execute(
        "INSERT INTO daily_features (trading_day, symbol, orb_minutes) VALUES (?, ?, ?)",
        [trading_day, symbol, orb_minutes],
    )


def _insert_orb_outcome(con, symbol, trading_day, apertures=(5, 15, 30)):
    """Insert minimal orb_outcomes rows for each aperture."""
    for ap in apertures:
        con.execute(
            "INSERT INTO orb_outcomes (trading_day, symbol, orb_label, orb_minutes, rr_target, "
            "confirm_bars, entry_model, outcome, pnl_r) VALUES (?, ?, 'CME_REOPEN', ?, 1.5, 1, 'E2', 'win', 1.0)",
            [trading_day, symbol, ap],
        )


# ---------------------------------------------------------------------------
# Staleness engine tests
# ---------------------------------------------------------------------------


class TestTradingDaysBetween:
    def test_weekend_gap(self):
        """Friday to Monday = 1 trading day (Monday itself)."""
        friday = date(2026, 3, 6)  # Friday
        monday = date(2026, 3, 9)  # Monday
        assert _trading_days_between(friday, monday) == 1

    def test_real_gap(self):
        """10 weekdays gap is correctly counted."""
        d1 = date(2026, 2, 20)  # Friday
        d2 = date(2026, 3, 6)  # Friday (2 weeks later)
        gap = _trading_days_between(d1, d2)
        assert gap == 10

    def test_same_day(self):
        """Same day returns 0."""
        d = date(2026, 3, 6)
        assert _trading_days_between(d, d) == 0

    def test_none_returns_zero(self):
        """None input returns 0."""
        assert _trading_days_between(None, date(2026, 3, 6)) == 0
        assert _trading_days_between(date(2026, 3, 6), None) == 0

    def test_reversed_returns_zero(self):
        """d1 > d2 returns 0."""
        assert _trading_days_between(date(2026, 3, 10), date(2026, 3, 6)) == 0


class TestIsStale:
    def test_none_table_date_is_stale(self):
        """Missing table data is stale when upstream exists."""
        assert is_stale(None, date(2026, 3, 6)) is True

    def test_none_reference_is_not_stale(self):
        """No upstream data means not stale."""
        assert is_stale(None, None) is False
        assert is_stale(date(2026, 3, 6), None) is False

    def test_same_date_not_stale(self):
        """Table at same date as upstream is fresh."""
        d = date(2026, 3, 6)
        assert is_stale(d, d) is False

    def test_weekend_gap_not_stale(self):
        """Friday-to-Monday is 1 trading day, within default threshold of 1."""
        assert is_stale(date(2026, 3, 6), date(2026, 3, 9)) is False


class TestStalenessEngine:
    def test_staleness_fresh(self, tmp_path):
        """All tables at same date -> stale_steps only flags things without data."""
        db_path, con = _create_test_db(tmp_path)
        sym = "MGC"
        day = "2026-03-06"

        _insert_bar_1m(con, sym, f"{day}T00:00:00+00:00")
        _insert_bar_5m(con, sym, f"{day}T00:00:00+00:00")
        for ap in [5, 15, 30]:
            _insert_daily_features(con, sym, day, ap)
        _insert_orb_outcome(con, sym, day)
        con.execute(
            "INSERT INTO experimental_strategies (strategy_id, instrument, created_at) "
            "VALUES ('s1', ?, ?::TIMESTAMPTZ)",
            [sym, f"{day}T00:00:00+00:00"],
        )
        con.execute(
            "INSERT INTO validated_setups (strategy_id, instrument, promoted_at, status) "
            "VALUES ('s1', ?, ?::TIMESTAMPTZ, 'active')",
            [sym, f"{day}T00:00:00+00:00"],
        )
        con.execute(
            "INSERT INTO edge_families (family_hash, instrument, head_strategy_id, member_count, trade_day_count, created_at) "
            "VALUES ('h1', ?, 's1', 1, 1, ?::TIMESTAMPTZ)",
            [sym, f"{day}T00:00:00+00:00"],
        )
        con.commit()

        status = staleness_engine(con, sym)
        con.close()

        # Everything at same date — nothing should be stale
        assert status["stale_steps"] == []
        assert status["bars_1m"] == date(2026, 3, 6)
        assert status["bars_5m"] == date(2026, 3, 6)
        assert status["daily_features_min"] == date(2026, 3, 6)

    def test_staleness_detects_gap(self, tmp_path):
        """orb_outcomes 14 days behind daily_features -> detected as stale."""
        db_path, con = _create_test_db(tmp_path)
        sym = "MGC"

        _insert_bar_1m(con, sym, "2026-03-06T00:00:00+00:00")
        _insert_bar_5m(con, sym, "2026-03-06T00:00:00+00:00")
        for ap in [5, 15, 30]:
            _insert_daily_features(con, sym, "2026-03-06", ap)
        # orb_outcomes 14 calendar days behind (10 trading days)
        _insert_orb_outcome(con, sym, "2026-02-20")
        con.commit()

        status = staleness_engine(con, sym)
        con.close()

        assert any(s.startswith("orb_outcomes_O") for s in status["stale_steps"])
        assert status["orb_outcomes"] == date(2026, 2, 20)
        assert status["daily_features_min"] == date(2026, 3, 6)

    def test_weekend_not_false_positive(self, tmp_path):
        """Friday bars_1m -> Monday bars_5m = not stale (0 trading day gap)."""
        db_path, con = _create_test_db(tmp_path)
        sym = "MGC"

        # bars_1m on Monday, bars_5m on previous Friday
        _insert_bar_1m(con, sym, "2026-03-09T00:00:00+00:00")  # Monday
        _insert_bar_5m(con, sym, "2026-03-06T00:00:00+00:00")  # Friday
        con.commit()

        status = staleness_engine(con, sym)
        con.close()

        # Friday to Monday = 0 trading days gap, should NOT be stale
        assert "bars_5m" not in status["stale_steps"]

    def test_staleness_real_gap(self, tmp_path):
        """10+ trading day gap IS stale."""
        db_path, con = _create_test_db(tmp_path)
        sym = "MGC"

        # bars_1m 2 weeks ahead of bars_5m (10 trading days)
        _insert_bar_1m(con, sym, "2026-03-06T00:00:00+00:00")
        _insert_bar_5m(con, sym, "2026-02-20T00:00:00+00:00")
        con.commit()

        status = staleness_engine(con, sym)
        con.close()

        assert "bars_5m" in status["stale_steps"]

    def test_staleness_last_rebuild_from_manifest(self, tmp_path):
        """staleness_engine returns last_rebuild date from COMPLETED manifest."""
        db_path, con = _create_test_db(tmp_path)
        sym = "MGC"

        _insert_bar_1m(con, sym, "2026-03-06T00:00:00+00:00")
        con.commit()

        # Write a COMPLETED manifest
        write_manifest(con, "rebuild-001", sym, "COMPLETED", trigger="CLI")

        status = staleness_engine(con, sym)
        con.close()

        # last_rebuild should NOT be None — the COMPLETED manifest exists
        assert status["last_rebuild"] is not None


# ---------------------------------------------------------------------------
# Pre-flight check tests
# ---------------------------------------------------------------------------


class TestPreflightCheck:
    def test_preflight_daily_features_missing(self, tmp_path):
        """No O15 daily_features data -> fails with helpful message."""
        db_path, con = _create_test_db(tmp_path)
        # Insert O5 data but NOT O15
        _insert_daily_features(con, "MGC", "2026-03-06", 5)
        con.commit()

        ok, msg = preflight_check(con, "MGC", "outcome_builder", orb_minutes=15)
        con.close()

        assert ok is False
        assert "PRE-FLIGHT FAIL" in msg
        assert "O15" in msg
        assert "build_daily_features" in msg

    def test_preflight_daily_features_present(self, tmp_path):
        """O15 daily_features data exists -> passes."""
        db_path, con = _create_test_db(tmp_path)
        _insert_daily_features(con, "MGC", "2026-03-06", 15)
        con.commit()

        ok, msg = preflight_check(con, "MGC", "outcome_builder", orb_minutes=15)
        con.close()

        assert ok is True
        assert "Pre-flight OK" in msg

    def test_preflight_no_rule(self, tmp_path):
        """Unknown step -> passes with 'no rule' message."""
        db_path, con = _create_test_db(tmp_path)
        ok, msg = preflight_check(con, "MGC", "nonexistent_step")
        con.close()

        assert ok is True
        assert "No pre-flight rule" in msg


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_write_and_read(self, tmp_path):
        """Write COMPLETED manifest, read back, verify all fields."""
        db_path, con = _create_test_db(tmp_path)

        rid = "test-rebuild-001"
        write_manifest(
            con,
            rebuild_id=rid,
            instrument="MGC",
            status="COMPLETED",
            steps_completed=["outcome_builder", "strategy_discovery"],
            trigger="MANUAL",
        )

        result = read_last_manifest(con, "MGC")
        con.close()

        assert result is not None
        assert result["rebuild_id"] == rid
        assert result["instrument"] == "MGC"
        assert result["status"] == "COMPLETED"
        assert result["completed_at"] is not None
        assert result["failed_step"] is None
        assert result["steps_completed"] == ["outcome_builder", "strategy_discovery"]
        assert result["trigger"] == "MANUAL"

    def test_manifest_resume_from_failed(self, tmp_path):
        """Write FAILED manifest with steps_completed, get_resume_point returns correct data."""
        db_path, con = _create_test_db(tmp_path)

        rid = "test-rebuild-fail-001"
        write_manifest(
            con,
            rebuild_id=rid,
            instrument="MNQ",
            status="FAILED",
            failed_step="strategy_validator",
            steps_completed=["outcome_builder", "strategy_discovery"],
            trigger="MANUAL",
        )

        result = get_resume_point(con, "MNQ")
        con.close()

        assert result is not None
        assert result["rebuild_id"] == rid
        assert result["failed_step"] == "strategy_validator"
        assert result["steps_completed"] == ["outcome_builder", "strategy_discovery"]

    def test_manifest_no_history(self, tmp_path):
        """read_last_manifest on instrument with no history -> None."""
        db_path, con = _create_test_db(tmp_path)
        result = read_last_manifest(con, "MGC")
        con.close()

        assert result is None


# ---------------------------------------------------------------------------
# Build step list tests
# ---------------------------------------------------------------------------


class TestBuildStepList:
    def test_build_step_list_full(self):
        """Full step list has all 9 steps (O15/O30 removed)."""
        steps = build_step_list("MGC")
        assert len(steps) == 9
        assert steps[0]["name"] == "outcome_builder_O5"
        assert "MGC" in steps[0]["cmd"]
        assert steps[-1]["name"] == "pinecone_sync"

    def test_build_step_list_resume(self):
        """Resume skips completed steps."""
        completed = ["outcome_builder_O5"]
        steps = build_step_list("MGC", resume_from=completed)
        assert len(steps) == 8
        assert steps[0]["name"] == "discovery_O5"

    def test_build_step_list_instrument_substitution(self):
        """Instrument name is substituted into all commands."""
        steps = build_step_list("MNQ")
        for step in steps:
            # Steps without {instrument} (retire_e3, repo_map, health_check, pinecone_sync, family_rr_locks) won't have it
            if "{instrument}" not in step["cmd"]:
                continue
            assert "MNQ" in step["cmd"]


# ---------------------------------------------------------------------------
# Rebuild dry run tests
# ---------------------------------------------------------------------------


class TestRebuildDryRun:
    def test_rebuild_dry_run(self, tmp_path, capsys):
        """Dry run prints steps without executing."""
        db_path, con = _create_test_db(tmp_path)
        result, con = run_rebuild(con, "MGC", dry_run=True)
        assert result is True
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "outcome_builder_O5" in captured.out
        con.close()

    def test_rebuild_dry_run_shows_all_steps(self, tmp_path, capsys):
        """Dry run lists all 9 steps (O15/O30 removed)."""
        db_path, con = _create_test_db(tmp_path)
        _, con = run_rebuild(con, "MGC", dry_run=True)
        captured = capsys.readouterr()
        assert "[9/9]" in captured.out
        assert "pinecone_sync" in captured.out
        con.close()


# ---------------------------------------------------------------------------
# Non-dry-run rebuild tests (mock subprocess)
# ---------------------------------------------------------------------------


class TestRebuildExecution:
    def test_rebuild_all_steps_pass(self, tmp_path, capsys):
        """All steps succeed -> COMPLETED manifest written."""
        from pipeline.dst import SESSION_CATALOG

        db_path, con = _create_test_db(tmp_path)
        sym = "MGC"
        # Seed prerequisites so preflight checks pass
        _insert_bar_1m(con, sym, "2026-03-06T00:00:00+00:00")
        _insert_bar_1m(con, sym, "2026-03-07T00:00:00+00:00")
        from pipeline.build_daily_features import ACTIVE_ORB_MINUTES, VALID_ORB_MINUTES

        for ap in VALID_ORB_MINUTES:
            _insert_daily_features(con, sym, "2026-03-06", ap)
        # Seed all session×aperture combos so A5 assertion passes (active apertures only)
        for session in SESSION_CATALOG:
            for ap in ACTIVE_ORB_MINUTES:
                con.execute(
                    "INSERT INTO orb_outcomes (trading_day, symbol, orb_label, orb_minutes, rr_target, "
                    "confirm_bars, entry_model, outcome, pnl_r) "
                    "VALUES ('2026-03-06', ?, ?, ?, 1.5, 1, 'E2', 'win', 1.0)",
                    [sym, session, ap],
                )
        con.execute("INSERT INTO experimental_strategies (strategy_id, instrument) VALUES ('s1', ?)", [sym])
        con.execute("INSERT INTO validated_setups (strategy_id, instrument, status) VALUES ('s1', ?, 'active')", [sym])
        con.execute(
            "INSERT INTO edge_families (family_hash, instrument, head_strategy_id, member_count, trade_day_count) "
            "VALUES ('h1', ?, 's1', 1, 1)",
            [sym],
        )
        con.commit()

        mock_result = type("Result", (), {"returncode": 0})()
        with patch("scripts.tools.pipeline_status.subprocess.run", return_value=mock_result):
            ok, con = run_rebuild(con, sym, db_path=str(db_path))

        assert ok is True
        manifest = read_last_manifest(con, sym)
        assert manifest is not None
        assert manifest["status"] == "COMPLETED"
        assert len(manifest["steps_completed"]) == 9
        con.close()

    def test_rebuild_step_fails_writes_manifest(self, tmp_path, capsys):
        """Step failure -> FAILED manifest with failed_step recorded."""
        _, con = _create_test_db(tmp_path)
        sym = "MGC"
        for ap in [5, 15, 30]:
            _insert_daily_features(con, sym, "2026-03-06", ap)
        con.commit()

        call_count = 0

        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on 2nd step (discovery_O5)
            rc = 1 if call_count == 2 else 0
            return type("Result", (), {"returncode": rc})()

        with patch("scripts.tools.pipeline_status.subprocess.run", side_effect=mock_run):
            ok, con = run_rebuild(con, sym)

        assert ok is False
        manifest = read_last_manifest(con, sym)
        assert manifest is not None
        assert manifest["status"] == "FAILED"
        assert manifest["failed_step"] == "discovery_O5"
        assert manifest["steps_completed"] == ["outcome_builder_O5"]
        con.close()

    def test_rebuild_timeout_writes_manifest(self, tmp_path, capsys):
        """Subprocess timeout -> FAILED manifest."""
        from subprocess import TimeoutExpired

        _, con = _create_test_db(tmp_path)
        sym = "MGC"
        for ap in [5, 15, 30]:
            _insert_daily_features(con, sym, "2026-03-06", ap)
        con.commit()

        def mock_run(*args, **kwargs):
            raise TimeoutExpired(cmd="test", timeout=3600)

        with patch("scripts.tools.pipeline_status.subprocess.run", side_effect=mock_run):
            ok, con = run_rebuild(con, sym)

        assert ok is False
        manifest = read_last_manifest(con, sym)
        assert manifest["status"] == "FAILED"
        assert manifest["failed_step"] == "outcome_builder_O5"
        captured = capsys.readouterr()
        assert "TIMED OUT" in captured.out
        con.close()


# ---------------------------------------------------------------------------
# Manifest started_at preservation
# ---------------------------------------------------------------------------


class TestManifestStartedAtPreservation:
    def test_update_preserves_started_at(self, tmp_path):
        """RUNNING -> COMPLETED update preserves the original started_at."""
        _, con = _create_test_db(tmp_path)

        rid = "test-preserve-001"
        # Write initial RUNNING record
        write_manifest(con, rid, "MGC", "RUNNING", trigger="CLI")
        row1 = read_last_manifest(con, "MGC")
        original_started = row1["started_at"]

        # Transition to COMPLETED
        write_manifest(con, rid, "MGC", "COMPLETED", steps_completed=["step1"], trigger="CLI")
        row2 = read_last_manifest(con, "MGC")

        assert row2["started_at"] == original_started
        assert row2["status"] == "COMPLETED"
        assert row2["completed_at"] is not None
        con.close()
