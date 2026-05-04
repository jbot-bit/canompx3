"""Tests for backfill_validated_trade_windows canonical refresh script."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from scripts.migrations import backfill_validated_trade_windows as migration
from trading_app.validation_provenance import StrategyTradeWindow


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE validated_setups (
            strategy_id VARCHAR PRIMARY KEY,
            instrument VARCHAR,
            orb_label VARCHAR,
            orb_minutes INTEGER,
            entry_model VARCHAR,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            filter_type VARCHAR,
            status VARCHAR,
            promotion_provenance VARCHAR,
            first_trade_day DATE,
            last_trade_day DATE,
            trade_day_count INTEGER
        )
        """
    )
    con.execute(
        """
        INSERT INTO validated_setups VALUES
            ('DRIFT_A', 'MNQ', 'EUROPE_FLOW', 5, 'E2', 1.0, 1, 'CROSS_SGP_MOMENTUM',
             'active', 'VALIDATOR_NATIVE', DATE '2019-05-08', DATE '2026-04-10', 1020),
            ('DRIFT_B', 'MNQ', 'EUROPE_FLOW', 5, 'E2', 1.5, 1, 'CROSS_SGP_MOMENTUM',
             'active', 'VALIDATOR_NATIVE', DATE '2019-05-08', DATE '2026-04-10', 1020),
            ('CLEAN_C', 'MES', 'NYSE_OPEN', 15, 'E2', 2.0, 1, 'ORB_G5',
             'active', 'VALIDATOR_NATIVE', DATE '2020-01-02', DATE '2026-04-14', 500),
            ('INACTIVE_D', 'MGC', 'COMEX_SETTLE', 5, 'E2', 1.0, 1, 'OVNRNG_100',
             'retired', 'VALIDATOR_NATIVE', DATE '2020-01-02', DATE '2023-04-10', 300),
            ('LEGACY_E', 'MNQ', 'TOKYO_OPEN', 5, 'E2', 1.0, 1, 'ORB_G5',
             'active', 'LEGACY', DATE '2020-01-02', DATE '2022-04-10', 100)
        """
    )
    con.close()
    return db


class _StubResolver:
    """Return canonical windows keyed by (sid implied by strategy signature)."""

    # Canonical recompute: DRIFT_A/B should move to 2026-04-14 N=1021, CLEAN_C stays,
    # INACTIVE/LEGACY should never be queried (filtered out by SQL WHERE).
    _canonical = {
        ("MNQ", "EUROPE_FLOW", 5, "E2", 1.0, 1, "CROSS_SGP_MOMENTUM"): StrategyTradeWindow(
            first_trade_day=date(2019, 5, 8),
            last_trade_day=date(2026, 4, 14),
            trade_day_count=1021,
        ),
        ("MNQ", "EUROPE_FLOW", 5, "E2", 1.5, 1, "CROSS_SGP_MOMENTUM"): StrategyTradeWindow(
            first_trade_day=date(2019, 5, 8),
            last_trade_day=date(2026, 4, 14),
            trade_day_count=1021,
        ),
        ("MES", "NYSE_OPEN", 15, "E2", 2.0, 1, "ORB_G5"): StrategyTradeWindow(
            first_trade_day=date(2020, 1, 2),
            last_trade_day=date(2026, 4, 14),
            trade_day_count=500,
        ),
    }

    def __init__(self, con):
        self.con = con
        self.calls: list[tuple] = []

    def resolve(
        self,
        *,
        instrument,
        orb_label,
        orb_minutes,
        entry_model,
        rr_target,
        confirm_bars,
        filter_type,
    ):
        key = (instrument, orb_label, orb_minutes, entry_model, rr_target, confirm_bars, filter_type)
        self.calls.append(key)
        return self._canonical[key]


@pytest.fixture
def stub_resolver(monkeypatch):
    monkeypatch.setattr(migration, "StrategyTradeWindowResolver", _StubResolver)
    return _StubResolver


def _read_row(db_path: Path, sid: str) -> tuple:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute(
            "SELECT first_trade_day, last_trade_day, trade_day_count FROM validated_setups WHERE strategy_id = ?",
            [sid],
        ).fetchone()
        assert row is not None, f"row not found: {sid}"
        return row
    finally:
        con.close()


@pytest.mark.usefixtures("stub_resolver")
class TestBackfillValidatedTradeWindows:
    def test_dry_run_reports_drift_without_writing(self, tmp_path):
        db = _make_db(tmp_path)
        report = migration.refresh_validated_trade_windows(db_path=db, dry_run=True)
        assert report.inspected == 3  # 2 drifted + 1 clean; excludes retired + LEGACY
        assert report.drifted == 2
        assert report.updated == 0
        # DB untouched
        assert _read_row(db, "DRIFT_A") == (date(2019, 5, 8), date(2026, 4, 10), 1020)

    def test_live_run_updates_only_drifted_rows(self, tmp_path):
        db = _make_db(tmp_path)
        report = migration.refresh_validated_trade_windows(db_path=db, dry_run=False)
        assert report.inspected == 3
        assert report.drifted == 2
        assert report.updated == 2
        assert _read_row(db, "DRIFT_A") == (date(2019, 5, 8), date(2026, 4, 14), 1021)
        assert _read_row(db, "DRIFT_B") == (date(2019, 5, 8), date(2026, 4, 14), 1021)
        # Clean row untouched
        assert _read_row(db, "CLEAN_C") == (date(2020, 1, 2), date(2026, 4, 14), 500)

    def test_rerun_is_noop(self, tmp_path):
        db = _make_db(tmp_path)
        migration.refresh_validated_trade_windows(db_path=db, dry_run=False)
        report = migration.refresh_validated_trade_windows(db_path=db, dry_run=False)
        assert report.drifted == 0
        assert report.updated == 0

    def test_retired_rows_excluded(self, tmp_path):
        db = _make_db(tmp_path)
        migration.refresh_validated_trade_windows(db_path=db, dry_run=False)
        # INACTIVE_D is retired — must remain untouched with its original stale window
        assert _read_row(db, "INACTIVE_D") == (date(2020, 1, 2), date(2023, 4, 10), 300)

    def test_legacy_rows_excluded(self, tmp_path):
        db = _make_db(tmp_path)
        migration.refresh_validated_trade_windows(db_path=db, dry_run=False)
        # LEGACY_E has promotion_provenance=LEGACY — excluded
        assert _read_row(db, "LEGACY_E") == (date(2020, 1, 2), date(2022, 4, 10), 100)

    def test_strategy_id_scopes_to_single_row(self, tmp_path):
        db = _make_db(tmp_path)
        report = migration.refresh_validated_trade_windows(db_path=db, strategy_id="DRIFT_A", dry_run=False)
        assert report.inspected == 1
        assert report.drifted == 1
        assert report.updated == 1
        # DRIFT_B should remain stale (out of scope)
        assert _read_row(db, "DRIFT_B") == (date(2019, 5, 8), date(2026, 4, 10), 1020)

    def test_main_exits_zero(self, tmp_path, monkeypatch, capsys):
        db = _make_db(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "backfill_validated_trade_windows.py",
                "--db-path",
                str(db),
                "--dry-run",
            ],
        )
        exit_code = migration.main()
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "[DRY-RUN]" in captured.out
        assert "inspected=3 drifted=2 updated=0" in captured.out
