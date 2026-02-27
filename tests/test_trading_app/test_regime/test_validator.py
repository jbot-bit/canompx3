"""
Tests for trading_app.regime.validator -- regime strategy validation.
"""

import sys
import json
from pathlib import Path
from datetime import date

import pytest
import duckdb

from trading_app.regime.schema import init_regime_schema
from trading_app.regime.validator import run_regime_validation

@pytest.fixture
def validated_db(tmp_path):
    """Create a DB with regime_strategies pre-populated for validation."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    init_regime_schema(con=con)

    # Insert a strategy with good metrics (should pass)
    yearly = json.dumps({"2025": {"trades": 30, "wins": 15, "total_r": 3.0,
                                   "win_rate": 0.5, "avg_r": 0.1}})
    con.execute(
        """INSERT INTO regime_strategies
           (run_label, strategy_id, start_date, end_date,
            instrument, orb_label, orb_minutes, rr_target, confirm_bars,
            entry_model, filter_type, filter_params,
            sample_size, win_rate, avg_win_r, avg_loss_r,
            expectancy_r, sharpe_ratio, max_drawdown_r,
            median_risk_points, avg_risk_points, yearly_results)
           VALUES ('test', 'MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G4', '2025-01-01', '2025-12-31',
                   'MGC', 'CME_REOPEN', 5, 2.0, 2, 'E1', 'ORB_G4', '{}',
                   30, 0.5, 1.5, 1.0, 0.25, 0.15, 2.5, 3.0, 3.2, ?)""",
        [yearly],
    )

    # Insert a strategy with bad metrics (should fail)
    yearly_bad = json.dumps({"2025": {"trades": 5, "wins": 1, "total_r": -3.0,
                                       "win_rate": 0.2, "avg_r": -0.6}})
    con.execute(
        """INSERT INTO regime_strategies
           (run_label, strategy_id, start_date, end_date,
            instrument, orb_label, orb_minutes, rr_target, confirm_bars,
            entry_model, filter_type, filter_params,
            sample_size, win_rate, avg_win_r, avg_loss_r,
            expectancy_r, sharpe_ratio, max_drawdown_r,
            median_risk_points, avg_risk_points, yearly_results)
           VALUES ('test', 'MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER', '2025-01-01', '2025-12-31',
                   'MGC', 'CME_REOPEN', 5, 2.0, 2, 'E1', 'NO_FILTER', '{}',
                   5, 0.2, 1.0, 1.0, -0.6, -0.3, 5.0, 3.0, 3.2, ?)""",
        [yearly_bad],
    )

    con.commit()
    con.close()
    return db_path

class TestRegimeValidation:

    def test_validates_strategies(self, validated_db):
        """Validation processes strategies and returns counts."""
        passed, rejected = run_regime_validation(
            db_path=validated_db,
            instrument="MGC",
            run_label="test",
            min_sample=5,
            min_years_positive_pct=1.0,
        )
        # At least the good one should pass; the bad one (only 5 trades) may fail
        assert passed + rejected == 2

    def test_passed_goes_to_regime_validated(self, validated_db):
        """PASSED strategies are inserted into regime_validated."""
        run_regime_validation(
            db_path=validated_db,
            instrument="MGC",
            run_label="test",
            min_sample=5,
            min_years_positive_pct=1.0,
        )

        con = duckdb.connect(str(validated_db), read_only=True)
        try:
            rows = con.execute(
                "SELECT run_label, strategy_id, status FROM regime_validated"
            ).fetchall()
            for run_label, sid, status in rows:
                assert run_label == "test"
                assert status == "active"
        finally:
            con.close()

    def test_updates_validation_status(self, validated_db):
        """Validation updates validation_status on regime_strategies."""
        run_regime_validation(
            db_path=validated_db,
            instrument="MGC",
            run_label="test",
            min_sample=5,
            min_years_positive_pct=1.0,
        )

        con = duckdb.connect(str(validated_db), read_only=True)
        try:
            rows = con.execute(
                "SELECT strategy_id, validation_status FROM regime_strategies WHERE run_label='test'"
            ).fetchall()
            statuses = {sid: status for sid, status in rows}
            # All should have a status set
            for sid, status in statuses.items():
                assert status is not None
                assert status != ""
        finally:
            con.close()

    def test_dry_run_no_writes(self, validated_db):
        """Dry run does not modify the database."""
        passed, rejected = run_regime_validation(
            db_path=validated_db,
            instrument="MGC",
            run_label="test",
            min_sample=5,
            dry_run=True,
        )
        assert passed + rejected == 2

        con = duckdb.connect(str(validated_db), read_only=True)
        try:
            # Validation status should still be NULL
            rows = con.execute(
                "SELECT validation_status FROM regime_strategies WHERE run_label='test'"
            ).fetchall()
            for (status,) in rows:
                assert status is None or status == ""
            # regime_validated should be empty
            count = con.execute(
                "SELECT COUNT(*) FROM regime_validated WHERE run_label='test'"
            ).fetchone()[0]
            assert count == 0
        finally:
            con.close()

    def test_revalidation_skips_already_validated(self, validated_db):
        """Running validation twice only processes unvalidated strategies."""
        p1, r1 = run_regime_validation(
            db_path=validated_db, instrument="MGC",
            run_label="test", min_sample=5,
        )
        # Second run: all already validated -> 0 processed
        p2, r2 = run_regime_validation(
            db_path=validated_db, instrument="MGC",
            run_label="test", min_sample=5,
        )
        assert p2 == 0
        assert r2 == 0
