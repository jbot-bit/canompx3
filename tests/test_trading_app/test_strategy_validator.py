"""
Tests for trading_app.strategy_validator module.
"""

import sys
import json
from pathlib import Path
from datetime import date

import pytest
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_app.strategy_validator import validate_strategy, run_validation
from pipeline.cost_model import get_cost_spec


def _cost():
    return get_cost_spec("MGC")


def _make_row(**overrides):
    """Build a strategy row dict with sane defaults."""
    base = {
        "strategy_id": "MGC_0900_E1_RR2.0_CB1_NO_FILTER",
        "instrument": "MGC",
        "orb_label": "0900",
        "orb_minutes": 5,
        "rr_target": 2.0,
        "confirm_bars": 1,
        "entry_model": "E1",
        "filter_type": "NO_FILTER",
        "filter_params": "{}",
        "sample_size": 150,
        "win_rate": 0.55,
        "avg_win_r": 1.8,
        "avg_loss_r": 1.0,
        "expectancy_r": 0.54,
        "sharpe_ratio": 0.3,
        "max_drawdown_r": 5.0,
        "median_risk_points": 10.0,
        "avg_risk_points": 10.5,
        "yearly_results": json.dumps({
            "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "win_rate": 0.56, "avg_r": 0.2},
            "2023": {"trades": 50, "wins": 27, "total_r": 8.0, "win_rate": 0.54, "avg_r": 0.16},
            "2024": {"trades": 50, "wins": 28, "total_r": 9.0, "win_rate": 0.56, "avg_r": 0.18},
        }),
    }
    base.update(overrides)
    return base


class TestValidateStrategy:
    """Tests for the 6-phase validation function."""

    def test_all_phases_pass(self):
        """Strategy that passes all phases."""
        status, notes = validate_strategy(_make_row(), _cost())
        assert status == "PASSED"

    def test_reject_low_sample(self):
        """Sample < 30 -> REJECT."""
        status, notes = validate_strategy(
            _make_row(sample_size=20), _cost()
        )
        assert status == "REJECTED"
        assert "Phase 1" in notes
        assert "20" in notes

    def test_warn_medium_sample(self):
        """30 <= sample < 100 -> PASS with warning."""
        status, notes = validate_strategy(
            _make_row(sample_size=50), _cost()
        )
        assert status == "PASSED"
        assert "WARN" in notes

    def test_reject_negative_expectancy(self):
        """ExpR <= 0 -> REJECT."""
        status, notes = validate_strategy(
            _make_row(expectancy_r=-0.1), _cost()
        )
        assert status == "REJECTED"
        assert "Phase 2" in notes

    def test_reject_zero_expectancy(self):
        """ExpR == 0 -> REJECT."""
        status, notes = validate_strategy(
            _make_row(expectancy_r=0.0), _cost()
        )
        assert status == "REJECTED"
        assert "Phase 2" in notes

    def test_reject_one_year_negative(self):
        """One year with avg_r <= 0 -> REJECT."""
        yearly = json.dumps({
            "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
            "2023": {"trades": 50, "wins": 20, "total_r": -5.0, "avg_r": -0.1},
            "2024": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
        })
        status, notes = validate_strategy(
            _make_row(yearly_results=yearly), _cost()
        )
        assert status == "REJECTED"
        assert "Phase 3" in notes
        assert "2023" in notes

    def test_reject_no_yearly_data(self):
        """No yearly data -> REJECT."""
        status, notes = validate_strategy(
            _make_row(yearly_results="{}"), _cost()
        )
        assert status == "REJECTED"
        assert "Phase 3" in notes

    def test_exclude_years_skips_bad_year(self):
        """Excluding a bad year lets strategy pass Phase 3."""
        yearly = json.dumps({
            "2021": {"trades": 30, "wins": 10, "total_r": -5.0, "avg_r": -0.17},
            "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
            "2023": {"trades": 50, "wins": 27, "total_r": 8.0, "avg_r": 0.16},
        })
        # Fails without exclusion
        status, _ = validate_strategy(
            _make_row(yearly_results=yearly), _cost()
        )
        assert status == "REJECTED"

        # Passes with 2021 excluded
        status, _ = validate_strategy(
            _make_row(yearly_results=yearly), _cost(), exclude_years={2021}
        )
        assert status == "PASSED"

    def test_min_years_positive_pct(self):
        """80% threshold allows 1 bad year out of 5."""
        yearly = json.dumps({
            "2021": {"trades": 30, "wins": 10, "total_r": -5.0, "avg_r": -0.17},
            "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
            "2023": {"trades": 50, "wins": 27, "total_r": 8.0, "avg_r": 0.16},
            "2024": {"trades": 50, "wins": 28, "total_r": 9.0, "avg_r": 0.18},
            "2025": {"trades": 50, "wins": 30, "total_r": 12.0, "avg_r": 0.24},
        })
        # Fails at 100%
        status, _ = validate_strategy(
            _make_row(yearly_results=yearly), _cost(), min_years_positive_pct=1.0
        )
        assert status == "REJECTED"

        # Passes at 80% (4/5 = 80%)
        status, _ = validate_strategy(
            _make_row(yearly_results=yearly), _cost(), min_years_positive_pct=0.8
        )
        assert status == "PASSED"

    def test_exclude_all_years_rejects(self):
        """Excluding all years -> REJECT."""
        yearly = json.dumps({
            "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "avg_r": 0.2},
        })
        status, notes = validate_strategy(
            _make_row(yearly_results=yearly), _cost(), exclude_years={2022}
        )
        assert status == "REJECTED"
        assert "Phase 3" in notes

    def test_reject_stress_test(self):
        """Marginal ExpR that fails stress test -> REJECT."""
        status, notes = validate_strategy(
            _make_row(expectancy_r=0.01), _cost()
        )
        assert status == "REJECTED"
        assert "Phase 4" in notes

    def test_pass_stress_test_high_exp(self):
        """High ExpR survives stress test."""
        status, notes = validate_strategy(
            _make_row(expectancy_r=0.8), _cost()
        )
        assert status == "PASSED"

    def test_reject_low_sharpe(self):
        """Sharpe below threshold -> REJECT when threshold set."""
        status, notes = validate_strategy(
            _make_row(sharpe_ratio=0.1), _cost(), min_sharpe=0.5
        )
        assert status == "REJECTED"
        assert "Phase 5" in notes

    def test_reject_high_drawdown(self):
        """Drawdown above threshold -> REJECT when threshold set."""
        status, notes = validate_strategy(
            _make_row(max_drawdown_r=15.0), _cost(), max_drawdown=10.0
        )
        assert status == "REJECTED"
        assert "Phase 6" in notes

    def test_stress_test_uses_outcome_risk(self):
        """Stress test uses median_risk_points when available."""
        # With large risk, stress delta is small -> passes
        status, _ = validate_strategy(
            _make_row(expectancy_r=0.10, median_risk_points=20.0), _cost()
        )
        assert status == "PASSED"

        # With tiny risk, stress delta is large -> rejects
        status, notes = validate_strategy(
            _make_row(expectancy_r=0.10, median_risk_points=0.5), _cost()
        )
        assert status == "REJECTED"
        assert "Phase 4" in notes

    def test_stress_test_falls_back_to_avg_risk(self):
        """Stress test uses avg_risk_points when median is None."""
        status, _ = validate_strategy(
            _make_row(expectancy_r=0.10, median_risk_points=None, avg_risk_points=20.0), _cost()
        )
        assert status == "PASSED"

    def test_stress_test_falls_back_to_tick_floor(self):
        """Stress test uses tick-based floor when both risk stats are None."""
        # tick floor = 10 * 0.10 = 1.0 point, risk $ = 10.0
        # stress delta = (8.40 * 0.5) / 10.0 = 0.42R -> needs ExpR > 0.42
        status, _ = validate_strategy(
            _make_row(expectancy_r=0.50, median_risk_points=None, avg_risk_points=None), _cost()
        )
        assert status == "PASSED"

        status, notes = validate_strategy(
            _make_row(expectancy_r=0.10, median_risk_points=None, avg_risk_points=None), _cost()
        )
        assert status == "REJECTED"
        assert "Phase 4" in notes

    def test_optional_phases_skipped_by_default(self):
        """Without min_sharpe/max_drawdown, phases 5/6 are skipped."""
        status, notes = validate_strategy(
            _make_row(sharpe_ratio=0.01, max_drawdown_r=50.0), _cost()
        )
        assert status == "PASSED"

    def test_validation_notes_contain_reason(self):
        """Rejection notes explain which phase failed."""
        status, notes = validate_strategy(
            _make_row(sample_size=10), _cost()
        )
        assert "Phase 1" in notes
        assert "10" in notes


class TestRunValidation:
    """Integration tests with temp DB."""

    def _setup_db(self, tmp_path, strategies):
        """Create temp DB with schema + strategies."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)
        con.close()

        init_trading_app_schema = __import__(
            "trading_app.db_manager", fromlist=["init_trading_app_schema"]
        ).init_trading_app_schema
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))
        for s in strategies:
            cols = list(s.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_str = ", ".join(cols)
            con.execute(
                f"INSERT INTO experimental_strategies ({col_str}) VALUES ({placeholders})",
                list(s.values()),
            )
        con.commit()
        con.close()
        return db_path

    def test_promotes_passing_strategy(self, tmp_path):
        """Passing strategy appears in validated_setups."""
        db_path = self._setup_db(tmp_path, [_make_row()])

        passed, rejected = run_validation(db_path=db_path, instrument="MGC",
                                          enable_walkforward=False)
        assert passed == 1
        assert rejected == 0

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
        con.close()
        assert count == 1

    def test_rejected_not_in_validated(self, tmp_path):
        """Rejected strategy does NOT appear in validated_setups."""
        db_path = self._setup_db(tmp_path, [_make_row(sample_size=10)])

        passed, rejected = run_validation(db_path=db_path, instrument="MGC")
        assert passed == 0
        assert rejected == 1

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
        con.close()
        assert count == 0

    def test_already_validated_not_reprocessed(self, tmp_path):
        """Strategy with existing validation_status is skipped."""
        row = _make_row(validation_status="PASSED", validation_notes="Already done")
        db_path = self._setup_db(tmp_path, [row])

        passed, rejected = run_validation(db_path=db_path, instrument="MGC")
        assert passed == 0
        assert rejected == 0

    def test_entry_model_in_validated_setups(self, tmp_path):
        """entry_model column is populated in validated_setups."""
        db_path = self._setup_db(tmp_path, [_make_row()])
        run_validation(db_path=db_path, instrument="MGC",
                       enable_walkforward=False)

        con = duckdb.connect(str(db_path), read_only=True)
        em = con.execute("SELECT entry_model FROM validated_setups").fetchone()[0]
        con.close()
        assert em == "E1"


class TestCLI:
    def test_help(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "trading_app/strategy_validator.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert r.returncode == 0
        assert "instrument" in r.stdout
