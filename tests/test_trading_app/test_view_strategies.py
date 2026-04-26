"""
Tests for trading_app.view_strategies module.
"""

import math
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from trading_app.view_strategies import (
    _fmt_signed,
    _safe_float,
    fetch_families,
    fetch_summary,
    fetch_total_count,
    fetch_unique_trade_count,
    format_families,
    format_summary,
    format_table,
)

# ============================================================================
# _safe_float tests
# ============================================================================


class TestSafeFloat:
    def test_none(self):
        assert _safe_float(None) is None

    def test_nan(self):
        assert _safe_float(float("nan")) is None

    def test_valid(self):
        assert _safe_float(1.5) == 1.5

    def test_zero(self):
        assert _safe_float(0.0) == 0.0

    def test_negative(self):
        assert _safe_float(-2.3) == -2.3


# ============================================================================
# _fmt_signed tests
# ============================================================================


class TestFmtSigned:
    def test_positive(self):
        assert _fmt_signed(1.234) == "+1.23"

    def test_negative(self):
        assert _fmt_signed(-0.5) == "-0.50"

    def test_zero(self):
        assert _fmt_signed(0.0) == "+0.00"

    def test_none(self):
        assert _fmt_signed(None) == "N/A"

    def test_nan(self):
        assert _fmt_signed(float("nan")) == "N/A"


# ============================================================================
# format_table tests
# ============================================================================


class TestFormatTable:
    def test_empty(self):
        result = format_table(pd.DataFrame())
        assert "No strategies found" in result

    def test_with_nan_sharpe_ann(self):
        """NaN values in sharpe_ann/trades_per_year must display as N/A, not nan."""
        df = pd.DataFrame(
            [
                {
                    "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4",
                    "orb_label": "CME_REOPEN",
                    "entry_model": "E1",
                    "confirm_bars": 1,
                    "rr_target": 2.0,
                    "filter_type": "ORB_G4",
                    "sample_size": 100,
                    "win_rate": 0.45,
                    "expectancy_r": 0.15,
                    "sharpe_ratio": 0.2,
                    "sharpe_ann": float("nan"),
                    "trades_per_year": float("nan"),
                    "max_drawdown_r": 3.5,
                    "years_tested": 5,
                    "stress_test_passed": True,
                }
            ]
        )
        result = format_table(df)
        assert "nan" not in result.lower() or "shann" in result.lower()
        assert "N/A" in result

    def test_with_none_sharpe_ann(self):
        """None values display as N/A."""
        df = pd.DataFrame(
            [
                {
                    "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4",
                    "orb_label": "CME_REOPEN",
                    "entry_model": "E1",
                    "confirm_bars": 1,
                    "rr_target": 2.0,
                    "filter_type": "ORB_G4",
                    "sample_size": 100,
                    "win_rate": 0.45,
                    "expectancy_r": 0.15,
                    "sharpe_ratio": 0.2,
                    "sharpe_ann": None,
                    "trades_per_year": None,
                    "max_drawdown_r": 3.5,
                    "years_tested": 5,
                    "stress_test_passed": True,
                }
            ]
        )
        result = format_table(df)
        assert "N/A" in result

    def test_with_valid_sharpe_ann(self):
        """Valid sharpe_ann displays with sign."""
        df = pd.DataFrame(
            [
                {
                    "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4",
                    "orb_label": "CME_REOPEN",
                    "entry_model": "E1",
                    "confirm_bars": 1,
                    "rr_target": 2.0,
                    "filter_type": "ORB_G4",
                    "sample_size": 100,
                    "win_rate": 0.45,
                    "expectancy_r": 0.15,
                    "sharpe_ratio": 0.2,
                    "sharpe_ann": 0.85,
                    "trades_per_year": 50.0,
                    "max_drawdown_r": 3.5,
                    "years_tested": 5,
                    "stress_test_passed": True,
                }
            ]
        )
        result = format_table(df)
        assert "+0.85" in result
        assert "50" in result


# ============================================================================
# format_families tests
# ============================================================================


class TestFormatFamilies:
    def test_empty(self):
        result = format_families(pd.DataFrame())
        assert "No families found" in result

    def test_nan_best_shann(self):
        """NaN in best_shann must display as N/A."""
        df = pd.DataFrame(
            [
                {
                    "orb_label": "CME_REOPEN",
                    "entry_model": "E1",
                    "rr_target": 2.0,
                    "confirm_bars": 1,
                    "filter_variants": 3,
                    "best_filter": "ORB_G4",
                    "best_shann": float("nan"),
                    "max_n": 100,
                    "best_wr_pct": 45.0,
                    "best_expr": 0.15,
                }
            ]
        )
        result = format_families(df)
        assert "nan" not in result.lower() or "shann" in result.lower()
        assert "N/A" in result


# ============================================================================
# format_summary tests
# ============================================================================


class TestFormatSummary:
    def test_empty(self):
        result = format_summary(pd.DataFrame())
        assert "No strategies found" in result

    def test_nan_shann(self):
        """NaN in avg_shann/best_shann must display as N/A."""
        df = pd.DataFrame(
            [
                {
                    "orb_label": "CME_REOPEN",
                    "count": 50,
                    "unique_trades": 15,
                    "avg_expr": 0.15,
                    "best_expr": 0.35,
                    "avg_shann": float("nan"),
                    "best_shann": float("nan"),
                    "avg_wr_pct": 42.0,
                }
            ]
        )
        result = format_summary(df)
        assert "N/A" in result
        assert "nan" not in result.replace("ShANN", "")


# ============================================================================
# CLI test
# ============================================================================


class TestCLI:
    def test_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["view_strategies", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from trading_app.view_strategies import main

            main()
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--family" in out
        assert "--db" in out
        assert "--sort" in out
        assert "sharpe_ann" in out


def _setup_validated_shelf_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "view_strategies.db"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id TEXT,
            orb_label TEXT,
            entry_model TEXT,
            confirm_bars INTEGER,
            rr_target DOUBLE,
            filter_type TEXT,
            sample_size INTEGER,
            win_rate DOUBLE,
            expectancy_r DOUBLE,
            sharpe_ratio DOUBLE,
            sharpe_ann DOUBLE,
            trades_per_year DOUBLE,
            max_drawdown_r DOUBLE,
            years_tested INTEGER,
            stress_test_passed BOOLEAN,
            status TEXT,
            deployment_scope TEXT
        )
    """)
    con.execute("""
        INSERT INTO validated_setups VALUES
        (
            'MNQ_US_DATA_830_E2_RR2.0_CB1_ORB_G5',
            'US_DATA_830', 'E2', 1, 2.0, 'ORB_G5', 120, 0.54, 0.31,
            0.40, 1.20, 60.0, 4.5, 5, TRUE, 'active', 'deployable'
        ),
        (
            'GC_US_DATA_830_E2_RR2.0_CB1_ORB_G5',
            'US_DATA_830', 'E2', 1, 2.0, 'ORB_G5', 120, 0.54, 0.31,
            0.40, 1.20, 60.0, 4.5, 5, TRUE, 'active', 'non_deployable'
        )
    """)
    con.close()
    return db_path


class TestDeployableShelfSemantics:
    def test_counts_exclude_non_deployable_active_rows(self, tmp_path):
        db_path = _setup_validated_shelf_db(tmp_path)
        assert fetch_total_count(db_path) == 1
        assert fetch_unique_trade_count(db_path) == 1

    def test_summary_and_family_views_exclude_non_deployable_active_rows(self, tmp_path):
        db_path = _setup_validated_shelf_db(tmp_path)

        summary = fetch_summary(db_path)
        families = fetch_families(db_path)

        assert list(summary["count"]) == [1]
        assert list(summary["unique_trades"]) == [1]
        assert list(families["filter_variants"]) == [1]
