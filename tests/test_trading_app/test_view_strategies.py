"""
Tests for trading_app.view_strategies module.
"""

import sys
import math
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_app.view_strategies import (
    _safe_float,
    _fmt_signed,
    format_table,
    format_families,
    format_summary,
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
        df = pd.DataFrame([{
            "strategy_id": "MGC_0900_E1_RR2.0_CB1_ORB_G4",
            "orb_label": "0900",
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
        }])
        result = format_table(df)
        assert "nan" not in result.lower() or "shann" in result.lower()
        assert "N/A" in result

    def test_with_none_sharpe_ann(self):
        """None values display as N/A."""
        df = pd.DataFrame([{
            "strategy_id": "MGC_0900_E1_RR2.0_CB1_ORB_G4",
            "orb_label": "0900",
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
        }])
        result = format_table(df)
        assert "N/A" in result

    def test_with_valid_sharpe_ann(self):
        """Valid sharpe_ann displays with sign."""
        df = pd.DataFrame([{
            "strategy_id": "MGC_0900_E1_RR2.0_CB1_ORB_G4",
            "orb_label": "0900",
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
        }])
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
        df = pd.DataFrame([{
            "orb_label": "0900",
            "entry_model": "E1",
            "rr_target": 2.0,
            "confirm_bars": 1,
            "filter_variants": 3,
            "best_filter": "ORB_G4",
            "best_shann": float("nan"),
            "max_n": 100,
            "best_wr_pct": 45.0,
            "best_expr": 0.15,
        }])
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
        df = pd.DataFrame([{
            "orb_label": "0900",
            "count": 50,
            "unique_trades": 15,
            "avg_expr": 0.15,
            "best_expr": 0.35,
            "avg_shann": float("nan"),
            "best_shann": float("nan"),
            "avg_wr_pct": 42.0,
        }])
        result = format_summary(df)
        assert "N/A" in result
        assert "nan" not in result.replace("ShANN", "")


# ============================================================================
# CLI test
# ============================================================================

class TestCLI:
    def test_help(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "trading_app/view_strategies.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert r.returncode == 0
        assert "--family" in r.stdout
        assert "--db" in r.stdout
        assert "--sort" in r.stdout
        assert "sharpe_ann" in r.stdout
