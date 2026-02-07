"""
Tests for trading_app.strategy_discovery module.
"""

import sys
import json
from pathlib import Path
from datetime import date

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_app.strategy_discovery import (
    compute_metrics,
    make_strategy_id,
    run_discovery,
)
from pipeline.cost_model import get_cost_spec


def _cost():
    return get_cost_spec("MGC")


# ============================================================================
# compute_metrics tests
# ============================================================================

class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_win_rate(self):
        """Win rate = wins / (wins + losses)."""
        outcomes = [
            {"trading_day": date(2024, 1, i), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.5}
            for i in range(1, 8)
        ] + [
            {"trading_day": date(2024, 1, i), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.2}
            for i in range(8, 11)
        ]

        m = compute_metrics(outcomes, _cost())
        assert m["win_rate"] == pytest.approx(7 / 10, abs=0.001)

    def test_expectancy(self):
        """E = (WR * AvgWin_R) - (LR * AvgLoss_R)."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0},
            {"trading_day": date(2024, 1, 2), "outcome": "win", "pnl_r": 3.0, "mae_r": 0.5, "mfe_r": 3.0},
            {"trading_day": date(2024, 1, 3), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0},
            {"trading_day": date(2024, 1, 4), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0},
        ]

        m = compute_metrics(outcomes, _cost())
        # WR=0.5, AvgWin=2.5, LR=0.5, AvgLoss=1.0
        # E = 0.5*2.5 - 0.5*1.0 = 1.25 - 0.5 = 0.75
        assert m["expectancy_r"] == pytest.approx(0.75, abs=0.01)

    def test_sharpe_ratio(self):
        """Sharpe = mean(R) / std(R)."""
        # All same R → std=0 → sharpe=None
        outcomes = [
            {"trading_day": date(2024, 1, i), "outcome": "win", "pnl_r": 1.0, "mae_r": 0.5, "mfe_r": 1.0}
            for i in range(1, 5)
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["sharpe_ratio"] is None  # std = 0

    def test_sharpe_ratio_valid(self):
        """Sharpe computes correctly with mixed results."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0},
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["sharpe_ratio"] is not None
        # mean = 0.5, std = sqrt(((2-0.5)^2 + (-1-0.5)^2)/1) = sqrt(4.5) ≈ 2.12
        # sharpe ≈ 0.5 / 2.12 ≈ 0.236
        assert abs(m["sharpe_ratio"]) < 1.0

    def test_max_drawdown(self):
        """Max drawdown tracks peak-to-trough in cumulative R."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 3.0, "mae_r": 0.5, "mfe_r": 3.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0},
            {"trading_day": date(2024, 1, 3), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0},
            {"trading_day": date(2024, 1, 4), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0},
        ]
        m = compute_metrics(outcomes, _cost())
        # Equity: 0→3→2→1→3. Peak=3 at idx 0, trough=1 at idx 2. DD=2.0
        assert m["max_drawdown_r"] == pytest.approx(2.0, abs=0.01)

    def test_yearly_breakdown(self):
        """Yearly results contain per-year stats."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0},
            {"trading_day": date(2024, 6, 1), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0},
            {"trading_day": date(2025, 1, 1), "outcome": "win", "pnl_r": 1.5, "mae_r": 0.5, "mfe_r": 1.5},
        ]
        m = compute_metrics(outcomes, _cost())
        yearly = json.loads(m["yearly_results"])
        assert "2024" in yearly
        assert "2025" in yearly
        assert yearly["2024"]["trades"] == 2
        assert yearly["2025"]["trades"] == 1

    def test_empty_outcomes(self):
        """Empty list returns zeroed metrics."""
        m = compute_metrics([], _cost())
        assert m["sample_size"] == 0
        assert m["win_rate"] is None

    def test_all_scratches(self):
        """Only scratches → no win/loss stats."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "scratch", "pnl_r": None, "mae_r": 0.1, "mfe_r": 0.1},
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["sample_size"] == 1
        assert m["win_rate"] is None


# ============================================================================
# make_strategy_id tests
# ============================================================================

class TestMakeStrategyId:
    """Tests for strategy ID generation."""

    def test_format(self):
        sid = make_strategy_id("MGC", "0900", 2.0, 1, "NO_FILTER")
        assert sid == "MGC_0900_RR2.0_CB1_NO_FILTER"

    def test_different_params_different_ids(self):
        s1 = make_strategy_id("MGC", "0900", 2.0, 1, "NO_FILTER")
        s2 = make_strategy_id("MGC", "0900", 2.0, 2, "NO_FILTER")
        s3 = make_strategy_id("MGC", "1000", 2.0, 1, "NO_FILTER")
        assert s1 != s2 != s3


# ============================================================================
# CLI test
# ============================================================================

class TestCLI:
    def test_help(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "trading_app/strategy_discovery.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert r.returncode == 0
        assert "instrument" in r.stdout
