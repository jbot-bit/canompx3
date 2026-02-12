"""
Tests for trading_app.strategy_discovery module.
"""

import sys
import json
from pathlib import Path
from datetime import date, datetime, timezone

import pytest
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_app.strategy_discovery import (
    compute_metrics,
    make_strategy_id,
    run_discovery,
    _compute_relative_volumes,
)
from trading_app.config import (
    ENTRY_MODELS, ALL_FILTERS, VolumeFilter,
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
            {"trading_day": date(2024, 1, i), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.5, "entry_price": 2703.0, "stop_price": 2690.0}
            for i in range(1, 8)
        ] + [
            {"trading_day": date(2024, 1, i), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.2, "entry_price": 2703.0, "stop_price": 2690.0}
            for i in range(8, 11)
        ]

        m = compute_metrics(outcomes, _cost())
        assert m["win_rate"] == pytest.approx(7 / 10, abs=0.001)

    def test_expectancy(self):
        """E = (WR * AvgWin_R) - (LR * AvgLoss_R)."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "win", "pnl_r": 3.0, "mae_r": 0.5, "mfe_r": 3.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 3), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 4), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
        ]

        m = compute_metrics(outcomes, _cost())
        assert m["expectancy_r"] == pytest.approx(0.75, abs=0.01)

    def test_sharpe_ratio(self):
        """Sharpe = mean(R) / std(R)."""
        outcomes = [
            {"trading_day": date(2024, 1, i), "outcome": "win", "pnl_r": 1.0, "mae_r": 0.5, "mfe_r": 1.0, "entry_price": 2703.0, "stop_price": 2690.0}
            for i in range(1, 5)
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["sharpe_ratio"] is None

    def test_sharpe_ratio_valid(self):
        """Sharpe computes correctly with mixed results."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["sharpe_ratio"] is not None
        assert abs(m["sharpe_ratio"]) < 1.0

    def test_max_drawdown(self):
        """Max drawdown tracks peak-to-trough in cumulative R."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 3.0, "mae_r": 0.5, "mfe_r": 3.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 3), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 4), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["max_drawdown_r"] == pytest.approx(2.0, abs=0.01)

    def test_yearly_breakdown(self):
        """Yearly results contain per-year stats."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 6, 1), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2025, 1, 1), "outcome": "win", "pnl_r": 1.5, "mae_r": 0.5, "mfe_r": 1.5, "entry_price": 2703.0, "stop_price": 2690.0},
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
        assert m["median_risk_points"] is None
        assert m["avg_risk_points"] is None

    def test_all_scratches(self):
        """Only scratches -> no win/loss stats."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "scratch", "pnl_r": None, "mae_r": 0.1, "mfe_r": 0.1, "entry_price": None, "stop_price": None},
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["sample_size"] == 0  # scratches excluded from sample_size
        assert m["win_rate"] is None

    def test_risk_stats_computed(self):
        """median_risk_points and avg_risk_points computed from entry/stop."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2705.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["median_risk_points"] == pytest.approx(14.0, abs=0.01)  # median of 13, 15
        assert m["avg_risk_points"] == pytest.approx(14.0, abs=0.01)  # avg of 13, 15

    def test_sharpe_ann_computed(self):
        """sharpe_ann = sharpe_ratio * sqrt(trades_per_year)."""
        outcomes = [
            {"trading_day": date(2024, 1, i), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0}
            for i in range(1, 6)
        ] + [
            {"trading_day": date(2024, 6, i), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0}
            for i in range(1, 4)
        ] + [
            {"trading_day": date(2025, 3, i), "outcome": "win", "pnl_r": 1.5, "mae_r": 0.5, "mfe_r": 1.5, "entry_price": 2703.0, "stop_price": 2690.0}
            for i in range(1, 4)
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["sharpe_ann"] is not None
        assert m["trades_per_year"] == pytest.approx(11 / 2, abs=0.1)  # 11 trades / 2 years
        # Identity: sharpe_ann = sharpe_ratio * sqrt(trades_per_year)
        expected = m["sharpe_ratio"] * (m["trades_per_year"] ** 0.5)
        assert m["sharpe_ann"] == pytest.approx(expected, abs=0.01)

    def test_sharpe_ann_none_when_no_variance(self):
        """If all trades have identical pnl_r, sharpe_ratio is None -> sharpe_ann is None."""
        outcomes = [
            {"trading_day": date(2024, 1, i), "outcome": "win", "pnl_r": 1.0, "mae_r": 0.5, "mfe_r": 1.0, "entry_price": 2703.0, "stop_price": 2690.0}
            for i in range(1, 5)
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["sharpe_ratio"] is None  # zero variance
        assert m["sharpe_ann"] is None

    def test_trades_per_year_in_empty(self):
        """Empty outcomes returns trades_per_year=0, sharpe_ann=None."""
        m = compute_metrics([], _cost())
        assert m["trades_per_year"] == 0
        assert m["sharpe_ann"] is None

    def test_sharpe_ann_single_year(self):
        """sharpe_ann works when all trades are in a single year."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 3, 1), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 6, 1), "outcome": "win", "pnl_r": 1.5, "mae_r": 0.5, "mfe_r": 1.5, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["trades_per_year"] == 3.0  # 3 trades / 1 year
        assert m["sharpe_ann"] is not None
        # With 1 year: sharpe_ann = sharpe_ratio * sqrt(3)
        expected = m["sharpe_ratio"] * (3.0 ** 0.5)
        assert m["sharpe_ann"] == pytest.approx(expected, abs=0.01)

    def test_sharpe_ann_negative(self):
        """Negative sharpe_ann when strategy is losing."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 3), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2025, 1, 1), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes, _cost())
        assert m["sharpe_ratio"] < 0
        assert m["sharpe_ann"] < 0
        expected = m["sharpe_ratio"] * (m["trades_per_year"] ** 0.5)
        assert m["sharpe_ann"] == pytest.approx(expected, abs=0.01)


# ============================================================================
# make_strategy_id tests
# ============================================================================

class TestMakeStrategyId:
    """Tests for strategy ID generation."""

    def test_format(self):
        sid = make_strategy_id("MGC", "0900", "E1", 2.0, 1, "NO_FILTER")
        assert sid == "MGC_0900_E1_RR2.0_CB1_NO_FILTER"

    def test_different_params_different_ids(self):
        s1 = make_strategy_id("MGC", "0900", "E1", 2.0, 1, "NO_FILTER")
        s2 = make_strategy_id("MGC", "0900", "E1", 2.0, 2, "NO_FILTER")
        s3 = make_strategy_id("MGC", "1000", "E1", 2.0, 1, "NO_FILTER")
        s4 = make_strategy_id("MGC", "0900", "E3", 2.0, 1, "NO_FILTER")
        assert len({s1, s2, s3, s4}) == 4

    def test_entry_model_in_id(self):
        for em in ENTRY_MODELS:
            sid = make_strategy_id("MGC", "0900", em, 2.0, 1, "NO_FILTER")
            assert f"_{em}_" in sid


# ============================================================================
# CLI test
# ============================================================================

# ============================================================================
# _compute_relative_volumes tests
# ============================================================================

def _make_bars_1m_table(con):
    """Create bars_1m table for testing."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS bars_1m (
            ts_utc TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            source_symbol TEXT,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)


class TestComputeRelativeVolumes:
    """Tests for _compute_relative_volumes relative volume enrichment."""

    def _setup_db(self, tmp_path):
        """Create a test DB with bars_1m data."""
        db_path = tmp_path / "test_vol.db"
        con = duckdb.connect(str(db_path))
        _make_bars_1m_table(con)
        return con

    def test_baseline_uses_same_minute_of_day(self, tmp_path):
        """Baseline median must use the SAME minute-of-day as the break bar."""
        con = self._setup_db(tmp_path)

        # Insert 20 prior days of volume at 23:05 UTC (break minute)
        # and different volumes at 23:10 UTC (different minute)
        for i in range(1, 22):  # days 1-21
            ts_05 = datetime(2024, 1, i, 23, 5, tzinfo=timezone.utc)
            ts_10 = datetime(2024, 1, i, 23, 10, tzinfo=timezone.utc)
            vol_05 = 200 if i == 21 else 100  # day 21 = break bar with 200
            con.execute(
                "INSERT INTO bars_1m VALUES (?::TIMESTAMPTZ, 'MGC', 'GCG4', 100, 101, 99, 100, ?)",
                [ts_05.isoformat(), vol_05],
            )
            con.execute(
                "INSERT INTO bars_1m VALUES (?::TIMESTAMPTZ, 'MGC', 'GCG4', 100, 101, 99, 100, 500)",
                [ts_10.isoformat()],
            )

        # Feature row: break at 23:05 on day 21
        features = [{
            "trading_day": date(2024, 1, 21),
            "orb_0900_break_ts": datetime(2024, 1, 21, 23, 5, tzinfo=timezone.utc),
        }]

        _compute_relative_volumes(con, features, "MGC", ["0900"], ALL_FILTERS)
        con.close()

        # Baseline = median of 100 (20 prior days at :05), break = 200
        # rel_vol = 200 / 100 = 2.0
        assert "rel_vol_0900" in features[0]
        assert features[0]["rel_vol_0900"] == pytest.approx(2.0, abs=0.01)

    def test_fail_closed_no_baseline(self, tmp_path):
        """If no prior data exists at the break minute, rel_vol is NOT set."""
        con = self._setup_db(tmp_path)

        # Only insert the break bar itself, no prior history
        con.execute(
            "INSERT INTO bars_1m VALUES (?::TIMESTAMPTZ, 'MGC', 'GCG4', 100, 101, 99, 100, 200)",
            [datetime(2024, 6, 15, 23, 5, tzinfo=timezone.utc).isoformat()],
        )

        features = [{
            "trading_day": date(2024, 6, 15),
            "orb_0900_break_ts": datetime(2024, 6, 15, 23, 5, tzinfo=timezone.utc),
        }]

        _compute_relative_volumes(con, features, "MGC", ["0900"], ALL_FILTERS)
        con.close()

        # No baseline -> fail-closed -> rel_vol not set
        assert "rel_vol_0900" not in features[0]

    def test_fail_closed_zero_volume_break_bar(self, tmp_path):
        """Break bar with volume=0 -> fail-closed."""
        con = self._setup_db(tmp_path)

        # Prior history
        for i in range(1, 21):
            con.execute(
                "INSERT INTO bars_1m VALUES (?::TIMESTAMPTZ, 'MGC', 'GCG4', 100, 101, 99, 100, 100)",
                [datetime(2024, 3, i, 23, 5, tzinfo=timezone.utc).isoformat()],
            )
        # Break bar with volume=0
        con.execute(
            "INSERT INTO bars_1m VALUES (?::TIMESTAMPTZ, 'MGC', 'GCG4', 100, 101, 99, 100, 0)",
            [datetime(2024, 3, 21, 23, 5, tzinfo=timezone.utc).isoformat()],
        )

        features = [{
            "trading_day": date(2024, 3, 21),
            "orb_0900_break_ts": datetime(2024, 3, 21, 23, 5, tzinfo=timezone.utc),
        }]

        _compute_relative_volumes(con, features, "MGC", ["0900"], ALL_FILTERS)
        con.close()

        assert "rel_vol_0900" not in features[0]

    def test_no_volume_filters_skips(self, tmp_path):
        """If no VolumeFilter in all_filters, function returns immediately."""
        con = self._setup_db(tmp_path)

        # Pass filters with NO VolumeFilter
        no_vol_filters = {k: v for k, v in ALL_FILTERS.items() if not isinstance(v, VolumeFilter)}

        features = [{
            "trading_day": date(2024, 1, 21),
            "orb_0900_break_ts": datetime(2024, 1, 21, 23, 5, tzinfo=timezone.utc),
        }]

        _compute_relative_volumes(con, features, "MGC", ["0900"], no_vol_filters)
        con.close()

        assert "rel_vol_0900" not in features[0]

    def test_volume_filter_reduces_eligible_days(self, tmp_path):
        """VOL_RV12_N20 produces fewer eligible days than NO_FILTER."""
        con = self._setup_db(tmp_path)

        from trading_app.strategy_discovery import _build_filter_day_sets

        # Create 30 days of data using March (31 days). First 20 have volume=100.
        # Days 21-30: some high volume, some low volume.
        for i in range(1, 31):
            vol = 100  # baseline volume
            if i > 20:
                # Days 21,23,25,27,29 = high volume (150), days 22,24,26,28,30 = low volume (50)
                vol = 150 if i % 2 == 1 else 50
            con.execute(
                "INSERT INTO bars_1m VALUES (?::TIMESTAMPTZ, 'MGC', 'GCG4', 100, 101, 99, 100, ?)",
                [datetime(2024, 3, i, 23, 5, tzinfo=timezone.utc).isoformat(), vol],
            )

        # Features: days 21-30 all have breaks
        features = []
        for i in range(21, 31):
            features.append({
                "trading_day": date(2024, 3, i),
                "orb_0900_break_ts": datetime(2024, 3, i, 23, 5, tzinfo=timezone.utc),
                "orb_0900_break_dir": "long",
                "orb_0900_size": 5.0,
            })

        _compute_relative_volumes(con, features, "MGC", ["0900"], ALL_FILTERS)

        nf_days = _build_filter_day_sets(features, ["0900"], {"NO_FILTER": ALL_FILTERS["NO_FILTER"]})
        vol_days = _build_filter_day_sets(features, ["0900"], {"VOL_RV12_N20": ALL_FILTERS["VOL_RV12_N20"]})
        con.close()

        nf_count = len(nf_days[("NO_FILTER", "0900")])
        vol_count = len(vol_days[("VOL_RV12_N20", "0900")])

        # NO_FILTER: all 10 days with breaks
        assert nf_count == 10
        # VOL_RV12_N20: only high-volume days (rel_vol >= 1.2)
        # Baseline = median(100...100) = 100. High vol days: 150/100 = 1.5 >= 1.2
        # Low vol days: 50/100 = 0.5 < 1.2
        assert vol_count < nf_count
        assert vol_count == 5  # only the 5 high-volume days


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
