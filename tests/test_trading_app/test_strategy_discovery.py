"""
Tests for trading_app.strategy_discovery module.
"""

import sys
import json
from pathlib import Path
from datetime import date, datetime, timezone

import pytest
import duckdb

from trading_app.strategy_discovery import (
    compute_metrics,
    make_strategy_id,
    run_discovery,
    _compute_relative_volumes,
)
from trading_app.config import (
    ENTRY_MODELS, ALL_FILTERS, VolumeFilter,
)

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

        m = compute_metrics(outcomes)
        assert m["win_rate"] == pytest.approx(7 / 10, abs=0.001)

    def test_expectancy(self):
        """E = (WR * AvgWin_R) - (LR * AvgLoss_R)."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "win", "pnl_r": 3.0, "mae_r": 0.5, "mfe_r": 3.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 3), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 4), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
        ]

        m = compute_metrics(outcomes)
        assert m["expectancy_r"] == pytest.approx(0.75, abs=0.01)

    def test_sharpe_ratio(self):
        """Sharpe = mean(R) / std(R)."""
        outcomes = [
            {"trading_day": date(2024, 1, i), "outcome": "win", "pnl_r": 1.0, "mae_r": 0.5, "mfe_r": 1.0, "entry_price": 2703.0, "stop_price": 2690.0}
            for i in range(1, 5)
        ]
        m = compute_metrics(outcomes)
        assert m["sharpe_ratio"] is None

    def test_sharpe_ratio_valid(self):
        """Sharpe computes correctly with mixed results."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes)
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
        m = compute_metrics(outcomes)
        assert m["max_drawdown_r"] == pytest.approx(2.0, abs=0.01)

    def test_yearly_breakdown(self):
        """Yearly results contain per-year stats."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 6, 1), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2025, 1, 1), "outcome": "win", "pnl_r": 1.5, "mae_r": 0.5, "mfe_r": 1.5, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes)
        yearly = json.loads(m["yearly_results"])
        assert "2024" in yearly
        assert "2025" in yearly
        assert yearly["2024"]["trades"] == 2
        assert yearly["2025"]["trades"] == 1

    def test_empty_outcomes(self):
        """Empty list returns zeroed metrics."""
        m = compute_metrics([])
        assert m["sample_size"] == 0
        assert m["win_rate"] is None
        assert m["median_risk_points"] is None
        assert m["avg_risk_points"] is None

    def test_all_scratches(self):
        """Only scratches -> no win/loss stats."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "scratch", "pnl_r": None, "mae_r": 0.1, "mfe_r": 0.1, "entry_price": None, "stop_price": None},
        ]
        m = compute_metrics(outcomes)
        assert m["sample_size"] == 0  # scratches excluded from sample_size
        assert m["win_rate"] is None

    def test_risk_stats_computed(self):
        """median_risk_points and avg_risk_points computed from entry/stop."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2705.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes)
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
        m = compute_metrics(outcomes)
        assert m["sharpe_ann"] is not None
        # Date range: 2024-01-01 to 2025-03-03 = 428 days ≈ 1.17 years
        span_years = ((date(2025, 3, 3) - date(2024, 1, 1)).days + 1) / 365.25
        assert m["trades_per_year"] == pytest.approx(11 / span_years, abs=0.5)
        # Identity: sharpe_ann = sharpe_ratio * sqrt(trades_per_year)
        expected = m["sharpe_ratio"] * (m["trades_per_year"] ** 0.5)
        assert m["sharpe_ann"] == pytest.approx(expected, abs=0.01)

    def test_sharpe_ann_none_when_no_variance(self):
        """If all trades have identical pnl_r, sharpe_ratio is None -> sharpe_ann is None."""
        outcomes = [
            {"trading_day": date(2024, 1, i), "outcome": "win", "pnl_r": 1.0, "mae_r": 0.5, "mfe_r": 1.0, "entry_price": 2703.0, "stop_price": 2690.0}
            for i in range(1, 5)
        ]
        m = compute_metrics(outcomes)
        assert m["sharpe_ratio"] is None  # zero variance
        assert m["sharpe_ann"] is None

    def test_trades_per_year_in_empty(self):
        """Empty outcomes returns trades_per_year=0, sharpe_ann=None."""
        m = compute_metrics([])
        assert m["trades_per_year"] == 0
        assert m["sharpe_ann"] is None

    def test_sharpe_ann_single_year(self):
        """sharpe_ann works when all trades are in a single year."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 3, 1), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 6, 1), "outcome": "win", "pnl_r": 1.5, "mae_r": 0.5, "mfe_r": 1.5, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes)
        # Date range: 2024-01-01 to 2024-06-01 = 153 days ≈ 0.42 years
        span_years = ((date(2024, 6, 1) - date(2024, 1, 1)).days + 1) / 365.25
        expected_tpy = 3 / span_years
        assert m["trades_per_year"] == pytest.approx(expected_tpy, abs=0.5)
        assert m["sharpe_ann"] is not None
        expected = m["sharpe_ratio"] * (m["trades_per_year"] ** 0.5)
        assert m["sharpe_ann"] == pytest.approx(expected, abs=0.01)

    def test_sharpe_ann_negative(self):
        """Negative sharpe_ann when strategy is losing."""
        outcomes = [
            {"trading_day": date(2024, 1, 1), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 2), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2024, 1, 3), "outcome": "win", "pnl_r": 2.0, "mae_r": 0.5, "mfe_r": 2.0, "entry_price": 2703.0, "stop_price": 2690.0},
            {"trading_day": date(2025, 1, 1), "outcome": "loss", "pnl_r": -1.0, "mae_r": 1.0, "mfe_r": 0.0, "entry_price": 2703.0, "stop_price": 2690.0},
        ]
        m = compute_metrics(outcomes)
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

class TestDoubleBreakNoExclusion:
    """Double-break days are NOT excluded from _build_filter_day_sets (Feb 2026).

    Double-break days are real losses in live trading — you can't predict
    them in advance. Including them gives honest discovery metrics.
    """

    def _make_feature(self, day_num, double_break):
        """Create a minimal feature row with break + double_break flag."""
        return {
            "trading_day": date(2024, 1, day_num),
            "orb_0900_break_dir": "long",
            "orb_0900_size": 5.0,
            "orb_0900_double_break": double_break,
        }

    def test_double_break_true_included(self):
        """Days with double_break=True are included (not filtered)."""
        from trading_app.strategy_discovery import _build_filter_day_sets

        features = [self._make_feature(1, True), self._make_feature(2, False)]
        result = _build_filter_day_sets(features, ["0900"], {"NO_FILTER": ALL_FILTERS["NO_FILTER"]})
        days = result[("NO_FILTER", "0900")]
        assert date(2024, 1, 1) in days
        assert date(2024, 1, 2) in days

    def test_double_break_none_included(self):
        """Days with double_break=None are included."""
        from trading_app.strategy_discovery import _build_filter_day_sets

        features = [self._make_feature(1, None)]
        result = _build_filter_day_sets(features, ["0900"], {"NO_FILTER": ALL_FILTERS["NO_FILTER"]})
        assert date(2024, 1, 1) in result[("NO_FILTER", "0900")]

    def test_all_days_with_break_included(self):
        """All days with a break direction are included regardless of double_break."""
        from trading_app.strategy_discovery import _build_filter_day_sets

        features = [
            self._make_feature(1, True),   # double break
            self._make_feature(2, False),  # single break
            self._make_feature(3, False),  # single break
        ]
        filters = {"NO_FILTER": ALL_FILTERS["NO_FILTER"], "ORB_G4": ALL_FILTERS["ORB_G4"]}
        result = _build_filter_day_sets(features, ["0900"], filters)

        assert len(result[("NO_FILTER", "0900")]) == 3
        assert len(result[("ORB_G4", "0900")]) == 3


class TestComputeMetricsScratchCounts:
    """Tests for entry_signals, scratch_count, early_exit_count in compute_metrics."""

    def _make_outcome(self, day_num, outcome, pnl_r=1.0):
        return {
            "trading_day": date(2024, 1, day_num),
            "outcome": outcome,
            "pnl_r": pnl_r if outcome in ("win", "loss", "early_exit") else 0.0,
            "mae_r": 0.5,
            "mfe_r": 1.0,
            "entry_price": 2703.0,
            "stop_price": 2690.0,
        }

    def test_sample_size_excludes_scratches(self):
        """10W, 5L, 3 scratches -> sample_size=15, entry_signals=18."""
        outcomes = (
            [self._make_outcome(i, "win", 2.0) for i in range(1, 11)]
            + [self._make_outcome(i, "loss", -1.0) for i in range(11, 16)]
            + [self._make_outcome(i, "scratch") for i in range(16, 19)]
        )
        m = compute_metrics(outcomes)
        assert m["sample_size"] == 15
        assert m["entry_signals"] == 18
        assert m["scratch_count"] == 3
        assert m["early_exit_count"] == 0

    def test_win_rate_uses_clean_sample(self):
        """4W, 6L, 5 scratches -> win_rate=0.40."""
        outcomes = (
            [self._make_outcome(i, "win", 2.0) for i in range(1, 5)]
            + [self._make_outcome(i, "loss", -1.0) for i in range(5, 11)]
            + [self._make_outcome(i, "scratch") for i in range(11, 16)]
        )
        m = compute_metrics(outcomes)
        assert m["win_rate"] == pytest.approx(0.4, abs=0.001)
        assert m["sample_size"] == 10
        assert m["entry_signals"] == 15
        assert m["scratch_count"] == 5

class TestZeroSampleNotWritten:
    """B2: All-scratch/early_exit outcomes should not produce strategy rows."""

    def test_zero_sample_strategies_not_written(self, tmp_path):
        """All-scratch outcomes should not produce a strategy row."""
        db_path = tmp_path / "test_b2.db"
        con = duckdb.connect(str(db_path))

        # Create minimal schema
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)
        con.close()

        from trading_app.db_manager import init_trading_app_schema
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))
        # Insert daily_features rows with ORB breaks
        for i in range(1, 11):
            con.execute(
                """INSERT INTO daily_features (trading_day, symbol, orb_minutes,
                    orb_0900_high, orb_0900_low, orb_0900_size, orb_0900_break_dir)
                   VALUES (?, 'MGC', 5, 2710.0, 2700.0, 10.0, 'long')""",
                [date(2024, 1, i)],
            )
        # Insert orb_outcomes that are ALL scratches
        for i in range(1, 11):
            con.execute(
                """INSERT INTO orb_outcomes
                    (trading_day, symbol, orb_minutes, orb_label, entry_model,
                     rr_target, confirm_bars, outcome, pnl_r, mae_r, mfe_r)
                   VALUES (?, 'MGC', 5, '0900', 'E1', 2.0, 1, 'scratch', 0.0, 0.1, 0.1)""",
                [date(2024, 1, i)],
            )
        con.commit()
        con.close()

        # Run discovery
        count = run_discovery(
            db_path=db_path, instrument="MGC",
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 31),
        )

        # Verify no strategies were written (all had sample_size=0)
        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute("SELECT COUNT(*) FROM experimental_strategies").fetchone()[0]
        con.close()
        assert rows == 0, f"Expected 0 strategies but got {rows}"

# ============================================================================
# Dedup tests
# ============================================================================

class TestDedup:
    """Tests for trade-day hash dedup (_mark_canonical, _compute_trade_day_hash)."""

    def test_dedup_identifies_identical_trade_sets(self):
        """3 strategies with same trade days -> 1 canonical + 2 aliases."""
        from trading_app.strategy_discovery import _mark_canonical
        from trading_app.db_manager import compute_trade_day_hash as _compute_trade_day_hash

        days = [date(2024, 1, i) for i in range(1, 11)]
        day_hash = _compute_trade_day_hash(days)

        strategies = [
            {"strategy_id": f"MGC_0900_E1_RR2.0_CB1_ORB_{f}",
             "instrument": "MGC", "orb_label": "0900",
             "entry_model": "E1", "rr_target": 2.0, "confirm_bars": 1,
             "filter_key": f"ORB_{f}", "trade_day_hash": day_hash,
             "metrics": {"expectancy_r": 0.5}}
            for f in ["G4", "G5", "G6"]
        ]

        _mark_canonical(strategies)
        canonical = [s for s in strategies if s["is_canonical"]]
        aliases = [s for s in strategies if not s["is_canonical"]]

        assert len(canonical) == 1
        assert len(aliases) == 2
        # G6 is highest specificity among G4, G5, G6
        assert canonical[0]["filter_key"] == "ORB_G6"
        # Aliases point to canonical
        for a in aliases:
            assert a["canonical_strategy_id"] == canonical[0]["strategy_id"]

    def test_dedup_preserves_different_trade_sets(self):
        """2 strategies with different trade days -> both canonical."""
        from trading_app.strategy_discovery import _mark_canonical
        from trading_app.db_manager import compute_trade_day_hash as _compute_trade_day_hash

        days1 = [date(2024, 1, i) for i in range(1, 6)]
        days2 = [date(2024, 1, i) for i in range(6, 11)]

        strategies = [
            {"strategy_id": "MGC_0900_E1_RR2.0_CB1_ORB_G4",
             "instrument": "MGC", "orb_label": "0900",
             "entry_model": "E1", "rr_target": 2.0, "confirm_bars": 1,
             "filter_key": "ORB_G4", "trade_day_hash": _compute_trade_day_hash(days1),
             "metrics": {"expectancy_r": 0.5}},
            {"strategy_id": "MGC_0900_E1_RR2.0_CB1_ORB_G8",
             "instrument": "MGC", "orb_label": "0900",
             "entry_model": "E1", "rr_target": 2.0, "confirm_bars": 1,
             "filter_key": "ORB_G8", "trade_day_hash": _compute_trade_day_hash(days2),
             "metrics": {"expectancy_r": 0.8}},
        ]

        _mark_canonical(strategies)
        canonical = [s for s in strategies if s["is_canonical"]]
        assert len(canonical) == 2

# ============================================================================
# Validator alias skipping
# ============================================================================

class TestValidatorSkipsAliases:
    """Test that strategy_validator skips non-canonical (alias) strategies."""

    def _setup_db(self, tmp_path, strategies):
        """Create temp DB with schema + strategies."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)
        con.close()

        from trading_app.db_manager import init_trading_app_schema
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

    def _make_row(self, **overrides):
        base = {
            "strategy_id": "MGC_0900_E1_RR2.0_CB1_ORB_G4",
            "instrument": "MGC",
            "orb_label": "0900",
            "orb_minutes": 5,
            "rr_target": 2.0,
            "confirm_bars": 1,
            "entry_model": "E1",
            "filter_type": "ORB_G4",
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
            "is_canonical": True,
        }
        base.update(overrides)
        return base

    def test_validator_skips_aliases(self, tmp_path):
        """Alias row is skipped without error, status set to SKIPPED."""
        from trading_app.strategy_validator import run_validation

        canonical = self._make_row(is_canonical=True)
        alias = self._make_row(
            strategy_id="MGC_0900_E1_RR2.0_CB1_NO_FILTER",
            filter_type="NO_FILTER",
            is_canonical=False,
            canonical_strategy_id="MGC_0900_E1_RR2.0_CB1_ORB_G4",
        )
        db_path = self._setup_db(tmp_path, [canonical, alias])

        passed, rejected = run_validation(
            db_path=db_path, instrument="MGC", enable_walkforward=False
        )

        # Canonical passes validation, alias is skipped
        assert passed == 1
        assert rejected == 0

        # Verify alias got SKIPPED status
        con = duckdb.connect(str(db_path), read_only=True)
        alias_status = con.execute(
            "SELECT validation_status FROM experimental_strategies WHERE strategy_id = ?",
            ["MGC_0900_E1_RR2.0_CB1_NO_FILTER"],
        ).fetchone()[0]
        con.close()
        assert alias_status == "SKIPPED"

class TestCLI:
    def test_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["strategy_discovery", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from trading_app.strategy_discovery import main
            main()
        assert exc_info.value.code == 0
        assert "instrument" in capsys.readouterr().out
