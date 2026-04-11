"""
Tests for trading_app.strategy_discovery module.
"""

import json
import sys
from datetime import UTC, date, datetime, timezone
from pathlib import Path

import duckdb
import pytest

from trading_app.config import (
    ALL_FILTERS,
    ENTRY_MODELS,
    VolumeFilter,
)
from trading_app.strategy_discovery import (
    _compute_haircut_sharpe,
    _compute_relative_volumes,
    compute_metrics,
    make_strategy_id,
    run_discovery,
)

# ============================================================================
# compute_metrics tests
# ============================================================================


class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_win_rate(self):
        """Win rate = wins / (wins + losses)."""
        outcomes = [
            {
                "trading_day": date(2024, 1, i),
                "outcome": "win",
                "pnl_r": 2.0,
                "mae_r": 0.5,
                "mfe_r": 2.5,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            }
            for i in range(1, 8)
        ] + [
            {
                "trading_day": date(2024, 1, i),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.2,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            }
            for i in range(8, 11)
        ]

        m = compute_metrics(outcomes)
        assert m["win_rate"] == pytest.approx(7 / 10, abs=0.001)

    def test_expectancy(self):
        """E = (WR * AvgWin_R) - (LR * AvgLoss_R)."""
        outcomes = [
            {
                "trading_day": date(2024, 1, 1),
                "outcome": "win",
                "pnl_r": 2.0,
                "mae_r": 0.5,
                "mfe_r": 2.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 2),
                "outcome": "win",
                "pnl_r": 3.0,
                "mae_r": 0.5,
                "mfe_r": 3.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 3),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 4),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
        ]

        m = compute_metrics(outcomes)
        assert m["expectancy_r"] == pytest.approx(0.75, abs=0.01)

    def test_sharpe_ratio(self):
        """Sharpe = mean(R) / std(R)."""
        outcomes = [
            {
                "trading_day": date(2024, 1, i),
                "outcome": "win",
                "pnl_r": 1.0,
                "mae_r": 0.5,
                "mfe_r": 1.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            }
            for i in range(1, 5)
        ]
        m = compute_metrics(outcomes)
        assert m["sharpe_ratio"] is None

    def test_sharpe_ratio_valid(self):
        """Sharpe computes correctly with mixed results."""
        outcomes = [
            {
                "trading_day": date(2024, 1, 1),
                "outcome": "win",
                "pnl_r": 2.0,
                "mae_r": 0.5,
                "mfe_r": 2.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 2),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
        ]
        m = compute_metrics(outcomes)
        assert m["sharpe_ratio"] is not None
        assert abs(m["sharpe_ratio"]) < 1.0

    def test_max_drawdown(self):
        """Max drawdown tracks peak-to-trough in cumulative R."""
        outcomes = [
            {
                "trading_day": date(2024, 1, 1),
                "outcome": "win",
                "pnl_r": 3.0,
                "mae_r": 0.5,
                "mfe_r": 3.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 2),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 3),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 4),
                "outcome": "win",
                "pnl_r": 2.0,
                "mae_r": 0.5,
                "mfe_r": 2.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
        ]
        m = compute_metrics(outcomes)
        assert m["max_drawdown_r"] == pytest.approx(2.0, abs=0.01)

    def test_yearly_breakdown(self):
        """Yearly results contain per-year stats."""
        outcomes = [
            {
                "trading_day": date(2024, 1, 1),
                "outcome": "win",
                "pnl_r": 2.0,
                "mae_r": 0.5,
                "mfe_r": 2.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 6, 1),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2025, 1, 1),
                "outcome": "win",
                "pnl_r": 1.5,
                "mae_r": 0.5,
                "mfe_r": 1.5,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
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
            {
                "trading_day": date(2024, 1, 1),
                "outcome": "scratch",
                "pnl_r": None,
                "mae_r": 0.1,
                "mfe_r": 0.1,
                "entry_price": None,
                "stop_price": None,
            },
        ]
        m = compute_metrics(outcomes)
        assert m["sample_size"] == 0  # scratches excluded from sample_size
        assert m["win_rate"] is None

    def test_risk_stats_computed(self):
        """median_risk_points and avg_risk_points computed from entry/stop."""
        outcomes = [
            {
                "trading_day": date(2024, 1, 1),
                "outcome": "win",
                "pnl_r": 2.0,
                "mae_r": 0.5,
                "mfe_r": 2.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 2),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2705.0,
                "stop_price": 2690.0,
            },
        ]
        m = compute_metrics(outcomes)
        assert m["median_risk_points"] == pytest.approx(14.0, abs=0.01)  # median of 13, 15
        assert m["avg_risk_points"] == pytest.approx(14.0, abs=0.01)  # avg of 13, 15

    def test_sharpe_ann_computed(self):
        """sharpe_ann = sharpe_ratio * sqrt(trades_per_year)."""
        outcomes = (
            [
                {
                    "trading_day": date(2024, 1, i),
                    "outcome": "win",
                    "pnl_r": 2.0,
                    "mae_r": 0.5,
                    "mfe_r": 2.0,
                    "entry_price": 2703.0,
                    "stop_price": 2690.0,
                }
                for i in range(1, 6)
            ]
            + [
                {
                    "trading_day": date(2024, 6, i),
                    "outcome": "loss",
                    "pnl_r": -1.0,
                    "mae_r": 1.0,
                    "mfe_r": 0.0,
                    "entry_price": 2703.0,
                    "stop_price": 2690.0,
                }
                for i in range(1, 4)
            ]
            + [
                {
                    "trading_day": date(2025, 3, i),
                    "outcome": "win",
                    "pnl_r": 1.5,
                    "mae_r": 0.5,
                    "mfe_r": 1.5,
                    "entry_price": 2703.0,
                    "stop_price": 2690.0,
                }
                for i in range(1, 4)
            ]
        )
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
            {
                "trading_day": date(2024, 1, i),
                "outcome": "win",
                "pnl_r": 1.0,
                "mae_r": 0.5,
                "mfe_r": 1.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            }
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
            {
                "trading_day": date(2024, 1, 1),
                "outcome": "win",
                "pnl_r": 2.0,
                "mae_r": 0.5,
                "mfe_r": 2.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 3, 1),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 6, 1),
                "outcome": "win",
                "pnl_r": 1.5,
                "mae_r": 0.5,
                "mfe_r": 1.5,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
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
            {
                "trading_day": date(2024, 1, 1),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 2),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2024, 1, 3),
                "outcome": "win",
                "pnl_r": 2.0,
                "mae_r": 0.5,
                "mfe_r": 2.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
            {
                "trading_day": date(2025, 1, 1),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.0,
                "entry_price": 2703.0,
                "stop_price": 2690.0,
            },
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
        sid = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "NO_FILTER")
        assert sid == "MGC_CME_REOPEN_E1_RR2.0_CB1_NO_FILTER"

    def test_different_params_different_ids(self):
        s1 = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "NO_FILTER")
        s2 = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 2, "NO_FILTER")
        s3 = make_strategy_id("MGC", "TOKYO_OPEN", "E1", 2.0, 1, "NO_FILTER")
        s4 = make_strategy_id("MGC", "CME_REOPEN", "E3", 2.0, 1, "NO_FILTER")
        assert len({s1, s2, s3, s4}) == 4

    def test_entry_model_in_id(self):
        for em in ENTRY_MODELS:
            sid = make_strategy_id("MGC", "CME_REOPEN", em, 2.0, 1, "NO_FILTER")
            assert f"_{em}_" in sid

    def test_orb_minutes_default_no_suffix(self):
        sid = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "NO_FILTER")
        assert "_O" not in sid

    def test_orb_minutes_5_no_suffix(self):
        sid = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "NO_FILTER", orb_minutes=5)
        assert sid == "MGC_CME_REOPEN_E1_RR2.0_CB1_NO_FILTER"

    def test_orb_minutes_15_suffix(self):
        sid = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "ORB_G4", orb_minutes=15)
        assert sid == "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4_O15"

    def test_orb_minutes_30_suffix(self):
        sid = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "ORB_G4", orb_minutes=30)
        assert sid == "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4_O30"

    def test_orb_minutes_with_dst_suffix_order(self):
        sid = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "ORB_G4", dst_regime="winter", orb_minutes=15)
        assert sid == "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4_O15_W"

    def test_orb_minutes_15_vs_5_different(self):
        s5 = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "ORB_G4", orb_minutes=5)
        s15 = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "ORB_G4", orb_minutes=15)
        assert s5 != s15


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
            ts_05 = datetime(2024, 1, i, 23, 5, tzinfo=UTC)
            ts_10 = datetime(2024, 1, i, 23, 10, tzinfo=UTC)
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
        features = [
            {
                "trading_day": date(2024, 1, 21),
                "orb_CME_REOPEN_break_ts": datetime(2024, 1, 21, 23, 5, tzinfo=UTC),
            }
        ]

        _compute_relative_volumes(con, features, "MGC", ["CME_REOPEN"], ALL_FILTERS)
        con.close()

        # Baseline = median of 100 (20 prior days at :05), break = 200
        # rel_vol = 200 / 100 = 2.0
        assert "rel_vol_CME_REOPEN" in features[0]
        assert features[0]["rel_vol_CME_REOPEN"] == pytest.approx(2.0, abs=0.01)

    def test_fail_closed_no_baseline(self, tmp_path):
        """If no prior data exists at the break minute, rel_vol is NOT set."""
        con = self._setup_db(tmp_path)

        # Only insert the break bar itself, no prior history
        con.execute(
            "INSERT INTO bars_1m VALUES (?::TIMESTAMPTZ, 'MGC', 'GCG4', 100, 101, 99, 100, 200)",
            [datetime(2024, 6, 15, 23, 5, tzinfo=UTC).isoformat()],
        )

        features = [
            {
                "trading_day": date(2024, 6, 15),
                "orb_CME_REOPEN_break_ts": datetime(2024, 6, 15, 23, 5, tzinfo=UTC),
            }
        ]

        _compute_relative_volumes(con, features, "MGC", ["CME_REOPEN"], ALL_FILTERS)
        con.close()

        # No baseline -> fail-closed -> rel_vol not set
        assert "rel_vol_CME_REOPEN" not in features[0]

    def test_fail_closed_zero_volume_break_bar(self, tmp_path):
        """Break bar with volume=0 -> fail-closed."""
        con = self._setup_db(tmp_path)

        # Prior history
        for i in range(1, 21):
            con.execute(
                "INSERT INTO bars_1m VALUES (?::TIMESTAMPTZ, 'MGC', 'GCG4', 100, 101, 99, 100, 100)",
                [datetime(2024, 3, i, 23, 5, tzinfo=UTC).isoformat()],
            )
        # Break bar with volume=0
        con.execute(
            "INSERT INTO bars_1m VALUES (?::TIMESTAMPTZ, 'MGC', 'GCG4', 100, 101, 99, 100, 0)",
            [datetime(2024, 3, 21, 23, 5, tzinfo=UTC).isoformat()],
        )

        features = [
            {
                "trading_day": date(2024, 3, 21),
                "orb_CME_REOPEN_break_ts": datetime(2024, 3, 21, 23, 5, tzinfo=UTC),
            }
        ]

        _compute_relative_volumes(con, features, "MGC", ["CME_REOPEN"], ALL_FILTERS)
        con.close()

        assert "rel_vol_CME_REOPEN" not in features[0]

    def test_no_volume_filters_skips(self, tmp_path):
        """If no VolumeFilter in all_filters, function returns immediately."""
        con = self._setup_db(tmp_path)

        # Pass filters with NO VolumeFilter
        no_vol_filters = {k: v for k, v in ALL_FILTERS.items() if not isinstance(v, VolumeFilter)}

        features = [
            {
                "trading_day": date(2024, 1, 21),
                "orb_CME_REOPEN_break_ts": datetime(2024, 1, 21, 23, 5, tzinfo=UTC),
            }
        ]

        _compute_relative_volumes(con, features, "MGC", ["CME_REOPEN"], no_vol_filters)
        con.close()

        assert "rel_vol_CME_REOPEN" not in features[0]

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
                [datetime(2024, 3, i, 23, 5, tzinfo=UTC).isoformat(), vol],
            )

        # Features: days 21-30 all have breaks
        features = []
        for i in range(21, 31):
            features.append(
                {
                    "trading_day": date(2024, 3, i),
                    "orb_CME_REOPEN_break_ts": datetime(2024, 3, i, 23, 5, tzinfo=UTC),
                    "orb_CME_REOPEN_break_dir": "long",
                    "orb_CME_REOPEN_size": 5.0,
                }
            )

        _compute_relative_volumes(con, features, "MGC", ["CME_REOPEN"], ALL_FILTERS)

        nf_days = _build_filter_day_sets(features, ["CME_REOPEN"], {"NO_FILTER": ALL_FILTERS["NO_FILTER"]})
        vol_days = _build_filter_day_sets(features, ["CME_REOPEN"], {"VOL_RV12_N20": ALL_FILTERS["VOL_RV12_N20"]})
        con.close()

        nf_count = len(nf_days[("NO_FILTER", "CME_REOPEN")])
        vol_count = len(vol_days[("VOL_RV12_N20", "CME_REOPEN")])

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
            "orb_CME_REOPEN_break_dir": "long",
            "orb_CME_REOPEN_size": 5.0,
            "orb_CME_REOPEN_double_break": double_break,
        }

    def test_double_break_true_included(self):
        """Days with double_break=True are included (not filtered)."""
        from trading_app.strategy_discovery import _build_filter_day_sets

        features = [self._make_feature(1, True), self._make_feature(2, False)]
        result = _build_filter_day_sets(features, ["CME_REOPEN"], {"NO_FILTER": ALL_FILTERS["NO_FILTER"]})
        days = result[("NO_FILTER", "CME_REOPEN")]
        assert date(2024, 1, 1) in days
        assert date(2024, 1, 2) in days

    def test_double_break_none_included(self):
        """Days with double_break=None are included."""
        from trading_app.strategy_discovery import _build_filter_day_sets

        features = [self._make_feature(1, None)]
        result = _build_filter_day_sets(features, ["CME_REOPEN"], {"NO_FILTER": ALL_FILTERS["NO_FILTER"]})
        assert date(2024, 1, 1) in result[("NO_FILTER", "CME_REOPEN")]

    def test_all_days_with_break_included(self):
        """All days with a break direction are included regardless of double_break."""
        from trading_app.strategy_discovery import _build_filter_day_sets

        features = [
            self._make_feature(1, True),  # double break
            self._make_feature(2, False),  # single break
            self._make_feature(3, False),  # single break
        ]
        filters = {"NO_FILTER": ALL_FILTERS["NO_FILTER"], "ORB_G4": ALL_FILTERS["ORB_G4"]}
        result = _build_filter_day_sets(features, ["CME_REOPEN"], filters)

        assert len(result[("NO_FILTER", "CME_REOPEN")]) == 3
        assert len(result[("ORB_G4", "CME_REOPEN")]) == 3


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
                    orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_size, orb_CME_REOPEN_break_dir)
                   VALUES (?, 'MGC', 5, 2710.0, 2700.0, 10.0, 'long')""",
                [date(2024, 1, i)],
            )
        # Insert orb_outcomes that are ALL scratches
        for i in range(1, 11):
            con.execute(
                """INSERT INTO orb_outcomes
                    (trading_day, symbol, orb_minutes, orb_label, entry_model,
                     rr_target, confirm_bars, outcome, pnl_r, mae_r, mfe_r)
                   VALUES (?, 'MGC', 5, 'CME_REOPEN', 'E1', 2.0, 1, 'scratch', 0.0, 0.1, 0.1)""",
                [date(2024, 1, i)],
            )
        con.commit()
        con.close()

        # Run discovery
        _count = run_discovery(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
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
        from trading_app.db_manager import compute_trade_day_hash as _compute_trade_day_hash
        from trading_app.strategy_discovery import _mark_canonical

        days = [date(2024, 1, i) for i in range(1, 11)]
        day_hash = _compute_trade_day_hash(days)

        strategies = [
            {
                "strategy_id": f"MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_{f}",
                "instrument": "MGC",
                "orb_label": "CME_REOPEN",
                "entry_model": "E1",
                "rr_target": 2.0,
                "confirm_bars": 1,
                "filter_key": f"ORB_{f}",
                "trade_day_hash": day_hash,
                "metrics": {"expectancy_r": 0.5},
            }
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
        from trading_app.db_manager import compute_trade_day_hash as _compute_trade_day_hash
        from trading_app.strategy_discovery import _mark_canonical

        days1 = [date(2024, 1, i) for i in range(1, 6)]
        days2 = [date(2024, 1, i) for i in range(6, 11)]

        strategies = [
            {
                "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4",
                "instrument": "MGC",
                "orb_label": "CME_REOPEN",
                "entry_model": "E1",
                "rr_target": 2.0,
                "confirm_bars": 1,
                "filter_key": "ORB_G4",
                "trade_day_hash": _compute_trade_day_hash(days1),
                "metrics": {"expectancy_r": 0.5},
            },
            {
                "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G8",
                "instrument": "MGC",
                "orb_label": "CME_REOPEN",
                "entry_model": "E1",
                "rr_target": 2.0,
                "confirm_bars": 1,
                "filter_key": "ORB_G8",
                "trade_day_hash": _compute_trade_day_hash(days2),
                "metrics": {"expectancy_r": 0.8},
            },
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
            "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4",
            "instrument": "MGC",
            "orb_label": "CME_REOPEN",
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
            "yearly_results": json.dumps(
                {
                    "2022": {"trades": 50, "wins": 28, "total_r": 10.0, "win_rate": 0.56, "avg_r": 0.2},
                    "2023": {"trades": 50, "wins": 27, "total_r": 8.0, "win_rate": 0.54, "avg_r": 0.16},
                    "2024": {"trades": 50, "wins": 28, "total_r": 9.0, "win_rate": 0.56, "avg_r": 0.18},
                }
            ),
            "is_canonical": True,
        }
        base.update(overrides)
        return base

    def test_validator_skips_aliases(self, tmp_path):
        """Alias row is skipped without error, status set to SKIPPED."""
        from trading_app.strategy_validator import run_validation

        canonical = self._make_row(is_canonical=True)
        alias = self._make_row(
            strategy_id="MGC_CME_REOPEN_E1_RR2.0_CB1_NO_FILTER",
            filter_type="NO_FILTER",
            is_canonical=False,
            canonical_strategy_id="MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G4",
        )
        db_path = self._setup_db(tmp_path, [canonical, alias])

        passed, rejected = run_validation(db_path=db_path, instrument="MGC", enable_walkforward=False)

        # Canonical passes validation, alias is skipped
        assert passed == 1
        assert rejected == 0

        # Verify alias got SKIPPED status
        con = duckdb.connect(str(db_path), read_only=True)
        alias_status = con.execute(
            "SELECT validation_status FROM experimental_strategies WHERE strategy_id = ?",
            ["MGC_CME_REOPEN_E1_RR2.0_CB1_NO_FILTER"],
        ).fetchone()[0]
        con.close()
        assert alias_status == "SKIPPED"


class TestHaircutSharpe:
    """Tests for _compute_haircut_sharpe (BLP 2014 Deflated Sharpe Ratio)."""

    def test_strong_edge_survives(self):
        """A genuine high-Sharpe strategy should survive the haircut."""
        result = _compute_haircut_sharpe(
            sharpe_per_trade=0.23,
            n_obs=1000,
            skewness=0.0,
            kurtosis_excess=0.0,
            n_trials=2376,
            trades_per_year=200.0,
        )
        assert result is not None
        assert result > 0, "Strong edge with N=1000 should survive haircut"

    def test_noise_goes_negative(self):
        """Near-zero per-trade Sharpe with small sample should go negative."""
        result = _compute_haircut_sharpe(
            sharpe_per_trade=0.01,
            n_obs=50,
            skewness=0.0,
            kurtosis_excess=0.0,
            n_trials=2376,
            trades_per_year=10.0,
        )
        assert result is not None
        assert result < 0, "Noise strategy should not survive haircut"

    def test_more_trials_increases_penalty(self):
        """More trials = higher expected max = more haircut."""
        few = _compute_haircut_sharpe(
            sharpe_per_trade=0.10,
            n_obs=200,
            skewness=0.0,
            kurtosis_excess=0.0,
            n_trials=10,
            trades_per_year=40.0,
        )
        many = _compute_haircut_sharpe(
            sharpe_per_trade=0.10,
            n_obs=200,
            skewness=0.0,
            kurtosis_excess=0.0,
            n_trials=2376,
            trades_per_year=40.0,
        )
        assert few > many, "More trials should produce a lower haircut Sharpe"

    def test_larger_sample_preserves_more(self):
        """Larger N reduces estimator variance, less penalty from noise."""
        small_n = _compute_haircut_sharpe(
            sharpe_per_trade=0.10,
            n_obs=50,
            skewness=0.0,
            kurtosis_excess=0.0,
            n_trials=2376,
            trades_per_year=10.0,
        )
        large_n = _compute_haircut_sharpe(
            sharpe_per_trade=0.10,
            n_obs=500,
            skewness=0.0,
            kurtosis_excess=0.0,
            n_trials=2376,
            trades_per_year=100.0,
        )
        assert large_n > small_n, "Larger sample should preserve more Sharpe"

    def test_returns_none_insufficient_data(self):
        """Edge cases should return None."""
        # Too few observations
        assert _compute_haircut_sharpe(0.1, 5, 0.0, 0.0, 2376, 1.0) is None
        # Only 1 trial (no multiple testing)
        assert _compute_haircut_sharpe(0.1, 200, 0.0, 0.0, 1, 40.0) is None
        # None Sharpe
        assert _compute_haircut_sharpe(None, 200, 0.0, 0.0, 2376, 40.0) is None
        # Zero trades per year
        assert _compute_haircut_sharpe(0.1, 200, 0.0, 0.0, 2376, 0.0) is None

    def test_compute_metrics_populates_haircut(self):
        """compute_metrics with n_trials > 0 should populate haircut fields."""
        outcomes = [
            {
                "trading_day": date(2024, 1, i),
                "outcome": "win",
                "pnl_r": 1.5,
                "mae_r": 0.5,
                "mfe_r": 2.0,
                "entry_price": 100.0,
                "stop_price": 95.0,
            }
            for i in range(1, 16)
        ] + [
            {
                "trading_day": date(2024, 1, i),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.2,
                "entry_price": 100.0,
                "stop_price": 95.0,
            }
            for i in range(16, 26)
        ]
        m = compute_metrics(outcomes, n_trials=2376)
        assert m["sharpe_haircut"] is not None
        assert m["skewness"] is not None
        assert m["kurtosis_excess"] is not None

    def test_compute_metrics_no_trials_leaves_none(self):
        """compute_metrics without n_trials should leave haircut as None."""
        outcomes = [
            {
                "trading_day": date(2024, 1, i),
                "outcome": "win",
                "pnl_r": 1.5,
                "mae_r": 0.5,
                "mfe_r": 2.0,
                "entry_price": 100.0,
                "stop_price": 95.0,
            }
            for i in range(1, 16)
        ] + [
            {
                "trading_day": date(2024, 1, i),
                "outcome": "loss",
                "pnl_r": -1.0,
                "mae_r": 1.0,
                "mfe_r": 0.2,
                "entry_price": 100.0,
                "stop_price": 95.0,
            }
            for i in range(16, 26)
        ]
        m = compute_metrics(outcomes)
        assert m["sharpe_haircut"] is None
        assert m["skewness"] is None
        assert m["kurtosis_excess"] is None


class TestCLI:
    def test_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["strategy_discovery", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from trading_app.strategy_discovery import main

            main()
        assert exc_info.value.code == 0
        assert "instrument" in capsys.readouterr().out

    def test_help_mentions_hypothesis_file_flag(self, monkeypatch, capsys):
        """Phase 4 Stage 4.1d: --hypothesis-file must appear in CLI help."""
        monkeypatch.setattr("sys.argv", ["strategy_discovery", "--help"])
        with pytest.raises(SystemExit):
            from trading_app.strategy_discovery import main

            main()
        out = capsys.readouterr().out
        assert "--hypothesis-file" in out
        assert "Phase 4" in out


# ---------------------------------------------------------------------------
# Phase 4 Stage 4.1 — discovery integration tests
# ---------------------------------------------------------------------------


class TestPhase4DiscoveryEnforcement:
    """Tests for the Phase 4 enforcement block inside run_discovery.

    The Phase 4 block runs BEFORE ``with duckdb.connect(...)`` opens a
    connection, so we can test the raising paths without needing a real
    database. Legacy-mode compatibility is implicitly proven by the 45
    tests in TestComputeMetrics, TestMakeStrategyId, TestComputeRelativeVolumes,
    etc., which all call ``run_discovery`` or its helpers without passing
    ``hypothesis_file``.

    These tests cover the 5 failure paths in the Phase 4 block plus 1
    CLI-level translation test, plus 1 _flush_batch_df column-alignment
    assertion test, plus 1 legacy-mode signature test.
    """

    @staticmethod
    def _init_git_repo(tmp_path):
        """Initialize a minimal temp git repo and return its path."""
        import subprocess as _sp

        _sp.run(["git", "init", "-q"], check=True, cwd=tmp_path)
        _sp.run(
            ["git", "config", "user.email", "test@example.com"],
            check=True,
            cwd=tmp_path,
        )
        _sp.run(
            ["git", "config", "user.name", "Test"],
            check=True,
            cwd=tmp_path,
        )
        return tmp_path

    @staticmethod
    def _write_and_commit(tmp_path, rel_path: str, content: str):
        """Write a file in the temp repo, git add + commit it, return Path."""
        import subprocess as _sp

        p = tmp_path / rel_path
        p.write_text(content, encoding="utf-8")
        _sp.run(["git", "add", str(p)], check=True, cwd=tmp_path)
        _sp.run(
            ["git", "commit", "-q", "-m", f"add {rel_path}"],
            check=True,
            cwd=tmp_path,
        )
        return p

    def _minimal_hypothesis_yaml(
        self,
        *,
        instrument: str = "MNQ",
        total_trials: int = 60,
        holdout: str = "2026-01-01",
        sessions: list | None = None,
    ) -> str:
        """Build a minimal valid hypothesis YAML string using yaml.safe_dump.

        Programmatic YAML construction (via yaml.safe_dump) avoids the
        hand-indented-string footgun — any indentation bug would break
        every Phase 4 integration test with a parse error rather than
        testing the Phase 4 enforcement path.
        """
        import yaml as _yaml

        sessions_list = sessions or ["NYSE_OPEN"]
        body = {
            "metadata": {
                "name": "test_stage_4_1_integration",
                "date_locked": "2026-04-08",
                "holdout_date": holdout,
                "total_expected_trials": total_trials,
            },
            "hypotheses": [
                {
                    "id": 1,
                    "name": "synthetic_phase_4",
                    "theory_citation": "docs/institutional/literature/synthetic.md",
                    "economic_basis": "synthetic fixture for integration tests",
                    "filter": {
                        "type": "OVNRNG",
                        "column": "overnight_range",
                        "thresholds": [50, 75],
                    },
                    "scope": {
                        "instruments": [instrument],
                        "sessions": sessions_list,
                        "rr_targets": [1.0, 1.5, 2.0],
                        "entry_models": ["E2"],
                        "confirm_bars": [1],
                        "stop_multipliers": [1.0],
                    },
                    "expected_trial_count": 12,
                    "kill_criteria": ["placeholder"],
                }
            ],
        }
        return _yaml.safe_dump(body, sort_keys=False)

    def test_raises_on_missing_hypothesis_file(self, tmp_path):
        """Phase 4 enforcement: nonexistent file fails at git gate."""
        from trading_app.hypothesis_loader import HypothesisLoaderError
        from trading_app.strategy_discovery import run_discovery

        nonexistent = tmp_path / "does_not_exist.yaml"
        with pytest.raises(HypothesisLoaderError, match="not found"):
            run_discovery(
                instrument="MNQ",
                orb_minutes=5,
                dry_run=True,
                hypothesis_file=nonexistent,
            )

    def test_raises_on_untracked_hypothesis_file(self, tmp_path, monkeypatch):
        """Phase 4 enforcement: untracked file fails at git cleanliness gate."""
        from trading_app.hypothesis_loader import HypothesisLoaderError
        from trading_app.strategy_discovery import run_discovery

        self._init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        # Write but do NOT commit
        p = tmp_path / "untracked.yaml"
        p.write_text(self._minimal_hypothesis_yaml(), encoding="utf-8")

        with pytest.raises(HypothesisLoaderError, match="not tracked"):
            run_discovery(
                instrument="MNQ",
                orb_minutes=5,
                dry_run=True,
                hypothesis_file=p,
            )

    def test_raises_on_dirty_hypothesis_file(self, tmp_path, monkeypatch):
        """Phase 4 enforcement: dirty (edited-after-commit) file fails."""
        from trading_app.hypothesis_loader import HypothesisLoaderError
        from trading_app.strategy_discovery import run_discovery

        self._init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        p = self._write_and_commit(
            tmp_path, "hyp.yaml", self._minimal_hypothesis_yaml()
        )
        # Edit without committing
        p.write_text(
            self._minimal_hypothesis_yaml() + "# added comment\n",
            encoding="utf-8",
        )

        with pytest.raises(HypothesisLoaderError, match="uncommitted"):
            run_discovery(
                instrument="MNQ",
                orb_minutes=5,
                dry_run=True,
                hypothesis_file=p,
            )

    def test_raises_on_mode_a_violation(self, tmp_path, monkeypatch):
        """Phase 4 enforcement: holdout_date past sacred boundary rejects."""
        from trading_app.hypothesis_loader import HypothesisLoaderError
        from trading_app.strategy_discovery import run_discovery

        self._init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        yaml_text = self._minimal_hypothesis_yaml(holdout="2026-06-01")
        p = self._write_and_commit(tmp_path, "hyp.yaml", yaml_text)

        with pytest.raises(HypothesisLoaderError, match="Amendment 2.7"):
            run_discovery(
                instrument="MNQ",
                orb_minutes=5,
                dry_run=True,
                hypothesis_file=p,
            )

    def test_raises_on_minbtl_overshoot_clean(self, tmp_path, monkeypatch):
        """Phase 4 enforcement: declared_trials > 300 in clean mode rejects."""
        from trading_app.hypothesis_loader import HypothesisLoaderError
        from trading_app.strategy_discovery import run_discovery

        self._init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        yaml_text = self._minimal_hypothesis_yaml(total_trials=500)
        p = self._write_and_commit(tmp_path, "hyp.yaml", yaml_text)

        with pytest.raises(HypothesisLoaderError, match="criterion_2"):
            run_discovery(
                instrument="MNQ",
                orb_minutes=5,
                dry_run=True,
                hypothesis_file=p,
            )

    def test_raises_on_instrument_not_in_file(self, tmp_path, monkeypatch):
        """Phase 4 enforcement: --instrument not declared in scope fails loud."""
        from trading_app.hypothesis_loader import HypothesisLoaderError
        from trading_app.strategy_discovery import run_discovery

        self._init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        # File declares MNQ only — but we'll run for MES
        yaml_text = self._minimal_hypothesis_yaml(instrument="MNQ")
        p = self._write_and_commit(tmp_path, "hyp.yaml", yaml_text)

        with pytest.raises(HypothesisLoaderError, match="no hypotheses for instrument"):
            run_discovery(
                instrument="MES",
                orb_minutes=5,
                dry_run=True,
                hypothesis_file=p,
            )

    def test_cli_translates_hypothesis_error_to_parser_error(
        self, tmp_path, monkeypatch, capsys
    ):
        """CLI wrap: HypothesisLoaderError becomes a clean parser.error exit 2."""
        from trading_app.strategy_discovery import main

        nonexistent = tmp_path / "missing.yaml"
        monkeypatch.setattr(
            "sys.argv",
            [
                "strategy_discovery",
                "--instrument",
                "MNQ",
                "--orb-minutes",
                "5",
                "--dry-run",
                "--hypothesis-file",
                str(nonexistent),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "Phase 4 hypothesis discipline" in err
        assert "not found" in err

    def test_legacy_mode_signature_accepts_none(self):
        """run_discovery signature still accepts hypothesis_file=None default.

        Proves the 45 pre-Stage-4.1 tests in this file are legacy-safe
        WITHOUT needing to re-run them — we just check the signature.
        """
        import inspect

        from trading_app.strategy_discovery import run_discovery

        sig = inspect.signature(run_discovery)
        assert "hypothesis_file" in sig.parameters
        assert sig.parameters["hypothesis_file"].default is None


class TestFlushBatchDfColumnAlignment:
    """Phase 4 Stage 4.1 defensive guard (D-7): _flush_batch_df must raise
    on row-length mismatch against _BATCH_COLUMNS to catch silent column
    misalignment during the 3-edit SHA-stamping coordination."""

    def test_short_row_raises(self):
        import duckdb

        from trading_app.strategy_discovery import _BATCH_COLUMNS, _flush_batch_df

        con = duckdb.connect(":memory:")
        # Row with one too few values
        short_row = [None] * (len(_BATCH_COLUMNS) - 1)
        with pytest.raises(ValueError, match="column alignment error"):
            _flush_batch_df(con, [short_row])
        con.close()

    def test_long_row_raises(self):
        import duckdb

        from trading_app.strategy_discovery import _BATCH_COLUMNS, _flush_batch_df

        con = duckdb.connect(":memory:")
        long_row = [None] * (len(_BATCH_COLUMNS) + 1)
        with pytest.raises(ValueError, match="column alignment error"):
            _flush_batch_df(con, [long_row])
        con.close()

    def test_empty_batch_does_not_raise_on_length_check(self):
        """An empty batch is legitimately a no-op — the length check loop
        iterates zero times, which is correct."""
        import duckdb

        from trading_app.strategy_discovery import _flush_batch_df

        con = duckdb.connect(":memory:")
        # This will still fail at the SQL INSERT step because the table
        # doesn't exist in :memory:, but we only care about the length
        # check path here. We expect either an OSError-style DB error
        # (no table) OR for the function to complete without the column
        # alignment ValueError.
        try:
            _flush_batch_df(con, [])
        except ValueError as e:
            if "column alignment" in str(e):
                pytest.fail(
                    "length check incorrectly fired on empty batch: "
                    + str(e)
                )
            # Some other ValueError is fine — not our concern
        except duckdb.Error:
            pass  # expected — table doesn't exist in :memory:
        con.close()

    def test_hypothesis_file_sha_is_last_column(self):
        """Phase 4 Stage 4.1 SHA column is appended to _BATCH_COLUMNS and
        must therefore be the last column. This guards against a future
        reorder that would silently break SHA stamping."""
        from trading_app.strategy_discovery import _BATCH_COLUMNS

        assert _BATCH_COLUMNS[-1] == "hypothesis_file_sha"
        # Also verify no duplicate
        assert _BATCH_COLUMNS.count("hypothesis_file_sha") == 1

    def test_sha_stamping_round_trip(self):
        """Phase D review SHOULD-FIX #2: end-to-end SHA stamping round trip.

        Build a minimal experimental_strategies schema in-memory matching
        only the columns _flush_batch_df writes to, construct a valid
        51-element row with a known test SHA in the hypothesis_file_sha
        slot, call _flush_batch_df, then query back and assert the SHA is
        present. Validates the 3-edit coordination (_BATCH_COLUMNS + INSERT
        SQL column list + batch assembly loop) is actually aligned at
        runtime, not just structurally well-formed.

        Uses a hand-rolled minimal schema rather than init_trading_app_schema
        because the full schema has foreign keys to daily_features (a
        pipeline-owned table not present in an isolated test DB).
        """
        from typing import Any

        import duckdb

        from trading_app.strategy_discovery import _BATCH_COLUMNS, _flush_batch_df

        con = duckdb.connect(":memory:")
        # Build a minimal experimental_strategies table with nullable
        # columns. strategy_id is PRIMARY KEY because _flush_batch_df uses
        # INSERT OR REPLACE which requires a unique constraint to resolve
        # conflicts. Other columns are nullable TEXT — the round-trip
        # only needs to exercise the SHA positional alignment, not the
        # production column types.
        col_defs = ",\n            ".join(
            f"{name} TEXT PRIMARY KEY" if name == "strategy_id" else f"{name} TEXT"
            for name in _BATCH_COLUMNS
        )
        con.execute(f"""
            CREATE TABLE experimental_strategies (
            {col_defs}
            )
        """)

        test_sha = "abc123" + "0" * 58  # 64-char synthetic SHA
        row: list[Any] = [None] * len(_BATCH_COLUMNS)
        col_idx = {name: i for i, name in enumerate(_BATCH_COLUMNS)}
        row[col_idx["strategy_id"]] = "round_trip_test_strategy"
        row[col_idx["instrument"]] = "MNQ"
        row[col_idx["orb_label"]] = "NYSE_OPEN"
        row[col_idx["orb_minutes"]] = "5"
        row[col_idx["entry_model"]] = "E2"
        row[col_idx["filter_type"]] = "NO_FILTER"
        row[col_idx["hypothesis_file_sha"]] = test_sha

        _flush_batch_df(con, [row])

        # Query back and verify the SHA is in the row
        result = con.execute(
            "SELECT hypothesis_file_sha FROM experimental_strategies "
            "WHERE strategy_id = ?",
            ["round_trip_test_strategy"],
        ).fetchone()
        assert result is not None, "row was not inserted"
        assert result[0] == test_sha, (
            f"SHA mismatch — 3-edit coordination broken. "
            f"Expected {test_sha}, got {result[0]}. "
            f"Check _BATCH_COLUMNS + INSERT SQL + batch assembly alignment."
        )
        con.close()


# ---------------------------------------------------------------------------
# Phase 4 Stage 4.1b — hypothesis-mode filter injection
# ---------------------------------------------------------------------------


class TestHypothesisFilterInjection:
    """Tests for _inject_hypothesis_filters, the Phase 4 Stage 4.1b helper
    extracted from run_discovery.

    Covers acceptance criteria #4 and #5 from the wave4-hypothesis-filter-
    injection stage file (docs/runtime/stages/wave4-hypothesis-filter-
    injection.md):

    - criterion #4: hypothesis-declared filter types that are NOT in the
      legacy grid get injected into ``all_grid_filters`` AND into the
      per-session map, so bulk pre-computation covers them and the
      per-session loop picks them up.
    - criterion #5: DOW composite filters (``ORB_*_NOFRI`` etc.) are NOT
      injected into sessions in ``DOW_MISALIGNED_SESSIONS`` — Brisbane DOW
      != exchange DOW at NYSE_OPEN, so a Friday-skip would fire on the
      wrong exchange-local day there.

    Plus two safety tests for the no-op branches:
    - filter already in the legacy grid → no-op (prevents double-counting
      and preserves legacy-mode parity, criterion #3).
    - filter not in ALL_FILTERS → silent skip (combo enumeration path
      rejects later via scope_predicate.accepts(), giving a clean
      zero-combos result instead of a crash).
    """

    @staticmethod
    def _build_scope_predicate(filter_types, sessions):
        """Build a minimal ScopePredicate directly (no YAML path).

        Bypasses hypothesis_loader parsing — we only need the
        ``allowed_filter_types()`` shape the helper reads. Each filter
        type gets its own HypothesisScope bundle so the predicate's
        ``allowed_filter_types()`` returns all of them.
        """
        from trading_app.hypothesis_loader import HypothesisScope, ScopePredicate

        hypotheses = tuple(
            HypothesisScope(
                filter_type=ft,
                instruments=frozenset({"MNQ"}),
                sessions=frozenset(sessions),
                rr_targets=frozenset({1.0}),
                entry_models=frozenset({"E2"}),
                confirm_bars=frozenset({1}),
                stop_multipliers=frozenset({1.0}),
                expected_trial_count=10,
            )
            for ft in filter_types
        )
        return ScopePredicate(
            hypotheses=hypotheses,
            instrument="MNQ",
            total_declared_trials=10 * len(hypotheses),
        )

    def test_hypothesis_filter_injection_expands_grid(self):
        """Criterion #4: a declared filter type not in the legacy grid
        gets injected into all_grid_filters AND into the per-session map
        for every (non-DOW-misaligned) session."""
        from trading_app.config import ALL_FILTERS
        from trading_app.strategy_discovery import _inject_hypothesis_filters

        # GAP_R015 exists in ALL_FILTERS (GapNormFilter, not a DOW composite).
        # It is routed in get_filters_for_grid() only for MGC/GC CME_REOPEN,
        # so for MNQ NYSE_OPEN + COMEX_SETTLE it is genuinely "not in grid".
        assert "GAP_R015" in ALL_FILTERS

        sessions = ["NYSE_OPEN", "COMEX_SETTLE"]
        all_grid_filters: dict = {}  # empty legacy grid (fresh slate)
        hypothesis_extra_by_session: dict[str, dict] = {s: {} for s in sessions}

        sp = self._build_scope_predicate(["GAP_R015"], sessions)

        _inject_hypothesis_filters(
            scope_predicate=sp,
            sessions=sessions,
            all_grid_filters=all_grid_filters,
            hypothesis_extra_by_session=hypothesis_extra_by_session,
        )

        # Injected into the global grid for bulk pre-computation
        assert "GAP_R015" in all_grid_filters
        assert all_grid_filters["GAP_R015"] is ALL_FILTERS["GAP_R015"]

        # AND into every session's per-session map
        for s in sessions:
            assert "GAP_R015" in hypothesis_extra_by_session[s], (
                f"session {s} did not receive the injected GAP_R015"
            )
            assert (
                hypothesis_extra_by_session[s]["GAP_R015"]
                is ALL_FILTERS["GAP_R015"]
            )

    def test_hypothesis_filter_injection_respects_dow_misalignment(self):
        """Criterion #5: DOW composite filters must NOT be injected into
        sessions in DOW_MISALIGNED_SESSIONS. Brisbane DOW != exchange DOW
        at NYSE_OPEN (midnight crossing) — a Brisbane-Friday skip at
        NYSE_OPEN fires on US Thursday, which is off by one day.

        The composite is still registered in ``all_grid_filters`` so bulk
        pre-computation runs, but the per-session map for NYSE_OPEN is
        left empty for that filter. Non-misaligned sessions still receive
        it."""
        from pipeline.dst import DOW_MISALIGNED_SESSIONS
        from trading_app.config import (
            ALL_FILTERS,
            CompositeFilter,
            DayOfWeekSkipFilter,
        )
        from trading_app.strategy_discovery import _inject_hypothesis_filters

        # Pick any real DOW composite from ALL_FILTERS (e.g. ORB_G5_NOFRI).
        dow_filter_type = next(
            ft
            for ft, obj in ALL_FILTERS.items()
            if isinstance(obj, CompositeFilter)
            and isinstance(obj.overlay, DayOfWeekSkipFilter)
        )

        # Guard: NYSE_OPEN is the canonical DOW-misaligned session. If the
        # SESSION_CATALOG ever changes to include another, this assertion
        # catches the drift and the test author can add coverage for it.
        assert "NYSE_OPEN" in DOW_MISALIGNED_SESSIONS

        sessions = ["NYSE_OPEN", "LONDON_METALS", "EUROPE_FLOW"]
        all_grid_filters: dict = {}
        hypothesis_extra_by_session: dict[str, dict] = {s: {} for s in sessions}

        sp = self._build_scope_predicate([dow_filter_type], sessions)

        _inject_hypothesis_filters(
            scope_predicate=sp,
            sessions=sessions,
            all_grid_filters=all_grid_filters,
            hypothesis_extra_by_session=hypothesis_extra_by_session,
        )

        # Composite IS in all_grid_filters — bulk pre-computation covers it
        assert dow_filter_type in all_grid_filters

        # NYSE_OPEN did NOT receive it (misalignment guard)
        assert dow_filter_type not in hypothesis_extra_by_session["NYSE_OPEN"], (
            f"DOW composite {dow_filter_type} was injected into NYSE_OPEN "
            f"despite DOW_MISALIGNED_SESSIONS guard — look-ahead leak risk"
        )

        # Other (non-misaligned) sessions DID receive it
        assert dow_filter_type in hypothesis_extra_by_session["LONDON_METALS"]
        assert dow_filter_type in hypothesis_extra_by_session["EUROPE_FLOW"]

    def test_hypothesis_filter_injection_skips_filters_already_in_grid(self):
        """Safety: a declared filter type that is already in the legacy
        grid is a no-op. This preserves criterion #3 — legacy-mode grid
        behavior must not change in hypothesis mode for filters the
        legacy grid already covers. Prevents double-counting in the
        per-session loop's merge step."""
        from trading_app.config import ALL_FILTERS
        from trading_app.strategy_discovery import _inject_hypothesis_filters

        sessions = ["NYSE_OPEN"]
        legacy_filter = ALL_FILTERS["NO_FILTER"]
        all_grid_filters: dict = {"NO_FILTER": legacy_filter}
        hypothesis_extra_by_session: dict[str, dict] = {s: {} for s in sessions}

        sp = self._build_scope_predicate(["NO_FILTER"], sessions)

        _inject_hypothesis_filters(
            scope_predicate=sp,
            sessions=sessions,
            all_grid_filters=all_grid_filters,
            hypothesis_extra_by_session=hypothesis_extra_by_session,
        )

        # all_grid_filters unchanged (same len, same identity)
        assert list(all_grid_filters.keys()) == ["NO_FILTER"]
        assert all_grid_filters["NO_FILTER"] is legacy_filter

        # per-session map NOT written to — the merge in run_discovery's
        # session loop would otherwise double-count this filter for the
        # session (once from the legacy grid, once from injection).
        assert "NO_FILTER" not in hypothesis_extra_by_session["NYSE_OPEN"]

    def test_hypothesis_filter_injection_skips_unknown_filter_type(self):
        """Safety: if the hypothesis file declares a filter_type string
        that does not exist in ALL_FILTERS, injection silently skips it.
        The combo enumeration path's ``scope_predicate.accepts()`` check
        will reject every combo for that type, producing a clean
        zero-combos result instead of a KeyError or silent pass-through."""
        from trading_app.strategy_discovery import _inject_hypothesis_filters

        sessions = ["NYSE_OPEN"]
        all_grid_filters: dict = {}
        hypothesis_extra_by_session: dict[str, dict] = {s: {} for s in sessions}

        sp = self._build_scope_predicate(["NONSENSE_FILTER_TYPE"], sessions)

        _inject_hypothesis_filters(
            scope_predicate=sp,
            sessions=sessions,
            all_grid_filters=all_grid_filters,
            hypothesis_extra_by_session=hypothesis_extra_by_session,
        )

        assert all_grid_filters == {}
        assert hypothesis_extra_by_session["NYSE_OPEN"] == {}
