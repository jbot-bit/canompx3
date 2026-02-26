"""Tests for trading_app.live_config — live portfolio configuration."""

import pytest
import duckdb
from pathlib import Path

from trading_app.live_config import (
    LiveStrategySpec,
    LIVE_PORTFOLIO,
    _load_best_regime_variant,
    _load_best_experimental_variant,
)


@pytest.fixture
def live_config_db(tmp_path):
    """Create temp DB with validated_setups and experimental_strategies tables."""
    db_path = tmp_path / "live_test.db"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id VARCHAR,
            instrument VARCHAR,
            orb_label VARCHAR,
            entry_model VARCHAR,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            filter_type VARCHAR,
            expectancy_r DOUBLE,
            win_rate DOUBLE,
            sample_size INTEGER,
            sharpe_ratio DOUBLE,
            max_drawdown_r DOUBLE,
            status VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE experimental_strategies (
            strategy_id VARCHAR,
            instrument VARCHAR,
            orb_label VARCHAR,
            entry_model VARCHAR,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            filter_type VARCHAR,
            expectancy_r DOUBLE,
            win_rate DOUBLE,
            sample_size INTEGER,
            sharpe_ratio DOUBLE,
            max_drawdown_r DOUBLE,
            median_risk_points DOUBLE
        )
    """)
    con.close()
    return db_path


class TestLiveStrategySpec:
    def test_spec_is_frozen(self):
        spec = LiveStrategySpec("fam1", "core", "1000", "E1", "ORB_G4", None)
        with pytest.raises(AttributeError):
            spec.tier = "regime"

    def test_spec_fields(self):
        spec = LiveStrategySpec("fam1", "core", "1000", "E1", "ORB_G4", "rolling")
        assert spec.family_id == "fam1"
        assert spec.regime_gate == "rolling"


class TestLivePortfolio:
    def test_portfolio_not_empty(self):
        assert len(LIVE_PORTFOLIO) > 0

    def test_all_entries_are_specs(self):
        for spec in LIVE_PORTFOLIO:
            assert isinstance(spec, LiveStrategySpec)

    def test_tiers_are_valid(self):
        valid_tiers = {"core", "hot", "regime"}
        for spec in LIVE_PORTFOLIO:
            assert spec.tier in valid_tiers, f"{spec.family_id} has invalid tier: {spec.tier}"


class TestLoadBestRegimeVariant:
    def test_found(self, live_config_db):
        """Matching active strategy returns dict."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO validated_setups VALUES (
                'MGC_1000_E1_RR2.0_CB1_ORB_G4', 'MGC', '1000', 'E1',
                2.0, 1, 'ORB_G4', 0.35, 0.52, 150, 1.2, 3.5, 'active'
            )
        """)
        con.execute("""
            INSERT INTO experimental_strategies VALUES (
                'MGC_1000_E1_RR2.0_CB1_ORB_G4', 'MGC', '1000', 'E1',
                2.0, 1, 'ORB_G4', 0.35, 0.52, 150, 1.2, 3.5, 4.2
            )
        """)
        con.close()

        result = _load_best_regime_variant(
            live_config_db, "MGC", "1000", "E1", "ORB_G4"
        )
        assert result is not None
        assert result["strategy_id"] == "MGC_1000_E1_RR2.0_CB1_ORB_G4"
        assert result["sharpe_ratio"] == 1.2

    def test_not_found(self, live_config_db):
        """No matching strategy returns None."""
        result = _load_best_regime_variant(
            live_config_db, "MGC", "9999", "E1", "ORB_G4"
        )
        assert result is None

    def test_inactive_filtered(self, live_config_db):
        """Inactive strategy is not returned."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO validated_setups VALUES (
                'test_inactive', 'MGC', '1000', 'E1',
                2.0, 1, 'ORB_G4', 0.35, 0.52, 150, 1.2, 3.5, 'purged'
            )
        """)
        con.close()

        result = _load_best_regime_variant(
            live_config_db, "MGC", "1000", "E1", "ORB_G4"
        )
        assert result is None

    def test_best_sharpe_selected(self, live_config_db):
        """When multiple candidates exist, highest Sharpe is returned."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('low_sharpe', 'MGC', '1000', 'E1', 2.0, 1, 'ORB_G4', 0.30, 0.50, 100, 0.8, 4.0, 'active'),
                ('high_sharpe', 'MGC', '1000', 'E1', 1.5, 2, 'ORB_G4', 0.40, 0.55, 120, 1.5, 3.0, 'active')
        """)
        con.close()

        result = _load_best_regime_variant(
            live_config_db, "MGC", "1000", "E1", "ORB_G4"
        )
        assert result["strategy_id"] == "high_sharpe"


class TestLoadBestExperimentalVariant:
    def test_found(self, live_config_db):
        """Matching experimental strategy with positive ExpR returns dict."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO experimental_strategies VALUES (
                'exp_test', 'MGC', '1000', 'E2', 2.0, 1, 'ORB_G5',
                0.25, 0.51, 200, 1.0, 4.0, 3.5
            )
        """)
        con.close()

        result = _load_best_experimental_variant(
            live_config_db, "MGC", "1000", "E2", "ORB_G5"
        )
        assert result is not None
        assert result["expectancy_r"] == 0.25

    def test_negative_expr_filtered(self, live_config_db):
        """Negative expectancy is filtered out."""
        con = duckdb.connect(str(live_config_db))
        con.execute("""
            INSERT INTO experimental_strategies VALUES (
                'neg_test', 'MGC', '1000', 'E2', 2.0, 1, 'ORB_G5',
                -0.10, 0.45, 200, -0.5, 5.0, 3.5
            )
        """)
        con.close()

        result = _load_best_experimental_variant(
            live_config_db, "MGC", "1000", "E2", "ORB_G5"
        )
        assert result is None

    def test_not_found(self, live_config_db):
        result = _load_best_experimental_variant(
            live_config_db, "MNQ", "0900", "E1", "ORB_G4"
        )
        assert result is None
