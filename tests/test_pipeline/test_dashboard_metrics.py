"""Focused tests for pipeline.dashboard strategy metrics."""

from pathlib import Path

import duckdb

from pipeline.dashboard import collect_strategy_metrics
from trading_app.db_manager import init_trading_app_schema


def _setup_dashboard_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "dashboard.db"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            bar_count_1m INTEGER,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)
    con.close()

    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path))
    con.execute("""
        INSERT INTO validated_setups (
            strategy_id, instrument, orb_label, orb_minutes, rr_target,
            confirm_bars, entry_model, filter_type, sample_size,
            win_rate, expectancy_r, years_tested, all_years_positive,
            stress_test_passed, sharpe_ratio, max_drawdown_r, status, deployment_scope
        )
        VALUES
            ('mnq_live', 'MNQ', 'US_DATA_830', 5, 2.0, 1, 'E2', 'ORB_G5',
             120, 0.54, 0.31, 5, TRUE, TRUE, 0.40, 4.5, 'active', 'deployable'),
            ('gc_research', 'GC', 'US_DATA_830', 5, 2.0, 1, 'E2', 'ORB_G5',
             120, 0.54, 0.31, 5, TRUE, TRUE, 0.40, 4.5, 'active', 'non_deployable')
    """)
    con.execute("""
        INSERT INTO experimental_strategies (
            strategy_id, instrument, orb_label, orb_minutes, rr_target,
            confirm_bars, entry_model
        )
        VALUES ('exp_1', 'MNQ', 'US_DATA_830', 5, 2.0, 1, 'E2')
    """)
    con.execute("""
        CREATE OR REPLACE VIEW deployable_validated_setups AS
        SELECT *
        FROM validated_setups
        WHERE LOWER(status) = 'active'
          AND LOWER(COALESCE(deployment_scope, 'deployable')) = 'deployable'
    """)
    con.commit()
    con.close()
    return db_path


class TestCollectStrategyMetrics:
    def test_excludes_non_deployable_active_rows(self, tmp_path):
        db_path = _setup_dashboard_db(tmp_path)

        metrics = collect_strategy_metrics(db_path)

        assert metrics["has_data"] is True
        assert metrics["validated_count"] == 1
        assert metrics["experimental_count"] == 1
        assert len(metrics["top_strategies"]) == 1
        assert metrics["top_strategies"][0]["orb"] == "US_DATA_830"
