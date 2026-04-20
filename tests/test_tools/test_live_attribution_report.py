import json
from pathlib import Path

import duckdb

from scripts.tools.live_attribution_report import build_report, render_report


def _write_allocation(path: Path) -> None:
    payload = {
        "rebalance_date": "2026-04-18",
        "profile_id": "topstep_50k_mnq_auto",
        "lanes": [
            {
                "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
                "instrument": "MNQ",
                "orb_label": "NYSE_OPEN",
                "rr_target": 1.0,
                "filter_type": "COST_LT12",
                "annual_r": 29.0,
                "trailing_expr": 0.12,
                "session_regime": "HOT",
                "status": "DEPLOY",
            },
            {
                "strategy_id": "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12",
                "instrument": "MNQ",
                "orb_label": "TOKYO_OPEN",
                "rr_target": 1.5,
                "filter_type": "COST_LT12",
                "annual_r": 20.3,
                "trailing_expr": 0.0934,
                "session_regime": "HOT",
                "status": "DEPLOY",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_gold_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE validated_setups (
            strategy_id VARCHAR,
            expectancy_r DOUBLE,
            win_rate DOUBLE,
            sample_size INTEGER,
            status VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE paper_trades (
            trading_day DATE,
            orb_label VARCHAR,
            entry_time TIMESTAMPTZ,
            direction VARCHAR,
            entry_price DOUBLE,
            stop_price DOUBLE,
            target_price DOUBLE,
            exit_price DOUBLE,
            exit_time TIMESTAMPTZ,
            exit_reason VARCHAR,
            pnl_r DOUBLE,
            slippage_ticks DOUBLE,
            strategy_id VARCHAR,
            lane_name VARCHAR,
            instrument VARCHAR,
            orb_minutes INTEGER,
            rr_target DOUBLE,
            filter_type VARCHAR,
            entry_model VARCHAR,
            execution_source VARCHAR,
            pnl_dollar DOUBLE,
            notes VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO validated_setups VALUES
        ('MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12', 0.087, 0.561, 1508, 'active'),
        ('MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', 0.1293, 0.4902, 918, 'active')
        """
    )
    con.execute(
        """
        INSERT INTO paper_trades VALUES
        ('2026-04-19', 'NYSE_OPEN', '2026-04-19 14:30:00+00', 'long', 20000, 19900, 20100, 20080,
         '2026-04-19 14:40:00+00', 'target', 0.8, 1.5, 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
         'NYSE_OPEN_test', 'MNQ', 5, 1.0, 'COST_LT12', 'E2', 'live', 160.0, 'ok'),
        ('2026-04-19', 'TOKYO_OPEN', '2026-04-19 00:30:00+00', 'long', 19000, 18900, 19150, 18950,
         '2026-04-19 00:45:00+00', 'stop', -0.5, 0.0, 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12',
         'TOKYO_OPEN_test', 'MNQ', 5, 1.5, 'COST_LT12', 'E2', 'backfill', NULL, 'modeled')
        """
    )
    con.close()


def _seed_journal_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE live_signal_events (
            event_id VARCHAR,
            trading_day DATE,
            instrument VARCHAR,
            strategy_id VARCHAR,
            event_type VARCHAR,
            reason VARCHAR,
            engine_price DOUBLE,
            fill_price DOUBLE,
            slippage_pts DOUBLE,
            contracts INTEGER,
            broker VARCHAR,
            order_id VARCHAR,
            trade_id VARCHAR,
            session_mode VARCHAR,
            created_at TIMESTAMPTZ
        )
        """
    )
    con.execute(
        """
        INSERT INTO live_signal_events VALUES
        ('e1', '2026-04-19', 'MNQ', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12', 'ENTRY_SUBMITTED', NULL, 20000, NULL, NULL, 1, 'tradovate', '1', 't1', 'live', '2026-04-19 14:30:01+00'),
        ('e2', '2026-04-19', 'MNQ', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12', 'ENTRY_FILLED', NULL, 20000, 20000.25, 1.0, 1, 'tradovate', '1', 't1', 'live', '2026-04-19 14:30:02+00'),
        ('e3', '2026-04-19', 'MNQ', 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', 'REGIME_PAUSED', 'allocator paused session regime', 19000, NULL, NULL, 1, 'tradovate', NULL, NULL, 'live', '2026-04-19 00:30:00+00')
        """
    )
    con.close()


def test_build_report_merges_modeled_realized_and_events(tmp_path):
    allocation = tmp_path / "lane_allocation.json"
    gold_db = tmp_path / "gold.db"
    journal_db = tmp_path / "live_journal.db"
    _write_allocation(allocation)
    _seed_gold_db(gold_db)
    _seed_journal_db(journal_db)

    report = build_report(allocation_path=allocation, db_path=gold_db, journal_path=journal_db, days=None)

    assert report["profile_id"] == "topstep_50k_mnq_auto"
    nyse, tokyo = report["lanes"]

    assert nyse["validated_expectancy_r"] == 0.087
    assert nyse["completed_trades"] == 1
    assert nyse["paper_trade_rows"] == 1
    assert nyse["avg_pnl_r"] == 0.8
    assert nyse["filled_events"] == 1
    assert nyse["mechanism_status"] == "COMPLETED_ROWS"

    assert tokyo["completed_trades"] == 0
    assert tokyo["paper_trade_rows"] == 1
    assert tokyo["avg_pnl_r"] is None
    assert tokyo["skipped_events"] == 1
    assert tokyo["mechanism_status"] == "EVENTS_ONLY"

    rendered = render_report(report)
    assert "LIVE ATTRIBUTION REPORT" in rendered
    assert "NYSE_OPEN" in rendered
    assert "TOKYO_OPEN" in rendered


def test_build_report_fail_closed_when_tables_missing(tmp_path):
    allocation = tmp_path / "lane_allocation.json"
    gold_db = tmp_path / "gold.db"
    journal_db = tmp_path / "live_journal.db"
    _write_allocation(allocation)
    duckdb.connect(str(gold_db)).close()
    duckdb.connect(str(journal_db)).close()

    report = build_report(allocation_path=allocation, db_path=gold_db, journal_path=journal_db, days=7)

    for lane in report["lanes"]:
        assert lane["completed_trades"] == 0
        assert lane["total_events"] == 0
        assert lane["mechanism_status"] == "NO_EVIDENCE"


def test_build_report_ignores_pre_rebalance_rows(tmp_path):
    allocation = tmp_path / "lane_allocation.json"
    gold_db = tmp_path / "gold.db"
    journal_db = tmp_path / "live_journal.db"
    _write_allocation(allocation)

    con = duckdb.connect(str(gold_db))
    con.execute(
        """
        CREATE TABLE paper_trades (
            trading_day DATE,
            orb_label VARCHAR,
            entry_time TIMESTAMPTZ,
            direction VARCHAR,
            entry_price DOUBLE,
            stop_price DOUBLE,
            target_price DOUBLE,
            exit_price DOUBLE,
            exit_time TIMESTAMPTZ,
            exit_reason VARCHAR,
            pnl_r DOUBLE,
            slippage_ticks DOUBLE,
            strategy_id VARCHAR,
            lane_name VARCHAR,
            instrument VARCHAR,
            orb_minutes INTEGER,
            rr_target DOUBLE,
            filter_type VARCHAR,
            entry_model VARCHAR,
            execution_source VARCHAR,
            pnl_dollar DOUBLE,
            notes VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO paper_trades VALUES
        ('2026-04-10', 'NYSE_OPEN', '2026-04-10 14:30:00+00', 'long', 20000, 19900, 20100, 20080,
         '2026-04-10 14:40:00+00', 'target', 0.8, 1.5, 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
         'NYSE_OPEN_test', 'MNQ', 5, 1.0, 'COST_LT12', 'E2', 'live', 160.0, 'old')
        """
    )
    con.close()
    duckdb.connect(str(journal_db)).close()

    report = build_report(allocation_path=allocation, db_path=gold_db, journal_path=journal_db, days=None)

    nyse = report["lanes"][0]
    assert report["window_start_trading_day"] == "2026-04-18"
    assert nyse["completed_trades"] == 0
    assert nyse["paper_trade_rows"] == 0
    assert nyse["mechanism_status"] == "NO_EVIDENCE"
