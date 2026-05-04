"""Tests for prop firm consistency rule and payout eligibility trackers."""

import duckdb
import pytest

from trading_app.consistency_tracker import (
    check_consistency,
    check_microscalp_compliance,
    check_payout_eligibility,
    check_profile_payout_eligibility,
)


@pytest.fixture
def db_with_trades(tmp_path):
    """Create a temp DB with paper_trades table and sample data."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE paper_trades (
            trading_day DATE,
            orb_label VARCHAR,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            direction VARCHAR,
            entry_price DOUBLE,
            stop_price DOUBLE,
            target_price DOUBLE,
            exit_price DOUBLE,
            exit_reason VARCHAR,
            pnl_r DOUBLE,
            pnl_dollar DOUBLE,
            slippage_ticks INTEGER,
            strategy_id VARCHAR,
            lane_name VARCHAR,
            instrument VARCHAR,
            orb_minutes INTEGER,
            rr_target DOUBLE,
            filter_type VARCHAR,
            entry_model VARCHAR,
            execution_source VARCHAR DEFAULT 'live',
            notes VARCHAR
        )
    """)

    # Insert sample trades across multiple days
    con.execute("""
        INSERT INTO paper_trades (trading_day, entry_time, exit_time, pnl_dollar, instrument, execution_source)
        VALUES
        ('2026-03-01', '2026-03-01 08:00:00', '2026-03-01 08:05:00', 100.0, 'MNQ', 'live'),
        ('2026-03-01', '2026-03-01 09:00:00', '2026-03-01 09:15:00', -30.0, 'MNQ', 'live'),
        ('2026-03-02', '2026-03-02 08:00:00', '2026-03-02 08:20:00', 200.0, 'MNQ', 'live'),
        ('2026-03-03', '2026-03-03 08:00:00', '2026-03-03 08:03:00', 50.0, 'MNQ', 'live'),
        ('2026-03-04', '2026-03-04 08:00:00', '2026-03-04 08:30:00', -20.0, 'MNQ', 'live'),
        ('2026-03-05', '2026-03-05 08:00:00', '2026-03-05 08:15:00', 80.0, 'MNQ', 'live'),
        ('2026-03-06', '2026-03-06 08:00:00', '2026-03-06 08:25:00', 60.0, 'MNQ', 'live'),
        ('2026-03-07', '2026-03-07 08:00:00', '2026-03-07 08:10:00', -10.0, 'MNQ', 'live'),
        ('2026-03-08', '2026-03-08 08:00:00', '2026-03-08 08:35:00', 90.0, 'MNQ', 'live')
    """)
    con.close()
    return db_path


class TestConsistency:
    def test_topstep_standard_has_no_consistency_rule(self, db_with_trades):
        result = check_consistency(firm="topstep", instrument="MNQ", db_path=db_with_trades)
        assert result is None

    def test_topstep_consistency_policy_warn(self, db_with_trades):
        result = check_consistency(
            firm="topstep",
            instrument="MNQ",
            db_path=db_with_trades,
            policy_id="topstep_express_consistency",
        )
        assert result is not None
        # Best day = $200 (Mar 2). Total profit = 100+200+50+80+60+90 = $580
        # Windfall = 200/580 = 34.5% < 40% limit but > 85% of 40% (=34%) -> WARN
        assert result.best_day_pnl == 200.0
        assert result.status == "WARN"

    def test_consistency_no_trades(self, tmp_path):
        db = tmp_path / "empty.db"
        con = duckdb.connect(str(db))
        con.execute("""
            CREATE TABLE paper_trades (
                trading_day DATE, entry_time TIMESTAMP, pnl_dollar DOUBLE, instrument VARCHAR, execution_source VARCHAR
            )
        """)
        con.close()
        result = check_consistency(
            firm="topstep",
            instrument="MNQ",
            db_path=db,
            policy_id="topstep_express_consistency",
        )
        assert result is None

    def test_tradeify_no_consistency_rule(self, db_with_trades):
        # Tradeify Select Flex has no consistency rule (consistency_rule=None in PROP_FIRM_SPECS)
        result = check_consistency(firm="tradeify", instrument="MNQ", db_path=db_with_trades)
        assert result is None


class TestPayoutEligibility:
    def test_tradeify_has_partial_policy_only(self, db_with_trades):
        result = check_payout_eligibility(firm="tradeify", instrument="MNQ", db_path=db_with_trades)
        assert result.policy_id == "tradeify_select_funded"
        assert result.policy_status == "partial"
        assert result.profit_threshold == 0.0
        assert not result.eligible
        assert any("partially modeled" in note for note in result.notes)

    def test_tradeify_not_eligible_few_days(self, tmp_path):
        db = tmp_path / "few.db"
        con = duckdb.connect(str(db))
        con.execute("""
            CREATE TABLE paper_trades (
                trading_day DATE, entry_time TIMESTAMP, pnl_dollar DOUBLE, instrument VARCHAR, execution_source VARCHAR
            )
        """)
        con.execute("""
            INSERT INTO paper_trades VALUES
            ('2026-03-01', '2026-03-01 08:00:00', 100.0, 'MNQ', 'live'),
            ('2026-03-02', '2026-03-02 08:00:00', 200.0, 'MNQ', 'live')
        """)
        con.close()
        result = check_payout_eligibility(firm="tradeify", instrument="MNQ", db_path=db)
        assert result.policy_status == "partial"
        assert not result.eligible
        assert result.trading_days == 2

    def test_topstep_standard_not_eligible_on_150_rule(self, db_with_trades):
        result = check_payout_eligibility(firm="topstep", instrument="MNQ", db_path=db_with_trades)
        assert result.policy_id == "topstep_express_standard"
        assert result.policy_status == "complete"
        assert result.profit_threshold == 150.0
        assert result.profitable_days_50 == 1  # only Mar 2 qualifies at $200
        assert not result.eligible

    def test_topstep_consistency_policy_is_eligible(self, db_with_trades):
        result = check_payout_eligibility(
            firm="topstep",
            instrument="MNQ",
            db_path=db_with_trades,
            policy_id="topstep_express_consistency",
        )
        assert result.policy_id == "topstep_express_consistency"
        assert result.policy_status == "complete"
        assert result.trading_days == 8
        assert result.eligible

    def test_profile_aware_topstep_uses_standard_policy(self, db_with_trades):
        result = check_profile_payout_eligibility("topstep_50k_mnq_auto", instrument="MNQ", db_path=db_with_trades)
        assert result.policy_id == "topstep_express_standard"
        assert result.policy_status == "complete"
        assert result.profit_threshold == 150.0


class TestMicroscalp:
    def test_microscalp_compliant(self, db_with_trades):
        result = check_microscalp_compliance(instrument="MNQ", db_path=db_with_trades)
        # Most trades held >5s (timestamps show >3min holds)
        assert result is not None
        assert result.pct_trades_over_10s > 50  # most >10s

    def test_microscalp_no_trades(self, tmp_path):
        db = tmp_path / "empty.db"
        con = duckdb.connect(str(db))
        con.execute("""
            CREATE TABLE paper_trades (
                entry_time TIMESTAMP, exit_time TIMESTAMP, pnl_dollar DOUBLE, instrument VARCHAR, execution_source VARCHAR
            )
        """)
        con.close()
        result = check_microscalp_compliance(instrument="MNQ", db_path=db)
        assert result is None
