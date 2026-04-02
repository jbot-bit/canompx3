"""Tests for trading_app.lane_allocator — 16 tests from spec.

Spec: docs/plans/2026-04-02-adaptive-lane-allocator-design.md
"""

from __future__ import annotations

import json
from datetime import date

import duckdb
import pytest

from trading_app.lane_allocator import (
    LaneScore,
    _classify_status,
    build_allocation,
    check_allocation_staleness,
    compute_lane_scores,
    generate_report,
)


# ── Factories ──────────────────────────────────────────────────────


def _make_score(**overrides) -> LaneScore:
    """Build a LaneScore with sane defaults (DEPLOY-ready)."""
    defaults = dict(
        strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER",
        instrument="MNQ",
        orb_label="COMEX_SETTLE",
        rr_target=1.0,
        filter_type="NO_FILTER",
        confirm_bars=1,
        stop_multiplier=1.0,
        trailing_expr=0.15,
        trailing_n=100,
        trailing_months=12,
        annual_r_estimate=30.0,
        trailing_wr=0.60,
        session_regime_expr=0.05,
        months_negative=0,
        months_positive_since_last_neg_streak=0,
        status="DEPLOY",
        status_reason="Test default",
    )
    defaults.update(overrides)
    return LaneScore(**defaults)


# ── DB Fixture ─────────────────────────────────────────────────────


@pytest.fixture()
def test_db(tmp_path):
    """Create a DuckDB with minimal schema for integration tests."""
    db_path = tmp_path / "test_alloc.db"
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
            stop_multiplier DOUBLE,
            sample_size INTEGER,
            status VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE orb_outcomes (
            trading_day DATE,
            symbol VARCHAR,
            orb_label VARCHAR,
            orb_minutes INTEGER,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            entry_model VARCHAR,
            pnl_r DOUBLE,
            mae_r DOUBLE,
            outcome VARCHAR,
            entry_price DOUBLE,
            stop_price DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE,
            symbol VARCHAR,
            orb_minutes INTEGER,
            orb_COMEX_SETTLE_break_dir VARCHAR,
            orb_COMEX_SETTLE_size DOUBLE
        )
    """)

    con.close()
    return db_path


def _seed_strategy(
    db_path,
    *,
    strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER",
    instrument="MNQ",
    orb_label="COMEX_SETTLE",
    entry_model="E2",
    rr_target=1.0,
    confirm_bars=1,
    filter_type="NO_FILTER",
    stop_multiplier=1.0,
    sample_size=100,
    status="active",
):
    """Insert a validated strategy into test DB."""
    con = duckdb.connect(str(db_path))
    con.execute(
        "INSERT INTO validated_setups VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            strategy_id,
            instrument,
            orb_label,
            entry_model,
            rr_target,
            confirm_bars,
            filter_type,
            stop_multiplier,
            sample_size,
            status,
        ],
    )
    con.close()


def _seed_outcomes(
    db_path, trades, *, symbol="MNQ", orb_label="COMEX_SETTLE", entry_model="E2", rr_target=1.0, confirm_bars=1
):
    """Insert multiple outcomes. trades = list of (trading_day, pnl_r, outcome, mae_r, entry_price, stop_price)."""
    con = duckdb.connect(str(db_path))
    for td, pnl_r, outcome, mae_r, entry_p, stop_p in trades:
        con.execute(
            "INSERT INTO orb_outcomes VALUES (?, ?, ?, 5, ?, ?, ?, ?, ?, ?, ?, ?)",
            [td, symbol, orb_label, rr_target, confirm_bars, entry_model, pnl_r, mae_r, outcome, entry_p, stop_p],
        )
    con.close()


def _seed_features(db_path, days, *, symbol="MNQ", break_dir="long", orb_size=10.0):
    """Insert daily_features for given days (all same break_dir/orb_size)."""
    con = duckdb.connect(str(db_path))
    for td in days:
        con.execute(
            "INSERT INTO daily_features VALUES (?, ?, 5, ?, ?)",
            [td, symbol, break_dir, orb_size],
        )
    con.close()


def _seed_features_mixed(db_path, day_sizes, *, symbol="MNQ", break_dir="long"):
    """Insert daily_features with varying orb sizes. day_sizes = list of (trading_day, orb_size)."""
    con = duckdb.connect(str(db_path))
    for td, orb_size in day_sizes:
        con.execute(
            "INSERT INTO daily_features VALUES (?, ?, 5, ?, ?)",
            [td, symbol, break_dir, orb_size],
        )
    con.close()


# ═══════════════════════════════════════════════════════════════════
# Tests 2, 3, 4, 5, 13, 16 — _classify_status unit tests
# ═══════════════════════════════════════════════════════════════════


class TestClassifyStatus:
    """Unit tests for the status classification logic."""

    def test_pause_after_2_negative_months(self):
        """Test 2: Strategy paused after 2 consecutive negative months."""
        monthly = [
            ("2025-06", -0.05, 20),
            ("2025-05", -0.08, 18),
            ("2025-04", 0.10, 22),
        ]
        status, reason = _classify_status(
            trailing_expr=-0.01,
            trailing_n=60,
            actual_months=3,
            months_neg=2,
            months_pos_since=0,
            annual_r=-2.0,
            session_regime_expr=0.05,
            monthly=monthly,
        )
        assert status == "PAUSE"
        assert "2 consecutive months negative" in reason

    def test_resume_after_3_positive_months(self):
        """Test 3: Strategy resumes after 3 positive months following a negative streak."""
        monthly = [
            ("2025-06", 0.12, 20),  # positive (most recent)
            ("2025-05", 0.08, 18),  # positive
            ("2025-04", 0.10, 22),  # positive → 3 positive since neg streak
            ("2025-03", -0.05, 15),  # negative streak start
            ("2025-02", -0.08, 16),  # negative streak (2 consecutive)
            ("2025-01", 0.15, 20),  # positive before streak
        ]
        status, reason = _classify_status(
            trailing_expr=0.05,
            trailing_n=111,
            actual_months=6,
            months_neg=0,  # no current negative streak
            months_pos_since=3,  # 3 positive months since last neg
            annual_r=12.0,
            session_regime_expr=0.05,
            monthly=monthly,
        )
        assert status == "RESUME"
        assert "Recovery confirmed" in reason
        assert "3 positive months" in reason

    def test_magnitude_override(self):
        """Test 4: 3mo avg ExpR < -0.10 triggers immediate pause even without consecutive negatives."""
        # Months alternate but are deeply negative on average
        monthly = [
            ("2025-06", -0.15, 30),  # negative
            ("2025-05", 0.02, 10),  # positive (but tiny N)
            ("2025-04", -0.20, 25),  # negative
            ("2025-03", 0.10, 20),  # positive
        ]
        # 3-month weighted avg: (-0.15*30 + 0.02*10 + -0.20*25) / (30+10+25)
        # = (-4.5 + 0.2 + -5.0) / 65 = -9.3/65 = -0.143
        status, reason = _classify_status(
            trailing_expr=-0.05,
            trailing_n=85,
            actual_months=4,
            months_neg=1,  # only 1 consecutive (most recent is negative)
            months_pos_since=0,
            annual_r=-15.0,
            session_regime_expr=0.05,
            monthly=monthly,
        )
        assert status == "PAUSE"
        assert "Magnitude override" in reason

    def test_regime_gate_hot_deploys(self):
        """Test 5a: Thin-data strategy deploys when session regime is HOT."""
        status, reason = _classify_status(
            trailing_expr=0.05,
            trailing_n=10,  # < MIN_TRAILING_N (20)
            actual_months=2,
            months_neg=0,
            months_pos_since=0,
            annual_r=3.0,
            session_regime_expr=0.05,  # positive → HOT
            monthly=[("2025-06", 0.05, 10)],
        )
        assert status == "DEPLOY"
        assert "session regime HOT" in reason

    def test_regime_gate_cold_pauses(self):
        """Test 5b: Thin-data strategy pauses when session regime is COLD."""
        status, reason = _classify_status(
            trailing_expr=-0.02,
            trailing_n=10,  # < MIN_TRAILING_N
            actual_months=2,
            months_neg=0,
            months_pos_since=0,
            annual_r=-1.0,
            session_regime_expr=-0.03,  # negative → COLD
            monthly=[("2025-06", -0.02, 10)],
        )
        assert status == "PAUSE"
        assert "session regime COLD" in reason

    def test_regime_gate_none_stale(self):
        """Test 5c: Thin-data strategy with no regime data → STALE."""
        status, reason = _classify_status(
            trailing_expr=0.0,
            trailing_n=5,
            actual_months=1,
            months_neg=0,
            months_pos_since=0,
            annual_r=0.0,
            session_regime_expr=None,
            monthly=[("2025-06", 0.0, 5)],
        )
        assert status == "STALE"
        assert "Insufficient trades" in reason

    def test_provisional_status(self):
        """Test 13: Strategy with <6 months data gets PROVISIONAL."""
        status, reason = _classify_status(
            trailing_expr=0.10,
            trailing_n=50,
            actual_months=3,  # < PROVISIONAL_MONTHS (6)
            months_neg=0,
            months_pos_since=0,
            annual_r=20.0,
            session_regime_expr=0.05,
            monthly=[
                ("2025-06", 0.10, 20),
                ("2025-05", 0.12, 15),
                ("2025-04", 0.08, 15),
            ],
        )
        assert status == "PROVISIONAL"
        assert "3 months" in reason

    def test_individual_month_negative(self):
        """Test 16: Both individual months negative → PAUSE (not just average).

        Even if the magnitude is small, two individually negative months
        trigger the 2-month consecutive pause rule.
        """
        monthly = [
            ("2025-06", -0.01, 25),  # barely negative
            ("2025-05", -0.01, 25),  # barely negative
            ("2025-04", 0.20, 25),  # positive
        ]
        status, reason = _classify_status(
            trailing_expr=0.06,  # overall positive
            trailing_n=75,
            actual_months=3,
            months_neg=2,  # 2 consecutive individual months negative
            months_pos_since=0,
            annual_r=6.0,
            session_regime_expr=0.05,
            monthly=monthly,
        )
        assert status == "PAUSE"
        assert "2 consecutive months negative" in reason


# ═══════════════════════════════════════════════════════════════════
# Tests 6, 7, 8, 9 — build_allocation unit tests
# ═══════════════════════════════════════════════════════════════════


class TestBuildAllocation:
    """Unit tests for greedy lane selection."""

    def test_respects_max_slots(self):
        """Test 6: Never returns more lanes than max_slots."""
        scores = [
            _make_score(
                strategy_id=f"MNQ_{session}_E2_RR1.0_CB1_NO_FILTER",
                orb_label=session,
                annual_r_estimate=30.0 - i,
            )
            for i, session in enumerate(
                [
                    "COMEX_SETTLE",
                    "NYSE_CLOSE",
                    "TOKYO_OPEN",
                    "SINGAPORE_OPEN",
                    "NYSE_OPEN",
                    "CME_PRECLOSE",
                    "EUROPE_FLOW",
                    "US_DATA_1000",
                ]
            )
        ]
        result = build_allocation(scores, max_slots=3, max_dd=100000.0)
        assert len(result) == 3

    def test_respects_allowed_sessions(self):
        """Test 7: Only includes lanes from allowed sessions."""
        scores = [
            _make_score(
                strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER",
                orb_label="COMEX_SETTLE",
                annual_r_estimate=30.0,
            ),
            _make_score(
                strategy_id="MNQ_NYSE_CLOSE_E2_RR1.0_CB1_NO_FILTER",
                orb_label="NYSE_CLOSE",
                annual_r_estimate=40.0,
            ),
            _make_score(
                strategy_id="MNQ_TOKYO_OPEN_E2_RR1.0_CB1_NO_FILTER",
                orb_label="TOKYO_OPEN",
                annual_r_estimate=25.0,
            ),
        ]
        result = build_allocation(
            scores,
            max_slots=5,
            max_dd=100000.0,
            allowed_sessions=frozenset({"COMEX_SETTLE"}),
        )
        assert len(result) == 1
        assert result[0].orb_label == "COMEX_SETTLE"

    def test_hysteresis_20pct(self):
        """Test 8: Lane not replaced for <20% improvement; replaced at >=20%."""
        old_strat = _make_score(
            strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER",
            orb_label="COMEX_SETTLE",
            annual_r_estimate=30.0,
        )
        # New candidate: 15% better (30 → 34.5) — below 20% threshold
        new_strat_small = _make_score(
            strategy_id="MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6",
            orb_label="COMEX_SETTLE",
            filter_type="ORB_G6",
            rr_target=1.5,
            annual_r_estimate=34.5,
        )
        scores = [old_strat, new_strat_small]
        result = build_allocation(
            scores,
            max_slots=5,
            max_dd=100000.0,
            prior_allocation=[old_strat.strategy_id],
        )
        # Old should stay — 15% improvement not enough
        selected_ids = [s.strategy_id for s in result]
        assert old_strat.strategy_id in selected_ids

        # Now test with 25% improvement (30 → 37.5) — above threshold
        new_strat_big = _make_score(
            strategy_id="MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G8",
            orb_label="COMEX_SETTLE",
            filter_type="ORB_G8",
            rr_target=2.0,
            annual_r_estimate=37.5,
        )
        scores = [old_strat, new_strat_big]
        result = build_allocation(
            scores,
            max_slots=5,
            max_dd=100000.0,
            prior_allocation=[old_strat.strategy_id],
        )
        selected_ids = [s.strategy_id for s in result]
        assert new_strat_big.strategy_id in selected_ids

    def test_annual_r_ranking(self):
        """Test 9: High-N medium-ExpR beats low-N high-ExpR via annual_r."""
        # Strategy A: lots of trades, moderate ExpR → high annual R
        strat_a = _make_score(
            strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER",
            orb_label="COMEX_SETTLE",
            trailing_expr=0.10,
            trailing_n=200,
            trailing_months=12,
            annual_r_estimate=20.0,  # 0.10 * 200 trades/year
        )
        # Strategy B: few trades, high ExpR → lower annual R
        strat_b = _make_score(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR1.0_CB1_NO_FILTER",
            orb_label="NYSE_CLOSE",
            trailing_expr=0.30,
            trailing_n=40,
            trailing_months=12,
            annual_r_estimate=12.0,  # 0.30 * 40 trades/year
        )
        result = build_allocation(
            [strat_a, strat_b],
            max_slots=2,
            max_dd=100000.0,
        )
        # A should be ranked first (annual_r 20 > 12)
        assert result[0].strategy_id == strat_a.strategy_id


# ═══════════════════════════════════════════════════════════════════
# Test 10 — Report completeness
# ═══════════════════════════════════════════════════════════════════


class TestReport:
    """Report generation tests."""

    def test_report_completeness(self):
        """Test 10: Report has all required sections and fields."""
        deployed = _make_score(status="DEPLOY", annual_r_estimate=30.0)
        paused = _make_score(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR1.0_CB1_NO_FILTER",
            orb_label="NYSE_CLOSE",
            status="PAUSE",
            status_reason="2 consecutive months negative",
            annual_r_estimate=-5.0,
        )
        scores = [deployed, paused]
        allocation = [deployed]

        report = generate_report(scores, allocation, date(2025, 7, 1), "apex_100k_manual")

        # Required sections
        assert "Selected Lanes" in report
        assert "Paused" in report
        assert "Session Regimes" in report

        # Must include profile and date
        assert "2025-07-01" in report
        assert "apex_100k_manual" in report

        # Deployed strategy appears in selected lanes
        assert deployed.strategy_id in report

        # Paused strategies collapsed by category
        assert "Consecutive months negative" in report


# ═══════════════════════════════════════════════════════════════════
# Tests 14, 15 — Staleness checks
# ═══════════════════════════════════════════════════════════════════


class TestStaleness:
    """Allocation staleness detection tests."""

    def test_staleness_warning(self, tmp_path):
        """Test 14: Allocation >35 days old triggers WARNING."""
        alloc_path = tmp_path / "lane_allocation.json"
        rebalance = date(2025, 6, 1)
        check_day = date(2025, 7, 7)  # 36 days later
        alloc_path.write_text(json.dumps({"rebalance_date": rebalance.isoformat()}))

        status, days_old = check_allocation_staleness(alloc_path, today=check_day)

        assert status == "WARNING"
        assert days_old == 36

    def test_staleness_block(self, tmp_path):
        """Test 15: Allocation >60 days old blocks trading."""
        alloc_path = tmp_path / "lane_allocation.json"
        rebalance = date(2025, 5, 1)
        check_day = date(2025, 7, 2)  # 62 days later
        alloc_path.write_text(json.dumps({"rebalance_date": rebalance.isoformat()}))

        status, days_old = check_allocation_staleness(alloc_path, today=check_day)

        assert status == "BLOCK"
        assert days_old == 62

    def test_staleness_ok(self, tmp_path):
        """Fresh allocation returns OK."""
        alloc_path = tmp_path / "lane_allocation.json"
        rebalance = date(2025, 7, 1)
        check_day = date(2025, 7, 10)  # 9 days later
        alloc_path.write_text(json.dumps({"rebalance_date": rebalance.isoformat()}))

        status, days_old = check_allocation_staleness(alloc_path, today=check_day)

        assert status == "OK"
        assert days_old == 9

    def test_staleness_missing_file(self, tmp_path):
        """Missing allocation file → BLOCK."""
        status, days_old = check_allocation_staleness(
            tmp_path / "nonexistent.json",
            today=date(2025, 7, 1),
        )
        assert status == "BLOCK"
        assert days_old == -1


# ═══════════════════════════════════════════════════════════════════
# Tests 1, 11, 12 — Integration tests (DB-backed)
# ═══════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests requiring a test database."""

    def test_zero_lookahead(self, test_db):
        """Test 1: Future trades excluded from trailing window.

        Rebalance date = 2025-07-01. Only trades with trading_day < 2025-07-01
        should count. Trades ON or AFTER the rebalance date must be excluded.
        """
        rebalance = date(2025, 7, 1)

        _seed_strategy(test_db)

        # Pre-rebalance trades (should be counted)
        pre_trades = [
            (date(2025, 5, 15), 1.0, "win", 0.3, 100.0, 99.0),
            (date(2025, 6, 15), 1.0, "win", 0.3, 100.0, 99.0),
        ]
        # Post-rebalance trades (must NOT be counted)
        post_trades = [
            (date(2025, 7, 1), -1.0, "loss", 0.8, 100.0, 99.0),  # ON date
            (date(2025, 7, 15), -1.0, "loss", 0.8, 100.0, 99.0),  # after
        ]
        _seed_outcomes(test_db, pre_trades + post_trades)

        all_days = [t[0] for t in pre_trades + post_trades]
        _seed_features(test_db, all_days)

        scores = compute_lane_scores(rebalance_date=rebalance, db_path=test_db)

        assert len(scores) == 1
        score = scores[0]
        # Only 2 pre-rebalance trades should count
        assert score.trailing_n == 2
        # Both were wins → trailing_expr = 1.0
        assert score.trailing_expr == 1.0
        assert score.trailing_wr == 1.0

    def test_sm_adjustment(self, test_db):
        """Test 11: SM=0.75 trailing ExpR differs from SM=1.0 for same outcomes.

        Uses MNQ cost model: point_value=2.0, total_friction=2.74.
        Trade: entry=100.0, stop=99.0, risk_pts=1.0.
        Some trades have mae_r=0.5 → max_adv_pts=1.185 >= 0.75 → stopped at SM=0.75.
        """
        rebalance = date(2025, 7, 1)

        # Two strategies: same underlying data, different stop multipliers
        _seed_strategy(
            test_db,
            strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER_SM10",
            stop_multiplier=1.0,
        )
        _seed_strategy(
            test_db,
            strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER_SM075",
            stop_multiplier=0.75,
        )

        trades = []
        days = []
        for i in range(10):
            td = date(2025, 6, 1 + i)
            # 5 trades with low mae (not stopped at either SM)
            # 5 trades with high mae (stopped at SM=0.75 only)
            mae = 0.1 if i < 5 else 0.5
            trades.append((td, 1.0, "win", mae, 100.0, 99.0))
            days.append(td)

        _seed_outcomes(test_db, trades)
        _seed_features(test_db, days)

        scores = compute_lane_scores(rebalance_date=rebalance, db_path=test_db)

        sm10 = next(s for s in scores if "SM10" in s.strategy_id)
        sm075 = next(s for s in scores if "SM075" in s.strategy_id)

        # SM=1.0: all 10 trades are wins → ExpR = 1.0
        assert sm10.trailing_expr == 1.0
        assert sm10.trailing_wr == 1.0
        assert sm10.trailing_n == 10

        # SM=0.75: 5 wins + 5 stopped (pnl_r=-0.75) → different ExpR
        assert sm075.trailing_n == 10
        assert sm075.trailing_expr < sm10.trailing_expr
        assert sm075.trailing_wr < sm10.trailing_wr

    def test_filter_applied_in_trailing(self, test_db):
        """Test 12: Filtered trailing ExpR uses only filter-eligible days.

        ORB_G6 filter requires orb_size >= 6. Days with smaller ORBs
        should be excluded from the trailing window entirely.
        """
        rebalance = date(2025, 7, 1)

        # NO_FILTER strategy — counts all trades
        _seed_strategy(
            test_db,
            strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER",
            filter_type="NO_FILTER",
        )
        # ORB_G6 strategy — only counts trades where ORB size >= 6
        _seed_strategy(
            test_db,
            strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6",
            filter_type="ORB_G6",
        )

        trades = []
        day_sizes = []
        for i in range(10):
            td = date(2025, 6, 1 + i)
            trades.append((td, 1.0, "win", 0.3, 100.0, 99.0))
            # First 5 days: large ORB (passes G6), last 5: small ORB (fails G6)
            orb_size = 10.0 if i < 5 else 3.0
            day_sizes.append((td, orb_size))

        _seed_outcomes(test_db, trades)
        _seed_features_mixed(test_db, day_sizes)

        scores = compute_lane_scores(rebalance_date=rebalance, db_path=test_db)

        no_filter = next(s for s in scores if s.filter_type == "NO_FILTER")
        orb_g6 = next(s for s in scores if s.filter_type == "ORB_G6")

        # NO_FILTER: all 10 trades counted
        assert no_filter.trailing_n == 10

        # ORB_G6: only 5 trades on large-ORB days counted
        assert orb_g6.trailing_n == 5
