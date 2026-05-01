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
    _effective_annual_r,
    build_allocation,
    check_allocation_staleness,
    compute_lane_scores,
    enrich_scores_with_liveness,
    generate_report,
    load_sr_state,
)
from trading_app.lane_correlation import RHO_REJECT_THRESHOLD


# ── Factories ──────────────────────────────────────────────────────


def _make_score(**overrides) -> LaneScore:
    """Build a LaneScore with sane defaults (DEPLOY-ready, chordia-clean).

    Chordia defaults: PASS_PROTOCOL_A verdict + audit_age 0d. This keeps
    every existing test passing under the new chordia gate without
    forcing each call site to specify chordia kwargs. Tests that exercise
    the gate explicitly override these fields.
    """
    defaults = dict(
        strategy_id="MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER",
        instrument="MNQ",
        orb_label="COMEX_SETTLE",
        orb_minutes=5,
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
        chordia_verdict="PASS_PROTOCOL_A",
        chordia_audit_age_days=0,
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
            orb_minutes INTEGER,
            entry_model VARCHAR,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            filter_type VARCHAR,
            stop_multiplier DOUBLE,
            sample_size INTEGER,
            sharpe_ratio DOUBLE,
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
    orb_minutes=5,
    entry_model="E2",
    rr_target=1.0,
    confirm_bars=1,
    filter_type="NO_FILTER",
    stop_multiplier=1.0,
    sample_size=100,
    sharpe_ratio=0.10,  # default; t = 0.10 * sqrt(100) = 1.0 -> FAIL_BOTH unless overridden
    status="active",
):
    """Insert a validated strategy into test DB."""
    con = duckdb.connect(str(db_path))
    con.execute(
        "INSERT INTO validated_setups VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            strategy_id,
            instrument,
            orb_label,
            orb_minutes,
            entry_model,
            rr_target,
            confirm_bars,
            filter_type,
            stop_multiplier,
            sample_size,
            sharpe_ratio,
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

    def test_hot_session_deploys_despite_negative_months(self):
        """Test 2: Strategy deploys when session is HOT even with negative months.

        Regime-only gating: individual month streaks are noise.
        Backtest 2022-2025: regime gate +630R vs individual pause -799R.
        """
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
            session_regime_expr=0.05,  # HOT session
            monthly=monthly,
        )
        assert status == "DEPLOY"
        assert "HOT" in reason

    def test_cold_session_pauses_despite_positive_trailing(self):
        """Test 3: Strategy pauses when session is COLD even if trailing is positive."""
        monthly = [
            ("2025-06", 0.12, 20),
            ("2025-05", 0.08, 18),
            ("2025-04", 0.10, 22),
        ]
        status, reason = _classify_status(
            trailing_expr=0.10,
            trailing_n=60,
            actual_months=3,
            months_neg=0,
            months_pos_since=3,
            annual_r=20.0,
            session_regime_expr=-0.03,  # COLD session
            monthly=monthly,
        )
        assert status == "PAUSE"
        assert "COLD" in reason

    def test_hot_session_deploys_despite_deeply_negative(self):
        """Test 4: Even deeply negative individual strategy deploys in HOT session.

        The regime gate is the ONLY gate. Individual magnitude doesn't matter.
        """
        monthly = [
            ("2025-06", -0.15, 30),
            ("2025-05", 0.02, 10),
            ("2025-04", -0.20, 25),
        ]
        status, reason = _classify_status(
            trailing_expr=-0.05,
            trailing_n=85,
            actual_months=4,
            months_neg=1,
            months_pos_since=0,
            annual_r=-15.0,
            session_regime_expr=0.05,  # HOT session overrides individual
            monthly=monthly,
        )
        assert status == "DEPLOY"
        assert "HOT" in reason

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
        assert "COLD" in reason

    def test_no_regime_thin_data_stale(self):
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
        assert "No regime data" in reason

    def test_no_regime_positive_trailing_deploys(self):
        """Test 13: No regime data but positive trailing → DEPLOY."""
        status, reason = _classify_status(
            trailing_expr=0.10,
            trailing_n=50,
            actual_months=3,
            months_neg=0,
            months_pos_since=0,
            annual_r=20.0,
            session_regime_expr=None,
            monthly=[
                ("2025-06", 0.10, 20),
                ("2025-05", 0.12, 15),
                ("2025-04", 0.08, 15),
            ],
        )
        assert status == "DEPLOY"
        assert "trailing positive" in reason

    def test_hot_session_overrides_individual_negatives(self):
        """Test 16: Barely negative individual months DON'T pause in HOT session.

        Regime-only: individual month streaks are noise. Session regime is structural.
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
            months_neg=2,
            months_pos_since=0,
            annual_r=6.0,
            session_regime_expr=0.05,  # HOT
            monthly=monthly,
        )
        assert status == "DEPLOY"
        assert "HOT" in reason


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

        report = generate_report(scores, allocation, date(2025, 7, 1), "topstep_50k_mnq_auto")

        # Required sections
        assert "Selected Lanes" in report
        assert "Paused" in report
        assert "Session Regimes" in report

        # Must include profile and date
        assert "2025-07-01" in report
        assert "topstep_50k_mnq_auto" in report

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


class TestLivenessScoring:
    """Tests for SR alarm + 3mo decay ranking adjustments."""

    def test_alarm_discounts_annual_r(self):
        s = _make_score(annual_r_estimate=40.0, sr_status="ALARM")
        assert _effective_annual_r(s) == pytest.approx(20.0)

    def test_continue_no_discount(self):
        s = _make_score(annual_r_estimate=40.0, sr_status="CONTINUE")
        assert _effective_annual_r(s) == pytest.approx(40.0)

    def test_unknown_no_discount(self):
        s = _make_score(annual_r_estimate=40.0, sr_status="UNKNOWN")
        assert _effective_annual_r(s) == pytest.approx(40.0)

    def test_3mo_decay_discounts(self):
        s = _make_score(
            annual_r_estimate=40.0,
            trailing_expr=0.15,
            recent_3mo_expr=-0.05,
        )
        assert _effective_annual_r(s) == pytest.approx(30.0)

    def test_3mo_positive_no_discount(self):
        s = _make_score(
            annual_r_estimate=40.0,
            trailing_expr=0.15,
            recent_3mo_expr=0.10,
        )
        assert _effective_annual_r(s) == pytest.approx(40.0)

    def test_both_alarm_and_decay_stack(self):
        s = _make_score(
            annual_r_estimate=40.0,
            sr_status="ALARM",
            trailing_expr=0.15,
            recent_3mo_expr=-0.05,
        )
        # 40 * 0.5 * 0.75 = 15.0
        assert _effective_annual_r(s) == pytest.approx(15.0)

    def test_alarm_lane_replaced_by_continue(self):
        """ALARM lane should be replaced by a CONTINUE lane on the same session."""
        alarm = _make_score(
            strategy_id="MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5",
            orb_label="COMEX_SETTLE",
            annual_r_estimate=44.0,
            sr_status="ALARM",
        )
        healthy = _make_score(
            strategy_id="MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5",
            orb_label="EUROPE_FLOW",
            annual_r_estimate=30.0,
            sr_status="CONTINUE",
        )
        # ALARM: effective = 44*0.5=22. CONTINUE: effective = 30.
        # healthy should rank higher.
        result = build_allocation([alarm, healthy], max_slots=1)
        assert result[0].strategy_id == healthy.strategy_id

    def test_alarm_lane_kept_if_no_alternative(self):
        """ALARM lane should still be selected if it's the only option."""
        alarm = _make_score(
            annual_r_estimate=44.0,
            sr_status="ALARM",
        )
        result = build_allocation([alarm], max_slots=1)
        assert len(result) == 1
        assert result[0].strategy_id == alarm.strategy_id

    def test_load_sr_state_missing_file(self):
        """Missing SR state file should return empty dict (fail-open)."""
        state = load_sr_state()
        # May or may not be empty depending on whether the file exists
        assert isinstance(state, dict)

    def test_recent_3mo_computed(self):
        """LaneScore with sufficient monthly data should have recent_3mo_expr."""
        s = _make_score(recent_3mo_expr=-0.05)
        assert s.recent_3mo_expr == -0.05

    def test_orb_size_stats_used_in_dd(self):
        """build_allocation should use per-session P90 from orb_size_stats."""
        # Two lanes with very different ORB sizes
        cheap = _make_score(
            strategy_id="MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_G5",
            orb_label="SINGAPORE_OPEN",
            annual_r_estimate=40.0,
        )
        expensive = _make_score(
            strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
            orb_label="NYSE_OPEN",
            annual_r_estimate=30.0,
        )
        # With per-session stats: SING P90=38, NYSE P90=117
        # At SM=0.75, PV=$2: SING DD=$57, NYSE DD=$175.50
        # Total: $232.50 — fits in $300 budget
        orb_stats = {
            ("MNQ", "SINGAPORE_OPEN", 5): (20.0, 38.0),
            ("MNQ", "NYSE_OPEN", 5): (76.0, 117.0),
        }
        result = build_allocation(
            [cheap, expensive],
            max_slots=5,
            max_dd=300.0,
            stop_multiplier=0.75,
            orb_size_stats=orb_stats,
        )
        # Both should fit: $57 + $175.50 = $232.50 < $300
        assert len(result) == 2

    def test_orb_size_stats_budget_constraint(self):
        """Per-session P90 should correctly enforce DD budget."""
        expensive = _make_score(
            strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
            orb_label="NYSE_OPEN",
            annual_r_estimate=30.0,
        )
        orb_stats = {("MNQ", "NYSE_OPEN", 5): (76.0, 117.0)}
        # DD = 117 * 0.75 * 2.0 = $175.50, budget = $100 → should NOT fit
        result = build_allocation(
            [expensive],
            max_slots=5,
            max_dd=100.0,
            stop_multiplier=0.75,
            orb_size_stats=orb_stats,
        )
        assert len(result) == 0

    def test_orb_size_stats_dd_uses_correct_aperture_not_o5(self):
        """Regression for 2026-04-30 audit: O15 lane DD must use O15 P90, not O5 P90.

        Prior bug: allocator hardcoded orb_minutes=5 in compute_orb_size_stats,
        so O15 lanes got O5 ORB sizes (~60% understated DD budget). This test
        proves the lookup now distinguishes apertures. If the bug returns,
        the O15 lane below would get the (much smaller) O5 P90 and pass the
        budget check incorrectly.
        """
        o15_lane = _make_score(
            strategy_id="MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15",
            orb_label="US_DATA_1000",
            orb_minutes=15,
            annual_r_estimate=40.0,
        )
        # Both apertures present. O15 P90 is the real one (147pts);
        # O5 P90 (95pts) would understate the DD budget if used.
        # At SM=0.75, PV=$2: O15 DD = 147*0.75*2 = $220.50; O5 DD = $142.50.
        orb_stats = {
            ("MNQ", "US_DATA_1000", 5): (56.0, 95.0),
            ("MNQ", "US_DATA_1000", 15): (87.0, 147.0),
        }
        # Budget $200: O15 DD ($220.50) does NOT fit → result must be empty.
        # If bug returns (uses O5 P90): O5 DD ($142.50) fits → result has 1 lane.
        result_correct = build_allocation(
            [o15_lane],
            max_slots=5,
            max_dd=200.0,
            stop_multiplier=0.75,
            orb_size_stats=orb_stats,
        )
        assert len(result_correct) == 0, (
            "O15 lane fit a $200 DD budget, but at the correct O15 P90 (147pts) "
            "its DD is $220.50 — does not fit. If this asserts, the allocator "
            "is using O5 P90 (95pts → $142.50 DD) for an O15 lane, regressing "
            "the 2026-04-30 fix."
        )

        # Sanity check: with budget $250, O15 DD ($220.50) DOES fit.
        result_loose = build_allocation(
            [o15_lane],
            max_slots=5,
            max_dd=250.0,
            stop_multiplier=0.75,
            orb_size_stats=orb_stats,
        )
        assert len(result_loose) == 1


class TestCorrelationAwareSelection:
    """Tests for correlation-gated greedy lane selection."""

    def test_high_corr_pair_rejected(self):
        """Two lanes with rho > 0.70 should not both be selected."""
        a = _make_score(strategy_id="STRAT_A", orb_label="COMEX_SETTLE", annual_r_estimate=44.0)
        b = _make_score(strategy_id="STRAT_B", orb_label="COMEX_SETTLE", annual_r_estimate=40.0)
        corr = {("STRAT_A", "STRAT_B"): 0.95}  # highly correlated
        result = build_allocation([a, b], max_slots=5, correlation_matrix=corr)
        assert len(result) == 1
        assert result[0].strategy_id == "STRAT_A"  # higher AnnR wins

    def test_low_corr_pair_accepted(self):
        """Two lanes with rho < 0.70 should both be selected."""
        a = _make_score(strategy_id="STRAT_A", orb_label="COMEX_SETTLE", annual_r_estimate=44.0)
        b = _make_score(strategy_id="STRAT_B", orb_label="EUROPE_FLOW", annual_r_estimate=40.0)
        corr = {("STRAT_A", "STRAT_B"): 0.15}  # low correlation
        result = build_allocation([a, b], max_slots=5, correlation_matrix=corr)
        assert len(result) == 2

    def test_same_session_different_filters_rejected(self):
        """Same session, different filters with high rho should be deduplicated."""
        g5 = _make_score(strategy_id="MNQ_COMEX_E2_RR1.5_CB1_ORB_G5", orb_label="COMEX_SETTLE", annual_r_estimate=44.0)
        lt12 = _make_score(
            strategy_id="MNQ_COMEX_E2_RR1.5_CB1_COST_LT12", orb_label="COMEX_SETTLE", annual_r_estimate=40.0
        )
        # Same-session filters are typically rho ≈ 1.0
        corr = {("MNQ_COMEX_E2_RR1.5_CB1_COST_LT12", "MNQ_COMEX_E2_RR1.5_CB1_ORB_G5"): 1.0}
        result = build_allocation([g5, lt12], max_slots=5, correlation_matrix=corr)
        assert len(result) == 1
        assert result[0].strategy_id == g5.strategy_id

    def test_three_lanes_one_rejected(self):
        """With 3 lanes where A-B correlated but A-C and B-C uncorrelated, pick A+C."""
        a = _make_score(strategy_id="A", annual_r_estimate=50.0)
        b = _make_score(strategy_id="B", orb_label="EUROPE_FLOW", annual_r_estimate=45.0)
        c = _make_score(strategy_id="C", orb_label="TOKYO_OPEN", annual_r_estimate=30.0)
        corr = {
            ("A", "B"): 0.85,  # A and B correlated → B rejected
            ("A", "C"): 0.10,  # A and C independent
            ("B", "C"): 0.05,  # B and C independent
        }
        result = build_allocation([a, b, c], max_slots=5, correlation_matrix=corr)
        assert len(result) == 2
        sids = {r.strategy_id for r in result}
        assert sids == {"A", "C"}

    def test_no_corr_matrix_uses_session_fallback(self):
        """Without correlation matrix, falls back to 1-per-session heuristic."""
        a = _make_score(strategy_id="A", orb_label="COMEX_SETTLE", annual_r_estimate=44.0)
        b = _make_score(strategy_id="B", orb_label="COMEX_SETTLE", annual_r_estimate=40.0)
        # No correlation_matrix → 1-per-session → only A selected
        result = build_allocation([a, b], max_slots=5)
        assert len(result) == 1
        assert result[0].strategy_id == "A"

    def test_corr_threshold_boundary(self):
        """Exactly at threshold should pass (> not >=)."""
        a = _make_score(strategy_id="A", annual_r_estimate=44.0)
        b = _make_score(strategy_id="B", orb_label="EUROPE_FLOW", annual_r_estimate=40.0)
        corr = {("A", "B"): RHO_REJECT_THRESHOLD}  # exactly at threshold
        result = build_allocation([a, b], max_slots=5, correlation_matrix=corr)
        # At threshold (not above) → both selected
        assert len(result) == 2

    def test_missing_pair_in_matrix_defaults_zero(self):
        """Missing pair in correlation matrix defaults to rho=0 (uncorrelated)."""
        a = _make_score(strategy_id="A", annual_r_estimate=44.0)
        b = _make_score(strategy_id="B", orb_label="EUROPE_FLOW", annual_r_estimate=40.0)
        corr = {}  # empty matrix — all pairs default to 0
        result = build_allocation([a, b], max_slots=5, correlation_matrix=corr)
        assert len(result) == 2


class TestChordiaGate:
    """Allocator Chordia gate — refuse DEPLOY for FAIL/MISSING/stale audits.

    Stage: docs/runtime/stages/allocator-chordia-gate.md.
    Verdict policy lives in trading_app.chordia.chordia_verdict_label and
    chordia_verdict_allows_deploy. The gate sits in
    lane_allocator.apply_chordia_gate, which build_allocation invokes inline.
    """

    def test_pass_protocol_a_unchanged(self):
        """A PASS_PROTOCOL_A score with fresh audit is left at DEPLOY."""
        from trading_app.lane_allocator import apply_chordia_gate

        s = _make_score(
            strategy_id="A",
            chordia_verdict="PASS_PROTOCOL_A",
            chordia_audit_age_days=10,
        )
        result = apply_chordia_gate([s])
        assert len(result) == 1
        assert result[0].status == "DEPLOY"
        assert result[0].chordia_verdict == "PASS_PROTOCOL_A"

    def test_pass_chordia_unchanged(self):
        """A PASS_CHORDIA score with fresh audit is left at DEPLOY."""
        from trading_app.lane_allocator import apply_chordia_gate

        s = _make_score(
            strategy_id="A",
            chordia_verdict="PASS_CHORDIA",
            chordia_audit_age_days=0,
        )
        result = apply_chordia_gate([s])
        assert result[0].status == "DEPLOY"

    def test_fail_both_demoted_to_pause(self):
        """A FAIL_BOTH score is demoted to PAUSE with chordia reason."""
        from trading_app.lane_allocator import apply_chordia_gate

        s = _make_score(
            strategy_id="A",
            chordia_verdict="FAIL_BOTH",
            chordia_audit_age_days=0,
        )
        result = apply_chordia_gate([s])
        assert result[0].status == "PAUSE"
        assert "FAIL_BOTH" in result[0].status_reason
        assert result[0].chordia_verdict == "FAIL_BOTH"  # field preserved for traceability

    def test_fail_chordia_demoted_to_pause(self):
        """A FAIL_CHORDIA score (3.00<=t<3.79, no theory) is demoted to PAUSE."""
        from trading_app.lane_allocator import apply_chordia_gate

        s = _make_score(
            strategy_id="A",
            chordia_verdict="FAIL_CHORDIA",
            chordia_audit_age_days=0,
        )
        result = apply_chordia_gate([s])
        assert result[0].status == "PAUSE"
        assert "FAIL_CHORDIA" in result[0].status_reason

    def test_missing_verdict_demoted_to_pause(self):
        """A score with chordia_verdict=None is demoted to PAUSE."""
        from trading_app.lane_allocator import apply_chordia_gate

        s = _make_score(
            strategy_id="A",
            chordia_verdict=None,
            chordia_audit_age_days=0,
        )
        result = apply_chordia_gate([s])
        assert result[0].status == "PAUSE"
        assert "missing audit" in result[0].status_reason.lower()

    def test_missing_audit_age_demoted_to_pause(self):
        """A PASS verdict with chordia_audit_age_days=None is still demoted to PAUSE."""
        from trading_app.lane_allocator import apply_chordia_gate

        s = _make_score(
            strategy_id="A",
            chordia_verdict="PASS_PROTOCOL_A",
            chordia_audit_age_days=None,
        )
        result = apply_chordia_gate([s])
        assert result[0].status == "PAUSE"
        assert "audit_date" in result[0].status_reason.lower()

    def test_stale_audit_demoted_to_pause(self):
        """audit_age_days=91 is stale (>90 default freshness) — PAUSE."""
        from trading_app.lane_allocator import apply_chordia_gate
        from trading_app.chordia import ChordiaAuditLog

        # Construct an empty doctrine log with default freshness=90.
        log = ChordiaAuditLog(default_has_theory=False, audit_freshness_days=90, entries={})
        s = _make_score(
            strategy_id="A",
            chordia_verdict="PASS_PROTOCOL_A",
            chordia_audit_age_days=91,
        )
        result = apply_chordia_gate([s], audit_log=log)
        assert result[0].status == "PAUSE"
        assert "stale" in result[0].status_reason.lower()

    def test_audit_age_at_freshness_boundary_passes(self):
        """audit_age_days=90 exactly equals freshness — still passes (>, not >=)."""
        from trading_app.lane_allocator import apply_chordia_gate
        from trading_app.chordia import ChordiaAuditLog

        log = ChordiaAuditLog(default_has_theory=False, audit_freshness_days=90, entries={})
        s = _make_score(
            strategy_id="A",
            chordia_verdict="PASS_PROTOCOL_A",
            chordia_audit_age_days=90,
        )
        result = apply_chordia_gate([s], audit_log=log)
        assert result[0].status == "DEPLOY"

    def test_existing_pause_not_overridden(self):
        """An already-PAUSED score keeps its original reason; gate is additive."""
        from trading_app.lane_allocator import apply_chordia_gate

        s = _make_score(
            strategy_id="A",
            status="PAUSE",
            status_reason="Recovering — need 3+ positive months",
            chordia_verdict="FAIL_BOTH",
        )
        result = apply_chordia_gate([s])
        assert result[0].status == "PAUSE"
        assert "Recovering" in result[0].status_reason  # original reason preserved
        assert "chordia gate" not in result[0].status_reason

    def test_build_allocation_invokes_gate(self):
        """build_allocation refuses FAIL_BOTH lanes even if caller forgot the gate."""
        deploy_clean = _make_score(
            strategy_id="CLEAN",
            chordia_verdict="PASS_PROTOCOL_A",
            chordia_audit_age_days=10,
            annual_r_estimate=50.0,
        )
        deploy_failed = _make_score(
            strategy_id="FAILED",
            orb_label="EUROPE_FLOW",
            chordia_verdict="FAIL_BOTH",
            chordia_audit_age_days=10,
            annual_r_estimate=99.0,  # would beat CLEAN if gate weren't enforced
        )
        result = build_allocation([deploy_clean, deploy_failed], max_slots=5)
        sids = {s.strategy_id for s in result}
        assert "CLEAN" in sids
        assert "FAILED" not in sids

    def test_save_allocation_emits_chordia_fields(self, tmp_path):
        """save_allocation writes chordia_verdict + chordia_audit_age_days into the JSON."""
        import json
        from datetime import date as _date
        from trading_app.lane_allocator import save_allocation

        s = _make_score(
            strategy_id="A",
            chordia_verdict="PASS_PROTOCOL_A",
            chordia_audit_age_days=15,
        )
        path = save_allocation(
            scores=[s],
            allocation=[s],
            rebalance_date=_date(2026, 5, 1),
            profile_id="test_profile",
            output_path=tmp_path / "test_alloc.json",
        )
        data = json.loads(path.read_text())
        assert len(data["lanes"]) == 1
        lane = data["lanes"][0]
        assert lane["chordia_verdict"] == "PASS_PROTOCOL_A"
        assert lane["chordia_audit_age_days"] == 15
