"""
Integration tests for the L1 -> L2 pipeline flow.

Tests the full connected chain:
  bars_1m -> build_5m -> build_daily_features -> outcome_builder -> discovery -> validation

Existing coverage:
  - test_trading_app/test_integration.py: L2 only (synthetic daily_features -> L2)
  - test_pipeline/test_full_pipeline.py: Orchestrator metadata only (step ordering)
  - test_trading_app/test_engine_risk_integration.py: Engine <-> Risk lifecycle

This file fills gaps:
  A. L1 pipeline integration (bars_1m -> bars_5m -> daily_features)
  B. Full L1->L2 end-to-end (raw bars through validated strategies)
  C. Multi-instrument isolation (MGC + MNQ in same DB)
  D. Idempotency across full L1+L2 pipeline
"""

import sys
import json
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import pytest
import duckdb

from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
from pipeline.build_bars_5m import build_5m_bars
from pipeline.build_daily_features import build_daily_features
from trading_app.db_manager import init_trading_app_schema
from trading_app.outcome_builder import build_outcomes
from trading_app.strategy_discovery import run_discovery
from trading_app.strategy_validator import run_validation

# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def _insert_bars_1m(con, n_days=30, instrument="MGC", start_year=2024,
                    start_month=1, start_day_of_month=7, base_price=2700.0):
    """
    Insert synthetic bars_1m for n_days of weekday trading days.

    Only inserts bars_1m -- the L1 pipeline builds bars_5m and daily_features.

    Price action per day:
      - First 5 bars (23:00-23:04 UTC): flat range forming ORB at 0900 Brisbane.
        All highs at base + 1.5, all lows at base - 0.5 => ORB size = 2.0
      - Bars 5+: steady uptrend so close > ORB high => triggers long break.
      - 200 bars total per day: enough room for RR4.0 to resolve.

    Args:
        con: Open DuckDB connection.
        n_days: Number of weekday trading days to generate.
        instrument: Symbol to use (MGC, MNQ, etc.).
        start_year: Year for first trading day.
        start_month: Month for first trading day.
        start_day_of_month: Day of month for first trading day (pick a Monday).
        base_price: Starting price level (e.g., 2700 for gold, 18000 for NQ).

    Returns:
        List of trading_day dates created.
    """
    source_sym = "GCG4" if instrument == "MGC" else f"{instrument}H5"
    trading_day = date(start_year, start_month, start_day_of_month)
    days_created = 0
    trading_days = []

    while days_created < n_days:
        if trading_day.weekday() >= 5:
            trading_day += timedelta(days=1)
            continue

        # Trading day starts at 09:00 Brisbane = 23:00 UTC previous calendar day
        td_start = datetime(
            trading_day.year, trading_day.month, trading_day.day,
            tzinfo=timezone.utc,
        ) - timedelta(hours=1)  # 23:00 UTC prev day

        price = base_price + days_created * 0.5
        bars = []

        for i in range(200):
            ts = td_start + timedelta(minutes=i)

            if i < 5:
                # ORB formation: flat range
                o = price
                h = price + 1.5
                l = price - 0.5
                c = price + 0.3
            else:
                # Post-ORB: uptrend to create long break and allow target hit
                o = price + 1.5 + (i - 5) * 0.10
                h = o + 1.5
                l = o - 0.3
                c = o + 1.0

            bars.append((
                ts.isoformat(), instrument, source_sym,
                round(o, 2), round(h, 2), round(l, 2), round(c, 2), 100,
            ))

        con.executemany(
            """INSERT OR REPLACE INTO bars_1m
               (ts_utc, symbol, source_symbol, open, high, low, close, volume)
               VALUES (?::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?)""",
            bars,
        )

        trading_days.append(trading_day)
        trading_day += timedelta(days=1)
        days_created += 1

    con.commit()
    return trading_days

def _create_l1_db(tmp_path, n_days=30, instrument="MGC", base_price=2700.0):
    """Create a temp DB with bars_1m only -- L1 pipeline builds the rest."""
    db_path = tmp_path / "l1_integration.db"
    con = duckdb.connect(str(db_path))

    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    con.close()

    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path))
    trading_days = _insert_bars_1m(
        con, n_days=n_days, instrument=instrument, base_price=base_price,
    )
    con.close()

    return db_path, trading_days

def _run_l1_pipeline(db_path, instrument="MGC", start_date=None, end_date=None):
    """Run L1 pipeline: build_5m -> build_daily_features. Returns (bars_5m_count, features_count)."""
    if start_date is None:
        start_date = date(2024, 1, 1)
    if end_date is None:
        end_date = date(2024, 12, 31)

    con = duckdb.connect(str(db_path))
    try:
        bars_5m_count = build_5m_bars(con, instrument, start_date, end_date, dry_run=False)
        features_count = build_daily_features(
            con, instrument, start_date, end_date, orb_minutes=5, dry_run=False,
        )
    finally:
        con.close()

    return bars_5m_count, features_count

def _run_l2_pipeline(db_path, instrument="MGC", start_date=None, end_date=None,
                     min_sample=5):
    """Run L2 pipeline: outcomes -> discovery -> validation. Returns (outcomes, strategies, passed, rejected)."""
    if start_date is None:
        start_date = date(2024, 1, 1)
    if end_date is None:
        end_date = date(2024, 12, 31)

    outcome_count = build_outcomes(
        db_path=db_path, instrument=instrument,
        start_date=start_date, end_date=end_date,
    )
    strategy_count = run_discovery(
        db_path=db_path, instrument=instrument,
        start_date=start_date, end_date=end_date,
    )
    passed, rejected = run_validation(
        db_path=db_path, instrument=instrument,
        min_sample=min_sample,
        enable_walkforward=False,
    )
    return outcome_count, strategy_count, passed, rejected

# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture(scope="class")
def l1_pipeline_db(tmp_path_factory):
    """30-day DB with L1 pipeline complete: bars_1m -> bars_5m -> daily_features."""
    tmp_dir = tmp_path_factory.mktemp("l1_integ")
    db_path, trading_days = _create_l1_db(tmp_dir, n_days=30)
    bars_5m_count, features_count = _run_l1_pipeline(db_path)
    return db_path, trading_days, bars_5m_count, features_count

@pytest.fixture(scope="class")
def full_pipeline_db(tmp_path_factory):
    """30-day DB with full L1+L2 pipeline complete."""
    tmp_dir = tmp_path_factory.mktemp("full_integ")
    db_path, trading_days = _create_l1_db(tmp_dir, n_days=30)
    bars_5m_count, features_count = _run_l1_pipeline(db_path)

    outcome_count, strategy_count, passed, rejected = _run_l2_pipeline(
        db_path, min_sample=5,
    )
    return {
        "db_path": db_path,
        "trading_days": trading_days,
        "bars_5m_count": bars_5m_count,
        "features_count": features_count,
        "outcome_count": outcome_count,
        "strategy_count": strategy_count,
        "passed": passed,
        "rejected": rejected,
    }

@pytest.fixture(scope="class")
def multi_instrument_db(tmp_path_factory):
    """DB with both MGC and MNQ bars + full pipeline."""
    tmp_dir = tmp_path_factory.mktemp("multi_integ")
    db_path = tmp_dir / "multi.db"

    con = duckdb.connect(str(db_path))
    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    con.close()

    init_trading_app_schema(db_path=db_path)

    # Insert bars for both instruments
    con = duckdb.connect(str(db_path))
    mgc_days = _insert_bars_1m(con, n_days=20, instrument="MGC", base_price=2700.0)
    mnq_days = _insert_bars_1m(con, n_days=20, instrument="MNQ", base_price=18000.0)
    con.close()

    # Run L1 for each instrument
    start_d, end_d = date(2024, 1, 1), date(2024, 12, 31)
    _run_l1_pipeline(db_path, instrument="MGC", start_date=start_d, end_date=end_d)
    _run_l1_pipeline(db_path, instrument="MNQ", start_date=start_d, end_date=end_d)

    # Run L2 for each instrument
    mgc_l2 = _run_l2_pipeline(db_path, instrument="MGC", min_sample=5)
    mnq_l2 = _run_l2_pipeline(db_path, instrument="MNQ", min_sample=5)

    return {
        "db_path": db_path,
        "mgc_days": mgc_days,
        "mnq_days": mnq_days,
        "mgc_l2": mgc_l2,  # (outcomes, strategies, passed, rejected)
        "mnq_l2": mnq_l2,
    }

# =============================================================================
# Test A: L1 Pipeline Integration (bars_1m -> bars_5m -> daily_features)
# =============================================================================

class TestL1Pipeline:
    """Tests the connected L1 flow: bars_1m -> bars_5m -> daily_features."""

    def test_5m_bars_built(self, l1_pipeline_db):
        db_path, _, bars_5m_count, _ = l1_pipeline_db
        assert bars_5m_count > 0

        con = duckdb.connect(str(db_path), read_only=True)
        actual = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        con.close()
        assert actual == bars_5m_count

    def test_5m_count_ratio(self, l1_pipeline_db):
        """bars_5m should be roughly 1/5 of bars_1m (5 1m bars per 5m bar)."""
        db_path, _, _, _ = l1_pipeline_db

        con = duckdb.connect(str(db_path), read_only=True)
        count_1m = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
        count_5m = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        con.close()

        ratio = count_1m / count_5m
        # Should be close to 5 (exact depends on bucket alignment)
        assert 4.0 <= ratio <= 6.0, f"1m/5m ratio {ratio:.1f} outside expected range"

    def test_daily_features_built(self, l1_pipeline_db):
        db_path, trading_days, _, features_count = l1_pipeline_db
        assert features_count > 0
        assert features_count == len(trading_days)

    def test_orb_columns_populated(self, l1_pipeline_db):
        """ORB CME_REOPEN columns should be populated (our synthetic data targets this session)."""
        db_path, _, _, _ = l1_pipeline_db

        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute("""
            SELECT orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_size
            FROM daily_features
            WHERE orb_CME_REOPEN_high IS NOT NULL
        """).fetchall()
        con.close()

        assert len(rows) > 0, "No rows with populated ORB CME_REOPEN columns"
        for high, low, size in rows:
            assert high > low
            assert size > 0
            assert abs(size - (high - low)) < 0.01

    def test_break_detected(self, l1_pipeline_db):
        """Synthetic uptrend data should produce long breaks at CME_REOPEN."""
        db_path, _, _, _ = l1_pipeline_db

        con = duckdb.connect(str(db_path), read_only=True)
        long_breaks = con.execute("""
            SELECT COUNT(*) FROM daily_features
            WHERE orb_CME_REOPEN_break_dir = 'long'
        """).fetchone()[0]
        con.close()

        assert long_breaks > 0, "No long breaks detected on synthetic uptrend data"

    def test_5m_alignment(self, l1_pipeline_db):
        """All bars_5m timestamps should be 5-minute aligned."""
        db_path, _, _, _ = l1_pipeline_db

        con = duckdb.connect(str(db_path), read_only=True)
        misaligned = con.execute("""
            SELECT COUNT(*) FROM bars_5m
            WHERE EXTRACT(EPOCH FROM ts_utc)::BIGINT % 300 != 0
        """).fetchone()[0]
        con.close()

        assert misaligned == 0, f"{misaligned} bars_5m timestamps not 5-min aligned"

    def test_bar_count_populated(self, l1_pipeline_db):
        """Every daily_features row should have bar_count_1m > 0."""
        db_path, _, _, _ = l1_pipeline_db

        con = duckdb.connect(str(db_path), read_only=True)
        bad_rows = con.execute("""
            SELECT COUNT(*) FROM daily_features
            WHERE bar_count_1m IS NULL OR bar_count_1m <= 0
        """).fetchone()[0]
        con.close()

        assert bad_rows == 0

# =============================================================================
# Test B: Full L1 -> L2 End-to-End
# =============================================================================

class TestFullL1L2:
    """Tests the complete pipeline: bars_1m -> ... -> validated_setups."""

    def test_outcomes_from_real_features(self, full_pipeline_db):
        """Outcomes should exist and reference real daily_features rows."""
        db_path = full_pipeline_db["db_path"]
        assert full_pipeline_db["outcome_count"] > 0

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        con.close()
        assert count > 0

    def test_discovery_produces_strategies(self, full_pipeline_db):
        assert full_pipeline_db["strategy_count"] > 0

    def test_validation_processes_all(self, full_pipeline_db):
        """passed + rejected should equal the number of canonical strategies."""
        db_path = full_pipeline_db["db_path"]
        passed = full_pipeline_db["passed"]
        rejected = full_pipeline_db["rejected"]

        con = duckdb.connect(str(db_path), read_only=True)
        # Count canonical strategies (aliases are not independently validated)
        canonical_count = con.execute("""
            SELECT COUNT(*) FROM experimental_strategies
            WHERE instrument = 'MGC'
            AND is_canonical = TRUE
        """).fetchone()[0]
        con.close()

        assert passed + rejected == canonical_count

    def test_no_orphan_outcomes(self, full_pipeline_db):
        """Every outcome should have a matching daily_features row."""
        db_path = full_pipeline_db["db_path"]

        con = duckdb.connect(str(db_path), read_only=True)
        orphans = con.execute("""
            SELECT COUNT(*) FROM orb_outcomes o
            WHERE NOT EXISTS (
                SELECT 1 FROM daily_features df
                WHERE df.symbol = o.symbol
                AND df.trading_day = o.trading_day
                AND df.orb_minutes = o.orb_minutes
            )
        """).fetchone()[0]
        con.close()

        assert orphans == 0, f"{orphans} orphan outcomes without matching daily_features"

    def test_yearly_results_valid_json(self, full_pipeline_db):
        """All strategies should have parseable yearly_results JSON."""
        db_path = full_pipeline_db["db_path"]

        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute(
            "SELECT yearly_results FROM experimental_strategies LIMIT 20"
        ).fetchall()
        con.close()

        assert len(rows) > 0
        for (yearly_json,) in rows:
            data = json.loads(yearly_json)
            assert isinstance(data, dict)
            for year_key, year_data in data.items():
                assert "trades" in year_data
                assert "wins" in year_data
                assert "total_r" in year_data

    def test_data_contract_chain(self, full_pipeline_db):
        """Spot-check: pick 1 strategy, trace it back through the pipeline."""
        db_path = full_pipeline_db["db_path"]

        con = duckdb.connect(str(db_path), read_only=True)

        # Pick a strategy with trades
        strat = con.execute("""
            SELECT strategy_id, orb_label, rr_target, confirm_bars, entry_model, sample_size
            FROM experimental_strategies
            WHERE sample_size > 0
            LIMIT 1
        """).fetchone()

        if strat is None:
            pytest.skip("No strategies with trades in synthetic data")

        strategy_id, orb_label, rr_target, confirm_bars, entry_model, sample_size = strat

        # Verify outcomes exist for this strategy's parameters
        outcome_count = con.execute("""
            SELECT COUNT(*) FROM orb_outcomes
            WHERE symbol = 'MGC'
            AND orb_label = ?
            AND rr_target = ?
            AND confirm_bars = ?
            AND entry_model = ?
        """, [orb_label, rr_target, confirm_bars, entry_model]).fetchone()[0]

        # Verify daily_features exist
        feature_count = con.execute(
            "SELECT COUNT(*) FROM daily_features WHERE symbol = 'MGC'"
        ).fetchone()[0]

        # Verify bars_1m exist
        bars_count = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MGC'"
        ).fetchone()[0]

        con.close()

        assert bars_count > 0, "No bars_1m for tracing"
        assert feature_count > 0, "No daily_features for tracing"
        assert outcome_count > 0, "No outcomes matching strategy params"

# =============================================================================
# Test C: Multi-Instrument Isolation
# =============================================================================

class TestMultiInstrument:
    """Tests MGC + MNQ in the same DB don't contaminate each other."""

    def test_no_cross_contamination_outcomes(self, multi_instrument_db):
        """MGC outcomes should only reference MGC features, and vice versa."""
        db_path = multi_instrument_db["db_path"]

        con = duckdb.connect(str(db_path), read_only=True)

        # Check MGC outcomes don't reference MNQ days
        mgc_orphans = con.execute("""
            SELECT COUNT(*) FROM orb_outcomes o
            WHERE o.symbol = 'MGC'
            AND NOT EXISTS (
                SELECT 1 FROM daily_features df
                WHERE df.symbol = 'MGC'
                AND df.trading_day = o.trading_day
                AND df.orb_minutes = o.orb_minutes
            )
        """).fetchone()[0]

        # Check MNQ outcomes don't reference MGC days
        mnq_orphans = con.execute("""
            SELECT COUNT(*) FROM orb_outcomes o
            WHERE o.symbol = 'MNQ'
            AND NOT EXISTS (
                SELECT 1 FROM daily_features df
                WHERE df.symbol = 'MNQ'
                AND df.trading_day = o.trading_day
                AND df.orb_minutes = o.orb_minutes
            )
        """).fetchone()[0]

        con.close()

        assert mgc_orphans == 0, f"MGC has {mgc_orphans} orphan outcomes"
        assert mnq_orphans == 0, f"MNQ has {mnq_orphans} orphan outcomes"

    def test_independent_counts(self, multi_instrument_db):
        """Each instrument should have independent row counts at every stage."""
        db_path = multi_instrument_db["db_path"]

        con = duckdb.connect(str(db_path), read_only=True)

        for table in ["bars_1m", "bars_5m", "daily_features", "orb_outcomes"]:
            mgc_count = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE symbol = 'MGC'"
            ).fetchone()[0]
            mnq_count = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE symbol = 'MNQ'"
            ).fetchone()[0]

            assert mgc_count > 0, f"MGC has 0 rows in {table}"
            assert mnq_count > 0, f"MNQ has 0 rows in {table}"

        con.close()

    def test_strategy_ids_prefixed(self, multi_instrument_db):
        """Strategy IDs should contain instrument prefix for disambiguation."""
        db_path = multi_instrument_db["db_path"]

        con = duckdb.connect(str(db_path), read_only=True)

        mgc_ids = con.execute("""
            SELECT strategy_id FROM experimental_strategies
            WHERE instrument = 'MGC' LIMIT 5
        """).fetchall()
        mnq_ids = con.execute("""
            SELECT strategy_id FROM experimental_strategies
            WHERE instrument = 'MNQ' LIMIT 5
        """).fetchall()

        con.close()

        for (sid,) in mgc_ids:
            assert "MGC" in sid, f"MGC strategy_id '{sid}' missing instrument prefix"
        for (sid,) in mnq_ids:
            assert "MNQ" in sid, f"MNQ strategy_id '{sid}' missing instrument prefix"

    def test_bars_price_ranges_distinct(self, multi_instrument_db):
        """MGC bars (~2700) and MNQ bars (~18000) should be in different price ranges."""
        db_path = multi_instrument_db["db_path"]

        con = duckdb.connect(str(db_path), read_only=True)

        mgc_avg = con.execute(
            "SELECT AVG(close) FROM bars_1m WHERE symbol = 'MGC'"
        ).fetchone()[0]
        mnq_avg = con.execute(
            "SELECT AVG(close) FROM bars_1m WHERE symbol = 'MNQ'"
        ).fetchone()[0]

        con.close()

        assert mgc_avg < 5000, f"MGC avg price {mgc_avg} unexpectedly high"
        assert mnq_avg > 10000, f"MNQ avg price {mnq_avg} unexpectedly low"

# =============================================================================
# Test D: Idempotency Across Full Pipeline
# =============================================================================

class TestFullPipelineIdempotent:
    """Tests that re-running pipeline stages produces identical results.

    L1 and L2 are tested separately because daily_features has FK references
    from orb_outcomes, so L1 DELETE+INSERT requires clearing L2 first.
    """

    def test_idempotent_l1(self, tmp_path):
        """L1 stages (build_5m, build_features) are idempotent when re-run."""
        db_path, _ = _create_l1_db(tmp_path, n_days=15)
        start_d, end_d = date(2024, 1, 1), date(2024, 12, 31)

        # First L1 run
        _run_l1_pipeline(db_path, start_date=start_d, end_date=end_d)

        con = duckdb.connect(str(db_path), read_only=True)
        bars_5m_1 = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        features_1 = con.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
        con.close()

        # Second L1 run (no L2 data, so no FK conflict)
        _run_l1_pipeline(db_path, start_date=start_d, end_date=end_d)

        con = duckdb.connect(str(db_path), read_only=True)
        bars_5m_2 = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        features_2 = con.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
        con.close()

        assert bars_5m_1 == bars_5m_2, f"bars_5m: {bars_5m_1} -> {bars_5m_2}"
        assert features_1 == features_2, f"daily_features: {features_1} -> {features_2}"

    def test_idempotent_l2(self, tmp_path):
        """L2 stages (outcomes, discovery) are idempotent when re-run."""
        db_path, _ = _create_l1_db(tmp_path, n_days=15)
        start_d, end_d = date(2024, 1, 1), date(2024, 12, 31)

        # Build L1 once
        _run_l1_pipeline(db_path, start_date=start_d, end_date=end_d)

        # First L2 run
        build_outcomes(db_path=db_path, instrument="MGC",
                       start_date=start_d, end_date=end_d)
        run_discovery(db_path=db_path, instrument="MGC",
                      start_date=start_d, end_date=end_d)

        con = duckdb.connect(str(db_path), read_only=True)
        outcomes_1 = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        strats_1 = con.execute("SELECT COUNT(*) FROM experimental_strategies").fetchone()[0]
        con.close()

        # Second L2 run (idempotent INSERT OR REPLACE)
        build_outcomes(db_path=db_path, instrument="MGC",
                       start_date=start_d, end_date=end_d)
        run_discovery(db_path=db_path, instrument="MGC",
                      start_date=start_d, end_date=end_d)

        con = duckdb.connect(str(db_path), read_only=True)
        outcomes_2 = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        strats_2 = con.execute("SELECT COUNT(*) FROM experimental_strategies").fetchone()[0]
        con.close()

        assert outcomes_1 == outcomes_2, f"orb_outcomes: {outcomes_1} -> {outcomes_2}"
        assert strats_1 == strats_2, f"experimental_strategies: {strats_1} -> {strats_2}"
