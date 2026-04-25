"""Tests for scripts/tools/pinecone_snapshots.py snapshot generators."""

from pathlib import Path

import duckdb
import pytest

from scripts.tools.pinecone_snapshots import (
    generate_fitness_report_snapshot,
    generate_live_config_snapshot,
    generate_portfolio_state_snapshot,
    generate_research_index_snapshot,
)


@pytest.fixture
def seeded_snapshot_db(tmp_path):
    """Temp gold.db with one ROBUST MNQ candidate seeded against canonical schema.

    Schema is built via canonical builders (pipeline.init_db.DAILY_FEATURES_SCHEMA
    + trading_app.db_manager.init_trading_app_schema) so future schema migrations
    automatically flow into this fixture without manual sync.

    Covers both portfolio_state and fitness_report snapshot assertions:
    they check for markdown table headers (always present) and at least one
    active instrument name (mentioned only if there's a row).
    """
    from pipeline.init_db import DAILY_FEATURES_SCHEMA
    from trading_app.db_manager import init_trading_app_schema

    db_path = tmp_path / "gold.db"
    # daily_features must exist first because orb_outcomes has an FK into it.
    con = duckdb.connect(str(db_path))
    try:
        con.execute(DAILY_FEATURES_SCHEMA)
    finally:
        con.close()
    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            INSERT INTO validated_setups (
                strategy_id, instrument, orb_label, entry_model, orb_minutes,
                rr_target, confirm_bars, filter_type, status,
                sample_size, win_rate, expectancy_r, sharpe_ann, max_drawdown_r,
                years_tested, all_years_positive, stress_test_passed,
                fdr_significant, wf_passed, stop_multiplier
            ) VALUES ('MNQ_TOKYO_OPEN_E2_CB1_ORB_G5_RR1.5', 'MNQ', 'TOKYO_OPEN',
                      'E2', 5, 1.5, 1, 'ORB_G5', 'active',
                      150, 0.52, 0.18, 1.1, 3.0,
                      6, TRUE, TRUE,
                      TRUE, TRUE, 1.0)
            """
        )
        con.execute(
            """
            INSERT INTO edge_families (
                family_hash, instrument, member_count, trade_day_count,
                head_strategy_id, head_expectancy_r, head_sharpe_ann,
                robustness_status, cv_expectancy, trade_tier, pbo
            ) VALUES ('MNQ_TOKYO_OPEN_ORB_G5', 'MNQ', 3, 150,
                      'MNQ_TOKYO_OPEN_E2_CB1_ORB_G5_RR1.5', 0.18, 1.1,
                      'ROBUST', 0.15, 'CORE', 0.20)
            """
        )
    finally:
        con.close()
    return db_path


def test_portfolio_state_snapshot(seeded_snapshot_db):
    """Verify markdown has Portfolio State header and instrument table."""
    md = generate_portfolio_state_snapshot(db_path=seeded_snapshot_db)

    assert "# Portfolio State Snapshot" in md
    assert "Generated:" in md

    # Must have strategy table headers
    assert "Instrument" in md
    assert "Active" in md
    assert "FDR" in md
    assert "CORE" in md
    assert "REGIME" in md

    # Must have edge families section
    assert "## Edge Families by Instrument" in md
    assert "ROBUST" in md
    assert "WHITELISTED" in md

    # Must have totals row
    assert "**TOTAL**" in md

    # Must mention at least one known instrument
    assert any(inst in md for inst in ("MGC", "MNQ", "MES", "M2K"))


def test_fitness_report_snapshot(seeded_snapshot_db):
    """Verify markdown has breakdown sections and top strategies."""
    md = generate_fitness_report_snapshot(db_path=seeded_snapshot_db)

    assert "# Fitness Report Snapshot" in md
    assert "Generated:" in md

    # Must have breakdown table
    assert "## Strategy Breakdown" in md
    assert "Session" in md
    assert "Entry" in md
    assert "Aperture" in md
    assert "Avg ExpR" in md
    assert "Avg Sharpe" in md

    # Must have top 10 section
    assert "## Top 10 Strategies" in md
    assert "Strategy ID" in md

    # Must mention at least one known instrument
    assert any(inst in md for inst in ("MGC", "MNQ", "MES", "M2K"))


def test_live_config_snapshot():
    """Verify markdown lists specs from LIVE_PORTFOLIO."""
    from trading_app.live_config import LIVE_PORTFOLIO

    md = generate_live_config_snapshot()

    assert "# Live Config Snapshot" in md
    assert "Generated:" in md

    # Must have gates section
    assert "## Portfolio Gates" in md
    assert "LIVE_MIN_EXPECTANCY_R" in md
    assert "LIVE_MIN_EXPECTANCY_DOLLARS_MULT" in md

    # Must have tier summary
    assert "## Tier Summary" in md
    assert "CORE" in md

    # Must have specs table
    assert "## Strategy Specs" in md
    assert "Family ID" in md
    assert "Filter" in md

    # Must list at least one actual spec from LIVE_PORTFOLIO
    assert any(spec.family_id in md for spec in LIVE_PORTFOLIO)

    # Total count must match
    assert f"**{len(LIVE_PORTFOLIO)}**" in md


def test_research_index_snapshot():
    """Verify markdown has file listing from research/output/."""
    md = generate_research_index_snapshot()

    assert "# Research Index Snapshot" in md

    # Should have file count
    assert "Total files:" in md

    # Should have at least one file table
    assert "File" in md
    assert "Size" in md
    assert "Summary" in md

    # Should list at least one known research output file
    # (we know HIGH_LEVEL_AUDIT_2026-02-22.md exists)
    assert ".md" in md or ".txt" in md
