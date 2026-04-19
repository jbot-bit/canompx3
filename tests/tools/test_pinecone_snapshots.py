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
    """Temp gold.db with minimal validated_setups + edge_families rows for MNQ.

    Covers both portfolio_state and fitness_report snapshot assertions:
    they check for markdown table headers (always present) and at least one
    active instrument name (mentioned only if there's a row). One MNQ row
    per table is sufficient.
    """
    db_path = tmp_path / "gold.db"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id VARCHAR PRIMARY KEY,
            instrument VARCHAR,
            orb_label VARCHAR,
            entry_model VARCHAR,
            orb_minutes INTEGER,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            filter_type VARCHAR,
            expectancy_r DOUBLE,
            win_rate DOUBLE,
            sample_size INTEGER,
            sharpe_ann DOUBLE,
            sharpe_ratio DOUBLE,
            max_drawdown_r DOUBLE,
            status VARCHAR,
            fdr_significant BOOLEAN,
            wf_passed BOOLEAN,
            years_tested INTEGER,
            all_years_positive BOOLEAN,
            yearly_results VARCHAR,
            fdr_adjusted_p DOUBLE,
            wf_windows VARCHAR,
            wfe DOUBLE,
            skewness DOUBLE,
            kurtosis_excess DOUBLE,
            stop_multiplier DOUBLE,
            oos_exp_r DOUBLE,
            noise_risk BOOLEAN
        )
    """)
    con.execute("""
        CREATE TABLE edge_families (
            family_id VARCHAR PRIMARY KEY,
            instrument VARCHAR,
            head_strategy_id VARCHAR,
            robustness_status VARCHAR,
            member_count INTEGER,
            pbo DOUBLE,
            cv_expectancy DOUBLE,
            trade_tier VARCHAR
        )
    """)
    con.execute(
        """
        INSERT INTO validated_setups
            (strategy_id, instrument, orb_label, entry_model, orb_minutes,
             rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
             sample_size, sharpe_ann, sharpe_ratio, max_drawdown_r, status,
             fdr_significant, wf_passed, stop_multiplier, oos_exp_r, noise_risk)
        VALUES ('MNQ_TOKYO_OPEN_E2_CB1_ORB_G5_RR1.5', 'MNQ', 'TOKYO_OPEN', 'E2',
                5, 1.5, 1, 'ORB_G5', 0.18, 0.52, 150, 1.1, 1.1, 3.0, 'active',
                TRUE, TRUE, 1.0, 0.20, FALSE)
        """
    )
    con.execute(
        """
        INSERT INTO edge_families
            (family_id, instrument, head_strategy_id, robustness_status,
             member_count, pbo, cv_expectancy, trade_tier)
        VALUES ('MNQ_TOKYO_OPEN_ORB_G5', 'MNQ',
                'MNQ_TOKYO_OPEN_E2_CB1_ORB_G5_RR1.5', 'ROBUST',
                3, 0.20, 0.15, 'CORE')
        """
    )
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
