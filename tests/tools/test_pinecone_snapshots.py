"""Tests for scripts/tools/pinecone_snapshots.py snapshot generators."""

from pathlib import Path

import pytest

# Import generators — these read from the live gold.db
from scripts.tools.pinecone_snapshots import (
    generate_portfolio_state_snapshot,
    generate_fitness_report_snapshot,
    generate_live_config_snapshot,
    generate_research_index_snapshot,
)


def test_portfolio_state_snapshot():
    """Verify markdown has Portfolio State header and instrument table."""
    md = generate_portfolio_state_snapshot()

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


def test_fitness_report_snapshot():
    """Verify markdown has breakdown sections and top strategies."""
    md = generate_fitness_report_snapshot()

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
