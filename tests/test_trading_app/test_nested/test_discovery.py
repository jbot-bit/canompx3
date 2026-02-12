"""
Tests for trading_app.nested.discovery — strategy ID format, grid iteration,
and nested table isolation.
"""

import sys
from pathlib import Path
from datetime import date

import pytest
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from trading_app.nested.discovery import (
    make_nested_strategy_id,
    ENTRY_RESOLUTION,
)
from trading_app.nested.schema import init_nested_schema


# ============================================================================
# Strategy ID format tests
# ============================================================================

class TestNestedStrategyId:
    """Tests for make_nested_strategy_id() format."""

    def test_format_15m(self):
        sid = make_nested_strategy_id("MGC", "0900", 15, "E1", 2.5, 2, "ORB_G4")
        assert sid == "NESTED_MGC_0900_15m_E1_RR2.5_CB2_ORB_G4"

    def test_format_30m(self):
        sid = make_nested_strategy_id("MGC", "1800", 30, "E3", 2.0, 5, "ORB_G5")
        assert sid == "NESTED_MGC_1800_30m_E3_RR2.0_CB5_ORB_G5"

    def test_format_no_filter(self):
        sid = make_nested_strategy_id("MGC", "1000", 15, "E1", 1.0, 1, "NO_FILTER")
        assert sid == "NESTED_MGC_1000_15m_E1_RR1.0_CB1_NO_FILTER"

    def test_nested_prefix_distinguishes_from_baseline(self):
        """Nested IDs start with NESTED_, baseline IDs don't."""
        nested_id = make_nested_strategy_id("MGC", "0900", 15, "E1", 2.5, 2, "ORB_G4")
        assert nested_id.startswith("NESTED_")

        # Baseline format for comparison (from strategy_discovery.make_strategy_id)
        from trading_app.strategy_discovery import make_strategy_id
        baseline_id = make_strategy_id("MGC", "0900", "E1", 2.5, 2, "ORB_G4")
        assert not baseline_id.startswith("NESTED_")

    def test_includes_orb_minutes(self):
        """Strategy ID embeds the ORB duration for disambiguation."""
        sid_15 = make_nested_strategy_id("MGC", "0900", 15, "E1", 2.5, 2, "ORB_G4")
        sid_30 = make_nested_strategy_id("MGC", "0900", 30, "E1", 2.5, 2, "ORB_G4")
        assert "15m" in sid_15
        assert "30m" in sid_30
        assert sid_15 != sid_30


class TestEntryResolution:
    """Verify the entry resolution constant."""

    def test_entry_resolution_is_5(self):
        assert ENTRY_RESOLUTION == 5


# ============================================================================
# Table isolation tests
# ============================================================================

class TestTableIsolation:
    """Verify nested tables don't interfere with production tables."""

    @pytest.fixture
    def db_with_both_schemas(self, tmp_path):
        """Create a DB with both production and nested schemas."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        # Minimal daily_features (required parent)
        con.execute("""
            CREATE TABLE daily_features (
                symbol TEXT NOT NULL,
                trading_day DATE NOT NULL,
                orb_minutes INTEGER NOT NULL,
                bar_count_1m INTEGER,
                PRIMARY KEY (symbol, trading_day, orb_minutes)
            )
        """)

        # Production tables
        con.execute("""
            CREATE TABLE experimental_strategies (
                strategy_id TEXT PRIMARY KEY,
                instrument TEXT,
                orb_label TEXT,
                orb_minutes INTEGER,
                sample_size INTEGER
            )
        """)

        con.commit()
        con.close()

        # Init nested schema
        init_nested_schema(db_path=db_path)

        return db_path

    def test_nested_insert_does_not_affect_production(self, db_with_both_schemas):
        """Inserting into nested_strategies does not insert into experimental_strategies."""
        db_path = db_with_both_schemas

        con = duckdb.connect(str(db_path))
        try:
            con.execute("""
                INSERT INTO nested_strategies
                (strategy_id, instrument, orb_label, orb_minutes, entry_resolution,
                 rr_target, confirm_bars, entry_model)
                VALUES ('NESTED_TEST', 'MGC', '0900', 15, 5, 2.5, 2, 'E1')
            """)
            con.commit()

            # nested_strategies should have the row
            n_nested = con.execute(
                "SELECT COUNT(*) FROM nested_strategies"
            ).fetchone()[0]
            assert n_nested == 1

            # experimental_strategies should be empty
            n_prod = con.execute(
                "SELECT COUNT(*) FROM experimental_strategies"
            ).fetchone()[0]
            assert n_prod == 0
        finally:
            con.close()

    def test_nested_and_production_pks_are_independent(self, db_with_both_schemas):
        """Same logical strategy can exist in both tables with different IDs."""
        db_path = db_with_both_schemas

        con = duckdb.connect(str(db_path))
        try:
            con.execute("""
                INSERT INTO experimental_strategies
                (strategy_id, instrument, orb_label, orb_minutes, sample_size)
                VALUES ('MGC_0900_E1_RR2.5_CB2_ORB_G4', 'MGC', '0900', 5, 100)
            """)
            con.execute("""
                INSERT INTO nested_strategies
                (strategy_id, instrument, orb_label, orb_minutes, entry_resolution,
                 rr_target, confirm_bars, entry_model)
                VALUES ('NESTED_MGC_0900_15m_E1_RR2.5_CB2_ORB_G4', 'MGC', '0900', 15, 5, 2.5, 2, 'E1')
            """)
            con.commit()

            assert con.execute("SELECT COUNT(*) FROM experimental_strategies").fetchone()[0] == 1
            assert con.execute("SELECT COUNT(*) FROM nested_strategies").fetchone()[0] == 1
        finally:
            con.close()
