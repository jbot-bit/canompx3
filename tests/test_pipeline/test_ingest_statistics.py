"""
Tests for pipeline.ingest_statistics module.

Tests contract pattern matching, stat type mappings, and schema integrity.
Does NOT test DBN file reading (requires real Databento files).
"""

import duckdb
import pytest

from pipeline.ingest_statistics import OUTRIGHT_PATTERNS, STAT_TYPES


class TestOutrightPatterns:
    """OUTRIGHT_PATTERNS correctly identify valid futures contract symbols."""

    def test_all_active_instruments_covered(self):
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        for inst in ACTIVE_ORB_INSTRUMENTS:
            assert inst in OUTRIGHT_PATTERNS, f"{inst} missing from OUTRIGHT_PATTERNS"

    def test_mes_matches_micro(self):
        patterns = OUTRIGHT_PATTERNS["MES"]
        micro_pat = patterns[1][0]  # MES pattern
        assert micro_pat.match("MESH6")
        assert micro_pat.match("MESZ5")
        assert micro_pat.match("MESM25")

    def test_mes_matches_parent(self):
        patterns = OUTRIGHT_PATTERNS["MES"]
        parent_pat = patterns[0][0]  # ES pattern
        assert parent_pat.match("ESH6")
        assert parent_pat.match("ESZ25")

    def test_mes_rejects_spreads(self):
        patterns = OUTRIGHT_PATTERNS["MES"]
        for pat, _ in patterns:
            assert not pat.match("ESH6-ESM6")
            assert not pat.match("MESH6-MESM6")

    def test_mgc_matches_micro(self):
        patterns = OUTRIGHT_PATTERNS["MGC"]
        micro_pat = patterns[1][0]  # MGC pattern
        assert micro_pat.match("MGCG6")
        assert micro_pat.match("MGCZ25")

    def test_mgc_matches_parent(self):
        patterns = OUTRIGHT_PATTERNS["MGC"]
        parent_pat = patterns[0][0]  # GC pattern
        assert parent_pat.match("GCG6")
        assert parent_pat.match("GCZ5")

    def test_mnq_matches_micro(self):
        patterns = OUTRIGHT_PATTERNS["MNQ"]
        micro_pat = patterns[1][0]  # MNQ pattern
        assert micro_pat.match("MNQH6")
        assert micro_pat.match("MNQM25")

    def test_mnq_matches_parent(self):
        patterns = OUTRIGHT_PATTERNS["MNQ"]
        parent_pat = patterns[0][0]  # NQ pattern
        assert parent_pat.match("NQH6")
        assert parent_pat.match("NQZ25")

    def test_rejects_invalid_months(self):
        """Month codes are limited to valid futures expiry months."""
        patterns = OUTRIGHT_PATTERNS["MNQ"]
        for pat, _ in patterns:
            # 'A' is not a valid month code
            assert not pat.match("MNQA6")
            assert not pat.match("NQA6")

    def test_prefix_lengths_correct(self):
        """Prefix lengths match expected symbol structure."""
        for inst, pats in OUTRIGHT_PATTERNS.items():
            for pat, plen in pats:
                # Parent symbols have shorter prefix (2), micros have 3
                assert plen in (2, 3), f"{inst} has unexpected prefix length {plen}"


class TestStatTypes:
    """STAT_TYPES maps Databento stat type codes to column names."""

    def test_required_types_present(self):
        expected = {"settlement", "session_low", "session_high", "cleared_volume", "open_interest"}
        actual = set(STAT_TYPES.values())
        assert expected.issubset(actual), f"Missing: {expected - actual}"

    def test_no_duplicate_names(self):
        names = list(STAT_TYPES.values())
        assert len(names) == len(set(names)), "Duplicate stat type names"

    def test_codes_are_integers(self):
        for code in STAT_TYPES:
            assert isinstance(code, int)


class TestExchangeStatisticsSchema:
    """exchange_statistics table schema matches ingest expectations."""

    def test_table_creation(self, tmp_path):
        from pipeline.init_db import EXCHANGE_STATISTICS_SCHEMA

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute(EXCHANGE_STATISTICS_SCHEMA)
        cols = {r[0] for r in con.execute("DESCRIBE exchange_statistics").fetchall()}
        expected = {
            "cal_date", "symbol", "session_high", "session_low", "settlement",
            "opening_price", "indicative_open", "cleared_volume", "open_interest",
            "total_cleared_volume", "front_contract",
        }
        assert expected == cols
        con.close()

    def test_primary_key_enforced(self, tmp_path):
        from pipeline.init_db import EXCHANGE_STATISTICS_SCHEMA

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute(EXCHANGE_STATISTICS_SCHEMA)
        con.execute(
            "INSERT INTO exchange_statistics (cal_date, symbol) VALUES ('2024-01-01', 'MNQ')"
        )
        with pytest.raises(duckdb.ConstraintException):
            con.execute(
                "INSERT INTO exchange_statistics (cal_date, symbol) VALUES ('2024-01-01', 'MNQ')"
            )
        con.close()

    def test_idempotent_delete_insert(self, tmp_path):
        """Verify the DELETE+INSERT pattern used by ingest_statistics."""
        from pipeline.init_db import EXCHANGE_STATISTICS_SCHEMA

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute(EXCHANGE_STATISTICS_SCHEMA)
        # First insert
        con.execute(
            "INSERT INTO exchange_statistics (cal_date, symbol, settlement) "
            "VALUES ('2024-01-01', 'MNQ', 100.0)"
        )
        # Idempotent re-insert
        con.execute("DELETE FROM exchange_statistics WHERE symbol = 'MNQ'")
        con.execute(
            "INSERT INTO exchange_statistics (cal_date, symbol, settlement) "
            "VALUES ('2024-01-01', 'MNQ', 105.0)"
        )
        result = con.execute(
            "SELECT settlement FROM exchange_statistics WHERE symbol = 'MNQ'"
        ).fetchone()
        assert result[0] == 105.0
        con.close()


class TestPitRangeAtrColumn:
    """pit_range_atr column in daily_features is correctly defined."""

    def test_column_exists_after_init(self, tmp_path):
        """init_db creates pit_range_atr column via migration."""
        from pipeline.init_db import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)
        con = duckdb.connect(str(db_path), read_only=True)
        cols = {r[0] for r in con.execute("DESCRIBE daily_features").fetchall()}
        assert "pit_range_atr" in cols
        con.close()

    def test_column_type_is_double(self, tmp_path):
        from pipeline.init_db import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)
        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute("DESCRIBE daily_features").fetchall()
        col_types = {r[0]: r[1] for r in rows}
        assert col_types["pit_range_atr"] == "DOUBLE"
        con.close()
