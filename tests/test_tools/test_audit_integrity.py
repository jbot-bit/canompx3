"""
WS2: Tests for data integrity audit (scripts/tools/audit_integrity.py).

Covers all 10 checks. Each test creates a temp DuckDB with injected data
and passes the connection directly (checks take `con` parameter).
"""

from contextlib import contextmanager
from pathlib import Path

import duckdb
import pytest

from scripts.tools import audit_integrity

# ── Shared schemas ───────────────────────────────────────────────────

VALIDATED_SCHEMA = """
    CREATE TABLE validated_setups (
        strategy_id VARCHAR PRIMARY KEY,
        instrument VARCHAR,
        orb_label VARCHAR,
        orb_minutes INTEGER,
        entry_model VARCHAR,
        confirm_bars INTEGER,
        filter_type VARCHAR,
        rr_target DOUBLE,
        stop_multiplier DOUBLE DEFAULT 1.0,
        status VARCHAR,
        win_rate DOUBLE,
        expectancy_r DOUBLE,
        fdr_significant BOOLEAN,
        family_hash VARCHAR,
        wf_tested BOOLEAN,
        retired_at TIMESTAMPTZ,
        retirement_reason VARCHAR
    );
"""

OUTCOMES_SCHEMA = """
    CREATE TABLE orb_outcomes (
        trading_day DATE,
        symbol VARCHAR,
        orb_minutes INTEGER,
        orb_label VARCHAR,
        entry_model VARCHAR,
        confirm_bars INTEGER
    );
"""

EXPERIMENTAL_SCHEMA = """
    CREATE TABLE experimental_strategies (
        instrument VARCHAR,
        strategy_id VARCHAR,
        entry_model VARCHAR,
        sample_size INTEGER,
        orb_minutes INTEGER,
        orb_label VARCHAR,
        confirm_bars INTEGER
    );
"""

ALL_SCHEMAS = VALIDATED_SCHEMA + OUTCOMES_SCHEMA + EXPERIMENTAL_SCHEMA


@contextmanager
def _make_con(tmp_path, schemas=ALL_SCHEMAS, inserts=""):
    """Create temp DuckDB and yield connection (auto-closes on exit)."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(schemas)
        if inserts:
            con.execute(inserts)
        yield con
    finally:
        con.close()


# ── Check 1: Outcome coverage ───────────────────────────────────────


class TestOutcomeCoverage:
    """Check 1: orb_outcomes sessions must match enabled_sessions."""

    def test_catches_missing_session(self, tmp_path, monkeypatch):
        """If enabled_sessions has a session not in DB → violation."""
        with _make_con(tmp_path) as con:
            con.execute("""
                INSERT INTO orb_outcomes VALUES
                ('2025-01-01', 'GC', 5, 'CME_REOPEN', 'E2', 1)
            """)
            monkeypatch.setattr(audit_integrity, "ACTIVE_INSTRUMENTS", ["MGC"])
            violations = audit_integrity.check_outcome_coverage(con)
            assert len(violations) > 0
            assert "missing" in violations[0]

    def test_passes_full_coverage(self, tmp_path, monkeypatch):
        """All enabled sessions present in DB → clean."""
        from pipeline.asset_configs import ASSET_CONFIGS

        with _make_con(tmp_path) as con:
            cfg = ASSET_CONFIGS["MGC"]
            for sess in cfg["enabled_sessions"]:
                con.execute(
                    "INSERT INTO orb_outcomes VALUES (?, ?, 5, ?, 'E2', 1)",
                    ["2025-01-01", cfg["symbol"], sess],
                )
            monkeypatch.setattr(audit_integrity, "ACTIVE_INSTRUMENTS", ["MGC"])
            violations = audit_integrity.check_outcome_coverage(con)
            assert len(violations) == 0


# ── Check 2: Validated session integrity ─────────────────────────────


class TestValidatedSessionIntegrity:
    """Check 2: Active validated_setups sessions must be in enabled_sessions."""

    def test_catches_invalid_session(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, status)
            VALUES ('s1', 'MGC', 'FAKE_SESSION', 'active')
        """,
        ) as con:
            violations = audit_integrity.check_validated_session_integrity(con)
            assert len(violations) > 0
            assert "FAKE_SESSION" in violations[0]

    def test_passes_valid_session(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, status)
            VALUES ('s1', 'MGC', 'CME_REOPEN', 'active')
        """,
        ) as con:
            violations = audit_integrity.check_validated_session_integrity(con)
            assert len(violations) == 0


# ── Check 3: Edge family integrity ───────────────────────────────────


class TestEdgeFamilyIntegrity:
    """Check 3: No active strategies with NULL family_hash."""

    def test_catches_null_family_hash(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, instrument, status, family_hash)
            VALUES ('s1', 'MGC', 'active', NULL)
        """,
        ) as con:
            violations = audit_integrity.check_edge_family_integrity(con)
            assert len(violations) > 0
            assert "NULL family_hash" in violations[0]

    def test_passes_with_family_hash(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, instrument, status, family_hash)
            VALUES ('s1', 'MGC', 'active', 'abc123')
        """,
        ) as con:
            violations = audit_integrity.check_edge_family_integrity(con)
            assert len(violations) == 0


# ── Check 4: E0 contamination ───────────────────────────────────────


class TestE0Contamination:
    """Check 4: No E0 rows in any trading table."""

    def test_catches_e0_in_outcomes(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO orb_outcomes VALUES
            ('2025-01-01', 'GC', 5, 'CME_REOPEN', 'E0', 1)
        """,
        ) as con:
            violations = audit_integrity.check_e0_contamination(con)
            assert len(violations) > 0
            assert "E0" in violations[0]

    def test_catches_e0_in_validated(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, entry_model)
            VALUES ('s1', 'E0')
        """,
        ) as con:
            violations = audit_integrity.check_e0_contamination(con)
            assert len(violations) > 0

    def test_passes_no_e0(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO orb_outcomes VALUES
            ('2025-01-01', 'GC', 5, 'CME_REOPEN', 'E2', 1)
        """,
        ) as con:
            violations = audit_integrity.check_e0_contamination(con)
            assert len(violations) == 0


# ── Check 5: Old session names ───────────────────────────────────────


class TestOldSessionNames:
    """Check 5: No old fixed-clock session names in DB."""

    def test_catches_old_0900(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO orb_outcomes VALUES
            ('2025-01-01', 'GC', 5, '0900', 'E2', 1)
        """,
        ) as con:
            violations = audit_integrity.check_old_session_names(con)
            assert len(violations) > 0
            assert "0900" in violations[0]

    def test_catches_old_1800(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO experimental_strategies VALUES
            ('MGC', 's1', 'E2', 100, 5, '1800', 1)
        """,
        ) as con:
            violations = audit_integrity.check_old_session_names(con)
            assert len(violations) > 0

    def test_passes_new_session_names(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO orb_outcomes VALUES
            ('2025-01-01', 'GC', 5, 'CME_REOPEN', 'E2', 1)
        """,
        ) as con:
            violations = audit_integrity.check_old_session_names(con)
            assert len(violations) == 0


# ── Check 6: E0 CB2+ contamination ──────────────────────────────────


class TestE0CB2Contamination:
    """Check 6: No E0 + confirm_bars > 1 rows."""

    def test_catches_e0_cb2(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO orb_outcomes VALUES
            ('2025-01-01', 'GC', 5, 'CME_REOPEN', 'E0', 2)
        """,
        ) as con:
            violations = audit_integrity.check_e0_cb2_contamination(con)
            assert len(violations) > 0
            assert "E0+CB>1" in violations[0]

    def test_passes_e0_cb1(self, tmp_path):
        """E0 with CB=1 is not flagged by this check (flagged by check 4)."""
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO orb_outcomes VALUES
            ('2025-01-01', 'GC', 5, 'CME_REOPEN', 'E0', 1)
        """,
        ) as con:
            violations = audit_integrity.check_e0_cb2_contamination(con)
            assert len(violations) == 0

    def test_passes_e2_cb2(self, tmp_path):
        """E2 with CB=2 is fine."""
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO orb_outcomes VALUES
            ('2025-01-01', 'GC', 5, 'CME_REOPEN', 'E2', 2)
        """,
        ) as con:
            violations = audit_integrity.check_e0_cb2_contamination(con)
            assert len(violations) == 0


# ── Check 11: Dead instrument contamination ──────────────────────────


class TestDeadInstrumentContamination:
    """Check 11: No dead instruments in active validated_setups."""

    def test_catches_dead_instrument(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, instrument, status)
            VALUES ('s1', 'MCL', 'active')
        """,
        ) as con:
            violations = audit_integrity.check_dead_instrument_contamination(con)
            assert len(violations) > 0
            assert "MCL" in violations[0]

    def test_passes_active_instrument(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, instrument, status)
            VALUES ('s1', 'MGC', 'active')
        """,
        ) as con:
            violations = audit_integrity.check_dead_instrument_contamination(con)
            assert len(violations) == 0


# ── Check 12: Duplicate strategy IDs ─────────────────────────────────


class TestDuplicateStrategyIds:
    """Check 12: No duplicate strategy_ids in validated_setups."""

    def test_catches_duplicate(self, tmp_path):
        # Use schema without PK to allow duplicate insertion
        with _make_con(
            tmp_path,
            schemas="""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR,
                instrument VARCHAR
            );
        """
            + OUTCOMES_SCHEMA
            + EXPERIMENTAL_SCHEMA,
            inserts="""
            INSERT INTO validated_setups VALUES ('s1', 'MGC');
            INSERT INTO validated_setups VALUES ('s1', 'MGC');
        """,
        ) as con:
            violations = audit_integrity.check_duplicate_strategy_ids(con)
            assert len(violations) > 0
            assert "duplicate" in violations[0]

    def test_passes_unique(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, instrument)
            VALUES ('s1', 'MGC');
            INSERT INTO validated_setups (strategy_id, instrument)
            VALUES ('s2', 'MGC');
        """,
        ) as con:
            violations = audit_integrity.check_duplicate_strategy_ids(con)
            assert len(violations) == 0


# ── Check 15: Win rate sanity ────────────────────────────────────────


class TestWinRateSanity:
    """Check 15: Win rates in sane range [20%-85%]."""

    def test_catches_extreme_high(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, status, win_rate, rr_target)
            VALUES ('s1', 'active', 0.95, 2.0)
        """,
        ) as con:
            violations = audit_integrity.check_win_rate_sanity(con)
            assert len(violations) > 0
            assert "extreme win rates" in violations[0]

    def test_catches_extreme_low(self, tmp_path):
        """WR=10% with RR=2.0 → breakeven=33%, 80% of that=26.7%. 10% < 26.7% → flagged."""
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, status, win_rate, rr_target)
            VALUES ('s1', 'active', 0.10, 2.0)
        """,
        ) as con:
            violations = audit_integrity.check_win_rate_sanity(con)
            assert len(violations) > 0

    def test_passes_normal_win_rate(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, status, win_rate, rr_target)
            VALUES ('s1', 'active', 0.45, 2.0)
        """,
        ) as con:
            violations = audit_integrity.check_win_rate_sanity(con)
            assert len(violations) == 0

    def test_passes_high_rr_tight_stop(self, tmp_path):
        """RR=4.0, stop_mult=0.75 → breakeven=15.8%. WR=19% is above → no violation."""
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, status, win_rate, rr_target, stop_multiplier)
            VALUES ('s1', 'active', 0.19, 4.0, 0.75)
        """,
        ) as con:
            violations = audit_integrity.check_win_rate_sanity(con)
            assert len(violations) == 0

    def test_catches_below_breakeven_high_rr_tight_stop(self, tmp_path):
        """RR=4.0, stop_mult=0.75 → breakeven=15.8%, 80% threshold=12.6%. WR=10% → flagged."""
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, status, win_rate, rr_target, stop_multiplier)
            VALUES ('s1', 'active', 0.10, 4.0, 0.75)
        """,
        ) as con:
            violations = audit_integrity.check_win_rate_sanity(con)
            assert len(violations) > 0


# ── Check 16: Negative expectancy ────────────────────────────────────


class TestNegativeExpectancy:
    """Check 16: No active strategies with expectancy_r <= 0."""

    def test_catches_negative(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, status, expectancy_r)
            VALUES ('s1', 'active', -0.05)
        """,
        ) as con:
            violations = audit_integrity.check_negative_expectancy(con)
            assert len(violations) > 0
            assert "expectancy_r" in violations[0]

    def test_catches_zero(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, status, expectancy_r)
            VALUES ('s1', 'active', 0.0)
        """,
        ) as con:
            violations = audit_integrity.check_negative_expectancy(con)
            assert len(violations) > 0

    def test_passes_positive(self, tmp_path):
        with _make_con(
            tmp_path,
            inserts="""
            INSERT INTO validated_setups (strategy_id, status, expectancy_r)
            VALUES ('s1', 'active', 0.15)
        """,
        ) as con:
            violations = audit_integrity.check_negative_expectancy(con)
            assert len(violations) == 0
