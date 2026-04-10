"""
WS2: Tests for DB-dependent drift checks in pipeline/check_drift.py.

Covers checks 29, 35, 42, 43, 50, 54-58.
Each test creates a temp DuckDB, injects data, and verifies the check.
"""

from pathlib import Path

import duckdb
import pytest

from pipeline import check_drift


def _create_db(tmp_path, tables_sql: str, inserts_sql: str = "") -> Path:
    """Create a temp DuckDB with given schema and data."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    con.execute(tables_sql)
    if inserts_sql:
        con.execute(inserts_sql)
    con.close()
    return db_path


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
        n_trials_at_discovery INTEGER,
        fst_hurdle DOUBLE,
        sharpe_haircut DOUBLE
    );
"""

DAILY_FEATURES_SCHEMA = """
    CREATE TABLE daily_features (
        trading_day DATE,
        symbol VARCHAR,
        orb_minutes INTEGER
    );
"""


# ── Check 29: Validated filters registered ────────────────────────────


class TestValidatedFiltersRegistered:
    """Check 29: filter_type in validated_setups must exist in ALL_FILTERS."""

    def test_current_db_passes(self):
        violations = check_drift.check_validated_filters_registered()
        assert len(violations) == 0


# ── Active shelf routability ──────────────────────────────────────────


class TestActiveValidatedFiltersRoutable:
    """Active shelf must stay inside the canonical session-aware filter grid."""

    def test_passes_routable_active_filter(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA,
            """INSERT INTO validated_setups (
                   strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   confirm_bars, filter_type, rr_target, status, win_rate,
                   expectancy_r, fdr_significant, family_hash, wf_tested,
                   retired_at, retirement_reason
               ) VALUES (
                   'MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5', 'MNQ', 'EUROPE_FLOW', 5, 'E2',
                   1, 'ORB_G5', 1.0, 'active', 0.55,
                   0.10, TRUE, 'fam1', TRUE,
                   NULL, NULL
               )""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_validated_filters_routable()
        assert violations == []

    def test_catches_non_routable_active_filter(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA,
            """INSERT INTO validated_setups (
                   strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   confirm_bars, filter_type, rr_target, status, win_rate,
                   expectancy_r, fdr_significant, family_hash, wf_tested,
                   retired_at, retirement_reason
               ) VALUES (
                   'MNQ_TOKYO_OPEN_E2_RR1.0_CB1_OVNRNG_25', 'MNQ', 'TOKYO_OPEN', 5, 'E2',
                   1, 'OVNRNG_25', 1.0, 'active', 0.55,
                   0.10, TRUE, 'fam2', TRUE,
                   NULL, NULL
               )""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_validated_filters_routable()
        assert len(violations) == 1
        assert "OVNRNG_25" in violations[0]
        assert "TOKYO_OPEN" in violations[0]

    def test_ignores_retired_non_routable_filter(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA,
            """INSERT INTO validated_setups (
                   strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   confirm_bars, filter_type, rr_target, status, win_rate,
                   expectancy_r, fdr_significant, family_hash, wf_tested,
                   retired_at, retirement_reason
               ) VALUES (
                   'MNQ_TOKYO_OPEN_E2_RR1.0_CB1_OVNRNG_25', 'MNQ', 'TOKYO_OPEN', 5, 'E2',
                   1, 'OVNRNG_25', 1.0, 'retired', 0.55,
                   0.10, TRUE, 'fam3', TRUE,
                   CURRENT_TIMESTAMP, 'retired for test'
               )""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_validated_filters_routable()
        assert violations == []


# ── Active micro-only filter discipline ───────────────────────────────


class TestActiveMicroOnlyFiltersOnRealMicros:
    """Micro-only filters must never survive on parent/proxy instruments."""

    def test_passes_micro_only_filter_on_real_micro(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA,
            """INSERT INTO validated_setups (
                   strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   confirm_bars, filter_type, rr_target, status, win_rate,
                   expectancy_r, fdr_significant, family_hash, wf_tested,
                   retired_at, retirement_reason
               ) VALUES (
                   'MNQ_EUROPE_FLOW_E1_RR1.0_CB1_VOL_RV12_N20', 'MNQ', 'EUROPE_FLOW', 5, 'E1',
                   1, 'VOL_RV12_N20', 1.0, 'active', 0.55,
                   0.10, TRUE, 'fam4', TRUE,
                   NULL, NULL
               )""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_micro_only_filters_on_real_micros()
        assert violations == []

    def test_catches_micro_only_filter_on_parent_instrument(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA,
            """INSERT INTO validated_setups (
                   strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   confirm_bars, filter_type, rr_target, status, win_rate,
                   expectancy_r, fdr_significant, family_hash, wf_tested,
                   retired_at, retirement_reason
               ) VALUES (
                   'GC_EUROPE_FLOW_E1_RR1.0_CB1_VOL_RV12_N20', 'GC', 'EUROPE_FLOW', 5, 'E1',
                   1, 'VOL_RV12_N20', 1.0, 'active', 0.55,
                   0.10, TRUE, 'fam5', TRUE,
                   NULL, NULL
               )""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_micro_only_filters_on_real_micros()
        assert len(violations) == 1
        assert "VOL_RV12_N20" in violations[0]
        assert "GC" in violations[0]

    def test_ignores_retired_parent_violation(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA,
            """INSERT INTO validated_setups (
                   strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   confirm_bars, filter_type, rr_target, status, win_rate,
                   expectancy_r, fdr_significant, family_hash, wf_tested,
                   retired_at, retirement_reason
               ) VALUES (
                   'GC_EUROPE_FLOW_E1_RR1.0_CB1_VOL_RV12_N20', 'GC', 'EUROPE_FLOW', 5, 'E1',
                   1, 'VOL_RV12_N20', 1.0, 'retired', 0.55,
                   0.10, TRUE, 'fam6', TRUE,
                   CURRENT_TIMESTAMP, 'retired for test'
               )""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_micro_only_filters_on_real_micros()
        assert violations == []

    def test_fails_closed_on_unknown_instrument(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA,
            """INSERT INTO validated_setups (
                   strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   confirm_bars, filter_type, rr_target, status, win_rate,
                   expectancy_r, fdr_significant, family_hash, wf_tested,
                   retired_at, retirement_reason
               ) VALUES (
                   'XYZ_EUROPE_FLOW_E1_RR1.0_CB1_VOL_RV12_N20', 'XYZ', 'EUROPE_FLOW', 5, 'E1',
                   1, 'VOL_RV12_N20', 1.0, 'active', 0.55,
                   0.10, TRUE, 'fam7', TRUE,
                   NULL, NULL
               )""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_micro_only_filters_on_real_micros()
        assert len(violations) == 2
        assert "could not resolve micro-era status" in violations[0]
        assert "XYZ" in violations[0]
        assert "micro-only filter_type" in violations[1]


# ── Active micro-only filters after launch ────────────────────────────


class TestActiveMicroOnlyFiltersAfterMicroLaunch:
    """Micro-only filters must only trade on/after the real micro launch date."""

    MICRO_LAUNCH_SCHEMA = """
        CREATE TABLE validated_setups (
            strategy_id VARCHAR PRIMARY KEY,
            instrument VARCHAR,
            orb_label VARCHAR,
            orb_minutes INTEGER,
            entry_model VARCHAR,
            confirm_bars INTEGER,
            filter_type VARCHAR,
            rr_target DOUBLE,
            status VARCHAR,
            win_rate DOUBLE,
            expectancy_r DOUBLE,
            fdr_significant BOOLEAN,
            family_hash VARCHAR,
            wf_tested BOOLEAN,
            retired_at TIMESTAMPTZ,
            retirement_reason VARCHAR
        );

        CREATE TABLE daily_features (
            trading_day DATE,
            symbol VARCHAR,
            orb_minutes INTEGER,
            orb_CME_REOPEN_break_dir VARCHAR,
            orb_CME_REOPEN_volume BIGINT
        );

        CREATE TABLE orb_outcomes (
            trading_day DATE,
            symbol VARCHAR,
            orb_minutes INTEGER,
            orb_label VARCHAR,
            entry_model VARCHAR,
            confirm_bars INTEGER,
            rr_target DOUBLE,
            outcome VARCHAR,
            pnl_r DOUBLE,
            mae_r DOUBLE,
            mfe_r DOUBLE,
            entry_price DOUBLE,
            stop_price DOUBLE
        );
    """

    def test_passes_when_first_trade_is_after_launch(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            self.MICRO_LAUNCH_SCHEMA,
            """
            INSERT INTO validated_setups (
                strategy_id, instrument, orb_label, orb_minutes, entry_model,
                confirm_bars, filter_type, rr_target, status, win_rate,
                expectancy_r, fdr_significant, family_hash, wf_tested,
                retired_at, retirement_reason
            ) VALUES (
                'MNQ_CME_REOPEN_E1_RR1.0_CB1_ORB_VOL_2K', 'MNQ', 'CME_REOPEN', 5, 'E1',
                1, 'ORB_VOL_2K', 1.0, 'active', 0.55,
                0.10, TRUE, 'fam8', TRUE,
                NULL, NULL
            );

            INSERT INTO daily_features VALUES
                ('2019-05-06', 'MNQ', 5, 'LONG', 2500);

            INSERT INTO orb_outcomes VALUES
                ('2019-05-06', 'MNQ', 5, 'CME_REOPEN', 'E1', 1, 1.0,
                 'win', 1.0, 0.2, 1.5, 100.0, 95.0);
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_micro_only_filters_after_micro_launch()
        assert violations == []

    def test_catches_first_trade_before_launch(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            self.MICRO_LAUNCH_SCHEMA,
            """
            INSERT INTO validated_setups (
                strategy_id, instrument, orb_label, orb_minutes, entry_model,
                confirm_bars, filter_type, rr_target, status, win_rate,
                expectancy_r, fdr_significant, family_hash, wf_tested,
                retired_at, retirement_reason
            ) VALUES (
                'MNQ_CME_REOPEN_E1_RR1.0_CB1_ORB_VOL_2K', 'MNQ', 'CME_REOPEN', 5, 'E1',
                1, 'ORB_VOL_2K', 1.0, 'active', 0.55,
                0.10, TRUE, 'fam9', TRUE,
                NULL, NULL
            );

            INSERT INTO daily_features VALUES
                ('2019-05-03', 'MNQ', 5, 'LONG', 2500);

            INSERT INTO orb_outcomes VALUES
                ('2019-05-03', 'MNQ', 5, 'CME_REOPEN', 'E1', 1, 1.0,
                 'win', 1.0, 0.2, 1.5, 100.0, 95.0);
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_micro_only_filters_after_micro_launch()
        assert len(violations) == 1
        assert "before MNQ micro launch 2019-05-06" in violations[0]

    def test_catches_missing_recomputable_trade_days(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            self.MICRO_LAUNCH_SCHEMA,
            """
            INSERT INTO validated_setups (
                strategy_id, instrument, orb_label, orb_minutes, entry_model,
                confirm_bars, filter_type, rr_target, status, win_rate,
                expectancy_r, fdr_significant, family_hash, wf_tested,
                retired_at, retirement_reason
            ) VALUES (
                'MNQ_CME_REOPEN_E1_RR1.0_CB1_ORB_VOL_2K', 'MNQ', 'CME_REOPEN', 5, 'E1',
                1, 'ORB_VOL_2K', 1.0, 'active', 0.55,
                0.10, TRUE, 'fam10', TRUE,
                NULL, NULL
            );

            INSERT INTO daily_features VALUES
                ('2019-05-06', 'MNQ', 5, 'LONG', 1500);

            INSERT INTO orb_outcomes VALUES
                ('2019-05-06', 'MNQ', 5, 'CME_REOPEN', 'E1', 1, 1.0,
                 'win', 1.0, 0.2, 1.5, 100.0, 95.0);
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_micro_only_filters_after_micro_launch()
        assert len(violations) == 1
        assert "no recomputable traded days" in violations[0]


# ── Check 35: No E0 in DB ────────────────────────────────────────────


class TestNoE0InDb:
    """Check 35: No E0 rows in trading tables."""

    def test_catches_e0_in_outcomes(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            OUTCOMES_SCHEMA + EXPERIMENTAL_SCHEMA + VALIDATED_SCHEMA,
            """INSERT INTO orb_outcomes VALUES
                ('2025-01-01', 'MGC', 5, 'CME_REOPEN', 'E0', 1)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_no_e0_in_db()
        assert len(violations) > 0
        assert "E0" in violations[0]

    def test_passes_no_e0(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            OUTCOMES_SCHEMA + EXPERIMENTAL_SCHEMA + VALIDATED_SCHEMA,
            """INSERT INTO orb_outcomes VALUES
                ('2025-01-01', 'MGC', 5, 'CME_REOPEN', 'E2', 1)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_no_e0_in_db()
        assert len(violations) == 0


# ── Check 42: Orphaned validated strategies ───────────────────────────


class TestOrphanedValidatedStrategies:
    """Check 42: Active strategies must have matching orb_outcomes."""

    def test_catches_orphan(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA + OUTCOMES_SCHEMA,
            """INSERT INTO validated_setups (strategy_id, instrument, orb_minutes, status)
               VALUES ('MGC_15m_1', 'MGC', 15, 'active')""",
        )
        # No orb_outcomes for MGC 15m
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_orphaned_validated_strategies()
        assert len(violations) > 0
        assert "15m" in violations[0]

    def test_catches_orphan_wrong_aperture(self, tmp_path, monkeypatch):
        """Outcomes exist for 5m but strategy is 15m — still orphaned."""
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA + OUTCOMES_SCHEMA,
            """INSERT INTO validated_setups (strategy_id, instrument, orb_minutes, status)
               VALUES ('MGC_15m_1', 'MGC', 15, 'active');
               INSERT INTO orb_outcomes (trading_day, symbol, orb_minutes)
               VALUES ('2025-01-01', 'MGC', 5)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_orphaned_validated_strategies()
        assert len(violations) > 0

    def test_passes_with_outcomes(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            VALIDATED_SCHEMA + OUTCOMES_SCHEMA,
            """INSERT INTO validated_setups (strategy_id, instrument, orb_minutes, status)
               VALUES ('MGC_5m_1', 'MGC', 5, 'active');
               INSERT INTO orb_outcomes (trading_day, symbol, orb_minutes)
               VALUES ('2025-01-01', 'MGC', 5)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_orphaned_validated_strategies()
        assert len(violations) == 0


# ── Check 43: Uncovered FDR strategies ────────────────────────────────


class TestUncoveredFdrStrategies:
    """Check 43: FDR-significant strategies must be in edge families."""

    def test_current_db_passes(self):
        violations = check_drift.check_uncovered_fdr_strategies()
        assert len(violations) == 0


# ── Check 50: Audit columns populated ─────────────────────────────────


class TestAuditColumnsPopulated:
    """Check 50: experimental_strategies must have audit columns."""

    def test_catches_unpopulated_n_trials(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            EXPERIMENTAL_SCHEMA,
            """INSERT INTO experimental_strategies
               VALUES ('MGC', 's1', 'E2', 100, NULL, NULL, NULL)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_audit_columns_populated()
        assert len(violations) > 0
        assert "n_trials" in violations[0]

    def test_passes_populated(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            EXPERIMENTAL_SCHEMA,
            """INSERT INTO experimental_strategies
               VALUES ('MGC', 's1', 'E2', 100, 2376, 0.05, 0.3)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_audit_columns_populated()
        assert len(violations) == 0


# ── Check 54: Live config spec validity ───────────────────────────────


class TestLiveConfigSpecValidity:
    """Check 54: LIVE_PORTFOLIO specs reference valid sessions/models."""

    def test_current_config_passes(self):
        violations = check_drift.check_live_config_spec_validity()
        assert len(violations) == 0


# ── Check 55: Cost model field ranges ─────────────────────────────────


class TestCostModelFieldRanges:
    """Check 55: Cost model values within sane ranges."""

    def test_current_config_passes(self):
        violations = check_drift.check_cost_model_field_ranges()
        assert len(violations) == 0


# ── Check 56: Session resolver sanity ─────────────────────────────────


class TestSessionResolverSanity:
    """Check 56: All resolvers return valid (hour, minute) tuples."""

    def test_current_config_passes(self):
        violations = check_drift.check_session_resolver_sanity()
        assert len(violations) == 0


# ── Check 57: Daily features row integrity ────────────────────────────


class TestDailyFeaturesRowIntegrity:
    """Check 57: daily_features must have exactly 3 rows per (day, symbol)."""

    def test_catches_partial_rows(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            DAILY_FEATURES_SCHEMA,
            """INSERT INTO daily_features VALUES
                ('2025-01-01', 'MGC', 5),
                ('2025-01-01', 'MGC', 15)""",
        )
        # Only 2 rows instead of 3
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_daily_features_row_integrity()
        assert len(violations) > 0
        assert "MGC" in violations[0]

    def test_passes_complete(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            DAILY_FEATURES_SCHEMA,
            """INSERT INTO daily_features VALUES
                ('2025-01-01', 'MGC', 5),
                ('2025-01-01', 'MGC', 15),
                ('2025-01-01', 'MGC', 30)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_daily_features_row_integrity()
        assert len(violations) == 0

    def test_skips_non_active_instruments(self, tmp_path, monkeypatch):
        """Proxy-only symbols (e.g. GC) with fewer apertures should not flag."""
        db_path = _create_db(
            tmp_path,
            DAILY_FEATURES_SCHEMA,
            """INSERT INTO daily_features VALUES
                ('2025-01-01', 'GC', 5)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_daily_features_row_integrity()
        assert len(violations) == 0


# ── Check 58: Data continuity ─────────────────────────────────────────


class TestDataContinuity:
    """Check 58: Advisory warning on large gaps in trading days."""

    def test_warns_on_gap(self, tmp_path, monkeypatch, capsys):
        db_path = _create_db(
            tmp_path,
            DAILY_FEATURES_SCHEMA,
            """INSERT INTO daily_features VALUES
                ('2025-01-01', 'MGC', 5),
                ('2025-01-01', 'MGC', 15),
                ('2025-01-01', 'MGC', 30),
                ('2025-01-20', 'MGC', 5),
                ('2025-01-20', 'MGC', 15),
                ('2025-01-20', 'MGC', 30)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_data_continuity()
        # Advisory — always returns []
        assert len(violations) == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.out and "gap" in captured.out.lower()

    def test_no_warning_for_small_gaps(self, tmp_path, monkeypatch, capsys):
        db_path = _create_db(
            tmp_path,
            DAILY_FEATURES_SCHEMA,
            """INSERT INTO daily_features VALUES
                ('2025-01-06', 'MGC', 5),
                ('2025-01-06', 'MGC', 15),
                ('2025-01-06', 'MGC', 30),
                ('2025-01-07', 'MGC', 5),
                ('2025-01-07', 'MGC', 15),
                ('2025-01-07', 'MGC', 30)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_data_continuity()
        assert len(violations) == 0
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out
