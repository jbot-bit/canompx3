"""
WS2: Tests for DB-dependent drift checks in pipeline/check_drift.py.

Covers checks 29, 35, 42, 43, 50, 54-58.
Each test creates a temp DuckDB, injects data, and verifies the check.
"""

from datetime import date
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

DAILY_FEATURES_HTF_SCHEMA = """
    CREATE TABLE daily_features (
        trading_day DATE,
        symbol VARCHAR,
        orb_minutes INTEGER,
        prev_week_high DOUBLE,
        prev_week_low DOUBLE,
        prev_week_open DOUBLE,
        prev_week_close DOUBLE,
        prev_week_range DOUBLE,
        prev_week_mid DOUBLE,
        prev_month_high DOUBLE,
        prev_month_low DOUBLE,
        prev_month_open DOUBLE,
        prev_month_close DOUBLE,
        prev_month_range DOUBLE,
        prev_month_mid DOUBLE
    );
"""

DAILY_FEATURES_GARCH_SCHEMA = """
    CREATE TABLE daily_features (
        trading_day DATE,
        symbol VARCHAR,
        orb_minutes INTEGER,
        garch_forecast_vol DOUBLE,
        garch_forecast_vol_pct DOUBLE
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


# ── Native promotion provenance discipline ────────────────────────────


class TestActiveNativePromotionProvenance:
    """Native rows must carry populated, linkable promotion provenance."""

    PROVENANCE_SCHEMA = """
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
            promotion_provenance VARCHAR,
            validation_run_id VARCHAR,
            promotion_git_sha VARCHAR,
            first_trade_day DATE,
            last_trade_day DATE,
            trade_day_count INTEGER
        );

        CREATE TABLE validation_run_log (
            run_id VARCHAR PRIMARY KEY
        );
    """

    def test_passes_populated_native_row(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            self.PROVENANCE_SCHEMA,
            """
            INSERT INTO validation_run_log VALUES ('run_1');
            INSERT INTO validated_setups VALUES (
                'sid1', 'MNQ', 'CME_REOPEN', 5, 'E1', 1, 'NO_FILTER', 1.0,
                'active', 'VALIDATOR_NATIVE', 'run_1', 'abc123def456',
                '2024-01-02', '2024-01-03', 2
            );
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_native_promotion_provenance_populated()
        assert violations == []

    def test_catches_missing_fields(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            self.PROVENANCE_SCHEMA,
            """
            INSERT INTO validated_setups VALUES (
                'sid1', 'MNQ', 'CME_REOPEN', 5, 'E1', 1, 'NO_FILTER', 1.0,
                'active', 'VALIDATOR_NATIVE', NULL, NULL,
                NULL, NULL, NULL
            );
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_native_promotion_provenance_populated()
        assert len(violations) == 1
        assert "missing promotion provenance fields" in violations[0]

    def test_catches_missing_validation_run(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            self.PROVENANCE_SCHEMA,
            """
            INSERT INTO validated_setups VALUES (
                'sid1', 'MNQ', 'CME_REOPEN', 5, 'E1', 1, 'NO_FILTER', 1.0,
                'active', 'VALIDATOR_NATIVE', 'missing_run', 'abc123def456',
                '2024-01-02', '2024-01-03', 2
            );
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_native_promotion_provenance_populated()
        assert len(violations) == 1
        assert "missing validation_run_log.run_id" in violations[0]

    def test_catches_legacy_schema_missing_columns(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            """
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                status VARCHAR
            );
            CREATE TABLE validation_run_log (
                run_id VARCHAR PRIMARY KEY
            );
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_native_promotion_provenance_populated()
        assert len(violations) == 1
        assert "missing native promotion provenance columns" in violations[0]


class TestActiveNativeTradeWindowProvenance:
    """Stored native trade-window provenance must match canonical recompute."""

    TRADE_WINDOW_SCHEMA = """
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
            promotion_provenance VARCHAR,
            first_trade_day DATE,
            last_trade_day DATE,
            trade_day_count INTEGER
        );

        CREATE TABLE daily_features (
            trading_day DATE,
            symbol VARCHAR,
            orb_minutes INTEGER,
            orb_CME_REOPEN_break_dir VARCHAR
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

    def test_passes_matching_window(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            self.TRADE_WINDOW_SCHEMA,
            """
            INSERT INTO validated_setups VALUES (
                'sid1', 'MNQ', 'CME_REOPEN', 5, 'E1', 1, 'NO_FILTER', 1.0,
                'active', 'VALIDATOR_NATIVE', '2024-01-02', '2024-01-03', 2
            );
            INSERT INTO daily_features VALUES
                ('2024-01-02', 'MNQ', 5, 'LONG'),
                ('2024-01-03', 'MNQ', 5, 'LONG');
            INSERT INTO orb_outcomes VALUES
                ('2024-01-02', 'MNQ', 5, 'CME_REOPEN', 'E1', 1, 1.0, 'win', 1.0, 0.2, 1.5, 100.0, 95.0),
                ('2024-01-03', 'MNQ', 5, 'CME_REOPEN', 'E1', 1, 1.0, 'win', 1.0, 0.2, 1.5, 100.0, 95.0);
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_native_trade_windows_match_provenance()
        assert violations == []

    def test_catches_mismatched_window(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            self.TRADE_WINDOW_SCHEMA,
            """
            INSERT INTO validated_setups VALUES (
                'sid1', 'MNQ', 'CME_REOPEN', 5, 'E1', 1, 'NO_FILTER', 1.0,
                'active', 'VALIDATOR_NATIVE', '2024-01-02', '2024-01-02', 1
            );
            INSERT INTO daily_features VALUES
                ('2024-01-02', 'MNQ', 5, 'LONG'),
                ('2024-01-03', 'MNQ', 5, 'LONG');
            INSERT INTO orb_outcomes VALUES
                ('2024-01-02', 'MNQ', 5, 'CME_REOPEN', 'E1', 1, 1.0, 'win', 1.0, 0.2, 1.5, 100.0, 95.0),
                ('2024-01-03', 'MNQ', 5, 'CME_REOPEN', 'E1', 1, 1.0, 'win', 1.0, 0.2, 1.5, 100.0, 95.0);
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_native_trade_windows_match_provenance()
        assert len(violations) == 1
        assert "stored trade window" in violations[0]

    def test_catches_legacy_schema_missing_columns(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            """
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                status VARCHAR
            );
            """,
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_active_native_trade_windows_match_provenance()
        assert len(violations) == 1
        assert "missing native trade-window provenance columns" in violations[0]


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


# ── Recent GARCH feature coverage ─────────────────────────────────────


class TestRecentGarchFeatureCoverage:
    """Late-history GARCH state should not revert to NULL on recent rows."""

    def test_catches_recent_nulls_after_warmup(self, tmp_path, monkeypatch):
        db_path = tmp_path / "test_garch_recent_nulls.db"
        con = duckdb.connect(str(db_path))
        con.execute(DAILY_FEATURES_GARCH_SCHEMA)
        for i in range(340):
            td = date.fromordinal(date(2024, 1, 1).toordinal() + i)
            garch = None if i >= 335 else 0.10 + i / 1000.0
            gpct = None if i >= 335 else 50.0
            con.execute(
                "INSERT INTO daily_features VALUES (?, 'MNQ', 5, ?, ?)",
                [td, garch, gpct],
            )
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_recent_garch_feature_coverage()
        assert len(violations) == 1
        assert "MNQ O5" in violations[0]
        assert "5 NULL row(s)" in violations[0]

    def test_ignores_early_history_warmup_and_passes_recent_coverage(self, tmp_path, monkeypatch):
        db_path = tmp_path / "test_garch_recent_covered.db"
        con = duckdb.connect(str(db_path))
        con.execute(DAILY_FEATURES_GARCH_SCHEMA)
        for i in range(320):
            td = date.fromordinal(date(2024, 1, 1).toordinal() + i)
            garch = None if i < 40 else 0.10 + i / 1000.0
            gpct = None if i < 40 else 50.0
            con.execute(
                "INSERT INTO daily_features VALUES (?, 'MNQ', 5, ?, ?)",
                [td, garch, gpct],
            )
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_recent_garch_feature_coverage()
        assert violations == []

    def test_ignores_legitimate_garch_pct_warmup_before_recent_window_is_safe(self, tmp_path, monkeypatch):
        db_path = tmp_path / "test_garch_pct_warmup.db"
        con = duckdb.connect(str(db_path))
        con.execute(DAILY_FEATURES_GARCH_SCHEMA)
        for i in range(320):
            td = date.fromordinal(date(2024, 1, 1).toordinal() + i)
            garch = None if i < 252 else 0.10 + i / 1000.0
            gpct = None if i < 312 else 50.0
            con.execute(
                "INSERT INTO daily_features VALUES (?, 'MNQ', 5, ?, ?)",
                [td, garch, gpct],
            )
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_recent_garch_feature_coverage()
        assert violations == []


# ── HTF aperture consistency ──────────────────────────────────────────


class TestHTFApertureConsistency:
    """HTF fields must match across O5/O15/O30 sibling rows."""

    def test_catches_aperture_divergence(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            DAILY_FEATURES_HTF_SCHEMA,
            """INSERT INTO daily_features VALUES
                ('2026-04-17', 'MGC', 5, 4887.3, 4625.1, 4673.2, 4680.7, 262.2, 4756.2, 5434.4, 4100.0, 5346.4, 4709.6, 1334.4, 4767.2),
                ('2026-04-17', 'MGC', 15, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
                ('2026-04-17', 'MGC', 30, 4887.3, 4625.1, 4673.2, 4680.7, 262.2, 4756.2, 5434.4, 4100.0, 5346.4, 4709.6, 1334.4, 4767.2)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_htf_aperture_consistency()
        assert len(violations) == 1
        assert "MGC 2026-04-17" in violations[0]
        assert "prev_week_high(aperture_diff)" in violations[0]

    def test_passes_when_all_apertures_agree(self, tmp_path, monkeypatch):
        db_path = _create_db(
            tmp_path,
            DAILY_FEATURES_HTF_SCHEMA,
            """INSERT INTO daily_features VALUES
                ('2026-04-17', 'MGC', 5, 4887.3, 4625.1, 4673.2, 4680.7, 262.2, 4756.2, 5434.4, 4100.0, 5346.4, 4709.6, 1334.4, 4767.2),
                ('2026-04-17', 'MGC', 15, 4887.3, 4625.1, 4673.2, 4680.7, 262.2, 4756.2, 5434.4, 4100.0, 5346.4, 4709.6, 1334.4, 4767.2),
                ('2026-04-17', 'MGC', 30, 4887.3, 4625.1, 4673.2, 4680.7, 262.2, 4756.2, 5434.4, 4100.0, 5346.4, 4709.6, 1334.4, 4767.2)""",
        )
        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_drift.check_htf_aperture_consistency()
        assert violations == []


# ── Stage 2: routed filter required columns populated ──────────────────────


class TestRoutedFilterColumnsPopulated:
    """New check: every routed filter's required daily_features column must
    be populated at >= 50% across ACTIVE_ORB_INSTRUMENTS. Catches ghost
    deployments like the 2026-04-06 PIT_MIN / pit_range_atr gap."""

    def test_passes_when_all_columns_populated(self):
        """Against the live DB after the 2026-04-20 backfill, the check
        must return no violations."""
        import duckdb
        from pipeline.paths import GOLD_DB_PATH

        if not GOLD_DB_PATH.exists():
            pytest.skip("gold.db not available")
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        try:
            violations = check_drift.check_routed_filter_columns_populated(con=con)
        finally:
            con.close()
        # No blocking violations expected after pit_range_atr backfill.
        assert violations == [], f"Expected clean pass, got: {violations}"

    def test_flags_zero_populated_column(self, tmp_path):
        """Inject a schema with a column present but 0%-populated; the
        check must flag it if that column is required by a routed filter."""
        import duckdb

        db_path = tmp_path / "ghost.db"
        con = duckdb.connect(str(db_path))
        # Replicate the minimal daily_features columns required by
        # routed filters, with pit_range_atr present but NULL.
        con.execute(
            """
            CREATE TABLE daily_features (
                trading_day DATE,
                symbol VARCHAR,
                orb_minutes INTEGER,
                pit_range_atr DOUBLE,
                atr_20 DOUBLE,
                atr_20_pct DOUBLE,
                overnight_range DOUBLE,
                overnight_range_pct DOUBLE,
                gap_open_points DOUBLE,
                garch_forecast_vol_pct DOUBLE,
                prev_day_range DOUBLE,
                day_of_week INTEGER,
                is_nfp_day BOOLEAN,
                is_opex_day BOOLEAN,
                is_friday BOOLEAN
            );
            INSERT INTO daily_features VALUES
                ('2025-01-01', 'MNQ', 5, NULL, 150.0, 75.0, 80.0, 60.0, 5.0, 80.0, 200.0, 2, FALSE, FALSE, FALSE),
                ('2025-01-01', 'MES', 5, NULL, 150.0, 75.0, 80.0, 60.0, 5.0, 80.0, 200.0, 2, FALSE, FALSE, FALSE),
                ('2025-01-01', 'MGC', 5, NULL, 150.0, 75.0, 80.0, 60.0, 5.0, 80.0, 200.0, 2, FALSE, FALSE, FALSE);
            """
        )
        violations = check_drift.check_routed_filter_columns_populated(con=con)
        con.close()

        # PIT_MIN is routed to CME_REOPEN; pit_range_atr is required but 0% populated.
        assert any("pit_range_atr" in v and "PIT_MIN" in v for v in violations), (
            f"Expected pit_range_atr/PIT_MIN violation, got: {violations}"
        )

    def test_con_none_returns_empty(self):
        """When DB is unavailable, check must fail-safe with empty output."""
        violations = check_drift.check_routed_filter_columns_populated(con=None)
        assert violations == []


# ── Stage 3: pooled-finding annotation schema ─────────────────────────────


class TestPooledFindingAnnotations:
    """New check: audit-result files on/after 2026-04-20 claiming pooled
    findings must carry per_cell_breakdown_path, flip_rate_pct, and
    (when flip_rate_pct >= 25) heterogeneity_ack front-matter."""

    def test_flags_missing_breakdown_path(self, tmp_path, monkeypatch):
        """File declares pooled_finding: true but omits per_cell_breakdown_path."""
        results_dir = tmp_path / "docs" / "audit" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "2026-05-01-example-pooled.md").write_text(
            "---\npooled_finding: true\nflip_rate_pct: 10\n---\n\n# Example\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        violations = check_drift.check_pooled_finding_annotations()
        assert any("per_cell_breakdown_path missing" in v for v in violations), (
            f"Expected per_cell_breakdown_path violation, got: {violations}"
        )

    def test_flags_high_flip_rate_without_ack(self, tmp_path, monkeypatch):
        """flip_rate_pct >= 25 without heterogeneity_ack=true must fail."""
        results_dir = tmp_path / "docs" / "audit" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "2026-05-01-example-pooled.md").write_text(
            "---\n"
            "pooled_finding: true\n"
            "per_cell_breakdown_path: docs/audit/results/2026-05-01-example-pooled.md\n"
            "flip_rate_pct: 30\n"
            "---\n\n# Example\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        violations = check_drift.check_pooled_finding_annotations()
        assert any("heterogeneity_ack" in v for v in violations), (
            f"Expected heterogeneity_ack violation, got: {violations}"
        )

    def test_accepts_complete_front_matter(self, tmp_path, monkeypatch):
        """Properly-annotated file with low flip rate must pass clean."""
        results_dir = tmp_path / "docs" / "audit" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "2026-05-01-example-pooled.md").write_text(
            "---\n"
            "pooled_finding: true\n"
            "per_cell_breakdown_path: docs/audit/results/2026-05-01-example-pooled.md\n"
            "flip_rate_pct: 8\n"
            "---\n\n# Example\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        violations = check_drift.check_pooled_finding_annotations()
        assert violations == [], f"Expected clean pass, got: {violations}"

    def test_exempts_pre_sentinel_files(self, tmp_path, monkeypatch):
        """Files dated before the sentinel are exempt from the schema."""
        results_dir = tmp_path / "docs" / "audit" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "2026-04-19-example-pooled.md").write_text(
            "---\npooled_finding: true\n---\n\n# Example\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        violations = check_drift.check_pooled_finding_annotations()
        assert violations == []

    def test_ignores_files_without_front_matter(self, tmp_path, monkeypatch):
        """A new file that omits front-matter entirely is not flagged by
        this check — rule requires the declaration to trigger. Undeclared
        pooled claims are a social-discipline problem handled by the rule
        file, not by this mechanical check."""
        results_dir = tmp_path / "docs" / "audit" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "2026-05-01-no-frontmatter.md").write_text(
            "# A new audit result with no front-matter at all.\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        violations = check_drift.check_pooled_finding_annotations()
        assert violations == []
