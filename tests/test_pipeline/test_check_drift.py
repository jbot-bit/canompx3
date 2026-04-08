"""
Tests for pipeline.check_drift drift detection rules.

Tests each drift check catches violations and passes clean code.
"""

import tempfile
from datetime import date
from pathlib import Path

import pytest

from pipeline.check_drift import (
    check_apply_iterrows,
    check_config_filter_sync,
    check_hardcoded_mgc_sql,
    check_holdout_policy_declaration_consistency,
    check_non_bars1m_writes,
    check_pipeline_never_imports_trading_app,
    check_pyright_config_exists,
    check_python_version_file,
    check_ruff_rules_minimum,
    check_trading_app_connection_leaks,
    check_trading_app_hardcoded_paths,
    check_uv_lock_exists,
)


class TestHardcodedMgcSql:
    """Tests for hardcoded 'MGC' SQL detection in generic files."""

    def test_catches_values_mgc(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("con.execute(\"INSERT INTO bars_1m VALUES ('MGC', ...)\")\n")
        violations = check_hardcoded_mgc_sql([f])
        assert len(violations) > 0

    def test_catches_where_symbol_mgc(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("con.execute(\"SELECT * FROM bars_1m WHERE symbol = 'MGC'\")\n")
        violations = check_hardcoded_mgc_sql([f])
        assert len(violations) > 0

    def test_passes_clean_code(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('con.execute("SELECT * FROM bars_1m WHERE symbol = ?")\n')
        violations = check_hardcoded_mgc_sql([f])
        assert len(violations) == 0

    def test_ignores_comments(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("# symbol = 'MGC' in a comment\n")
        violations = check_hardcoded_mgc_sql([f])
        assert len(violations) == 0

    def test_missing_file_no_crash(self, tmp_path):
        f = tmp_path / "nonexistent.py"
        violations = check_hardcoded_mgc_sql([f])
        assert len(violations) == 0


class TestApplyIterrows:
    """Tests for .apply()/.iterrows() anti-pattern detection."""

    def test_catches_iterrows(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("for idx, row in df.iterrows():\n    pass\n")
        violations = check_apply_iterrows([f])
        assert len(violations) > 0

    def test_allows_front_df_iterrows(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("for ts_utc, row in front_df.iterrows():\n    pass\n")
        violations = check_apply_iterrows([f])
        assert len(violations) == 0

    def test_catches_apply(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("result = df['price'].apply(lambda x: x * 2)\n")
        violations = check_apply_iterrows([f])
        assert len(violations) > 0

    def test_allows_symbol_apply(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("mask = chunk_df['symbol'].apply(lambda s: bool(pattern.match(s)))\n")
        violations = check_apply_iterrows([f])
        assert len(violations) == 0

    def test_passes_clean_code(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("result = df[df['high'] > 100]\n")
        violations = check_apply_iterrows([f])
        assert len(violations) == 0


class TestNonBars1mWrites:
    """Tests for non-bars_1m write detection in ingest scripts."""

    def test_catches_insert_into_bars_5m(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('con.execute("INSERT INTO bars_5m (ts_utc) VALUES (?)")\n')
        violations = check_non_bars1m_writes([f])
        assert len(violations) > 0

    def test_catches_delete_from_other_table(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('con.execute("DELETE FROM daily_features WHERE date = ?")\n')
        violations = check_non_bars1m_writes([f])
        assert len(violations) > 0

    def test_allows_bars_1m_writes(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('con.execute("INSERT OR REPLACE INTO bars_1m (ts_utc) VALUES (?)")\n')
        violations = check_non_bars1m_writes([f])
        assert len(violations) == 0

    def test_passes_clean_code(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('count = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]\n')
        violations = check_non_bars1m_writes([f])
        assert len(violations) == 0


class TestPipelineNeverImportsTradingApp:
    """Tests for one-way dependency: pipeline must never import trading_app."""

    def test_catches_from_import(self, tmp_path):
        f = tmp_path / "run_pipeline.py"
        f.write_text("from trading_app.config import ALL_FILTERS\n")
        violations = check_pipeline_never_imports_trading_app(tmp_path)
        assert len(violations) > 0

    def test_catches_import_statement(self, tmp_path):
        f = tmp_path / "run_pipeline.py"
        f.write_text("import trading_app\n")
        violations = check_pipeline_never_imports_trading_app(tmp_path)
        assert len(violations) > 0

    def test_passes_pipeline_imports(self, tmp_path):
        f = tmp_path / "build_bars_5m.py"
        f.write_text("from pipeline.paths import GOLD_DB_PATH\n")
        violations = check_pipeline_never_imports_trading_app(tmp_path)
        assert len(violations) == 0

    def test_ignores_comments(self, tmp_path):
        f = tmp_path / "run_pipeline.py"
        f.write_text("# from trading_app import something\n")
        violations = check_pipeline_never_imports_trading_app(tmp_path)
        assert len(violations) == 0

    def test_skips_check_drift(self, tmp_path):
        """check_drift.py itself references trading_app dir — should be skipped."""
        f = tmp_path / "check_drift.py"
        f.write_text("from trading_app import db_manager\n")
        violations = check_pipeline_never_imports_trading_app(tmp_path)
        assert len(violations) == 0


class TestTradingAppConnectionLeaks:
    """Tests for connection leak detection in trading_app/."""

    def test_catches_no_cleanup(self, tmp_path):
        f = tmp_path / "bad_module.py"
        f.write_text("con = duckdb.connect(str(db_path))\ncon.execute('SELECT 1')\n")
        violations = check_trading_app_connection_leaks(tmp_path)
        assert len(violations) > 0

    def test_passes_with_finally(self, tmp_path):
        f = tmp_path / "good_module.py"
        f.write_text("con = duckdb.connect(str(db_path))\ntry:\n    pass\nfinally:\n    con.close()\n")
        violations = check_trading_app_connection_leaks(tmp_path)
        assert len(violations) == 0

    def test_passes_with_close(self, tmp_path):
        f = tmp_path / "good_module.py"
        f.write_text("con = duckdb.connect(str(db_path))\ncon.execute('SELECT 1')\ncon.close()\n")
        violations = check_trading_app_connection_leaks(tmp_path)
        assert len(violations) == 0

    def test_skips_init(self, tmp_path):
        f = tmp_path / "__init__.py"
        f.write_text("con = duckdb.connect('test.db')\n")
        violations = check_trading_app_connection_leaks(tmp_path)
        assert len(violations) == 0

    def test_no_dir(self, tmp_path):
        nonexistent = tmp_path / "trading_app_fake"
        violations = check_trading_app_connection_leaks(nonexistent)
        assert len(violations) == 0


class TestTradingAppHardcodedPaths:
    """Tests for hardcoded absolute paths in trading_app/."""

    def test_catches_windows_path(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("DB_PATH = 'C:\\Users\\josh\\gold.db'\n")
        violations = check_trading_app_hardcoded_paths(tmp_path)
        assert len(violations) > 0

    def test_catches_forward_slash_path(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("DB_PATH = 'C:/Users/josh/gold.db'\n")
        violations = check_trading_app_hardcoded_paths(tmp_path)
        assert len(violations) > 0

    def test_passes_relative_path(self, tmp_path):
        f = tmp_path / "good.py"
        f.write_text("DB_PATH = Path(__file__).parent.parent / 'gold.db'\n")
        violations = check_trading_app_hardcoded_paths(tmp_path)
        assert len(violations) == 0

    def test_ignores_comments(self, tmp_path):
        f = tmp_path / "good.py"
        f.write_text("# path was C:\\\\Users\\\\old_path\n")
        violations = check_trading_app_hardcoded_paths(tmp_path)
        assert len(violations) == 0

    def test_no_dir(self, tmp_path):
        nonexistent = tmp_path / "fake_dir"
        violations = check_trading_app_hardcoded_paths(nonexistent)
        assert len(violations) == 0


class TestConfigFilterSync:
    """Tests for check 12: config filter_type sync."""

    def test_current_config_passes(self):
        """Current ALL_FILTERS config has no sync violations."""
        violations = check_config_filter_sync()
        assert len(violations) == 0

    def test_detects_real_sync(self):
        """Verify the check actually inspects ALL_FILTERS."""
        # If ALL_FILTERS is importable and has entries, check should return empty
        from trading_app.config import ALL_FILTERS

        assert len(ALL_FILTERS) > 0
        violations = check_config_filter_sync()
        assert len(violations) == 0


class TestClaudeMdSizeCap:
    """Tests for check 23: CLAUDE.md size cap."""

    def test_real_claude_md_exists_and_under_cap(self):
        """Verify actual CLAUDE.md is under 12KB."""
        from pipeline.check_drift import check_claude_md_size_cap

        violations = check_claude_md_size_cap()
        assert len(violations) == 0

    def test_catches_oversized_file(self, tmp_path, monkeypatch):
        """Oversized CLAUDE.md triggers violation."""
        from pipeline import check_drift

        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        (tmp_path / "CLAUDE.md").write_text("x" * 13000)
        violations = check_drift.check_claude_md_size_cap()
        assert len(violations) == 1
        assert "12KB" in violations[0]

    def test_passes_small_file(self, tmp_path, monkeypatch):
        """Small CLAUDE.md passes."""
        from pipeline import check_drift

        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        (tmp_path / "CLAUDE.md").write_text("small")
        violations = check_drift.check_claude_md_size_cap()
        assert len(violations) == 0


class TestDiscoverySessionAwareFilters:
    """Tests for check 28: discovery scripts must use get_filters_for_grid, not ALL_FILTERS."""

    def _patch(self, monkeypatch, check_drift, tmp_path):
        """Patch both TRADING_APP_DIR and PROJECT_ROOT to tmp_path so
        fpath.relative_to(PROJECT_ROOT) works in violation messages."""
        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", tmp_path)
        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)

    def test_catches_all_filters_items(self, tmp_path, monkeypatch):
        """ALL_FILTERS.items() in strategy_discovery.py triggers a violation."""
        from pipeline import check_drift

        (tmp_path / "strategy_discovery.py").write_text("for filter_key, filt in ALL_FILTERS.items():\n    pass\n")
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 1
        assert "iterates ALL_FILTERS" in violations[0]

    def test_catches_all_filters_values(self, tmp_path, monkeypatch):
        """ALL_FILTERS.values() in a discovery file triggers a violation."""
        from pipeline import check_drift

        (tmp_path / "strategy_discovery.py").write_text("for filt in ALL_FILTERS.values():\n    pass\n")
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 1
        assert "iterates ALL_FILTERS" in violations[0]

    def test_catches_len_all_filters(self, tmp_path, monkeypatch):
        """len(ALL_FILTERS) in a discovery file triggers a violation."""
        from pipeline import check_drift

        (tmp_path / "strategy_discovery.py").write_text("total_combos = len(ALL_FILTERS) * len(RR_TARGETS)\n")
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 1
        assert "len(ALL_FILTERS)" in violations[0]

    def test_passes_all_filters_get(self, tmp_path, monkeypatch):
        """ALL_FILTERS.get() is a registry lookup — not a grid iteration."""
        from pipeline import check_drift

        (tmp_path / "strategy_discovery.py").write_text("filt = ALL_FILTERS.get(strategy.filter_type)\n")
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 0

    def test_passes_commented_out(self, tmp_path, monkeypatch):
        """Commented-out ALL_FILTERS.items() is not a violation."""
        from pipeline import check_drift

        (tmp_path / "strategy_discovery.py").write_text("# for k, v in ALL_FILTERS.items():  # old pattern\n")
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 0

    def test_passes_missing_file(self, tmp_path, monkeypatch):
        """Missing discovery file is silently skipped."""
        from pipeline import check_drift

        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 0

    def test_catches_nested_discovery(self, tmp_path, monkeypatch):
        """Violation in nested/discovery.py is detected."""
        from pipeline import check_drift

        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "discovery.py").write_text("for k, v in ALL_FILTERS.items():\n    pass\n")
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 1
        assert "nested" in violations[0]

    def test_real_discovery_files_pass(self):
        """Current codebase discovery scripts are clean."""
        from pipeline.check_drift import check_discovery_session_aware_filters

        violations = check_discovery_session_aware_filters()
        assert len(violations) == 0


class TestWfCoverage:
    """Tests for check #40: WF coverage for MGC/MES."""

    def test_warns_on_untested_mgc(self, tmp_path, monkeypatch, capsys):
        """MGC with 1 NULL + 1 TRUE -> warning printed, but no blocking violation."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_wf_coverage

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                status VARCHAR,
                wf_tested BOOLEAN
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('MGC_1', 'MGC', 'active', TRUE),
                ('MGC_2', 'MGC', 'active', NULL)
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_wf_coverage()
        assert len(violations) == 0, "Soft gate must never block"
        captured = capsys.readouterr()
        assert "MGC" in captured.out
        assert "1/2" in captured.out
        assert "WARNING" in captured.out

    def test_passes_all_tested(self, tmp_path, monkeypatch):
        """MGC with 1 TRUE -> no violation."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_wf_coverage

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                status VARCHAR,
                wf_tested BOOLEAN
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('MGC_1', 'MGC', 'active', TRUE)
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_wf_coverage()
        assert len(violations) == 0

    def test_ignores_mnq_m2k(self, tmp_path, monkeypatch):
        """MNQ with NULL -> no violation (not in required set)."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_wf_coverage

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                status VARCHAR,
                wf_tested BOOLEAN
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('MNQ_1', 'MNQ', 'active', NULL),
                ('M2K_1', 'M2K', 'active', NULL)
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_wf_coverage()
        assert len(violations) == 0


class TestNoActiveE3:
    """Tests for check #39: no active E3 strategies in validated_setups."""

    def test_catches_active_e3(self, tmp_path, monkeypatch):
        """Active E3 row triggers violation."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_no_active_e3

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                entry_model VARCHAR,
                status VARCHAR,
                retired_at TIMESTAMPTZ,
                retirement_reason VARCHAR,
                fdr_significant BOOLEAN
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('E3_MGC_1', 'MGC', 'E3', 'active', NULL, NULL, FALSE)
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_no_active_e3()
        assert len(violations) == 1
        assert "active E3" in violations[0]

    def test_passes_retired_e3(self, tmp_path, monkeypatch):
        """Retired E3 row does not trigger violation."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_no_active_e3

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                entry_model VARCHAR,
                status VARCHAR,
                retired_at TIMESTAMPTZ,
                retirement_reason VARCHAR,
                fdr_significant BOOLEAN
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('E3_MGC_1', 'MGC', 'E3', 'RETIRED', '2026-02-28', 'test', FALSE)
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_no_active_e3()
        assert len(violations) == 0


class TestDataYearsDisclosure:
    """Tests for check #41: data years disclosure (warning-only)."""

    def test_warns_on_short_history(self, tmp_path, monkeypatch, capsys):
        """MNQ with years_tested=5 -> warning printed, no blocking violation."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_data_years_disclosure

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id TEXT, instrument TEXT, status TEXT,
                years_tested INTEGER
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
            ('S1', 'MNQ', 'active', 5),
            ('S2', 'MGC', 'active', 10)
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_data_years_disclosure()
        assert len(violations) == 0, "Soft gate must never block"
        captured = capsys.readouterr()
        assert "MNQ" in captured.out
        assert "WARNING" in captured.out

    def test_passes_long_history(self, tmp_path, monkeypatch, capsys):
        """MGC with years_tested=10 -> no warning."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_data_years_disclosure

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id TEXT, instrument TEXT, status TEXT,
                years_tested INTEGER
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
            ('S1', 'MGC', 'active', 10)
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_data_years_disclosure()
        assert len(violations) == 0
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out


# ============================================================================
# Tooling config drift checks
# ============================================================================


class TestPyrightConfigExists:
    """Tests for pyrightconfig.json existence and mode check."""

    def test_passes_with_basic_mode(self, tmp_path):
        config = tmp_path / "pyrightconfig.json"
        config.write_text('{"typeCheckingMode": "basic"}')
        assert check_pyright_config_exists(tmp_path) == []

    def test_passes_with_strict_mode(self, tmp_path):
        config = tmp_path / "pyrightconfig.json"
        config.write_text('{"typeCheckingMode": "strict"}')
        assert check_pyright_config_exists(tmp_path) == []

    def test_fails_when_missing(self, tmp_path):
        violations = check_pyright_config_exists(tmp_path)
        assert len(violations) == 1
        assert "missing" in violations[0]

    def test_fails_with_off_mode(self, tmp_path):
        config = tmp_path / "pyrightconfig.json"
        config.write_text('{"typeCheckingMode": "off"}')
        violations = check_pyright_config_exists(tmp_path)
        assert len(violations) == 1
        assert "off" in violations[0]

    def test_fails_with_no_mode_key(self, tmp_path):
        config = tmp_path / "pyrightconfig.json"
        config.write_text('{"include": ["pipeline"]}')
        violations = check_pyright_config_exists(tmp_path)
        assert len(violations) == 1
        assert "off" in violations[0]


class TestRuffRulesMinimum:
    """Tests for ruff.toml minimum rule set check."""

    def test_passes_with_all_required(self, tmp_path):
        ruff = tmp_path / "ruff.toml"
        ruff.write_text('[lint]\nselect = ["F", "E", "W", "I", "B", "UP", "SIM"]')
        assert check_ruff_rules_minimum(tmp_path) == []

    def test_fails_when_missing_file(self, tmp_path):
        violations = check_ruff_rules_minimum(tmp_path)
        assert len(violations) == 1
        assert "missing" in violations[0]

    def test_fails_when_missing_rules(self, tmp_path):
        ruff = tmp_path / "ruff.toml"
        ruff.write_text('[lint]\nselect = ["F", "E"]')
        violations = check_ruff_rules_minimum(tmp_path)
        assert len(violations) == 1
        assert "I" in violations[0]
        assert "B" in violations[0]
        assert "UP" in violations[0]


class TestPythonVersionFile:
    """Tests for .python-version file check."""

    def test_passes_with_313(self, tmp_path):
        pv = tmp_path / ".python-version"
        pv.write_text("3.13\n")
        assert check_python_version_file(tmp_path) == []

    def test_fails_when_missing(self, tmp_path):
        violations = check_python_version_file(tmp_path)
        assert len(violations) == 1
        assert "missing" in violations[0]

    def test_fails_with_wrong_version(self, tmp_path):
        pv = tmp_path / ".python-version"
        pv.write_text("3.11\n")
        violations = check_python_version_file(tmp_path)
        assert len(violations) == 1
        assert "3.11" in violations[0]


class TestUvLockExists:
    """Tests for uv.lock existence check."""

    def test_passes_with_real_lock(self, tmp_path):
        lock = tmp_path / "uv.lock"
        lock.write_text("\n".join([f"[[package]]\nname = 'pkg{i}'" for i in range(10)]))
        assert check_uv_lock_exists(tmp_path) == []

    def test_fails_when_missing(self, tmp_path):
        violations = check_uv_lock_exists(tmp_path)
        assert len(violations) == 1
        assert "missing" in violations[0]

    def test_fails_with_skeleton(self, tmp_path):
        lock = tmp_path / "uv.lock"
        lock.write_text("version = 1\n[[package]]\nname = 'canompx3'\n")
        violations = check_uv_lock_exists(tmp_path)
        assert len(violations) == 1
        assert "skeleton" in violations[0]


# ============================================================================
# E2 canonical-window fix structural-lock checks (Stage 8 of refactor 2026-04-07)
# ============================================================================
#
# Each test injects a controlled violation of one of the 5 new structural
# locks, then asserts the corresponding drift check detects it. This proves
# the checks actually catch their target — without negative tests, the
# checks could silently rot into "always passes" no-ops.
#
# Reference: docs/postmortems/2026-04-07-e2-canonical-window-fix.md.


class TestCanonicalOrbUtcWindowSource:
    """Test that check_canonical_orb_utc_window_source catches non-canonical defs."""

    def test_passes_when_only_canonical_defines_it(self):
        """The current repo state must pass — pipeline/dst.py is the only definer."""
        from pipeline.check_drift import check_canonical_orb_utc_window_source

        violations = check_canonical_orb_utc_window_source()
        assert violations == [], f"Current repo should be canonical-clean: {violations}"

    def test_catches_duplicate_definition_in_trading_app(self, tmp_path, monkeypatch):
        """Inject a fake `def orb_utc_window(` in trading_app/ — must be flagged."""
        from pipeline import check_drift

        fake_trading = tmp_path / "trading_app"
        fake_trading.mkdir()
        offender = fake_trading / "rogue_orb.py"
        offender.write_text(
            "def orb_utc_window(trading_day, orb_label, orb_minutes):\n"
            "    return None, None\n"
        )
        # Keep PIPELINE_DIR/SCRIPTS_DIR pointing somewhere harmless so the
        # canonical pipeline/dst.py definer doesn't appear in the scan
        # (we want to test detection of duplicates only).
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_trading)
        monkeypatch.setattr(check_drift, "PIPELINE_DIR", empty)
        monkeypatch.setattr(check_drift, "SCRIPTS_DIR", empty)

        violations = check_drift.check_canonical_orb_utc_window_source()
        assert len(violations) > 0
        assert any("rogue_orb.py" in v for v in violations)

    def test_catches_duplicate_definition_in_scripts(self, tmp_path, monkeypatch):
        """Inject a fake `def orb_utc_window(` in scripts/ — must be flagged."""
        from pipeline import check_drift

        fake_scripts = tmp_path / "scripts"
        fake_scripts.mkdir()
        offender = fake_scripts / "tools" / "bad_orb.py"
        offender.parent.mkdir()
        offender.write_text(
            "def orb_utc_window(td, lbl, m):\n"
            "    pass\n"
        )
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(check_drift, "SCRIPTS_DIR", fake_scripts)
        monkeypatch.setattr(check_drift, "PIPELINE_DIR", empty)
        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", empty)

        violations = check_drift.check_canonical_orb_utc_window_source()
        assert len(violations) > 0
        assert any("bad_orb.py" in v for v in violations)


class TestNoSilentBreakTsFallback:
    """Test that check_no_silent_break_ts_fallback catches Stage 5 regressions."""

    def test_passes_on_current_outcome_builder(self):
        """The current Stage 5 fix must not contain any forbidden pattern."""
        from pipeline.check_drift import check_no_silent_break_ts_fallback

        violations = check_no_silent_break_ts_fallback()
        assert violations == [], (
            f"trading_app/outcome_builder.py contains a forbidden silent-fallback "
            f"pattern — Stage 5 of the E2 canonical-window refactor is broken: {violations}"
        )

    def test_catches_if_else_silent_fallback(self, tmp_path, monkeypatch):
        """Inject the L455-style silent fallback — must be flagged."""
        from pipeline import check_drift

        fake_dir = tmp_path / "trading_app"
        fake_dir.mkdir()
        (fake_dir / "outcome_builder.py").write_text(
            "scan_start = orb_end_utc if orb_end_utc is not None else break_ts\n"
        )
        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_dir)

        violations = check_drift.check_no_silent_break_ts_fallback()
        assert len(violations) > 0
        assert any("if orb_end_utc is not None else break_ts" in v for v in violations)

    def test_catches_or_shorthand_fallback(self, tmp_path, monkeypatch):
        """Inject the `orb_end_utc or break_ts` shorthand — must be flagged."""
        from pipeline import check_drift

        fake_dir = tmp_path / "trading_app"
        fake_dir.mkdir()
        (fake_dir / "outcome_builder.py").write_text(
            "x = orb_end_utc or break_ts\n"
        )
        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_dir)

        violations = check_drift.check_no_silent_break_ts_fallback()
        assert len(violations) > 0
        assert any("orb_end_utc or break_ts" in v for v in violations)

    def test_catches_break_delay_derivation(self, tmp_path, monkeypatch):
        """Inject the L782-style derivation from break_delay_min — must be flagged."""
        from pipeline import check_drift

        fake_dir = tmp_path / "trading_app"
        fake_dir.mkdir()
        (fake_dir / "outcome_builder.py").write_text(
            "orb_end_utc = break_ts - timedelta(minutes=break_delay)\n"
        )
        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_dir)

        violations = check_drift.check_no_silent_break_ts_fallback()
        assert len(violations) > 0
        assert any("break_ts - timedelta(minutes=break_delay)" in v for v in violations)


class TestComputeSingleOutcomeCanonicalKwargs:
    """Test that check_compute_single_outcome_canonical_kwargs catches signature drift."""

    def test_passes_on_current_signature(self):
        """The Stage 5 signature must include all 4 canonical kwargs."""
        from pipeline.check_drift import (
            check_compute_single_outcome_canonical_kwargs,
        )

        violations = check_compute_single_outcome_canonical_kwargs()
        assert violations == [], (
            f"compute_single_outcome signature is missing a required canonical kwarg — "
            f"Stage 5 of the E2 canonical-window refactor is broken: {violations}"
        )

    def test_catches_missing_kwarg_via_monkeypatched_signature(self, monkeypatch):
        """Replace compute_single_outcome with a stub missing canonical kwargs."""
        import trading_app.outcome_builder as ob

        def stub(bars_df, break_ts, orb_high, orb_low, break_dir, rr_target,
                 confirm_bars, trading_day_end, cost_spec, entry_model="E1"):
            # Deliberately missing trading_day, orb_label, orb_minutes, orb_end_utc
            return {}

        monkeypatch.setattr(ob, "compute_single_outcome", stub)
        from pipeline.check_drift import (
            check_compute_single_outcome_canonical_kwargs,
        )

        violations = check_compute_single_outcome_canonical_kwargs()
        assert len(violations) > 0
        assert any("trading_day" in v or "orb_label" in v or "orb_minutes" in v
                   or "orb_end_utc" in v for v in violations)


class TestNestedBuilderAbsent:
    """Test that check_nested_builder_absent catches re-creation of dead module."""

    def test_passes_when_file_absent(self):
        """Stage 7 deleted nested/builder.py — current state must pass."""
        from pipeline.check_drift import check_nested_builder_absent

        violations = check_nested_builder_absent()
        assert violations == [], (
            f"trading_app/nested/builder.py exists but Stage 7 of the E2 "
            f"canonical-window refactor deleted it: {violations}"
        )

    def test_catches_re_creation(self, tmp_path, monkeypatch):
        """Re-create nested/builder.py — must be flagged."""
        from pipeline import check_drift

        fake_trading = tmp_path / "trading_app"
        nested_dir = fake_trading / "nested"
        nested_dir.mkdir(parents=True)
        builder = nested_dir / "builder.py"
        builder.write_text("# accidentally re-created dead module\n")
        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_trading)

        violations = check_drift.check_nested_builder_absent()
        assert len(violations) > 0
        assert any("nested" in v and "builder.py" in v for v in violations)


class TestResampleHelpersInEntryRules:
    """Test that resample helpers are pinned to trading_app.entry_rules."""

    def test_passes_on_current_module_location(self):
        """Both helpers live in trading_app.entry_rules per Stage 4."""
        from pipeline.check_drift import check_resample_helpers_in_entry_rules

        violations = check_resample_helpers_in_entry_rules()
        assert violations == [], (
            f"resample_to_5m or _verify_e3_sub_bar_fill is not in "
            f"trading_app.entry_rules — Stage 4 of the E2 canonical-window "
            f"refactor placed them there: {violations}"
        )

    def test_catches_helper_relocation(self, monkeypatch):
        """Monkeypatch resample_to_5m's __module__ to a wrong module — must be flagged."""
        import trading_app.entry_rules as er
        from pipeline.check_drift import check_resample_helpers_in_entry_rules

        original_module = er.resample_to_5m.__module__
        # Wrap and override __module__
        wrapped = type(er.resample_to_5m)(
            er.resample_to_5m.__code__,
            er.resample_to_5m.__globals__,
            er.resample_to_5m.__name__,
            er.resample_to_5m.__defaults__,
            er.resample_to_5m.__closure__,
        )
        wrapped.__module__ = "rogue.module"
        monkeypatch.setattr(er, "resample_to_5m", wrapped)

        violations = check_resample_helpers_in_entry_rules()
        # Restore handled by monkeypatch teardown.
        assert len(violations) > 0
        assert any("resample_to_5m" in v for v in violations)
        # Sanity: confirm the original was correct (otherwise the test is moot)
        assert original_module == "trading_app.entry_rules"


class TestHoldoutPolicyDeclarationConsistency:
    """Tests for ``check_holdout_policy_declaration_consistency`` (drift check #83).

    The function asserts four invariants:
    1. ``trading_app.holdout_policy`` is importable and exports the three
       canonical names.
    2. ``HOLDOUT_SACRED_FROM`` equals ``date(2026, 1, 1)`` (Amendment 2.7 lock).
    3. ``docs/institutional/pre_registered_criteria.md`` exists and mentions
       ``Amendment 2.7``.
    4. ``RESEARCH_RULES.md`` exists, mentions the sacred-from date in
       ISO format, and cites ``Amendment 2.7``.

    The function reads project-relative paths via ``Path(__file__).parent.parent``,
    so failure-case tests redirect ``project_root`` by monkeypatching
    ``pipeline.check_drift.__file__`` to point inside a synthetic ``tmp_path``
    repo skeleton. This avoids touching the real repo files (which are
    canonical and asserted by the happy-path test).
    """

    AMENDMENT = "Amendment 2.7"
    SACRED_DATE_STR = "2026-01-01"

    def _build_fake_root(
        self,
        tmp_path: Path,
        criteria_text: str | None,
        rules_text: str | None,
    ) -> Path:
        """Build a fake repo skeleton with optional doc files.

        Returns the path to the fake ``pipeline/check_drift.py`` so the caller
        can monkeypatch ``pipeline.check_drift.__file__`` to it.
        """
        fake_root = tmp_path / "fake_repo"
        pipeline_dir = fake_root / "pipeline"
        pipeline_dir.mkdir(parents=True)
        if criteria_text is not None:
            criteria_dir = fake_root / "docs" / "institutional"
            criteria_dir.mkdir(parents=True)
            (criteria_dir / "pre_registered_criteria.md").write_text(criteria_text, encoding="utf-8")
        if rules_text is not None:
            (fake_root / "RESEARCH_RULES.md").write_text(rules_text, encoding="utf-8")
        return pipeline_dir / "check_drift.py"

    def test_happy_path_against_real_repo(self):
        """The function MUST pass against the actual repo state — this is the
        gold-standard mirror of ``python pipeline/check_drift.py`` 84/0/0."""
        violations = check_holdout_policy_declaration_consistency()
        assert violations == [], (
            "Real repo declaration consistency drifted — fix the canonical sources "
            f"before running other tests. Violations: {violations}"
        )

    def test_missing_export_returns_violation(self, monkeypatch):
        """If the canonical module loses one of the three required exports,
        the check must flag the missing name and bail before doc checks."""
        import trading_app.holdout_policy as hp
        monkeypatch.delattr(hp, "HOLDOUT_SACRED_FROM", raising=True)
        violations = check_holdout_policy_declaration_consistency()
        assert any("HOLDOUT_SACRED_FROM" in v and "EXPORT MISSING" in v for v in violations)

    def test_sacred_from_drift_returns_violation(self, monkeypatch):
        """If ``HOLDOUT_SACRED_FROM`` is changed without a new Amendment, the
        check must flag the drift and cite the lock value."""
        import trading_app.holdout_policy as hp
        monkeypatch.setattr(hp, "HOLDOUT_SACRED_FROM", date(2027, 1, 1))
        violations = check_holdout_policy_declaration_consistency()
        assert any("HOLDOUT_SACRED_FROM drifted" in v for v in violations)
        assert any("2026-01-01" in v for v in violations)

    def test_criteria_md_missing_amendment_returns_violation(self, monkeypatch, tmp_path):
        """If ``pre_registered_criteria.md`` exists but does not cite
        Amendment 2.7, the check must flag the missing citation."""
        fake_check_drift = self._build_fake_root(
            tmp_path,
            criteria_text="some criteria but no amendment marker",
            rules_text=f"{self.AMENDMENT}\n{self.SACRED_DATE_STR}\n",
        )
        monkeypatch.setattr("pipeline.check_drift.__file__", str(fake_check_drift))
        violations = check_holdout_policy_declaration_consistency()
        assert any(
            "pre_registered_criteria.md" in v and self.AMENDMENT in v for v in violations
        )

    def test_criteria_md_missing_file_returns_violation(self, monkeypatch, tmp_path):
        """If ``pre_registered_criteria.md`` does not exist on disk, the
        check must surface the missing file (not silently pass)."""
        fake_check_drift = self._build_fake_root(
            tmp_path,
            criteria_text=None,  # don't create the file
            rules_text=f"{self.AMENDMENT}\n{self.SACRED_DATE_STR}\n",
        )
        monkeypatch.setattr("pipeline.check_drift.__file__", str(fake_check_drift))
        violations = check_holdout_policy_declaration_consistency()
        assert any("pre_registered_criteria.md missing" in v for v in violations)

    def test_research_rules_missing_sacred_date_returns_violation(self, monkeypatch, tmp_path):
        """If ``RESEARCH_RULES.md`` does not contain the sacred-from ISO date,
        the check must flag the doc-code drift."""
        fake_check_drift = self._build_fake_root(
            tmp_path,
            criteria_text=f"valid criteria {self.AMENDMENT}",
            rules_text=f"{self.AMENDMENT}\nbut no sacred date here\n",
        )
        monkeypatch.setattr("pipeline.check_drift.__file__", str(fake_check_drift))
        violations = check_holdout_policy_declaration_consistency()
        assert any(
            "RESEARCH_RULES.md" in v and self.SACRED_DATE_STR in v for v in violations
        )

    def test_research_rules_missing_amendment_returns_violation(self, monkeypatch, tmp_path):
        """If ``RESEARCH_RULES.md`` does not cite Amendment 2.7, the check
        must flag the missing top-level declaration."""
        fake_check_drift = self._build_fake_root(
            tmp_path,
            criteria_text=f"valid criteria {self.AMENDMENT}",
            rules_text=f"sacred date {self.SACRED_DATE_STR} but no amendment marker\n",
        )
        monkeypatch.setattr("pipeline.check_drift.__file__", str(fake_check_drift))
        violations = check_holdout_policy_declaration_consistency()
        assert any(
            "RESEARCH_RULES.md" in v and self.AMENDMENT in v for v in violations
        )

    def test_research_rules_missing_file_returns_violation(self, monkeypatch, tmp_path):
        """If ``RESEARCH_RULES.md`` does not exist on disk, the check must
        surface the missing file (not silently pass)."""
        fake_check_drift = self._build_fake_root(
            tmp_path,
            criteria_text=f"valid criteria {self.AMENDMENT}",
            rules_text=None,  # don't create the file
        )
        monkeypatch.setattr("pipeline.check_drift.__file__", str(fake_check_drift))
        violations = check_holdout_policy_declaration_consistency()
        assert any("RESEARCH_RULES.md missing" in v for v in violations)

    def test_all_invariants_violated_yields_multiple(self, monkeypatch, tmp_path):
        """When the criteria doc, the rules doc, AND the sacred date all
        diverge, the check must return multiple violations (no early bail
        once we're past the export check)."""
        import trading_app.holdout_policy as hp
        monkeypatch.setattr(hp, "HOLDOUT_SACRED_FROM", date(2025, 7, 1))
        fake_check_drift = self._build_fake_root(
            tmp_path,
            criteria_text="missing the marker",
            rules_text="also missing both",
        )
        monkeypatch.setattr("pipeline.check_drift.__file__", str(fake_check_drift))
        violations = check_holdout_policy_declaration_consistency()
        # Expect: sacred-from drift, criteria amendment-missing,
        # rules amendment-missing, rules sacred-date-missing.
        assert len(violations) >= 3, f"Expected multiple violations, got: {violations}"
        assert any("HOLDOUT_SACRED_FROM drifted" in v for v in violations)
        assert any("pre_registered_criteria.md" in v for v in violations)
        assert any("RESEARCH_RULES.md" in v for v in violations)
