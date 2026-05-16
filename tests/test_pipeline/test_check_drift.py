"""
Tests for pipeline.check_drift drift detection rules.

Tests each drift check catches violations and passes clean code.
"""

import tempfile
import textwrap
from datetime import date
from pathlib import Path

import pytest

from pipeline.check_drift import (
    _parse_stage_acceptance_commands,
    _stage_acceptance_all_pass,
    check_apply_iterrows,
    check_config_filter_sync,
    check_daily_features_row_integrity,
    check_hardcoded_mgc_sql,
    check_holdout_policy_declaration_consistency,
    check_non_bars1m_writes,
    check_pipeline_never_imports_trading_app,
    check_prereg_present_for_recent_runs,
    check_pyright_config_exists,
    check_python_version_file,
    check_ruff_rules_minimum,
    check_trading_app_connection_leaks,
    check_trading_app_hardcoded_paths,
    check_checks_list_labels_are_ascii,
    check_chordia_result_threshold_matches_prereg,
    check_uv_lock_exists,
    check_verdict_vocabulary_md_matches_code,
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


class TestDailyFeaturesRowIntegrity:
    """Tests for active-universe daily_features aperture completeness."""

    def test_active_symbol_missing_aperture_is_flagged(self, tmp_path, monkeypatch):
        import duckdb

        from pipeline import check_drift

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE daily_features (
                trading_day DATE,
                symbol VARCHAR,
                orb_minutes INTEGER
            )
        """)
        con.execute("""
            INSERT INTO daily_features VALUES
                ('2026-04-01', 'MNQ', 5),
                ('2026-04-01', 'MNQ', 15),
                ('2026-04-01', 'MES', 5),
                ('2026-04-01', 'MES', 15),
                ('2026-04-01', 'MES', 30),
                ('2026-04-01', 'MGC', 5),
                ('2026-04-01', 'MGC', 15),
                ('2026-04-01', 'MGC', 30)
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_daily_features_row_integrity()
        assert len(violations) == 1
        assert "MNQ" in violations[0]

    def test_proxy_symbol_is_excluded_from_active_integrity_gate(self, tmp_path, monkeypatch):
        import duckdb

        from pipeline import check_drift

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE daily_features (
                trading_day DATE,
                symbol VARCHAR,
                orb_minutes INTEGER
            )
        """)
        con.execute("""
            INSERT INTO daily_features VALUES
                ('2026-04-01', 'GC', 5),
                ('2026-04-02', 'GC', 5),
                ('2026-04-01', 'MNQ', 5),
                ('2026-04-01', 'MNQ', 15),
                ('2026-04-01', 'MNQ', 30),
                ('2026-04-01', 'MES', 5),
                ('2026-04-01', 'MES', 15),
                ('2026-04-01', 'MES', 30),
                ('2026-04-01', 'MGC', 5),
                ('2026-04-01', 'MGC', 15),
                ('2026-04-01', 'MGC', 30)
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        assert check_daily_features_row_integrity() == []


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


class TestGarchDependencyImportable:
    """Tests for check_garch_dependency_importable.

    Guards against the 2026-04-29 incident where `arch` went missing from
    the canonical venv and every daily build silently NULLed
    `garch_forecast_vol` until Check 65 surfaced the late-history NULLs.
    """

    def test_passes_when_arch_installed(self):
        """Live venv must have arch installed (it's a hard pyproject dep)."""
        from pipeline.check_drift import check_garch_dependency_importable

        violations = check_garch_dependency_importable()
        assert violations == [], f"arch package missing from venv: {violations}. Run: pip install 'arch>=8.0.0'"

    def test_catches_missing_arch_package(self, monkeypatch):
        """If importlib.metadata.version('arch') raises PackageNotFoundError, fail loudly."""
        from importlib.metadata import PackageNotFoundError

        from pipeline import check_drift

        def fake_version(name):
            if name == "arch":
                raise PackageNotFoundError("arch")
            from importlib.metadata import version as real_version

            return real_version(name)

        monkeypatch.setattr("importlib.metadata.version", fake_version)
        violations = check_drift.check_garch_dependency_importable()
        assert len(violations) == 1
        assert "arch package not installed" in violations[0]
        assert "pyproject.toml" in violations[0]
        assert "uv sync" in violations[0]

    def test_catches_broken_install(self, monkeypatch):
        """version() succeeds but import fails (partial wheel / corruption)."""
        import builtins

        from pipeline import check_drift

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "arch":
                raise ImportError("DLL load failed: arch broken install")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        violations = check_drift.check_garch_dependency_importable()
        assert len(violations) == 1
        assert "arch importable check failed" in violations[0]
        assert "force-reinstall" in violations[0]


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


class TestNoActiveE2LookaheadFilters:
    """Tests for the active-shelf E2 look-ahead contamination check."""

    def test_catches_active_e2_break_speed_filter(self, tmp_path, monkeypatch):
        """An active E2 strategy with a FAST filter must trigger a violation."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_no_active_e2_lookahead_filters

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                orb_label VARCHAR,
                entry_model VARCHAR,
                filter_type VARCHAR,
                status VARCHAR
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('MNQ_NYSE_OPEN_E2_FAST5', 'MNQ', 'NYSE_OPEN', 'E2', 'ORB_G8_FAST5', 'active')
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_no_active_e2_lookahead_filters()
        assert len(violations) == 1
        assert "look-ahead filter_type" in violations[0]
        assert "ORB_G8_FAST5" in violations[0]

    def test_catches_active_e2_relative_volume_filter(self, tmp_path, monkeypatch):
        """An active E2 strategy with a rel-vol filter must trigger a violation."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_no_active_e2_lookahead_filters

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                orb_label VARCHAR,
                entry_model VARCHAR,
                filter_type VARCHAR,
                status VARCHAR
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('MNQ_CLOSE_E2_VOL', 'MNQ', 'NYSE_CLOSE', 'E2', 'VOL_RV12_N20', 'active')
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_no_active_e2_lookahead_filters()
        assert len(violations) == 1
        assert "VOL_RV12_N20" in violations[0]

    def test_passes_active_e1_break_speed_filter(self, tmp_path, monkeypatch):
        """The same filter family is fine for E1 because the bar has closed."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_no_active_e2_lookahead_filters

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                orb_label VARCHAR,
                entry_model VARCHAR,
                filter_type VARCHAR,
                status VARCHAR
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('MNQ_NYSE_OPEN_E1_FAST5', 'MNQ', 'NYSE_OPEN', 'E1', 'ORB_G8_FAST5', 'active')
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_no_active_e2_lookahead_filters()
        assert len(violations) == 0

    def test_passes_retired_e2_lookahead_filter(self, tmp_path, monkeypatch):
        """Retired contamination should not block the active shelf check."""
        import duckdb

        from pipeline import check_drift
        from pipeline.check_drift import check_no_active_e2_lookahead_filters

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR PRIMARY KEY,
                instrument VARCHAR,
                orb_label VARCHAR,
                entry_model VARCHAR,
                filter_type VARCHAR,
                status VARCHAR
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
                ('MNQ_NYSE_OPEN_E2_FAST5', 'MNQ', 'NYSE_OPEN', 'E2', 'ORB_G8_FAST5', 'retired')
        """)
        con.close()

        monkeypatch.setattr(check_drift, "GOLD_DB_PATH_FOR_CHECKS", db_path)
        violations = check_no_active_e2_lookahead_filters()
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
        offender.write_text("def orb_utc_window(trading_day, orb_label, orb_minutes):\n    return None, None\n")
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
        offender.write_text("def orb_utc_window(td, lbl, m):\n    pass\n")
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
        (fake_dir / "outcome_builder.py").write_text("x = orb_end_utc or break_ts\n")
        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_dir)

        violations = check_drift.check_no_silent_break_ts_fallback()
        assert len(violations) > 0
        assert any("orb_end_utc or break_ts" in v for v in violations)

    def test_catches_break_delay_derivation(self, tmp_path, monkeypatch):
        """Inject the L782-style derivation from break_delay_min — must be flagged."""
        from pipeline import check_drift

        fake_dir = tmp_path / "trading_app"
        fake_dir.mkdir()
        (fake_dir / "outcome_builder.py").write_text("orb_end_utc = break_ts - timedelta(minutes=break_delay)\n")
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

        def stub(
            bars_df,
            break_ts,
            orb_high,
            orb_low,
            break_dir,
            rr_target,
            confirm_bars,
            trading_day_end,
            cost_spec,
            entry_model="E1",
        ):
            # Deliberately missing trading_day, orb_label, orb_minutes, orb_end_utc
            return {}

        monkeypatch.setattr(ob, "compute_single_outcome", stub)
        from pipeline.check_drift import (
            check_compute_single_outcome_canonical_kwargs,
        )

        violations = check_compute_single_outcome_canonical_kwargs()
        assert len(violations) > 0
        assert any(
            "trading_day" in v or "orb_label" in v or "orb_minutes" in v or "orb_end_utc" in v for v in violations
        )


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
        assert any("pre_registered_criteria.md" in v and self.AMENDMENT in v for v in violations)

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
        assert any("RESEARCH_RULES.md" in v and self.SACRED_DATE_STR in v for v in violations)

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
        assert any("RESEARCH_RULES.md" in v and self.AMENDMENT in v for v in violations)

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


class TestPreregPresentForRecentRuns:
    """Tests for ``check_prereg_present_for_recent_runs`` (Criterion 1 advisory).

    The check scans ``experimental_strategies`` for rows created after
    ``HOLDOUT_GRANDFATHER_CUTOFF`` and reports any (instrument, discovery_date)
    that has no matching prereg yaml at
    ``docs/audit/hypotheses/<date>-<instrument>-*.yaml``.

    These tests inject a fake DuckDB-like connection (``execute`` returning a
    pre-canned fetchall) and monkeypatch ``PROJECT_ROOT`` so the prereg
    glob hits a temp dir.
    """

    class _FakeCursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _FakeCon:
        def __init__(self, rows_by_instrument):
            self._rows_by_instrument = rows_by_instrument

        def execute(self, _sql, params):
            instrument = params[0]
            return TestPreregPresentForRecentRuns._FakeCursor(self._rows_by_instrument.get(instrument, []))

        def close(self):
            pass

    def _patch_project_root(self, monkeypatch, fake_root: Path) -> None:
        import pipeline.check_drift as cd

        monkeypatch.setattr(cd, "PROJECT_ROOT", fake_root)

    def test_no_post_grandfather_rows_passes(self, monkeypatch, tmp_path):
        """If experimental_strategies has no rows past the grandfather cutoff
        for the active instruments, the check returns no violations."""
        fake_root = tmp_path / "fake_repo"
        (fake_root / "docs" / "audit" / "hypotheses").mkdir(parents=True)
        self._patch_project_root(monkeypatch, fake_root)

        con = self._FakeCon(rows_by_instrument={})  # all instruments empty
        violations = check_prereg_present_for_recent_runs(con=con)
        assert violations == [], f"unexpected violations: {violations}"

    def test_missing_prereg_yields_violation(self, monkeypatch, tmp_path):
        """A post-grandfather discovery date with no matching yaml is reported."""
        fake_root = tmp_path / "fake_repo"
        (fake_root / "docs" / "audit" / "hypotheses").mkdir(parents=True)
        self._patch_project_root(monkeypatch, fake_root)

        con = self._FakeCon(rows_by_instrument={"MGC": [(date(2026, 5, 4),)]})
        violations = check_prereg_present_for_recent_runs(con=con)
        assert any("PREREG MISSING" in v and "MGC" in v and "2026-05-04" in v for v in violations), (
            f"expected PREREG MISSING for MGC 2026-05-04, got: {violations}"
        )

    def test_present_prereg_passes(self, monkeypatch, tmp_path):
        """If a matching yaml exists, no violation for that (instrument, date)."""
        fake_root = tmp_path / "fake_repo"
        hyp_dir = fake_root / "docs" / "audit" / "hypotheses"
        hyp_dir.mkdir(parents=True)
        (hyp_dir / "2026-05-04-mgc-cme-reopen-v1.yaml").write_text(
            "hypotheses:\n  - id: 1\n",
            encoding="utf-8",
        )
        self._patch_project_root(monkeypatch, fake_root)

        con = self._FakeCon(rows_by_instrument={"MGC": [(date(2026, 5, 4),)]})
        violations = check_prereg_present_for_recent_runs(con=con)
        # No violation specifically for MGC 2026-05-04
        assert not any("MGC" in v and "2026-05-04" in v for v in violations), (
            f"expected no MGC 2026-05-04 violation, got: {violations}"
        )

    def test_missing_hypotheses_dir_returns_violation(self, monkeypatch, tmp_path):
        """A repo with no docs/audit/hypotheses/ directory cannot enforce
        Criterion 1 -- the check must surface that as a violation rather than
        silently passing (fail-closed)."""
        fake_root = tmp_path / "fake_repo"
        fake_root.mkdir()  # no hypotheses subdir
        self._patch_project_root(monkeypatch, fake_root)

        con = self._FakeCon(rows_by_instrument={})
        violations = check_prereg_present_for_recent_runs(con=con)
        assert any("hypotheses dir missing" in v for v in violations), (
            f"expected missing-dir violation, got: {violations}"
        )


# ─── Check 92: @canonical-source annotation integrity (F-1..F-9 stage 8) ──
# @canonical-source docs/research-input/topstep/topstep_dll_article.md  (referenced for self-test)


class TestCanonicalSourceAnnotations:
    """Drift check 92 — verifies @canonical-source refs point to existing files.

    Established 2026-04-08 by stage 8 of docs/plans/2026-04-08-topstep-canonical-fixes.md.
    """

    def test_baseline_passes(self):
        """The current repo has no broken @canonical-source refs."""
        from pipeline.check_drift import check_canonical_source_annotations

        violations = check_canonical_source_annotations()
        assert violations == [], (
            f"Expected zero broken @canonical-source refs but found {len(violations)}: {violations[:5]}"
        )

    def test_catches_missing_path(self, tmp_path, monkeypatch):
        """Inject a fake @canonical-source ref to a nonexistent file → flagged."""
        from pipeline import check_drift

        fake_pkg = tmp_path / "trading_app"
        fake_pkg.mkdir()
        bad_file = fake_pkg / "_drift92_negative.py"
        bad_file.write_text("# @canonical-source docs/research-input/topstep/THIS_FILE_DOES_NOT_EXIST.md\n")

        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_pkg)
        monkeypatch.setattr(check_drift, "PIPELINE_DIR", tmp_path / "pipeline")
        monkeypatch.setattr(check_drift, "SCRIPTS_DIR", tmp_path / "scripts")
        # PROJECT_ROOT must remain at the real repo so the check can resolve
        # the canonical paths under docs/research-input/.

        violations = check_drift.check_canonical_source_annotations()
        assert len(violations) == 1
        assert "THIS_FILE_DOES_NOT_EXIST" in violations[0]
        assert "_drift92_negative.py" in violations[0]

    def test_ignores_archive_dirs(self, tmp_path, monkeypatch):
        """Files under archive/ are NOT scanned even with broken refs."""
        from pipeline import check_drift

        fake_pkg = tmp_path / "trading_app"
        archive = fake_pkg / "archive"
        archive.mkdir(parents=True)
        (archive / "old.py").write_text("# @canonical-source docs/research-input/MISSING.md\n")

        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_pkg)
        monkeypatch.setattr(check_drift, "PIPELINE_DIR", tmp_path / "pipeline")
        monkeypatch.setattr(check_drift, "SCRIPTS_DIR", tmp_path / "scripts")

        violations = check_drift.check_canonical_source_annotations()
        assert violations == []

    def test_ignores_placeholder_refs(self, tmp_path, monkeypatch):
        """Refs starting with `<` (template placeholders) are skipped."""
        from pipeline import check_drift

        fake_pkg = tmp_path / "trading_app"
        fake_pkg.mkdir()
        (fake_pkg / "doc_template.py").write_text("# @canonical-source <relative path from project root>\n")

        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_pkg)
        monkeypatch.setattr(check_drift, "PIPELINE_DIR", tmp_path / "pipeline")
        monkeypatch.setattr(check_drift, "SCRIPTS_DIR", tmp_path / "scripts")

        violations = check_drift.check_canonical_source_annotations()
        assert violations == []


class TestPhase4ShaIntegrity:
    """Phase 4 Stage 4.1 drift check #94: stamped hypothesis_file_sha must
    reference a real file in docs/audit/hypotheses/.

    These tests use an in-memory DuckDB fixture rather than monkeypatching
    the default gold.db path. Each test builds a minimal
    experimental_strategies table, inserts synthetic rows, and calls
    check_phase_4_sha_integrity with the fixture connection. The tests
    exercise the actual SQL and resolver logic without touching gold.db.
    """

    @staticmethod
    def _make_experimental_db():
        """In-memory DB with a minimal experimental_strategies schema
        containing the two columns the check reads: hypothesis_file_sha
        and created_at."""
        import duckdb

        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE experimental_strategies (
                strategy_id TEXT,
                hypothesis_file_sha TEXT,
                created_at TIMESTAMPTZ
            )
            """
        )
        return con

    def test_empty_db_passes(self):
        """Zero rows → zero violations."""
        from pipeline import check_drift

        con = self._make_experimental_db()
        violations = check_drift.check_phase_4_sha_integrity(con=con)
        assert violations == []
        con.close()

    def test_pre_ship_row_with_orphan_sha_is_grandfathered(self):
        """Rows with created_at < PHASE_4_1_SHIP_DATE are outside the
        check's scope even if their SHA is orphaned."""
        from datetime import UTC, datetime

        from pipeline import check_drift

        con = self._make_experimental_db()
        # Pre-ship-date row with an orphaned SHA → should be ignored
        con.execute(
            "INSERT INTO experimental_strategies VALUES (?, ?, ?)",
            ["s1", "orphan_sha_pre_ship", datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)],
        )
        violations = check_drift.check_phase_4_sha_integrity(con=con)
        assert violations == []
        con.close()

    def test_post_ship_row_with_null_sha_is_out_of_scope(self):
        """NULL hypothesis_file_sha is legacy-mode and outside the check's
        scope regardless of created_at. The validator handles legacy rows
        via _is_phase_4_grandfathered."""
        from datetime import UTC, datetime

        from pipeline import check_drift

        con = self._make_experimental_db()
        # Post-ship row with NULL SHA → legacy-mode, out of scope
        con.execute(
            "INSERT INTO experimental_strategies VALUES (?, ?, ?)",
            ["s1", None, datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)],
        )
        violations = check_drift.check_phase_4_sha_integrity(con=con)
        assert violations == []
        con.close()

    def test_post_ship_row_with_orphan_sha_is_flagged(self):
        """Post-ship-date + non-null SHA + no matching file → violation."""
        from datetime import UTC, datetime

        from pipeline import check_drift

        con = self._make_experimental_db()
        # Post-ship row with an orphaned SHA → should fire
        con.execute(
            "INSERT INTO experimental_strategies VALUES (?, ?, ?)",
            ["s1", "orphan_sha_" + "a" * 53, datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)],
        )
        violations = check_drift.check_phase_4_sha_integrity(con=con)
        assert len(violations) == 1
        assert "orphaned SHA" in violations[0]
        assert "1 row" in violations[0]
        con.close()

    def test_post_ship_row_with_valid_sha_passes(self, tmp_path, monkeypatch):
        """Post-ship-date + non-null SHA + SHA resolves to a real file → pass.

        Uses monkeypatch to redirect the hypothesis registry directory to
        tmp_path where we can control what files exist.
        """
        import hashlib
        from datetime import UTC, datetime

        from pipeline import check_drift
        from trading_app import hypothesis_loader

        # Write a real hypothesis file in tmp_path and compute its SHA
        hyp_file = tmp_path / "2026-04-09-valid.yaml"
        hyp_file.write_text("metadata:\n  name: valid\n", encoding="utf-8")
        real_sha = hashlib.sha256(hyp_file.read_bytes()).hexdigest()

        # Redirect the registry directory
        monkeypatch.setattr(hypothesis_loader, "_HYPOTHESIS_DIR", tmp_path)

        con = self._make_experimental_db()
        con.execute(
            "INSERT INTO experimental_strategies VALUES (?, ?, ?)",
            ["s1", real_sha, datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)],
        )
        violations = check_drift.check_phase_4_sha_integrity(con=con)
        assert violations == [], f"expected pass, got {violations}"
        con.close()

    def test_multiple_rows_sharing_orphan_sha_reports_once(self):
        """When multiple rows share the same orphaned SHA, the check
        deduplicates via SELECT DISTINCT and reports once with a count."""
        from datetime import UTC, datetime

        from pipeline import check_drift

        con = self._make_experimental_db()
        shared_sha = "shared_orphan_" + "b" * 50
        for i in range(5):
            con.execute(
                "INSERT INTO experimental_strategies VALUES (?, ?, ?)",
                [f"s{i}", shared_sha, datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)],
            )
        violations = check_drift.check_phase_4_sha_integrity(con=con)
        assert len(violations) == 1  # DISTINCT — one violation for 5 rows
        assert "5 row" in violations[0]
        con.close()


class TestPropProfilesValidatedSetupsAlignment:
    """Drift check #95: every DailyLaneSpec in an ``active=True`` AccountProfile
    must exist in ``validated_setups`` with ``status='active'``.

    Rationale: the 2026-04-09 alignment audit found that all 5 deployed lanes
    in ``topstep_50k_mnq_auto`` were GHOSTS — strategy_ids not present in
    validated_setups or experimental_strategies. The bot was operating with
    zero current validation backing against real money. This check prevents
    that class of drift from recurring: if prop_profiles.py references a lane
    that is not in the current validated book, the check fires before the bot
    can be launched.

    Inactive profiles (``active=False``) are exempt because they don't affect
    runtime — they're held as reference templates for future activation. Any
    lane in an inactive profile is re-validated at the point of activation
    via this same check (the profile flip to ``active=True`` will re-run the
    drift check).
    """

    @staticmethod
    def _make_validated_setups_db():
        """In-memory DB with a minimal validated_setups schema containing the
        two columns the check reads: strategy_id and status."""
        import duckdb

        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE validated_setups (
                strategy_id TEXT,
                status TEXT
            )
            """
        )
        return con

    @staticmethod
    def _runtime_lanes_for_profile(profile):
        """Mirror the drift check's lane source: daily_lanes, else allocation lanes."""
        from trading_app.prop_profiles import load_allocation_lanes

        lanes = profile.daily_lanes
        if not lanes:
            lanes = load_allocation_lanes(profile.profile_id)
        return lanes

    def test_all_active_lanes_present_passes(self):
        """Happy path: every active lane is in validated_setups with status='active'."""
        from pipeline import check_drift
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        con = self._make_validated_setups_db()
        # Insert every lane from every active profile with status='active'
        for profile in ACCOUNT_PROFILES.values():
            if not profile.active:
                continue
            for lane in self._runtime_lanes_for_profile(profile):
                con.execute(
                    "INSERT INTO validated_setups VALUES (?, ?)",
                    [lane.strategy_id, "active"],
                )
        violations = check_drift.check_prop_profiles_validated_alignment(con=con)
        assert violations == [], f"unexpected violations: {violations}"
        con.close()

    def test_missing_lane_flagged(self):
        """A deployed lane absent from validated_setups fires a violation."""
        from pipeline import check_drift
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        con = self._make_validated_setups_db()
        # Insert every active lane EXCEPT one chosen victim
        active_profiles = [p for p in ACCOUNT_PROFILES.values() if p.active]
        assert active_profiles, "No active profiles — test fixture assumption broken"
        victim_profile = active_profiles[0]
        victim_lanes = self._runtime_lanes_for_profile(victim_profile)
        assert victim_lanes, "Active profile has no runtime lanes — test fixture broken"
        victim_lane = victim_lanes[0]
        for profile in active_profiles:
            for lane in self._runtime_lanes_for_profile(profile):
                if lane.strategy_id == victim_lane.strategy_id:
                    continue
                con.execute(
                    "INSERT INTO validated_setups VALUES (?, ?)",
                    [lane.strategy_id, "active"],
                )
        violations = check_drift.check_prop_profiles_validated_alignment(con=con)
        assert len(violations) == 1, f"expected 1 violation, got {len(violations)}: {violations}"
        assert victim_lane.strategy_id in violations[0]
        assert victim_profile.profile_id in violations[0]
        con.close()

    def test_retired_lane_flagged(self):
        """A deployed lane present in validated_setups but with status='retired'
        fires a violation — only status='active' counts as backing."""
        from pipeline import check_drift
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        con = self._make_validated_setups_db()
        active_profiles = [p for p in ACCOUNT_PROFILES.values() if p.active]
        assert active_profiles
        for profile in active_profiles:
            for i, lane in enumerate(self._runtime_lanes_for_profile(profile)):
                # First lane gets status='retired', rest get 'active'
                status = "retired" if i == 0 else "active"
                con.execute(
                    "INSERT INTO validated_setups VALUES (?, ?)",
                    [lane.strategy_id, status],
                )
        violations = check_drift.check_prop_profiles_validated_alignment(con=con)
        assert len(violations) >= 1
        assert any("retired" in v or "not active" in v for v in violations)
        con.close()

    def test_inactive_profile_exempt(self):
        """Inactive profiles are exempt — their lanes may be ghosts without
        firing a violation."""
        from pipeline import check_drift
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        con = self._make_validated_setups_db()
        # Populate only active profile lanes. Inactive profile lanes are
        # deliberately NOT inserted → they would fire if the check were
        # scanning them, but should be exempt.
        for profile in ACCOUNT_PROFILES.values():
            if not profile.active:
                continue
            for lane in self._runtime_lanes_for_profile(profile):
                con.execute(
                    "INSERT INTO validated_setups VALUES (?, ?)",
                    [lane.strategy_id, "active"],
                )
        violations = check_drift.check_prop_profiles_validated_alignment(con=con)
        # Should pass because only inactive profiles have unbacked lanes
        assert violations == [], f"inactive profile lanes leaked into check: {violations}"
        con.close()


class TestCanonicalClaudeClientSource:
    """Stage 4 of claude-api-modernization: lock trading_app/ai/claude_client.py
    as the sole source of hardcoded Claude model IDs and direct
    anthropic.Anthropic(...) client constructions.
    """

    def test_catches_offenders_via_injection(self, tmp_path, monkeypatch):
        """Inject a file with a hardcoded model ID AND a direct
        anthropic.Anthropic() construction — both must be flagged.

        Also verify the current repo is clean: before injection, the check
        returns zero violations (Stages 1-3 migrated every call site).
        """
        from pipeline import check_drift

        # Baseline: current repo must be clean.
        baseline = check_drift.check_canonical_claude_client_source()
        assert baseline == [], f"Clean repo expected zero violations. Offenders: {baseline}"

        # Inject a rogue file with both offending patterns.
        fake_trading = tmp_path / "trading_app"
        fake_trading.mkdir()
        offender = fake_trading / "rogue_ai.py"
        offender.write_text(
            "import anthropic\n"
            "\n"
            "def bad():\n"
            '    client = anthropic.Anthropic(api_key="k")\n'
            '    return client.messages.create(model="claude-opus-4-7", max_tokens=10)\n'
        )
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(check_drift, "TRADING_APP_DIR", fake_trading)
        monkeypatch.setattr(check_drift, "PIPELINE_DIR", empty)
        monkeypatch.setattr(check_drift, "SCRIPTS_DIR", empty)
        monkeypatch.setattr(check_drift, "RESEARCH_DIR", empty)

        violations = check_drift.check_canonical_claude_client_source()
        assert len(violations) == 2, f"Expected 2 violations, got {len(violations)}: {violations}"
        assert any("rogue_ai.py" in v and "claude-opus-4-7" in v for v in violations), (
            f"Missing model-ID violation: {violations}"
        )
        assert any("rogue_ai.py" in v and "anthropic.Anthropic(" in v for v in violations), (
            f"Missing direct-construction violation: {violations}"
        )


class TestLaneAllocationChordiaGate:
    """check_lane_allocation_chordia_gate refuses lanes failing the gate.

    The check reads ``PROJECT_ROOT / "docs/runtime/lane_allocation.json"``
    (absolute path via the module's PROJECT_ROOT constant — explicitly NOT
    CWD-relative). Each test monkeypatches PROJECT_ROOT to point at tmp_path
    and writes a controlled fixture file there.

    Stage: docs/runtime/stages/allocator-chordia-gate.md.
    Companion to trading_app.lane_allocator.apply_chordia_gate.
    """

    def _write_alloc(self, tmp_path: Path, lanes: list[dict]) -> None:
        import json

        runtime = tmp_path / "docs" / "runtime"
        runtime.mkdir(parents=True)
        (runtime / "lane_allocation.json").write_text(
            json.dumps(
                {
                    "rebalance_date": "2026-05-01",
                    "trailing_window_months": 12,
                    "profile_id": "test_profile",
                    "lanes": lanes,
                    "paused": [],
                    "all_scores_count": len(lanes),
                }
            )
        )

    def _patch_root(self, monkeypatch, tmp_path: Path) -> None:
        from pipeline import check_drift

        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)

    def test_passes_when_file_absent(self, tmp_path, monkeypatch):
        """Missing lane_allocation.json returns no violations (allowed in fresh worktrees)."""
        from pipeline.check_drift import check_lane_allocation_chordia_gate

        self._patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_chordia_gate() == []

    def test_passes_with_clean_lane(self, tmp_path, monkeypatch):
        """A lane with PASS_PROTOCOL_A + fresh audit passes the check."""
        from pipeline.check_drift import check_lane_allocation_chordia_gate

        self._write_alloc(
            tmp_path,
            [
                {
                    "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
                    "chordia_verdict": "PASS_PROTOCOL_A",
                    "chordia_audit_age_days": 10,
                }
            ],
        )
        self._patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_chordia_gate() == []

    def test_fails_on_fail_both(self, tmp_path, monkeypatch):
        """A FAIL_BOTH lane in lane_allocation.json triggers a violation."""
        from pipeline.check_drift import check_lane_allocation_chordia_gate

        self._write_alloc(
            tmp_path,
            [
                {
                    "strategy_id": "BAD_LANE",
                    "chordia_verdict": "FAIL_BOTH",
                    "chordia_audit_age_days": 5,
                }
            ],
        )
        self._patch_root(monkeypatch, tmp_path)
        violations = check_lane_allocation_chordia_gate()
        assert len(violations) == 1
        assert "BAD_LANE" in violations[0]
        assert "FAIL_BOTH" in violations[0]

    def test_fails_on_missing_verdict(self, tmp_path, monkeypatch):
        """A lane with chordia_verdict missing entirely is rejected."""
        from pipeline.check_drift import check_lane_allocation_chordia_gate

        self._write_alloc(
            tmp_path,
            [
                {
                    "strategy_id": "OLD_LANE",
                    # no chordia_verdict, no chordia_audit_age_days
                }
            ],
        )
        self._patch_root(monkeypatch, tmp_path)
        violations = check_lane_allocation_chordia_gate()
        assert len(violations) == 1
        assert "OLD_LANE" in violations[0]
        assert "missing chordia_verdict" in violations[0]

    def test_fails_on_stale_audit(self, tmp_path, monkeypatch):
        """A PASS_PROTOCOL_A lane with audit_age > 90d is rejected as stale."""
        from pipeline.check_drift import check_lane_allocation_chordia_gate

        self._write_alloc(
            tmp_path,
            [
                {
                    "strategy_id": "STALE_LANE",
                    "chordia_verdict": "PASS_PROTOCOL_A",
                    "chordia_audit_age_days": 91,
                }
            ],
        )
        self._patch_root(monkeypatch, tmp_path)
        violations = check_lane_allocation_chordia_gate()
        assert len(violations) == 1
        assert "STALE_LANE" in violations[0]
        assert "stale" in violations[0].lower()

    def test_robust_to_non_root_cwd(self, tmp_path, monkeypatch):
        """Regression: check must not silently pass from a non-root cwd.

        Pre-fix the check used a CWD-relative path; this test would have
        passed with the bug because tmp_path / 'subdir' has no
        lane_allocation.json under it. Post-fix we patch PROJECT_ROOT at
        tmp_path AND chdir into a subdir — the check must still find the
        file at the patched root.
        """
        from pipeline.check_drift import check_lane_allocation_chordia_gate

        self._write_alloc(
            tmp_path,
            [
                {
                    "strategy_id": "BAD_LANE",
                    "chordia_verdict": "FAIL_BOTH",
                    "chordia_audit_age_days": 0,
                }
            ],
        )
        self._patch_root(monkeypatch, tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        violations = check_lane_allocation_chordia_gate()
        assert len(violations) == 1, (
            "Drift check must resolve via PROJECT_ROOT, not CWD — otherwise it "
            "fail-opens whenever invoked from a non-root directory."
        )
        assert "BAD_LANE" in violations[0]

    def test_fails_on_empty_lanes(self, tmp_path, monkeypatch):
        """Regression: empty lanes[] must NOT silently pass.

        Pre-fix the check returned [] when lanes was empty — meaning a
        producer crash mid-write or a hand-edit that emptied the array
        certified the broken state as healthy. Post-fix the check fails
        loud: file-exists + empty-lanes is not a legitimate state.
        """
        from pipeline.check_drift import check_lane_allocation_chordia_gate

        self._write_alloc(tmp_path, [])  # explicit empty lanes
        self._patch_root(monkeypatch, tmp_path)
        violations = check_lane_allocation_chordia_gate()
        assert len(violations) == 1
        assert "empty lanes[]" in violations[0]

    def test_fails_when_chordia_doctrine_load_raises(self, tmp_path, monkeypatch):
        """Regression: if load_chordia_audit_log raises, the check must
        emit a violation rather than silently fall back to freshness=90.

        Pre-fix the broad except swallowed any error and continued the
        audit with a hardcoded threshold — hiding doctrine corruption
        behind a passing-looking check.
        """
        from pipeline import check_drift
        from pipeline.check_drift import check_lane_allocation_chordia_gate

        # Real lane (one that would otherwise pass) so we test the load
        # path specifically, not other validations.
        self._write_alloc(
            tmp_path,
            [
                {
                    "strategy_id": "WOULD_PASS",
                    "chordia_verdict": "PASS_PROTOCOL_A",
                    "chordia_audit_age_days": 10,
                }
            ],
        )
        self._patch_root(monkeypatch, tmp_path)

        # Force the doctrine import to fail. Patching the module's import
        # cache means `from trading_app.chordia import ...` inside the
        # check raises ImportError.
        import sys

        monkeypatch.setitem(sys.modules, "trading_app.chordia", None)

        violations = check_lane_allocation_chordia_gate()
        assert len(violations) == 1
        assert "Cannot load chordia freshness threshold" in violations[0]
        assert "audit threshold unverified" in violations[0]


class TestStageAcceptanceParser:
    """Tests for stage-file acceptance command parsing (Check #121 Mode A)."""

    def test_parses_yaml_acceptance_list_with_backtick_command(self):
        text = (
            "---\n"
            "task: foo\n"
            "mode: IMPLEMENTATION\n"
            "acceptance:\n"
            "  - `pytest tests/test_foo.py -q` passes\n"
            "  - drift check exits 0\n"
            "---\n"
            "## Body\n"
            "blah\n"
        )
        cmds = _parse_stage_acceptance_commands(text)
        assert "pytest tests/test_foo.py -q" in cmds

    def test_parses_markdown_acceptance_section(self):
        text = (
            "---\n"
            "task: foo\n"
            "---\n"
            "## Acceptance\n"
            "- `pytest tests/test_bar.py -q` green\n"
            "- `python pipeline/check_drift.py` exits 0\n"
            "## Notes\n"
            "- ignored: `rm -rf /`\n"
        )
        cmds = _parse_stage_acceptance_commands(text)
        assert "pytest tests/test_bar.py -q" in cmds
        assert "python pipeline/check_drift.py" in cmds
        # Ignore section: rm command must NOT be picked up regardless of section
        assert not any("rm" in c for c in cmds)

    def test_drops_dangerous_shell_metachars(self):
        text = (
            "---\n"
            "task: bad\n"
            "---\n"
            "## Acceptance\n"
            "- `pytest foo > /tmp/out` ignored due to redirect\n"
            "- `python a.py | grep b` ignored due to pipe\n"
            "- `python a.py; rm b` ignored due to semicolon\n"
            "- `python clean.py` kept\n"
        )
        cmds = _parse_stage_acceptance_commands(text)
        assert "python clean.py" in cmds
        assert len(cmds) == 1

    def test_non_runnable_acceptance_returns_empty(self):
        text = (
            "---\n"
            "task: prose-only\n"
            "---\n"
            "## Acceptance\n"
            "- Verdict committed to docs/audit/results/.\n"
            "- User confirmed via /capital-review.\n"
        )
        cmds = _parse_stage_acceptance_commands(text)
        assert cmds == []


class TestStageAcceptanceRunner:
    """Tests for _stage_acceptance_all_pass (Check #121 Mode A executor)."""

    def test_empty_command_list_returns_false_zero(self):
        all_pass, n_ran = _stage_acceptance_all_pass([])
        assert all_pass is False
        assert n_ran == 0

    def test_all_passing_returns_true_with_count(self):
        # python -c is rejected by allowlist? No — head is "python".
        # But we don't want to actually allow `-c` injection in production;
        # the allowlist guards by HEAD only. This test verifies the executor
        # contract on a benign all-pass case.
        cmds = ["python --version"]
        all_pass, n_ran = _stage_acceptance_all_pass(cmds)
        assert all_pass is True
        assert n_ran == 1

    def test_failing_command_short_circuits(self):
        # Keep this portable: GitHub's Windows runner has ls.exe from Git on PATH,
        # but local Windows shells may not.
        cmds = ['python -c "raise SystemExit(7)"']
        all_pass, n_ran = _stage_acceptance_all_pass(cmds)
        assert all_pass is False
        assert n_ran == 1

    def test_disallowed_head_short_circuits(self):
        # `cat` is not in the allowlist; the runner must reject it even if
        # the parser somehow let it through.
        cmds = ["cat /etc/hostname"]
        all_pass, n_ran = _stage_acceptance_all_pass(cmds)
        assert all_pass is False
        assert n_ran == 0


class TestIsoUtcFormatterSilentNone:
    """Tests for the iso_utc silent-None formatter class-bug check."""

    @staticmethod
    def _parse_first_func(src: str):
        import ast as _ast

        tree = _ast.parse(src)
        for node in tree.body:
            if isinstance(node, _ast.FunctionDef | _ast.AsyncFunctionDef):
                return node
        raise AssertionError("no function in source")

    def test_predicate_triggers_on_silent_formatter_shape(self):
        from pipeline.check_drift import _function_has_isinstance_then_silent_none

        src = "def f(v):\n    if isinstance(v, datetime):\n        return v.isoformat()\n    return None\n"
        node = self._parse_first_func(src)
        assert _function_has_isinstance_then_silent_none(node) is True

    def test_predicate_passes_when_log_warning_present(self):
        from pipeline.check_drift import _function_has_isinstance_then_silent_none

        src = (
            "def f(v):\n"
            "    if isinstance(v, datetime):\n"
            "        return v.isoformat()\n"
            "    log.warning('bad type %s', type(v).__name__)\n"
            "    return None\n"
        )
        node = self._parse_first_func(src)
        assert _function_has_isinstance_then_silent_none(node) is False

    def test_predicate_passes_when_log_critical_present(self):
        # Round 1 regression guard — predicate must accept .critical not just .warning
        from pipeline.check_drift import _function_has_isinstance_then_silent_none

        src = (
            "def f(v):\n"
            "    if isinstance(v, datetime):\n"
            "        return v.isoformat()\n"
            "    log.critical('unrecoverable type %s', type(v).__name__)\n"
            "    return None\n"
        )
        node = self._parse_first_func(src)
        assert _function_has_isinstance_then_silent_none(node) is False

    def test_predicate_passes_when_log_error_present(self):
        from pipeline.check_drift import _function_has_isinstance_then_silent_none

        src = (
            "def f(v):\n"
            "    if isinstance(v, datetime):\n"
            "        return v.isoformat()\n"
            "    logger.error('bad type %s', type(v).__name__)\n"
            "    return None\n"
        )
        node = self._parse_first_func(src)
        assert _function_has_isinstance_then_silent_none(node) is False

    def test_predicate_passes_on_pure_null_passthrough(self):
        # Regression guard: _iso_utc's `if v is None: return None` short-circuit
        # must NOT be confused with the silent-formatter shape — the
        # null-passthrough has no isinstance call.
        from pipeline.check_drift import _function_has_isinstance_then_silent_none

        src = "def f(v):\n    if v is None:\n        return None\n    return str(v)\n"
        node = self._parse_first_func(src)
        assert _function_has_isinstance_then_silent_none(node) is False

    def test_check_respects_silent_none_policy_annotation(self, tmp_path, monkeypatch):
        # End-to-end: a synthetic file matching the trigger shape but bearing
        # the `# silent-none-policy: <reason>` annotation must produce no
        # violation.
        from pipeline import check_drift
        from pipeline.check_drift import check_iso_utc_formatter_silent_none

        synthetic = tmp_path / "fake_module.py"
        synthetic.write_text(
            "from datetime import datetime\n"
            "log = None\n"
            "# silent-none-policy: upstream-coerced\n"
            "def f(v):\n"
            "    if isinstance(v, datetime):\n"
            "        return v.isoformat()\n"
            "    return None\n",
            encoding="utf-8",
        )
        rel = synthetic.relative_to(tmp_path).as_posix()
        monkeypatch.setattr(check_drift, "_ISO_UTC_FORMATTER_SCAN_FILES", (rel,))
        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        violations = check_iso_utc_formatter_silent_none()
        assert violations == []

    def test_check_clean_against_real_canonical_files(self):
        # Smoke test: against the real codebase, `check_iso_utc_formatter_silent_none`
        # must return [] — proves the predicate is correctly tuned against
        # `_iso_utc` (logs warning) and that no other formatter helper in the
        # scan set silently drops type-mismatched values.
        from pipeline.check_drift import check_iso_utc_formatter_silent_none

        violations = check_iso_utc_formatter_silent_none()
        assert violations == [], f"unexpected violations: {violations}"


class TestQuietModeOutputSanitization:
    """`python pipeline/check_drift.py --quiet` emits sanitized lines only.

    Acceptance criterion 5: every emitted line must match `PASS: <name>`,
    `FAIL: <name> (count=N)`, `ADVISORY: <name>`, `SKIP: <name>`, or the
    final `SUMMARY: ...` line. No file paths, SQL fragments, or DB internals.
    """

    def test_quiet_mode_lines_are_sanitized(self, tmp_path):
        import re
        import subprocess
        import sys

        proj_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(proj_root / "pipeline" / "check_drift.py"), "--quiet", "--fast"],
            capture_output=True,
            text=True,
            cwd=str(proj_root),
        )
        # Exit 0 = clean, 1 = drift; either is fine for sanitization test.
        assert result.returncode in (0, 1), f"unexpected exit: {result.stderr}"
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert lines, "quiet mode produced no output"
        allowed = re.compile(r"^(?:PASS|FAIL|ADVISORY|SKIP):\s.+$|^SUMMARY:\s.+$")
        for line in lines:
            assert allowed.match(line), f"unsanitized line leaked: {line!r}"
        # The summary line is required and last-emitted.
        assert lines[-1].startswith("SUMMARY:"), f"missing summary line: {lines[-1]!r}"

    def test_quiet_mode_summary_carries_passed_count(self, tmp_path):
        import subprocess
        import sys

        proj_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(proj_root / "pipeline" / "check_drift.py"), "--quiet", "--fast"],
            capture_output=True,
            text=True,
            cwd=str(proj_root),
        )
        summary = [line for line in result.stdout.splitlines() if line.startswith("SUMMARY:")]
        assert len(summary) == 1
        assert "passed=" in summary[0]


class TestLiteratureExtractsModeABFraming:
    """Check 142 — literature extracts citing research/output/ must carry
    explicit Mode A / Mode B / HOLDOUT_SACRED_FROM / grandfathered framing.

    Origin: 2026-05-07 self-review commit 2ea6fc5e — composite-N + Mode A/B
    conflation class bug.
    """

    def _setup(self, tmp_path, monkeypatch):
        from pipeline import check_drift as cd

        lit_dir = tmp_path / "docs" / "institutional" / "literature"
        lit_dir.mkdir(parents=True)
        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        return cd, lit_dir

    def test_passes_when_no_research_output_citation(self, tmp_path, monkeypatch):
        cd, lit_dir = self._setup(tmp_path, monkeypatch)
        (lit_dir / "external_only_paper.md").write_text(
            "# External Paper\n\nQuoted from a 2024 SSRN paper. No internal result references at all.\n",
            encoding="utf-8",
        )
        assert cd.check_literature_extracts_mode_a_b_framing() == []

    def test_fails_when_cites_research_output_without_framing(self, tmp_path, monkeypatch):
        cd, lit_dir = self._setup(tmp_path, monkeypatch)
        (lit_dir / "bad_extract.md").write_text(
            "# Bad Extract\n\nCites research/output/dalton_summary.csv with no holdout framing at all. Just numbers.\n",
            encoding="utf-8",
        )
        violations = cd.check_literature_extracts_mode_a_b_framing()
        assert len(violations) == 1
        assert "bad_extract.md" in violations[0]
        assert "Mode A" in violations[0] or "Mode B" in violations[0]

    @pytest.mark.parametrize(
        "framing_token",
        ["Mode A", "Mode B", "HOLDOUT_SACRED_FROM", "grandfathered"],
    )
    def test_passes_with_any_recognized_framing_token(self, tmp_path, monkeypatch, framing_token):
        cd, lit_dir = self._setup(tmp_path, monkeypatch)
        (lit_dir / "good_extract.md").write_text(
            f"# Good Extract\n\nCites research/output/dalton_summary.csv. "
            f"Note: this result was produced under {framing_token} regime.\n",
            encoding="utf-8",
        )
        assert cd.check_literature_extracts_mode_a_b_framing() == []

    def test_case_insensitive_framing_match(self, tmp_path, monkeypatch):
        cd, lit_dir = self._setup(tmp_path, monkeypatch)
        (lit_dir / "good_lower.md").write_text(
            "Cites research/output/foo.csv under mode a holdout window.\n",
            encoding="utf-8",
        )
        assert cd.check_literature_extracts_mode_a_b_framing() == []

    def test_pending_acquisition_files_exempt(self, tmp_path, monkeypatch):
        cd, lit_dir = self._setup(tmp_path, monkeypatch)
        (lit_dir / "PENDING_ACQUISITION_foo.md").write_text(
            "Cites research/output/bar.csv with no framing — exempt by filename prefix.\n",
            encoding="utf-8",
        )
        assert cd.check_literature_extracts_mode_a_b_framing() == []

    def test_check_callable_imports_cleanly(self):
        from pipeline.check_drift import check_literature_extracts_mode_a_b_framing

        assert callable(check_literature_extracts_mode_a_b_framing)


class TestCheckSrPausesHaveRecentEvidence:
    """Drift check for stale SR-monitor pauses (2026-05-11 misread incident)."""

    THRESHOLD = 31.96

    def _setup(self, tmp_path, monkeypatch):
        import pipeline.check_drift as cd

        state_dir = tmp_path / "data" / "state"
        state_dir.mkdir(parents=True)
        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        return cd, state_dir

    def _write_sr_state(self, state_dir, profile_id, lanes):
        import json

        state_dir.joinpath("sr_state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state_type": "sr_monitor",
                    "canonical_inputs": {"profile_id": profile_id, "lane_ids": [l["strategy_id"] for l in lanes]},
                    "payload": {"results": lanes},
                }
            ),
            encoding="utf-8",
        )

    def _write_overrides(self, state_dir, profile_id, overrides):
        import json

        state_dir.joinpath(f"lane_overrides_{profile_id}.json").write_text(
            json.dumps(overrides),
            encoding="utf-8",
        )

    def _stale_lane(self, sid):
        return {
            "strategy_id": sid,
            "status": "ALARM",
            "sr_stat": 41.6,
            "current_sr_stat": 0.5,
            "trades_since_alarm": 34,
            "recent_10_mean_r": 0.47,
            "threshold": self.THRESHOLD,
        }

    def _pause_entry(self, source="sr_monitor"):
        return {"active": False, "source": source, "reason": "SR alarm: stat=41.60 ..."}

    def test_passes_when_no_state_files_exist(self, tmp_path, monkeypatch):
        cd, _state_dir = self._setup(tmp_path, monkeypatch)
        assert cd.check_sr_pauses_have_recent_evidence() == []

    def test_fires_when_paused_lane_has_recovered(self, tmp_path, monkeypatch):
        cd, state_dir = self._setup(tmp_path, monkeypatch)
        sid = "MNQ_FOO_BAR"
        self._write_sr_state(state_dir, "p1", [self._stale_lane(sid)])
        self._write_overrides(state_dir, "p1", {sid: self._pause_entry()})
        violations = cd.check_sr_pauses_have_recent_evidence()
        assert len(violations) == 1
        assert sid in violations[0]
        assert "stale" in violations[0].lower()
        assert "watch" in violations[0]

    def test_does_not_fire_when_sr_still_above_half_threshold(self, tmp_path, monkeypatch):
        cd, state_dir = self._setup(tmp_path, monkeypatch)
        sid = "MNQ_FOO_BAR"
        lane = self._stale_lane(sid)
        lane["current_sr_stat"] = self.THRESHOLD * 0.75  # 75% of threshold — not recovered
        self._write_sr_state(state_dir, "p1", [lane])
        self._write_overrides(state_dir, "p1", {sid: self._pause_entry()})
        assert cd.check_sr_pauses_have_recent_evidence() == []

    def test_does_not_fire_when_too_few_trades_since_alarm(self, tmp_path, monkeypatch):
        cd, state_dir = self._setup(tmp_path, monkeypatch)
        sid = "MNQ_FOO_BAR"
        lane = self._stale_lane(sid)
        lane["trades_since_alarm"] = 5  # < 10 minimum
        self._write_sr_state(state_dir, "p1", [lane])
        self._write_overrides(state_dir, "p1", {sid: self._pause_entry()})
        assert cd.check_sr_pauses_have_recent_evidence() == []

    def test_does_not_fire_when_recent_mean_r_negative(self, tmp_path, monkeypatch):
        cd, state_dir = self._setup(tmp_path, monkeypatch)
        sid = "MNQ_FOO_BAR"
        lane = self._stale_lane(sid)
        lane["recent_10_mean_r"] = -0.27  # negative recent performance
        self._write_sr_state(state_dir, "p1", [lane])
        self._write_overrides(state_dir, "p1", {sid: self._pause_entry()})
        assert cd.check_sr_pauses_have_recent_evidence() == []

    def test_does_not_fire_for_active_overrides(self, tmp_path, monkeypatch):
        cd, state_dir = self._setup(tmp_path, monkeypatch)
        sid = "MNQ_FOO_BAR"
        self._write_sr_state(state_dir, "p1", [self._stale_lane(sid)])
        self._write_overrides(state_dir, "p1", {sid: {"active": True, "source": "sr_monitor"}})
        assert cd.check_sr_pauses_have_recent_evidence() == []

    def test_does_not_fire_for_non_sr_monitor_pauses(self, tmp_path, monkeypatch):
        cd, state_dir = self._setup(tmp_path, monkeypatch)
        sid = "MNQ_FOO_BAR"
        self._write_sr_state(state_dir, "p1", [self._stale_lane(sid)])
        self._write_overrides(state_dir, "p1", {sid: self._pause_entry(source="manual")})
        assert cd.check_sr_pauses_have_recent_evidence() == []

    def test_skips_lanes_missing_new_fields(self, tmp_path, monkeypatch):
        """Backwards-compat: legacy sr_state.json without current_sr_stat/etc must not fire false positives."""
        cd, state_dir = self._setup(tmp_path, monkeypatch)
        sid = "MNQ_FOO_BAR"
        legacy_lane = {
            "strategy_id": sid,
            "status": "ALARM",
            "sr_stat": 41.6,
            "threshold": self.THRESHOLD,
            # no current_sr_stat / trades_since_alarm / recent_10_mean_r
        }
        self._write_sr_state(state_dir, "p1", [legacy_lane])
        self._write_overrides(state_dir, "p1", {sid: self._pause_entry()})
        assert cd.check_sr_pauses_have_recent_evidence() == []

    def test_check_callable_imports_cleanly(self):
        from pipeline.check_drift import check_sr_pauses_have_recent_evidence

        assert callable(check_sr_pauses_have_recent_evidence)


# --- Routine-TBBO slippage registry coverage (Stage 1, 2026-05-11) ---


class TestCheckRoutineTbboSlippageRegistryCoverage:
    """Fail-closed coverage check for `ROUTINE_TBBO_SLIPPAGE_REGISTRY`.

    Verifies the check fires on:
      - PASS pilot doc whose instrument is missing from the registry
      - WARN/FAIL pilot doc whose instrument is incorrectly registered
    And passes on:
      - matched PASS-and-registered
      - WARN/FAIL when the same instrument also has a PASS pilot
    """

    @staticmethod
    def _write_pilot_doc(results_dir, name: str, verdict: str) -> None:
        path = results_dir / name
        path.write_text(
            f"# pilot fixture\n\n## Verdict: **{verdict}**\n\nbody body\n",
            encoding="utf-8",
        )

    def _patch(self, monkeypatch, tmp_path, registry):
        from pipeline import check_drift as cd
        from trading_app import deployability as dep

        # Redirect PROJECT_ROOT so PROJECT_ROOT/docs/audit/results/ resolves to tmp_path.
        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dep, "ROUTINE_TBBO_SLIPPAGE_REGISTRY", registry)

        results_dir = tmp_path / "docs" / "audit" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def test_pass_pilot_with_matching_registry_passes(self, monkeypatch, tmp_path):
        from pipeline.check_drift import check_routine_tbbo_slippage_registry_coverage
        from trading_app.deployability import RoutineTbboPilot

        registry = {
            "MNQ": RoutineTbboPilot("MNQ", "E2", frozenset({"NYSE_OPEN"}), "basis"),
        }
        results_dir = self._patch(monkeypatch, tmp_path, registry)
        self._write_pilot_doc(results_dir, "2026-04-20-mnq-e2-slippage-pilot-v1.md", "PASS")

        violations = check_routine_tbbo_slippage_registry_coverage()
        assert violations == []

    def test_pass_pilot_with_missing_registry_fails_closed(self, monkeypatch, tmp_path):
        from pipeline.check_drift import check_routine_tbbo_slippage_registry_coverage

        registry = {}  # empty — MES PASS pilot has no registry coverage
        results_dir = self._patch(monkeypatch, tmp_path, registry)
        self._write_pilot_doc(results_dir, "2026-04-24-mes-e2-slippage-pilot-v1.md", "PASS")

        violations = check_routine_tbbo_slippage_registry_coverage()
        assert len(violations) == 1
        assert "MES" in violations[0]
        assert "missing from trading_app.deployability.ROUTINE_TBBO_SLIPPAGE_REGISTRY" in violations[0]

    def test_warn_pilot_with_registry_entry_fails_closed(self, monkeypatch, tmp_path):
        from pipeline.check_drift import check_routine_tbbo_slippage_registry_coverage
        from trading_app.deployability import RoutineTbboPilot

        # Instrument is registered but its only pilot v1 doc is WARN — should fail closed.
        registry = {
            "MGC": RoutineTbboPilot("MGC", "E2", frozenset({"LONDON_METALS"}), "basis"),
        }
        results_dir = self._patch(monkeypatch, tmp_path, registry)
        self._write_pilot_doc(results_dir, "2026-04-24-mgc-e2-slippage-pilot-v1.md", "WARN")

        violations = check_routine_tbbo_slippage_registry_coverage()
        assert len(violations) == 1
        assert "MGC" in violations[0]
        assert "non-PASS" in violations[0]
        assert "verdict=WARN" in violations[0]

    def test_newer_pass_supersedes_older_warn_for_same_instrument(self, monkeypatch, tmp_path):
        """LATEST-by-filename-date pilot is authoritative. Newer PASS overrides older WARN."""
        from pipeline.check_drift import check_routine_tbbo_slippage_registry_coverage
        from trading_app.deployability import RoutineTbboPilot

        registry = {
            "MGC": RoutineTbboPilot("MGC", "E2", frozenset({"LONDON_METALS"}), "basis"),
        }
        results_dir = self._patch(monkeypatch, tmp_path, registry)
        self._write_pilot_doc(results_dir, "2026-04-01-mgc-e2-slippage-pilot-v1.md", "WARN")
        self._write_pilot_doc(results_dir, "2026-05-01-mgc-e2-slippage-pilot-v1.md", "PASS")

        violations = check_routine_tbbo_slippage_registry_coverage()
        assert violations == []

    def test_newer_warn_supersedes_older_pass_for_same_instrument(self, monkeypatch, tmp_path):
        """LATEST-by-filename-date pilot is authoritative. Newer WARN refutes older PASS,
        and the registry entry must be flagged for removal — staleness cannot keep a
        registry entry alive after newer evidence refutes the original PASS verdict.
        """
        from pipeline.check_drift import check_routine_tbbo_slippage_registry_coverage
        from trading_app.deployability import RoutineTbboPilot

        # Same registry as the supersedes-PASS test, but the date ordering of the
        # pilot docs is REVERSED: an OLDER PASS exists, but the LATEST evidence is WARN.
        registry = {
            "MGC": RoutineTbboPilot("MGC", "E2", frozenset({"LONDON_METALS"}), "basis"),
        }
        results_dir = self._patch(monkeypatch, tmp_path, registry)
        self._write_pilot_doc(results_dir, "2026-04-01-mgc-e2-slippage-pilot-v1.md", "PASS")
        self._write_pilot_doc(results_dir, "2026-05-01-mgc-e2-slippage-pilot-v1.md", "WARN")

        violations = check_routine_tbbo_slippage_registry_coverage()
        assert len(violations) == 1
        assert "MGC" in violations[0]
        # Should cite the LATEST doc and verdict, not the older PASS.
        assert "2026-05-01-mgc-e2-slippage-pilot-v1.md" in violations[0]
        assert "verdict=WARN" in violations[0]

    def test_pilot_doc_without_verdict_line_fails_closed(self, monkeypatch, tmp_path):
        from pipeline.check_drift import check_routine_tbbo_slippage_registry_coverage

        results_dir = self._patch(monkeypatch, tmp_path, {})
        path = results_dir / "2026-05-01-mes-e2-slippage-pilot-v1.md"
        path.write_text("# fixture\n\n(no verdict line)\n", encoding="utf-8")

        violations = check_routine_tbbo_slippage_registry_coverage()
        assert len(violations) == 1
        assert "no `## Verdict:" in violations[0]

    def test_no_pilot_docs_returns_clean(self, monkeypatch, tmp_path):
        from pipeline.check_drift import check_routine_tbbo_slippage_registry_coverage

        self._patch(monkeypatch, tmp_path, {})
        # results_dir exists but is empty
        assert check_routine_tbbo_slippage_registry_coverage() == []

    def test_registered_instrument_with_no_pilot_doc_fails_closed(self, monkeypatch, tmp_path):
        """Registry membership without ANY committed pilot doc is silent over-coverage.

        Catches the no-evidence-at-all case the original set-difference loop missed:
        if a future maintainer adds an instrument to the registry without committing
        its pilot v1 doc (or a doc gets deleted/moved/renamed), the registry entry
        persists with no evidence backing it. The post-self-review symmetric loop
        catches this; the original implementation silently passed.
        """
        from pipeline.check_drift import check_routine_tbbo_slippage_registry_coverage
        from trading_app.deployability import RoutineTbboPilot

        registry = {
            "MES": RoutineTbboPilot("MES", "E2", frozenset({"COMEX_SETTLE"}), "basis"),
        }
        results_dir = self._patch(monkeypatch, tmp_path, registry)
        # results_dir is empty — no pilot docs at all
        assert results_dir.is_dir()

        violations = check_routine_tbbo_slippage_registry_coverage()
        assert len(violations) == 1
        assert "MES" in violations[0]
        assert "no `mes-...slippage-pilot-v1.md` doc was found" in violations[0]
        assert "silent over-coverage" in violations[0]


class TestVerdictVocabularyDrift:
    """Doctrine parity between RESEARCH_RULES.md and MCP-server vocab constants.

    Source-of-truth is the Python constants in
    ``scripts/tools/research_catalog_mcp_server.py``; the MD section is the
    documented mirror. Promotion landed 2026-05-14 (action-queue item
    ``research_catalog_verdict_vocabulary_doctrine_2026_05_12``).
    """

    def _canonical_md(self) -> Path:
        from pipeline.check_drift import PROJECT_ROOT

        return PROJECT_ROOT / "RESEARCH_RULES.md"

    def test_drift_clean_against_live_repo(self):
        """Live repo must be parity-clean — doubles as tripwire."""
        violations = check_verdict_vocabulary_md_matches_code()
        assert violations == [], "Verdict-vocab drift in live repo:\n  " + "\n  ".join(violations)

    def test_detects_missing_mapping_row(self, tmp_path):
        """Deleting one mapping row triggers VERDICT_VOCAB_MISSING_IN_MD."""
        import shutil

        src = self._canonical_md()
        dst = tmp_path / "RESEARCH_RULES.md"
        shutil.copy(src, dst)
        text = dst.read_text(encoding="utf-8")
        # Delete the canonical NO-GO -> NO-GO row by raw spelling.
        target_row = "| `NO-GO` | `NO-GO` |\n"
        assert target_row in text, "fixture row missing — update test if doctrine changed"
        tampered = text.replace(target_row, "", 1)
        dst.write_text(tampered, encoding="utf-8")

        violations = check_verdict_vocabulary_md_matches_code(md_path=dst)
        assert any("VERDICT_VOCAB_MISSING_IN_MD" in v and "'NO-GO'" in v for v in violations), violations

    def test_detects_priority_order_mismatch(self, tmp_path):
        """Swapping two adjacent priority rows triggers ORDER_MISMATCH."""
        import shutil

        src = self._canonical_md()
        dst = tmp_path / "RESEARCH_RULES.md"
        shutil.copy(src, dst)
        text = dst.read_text(encoding="utf-8")
        # Swap rows 1 and 2 of the priority table (NO-GO and NOGO are adjacent
        # and both retained in code, so swapping them stays a pure ORDER bug).
        row_1 = "| 1 | `NO-GO` |\n"
        row_2 = "| 2 | `NOGO` |\n"
        assert row_1 in text and row_2 in text, "fixture priority rows missing — update test if doctrine reorders"
        # Use placeholders to avoid the second replace eating the first row.
        swapped = text.replace(row_1, "<<TMP1>>").replace(row_2, "<<TMP2>>")
        # Swap the slot contents but keep the index cells intact.
        swapped = swapped.replace("<<TMP1>>", "| 1 | `NOGO` |\n")
        swapped = swapped.replace("<<TMP2>>", "| 2 | `NO-GO` |\n")
        dst.write_text(swapped, encoding="utf-8")

        violations = check_verdict_vocabulary_md_matches_code(md_path=dst)
        assert any("VERDICT_PRIORITY_ORDER_MISMATCH" in v for v in violations), violations

    def test_detects_extra_mapping_row(self, tmp_path):
        """Adding a mapping row not present in code triggers EXTRA_IN_MD."""
        import shutil

        src = self._canonical_md()
        dst = tmp_path / "RESEARCH_RULES.md"
        shutil.copy(src, dst)
        text = dst.read_text(encoding="utf-8")
        anchor = "| `NO-GO` | `NO-GO` |\n"
        assert anchor in text
        tampered = text.replace(anchor, anchor + "| `FABRICATED` | `NO-GO` |\n", 1)
        dst.write_text(tampered, encoding="utf-8")

        violations = check_verdict_vocabulary_md_matches_code(md_path=dst)
        assert any("VERDICT_VOCAB_EXTRA_IN_MD" in v and "FABRICATED" in v for v in violations), violations

    def test_detects_mapping_mismatch(self, tmp_path):
        """Same raw key mapped to a different canonical tag triggers MISMATCH."""
        import shutil

        src = self._canonical_md()
        dst = tmp_path / "RESEARCH_RULES.md"
        shutil.copy(src, dst)
        text = dst.read_text(encoding="utf-8")
        dead_row = "| `DEAD` | `NO-GO` |\n"
        assert dead_row in text
        tampered = text.replace(dead_row, "| `DEAD` | `PARK` |\n", 1)
        dst.write_text(tampered, encoding="utf-8")

        violations = check_verdict_vocabulary_md_matches_code(md_path=dst)
        assert any("VERDICT_VOCAB_MAPPING_MISMATCH" in v and "'DEAD'" in v for v in violations), violations

    def test_detects_section_missing(self, tmp_path):
        """Deleting the whole § Verdict Token Vocabulary triggers SECTION_MISSING."""
        import shutil

        src = self._canonical_md()
        dst = tmp_path / "RESEARCH_RULES.md"
        shutil.copy(src, dst)
        text = dst.read_text(encoding="utf-8")
        tampered = text.replace("## Verdict Token Vocabulary", "## Removed Header")
        dst.write_text(tampered, encoding="utf-8")

        violations = check_verdict_vocabulary_md_matches_code(md_path=dst)
        assert any("VERDICT_VOCAB_SECTION_MISSING" in v for v in violations), violations

    def test_detects_priority_duplicate_in_md_does_not_crash(self, tmp_path):
        """Duplicate priority row must NOT crash the check (was a strict-zip bug).

        Pre-hardening, a duplicate row in the priority table passed
        ``md_set == code_set`` (sets dedupe) but failed ``zip(strict=True)``
        because list lengths differed. The ValueError propagated up because
        non-DB checks have no try/except wrapper in the runner — taking down
        the whole drift gate.
        """
        import shutil

        src = self._canonical_md()
        dst = tmp_path / "RESEARCH_RULES.md"
        shutil.copy(src, dst)
        text = dst.read_text(encoding="utf-8")
        anchor = "| 1 | `NO-GO` |\n"
        assert anchor in text
        tampered = text.replace(anchor, anchor + "| 1 | `NO-GO` |\n", 1)
        dst.write_text(tampered, encoding="utf-8")

        violations = check_verdict_vocabulary_md_matches_code(md_path=dst)
        assert any("VERDICT_PRIORITY_DUPLICATE_IN_MD" in v and "NO-GO" in v for v in violations), violations

    def test_detects_unexpected_subsection(self, tmp_path):
        """Adding a third `### …` block inside the section triggers UNEXPECTED_SUBSECTION."""
        import shutil

        src = self._canonical_md()
        dst = tmp_path / "RESEARCH_RULES.md"
        shutil.copy(src, dst)
        text = dst.read_text(encoding="utf-8")
        injection = "\n### Surprise Examples\n\nSome content here.\n"
        marker = "### Priority Resolution Order"
        assert marker in text
        tampered = text.replace(marker, injection + "\n" + marker, 1)
        dst.write_text(tampered, encoding="utf-8")

        violations = check_verdict_vocabulary_md_matches_code(md_path=dst)
        assert any("VERDICT_VOCAB_UNEXPECTED_SUBSECTION" in v and "Surprise Examples" in v for v in violations), (
            violations
        )

    def test_order_check_runs_despite_missing_entries(self, tmp_path):
        """Order check must still report a swap even when a different token is missing.

        Regression guard for the pre-hardening behaviour where a single
        MISSING_IN_MD entry silently suppressed the entire ORDER_MISMATCH
        scan, forcing two fix cycles per divergence event.
        """
        import shutil

        src = self._canonical_md()
        dst = tmp_path / "RESEARCH_RULES.md"
        shutil.copy(src, dst)
        text = dst.read_text(encoding="utf-8")

        weak_row = "| `WEAK` | `WEAK` |\n"
        assert weak_row in text
        text = text.replace(weak_row, "", 1)

        row_1 = "| 1 | `NO-GO` |\n"
        row_2 = "| 2 | `NOGO` |\n"
        assert row_1 in text and row_2 in text
        text = text.replace(row_1, "<<TMP1>>").replace(row_2, "<<TMP2>>")
        text = text.replace("<<TMP1>>", "| 1 | `NOGO` |\n")
        text = text.replace("<<TMP2>>", "| 2 | `NO-GO` |\n")

        dst.write_text(text, encoding="utf-8")

        violations = check_verdict_vocabulary_md_matches_code(md_path=dst)
        assert any("VERDICT_VOCAB_MISSING_IN_MD" in v and "WEAK" in v for v in violations), violations
        assert any("VERDICT_PRIORITY_ORDER_MISMATCH" in v for v in violations), violations


class TestChecksListLabelsAreAscii:
    """All CHECKS labels must be pure ASCII to survive Windows cp1252 console.

    Origin: 2026-05-14. The drift runner crashes mid-loop with
    UnicodeEncodeError on the first non-ASCII label any time check
    ordering shifts. This invariant is now binding.
    """

    def test_live_repo_is_ascii_clean(self):
        """Live CHECKS list must be pure ASCII -- direct tripwire."""
        violations = check_checks_list_labels_are_ascii()
        assert violations == [], "Non-ASCII CHECKS labels in live repo:\n  " + "\n  ".join(violations)

    def test_detects_injected_non_ascii(self, monkeypatch):
        """Injecting a non-ASCII label into CHECKS triggers NON_ASCII_CHECK_LABEL."""
        from pipeline import check_drift as drift_mod

        original = list(drift_mod.CHECKS)
        # Synthetic offender with U+00D7 multiplication sign -- the exact
        # codepoint that crashed the runner before the 2026-05-14 cleanup.
        offender = ("Synthetic check with x glyph × and a tail", lambda: [], False, False)
        monkeypatch.setattr(drift_mod, "CHECKS", original + [offender])

        violations = check_checks_list_labels_are_ascii()
        assert any("NON_ASCII_CHECK_LABEL" in v and "0xd7" in v for v in violations), violations

    def test_handles_multiple_offending_codepoints(self, monkeypatch):
        """Multiple non-ASCII chars in one label all reported via hex codes."""
        from pipeline import check_drift as drift_mod

        original = list(drift_mod.CHECKS)
        offender = ("label with × and ↔ and —", lambda: [], False, False)
        monkeypatch.setattr(drift_mod, "CHECKS", original + [offender])

        violations = check_checks_list_labels_are_ascii()
        offender_violations = [v for v in violations if "NON_ASCII_CHECK_LABEL" in v and "label with" in v]
        assert len(offender_violations) == 1, offender_violations
        msg = offender_violations[0]
        assert "0xd7" in msg and "0x2194" in msg and "0x2014" in msg, msg

    def test_label_prints_under_cp1252_after_cleanup(self):
        """Sanity: every live label can encode under cp1252 (the underlying property)."""
        from pipeline.check_drift import CHECKS

        for label, *_ in CHECKS:
            label.encode("cp1252")  # raises UnicodeEncodeError if not encodable


class TestChordiaResultThresholdMatchesPrereg:
    """Paired-file parity: prereg `chordia_threshold_basis` <-> result `MEASURED threshold applied`.

    Origin: 2026-05-12 MGC LONDON_METALS Stage 1 run — prereg declared
    t>=3.79 strict, runner applied t>=3.00 due to theory_citation field-presence
    trap in the loader. Drift check landed in commit 9633fee6 without injection
    tests; this class closes the immune-system-immune-system gap.
    """

    @staticmethod
    def _seed_pair(
        root: Path,
        stem: str,
        *,
        prereg_threshold: str | None,
        result_threshold: str | None,
    ) -> None:
        """Write a minimal prereg + result MD pair under `root`."""
        hyp_dir = root / "docs" / "audit" / "hypotheses"
        res_dir = root / "docs" / "audit" / "results"
        hyp_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)
        if prereg_threshold is not None:
            (hyp_dir / f"{stem}.yaml").write_text(
                textwrap.dedent(
                    f"""\
                    hypotheses:
                      - id: test_fixture
                        chordia_threshold_basis: "Criterion 4 theory-backed threshold (t >= {prereg_threshold})"
                    """
                ),
                encoding="utf-8",
            )
        if result_threshold is not None:
            (res_dir / f"{stem}.md").write_text(
                f"# Test fixture result\n\n**MEASURED threshold applied:** `{result_threshold}`\n",
                encoding="utf-8",
            )

    def test_drift_clean_against_live_repo(self):
        """Live repo prereg/result pairs must be parity-clean — direct tripwire."""
        violations = check_chordia_result_threshold_matches_prereg()
        assert violations == [], "Chordia threshold drift in live repo:\n  " + "\n  ".join(violations)

    def test_detects_mismatch_on_binding_date(self, monkeypatch, tmp_path):
        """A pair dated >= 2026-05-12 with threshold divergence becomes a binding violation."""
        from pipeline import check_drift as cd

        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        self._seed_pair(
            tmp_path,
            "2026-05-12-fixture-binding",
            prereg_threshold="3.79",
            result_threshold="3.00",
        )
        violations = check_chordia_result_threshold_matches_prereg()
        assert len(violations) == 1, violations
        assert "2026-05-12-fixture-binding.yaml" in violations[0]
        assert "t>=3.79" in violations[0] and "t>=3.00" in violations[0]
        assert "theory_citation" in violations[0]

    def test_pre_sentinel_pair_is_advisory_not_violation(self, monkeypatch, tmp_path, capsys):
        """A pair dated before 2026-05-12 with the same mismatch routes to advisory, not violation."""
        from pipeline import check_drift as cd

        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        self._seed_pair(
            tmp_path,
            "2026-05-01-fixture-grandfathered",
            prereg_threshold="3.79",
            result_threshold="3.00",
        )
        violations = check_chordia_result_threshold_matches_prereg()
        assert violations == [], (
            "Pre-sentinel mismatch must NOT escalate to binding violation (was grandfathered): " + repr(violations)
        )
        captured = capsys.readouterr()
        assert "Chordia threshold mismatch advisory" in captured.out
        assert "2026-05-01-fixture-grandfathered.yaml" in captured.out

    def test_matching_threshold_passes(self, monkeypatch, tmp_path):
        """No mismatch on a binding-date pair with identical thresholds — no false positive."""
        from pipeline import check_drift as cd

        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        self._seed_pair(
            tmp_path,
            "2026-05-15-fixture-clean",
            prereg_threshold="3.00",
            result_threshold="3.00",
        )
        violations = check_chordia_result_threshold_matches_prereg()
        assert violations == [], violations

    def test_prereg_without_result_md_silent_skip(self, monkeypatch, tmp_path):
        """A prereg without a matching result MD is pre-run state — silent skip, no violation."""
        from pipeline import check_drift as cd

        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        self._seed_pair(
            tmp_path,
            "2026-05-15-fixture-pre-run",
            prereg_threshold="3.79",
            result_threshold=None,
        )
        violations = check_chordia_result_threshold_matches_prereg()
        assert violations == [], violations

    def test_non_chordia_prereg_silent_skip(self, monkeypatch, tmp_path):
        """A prereg without chordia_threshold_basis is out-of-scope — no false positive."""
        from pipeline import check_drift as cd

        monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
        hyp_dir = tmp_path / "docs" / "audit" / "hypotheses"
        res_dir = tmp_path / "docs" / "audit" / "results"
        hyp_dir.mkdir(parents=True)
        res_dir.mkdir(parents=True)
        (hyp_dir / "2026-05-15-fixture-non-chordia.yaml").write_text(
            "hypotheses:\n  - id: not_a_chordia_run\n    notes: nothing to compare\n",
            encoding="utf-8",
        )
        (res_dir / "2026-05-15-fixture-non-chordia.md").write_text(
            "# Result with no measured threshold line\n",
            encoding="utf-8",
        )
        violations = check_chordia_result_threshold_matches_prereg()
        assert violations == [], violations


class TestCheckDashboardLocalhostOnlyBinding:
    """check_dashboard_localhost_only_binding — pass, fail, and mutation proofs."""

    _GOOD = textwrap.dedent("""\
        def run_dashboard(host: str = "127.0.0.1", port: int = 8765):
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError("Refusing to start dashboard on non-localhost host")
            pass

        parser.add_argument("--host", default="127.0.0.1")
    """)

    def _check(self, tmp_path: Path, content: str) -> list[str]:
        from pipeline.check_drift import check_dashboard_localhost_only_binding

        dash = tmp_path / "live" / "bot_dashboard.py"
        dash.parent.mkdir(parents=True, exist_ok=True)
        dash.write_text(content, encoding="utf-8")
        return check_dashboard_localhost_only_binding(tmp_path)

    def test_clean_file_passes(self, tmp_path):
        assert self._check(tmp_path, self._GOOD) == []

    def test_live_repo_passes(self):
        """Direct tripwire against the real bot_dashboard.py."""
        from pipeline.check_drift import check_dashboard_localhost_only_binding
        from pipeline.paths import PROJECT_ROOT

        violations = check_dashboard_localhost_only_binding(PROJECT_ROOT / "trading_app")
        assert violations == [], "\n".join(violations)

    def test_detects_non_loopback_signature_default(self, tmp_path):
        bad = self._GOOD.replace('host: str = "127.0.0.1"', 'host: str = "0.0.0.0"')
        violations = self._check(tmp_path, bad)
        assert any("0.0.0.0" in v for v in violations), violations

    def test_detects_missing_signature(self, tmp_path):
        bad = self._GOOD.replace('def run_dashboard(host: str = "127.0.0.1"', "def run_dashboard(")
        violations = self._check(tmp_path, bad)
        assert any("signature" in v for v in violations), violations

    def test_detects_absent_argparse_host(self, tmp_path):
        """Removing --host argparse line entirely must fire a violation (T3-W1 fix)."""
        bad = "\n".join(line for line in self._GOOD.splitlines() if "--host" not in line)
        violations = self._check(tmp_path, bad)
        assert any("argparse" in v for v in violations), violations

    def test_detects_non_loopback_argparse_default(self, tmp_path):
        bad = self._GOOD.replace('default="127.0.0.1"', 'default="0.0.0.0"')
        violations = self._check(tmp_path, bad)
        assert any("0.0.0.0" in v for v in violations), violations

    def test_detects_missing_runtime_error_guard(self, tmp_path):
        bad = self._GOOD.replace("Refusing to start dashboard on non-localhost host", "nope")
        violations = self._check(tmp_path, bad)
        assert any("RuntimeError" in v for v in violations), violations

    def test_missing_file_returns_clean(self, tmp_path):
        from pipeline.check_drift import check_dashboard_localhost_only_binding

        violations = check_dashboard_localhost_only_binding(tmp_path)
        assert violations == []


class TestCheckDashboardSseSingleWorker:
    """check_dashboard_sse_single_worker — pass, fail, and mutation proofs."""

    _GOOD = textwrap.dedent("""\
        import uvicorn
        uvicorn.run(app, host=host, port=port, workers=1)
    """)

    def _check(self, tmp_path: Path, content: str) -> list[str]:
        from pipeline.check_drift import check_dashboard_sse_single_worker

        dash = tmp_path / "live" / "bot_dashboard.py"
        dash.parent.mkdir(parents=True, exist_ok=True)
        dash.write_text(content, encoding="utf-8")
        return check_dashboard_sse_single_worker(tmp_path)

    def test_clean_file_passes(self, tmp_path):
        assert self._check(tmp_path, self._GOOD) == []

    def test_live_repo_passes(self):
        """Direct tripwire against the real bot_dashboard.py."""
        from pipeline.check_drift import check_dashboard_sse_single_worker
        from pipeline.paths import PROJECT_ROOT

        violations = check_dashboard_sse_single_worker(PROJECT_ROOT / "trading_app")
        assert violations == [], "\n".join(violations)

    def test_detects_workers_greater_than_one(self, tmp_path):
        bad = self._GOOD.replace("workers=1", "workers=4")
        violations = self._check(tmp_path, bad)
        assert any("workers=4" in v for v in violations), violations

    def test_detects_missing_workers_pin(self, tmp_path):
        bad = "uvicorn.run(app, host=host, port=port)\n"
        violations = self._check(tmp_path, bad)
        assert any("missing explicit workers=1" in v for v in violations), violations

    def test_missing_file_returns_clean(self, tmp_path):
        from pipeline.check_drift import check_dashboard_sse_single_worker

        violations = check_dashboard_sse_single_worker(tmp_path)
        assert violations == []
