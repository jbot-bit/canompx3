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
    check_daily_features_row_integrity,
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
