"""
Tests for pipeline.check_drift drift detection rules.

Tests each drift check catches violations and passes clean code.
"""

import pytest
import tempfile
from pathlib import Path

from pipeline.check_drift import (
    check_hardcoded_mgc_sql,
    check_apply_iterrows,
    check_non_bars1m_writes,
    check_pipeline_never_imports_trading_app,
    check_trading_app_connection_leaks,
    check_trading_app_hardcoded_paths,
    check_config_filter_sync,
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
        f.write_text("con.execute(\"SELECT * FROM bars_1m WHERE symbol = ?\")\n")
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
        f.write_text("con.execute(\"INSERT INTO bars_5m (ts_utc) VALUES (?)\")\n")
        violations = check_non_bars1m_writes([f])
        assert len(violations) > 0

    def test_catches_delete_from_other_table(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("con.execute(\"DELETE FROM daily_features WHERE date = ?\")\n")
        violations = check_non_bars1m_writes([f])
        assert len(violations) > 0

    def test_allows_bars_1m_writes(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("con.execute(\"INSERT OR REPLACE INTO bars_1m (ts_utc) VALUES (?)\")\n")
        violations = check_non_bars1m_writes([f])
        assert len(violations) == 0

    def test_passes_clean_code(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("count = con.execute(\"SELECT COUNT(*) FROM bars_1m\").fetchone()[0]\n")
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
        (tmp_path / "strategy_discovery.py").write_text(
            "for filter_key, filt in ALL_FILTERS.items():\n    pass\n"
        )
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 1
        assert "iterates ALL_FILTERS" in violations[0]

    def test_catches_all_filters_values(self, tmp_path, monkeypatch):
        """ALL_FILTERS.values() in a discovery file triggers a violation."""
        from pipeline import check_drift
        (tmp_path / "strategy_discovery.py").write_text(
            "for filt in ALL_FILTERS.values():\n    pass\n"
        )
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 1
        assert "iterates ALL_FILTERS" in violations[0]

    def test_catches_len_all_filters(self, tmp_path, monkeypatch):
        """len(ALL_FILTERS) in a discovery file triggers a violation."""
        from pipeline import check_drift
        (tmp_path / "strategy_discovery.py").write_text(
            "total_combos = len(ALL_FILTERS) * len(RR_TARGETS)\n"
        )
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 1
        assert "len(ALL_FILTERS)" in violations[0]

    def test_passes_all_filters_get(self, tmp_path, monkeypatch):
        """ALL_FILTERS.get() is a registry lookup — not a grid iteration."""
        from pipeline import check_drift
        (tmp_path / "strategy_discovery.py").write_text(
            "filt = ALL_FILTERS.get(strategy.filter_type)\n"
        )
        self._patch(monkeypatch, check_drift, tmp_path)
        violations = check_drift.check_discovery_session_aware_filters()
        assert len(violations) == 0

    def test_passes_commented_out(self, tmp_path, monkeypatch):
        """Commented-out ALL_FILTERS.items() is not a violation."""
        from pipeline import check_drift
        (tmp_path / "strategy_discovery.py").write_text(
            "# for k, v in ALL_FILTERS.items():  # old pattern\n"
        )
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
        (nested / "discovery.py").write_text(
            "for k, v in ALL_FILTERS.items():\n    pass\n"
        )
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
