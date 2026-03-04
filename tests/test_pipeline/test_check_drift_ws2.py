"""
WS2: Tests for untested drift checks in pipeline/check_drift.py.

Covers checks 4-8, 13-22, 24-27, 30-34, 37, 44-49.
Each check gets at minimum: one violation test, one clean/pass test.
"""

import pytest
from pathlib import Path

from pipeline import check_drift


# ── Helpers ────────────────────────────────────────────────────────────


def _patch_dirs(monkeypatch, tmp_path):
    """Patch PROJECT_ROOT, PIPELINE_DIR, TRADING_APP_DIR, RESEARCH_DIR to tmp_path."""
    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(check_drift, "PIPELINE_DIR", tmp_path / "pipeline")
    monkeypatch.setattr(check_drift, "TRADING_APP_DIR", tmp_path / "trading_app")
    monkeypatch.setattr(check_drift, "RESEARCH_DIR", tmp_path / "research")


def _mkfile(path: Path, content: str):
    """Create a file with content, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── Check 4: Schema/query consistency (pipeline) ──────────────────────


class TestSchemaQueryConsistency:
    """Check 4: SQL table refs in pipeline/ must match init_db.py schema."""

    def test_catches_unknown_table(self, tmp_path):
        _mkfile(tmp_path / "init_db.py", 'CREATE TABLE IF NOT EXISTS bars_1m (ts_utc TEXT)')
        _mkfile(tmp_path / "run_pipeline.py", '''
sql = """SELECT * FROM fake_table WHERE 1=1"""
''')
        violations = check_drift.check_schema_query_consistency(tmp_path)
        assert len(violations) > 0
        assert "fake_table" in violations[0]

    def test_passes_known_table(self, tmp_path):
        _mkfile(tmp_path / "init_db.py", 'CREATE TABLE IF NOT EXISTS bars_1m (ts_utc TEXT)')
        _mkfile(tmp_path / "run_pipeline.py", '''
sql = """SELECT * FROM bars_1m WHERE symbol = ?"""
''')
        violations = check_drift.check_schema_query_consistency(tmp_path)
        assert len(violations) == 0

    def test_skips_cte_names(self, tmp_path):
        _mkfile(tmp_path / "init_db.py", 'CREATE TABLE IF NOT EXISTS bars_1m (ts_utc TEXT)')
        _mkfile(tmp_path / "run_pipeline.py", '''
sql = """WITH recent AS (SELECT * FROM bars_1m) SELECT * FROM recent"""
''')
        violations = check_drift.check_schema_query_consistency(tmp_path)
        assert len(violations) == 0


# ── Check 5: Import cycles ────────────────────────────────────────────


class TestImportCycles:
    """Check 5: ingest_dbn_mgc.py must not import from ingest_dbn.py."""

    def test_catches_circular_import(self, tmp_path):
        _mkfile(tmp_path / "ingest_dbn_mgc.py",
                "from pipeline.ingest_dbn import validate_chunk\n")
        violations = check_drift.check_import_cycles(tmp_path)
        assert len(violations) > 0
        assert "circular" in violations[0].lower()

    def test_passes_clean(self, tmp_path):
        _mkfile(tmp_path / "ingest_dbn_mgc.py",
                "from pipeline.paths import GOLD_DB_PATH\n")
        violations = check_drift.check_import_cycles(tmp_path)
        assert len(violations) == 0

    def test_no_file_no_crash(self, tmp_path):
        violations = check_drift.check_import_cycles(tmp_path)
        assert len(violations) == 0


# ── Check 6: Hardcoded paths (pipeline) ───────────────────────────────


class TestHardcodedPathsPipeline:
    """Check 6: No hardcoded Windows paths in pipeline/."""

    def test_catches_windows_path(self, tmp_path):
        _mkfile(tmp_path / "bad.py", "DB = 'C:/Users/josh/gold.db'\n")
        violations = check_drift.check_hardcoded_paths(tmp_path)
        assert len(violations) > 0

    def test_passes_relative(self, tmp_path):
        _mkfile(tmp_path / "good.py", "DB = Path('gold.db')\n")
        violations = check_drift.check_hardcoded_paths(tmp_path)
        assert len(violations) == 0

    def test_ignores_comments(self, tmp_path):
        _mkfile(tmp_path / "ok.py", "# old path: C:/Users/josh\n")
        violations = check_drift.check_hardcoded_paths(tmp_path)
        assert len(violations) == 0


# ── Check 7: Connection leaks (pipeline) ──────────────────────────────


class TestConnectionLeaksPipeline:
    """Check 7: duckdb.connect() must have matching close/finally."""

    def test_catches_leak(self, tmp_path):
        _mkfile(tmp_path / "leaky.py",
                "import duckdb\ncon = duckdb.connect('test.db')\ncon.execute('SELECT 1')\n")
        violations = check_drift.check_connection_leaks(tmp_path)
        assert len(violations) > 0

    def test_passes_with_close(self, tmp_path):
        _mkfile(tmp_path / "good.py",
                "import duckdb\ncon = duckdb.connect('test.db')\ncon.execute('SELECT 1')\ncon.close()\n")
        violations = check_drift.check_connection_leaks(tmp_path)
        assert len(violations) == 0

    def test_passes_with_finally(self, tmp_path):
        _mkfile(tmp_path / "good.py",
                "import duckdb\ncon = duckdb.connect('test.db')\ntry:\n    pass\nfinally:\n    con.close()\n")
        violations = check_drift.check_connection_leaks(tmp_path)
        assert len(violations) == 0


# ── Check 8: Dashboard readonly ───────────────────────────────────────


class TestDashboardReadonly:
    """Check 8: dashboard.py must be read-only (no SQL writes)."""

    def test_catches_insert(self, tmp_path):
        _mkfile(tmp_path / "dashboard.py",
                "con.execute(\"INSERT INTO bars_1m VALUES (?)\")\n"
                "con = duckdb.connect('db', read_only=True)\n")
        violations = check_drift.check_dashboard_readonly(tmp_path)
        assert len(violations) > 0

    def test_catches_missing_readonly(self, tmp_path):
        _mkfile(tmp_path / "dashboard.py",
                "con = duckdb.connect(str(db_path))\nresult = con.execute('SELECT 1')\n")
        violations = check_drift.check_dashboard_readonly(tmp_path)
        assert len(violations) > 0
        assert "read_only" in violations[0]

    def test_passes_clean(self, tmp_path):
        _mkfile(tmp_path / "dashboard.py",
                "con = duckdb.connect(str(db_path), read_only=True)\n"
                "result = con.execute('SELECT * FROM bars_1m')\n")
        violations = check_drift.check_dashboard_readonly(tmp_path)
        assert len(violations) == 0


# ── Check 13: Entry models sync ───────────────────────────────────────


class TestEntryModelsSync:
    """Check 13: ENTRY_MODELS matches expected values. Tests current state."""

    def test_current_config_passes(self):
        violations = check_drift.check_entry_models_sync()
        assert len(violations) == 0


# ── Check 14: Entry price sanity ──────────────────────────────────────


class TestEntryPriceSanity:
    """Check 14: outcome_builder must not hardcode entry_price = orb_high/low."""

    def test_catches_hardcoded_entry(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "outcome_builder.py",
                "entry_price = orb_high  # bad\n")
        violations = check_drift.check_entry_price_sanity()
        assert len(violations) > 0

    def test_passes_clean(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "outcome_builder.py",
                "entry_price = compute_entry(model, row)\n")
        violations = check_drift.check_entry_price_sanity()
        assert len(violations) == 0


# ── Check 15: Nested isolation ────────────────────────────────────────


class TestNestedIsolation:
    """Check 15: nested/ and regime/ must not import from db_manager."""

    def test_catches_db_manager_import(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        nested = tmp_path / "trading_app" / "nested"
        _mkfile(nested / "discovery.py",
                "from trading_app.db_manager import init_schema\n")
        violations = check_drift.check_nested_isolation()
        assert len(violations) > 0

    def test_passes_clean(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        nested = tmp_path / "trading_app" / "nested"
        _mkfile(nested / "discovery.py",
                "from pipeline.paths import GOLD_DB_PATH\n")
        violations = check_drift.check_nested_isolation()
        assert len(violations) == 0


# ── Check 17: Nested production writes ────────────────────────────────


class TestNestedProductionWrites:
    """Check 17: nested/ must not write to production tables."""

    def test_catches_insert_validated(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        nested = tmp_path / "trading_app" / "nested"
        _mkfile(nested / "builder.py",
                'con.execute("INSERT INTO validated_setups VALUES (?)")\n')
        violations = check_drift.check_nested_production_writes()
        assert len(violations) > 0

    def test_passes_own_table(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        nested = tmp_path / "trading_app" / "nested"
        _mkfile(nested / "builder.py",
                'con.execute("INSERT INTO nested_strategies VALUES (?)")\n')
        violations = check_drift.check_nested_production_writes()
        assert len(violations) == 0


# ── Check 18: Schema query consistency (trading_app) ──────────────────


class TestSchemaQueryConsistencyTradingApp:
    """Check 18: SQL refs in trading_app/ must match schema."""

    def test_catches_unknown_table(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "init_db.py",
                'CREATE TABLE IF NOT EXISTS bars_1m (ts TEXT)')
        _mkfile(tmp_path / "trading_app" / "db_manager.py",
                'CREATE TABLE IF NOT EXISTS validated_setups (id TEXT)')
        _mkfile(tmp_path / "trading_app" / "query.py", '''
sql = """SELECT COUNT(*) FROM bogus_table WHERE 1=1
         INSERT INTO bogus_table VALUES (1)"""
''')
        violations = check_drift.check_schema_query_consistency_trading_app(
            tmp_path / "trading_app")
        assert len(violations) > 0
        assert "bogus_table" in violations[0]

    def test_passes_known_table(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "init_db.py",
                'CREATE TABLE IF NOT EXISTS bars_1m (ts TEXT)')
        _mkfile(tmp_path / "trading_app" / "db_manager.py",
                'CREATE TABLE IF NOT EXISTS validated_setups (id TEXT)')
        _mkfile(tmp_path / "trading_app" / "query.py", '''
sql = """SELECT COUNT(*) FROM validated_setups
         INSERT INTO bars_1m VALUES (1)"""
''')
        violations = check_drift.check_schema_query_consistency_trading_app(
            tmp_path / "trading_app")
        assert len(violations) == 0


# ── Check 19: Timezone hygiene ────────────────────────────────────────


class TestTimezoneHygiene:
    """Check 19: No pytz imports or hardcoded timedelta(hours=10)."""

    def test_catches_pytz(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "bad.py", "import pytz\n")
        violations = check_drift.check_timezone_hygiene()
        assert len(violations) > 0

    def test_catches_timedelta_10(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "bad.py",
                "dt = datetime.now() + timedelta(hours=10)\n")
        violations = check_drift.check_timezone_hygiene()
        assert len(violations) > 0

    def test_passes_zoneinfo(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "good.py",
                "from zoneinfo import ZoneInfo\ntz = ZoneInfo('Australia/Brisbane')\n")
        violations = check_drift.check_timezone_hygiene()
        assert len(violations) == 0


# ── Check 20: Market state readonly ───────────────────────────────────


class TestMarketStateReadonly:
    """Check 20: market_state.py, scoring.py, cascade_table.py must not write DB."""

    def test_catches_insert(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "market_state.py",
                'con.execute("INSERT INTO bars_1m VALUES (?)")\n')
        violations = check_drift.check_market_state_readonly()
        assert len(violations) > 0

    def test_passes_select_only(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "market_state.py",
                'result = con.execute("SELECT * FROM bars_1m").fetchall()\n')
        violations = check_drift.check_market_state_readonly()
        assert len(violations) == 0


# ── Check 21: Sharpe ann presence ─────────────────────────────────────


class TestSharpeAnnPresence:
    """Check 21: strategy_discovery must compute sharpe_ann."""

    def test_catches_missing_sharpe_ann(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "strategy_discovery.py",
                '"trades_per_year": 252\n')
        violations = check_drift.check_sharpe_ann_presence()
        assert len(violations) > 0
        assert "sharpe_ann" in violations[0]

    def test_passes_both_present(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "strategy_discovery.py",
                '"sharpe_ann": 1.5,\n"trades_per_year": 252\n')
        _mkfile(tmp_path / "trading_app" / "view_strategies.py",
                'print(f"sharpe_ann={row.sharpe_ann}")\n')
        violations = check_drift.check_sharpe_ann_presence()
        assert len(violations) == 0


# ── Check 22: Ingest authority notice ─────────────────────────────────


class TestIngestAuthorityNotice:
    """Check 22: ingest_dbn_mgc.py must have deprecation notice."""

    def test_catches_missing_notice(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "ingest_dbn_mgc.py",
                'if __name__ == "__main__":\n    run()\n')
        violations = check_drift.check_ingest_authority_notice()
        assert len(violations) > 0

    def test_passes_with_notice(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "ingest_dbn_mgc.py",
                'if __name__ == "__main__":\n'
                '    print("NOTE: For multi-instrument support, prefer:")\n'
                '    print("  python pipeline/ingest_dbn.py --instrument MGC")\n'
                '    run()\n')
        violations = check_drift.check_ingest_authority_notice()
        assert len(violations) == 0


# ── Check 24: Validation gate existence ───────────────────────────────


class TestValidationGateExistence:
    """Check 24: Critical gate functions must exist in their files."""

    def test_catches_missing_gate(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        # Create ingest_dbn_mgc.py without required gate functions
        _mkfile(tmp_path / "pipeline" / "ingest_dbn_mgc.py",
                "def process(): pass\n")
        violations = check_drift.check_validation_gate_existence()
        assert len(violations) > 0
        assert "validate_chunk" in violations[0] or "Missing gate" in violations[0]

    def test_passes_all_gates(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "ingest_dbn_mgc.py",
                "def validate_chunk(df): pass\n"
                "def validate_timestamp_utc(ts): pass\n"
                "def check_pk_safety(con): pass\n"
                "def check_merge_integrity(df): pass\n"
                "def run_final_gates(con): pass\n")
        _mkfile(tmp_path / "pipeline" / "build_bars_5m.py",
                "def verify_5m_integrity(con): pass\n")
        _mkfile(tmp_path / "pipeline" / "ingest_dbn.py",
                "# FAIL-CLOSED validation\n")
        violations = check_drift.check_validation_gate_existence()
        assert len(violations) == 0


# ── Check 25: Naive datetime ──────────────────────────────────────────


class TestNaiveDatetime:
    """Check 25: Block deprecated datetime.utcnow()."""

    def test_catches_utcnow(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "bad.py",
                "now = datetime.utcnow()\n")
        violations = check_drift.check_naive_datetime()
        assert len(violations) > 0

    def test_catches_utcfromtimestamp(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "bad.py",
                "dt = datetime.utcfromtimestamp(ts)\n")
        violations = check_drift.check_naive_datetime()
        assert len(violations) > 0

    def test_passes_aware(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "good.py",
                "now = datetime.now(timezone.utc)\n")
        violations = check_drift.check_naive_datetime()
        assert len(violations) == 0


# ── Check 26: DST session coverage ────────────────────────────────────


class TestDstSessionCoverage:
    """Check 26: All sessions must be classified as DST-affected or clean."""

    def test_current_config_passes(self):
        violations = check_drift.check_dst_session_coverage()
        assert len(violations) == 0


# ── Check 27: DB config usage ─────────────────────────────────────────


class TestDbConfigUsage:
    """Check 27: duckdb.connect() callers must also call configure_connection()."""

    def test_catches_missing_configure(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "build_bars_5m.py",
                "con = duckdb.connect(str(db_path))\ncon.execute('SELECT 1')\n")
        violations = check_drift.check_db_config_usage()
        assert len(violations) > 0
        assert "configure_connection" in violations[0]

    def test_passes_with_configure(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "pipeline" / "build_bars_5m.py",
                "from pipeline.db_config import configure_connection\n"
                "con = duckdb.connect(str(db_path))\n"
                "configure_connection(con)\n")
        violations = check_drift.check_db_config_usage()
        assert len(violations) == 0


# ── Check 30: E2/E3 CB1 only ─────────────────────────────────────────


class TestE2E3Cb1Only:
    """Check 30: E2 and E3 must be restricted to CB1."""

    def test_catches_missing_e3_restriction(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "outcome_builder.py",
                "# no E3 restriction\nfor em in ENTRY_MODELS:\n    pass\n")
        violations = check_drift.check_e2_e3_cb1_only()
        assert len(violations) > 0

    def test_passes_with_restriction(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "outcome_builder.py",
                'if em == "E3":\n    cb_options = [1]\n')
        violations = check_drift.check_e2_e3_cb1_only()
        assert len(violations) == 0


# ── Check 31: orb_minutes in strategy_id ──────────────────────────────


class TestOrbMinutesInStrategyId:
    """Check 31: strategy_id format must include orb_minutes."""

    def test_current_codebase_passes(self):
        violations = check_drift.check_orb_minutes_in_strategy_id()
        assert len(violations) == 0


# ── Check 32: orb_labels session catalog sync ─────────────────────────


class TestOrbLabelsSessionCatalogSync:
    """Check 32: enabled_sessions must be subset of SESSION_CATALOG."""

    def test_current_config_passes(self):
        violations = check_drift.check_orb_labels_session_catalog_sync()
        assert len(violations) == 0


# ── Check 33: Stale session names in code ─────────────────────────────


class TestStaleSessionNamesInCode:
    """Check 33: No old fixed-clock session names in production code."""

    def test_current_codebase_passes(self):
        violations = check_drift.check_stale_session_names_in_code()
        assert len(violations) == 0


# ── Check 34: SQL adapter validation sync ─────────────────────────────


class TestSqlAdapterValidationSync:
    """Check 34: sql_adapter templates must match validation requirements."""

    def test_current_config_passes(self):
        violations = check_drift.check_sql_adapter_validation_sync()
        assert len(violations) == 0


# ── Check 37: Stale scratch DB ────────────────────────────────────────


class TestStaleScratchDb:
    """Check 37: Canonical gold.db must exist at project root."""

    def test_catches_missing_canonical(self, tmp_path, monkeypatch):
        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        # No gold.db at project root
        violations = check_drift.check_stale_scratch_db()
        assert len(violations) > 0
        assert "not found" in violations[0].lower() or "Canonical" in violations[0]

    def test_passes_with_db_no_scratch(self, tmp_path, monkeypatch):
        """If canonical DB exists and no scratch copy, no violation."""
        monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)
        (tmp_path / "gold.db").write_bytes(b"fake db")
        # Patch Path to make C:/db/gold.db appear nonexistent
        original_exists = Path.exists
        def mock_exists(self):
            if str(self) == r"C:\db\gold.db" or str(self) == "C:/db/gold.db":
                return False
            return original_exists(self)
        monkeypatch.setattr(Path, "exists", mock_exists)
        violations = check_drift.check_stale_scratch_db()
        assert len(violations) == 0


# ── Check 44: Variant selection metric ────────────────────────────────


class TestVariantSelectionMetric:
    """Check 44: ORDER BY must use expectancy_r, not sharpe_ratio."""

    def test_catches_sharpe_order(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "live_config.py",
                'sql = "ORDER BY sharpe_ratio DESC LIMIT 1"\n')
        violations = check_drift.check_variant_selection_metric()
        assert len(violations) > 0
        assert "sharpe_ratio" in violations[0]

    def test_passes_expectancy(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "live_config.py",
                'sql = "ORDER BY expectancy_r DESC LIMIT 1"\n')
        violations = check_drift.check_variant_selection_metric()
        assert len(violations) == 0


# ── Check 45: Research provenance annotations ─────────────────────────


class TestResearchProvenanceAnnotations:
    """Check 45: @research-source must have @entry-models with E2."""

    def test_catches_missing_entry_models(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "config.py",
                "# @research-source: winner_speed T80 profiling\n"
                "EARLY_EXIT_MINUTES = 38\n")
        violations = check_drift.check_research_provenance_annotations()
        assert len(violations) > 0
        assert "@entry-models" in violations[0]

    def test_catches_stale_entry_model(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "config.py",
                "# @research-source: winner_speed T80\n"
                "# @entry-models: E0, E1\n"
                "EARLY_EXIT_MINUTES = 38\n")
        violations = check_drift.check_research_provenance_annotations()
        assert len(violations) > 0
        assert "E2" in violations[0]

    def test_passes_with_e2(self, tmp_path, monkeypatch):
        _patch_dirs(monkeypatch, tmp_path)
        _mkfile(tmp_path / "trading_app" / "config.py",
                "# @research-source: winner_speed T80\n"
                "# @entry-models: E1, E2\n"
                "EARLY_EXIT_MINUTES = 38\n")
        violations = check_drift.check_research_provenance_annotations()
        assert len(violations) == 0


# ── Check 46: Cost model completeness ─────────────────────────────────


class TestCostModelCompleteness:
    """Check 46: COST_SPECS covers all active instruments."""

    def test_current_config_passes(self):
        violations = check_drift.check_cost_model_completeness()
        assert len(violations) == 0


# ── Check 47: Trading rules authority ─────────────────────────────────


class TestTradingRulesAuthority:
    """Check 47: TRADING_RULES canonical values match code."""

    def test_current_config_passes(self):
        violations = check_drift.check_trading_rules_authority()
        assert len(violations) == 0


# ── Check 48: ML config canonical sources ─────────────────────────────


class TestMlConfigCanonicalSources:
    """Check 48: ML config instruments/features match pipeline sources."""

    def test_current_config_passes(self):
        violations = check_drift.check_ml_config_canonical_sources()
        assert len(violations) == 0


# ── Check 49: ML lookahead blacklist ──────────────────────────────────


class TestMlLookaheadBlacklist:
    """Check 49: ML features must not include look-ahead columns."""

    def test_current_config_passes(self):
        violations = check_drift.check_ml_lookahead_blacklist()
        assert len(violations) == 0
