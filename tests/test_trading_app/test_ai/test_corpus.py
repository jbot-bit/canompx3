"""Tests for trading_app.ai.corpus."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_app.ai.corpus import (
    CORPUS_FILES,
    PROJECT_ROOT,
    get_corpus_file_paths,
    load_corpus,
)

# Stage 2 of claude-api-modernization: corpus expanded from 4 to 8 entries to
# include institutional-rigor / research-discipline documents alongside the
# trading-rule docs.
EXPECTED_CORPUS_KEYS = {
    "TRADING_RULES",
    "TRADE_MANAGEMENT_RULES",
    "CONFIG",
    "COST_MODEL",
    "RESEARCH_RULES",
    "CLAUDE_MD",
    "PRE_REGISTERED_CRITERIA",
    "MECHANISM_PRIORS",
}


class TestCorpusFiles:
    """Verify all canonical files exist on disk."""

    def test_corpus_keys_match_expected(self):
        """CORPUS_FILES must carry exactly the 8 expected entries."""
        assert set(CORPUS_FILES.keys()) == EXPECTED_CORPUS_KEYS

    def test_all_corpus_files_exist(self):
        for name, info in CORPUS_FILES.items():
            fpath = PROJECT_ROOT / info["path"]
            assert fpath.exists(), f"Corpus file missing: {info['path']} ({name})"

    def test_corpus_file_paths_returns_all(self):
        paths = get_corpus_file_paths()
        assert len(paths) == len(CORPUS_FILES)
        for p in paths:
            assert isinstance(p, str)


class TestLoadCorpus:
    """Test corpus loading."""

    def test_missing_file_logs_warning(self, tmp_path, caplog):
        """load_corpus() must emit a WARNING when a corpus file is missing."""
        fake_files = {
            "TEST_DOC": {
                "path": "nonexistent_file_ralph_test.md",
                "priority": "HIGH",
                "description": "test",
            }
        }
        with patch("trading_app.ai.corpus.CORPUS_FILES", fake_files):
            with caplog.at_level(logging.WARNING, logger="trading_app.ai.corpus"):
                result = load_corpus()
        assert result["TEST_DOC"].startswith("[MISSING:")
        assert any("nonexistent_file_ralph_test.md" in r.message for r in caplog.records)

    def test_critical_missing_raises(self):
        """load_corpus() MUST fail-closed when a CRITICAL file is missing."""
        fake_files = {
            "CRITICAL_DOC": {
                "path": "nonexistent_critical_doc.md",
                "priority": "CRITICAL",
                "description": "test",
            },
            "HIGH_DOC": {
                "path": "nonexistent_high_doc.md",
                "priority": "HIGH",
                "description": "test",
            },
        }
        with patch("trading_app.ai.corpus.CORPUS_FILES", fake_files):
            with pytest.raises(RuntimeError, match="CRITICAL corpus files missing"):
                load_corpus()

    def test_load_corpus_returns_dict(self):
        corpus = load_corpus()
        assert isinstance(corpus, dict)
        assert len(corpus) == len(CORPUS_FILES)

    def test_all_entries_have_content(self):
        corpus = load_corpus()
        for name, content in corpus.items():
            assert not content.startswith("[MISSING:"), f"{name} is missing"
            assert len(content) > 100, f"{name} has suspiciously little content"

    def test_config_contains_entry_models(self):
        corpus = load_corpus()
        assert "ENTRY_MODELS" in corpus["CONFIG"]

    def test_cost_model_contains_friction(self):
        corpus = load_corpus()
        assert "total_friction" in corpus["COST_MODEL"]


class TestSchemaDefinitions:
    """Test schema extraction (requires gold.db)."""

    @pytest.fixture
    def db_path(self):
        from pipeline.paths import GOLD_DB_PATH

        if not GOLD_DB_PATH.exists():
            pytest.skip("gold.db not available")
        return str(GOLD_DB_PATH)

    def test_get_schema_definitions(self, db_path):
        from trading_app.ai.corpus import get_schema_definitions

        schema = get_schema_definitions(db_path)
        assert "bars_1m" in schema
        assert "ts_utc" in schema

    def test_get_db_stats(self, db_path):
        from trading_app.ai.corpus import get_db_stats

        stats = get_db_stats(db_path)
        assert "bars_1m" in stats
        assert "rows" in stats
