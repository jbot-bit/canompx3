"""Tests for trading_app.ai.corpus."""

import pytest
from pathlib import Path

from trading_app.ai.corpus import (
    load_corpus,
    get_corpus_file_paths,
    CORPUS_FILES,
    PROJECT_ROOT,
)


class TestCorpusFiles:
    """Verify all canonical files exist on disk."""

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
        assert "8.40" in corpus["COST_MODEL"] or "total_friction" in corpus["COST_MODEL"]


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
