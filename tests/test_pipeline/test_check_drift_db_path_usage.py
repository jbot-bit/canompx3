"""Tests for the active-code canonical DB path drift guard."""

from pipeline.check_drift import check_active_code_uses_canonical_db_path


class TestActiveCodeUsesCanonicalDbPath:
    """Active research scripts must delegate DB resolution to pipeline.paths."""

    def _patch_scan_dirs(self, monkeypatch, tmp_path, research_dir):
        monkeypatch.setattr("pipeline.check_drift.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("pipeline.check_drift.RESEARCH_DIR", research_dir)
        monkeypatch.setattr("pipeline.check_drift.SCRIPTS_DIR", tmp_path / "scripts")
        monkeypatch.setattr(
            "pipeline.check_drift.ACTIVE_DB_PATH_SCAN_DIRS",
            [research_dir, tmp_path / "scripts" / "research"],
        )

    def test_catches_repo_root_gold_db_join(self, tmp_path, monkeypatch):
        research_dir = tmp_path / "research"
        research_dir.mkdir()
        f = research_dir / "bad.py"
        f.write_text("PROJECT_ROOT = Path(__file__).resolve().parent.parent\nDB_PATH = PROJECT_ROOT / 'gold.db'\n")
        self._patch_scan_dirs(monkeypatch, tmp_path, research_dir)
        violations = check_active_code_uses_canonical_db_path()
        assert len(violations) == 1
        assert "delegate to pipeline.paths.GOLD_DB_PATH" in violations[0]

    def test_catches_manual_duckdb_path_fallback(self, tmp_path, monkeypatch):
        research_dir = tmp_path / "research"
        research_dir.mkdir()
        f = research_dir / "bad.py"
        f.write_text('DB_PATH = os.environ.get("DUCKDB_PATH", str(PROJECT_ROOT / "gold.db"))\n')
        self._patch_scan_dirs(monkeypatch, tmp_path, research_dir)
        violations = check_active_code_uses_canonical_db_path()
        assert len(violations) == 1

    def test_passes_pipeline_paths_usage(self, tmp_path, monkeypatch):
        research_dir = tmp_path / "research"
        research_dir.mkdir()
        f = research_dir / "good.py"
        f.write_text("from pipeline.paths import GOLD_DB_PATH\nDB_PATH = GOLD_DB_PATH\n")
        self._patch_scan_dirs(monkeypatch, tmp_path, research_dir)
        violations = check_active_code_uses_canonical_db_path()
        assert violations == []
