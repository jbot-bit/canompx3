"""Tests for scripts/tools/check_referenced_paths.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tools import check_referenced_paths as crp


class TestLooksLikePath:
    def test_accepts_pipeline_prefix(self) -> None:
        assert crp._looks_like_path("pipeline/check_drift.py") is True

    def test_accepts_dot_claude_prefix(self) -> None:
        assert crp._looks_like_path(".claude/rules/foo.md") is True

    def test_accepts_docs_prefix(self) -> None:
        assert crp._looks_like_path("docs/ARCHITECTURE.md") is True

    def test_rejects_url(self) -> None:
        assert crp._looks_like_path("https://example.com/foo") is False

    def test_rejects_glob(self) -> None:
        assert crp._looks_like_path("docs/specs/*.md") is False

    def test_rejects_python_module(self) -> None:
        assert crp._looks_like_path("pipeline.dst.SESSION_CATALOG") is False

    def test_rejects_shell_var(self) -> None:
        assert crp._looks_like_path("$DUCKDB_PATH") is False

    def test_rejects_empty(self) -> None:
        assert crp._looks_like_path("") is False


class TestExtractRefs:
    def test_extracts_backtick_path(self) -> None:
        refs = crp._extract_refs("See `pipeline/check_drift.py` for details.")
        assert "pipeline/check_drift.py" in refs

    def test_skips_inline_code_non_path(self) -> None:
        refs = crp._extract_refs("Use `SELECT *` to query.")
        assert refs == []

    def test_extracts_multiple(self) -> None:
        refs = crp._extract_refs("`pipeline/dst.py` and `.claude/rules/foo.md`")
        assert len(refs) == 2


class TestMainFunction:
    def test_all_refs_exist_returns_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Create a real file
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "dst.py").write_text("", encoding="utf-8")

        doc = tmp_path / "CLAUDE.md"
        doc.write_text("See `pipeline/dst.py` for details.", encoding="utf-8")

        monkeypatch.setattr(crp, "PROJECT_ROOT", tmp_path)

        def fake_files():
            return [doc]

        monkeypatch.setattr(crp, "_files_to_scan", fake_files)

        result = crp.main(verbose=False)
        assert result == 0

    def test_missing_ref_returns_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = tmp_path / "CLAUDE.md"
        doc.write_text("See `pipeline/nonexistent.py` for details.", encoding="utf-8")

        monkeypatch.setattr(crp, "PROJECT_ROOT", tmp_path)

        def fake_files():
            return [doc]

        monkeypatch.setattr(crp, "_files_to_scan", fake_files)

        result = crp.main(verbose=False)
        assert result == 1

    def test_missing_ref_verbose_reports_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        doc = tmp_path / "CLAUDE.md"
        doc.write_text("See `pipeline/ghost.py` for details.", encoding="utf-8")

        monkeypatch.setattr(crp, "PROJECT_ROOT", tmp_path)

        def fake_files():
            return [doc]

        monkeypatch.setattr(crp, "_files_to_scan", fake_files)

        result = crp.main(verbose=True)
        assert result == 1
        captured = capsys.readouterr()
        assert "ghost.py" in captured.out

    def test_no_files_to_scan_returns_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(crp, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(crp, "_files_to_scan", lambda: [])
        result = crp.main()
        assert result == 0
