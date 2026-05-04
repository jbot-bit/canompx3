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


class TestNormalize:
    def test_strips_cli_prefix(self) -> None:
        assert crp._normalize("python pipeline/check_drift.py") == "pipeline/check_drift.py"

    def test_strips_cli_args(self) -> None:
        assert crp._normalize("research/foo.py --quantile-method is_only") == "research/foo.py"

    def test_strips_qualified_name(self) -> None:
        assert crp._normalize("pipeline/check_drift.py::check_x") == "pipeline/check_drift.py"

    def test_strips_qualified_name_with_parens(self) -> None:
        assert crp._normalize("research/foo.py::test_cell()") == "research/foo.py"

    def test_strips_line_number(self) -> None:
        assert crp._normalize("pipeline/foo.py:510") == "pipeline/foo.py"

    def test_strips_line_range(self) -> None:
        assert crp._normalize("pipeline/foo.py:1600-1660") == "pipeline/foo.py"

    def test_strips_multi_line_ranges(self) -> None:
        assert crp._normalize("pipeline/foo.py:586-594, :612-616, :357-381") == "pipeline/foo.py"

    def test_strips_trailing_comma(self) -> None:
        assert crp._normalize("pipeline/foo.py:586-594,") == "pipeline/foo.py"

    def test_combined_cli_qualified_line(self) -> None:
        # Order: CLI split, then qualified split, then line strip
        assert crp._normalize("python pipeline/foo.py::bar:42") == "pipeline/foo.py"

    def test_no_normalization_when_clean(self) -> None:
        assert crp._normalize("pipeline/foo.py") == "pipeline/foo.py"


class TestSkipRules:
    def test_anti_pattern_skipped(self) -> None:
        # /tmp/gold.db is cited as anti-pattern, never validated
        refs = crp._extract_refs("Never use `/tmp/gold.db` as default.")
        assert refs == []

    def test_anti_pattern_windows_skipped(self) -> None:
        refs = crp._extract_refs("Avoid `C:\\db\\gold.db` scratch copy.")
        assert refs == []

    def test_memory_path_skipped(self) -> None:
        # memory/ files live in user dir, not repo
        refs = crp._extract_refs("See `memory/feedback_foo.md` for details.")
        assert refs == []

    def test_stages_relative_path_skipped(self) -> None:
        # stages/ is relative to docs/runtime/, not project root
        refs = crp._extract_refs("Auto-creates `stages/auto_trivial.md`.")
        assert refs == []

    def test_template_with_angle_brackets_skipped(self) -> None:
        refs = crp._extract_refs("Write `docs/audit/hypotheses/<slug>.md`.")
        assert refs == []

    def test_template_with_yyyy_skipped(self) -> None:
        refs = crp._extract_refs("Path: `docs/audit/YYYY-MM-DD-foo.md`.")
        assert refs == []

    def test_template_with_pipe_skipped(self) -> None:
        refs = crp._extract_refs("`docs/audit/hypotheses/foo.yaml|.md` template.")
        assert refs == []


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

    def test_extracts_normalized_form(self) -> None:
        # Verify normalization is applied during extraction, not just file-existence check
        refs = crp._extract_refs("Run `python pipeline/check_drift.py --fast`.")
        assert refs == ["pipeline/check_drift.py"]


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
