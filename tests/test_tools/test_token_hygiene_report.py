from __future__ import annotations

from pathlib import Path

from scripts.tools import token_hygiene_report


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_report_counts_only_active_stage_files(monkeypatch, tmp_path: Path, capsys) -> None:
    _mkfile(tmp_path / ".claude" / "settings.json", "{}")
    _mkfile(tmp_path / ".gitignore", "SOUL.md\nUSER.md\nmemory/\n")
    _mkfile(tmp_path / "CLAUDE.md", "# Claude\n")
    _mkfile(tmp_path / "CODEX.md", "# Codex\n")
    _mkfile(tmp_path / "AGENTS.md", "# Agents\n")
    _mkfile(
        tmp_path / "docs" / "runtime" / "stages" / "active-stage.md",
        "---\ntask: Active stage\nmode: IMPLEMENTATION\n---\n",
    )
    _mkfile(
        tmp_path / "docs" / "runtime" / "stages" / "closed-stage.md",
        "---\ntask: Closed stage\nmode: IMPLEMENTATION\nstatus: closed\n---\n",
    )
    _mkfile(
        tmp_path / "docs" / "runtime" / "stages" / "loose-template.md",
        "task: Loose template\nmode: IMPLEMENTATION\n",
    )

    monkeypatch.setattr(token_hygiene_report, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(token_hygiene_report, "RULES_DIR", tmp_path / ".claude" / "rules")
    monkeypatch.setattr(token_hygiene_report, "STAGES_DIR", tmp_path / "docs" / "runtime" / "stages")
    monkeypatch.setattr(token_hygiene_report, "SETTINGS_PATH", tmp_path / ".claude" / "settings.json")
    monkeypatch.setattr(token_hygiene_report, "GITIGNORE_PATH", tmp_path / ".gitignore")
    monkeypatch.setattr(token_hygiene_report, "_home_claude_candidates", lambda: [])

    assert token_hygiene_report.main() == 0

    out = capsys.readouterr().out
    assert "- active stage files: 1" in out
    assert "active-stage" in out
    assert "closed-stage" not in out
    assert "loose-template" not in out
