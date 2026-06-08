from __future__ import annotations

from pathlib import Path

from scripts.infra import codex_parity


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_parity_check_passes_when_codex_indexes_claude_surfaces(tmp_path: Path) -> None:
    _write(tmp_path / ".claude/commands/check.md")
    _write(tmp_path / ".claude/agents/evidence-auditor.md")
    _write(tmp_path / ".claude/rules/mcp-usage.md")
    _write(tmp_path / ".claude/skills/verify/SKILL.md")
    _write(tmp_path / ".claude/hooks/session-start.py")

    _write(tmp_path / ".codex/COMMANDS.md", ".claude/commands/check.md")
    _write(tmp_path / ".codex/AGENTS.md", ".claude/agents/evidence-auditor.md")
    _write(tmp_path / ".codex/RULES.md", ".claude/rules/mcp-usage.md")
    _write(tmp_path / ".codex/HOOKS.md", ".claude/hooks/session-start.py")
    _write(tmp_path / ".codex/WORKFLOWS.md")
    _write(tmp_path / ".codex/INTEGRATIONS.md")
    _write(tmp_path / ".codex/skills/canompx3-claude-parity/SKILL.md", ".claude/skills/verify/SKILL.md")
    _write(tmp_path / ".codex/skills/canompx3-claude-parity/agents/openai.yaml")
    _write(tmp_path / ".agents/skills/canompx3-claude-parity/SKILL.md")

    report = codex_parity.check_parity(tmp_path)

    assert report["ok"] is True
    assert report["missing_refs"] == {}


def test_parity_check_reports_missing_references(tmp_path: Path) -> None:
    _write(tmp_path / ".claude/commands/check.md")
    _write(tmp_path / ".claude/agents/evidence-auditor.md")
    _write(tmp_path / ".claude/rules/mcp-usage.md")
    _write(tmp_path / ".claude/skills/verify/SKILL.md")
    _write(tmp_path / ".claude/hooks/session-start.py")

    for required in codex_parity.REQUIRED_CODEX_FILES:
        _write(tmp_path / required)

    report = codex_parity.check_parity(tmp_path)

    assert report["ok"] is False
    assert report["missing_refs"]["commands"] == [".claude/commands/check.md"]
    assert report["missing_refs"]["agents"] == [".claude/agents/evidence-auditor.md"]
    assert report["missing_refs"]["rules"] == [".claude/rules/mcp-usage.md"]
    assert report["missing_refs"]["skills"] == [".claude/skills/verify/SKILL.md"]
    assert report["missing_refs"]["hooks"] == [".claude/hooks/session-start.py"]
