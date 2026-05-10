from __future__ import annotations

from pathlib import Path

from scripts.tools import agent_prompt_audit


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_flags_read_only_markdown_with_write_tools(tmp_path: Path) -> None:
    agent = write(
        tmp_path / ".claude" / "agents" / "bad.md",
        """---
name: bad
tools: Read, Write, Edit
---
You are read-only. Never edit files.
""",
    )

    findings = agent_prompt_audit.audit_path(agent)

    assert [finding.rule for finding in findings] == ["read-only-tools"]


def test_flags_silent_skip_and_volatile_count(tmp_path: Path) -> None:
    agent = write(
        tmp_path / ".claude" / "agents" / "bad.md",
        """---
name: bad
tools: Read, Bash
---
If unavailable, skip silently.
Query all 4 instruments.
""",
    )

    findings = agent_prompt_audit.audit_path(agent)

    assert {finding.rule for finding in findings} == {"silent-skip", "volatile-fact"}


def test_passes_hardened_codex_read_only_agent(tmp_path: Path) -> None:
    agent = write(
        tmp_path / ".codex" / "agents" / "reviewer.toml",
        '''name = "reviewer"
description = "Read-only reviewer."
sandbox_mode = "read-only"
developer_instructions = """
Stay read-only.
Report skipped checks as `SKIPPED - reason - residual risk`.
"""
''',
    )

    assert agent_prompt_audit.audit_path(agent) == []


def test_flags_research_agent_without_local_literature_grounding(tmp_path: Path) -> None:
    agent = write(
        tmp_path / ".claude" / "agents" / "research.md",
        """---
name: research
tools: Read, Bash
---
Review strategy claims for FDR, DSR, holdout, and leakage.
""",
    )

    findings = agent_prompt_audit.audit_path(agent)

    assert {finding.rule for finding in findings} == {"research-lit-grounding"}


def test_main_scans_project_agent_dirs(tmp_path: Path, capsys) -> None:
    write(
        tmp_path / ".claude" / "agents" / "ok.md",
        """---
name: ok
tools: Read, Bash
---
Read-only review. Report skipped checks as SKIPPED with reason.
""",
    )

    assert agent_prompt_audit.main(["--root", str(tmp_path)]) == 0
    assert "PASS" in capsys.readouterr().out
