"""Regression tests for automatic second-pass/grounding prompt hooks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_HOOK = PROJECT_ROOT / ".claude" / "hooks" / "targeted-grounding-router.py"
CODEX_HOOK = PROJECT_ROOT / ".codex" / "hooks" / "user_prompt_submit_grounding.py"


def _run_hook(path: Path, prompt: str) -> str | None:
    result = subprocess.run(
        [sys.executable, str(path)],
        input=json.dumps({"prompt": prompt}),
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout.strip()
    if not stdout:
        return None
    payload = json.loads(stdout)
    return payload["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize(
    "prompt",
    [
        "2P this plan before you implement",
        "second pass this please",
        "can you improve this?",
        "is this good?",
        "take a look at this",
        "look over this plan",
        "what am I missing?",
        "poke holes in this",
        "find the blind spots",
        "does this approach hold up?",
        "sense check this",
        "fresh eyes on this",
        "red team this",
        "can you make this better",
        "thoughts on this implementation?",
        "QA this before we ship",
        "will this work?",
        "spot flaws in this",
        "any gotchas or risks?",
    ],
)
def test_claude_targeted_grounding_semantics(prompt: str) -> None:
    context = _run_hook(CLAUDE_HOOK, prompt)
    assert context is not None
    assert "compact truth check" in context
    assert "second-pass" in context


@pytest.mark.parametrize(
    "prompt",
    [
        "2P this plan before you implement",
        "is this good?",
        "what am I missing?",
        "poke holes in this",
        "sense check this",
        "will this work?",
        "spot flaws in this",
    ],
)
def test_codex_targeted_grounding_semantics(prompt: str) -> None:
    context = _run_hook(CODEX_HOOK, prompt)
    assert context is not None
    assert "compact truth check" in context
    assert "second-pass" in context


@pytest.mark.parametrize("prompt", ["/resource", "/lit", "read the trading bible grounding truth"])
def test_claude_resource_lit_route(prompt: str) -> None:
    context = _run_hook(CLAUDE_HOOK, prompt)
    assert context is not None
    assert "local-PC corpus" in context
    assert "raw resources only if present locally" in context
    assert "no skim/guess" in context


@pytest.mark.parametrize("prompt", ["/resource", "/lit", "read the local literature first"])
def test_codex_resource_lit_route(prompt: str) -> None:
    context = _run_hook(CODEX_HOOK, prompt)
    assert context is not None
    assert "local-PC corpus" in context
    assert "raw resources only if present locally" in context
    assert "no skim/guess" in context


@pytest.mark.parametrize(
    "prompt",
    [
        "research this and include user comments",
        "fetch docs changelog upgrades fixes and github issues",
        "look up official sources and forum reports",
    ],
)
def test_claude_research_fetch_separates_source_classes(prompt: str) -> None:
    context = _run_hook(CLAUDE_HOOK, prompt)
    assert context is not None
    assert "official" in context
    assert "unofficial" in context
    assert "cautionary signals" in context


@pytest.mark.parametrize(
    "prompt",
    [
        "research this and include user comments",
        "fetch docs changelog upgrades fixes and github issues",
        "look up official sources and forum reports",
    ],
)
def test_codex_research_fetch_separates_source_classes(prompt: str) -> None:
    context = _run_hook(CODEX_HOOK, prompt)
    assert context is not None
    assert "official" in context
    assert "unofficial" in context
    assert "cautionary signals" in context
