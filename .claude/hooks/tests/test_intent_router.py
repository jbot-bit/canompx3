"""Tests for `.claude/hooks/intent-router.py`.

The hook is shell-invoked: tests drive it via subprocess + stdin JSON, the
same way Claude Code does. Fail-open paths are tested explicitly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "intent-router.py"
HOOKS_DIR = HOOK.parent
STATE_FILE = HOOKS_DIR / "state" / "intent-router.json"


def _run_hook(payload: dict, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
        timeout=10,
        check=False,
    )


@pytest.fixture(autouse=True)
def _clear_state():
    """Always start with no cooldown state — each test runs in isolation."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    yield
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def _parse_context(stdout: str) -> str | None:
    """Extract additionalContext from hook stdout, or None if no output."""
    stdout = stdout.strip()
    if not stdout:
        return None
    obj = json.loads(stdout)
    return obj["hookSpecificOutput"]["additionalContext"]


def test_match_orient_status():
    r = _run_hook({"prompt": "where are we at?"})
    assert r.returncode == 0
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/orient" in ctx
    assert "auto-skill-routing.md L" in ctx


def test_match_next():
    r = _run_hook({"prompt": "what now should I do"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/next" in ctx


def test_match_trade_book():
    r = _run_hook({"prompt": "show me the trade book"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/trade-book" in ctx


def test_match_regime_check():
    r = _run_hook({"prompt": "check portfolio fitness please"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/regime-check" in ctx


def test_match_quant_debug():
    r = _run_hook({"prompt": "something weird is happening with the cost model"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/quant-debug" in ctx


def test_match_design():
    r = _run_hook({"prompt": "let's brainstorm an approach"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/design" in ctx


def test_match_pinecone():
    r = _run_hook({"prompt": "remind me about the IBS history"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/pinecone-assistant" in ctx


def test_match_nogo():
    r = _run_hook({"prompt": "kill verdict on IBS"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/nogo" in ctx


def test_match_research():
    r = _run_hook({"prompt": "let me test a hypothesis on MNQ"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/research" in ctx


def test_match_capital_review():
    r = _run_hook({"prompt": "do a bias check before we go live"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/capital-review" in ctx


def test_match_code_review():
    r = _run_hook({"prompt": "check my work before I commit"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/code-review" in ctx


def test_match_crg_search():
    r = _run_hook({"prompt": "where is the orb_outcomes builder defined"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/crg-search" in ctx


def test_match_crg_blast():
    r = _run_hook({"prompt": "what will this break if I edit asset_configs"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/crg-blast" in ctx


def test_no_match_random_prompt():
    r = _run_hook({"prompt": "tell me a joke about turtles"})
    assert r.returncode == 0
    assert r.stdout.strip() == ""


def test_skip_slash_command():
    r = _run_hook({"prompt": "/orient now please"})
    assert r.returncode == 0
    assert r.stdout.strip() == ""


def test_skip_one_word():
    r = _run_hook({"prompt": "hello"})
    assert r.returncode == 0
    assert r.stdout.strip() == ""


def test_skip_empty():
    r = _run_hook({"prompt": ""})
    assert r.returncode == 0
    assert r.stdout.strip() == ""


def test_fail_open_malformed_json():
    """Hook receives garbage on stdin — exits 0 silently."""
    r = subprocess.run(
        [sys.executable, str(HOOK)],
        input="not json at all",
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert r.returncode == 0
    assert r.stdout.strip() == ""


def test_fail_open_empty_event():
    r = _run_hook({})
    assert r.returncode == 0
    assert r.stdout.strip() == ""


def test_cooldown_silences_repeat():
    """Second matching prompt for same skill within cooldown returns no output."""
    r1 = _run_hook({"prompt": "where are we at"})
    assert _parse_context(r1.stdout) is not None
    # Per-skill cooldown is 5min — within 30s, same skill, must be silent.
    r2 = _run_hook({"prompt": "status please catch me up"})
    assert r2.returncode == 0
    assert r2.stdout.strip() == ""


def test_longest_pattern_wins():
    """A prompt that matches multiple rules picks the longest literal phrase.

    'predicate lineage' (predicate-lineage rule) is longer than 'where is '
    (crg-search rule). When both match, predicate-lineage should win.
    """
    r = _run_hook({"prompt": "predicate lineage for break_dir"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "/crg-lineage" in ctx


def test_context_under_char_cap():
    """Injected line must stay within MAX_CONTEXT_CHARS budget."""
    r = _run_hook({"prompt": "where are we at"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert len(ctx) <= 100


def test_response_includes_line_reference():
    """Every match should cite the rules-file line for verifiability."""
    r = _run_hook({"prompt": "where are we at"})
    ctx = _parse_context(r.stdout)
    assert ctx is not None
    assert "auto-skill-routing.md L" in ctx
