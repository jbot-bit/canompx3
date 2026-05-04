"""Tests for prompt-broker.py — round-trip + per-guard + token-cap.

Mirrors the importlib pattern in test_main_ci_preflight.py since
`prompt-broker.py` has a hyphenated filename.

Run: `pytest .claude/hooks/tests/test_prompt_broker.py -v`
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

HOOK_PATH = Path(__file__).resolve().parents[1] / "prompt-broker.py"


def _load_broker():
    spec = importlib.util.spec_from_file_location("prompt_broker", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prompt_broker"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def broker(tmp_path, monkeypatch):
    """Fresh broker module per test, with STATE_DIR + STAGES_DIR pinned to
    a temp path so each test starts clean.
    """
    module = _load_broker()
    # All state writes redirected to the temp dir
    monkeypatch.setattr(module, "STATE_DIR", tmp_path / "state")
    # Stage paths redirected so we control the stage population per test
    monkeypatch.setattr(module, "STAGES_DIR", tmp_path / "stages")
    monkeypatch.setattr(
        module, "STAGE_STATE_FILE", tmp_path / "STAGE_STATE.md"
    )
    return module


def _extract(payload_str: str) -> str:
    """Parse stdout JSON envelope and return additionalContext text."""
    payload = json.loads(payload_str)
    return payload["hookSpecificOutput"]["additionalContext"]


# ───────────────────── 1. Empty / no-match prompt ────────────────────

def test_empty_prompt_emits_nothing(broker, capsys):
    """Stage 'none' branch fires for any non-empty prompt; truly-empty
    prompts exit before any classifier."""
    with patch("sys.stdin.read", return_value=""):
        # main() calls json.load(sys.stdin) — feed empty string -> JSONDecodeError -> exit 0
        with pytest.raises(SystemExit) as exc:
            with patch.object(broker.sys, "stdin", _StubStdin("")):
                broker.main()
    assert exc.value.code == 0
    assert capsys.readouterr().out == ""


def test_whitespace_prompt_emits_nothing(broker, capsys):
    event = {"hook_event_name": "UserPromptSubmit", "prompt": "   "}
    with pytest.raises(SystemExit) as exc:
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    assert exc.value.code == 0
    assert capsys.readouterr().out == ""


# ───────────────────── 2. Single-classifier hits ─────────────────────

def test_risk_critical_only(broker, capsys):
    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "deploy to production now",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    out = capsys.readouterr().out
    text = _extract(out)
    assert "[risk:critical]" in text
    assert "RISK TIER: critical" in text


def test_risk_high_only(broker, capsys):
    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "let's review the schema migration",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    text = _extract(capsys.readouterr().out)
    assert "[risk:high]" in text
    assert "RISK TIER: high" in text


def test_bias_research_fires(broker, capsys):
    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "what are the OOS results for the audit",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    text = _extract(capsys.readouterr().out)
    assert "[bias:research]" in text


def test_intent_design(broker, capsys):
    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "let's brainstorm the approach",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    text = _extract(capsys.readouterr().out)
    assert "[intent:design]" in text
    assert "DESIGN MODE" in text


def test_intent_commit(broker, capsys):
    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "commit and push",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    text = _extract(capsys.readouterr().out)
    assert "[intent:commit]" in text
    assert "GIT OPERATION" in text


# ───────────────────── 3. Multi-classifier (all 4) ───────────────────

def test_all_four_match(broker, capsys, tmp_path):
    """A crafted prompt + active stage file should hit risk + bias +
    intent + stage simultaneously."""
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    (stages_dir / "demo.md").write_text(
        "mode: IMPLEMENTATION\n"
        "task: example\n"
        "stage: 1\n"
        "stage_of: 2\n"
        "## Blast Radius\n"
        "- file_a.py changes touching lots of downstream consumers\n",
        encoding="utf-8",
    )

    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "review the deploy readiness OOS audit and brainstorm",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    text = _extract(capsys.readouterr().out)
    assert "[risk:" in text
    assert "[bias:research]" in text
    assert "[intent:" in text
    assert "[stage:" in text
    assert len(text) <= broker.GLOBAL_CAP


# ───────────────────── 4. Token-cap enforcement ──────────────────────

def test_global_cap_enforced(broker, capsys, tmp_path, monkeypatch):
    """Pathological stage state (huge content) — risk-tier line MUST survive."""
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    huge_blast = "\n".join(f"- file_{i}.py downstream churn" for i in range(500))
    (stages_dir / "huge.md").write_text(
        f"mode: IMPLEMENTATION\ntask: x\n## Blast Radius\n{huge_blast}\n",
        encoding="utf-8",
    )

    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "deploy to production",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    text = _extract(capsys.readouterr().out)
    assert len(text) <= broker.GLOBAL_CAP
    assert "[risk:critical]" in text  # safety-priority line survives


# ───────────────────── 5. Per-guard dedup state ──────────────────────

def test_risk_tier_dedup_within_cooldown(broker, capsys):
    """Same prompt twice within 20m -> second invocation emits no risk line."""
    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "deploy to production",
    }
    payload = json.dumps(event)

    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(payload)):
            broker.main()
    first = capsys.readouterr().out
    assert "[risk:critical]" in _extract(first)

    # Second call — state populated, cooldown active
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(payload)):
            broker.main()
    second = capsys.readouterr().out
    if second.strip():
        # If any output, it must NOT contain the risk message
        assert "RISK TIER" not in second


def test_risk_tier_change_bypasses_cooldown(broker, capsys):
    """high -> critical must emit despite recent fire (tier changed)."""
    high_event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "review the schema",
    }
    crit_event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "deploy to production",
    }

    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(high_event))):
            broker.main()
    first = capsys.readouterr().out
    assert "[risk:high]" in _extract(first)

    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(crit_event))):
            broker.main()
    second = capsys.readouterr().out
    assert "[risk:critical]" in _extract(second)


# ───────────────────── 6. Stage hash cooldown ────────────────────────

def test_stage_hash_cooldown(broker, capsys, tmp_path):
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    stage_file = stages_dir / "demo.md"
    stage_file.write_text(
        "mode: IMPLEMENTATION\n"
        "task: example\n"
        "## Blast Radius\n"
        "- file_a.py with sufficient detail to satisfy the 30-char floor\n",
        encoding="utf-8",
    )

    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "review",  # only triggers risk:high + stage
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    first_text = _extract(capsys.readouterr().out)
    assert "[stage:" in first_text

    # Same mtime -> cooldown -> no stage line on next emit
    # Use a different prompt so risk-tier cooldown doesn't suppress everything
    event2 = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "audit the holdout",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event2))):
            broker.main()
    second_text = capsys.readouterr().out
    if second_text.strip():
        # bias may still fire; stage should not
        body = _extract(second_text)
        assert "[stage:" not in body


# ───────────────────── 7. Failure isolation ──────────────────────────

def test_classifier_failure_isolated(broker, capsys, monkeypatch):
    """If _classify_bias raises, the broker still emits the other 3 tags."""
    def boom(_prompt):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(broker, "_classify_bias", boom)

    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "deploy to production and brainstorm the audit",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    text = _extract(capsys.readouterr().out)
    assert "[risk:critical]" in text
    assert "[intent:" in text
    # bias subroutine crashed -> tag 'error' or just absent
    assert "[bias:research]" not in text


# ───────────────────── 8. Wrong event name ───────────────────────────

def test_non_userpromptsubmit_event_exits_silently(broker, capsys):
    event = {
        "hook_event_name": "Notification",
        "prompt": "deploy to production",
    }
    with pytest.raises(SystemExit) as exc:
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    assert exc.value.code == 0
    assert capsys.readouterr().out == ""


# ───────────────────── helpers ───────────────────────────────────────

class _StubStdin:
    """Minimal stdin replacement for json.load(). Implements the
    `read()` method that json.load delegates to.
    """

    def __init__(self, payload: str):
        self._payload = payload

    def read(self, *_args, **_kwargs) -> str:
        return self._payload
