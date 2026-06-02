"""Tests for prompt-broker.py: round-trip, dedup, and cap enforcement."""

from __future__ import annotations

import importlib.util
import json
import sys
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
    module = _load_broker()
    monkeypatch.setattr(module, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(module, "STAGES_DIR", tmp_path / "stages")
    monkeypatch.setattr(module, "STAGE_STATE_FILE", tmp_path / "STAGE_STATE.md")
    return module


def _extract(payload_str: str) -> str:
    payload = json.loads(payload_str)
    return payload["hookSpecificOutput"]["additionalContext"]


def test_empty_prompt_emits_nothing(broker, capsys):
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


def test_risk_critical_only(broker, capsys):
    event = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "deploy to production now",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    text = _extract(capsys.readouterr().out)
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


def test_all_four_match(broker, capsys, tmp_path):
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


def test_global_cap_enforced(broker, capsys, tmp_path):
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
    assert "[risk:critical]" in text


def test_risk_tier_dedup_within_cooldown(broker, capsys):
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

    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(payload)):
            broker.main()
    second = capsys.readouterr().out
    if second.strip():
        assert "RISK TIER" not in second


def test_risk_tier_change_bypasses_cooldown(broker, capsys):
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
        "prompt": "review",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event))):
            broker.main()
    first_text = _extract(capsys.readouterr().out)
    assert "[stage:" in first_text

    event2 = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "audit the holdout",
    }
    with pytest.raises(SystemExit):
        with patch.object(broker.sys, "stdin", _StubStdin(json.dumps(event2))):
            broker.main()
    second_text = capsys.readouterr().out
    if second_text.strip():
        body = _extract(second_text)
        assert "[stage:" not in body


def test_classifier_failure_isolated(broker, capsys, monkeypatch):
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
    assert "[bias:research]" not in text


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


class _StubStdin:
    def __init__(self, payload: str):
        self._payload = payload

    def read(self, *_args, **_kwargs) -> str:
        return self._payload
