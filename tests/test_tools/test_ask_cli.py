"""Tests for scripts/tools/ask.py — operator CLI.

Covers dotenv loading, profile alias resolution, default-model env injection,
chat dry-run path, grounded mode dispatch, --models capability filtering, and
REPL command parsing surface (the parser, not the input loop).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ask = importlib.import_module("scripts.tools.ask")


def test_load_dotenv_skips_comments_and_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "OPENROUTER_API_KEY=sk-or-from-file\n"
        "ALREADY_SET=from-file\n"
        "BAD_LINE_NO_EQUALS\n"
        '  QUOTED_VALUE = "with spaces"  \n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ALREADY_SET", "from-env")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)

    ask._load_dotenv(tmp_path)

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-file"
    assert os.environ["ALREADY_SET"] == "from-env"
    assert os.environ["QUOTED_VALUE"] == "with spaces"


def test_resolve_profile_aliases() -> None:
    assert ask._resolve_profile("plan") == "deepseek_planning"
    assert ask._resolve_profile("long") == "deepseek_research_long_context"
    assert ask._resolve_profile("extract") == "deepseek_structured_extraction"
    assert ask._resolve_profile("research") == "deepseek_research_long_context"
    assert ask._resolve_profile("deepseek_planning") == "deepseek_planning"


def test_ensure_default_models_does_not_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL", "explicit/model")
    monkeypatch.delenv("CANOMPX3_AI_DEEPSEEK_RESEARCH_LONG_CONTEXT_MODEL", raising=False)

    ask._ensure_default_models()

    assert os.environ["CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL"] == "explicit/model"
    assert (
        os.environ["CANOMPX3_AI_DEEPSEEK_RESEARCH_LONG_CONTEXT_MODEL"]
        == ask.DEFAULT_MODELS["deepseek_research_long_context"]
    )


def test_check_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    err = ask._check_api_key()
    assert err is not None
    assert "OPENROUTER_API_KEY" in err


def test_check_api_key_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-your-key-here")
    err = ask._check_api_key()
    assert err is not None


def test_check_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real-looking")
    assert ask._check_api_key() is None


def test_chat_dry_run_returns_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="hello",
        history=None,
        model="some/model",
        system="be concise",
        temperature=0.5,
        max_tokens=100,
        stream=False,
        dry=True,
    )
    assert envelope["status"] == "dry_run"
    request = envelope["request"]
    assert request["model"] == "some/model"
    assert request["temperature"] == 0.5
    assert request["max_tokens"] == 100
    assert request["messages"][0] == {"role": "system", "content": "be concise"}
    assert request["messages"][-1] == {"role": "user", "content": "hello"}


def test_chat_history_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    envelope = ask._chat_request(
        question="follow-up",
        history=history,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
    )
    messages = envelope["request"]["messages"]
    assert messages[0]["content"] == "earlier question"
    assert messages[1]["content"] == "earlier answer"
    assert messages[-1] == {"role": "user", "content": "follow-up"}


def test_parser_mutually_exclusive_modes() -> None:
    parser = ask.build_parser()
    args = parser.parse_args(["--plan", "test question"])
    assert args.plan is True
    assert args.long is False
    assert args.code is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--plan", "--long", "x"])


def test_parser_defaults() -> None:
    args = ask.build_parser().parse_args(["hello", "world"])
    assert args.question == ["hello", "world"]
    assert args.temp == pytest.approx(0.7)
    assert args.max_tokens == 2048
    assert args.max_turns == 4
    assert args.no_stream is False
    assert args.dry is False


def test_main_dry_chat_no_api_required(capsys: pytest.CaptureFixture[str]) -> None:
    rc = ask.main(["--dry", "what", "is", "2+2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "what is 2+2" in out


def test_main_grounded_dry_uses_planning_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setenv("CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL", "deepseek/deepseek-chat")
    captured: dict = {}

    def fake_run(question: str, profile_id: str, max_turns: int, dry: bool, schema: str | None):
        captured["question"] = question
        captured["profile"] = profile_id
        captured["dry"] = dry
        return {"status": "dry_run", "request": {"profile": profile_id}}

    monkeypatch.setattr(ask, "_run_grounded", fake_run)
    rc = ask.main(["--plan", "--dry", "test"])
    assert rc == 0
    assert captured["profile"] == "deepseek_planning"
    assert captured["dry"] is True


def test_main_no_key_returns_3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ask, "_load_dotenv", lambda root: None)
    rc = ask.main(["hello"])
    assert rc == 3
