"""Tests for scripts/tools/ask.py — operator CLI.

Covers dotenv loading, profile alias resolution, default-model env injection,
chat dry-run path (incl. provider safety, web/reasoning/multimodal/fallback
flags, stdin pipe), grounded mode dispatch, --models capability filtering, and
REPL command parsing surface (the parser, not the input loop).
"""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ask = importlib.import_module("scripts.tools.ask")


# ---------- env / profile basics ----------


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


def test_resolve_chat_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CANOMPX3_AI_CHAT_MODEL", "anthropic/claude-opus-4.7")
    assert ask._resolve_chat_model() == "anthropic/claude-opus-4.7"


def test_resolve_chat_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CANOMPX3_AI_CHAT_MODEL", raising=False)
    assert ask._resolve_chat_model() == ask._DEFAULT_CHAT_MODEL_FALLBACK


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


# ---------- chat payload shape ----------


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


def test_chat_dry_run_includes_safety_provider_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat path must default to data_collection=deny + allow_fallbacks=false."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="x",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
    )
    provider = envelope["request"]["provider"]
    assert provider["data_collection"] == "deny"
    assert provider["allow_fallbacks"] is False


def test_chat_allow_fallbacks_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="x",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
        allow_fallbacks=True,
    )
    assert envelope["request"]["provider"]["allow_fallbacks"] is True


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


# ---------- web search ----------


def test_web_plugin_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="news",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
        web=True,
        web_engine="exa",
        web_results=3,
    )
    plugins = envelope["request"]["plugins"]
    assert plugins == [{"id": "web", "engine": "exa", "max_results": 3}]


def test_web_plugin_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="news",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
        web=True,
    )
    assert envelope["request"]["plugins"] == [{"id": "web"}]


# ---------- reasoning ----------


def test_reasoning_effort_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="hard",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=2000,
        stream=False,
        dry=True,
        think=True,
        effort="medium",
    )
    assert envelope["request"]["reasoning"] == {"enabled": True, "effort": "medium"}


def test_reasoning_max_tokens_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="hard",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=10000,
        stream=False,
        dry=True,
        think=True,
        reasoning_tokens=2048,
    )
    assert envelope["request"]["reasoning"] == {"enabled": True, "max_tokens": 2048}


def test_reasoning_default_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """--think with no effort/tokens defaults to effort=high."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="hard",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=2000,
        stream=False,
        dry=True,
        think=True,
    )
    assert envelope["request"]["reasoning"] == {"enabled": True, "effort": "high"}


# ---------- multimodal ----------


def test_image_attachment_encodes_base64(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    img_path = tmp_path / "fake.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")

    envelope = ask._chat_request(
        question="what is this",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
        images=[img_path],
    )
    user_content = envelope["request"]["messages"][-1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0] == {"type": "text", "text": "what is this"}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_pdf_attachment_encodes_base64(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    envelope = ask._chat_request(
        question="summarize",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
        pdfs=[pdf_path],
    )
    user_content = envelope["request"]["messages"][-1]["content"]
    file_part = next(p for p in user_content if p["type"] == "file")
    assert file_part["file"]["filename"] == "doc.pdf"
    assert file_part["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_attachment_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ask._attachment_data_url(tmp_path / "nope.png")


def test_user_content_text_only_when_no_attachments() -> None:
    content = ask._build_user_content("hello", None, None)
    assert content == "hello"


# ---------- fallback / transforms ----------


def test_models_fallback_array(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="x",
        history=None,
        model="primary/model",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
        fallback_models=["a/b", "c/d"],
    )
    assert envelope["request"]["models"] == ["a/b", "c/d"]


def test_route_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="x",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
        route_fallback=True,
    )
    assert envelope["request"]["route"] == "fallback"


def test_transforms_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    envelope = ask._chat_request(
        question="x",
        history=None,
        model="m",
        system="",
        temperature=0.7,
        max_tokens=50,
        stream=False,
        dry=True,
        transforms=["middle-out"],
    )
    assert envelope["request"]["transforms"] == ["middle-out"]


# ---------- model filtering ----------


def test_filter_models_required_params() -> None:
    models = [
        {"id": "a", "supported_parameters": ["tools", "reasoning"], "context_length": 100000},
        {"id": "b", "supported_parameters": ["tools"], "context_length": 100000},
    ]
    out = ask._filter_models(models, required_params={"reasoning"})
    assert [m["id"] for m in out] == ["a"]


def test_filter_models_min_ctx_and_tools() -> None:
    models = [
        {"id": "small", "supported_parameters": ["tools"], "context_length": 8000},
        {"id": "big", "supported_parameters": ["tools"], "context_length": 200000},
        {"id": "no-tools", "supported_parameters": [], "context_length": 200000},
    ]
    out = ask._filter_models(models, tools_only=True, min_ctx=100000)
    assert [m["id"] for m in out] == ["big"]


def test_filter_models_max_cost() -> None:
    models = [
        {
            "id": "cheap",
            "supported_parameters": [],
            "context_length": 0,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
        },
        {
            "id": "expensive",
            "supported_parameters": [],
            "context_length": 0,
            "pricing": {"prompt": "0.00005", "completion": "0.00015"},
        },
    ]
    # 1.0 $/M prompt = 0.000001 $/token. cheap=0.1 $/M, expensive=50 $/M.
    out = ask._filter_models(models, max_prompt_cost=1.0)
    assert [m["id"] for m in out] == ["cheap"]


def test_model_pricing_handles_missing() -> None:
    assert ask._model_pricing({"id": "x"}) == (0.0, 0.0)
    assert ask._model_pricing({"id": "x", "pricing": {}}) == (0.0, 0.0)
    assert ask._model_pricing({"id": "x", "pricing": {"prompt": "bad", "completion": "0.0"}}) == (0.0, 0.0)


# ---------- stdin / pipe ----------


def test_pipe_reads_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("piped content here"))
    args = ask.build_parser().parse_args(["--pipe"])
    q = ask._read_stdin_if_pipe(args)
    assert q == "piped content here"


def test_pipe_appends_to_positional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("error log content"))
    args = ask.build_parser().parse_args(["--pipe", "what", "failed?"])
    q = ask._read_stdin_if_pipe(args)
    assert q.startswith("what failed?")
    assert "error log content" in q


def test_no_pipe_uses_positional_only() -> None:
    args = ask.build_parser().parse_args(["just", "text"])
    q = ask._read_stdin_if_pipe(args)
    assert q == "just text"


# ---------- parser ----------


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
    assert args.web is False
    assert args.think is False
    assert args.image is None
    assert args.pdf is None
    assert args.pipe is False
    assert args.allow_fallbacks is False


def test_parser_repeatable_image_pdf() -> None:
    args = ask.build_parser().parse_args(["--image", "a.png", "--image", "b.png", "--pdf", "c.pdf", "q"])
    assert args.image == ["a.png", "b.png"]
    assert args.pdf == ["c.pdf"]


def test_split_csv() -> None:
    assert ask._split_csv("a,b, c") == ["a", "b", "c"]
    assert ask._split_csv("") is None
    assert ask._split_csv(None) is None
    assert ask._split_csv("  ,  ") is None


# ---------- main entry ----------


def test_main_dry_chat_no_api_required(capsys: pytest.CaptureFixture[str]) -> None:
    rc = ask.main(["--dry", "what", "is", "2+2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "what is 2+2" in out


def test_main_dry_includes_provider_safety(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = ask.main(["--dry", "hello"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.split("\n\n", 1)[1])
    assert payload["provider"]["data_collection"] == "deny"
    assert payload["provider"]["allow_fallbacks"] is False


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


def test_main_effort_and_tokens_mutex(monkeypatch: pytest.MonkeyPatch) -> None:
    """argparse declares --reasoning-tokens in the mutex group; --effort isn't.
    The runtime guard rejects the combination explicitly."""
    monkeypatch.setattr(ask, "_load_dotenv", lambda root: None)
    rc = ask.main(["--dry", "--think", "--effort", "high", "--reasoning-tokens", "1024", "x"])
    assert rc == 2


# ---------- history persistence ----------


def test_history_save_and_load_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ask, "HISTORY_DIR", tmp_path)
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    saved_path = ask._save_history("test-session", history)
    assert saved_path.exists()
    loaded = ask._load_history("test-session")
    assert loaded == history
    listed = ask._list_histories()
    assert "test-session" in listed


def test_history_load_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ask, "HISTORY_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        ask._load_history("does-not-exist")


def test_history_path_sanitizes() -> None:
    """Hostile session names must not escape the history dir."""
    p = ask._history_path("../../etc/passwd")
    # Sanitization replaces / and . — file must end up directly under HISTORY_DIR.
    assert p.parent.resolve() == ask.HISTORY_DIR.resolve()
