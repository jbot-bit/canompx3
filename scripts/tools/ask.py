#!/usr/bin/env python3
"""ask — operator CLI for OpenRouter.

Two layers:

  CHAT (default)         Free chat. Any of OpenRouter's 300+ models. Streams.
  GROUNDED (--repo / --plan / --long / --extract)
                         Bounded read-only research path through the canonical
                         provider_registry + research_packet runtime.

Power features (chat path):
  --web                  Real-time web search (OpenRouter web plugin).
  --think [--effort H]   Reasoning tokens (effort: minimal/low/medium/high/xhigh).
  --image PATH           Attach images (repeatable).
  --pdf PATH             Attach PDFs (repeatable).
  --pipe                 Read stdin as the question (or appended to it).
  --models-fallback      Comma-separated fallback model ids.

Safety:
  Chat path defaults to provider.data_collection=deny + allow_fallbacks=false.
  Grounded path keeps the strict provider_registry contract.

Usage:
    ask "what does this regex do: r'\\d+'"
    ask --code "review this python snippet"
    ask --web "latest 4.6 → 4.7 migration notes"
    ask --think --effort high "design a lock-free queue"
    ask --image chart.png "what regime is this?"
    cat error.log | ask --pipe --code "what failed?"
    ask --plan "scale prop firm setup"
    ask --models --tools-only --min-ctx 100000
    ask                                  # interactive REPL
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _preferred_repo_python() -> Path | None:
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv-wsl" / "bin" / "python"
    return candidate if candidate.exists() else None


def _ensure_repo_python() -> None:
    if "pytest" in sys.modules:
        return
    expected = _preferred_repo_python()
    if expected is None:
        return
    current_prefix = Path(sys.prefix).resolve()
    expected_prefix = expected.parent.parent.resolve()
    if current_prefix == expected_prefix or os.environ.get("CANOMPX3_BOOTSTRAP_DONE") == "1":
        return
    env = os.environ.copy()
    env["CANOMPX3_BOOTSTRAP_DONE"] = "1"
    raise SystemExit(
        subprocess.call(
            [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    )


def _load_dotenv(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


PROFILE_ALIASES = {
    "plan": "deepseek_planning",
    "planning": "deepseek_planning",
    "long": "deepseek_research_long_context",
    "longctx": "deepseek_research_long_context",
    "research": "deepseek_research_long_context",
    "extract": "deepseek_structured_extraction",
    "structured": "deepseek_structured_extraction",
}

DEFAULT_MODELS = {
    "deepseek_planning": "deepseek/deepseek-chat",
    "deepseek_research_long_context": "deepseek/deepseek-chat",
    "deepseek_structured_extraction": "deepseek/deepseek-chat",
}

# Resolved at runtime — env var CANOMPX3_AI_CHAT_MODEL overrides.
_DEFAULT_CHAT_MODEL_FALLBACK = "deepseek/deepseek-chat"

CACHE_DIR = PROJECT_ROOT / ".cache"
MODELS_CACHE = CACHE_DIR / "openrouter_models.json"
MODELS_CACHE_TTL_S = 24 * 3600
HISTORY_DIR = CACHE_DIR / "ask_history"

# Default safety contract for chat path. Mirrors provider_registry.ProviderRouting
# contract used by grounded profiles: deny data collection, no silent fallbacks.
CHAT_PROVIDER_DEFAULTS: dict[str, Any] = {
    "data_collection": "deny",
    "allow_fallbacks": False,
}


def _resolve_chat_model() -> str:
    raw = os.environ.get("CANOMPX3_AI_CHAT_MODEL", "").strip()
    if not raw:
        return _DEFAULT_CHAT_MODEL_FALLBACK
    # OpenRouter requires ``provider/model`` shape (e.g. ``deepseek/deepseek-chat``).
    # A typo without the provider prefix returns a 400 from the API with no
    # operator-visible hint that the env var was the cause. Warn + fall back
    # so the operator sees the misconfiguration in the same line of output.
    if "/" not in raw:
        logger = logging.getLogger(__name__)
        logger.warning(
            "CANOMPX3_AI_CHAT_MODEL=%r missing 'provider/' prefix — "
            "falling back to %s. Expected format: deepseek/deepseek-chat.",
            raw,
            _DEFAULT_CHAT_MODEL_FALLBACK,
        )
        return _DEFAULT_CHAT_MODEL_FALLBACK
    return raw


def _ensure_default_models() -> None:
    for profile_id, default_model in DEFAULT_MODELS.items():
        env_key = f"CANOMPX3_AI_{profile_id.upper()}_MODEL"
        if not os.environ.get(env_key):
            os.environ[env_key] = default_model


def _check_api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key.startswith("sk-or-your"):
        return (
            "OPENROUTER_API_KEY missing or placeholder.\n"
            "  Get one: https://openrouter.ai/keys\n"
            "  Add to .env: OPENROUTER_API_KEY=sk-or-...\n"
        )
    return None


def _resolve_profile(name: str) -> str:
    return PROFILE_ALIASES.get(name.lower(), name)


# ---------- model metadata cache ----------


def _fetch_models_cached(*, force_refresh: bool = False) -> list[dict]:
    """Return the OpenRouter /models payload, cached on disk for 24h."""
    import httpx

    if not force_refresh and MODELS_CACHE.exists():
        age = time.time() - MODELS_CACHE.stat().st_mtime
        if age < MODELS_CACHE_TTL_S:
            try:
                return json.loads(MODELS_CACHE.read_text(encoding="utf-8")).get("data", [])
            except (json.JSONDecodeError, OSError):
                # Corrupt cache — fall through to refetch.
                pass

    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=15.0)
    response.raise_for_status()
    payload = response.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE.write_text(json.dumps(payload), encoding="utf-8")
    return payload.get("data", [])


def _model_pricing(entry: dict) -> tuple[float, float]:
    """Return (prompt $/M, completion $/M) for a model entry. OpenRouter prices in $/token."""
    pricing = entry.get("pricing") or {}
    try:
        prompt_per_token = float(pricing.get("prompt", "0") or 0)
        completion_per_token = float(pricing.get("completion", "0") or 0)
    except (TypeError, ValueError):
        return (0.0, 0.0)
    return (prompt_per_token * 1_000_000, completion_per_token * 1_000_000)


def _filter_models(
    models: list[dict],
    *,
    required_params: set[str] | None = None,
    tools_only: bool = False,
    reasoning_only: bool = False,
    min_ctx: int | None = None,
    max_prompt_cost: float | None = None,
    max_completion_cost: float | None = None,
    provider_substr: str | None = None,
) -> list[dict]:
    out = []
    for m in models:
        params = set(m.get("supported_parameters") or [])
        if required_params and not required_params.issubset(params):
            continue
        if tools_only and "tools" not in params:
            continue
        if reasoning_only and "reasoning" not in params:
            continue
        ctx = m.get("context_length") or 0
        if min_ctx is not None and ctx < min_ctx:
            continue
        prompt_cost, completion_cost = _model_pricing(m)
        if max_prompt_cost is not None and prompt_cost > max_prompt_cost:
            continue
        if max_completion_cost is not None and completion_cost > max_completion_cost:
            continue
        if provider_substr and provider_substr.lower() not in (m.get("id", "").lower()):
            continue
        out.append(m)
    return sorted(out, key=lambda x: x.get("id", ""))


def _list_models_for_profile(profile_id: str, *, force_refresh: bool = False) -> list[dict]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from trading_app.ai.provider_registry import assert_openrouter_research_profile

    profile = assert_openrouter_research_profile(profile_id)
    required = set(profile.required_parameters)
    if profile.response_mode == "json_schema":
        required.update({"response_format", "structured_outputs"})

    models = _fetch_models_cached(force_refresh=force_refresh)
    return _filter_models(models, required_params=required)


def _show_models(args: argparse.Namespace) -> int:
    """List OpenRouter models passing capability gates per profile + optional chat-mode filters."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from trading_app.ai.provider_registry import list_openrouter_research_profiles

    sys.stderr.write("[ask] fetching OpenRouter model catalog (cached 24h)...\n")

    has_filters = (
        args.tools_only
        or args.reasoning_only
        or args.min_ctx is not None
        or args.max_prompt_cost is not None
        or args.max_completion_cost is not None
        or args.provider_filter is not None
    )

    if has_filters:
        # Filter the universe directly, ignore profile gates.
        all_models = _fetch_models_cached(force_refresh=args.refresh)
        filtered = _filter_models(
            all_models,
            tools_only=args.tools_only,
            reasoning_only=args.reasoning_only,
            min_ctx=args.min_ctx,
            max_prompt_cost=args.max_prompt_cost,
            max_completion_cost=args.max_completion_cost,
            provider_substr=args.provider_filter,
        )
        sys.stdout.write(f"\n## filtered ({len(filtered)} matches)\n")
        for m in filtered[:60]:
            ctx = m.get("context_length") or 0
            pp, cp = _model_pricing(m)
            sys.stdout.write(f"   - {m.get('id', '')}  ctx={ctx}  ${pp:.2f}/M in  ${cp:.2f}/M out\n")
        if len(filtered) > 60:
            sys.stdout.write(f"   ... +{len(filtered) - 60} more\n")
        return 0

    for profile_id in list_openrouter_research_profiles():
        try:
            entries = _list_models_for_profile(profile_id, force_refresh=args.refresh)
        except (ValueError, KeyError, RuntimeError) as exc:
            sys.stdout.write(f"\n## {profile_id}\n  ERROR: {exc}\n")
            continue
        sys.stdout.write(f"\n## {profile_id}  ({len(entries)} compatible)\n")
        env_key = f"CANOMPX3_AI_{profile_id.upper()}_MODEL"
        current = os.environ.get(env_key, "(unset)")
        sys.stdout.write(f"   currently: {current}  (set via {env_key})\n")
        for m in entries[:30]:
            mid = m.get("id", "")
            ctx = m.get("context_length") or 0
            marker = " <-- current" if mid == current else ""
            sys.stdout.write(f"   - {mid}  ctx={ctx}{marker}\n")
        if len(entries) > 30:
            sys.stdout.write(f"   ... +{len(entries) - 30} more\n")
    return 0


# ---------- attachment encoding ----------


def _attachment_data_url(path: Path) -> tuple[str, str]:
    """Return (mime, data_url) for a file. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"attachment not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        # Conservative defaults — OpenRouter accepts these for image/pdf.
        suffix = path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".pdf": "application/pdf",
        }.get(suffix)
        if mime is None:
            raise ValueError(f"cannot infer MIME for {path}; pass an image/* or application/pdf")
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return (mime, f"data:{mime};base64,{encoded}")


def _build_user_content(
    text: str,
    images: list[Path] | None,
    pdfs: list[Path] | None,
) -> str | list[dict[str, Any]]:
    """Build the user message content. Returns plain string when no attachments."""
    if not images and not pdfs:
        return text
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for img in images or []:
        _mime, data_url = _attachment_data_url(img)
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
    for pdf in pdfs or []:
        mime, data_url = _attachment_data_url(pdf)
        if mime != "application/pdf":
            raise ValueError(f"--pdf expects application/pdf, got {mime} for {pdf}")
        parts.append(
            {
                "type": "file",
                "file": {"filename": pdf.name, "file_data": data_url},
            }
        )
    return parts


# ---------- request shaping ----------


CODE_SYSTEM = (
    "You are a concise, expert software engineer. Answer directly. "
    "Show working code when helpful. Use current best practices. Keep prose tight."
)
GENERAL_SYSTEM = (
    "You are a helpful, concise assistant. Answer directly. "
    "Use markdown for code blocks but keep prose tight. No filler."
)


def _build_chat_payload(
    *,
    question: str,
    history: list[dict] | None,
    model: str,
    system: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    images: list[Path] | None = None,
    pdfs: list[Path] | None = None,
    web: bool = False,
    web_engine: str | None = None,
    web_results: int | None = None,
    think: bool = False,
    effort: str | None = None,
    reasoning_tokens: int | None = None,
    fallback_models: list[str] | None = None,
    route_fallback: bool = False,
    transforms: list[str] | None = None,
    allow_fallbacks: bool = False,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": _build_user_content(question, images, pdfs)})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "provider": {**CHAT_PROVIDER_DEFAULTS, "allow_fallbacks": allow_fallbacks},
    }

    if fallback_models:
        payload["models"] = fallback_models
    if route_fallback:
        payload["route"] = "fallback"
    if transforms:
        payload["transforms"] = transforms

    if web:
        plugin: dict[str, Any] = {"id": "web"}
        if web_engine:
            plugin["engine"] = web_engine
        if web_results is not None:
            plugin["max_results"] = web_results
        payload["plugins"] = [plugin]

    if think:
        reasoning: dict[str, Any] = {"enabled": True}
        if reasoning_tokens is not None:
            reasoning["max_tokens"] = reasoning_tokens
        elif effort is not None:
            reasoning["effort"] = effort
        else:
            reasoning["effort"] = "high"
        payload["reasoning"] = reasoning

    return payload


def _chat_request(
    *,
    question: str,
    history: list[dict] | None,
    model: str,
    system: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    dry: bool,
    show_reasoning: bool = False,
    **payload_extras: Any,
) -> dict:
    """Send a chat completion. payload_extras forwards image/web/think/etc. flags."""
    import httpx

    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    payload = _build_chat_payload(
        question=question,
        history=history,
        model=model,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream and not dry,
        **payload_extras,
    )

    if dry:
        return {"status": "dry_run", "request": payload}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/jbot-bit/canompx3",
        "X-Title": "canompx3-ask",
    }
    url = f"{base_url}/chat/completions"
    started = time.perf_counter()

    if stream:
        return _stream_chat(url, headers, payload, started, show_reasoning=show_reasoning, model=model)

    response = httpx.post(url, headers=headers, json=payload, timeout=180.0)
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]["message"]
    return {
        "status": "completed",
        "result": choice.get("content", ""),
        "reasoning": choice.get("reasoning") or "",
        "annotations": choice.get("annotations") or [],
        "usage": data.get("usage", {}),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "model": data.get("model") or model,
    }


def _stream_chat(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    started: float,
    *,
    show_reasoning: bool,
    model: str,
) -> dict:
    import httpx

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict = {}
    response_model = model
    in_reasoning_block = False

    with httpx.stream("POST", url, headers=headers, json=payload, timeout=180.0) as response:
        if response.status_code >= 400:
            body = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {body[:500]}")
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            response_model = chunk.get("model") or response_model
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {}) or {}
                # Reasoning content can arrive as either delta.reasoning (string)
                # or delta.reasoning_details (list of {type, text}). Render both.
                rtext = delta.get("reasoning")
                rdetails = delta.get("reasoning_details") or []
                rtext_combined = ""
                if isinstance(rtext, str):
                    rtext_combined += rtext
                for detail in rdetails:
                    if isinstance(detail, dict) and isinstance(detail.get("text"), str):
                        rtext_combined += detail["text"]
                if rtext_combined:
                    reasoning_parts.append(rtext_combined)
                    if show_reasoning:
                        if not in_reasoning_block:
                            sys.stderr.write("\033[2m[reasoning] ")
                            in_reasoning_block = True
                        sys.stderr.write(rtext_combined)
                        sys.stderr.flush()

                content_delta = delta.get("content")
                if content_delta:
                    if in_reasoning_block:
                        sys.stderr.write("\033[0m\n")
                        sys.stderr.flush()
                        in_reasoning_block = False
                    sys.stdout.write(content_delta)
                    sys.stdout.flush()
                    text_parts.append(content_delta)
            if chunk.get("usage"):
                usage = chunk["usage"]
    if in_reasoning_block:
        sys.stderr.write("\033[0m\n")
        sys.stderr.flush()
    sys.stdout.write("\n")
    return {
        "status": "completed",
        "result": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "usage": usage,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "model": response_model,
    }


# ---------- grounded path (bounded research runtime) ----------


def _run_grounded(question: str, profile_id: str, max_turns: int, dry: bool, schema: str | None) -> dict:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.paths import GOLD_DB_PATH
    from trading_app.ai.openrouter_runtime import run_openrouter_task

    return run_openrouter_task(
        task_text=question,
        profile_id=profile_id,
        root=PROJECT_ROOT,
        db_path=GOLD_DB_PATH,
        schema_name=schema,
        max_turns=max_turns,
        execute=not dry,
    )


# ---------- output ----------


def _print_friendly(envelope: dict) -> None:
    status = envelope.get("status", "unknown")
    if status == "completed":
        result = envelope.get("result", "")
        if isinstance(result, str) and result.strip():
            sys.stdout.write(result.rstrip() + "\n")
            footer = []
            usage = envelope.get("usage") or {}
            pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            ct = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            if pt or ct:
                footer.append(f"tokens: {pt}+{ct}")
            tools = envelope.get("tool_history") or []
            if tools:
                footer.append("tools: " + ", ".join(t.get("name", "?") for t in tools))
            if envelope.get("turns"):
                footer.append(f"turns: {envelope['turns']}")
            if envelope.get("model"):
                footer.append(f"model: {envelope['model']}")
            ms = envelope.get("latency_ms") or 0
            if ms:
                footer.append(f"{ms / 1000:.1f}s")
            if footer:
                sys.stderr.write("--- " + " | ".join(footer) + "\n")
            return
    if status == "max_turns_exceeded":
        sys.stderr.write(f"[WARN] max-turns ({envelope.get('turns')}) hit.\n")
        for entry in envelope.get("tool_history", []):
            sys.stderr.write(f"  - {entry.get('name')}({entry.get('arguments')})\n")
        return
    if status == "dry_run":
        sys.stdout.write("[dry-run] Request preview:\n\n")
        sys.stdout.write(json.dumps(envelope.get("request", {}), indent=2, sort_keys=True) + "\n")
        return
    sys.stdout.write(json.dumps(envelope, indent=2, sort_keys=True) + "\n")


# ---------- REPL history persistence ----------


def _history_path(name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_") or "session"
    return HISTORY_DIR / f"{safe}.jsonl"


def _save_history(name: str, history: list[dict]) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _history_path(name)
    with path.open("w", encoding="utf-8") as fh:
        for entry in history:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def _load_history(name: str) -> list[dict]:
    path = _history_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no saved history at {path}")
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        out.append(json.loads(raw))
    return out


def _list_histories() -> list[str]:
    if not HISTORY_DIR.exists():
        return []
    return sorted(p.stem for p in HISTORY_DIR.glob("*.jsonl"))


# ---------- REPL ----------


def _repl(args: argparse.Namespace, mode: str, profile_id: str | None, model: str, system: str) -> int:
    sys.stderr.write(
        f"[ask] interactive | mode={mode} | model={model}\n"
        "[ask] /clear  /model <id>  /system <text>  /mode chat|code|plan|long\n"
        "       /save <name>  /load <name>  /history  /quit\n\n"
    )
    history: list[dict] = []
    while True:
        try:
            sys.stderr.write("ask> ")
            sys.stderr.flush()
            line = input()
        except (EOFError, KeyboardInterrupt):
            sys.stderr.write("\n")
            return 0
        line = line.strip()
        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            return 0
        if line == "/clear":
            history.clear()
            sys.stderr.write("[ask] history cleared\n")
            continue
        if line.startswith("/model "):
            model = line[len("/model ") :].strip()
            sys.stderr.write(f"[ask] model -> {model}\n")
            continue
        if line.startswith("/system "):
            system = line[len("/system ") :].strip()
            sys.stderr.write("[ask] system updated\n")
            continue
        if line.startswith("/mode "):
            new_mode = line[len("/mode ") :].strip()
            if new_mode in {"chat", "code"}:
                mode = new_mode
                profile_id = None
                system = CODE_SYSTEM if new_mode == "code" else GENERAL_SYSTEM
                sys.stderr.write(f"[ask] mode -> {mode}\n")
                continue
            if new_mode in {"plan", "long", "extract"}:
                mode = "grounded"
                profile_id = _resolve_profile(new_mode)
                sys.stderr.write(f"[ask] mode -> grounded ({profile_id})\n")
                continue
            sys.stderr.write("[ask] mode must be: chat | code | plan | long | extract\n")
            continue
        if line.startswith("/save "):
            name = line[len("/save ") :].strip() or datetime.now(UTC).strftime("session-%Y%m%dT%H%M%S")
            path = _save_history(name, history)
            sys.stderr.write(f"[ask] saved {len(history)} turns -> {path}\n")
            continue
        if line.startswith("/load "):
            name = line[len("/load ") :].strip()
            try:
                history = _load_history(name)
                sys.stderr.write(f"[ask] loaded {len(history)} turns from {name}\n")
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                sys.stderr.write(f"[ask] load failed: {exc}\n")
            continue
        if line == "/history":
            saved = _list_histories()
            if not saved:
                sys.stderr.write(f"[ask] no saved sessions in {HISTORY_DIR}\n")
            else:
                sys.stderr.write("[ask] saved sessions: " + ", ".join(saved) + "\n")
            continue
        try:
            if mode in {"chat", "code"}:
                envelope = _chat_request(
                    question=line,
                    history=history,
                    model=model,
                    system=system,
                    temperature=args.temp,
                    max_tokens=args.max_tokens,
                    stream=not args.no_stream,
                    dry=args.dry,
                    show_reasoning=args.think,
                    images=[Path(p) for p in (args.image or [])],
                    pdfs=[Path(p) for p in (args.pdf or [])],
                    web=args.web,
                    web_engine=args.web_engine,
                    web_results=args.web_results,
                    think=args.think,
                    effort=args.effort,
                    reasoning_tokens=args.reasoning_tokens,
                    fallback_models=_split_csv(args.models_fallback),
                    route_fallback=args.route_fallback,
                    transforms=_split_csv(args.transforms),
                    allow_fallbacks=args.allow_fallbacks,
                )
                if envelope.get("status") == "completed":
                    history.append({"role": "user", "content": line})
                    history.append({"role": "assistant", "content": envelope.get("result", "")})
            else:
                envelope = _run_grounded(line, profile_id or "deepseek_planning", args.max_turns, args.dry, args.schema)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            sys.stderr.write(f"[ERROR] {type(exc).__name__}: {exc}\n")
            continue
        if mode in {"chat", "code"} and not args.no_stream and envelope.get("status") == "completed":
            usage = envelope.get("usage") or {}
            pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            ct = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            ms = envelope.get("latency_ms") or 0
            footer = []
            if pt or ct:
                footer.append(f"tokens: {pt}+{ct}")
            if envelope.get("model"):
                footer.append(f"model: {envelope['model']}")
            if ms:
                footer.append(f"{ms / 1000:.1f}s")
            if footer:
                sys.stderr.write("--- " + " | ".join(footer) + "\n")
        else:
            _print_friendly(envelope)
        sys.stderr.write("\n")


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    out = [item.strip() for item in value.split(",") if item.strip()]
    return out or None


def _read_stdin_if_pipe(args: argparse.Namespace) -> str:
    """Return the question, optionally appending stdin contents when --pipe is set."""
    typed = " ".join(args.question).strip() if args.question else ""
    if not args.pipe:
        return typed
    if sys.stdin.isatty():
        # User said --pipe but there's no piped input. Honor --pipe but warn.
        sys.stderr.write("[ask] --pipe set but stdin is a tty; using positional question only\n")
        return typed
    piped = sys.stdin.read().rstrip()
    if typed and piped:
        return f"{typed}\n\n{piped}"
    return typed or piped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask",
        description=(
            "ask — operator CLI for OpenRouter. Free chat by default; "
            "--repo/--plan/--long/--extract for bounded grounded research."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  ask "explain a python decorator"\n'
            "  ask --code \"review this regex r'\\d+'\"\n"
            '  ask --web "latest 4.7 release notes"\n'
            '  ask --think --effort high "design a lock-free queue"\n'
            '  ask --image chart.png "what regime is this?"\n'
            '  cat error.log | ask --pipe --code "what failed?"\n'
            '  ask --plan "scale prop firms"\n'
            "  ask --models --tools-only --min-ctx 100000\n"
            "  ask                                        # interactive REPL\n"
        ),
    )
    parser.add_argument("question", nargs="*")

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--code", action="store_true", help="Coding system prompt (chat).")
    grp.add_argument("--repo", action="store_true", help="Repo-grounded research (long-context profile).")
    grp.add_argument("--plan", action="store_true", help="Planning profile (grounded).")
    grp.add_argument("--long", action="store_true", help="Long-context research profile (grounded).")
    grp.add_argument("--extract", action="store_true", help="Structured extraction profile (grounded).")

    parser.add_argument("--model", help="Any OpenRouter model id (chat mode).")
    parser.add_argument("--system", help="Custom system prompt (chat mode).")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=4, help="Tool-loop turns (grounded modes).")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--dry", action="store_true", help="Preview request without API call.")
    parser.add_argument("--raw", action="store_true", help="Full JSON envelope.")
    parser.add_argument("--schema", help="Structured-output schema (extract mode).")

    parser.add_argument("--pipe", action="store_true", help="Read stdin and use as / append to question.")

    parser.add_argument("--web", action="store_true", help="Enable real-time web search (OpenRouter web plugin).")
    parser.add_argument("--web-engine", choices=["native", "exa", "firecrawl", "parallel"], help="Web search engine.")
    parser.add_argument("--web-results", type=int, help="Max web results (default 5).")

    reasoning_grp = parser.add_mutually_exclusive_group()
    reasoning_grp.add_argument(
        "--reasoning-tokens",
        type=int,
        help="Reasoning budget in tokens (Anthropic/Gemini style). Implies --think.",
    )
    parser.add_argument("--think", action="store_true", help="Enable reasoning tokens.")
    parser.add_argument(
        "--effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort (OpenAI/Grok style). Default high when --think is set.",
    )

    parser.add_argument("--image", action="append", help="Attach image (repeatable).")
    parser.add_argument("--pdf", action="append", help="Attach PDF (repeatable).")

    parser.add_argument("--models-fallback", help="Comma-separated fallback model ids.")
    parser.add_argument(
        "--route-fallback",
        action="store_true",
        help="Set route=fallback (let OpenRouter pick fallback if primary fails).",
    )
    parser.add_argument(
        "--transforms",
        help="Comma-separated transforms (e.g. middle-out for context compression).",
    )
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help="Allow OpenRouter provider fallbacks (default deny for safety).",
    )

    parser.add_argument("--models", action="store_true", help="List models passing capability gates per profile.")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh model catalog cache.")
    parser.add_argument("--tools-only", action="store_true", help="--models filter: tools support.")
    parser.add_argument("--reasoning-only", action="store_true", help="--models filter: reasoning support.")
    parser.add_argument("--min-ctx", type=int, help="--models filter: minimum context length.")
    parser.add_argument("--max-prompt-cost", type=float, help="--models filter: max $/M prompt tokens.")
    parser.add_argument("--max-completion-cost", type=float, help="--models filter: max $/M completion tokens.")
    parser.add_argument("--provider-filter", help="--models filter: substring match on model id.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    _load_dotenv(PROJECT_ROOT)
    _ensure_default_models()

    # --reasoning-tokens implies --think (mutually exclusive with --effort handled below).
    if args.reasoning_tokens is not None:
        args.think = True
    if args.think and args.effort is not None and args.reasoning_tokens is not None:
        sys.stderr.write("[ask] --effort and --reasoning-tokens are mutually exclusive\n")
        return 2

    if args.models:
        if err := _check_api_key():
            sys.stderr.write(err)
            return 3
        return _show_models(args)

    if args.repo or args.long:
        mode = "grounded"
        profile_id = _resolve_profile("long")
    elif args.plan:
        mode = "grounded"
        profile_id = _resolve_profile("plan")
    elif args.extract:
        mode = "grounded"
        profile_id = _resolve_profile("extract")
    elif args.code:
        mode = "code"
        profile_id = None
    else:
        mode = "chat"
        profile_id = None

    if not args.dry and (err := _check_api_key()):
        sys.stderr.write(err)
        return 3

    model = args.model or _resolve_chat_model()
    system = args.system or (CODE_SYSTEM if mode == "code" else GENERAL_SYSTEM)
    question = _read_stdin_if_pipe(args)

    if not question:
        return _repl(args, mode, profile_id, model, system)

    try:
        if mode in {"chat", "code"}:
            sys.stderr.write(f"[ask] {mode} | {model}\n")
            envelope = _chat_request(
                question=question,
                history=None,
                model=model,
                system=system,
                temperature=args.temp,
                max_tokens=args.max_tokens,
                stream=not args.no_stream,
                dry=args.dry,
                show_reasoning=args.think,
                images=[Path(p) for p in (args.image or [])],
                pdfs=[Path(p) for p in (args.pdf or [])],
                web=args.web,
                web_engine=args.web_engine,
                web_results=args.web_results,
                think=args.think,
                effort=args.effort,
                reasoning_tokens=args.reasoning_tokens,
                fallback_models=_split_csv(args.models_fallback),
                route_fallback=args.route_fallback,
                transforms=_split_csv(args.transforms),
                allow_fallbacks=args.allow_fallbacks,
            )
            if not args.no_stream and not args.raw and envelope.get("status") == "completed":
                usage = envelope.get("usage") or {}
                pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                ct = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                ms = envelope.get("latency_ms") or 0
                footer = []
                if pt or ct:
                    footer.append(f"tokens: {pt}+{ct}")
                if envelope.get("model"):
                    footer.append(f"model: {envelope['model']}")
                if ms:
                    footer.append(f"{ms / 1000:.1f}s")
                if footer:
                    sys.stderr.write("--- " + " | ".join(footer) + "\n")
                return 0
        else:
            sys.stderr.write(
                f"[ask] grounded | profile={profile_id} | "
                f"model={os.environ.get('CANOMPX3_AI_' + profile_id.upper() + '_MODEL', '?')}\n"
            )
            sys.stderr.write("[ask] thinking...\n")
            envelope = _run_grounded(question, profile_id, args.max_turns, args.dry, args.schema)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"[ERROR] {type(exc).__name__}: {exc}\n")
        return 4

    if args.raw:
        sys.stdout.write(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    else:
        _print_friendly(envelope)
    return 0 if envelope.get("status") in {"completed", "dry_run"} else 1


if __name__ == "__main__":
    _ensure_repo_python()
    raise SystemExit(main())
