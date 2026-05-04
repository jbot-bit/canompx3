#!/usr/bin/env python3
"""ask — operator CLI for OpenRouter / DeepSeek. Two modes:

  CHAT (default)         Free-form. Coding, explanations, anything. No repo grounding.
  REPO  (--repo / --plan / --long / --extract)
                         Bounded read-only research path through the canonical
                         provider_registry + research_packet runtime.

Just type a question. No env ceremony. Streams by default.

Usage:
    ask "what does this regex do: r'\\d+'"      # chat
    ask --code "review this python snippet"     # chat with coding system prompt
    ask --repo "summarize live allocator state" # repo-grounded
    ask --plan "how to scale prop firm setup"   # planning profile (grounded)
    ask --long "what literature backs t>=3.79"  # long-context grounded
    ask --models                                # list models that pass capability gates
    ask                                         # interactive REPL
    ask --model anthropic/claude-3.5-sonnet "..."  # any OpenRouter model
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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

DEFAULT_CHAT_MODEL = "deepseek/deepseek-chat"

CACHE_DIR = PROJECT_ROOT / ".cache"
MODELS_CACHE = CACHE_DIR / "openrouter_models.json"
MODELS_CACHE_TTL_S = 24 * 3600


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


def _fetch_models_cached() -> list[dict]:
    """Returns the OpenRouter /models payload, cached on disk for 24h."""
    import httpx

    if MODELS_CACHE.exists():
        age = time.time() - MODELS_CACHE.stat().st_mtime
        if age < MODELS_CACHE_TTL_S:
            try:
                return json.loads(MODELS_CACHE.read_text(encoding="utf-8")).get("data", [])
            except (json.JSONDecodeError, OSError):
                pass

    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=15.0)
    response.raise_for_status()
    payload = response.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_CACHE.write_text(json.dumps(payload), encoding="utf-8")
    return payload.get("data", [])


def _list_models_for_profile(profile_id: str) -> list[tuple[str, list[str], int | None]]:
    """For a profile, return [(model_id, supported_params, context_length), ...] of models that pass capability gates."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from trading_app.ai.provider_registry import assert_openrouter_research_profile

    profile = assert_openrouter_research_profile(profile_id)
    required = set(profile.required_parameters)
    if profile.response_mode == "json_schema":
        required.update({"response_format", "structured_outputs"})

    models = _fetch_models_cached()
    out: list[tuple[str, list[str], int | None]] = []
    for m in models:
        params = set(m.get("supported_parameters", []))
        if required.issubset(params):
            out.append((m.get("id", ""), sorted(params), m.get("context_length")))
    return sorted(out, key=lambda x: x[0])


def _show_models() -> int:
    """Print which OpenRouter models pass capability gates for each profile."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from trading_app.ai.provider_registry import list_openrouter_research_profiles

    sys.stderr.write("[ask] fetching OpenRouter model catalog (cached 24h)...\n")
    for profile_id in list_openrouter_research_profiles():
        try:
            entries = _list_models_for_profile(profile_id)
        except Exception as exc:
            sys.stdout.write(f"\n## {profile_id}\n  ERROR: {exc}\n")
            continue
        sys.stdout.write(f"\n## {profile_id}  ({len(entries)} compatible)\n")
        env_key = f"CANOMPX3_AI_{profile_id.upper()}_MODEL"
        current = os.environ.get(env_key, "(unset)")
        sys.stdout.write(f"   currently: {current}  (set via {env_key})\n")
        for model_id, _params, ctx in entries[:30]:
            ctx_str = f"  ctx={ctx}" if ctx else ""
            marker = " <-- current" if model_id == current else ""
            sys.stdout.write(f"   - {model_id}{ctx_str}{marker}\n")
        if len(entries) > 30:
            sys.stdout.write(f"   ... +{len(entries) - 30} more\n")
    return 0


# ---------- chat path (free, direct OpenRouter) ----------


CODE_SYSTEM = (
    "You are a concise, expert software engineer. Answer directly. "
    "Show working code when helpful. Use current best practices. Keep prose tight."
)
GENERAL_SYSTEM = (
    "You are a helpful, concise assistant. Answer directly. "
    "Use markdown for code blocks but keep prose tight. No filler."
)


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
) -> dict:
    import httpx

    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream and not dry,
    }
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
        text_parts: list[str] = []
        usage: dict = {}
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
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        text_parts.append(delta)
                if chunk.get("usage"):
                    usage = chunk["usage"]
        sys.stdout.write("\n")
        return {
            "status": "completed",
            "result": "".join(text_parts),
            "usage": usage,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "model": model,
        }

    response = httpx.post(url, headers=headers, json=payload, timeout=180.0)
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]["message"]
    return {
        "status": "completed",
        "result": choice.get("content", ""),
        "usage": data.get("usage", {}),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "model": model,
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


# ---------- REPL ----------


def _repl(args: argparse.Namespace, mode: str, profile_id: str | None, model: str, system: str) -> int:
    sys.stderr.write(
        f"[ask] interactive | mode={mode} | model={model}\n"
        "[ask] /clear  /model <id>  /system <text>  /mode chat|code|plan|long  /quit\n\n"
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
                if new_mode == "code":
                    system = CODE_SYSTEM
                else:
                    system = GENERAL_SYSTEM
                sys.stderr.write(f"[ask] mode -> {mode}\n")
                continue
            if new_mode in {"plan", "long", "extract"}:
                mode = "grounded"
                profile_id = _resolve_profile(new_mode)
                sys.stderr.write(f"[ask] mode -> grounded ({profile_id})\n")
                continue
            sys.stderr.write("[ask] mode must be: chat | code | plan | long | extract\n")
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
                )
                if envelope.get("status") == "completed":
                    history.append({"role": "user", "content": line})
                    history.append({"role": "assistant", "content": envelope.get("result", "")})
            else:
                envelope = _run_grounded(line, profile_id or "deepseek_planning", args.max_turns, args.dry, args.schema)
        except Exception as exc:
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
            if ms:
                footer.append(f"{ms / 1000:.1f}s")
            if footer:
                sys.stderr.write("--- " + " | ".join(footer) + "\n")
        else:
            _print_friendly(envelope)
        sys.stderr.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask",
        description="ask — operator CLI for OpenRouter / DeepSeek. Free chat by default; --repo/--plan/--long/--extract for bounded grounded research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  ask "explain a python decorator"\n'
            "  ask --code \"review this regex r'\\d+'\"\n"
            '  ask --plan "scale prop firms"\n'
            '  ask --long "literature for chordia t>=3.79"\n'
            "  ask --models                              # list capability-passing models\n"
            "  ask                                        # interactive REPL\n"
            '  ask --model anthropic/claude-3.5-sonnet "..."\n'
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
    parser.add_argument("--models", action="store_true", help="List models passing capability gates per profile.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    _load_dotenv(PROJECT_ROOT)
    _ensure_default_models()

    if args.models:
        if err := _check_api_key():
            sys.stderr.write(err)
            return 3
        return _show_models()

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

    model = args.model or DEFAULT_CHAT_MODEL
    system = args.system or (CODE_SYSTEM if mode == "code" else GENERAL_SYSTEM)
    question = " ".join(args.question).strip() if args.question else ""

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
            )
            if not args.no_stream and not args.raw and envelope.get("status") == "completed":
                usage = envelope.get("usage") or {}
                pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                ct = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                ms = envelope.get("latency_ms") or 0
                footer = []
                if pt or ct:
                    footer.append(f"tokens: {pt}+{ct}")
                if ms:
                    footer.append(f"{ms / 1000:.1f}s")
                if footer:
                    sys.stderr.write("--- " + " | ".join(footer) + "\n")
                return 0
        else:
            sys.stderr.write(
                f"[ask] grounded | profile={profile_id} | model={os.environ.get('CANOMPX3_AI_' + profile_id.upper() + '_MODEL', '?')}\n"
            )
            sys.stderr.write("[ask] thinking...\n")
            envelope = _run_grounded(question, profile_id, args.max_turns, args.dry, args.schema)
    except Exception as exc:
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
