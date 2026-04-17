#!/usr/bin/env python3
"""Interactive AI trading coach — conversational CLI.

Loads trader profile + recent coaching digests into system prompt.
Uses Claude API for conversation.

Stage 3 of claude-api-modernization:
  - Canonical `claude_client` (Opus 4.7 reasoning model)
  - Adaptive thinking OFF by default — interactive chat latency matters.
    Set `TRADING_COACH_THINKING=adaptive` to opt in per-session.
  - Cache_control on system prompt (stable across turns — ~90% cost cut)
  - Typed Anthropic exceptions replace silent `except Exception`

Usage:
    python scripts/tools/trading_coach.py          # start chat
    python scripts/tools/trading_coach.py --query "Why do I keep losing on Fridays?"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

from scripts.tools.coaching_prompts import (  # noqa: E402
    COACHING_RULES,
    EMOTION_CATEGORIES,
    INTERVENTION_PROTOCOLS,
    TENDLER_FRAMEWORK,
)
from trading_app.ai.claude_client import CLAUDE_REASONING_MODEL, get_client  # noqa: E402

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROFILE_PATH = DATA_DIR / "trader_profile.json"
DIGESTS_PATH = DATA_DIR / "coaching_digests.jsonl"


def load_recent_digests(n: int = 5, *, path: Path = DIGESTS_PATH) -> list[dict]:
    """Load the last N coaching digests."""
    if not path.exists():
        return []
    digests = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                digests.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return digests[-n:]


COACH_SYSTEM_PROMPT = f"""\
You are a trading performance coach grounded in Tendler's Mental Game of Trading framework.

{TENDLER_FRAMEWORK}

{EMOTION_CATEGORIES}

{INTERVENTION_PROTOCOLS}

{COACHING_RULES}
- Be honest and direct — the trader needs truth, not comfort.
- If asked about something not in the data, say so clearly."""


def build_chat_system_prompt(profile: dict, recent_digests: list[dict]) -> str:
    """Build the system prompt for interactive chat."""
    parts = [COACH_SYSTEM_PROMPT, "", "## Trader Profile"]
    parts.append(f"```json\n{json.dumps(profile, indent=2)}\n```")

    if recent_digests:
        parts.append("\n## Recent Coaching Digests")
        for d in recent_digests:
            parts.append(f"\n### {d.get('date', 'Unknown date')}")
            if d.get("coaching_note"):
                parts.append(d["coaching_note"])
            if d.get("patterns_observed"):
                parts.append(f"Patterns: {json.dumps(d['patterns_observed'])}")
            if d.get("metrics"):
                parts.append(f"Metrics: {json.dumps(d['metrics'])}")

    return "\n".join(parts)


def _thinking_param():
    """Honor TRADING_COACH_THINKING=adaptive for opt-in deep reasoning.

    Default (absent / any other value): thinking off — prioritize interactive
    latency. The user can set `TRADING_COACH_THINKING=adaptive` before launching
    the chat to trade latency for reasoning depth.
    """
    if os.environ.get("TRADING_COACH_THINKING", "").lower() == "adaptive":
        return {"type": "adaptive"}
    return None


def _get_client_and_profile():
    """Load Anthropic client + trader profile. Exits on API-key failure."""
    from scripts.tools.coaching_digest import load_trader_profile

    try:
        client = get_client()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    profile = load_trader_profile()
    digests = load_recent_digests()
    system_prompt = build_chat_system_prompt(profile, digests)
    return client, system_prompt


def _call_coach(client, system_prompt: str, messages: list[dict], *, max_tokens: int = 2048):
    """Single Claude call with typed exception handling.

    Returns response on success, None on recoverable failure (logged).
    """
    kwargs = {
        "model": CLAUDE_REASONING_MODEL,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": messages,
    }
    thinking = _thinking_param()
    if thinking is not None:
        kwargs["thinking"] = thinking

    try:
        return client.messages.create(**kwargs)
    except anthropic.BadRequestError as exc:
        logger.error("Claude BadRequestError: %s", exc)
        print(f"\nERROR: malformed request — {exc}", file=sys.stderr)
    except anthropic.AuthenticationError as exc:
        logger.error("Claude AuthenticationError: %s", exc)
        print(f"\nERROR: bad API key — {exc}", file=sys.stderr)
    except anthropic.RateLimitError as exc:
        logger.warning("Claude RateLimitError: %s", exc)
        print("\nERROR: rate limited — retry shortly.", file=sys.stderr)
    except anthropic.APIStatusError as exc:
        logger.error("Claude APIStatusError (status=%s): %s", exc.status_code, exc)
        print(f"\nERROR: API status {exc.status_code}", file=sys.stderr)
    except anthropic.APIConnectionError as exc:
        logger.warning("Claude APIConnectionError: %s", exc)
        print("\nERROR: connection lost — check network.", file=sys.stderr)
    return None


def _extract_text(response) -> str:
    """Pull the first text block from a response (skips thinking blocks if any)."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def chat_loop():
    """Run interactive chat with the trading coach."""
    client, system_prompt = _get_client_and_profile()
    messages: list[dict] = []

    print("Trading Coach (type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = _call_coach(client, system_prompt, messages)
        if response is None:
            messages.pop()  # roll back failed user turn
            continue

        assistant_text = _extract_text(response)
        messages.append({"role": "assistant", "content": assistant_text})
        print(f"\nCoach: {assistant_text}")


def single_query(query: str):
    """Ask a single question and print the response."""
    client, system_prompt = _get_client_and_profile()

    response = _call_coach(
        client,
        system_prompt,
        [{"role": "user", "content": query}],
    )
    if response is None:
        sys.exit(1)
    print(_extract_text(response))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="AI Trading Coach")
    parser.add_argument("--query", "-q", help="Single question (no interactive mode)")
    args = parser.parse_args()

    if args.query:
        single_query(args.query)
    else:
        chat_loop()


if __name__ == "__main__":
    main()
