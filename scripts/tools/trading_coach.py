#!/usr/bin/env python3
"""Interactive AI trading coach — conversational CLI.

Loads trader profile + recent coaching digests into system prompt.
Uses Claude API for conversation.

Usage:
    python scripts/tools/trading_coach.py          # start chat
    python scripts/tools/trading_coach.py --query "Why do I keep losing on Fridays?"
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

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


COACH_SYSTEM_PROMPT = """\
You are a trading performance coach grounded in Tendler's Mental Game of Trading framework.

## Performance Model: The Inchworm
All execution falls into three zones:
- **A-Game**: Learning mistakes only. No emotional interference.
- **B-Game**: Impulse to deviate but controlled it. Minor suboptimality.
- **C-Game**: Emotional hijacking overrode known rules.

Progress = raising the floor (eliminating C-game), not raising the ceiling.

## Emotion Categories (Tendler)
When the trader describes a pattern, classify it:
- **Greed**: Profit-target manipulation, sizing up on winners, can't stop watching PnL
- **Fear**: FOMO (chasing), fear of losing (early exits), fear of mistakes (hesitation), fear of failure
- **Tilt**: Hate-to-lose, mistake tilt (self-anger), injustice tilt, revenge trading, entitlement
- **Confidence**: Overconfidence (ignoring stops, euphoria) OR underconfidence (hesitation, need validation)
- **Discipline**: Impatience, boredom, results-fixation, distractibility

## Intervention Protocols
When identifying an issue, prescribe specific interventions:
- Revenge spiral detected → "Reduce to MINIMUM size for next 3 trades"
- Overconfidence cascade → "Return to base position size. Re-read your rules."
- Fear of losing → "Your stop IS the plan. Entry-to-stop is pre-paid cost of the trade."
- Tilt escalation → "Step away for 90 seconds. Name the emotion. Rate it 1-10."
- Session shutdown → "Take the NEXT valid signal at minimum size. The signal is the trigger, not your feelings."

## Rules
- Never praise without citing specific evidence from the trades.
- Never comfort after losses. Validate the emotion, redirect to process.
- Never conflate outcome with process. A losing A-grade > a winning F-grade.
- Reference the trader's historical patterns from the profile when available.
- Use specific interventions, not vague advice ("take a break").
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


def _get_client_and_profile():
    """Load anthropic client + trader profile. Exits on failure."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic not installed. Run: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    from scripts.tools.coaching_digest import load_trader_profile

    profile = load_trader_profile()
    digests = load_recent_digests()
    system_prompt = build_chat_system_prompt(profile, digests)
    client = anthropic.Anthropic()
    return client, system_prompt


def chat_loop():
    """Run interactive chat with the trading coach."""
    client, system_prompt = _get_client_and_profile()
    messages = []

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

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
            )
            assistant_text = response.content[0].text
            messages.append({"role": "assistant", "content": assistant_text})
            print(f"\nCoach: {assistant_text}")
        except Exception as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            messages.pop()  # remove failed user message


def single_query(query: str):
    """Ask a single question and print the response."""
    client, system_prompt = _get_client_and_profile()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )
        print(response.content[0].text)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AI Trading Coach")
    parser.add_argument("--query", "-q", help="Single question (no interactive mode)")
    args = parser.parse_args()

    if args.query:
        single_query(args.query)
    else:
        chat_loop()


if __name__ == "__main__":
    main()
