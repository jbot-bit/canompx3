#!/usr/bin/env python3
"""AI coaching digest engine — generates session analysis via Claude API.

Reads broker_trades.jsonl, trader_profile.json, and daily_features.
Generates coaching digest + profile update patch.

Usage:
    python scripts/tools/coaching_digest.py                    # today's session
    python scripts/tools/coaching_digest.py --date 2026-03-06  # specific date
"""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = PROJECT_ROOT / "data"
PROFILE_PATH = DATA_DIR / "trader_profile.json"
TRADES_PATH = DATA_DIR / "broker_trades.jsonl"
DIGESTS_PATH = DATA_DIR / "coaching_digests.jsonl"
TRADING_RULES_PATH = PROJECT_ROOT / "TRADING_RULES.md"

from scripts.tools.coaching_prompts import (
    BEHAVIORAL_PATTERNS,
    COACHING_RULES,
    EMOTION_CATEGORIES,
    TENDLER_FRAMEWORK,
    TRADE_GRADING_RUBRIC,
)

SYSTEM_PROMPT = f"""\
You are a trading performance coach grounded in Tendler's Mental Game of Trading framework.

{TENDLER_FRAMEWORK}

{TRADE_GRADING_RUBRIC}

{BEHAVIORAL_PATTERNS}

{EMOTION_CATEGORIES}

{COACHING_RULES}

RESPOND WITH VALID JSON ONLY — no markdown fencing, no commentary outside the JSON."""

DIGEST_SCHEMA = """{
  "digest": {
    "summary": "1-2 sentence session summary",
    "trade_grades": [{"trade_id": "...", "grade": "A|B|C|D|F", "zone": "A-game|B-game|C-game", "reason": "..."}],
    "patterns_observed": [{"name": "revenge_spiral|overconfidence_cascade|fear_of_losing|...", "emotion": "greed|fear|tilt|confidence|discipline", "severity": 1-10, "evidence": "..."}],
    "coaching_note": "2-3 paragraphs of specific coaching feedback with interventions",
    "metrics": {"trades": N, "win_rate": 0.XX, "gross_pnl": X, "fees": X, "net_pnl": X, "a_game_pct": 0.XX, "c_game_pct": 0.XX}
  },
  "profile_patch": {
    "inchworm": {"c_game_patterns": ["..."], "b_game_patterns": ["..."], "a_game_indicators": ["..."]},
    "strengths": [{"trait": "...", "confidence": 0.0-1.0, "evidence_count": N}],
    "growth_edges": [{"trait": "...", "confidence": 0.0-1.0, "evidence_count": N}],
    "behavioral_patterns": [{"pattern": "...", "emotion": "greed|fear|tilt|confidence|discipline", "trigger": "...", "frequency": "...", "avg_cost_r": X}]
  }
}"""


# ---------------------------------------------------------------------------
# Profile I/O
# ---------------------------------------------------------------------------


def load_trader_profile(*, path: Path = PROFILE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "last_updated": "",
        "strengths": [],
        "growth_edges": [],
        "behavioral_patterns": [],
        "goals": [],
        "session_tendencies": {},
        "inchworm": {
            "c_game_patterns": [],
            "b_game_patterns": [],
            "a_game_indicators": [],
        },
        "emotional_profile": {
            "primary_emotion": None,
            "tilt_subtype": None,
            "escalation_speed": None,
            "recovery_pattern": None,
            "tilt_indicators": [],
            "calm_indicators": [],
            "known_triggers": [],
            "effective_interventions": [],
        },
        "account_summary": {},
    }


def save_trader_profile(profile: dict, *, path: Path = PROFILE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile["last_updated"] = date.today().isoformat()
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_digest_prompt(profile: dict, trades: list[dict], trading_rules_excerpt: str = "") -> str:
    """Build the user prompt for the Claude API call."""
    parts = []
    if trading_rules_excerpt:
        parts.append(f"## Trading Rules\n{trading_rules_excerpt[:2000]}")
    parts.append(f"## Current Trader Profile\n```json\n{json.dumps(profile, indent=2)}\n```")
    parts.append(f"## Today's Trades\n```json\n{json.dumps(trades, indent=2)}\n```")
    parts.append(f"## Required Output Schema\n```json\n{DIGEST_SCHEMA}\n```")
    parts.append("Generate the digest and profile patch now. JSON only.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_digest_response(raw: str) -> tuple[dict, dict]:
    """Parse Claude's response into (digest, profile_patch).

    Raises ValueError if response is not valid JSON or missing required keys.
    """
    # Strip markdown fencing if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude returned invalid JSON: {cleaned[:200]}") from exc

    if "digest" not in data:
        raise ValueError(f"Claude response missing 'digest' key. Got keys: {list(data.keys())}")

    return data["digest"], data.get("profile_patch", {})


# ---------------------------------------------------------------------------
# Profile patching
# ---------------------------------------------------------------------------


def apply_profile_patch(profile: dict, patch: dict) -> None:
    """Merge a profile patch into the existing profile. Mutates in-place."""
    if not patch:
        return

    changed = False

    # Merge list fields (strengths, growth_edges, behavioral_patterns)
    for list_field in ("strengths", "growth_edges", "behavioral_patterns"):
        if list_field not in patch:
            continue
        existing = profile.setdefault(list_field, [])
        for new_item in patch[list_field]:
            key_field = "trait" if list_field != "behavioral_patterns" else "pattern"
            match = next(
                (e for e in existing if e.get(key_field) == new_item.get(key_field)),
                None,
            )
            if match:
                match.update(new_item)
            else:
                existing.append(new_item)
            changed = True

    # Merge dict fields (inchworm, emotional_profile) — shallow merge of sub-keys
    for dict_field in ("inchworm", "emotional_profile"):
        if dict_field not in patch:
            continue
        existing = profile.setdefault(dict_field, {})
        for key, value in patch[dict_field].items():
            if isinstance(value, list) and isinstance(existing.get(key), list):
                # Deduplicate list values (e.g. c_game_patterns)
                combined = existing[key] + [v for v in value if v not in existing[key]]
                existing[key] = combined
            else:
                existing[key] = value
            changed = True

    if changed:
        profile["version"] = profile.get("version", 0) + 1


# ---------------------------------------------------------------------------
# Digest generation
# ---------------------------------------------------------------------------


def generate_digest(trades: list[dict], *, profile_path: Path = PROFILE_PATH) -> dict | None:
    """Generate a coaching digest via Claude API. Returns the digest dict or None on failure."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return None

    profile = load_trader_profile(path=profile_path)
    rules_excerpt = ""
    if TRADING_RULES_PATH.exists():
        rules_excerpt = TRADING_RULES_PATH.read_text(encoding="utf-8")[:2000]

    user_prompt = build_digest_prompt(profile, trades, rules_excerpt)

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:
        print(f"ERROR: Claude API call failed: {exc}")
        return None

    raw_text = response.content[0].text
    try:
        digest, patch = parse_digest_response(raw_text)
    except ValueError as exc:
        print(f"ERROR: Failed to parse Claude response: {exc}")
        return None

    # Apply profile patch
    version_before = profile.get("version", 1)
    apply_profile_patch(profile, patch)
    save_trader_profile(profile, path=profile_path)

    # Enrich digest with metadata
    digest["date"] = trades[0]["entry_time"][:10] if trades else date.today().isoformat()
    digest["accounts"] = list({t.get("account_name", "") for t in trades})
    digest["profile_version_before"] = version_before
    digest["profile_version_after"] = profile.get("version", version_before)

    return digest


def save_digest(digest: dict, *, path: Path = DIGESTS_PATH) -> None:
    """Append digest to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(digest) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate coaching digest")
    parser.add_argument("--date", help="Date to analyze (YYYY-MM-DD), default today")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()

    # Load trades for the target date
    if not TRADES_PATH.exists():
        print(f"No trades file at {TRADES_PATH}. Run trade_matcher.py first.")
        return

    all_trades = []
    for line in TRADES_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            all_trades.append(json.loads(line))

    day_trades = [t for t in all_trades if t.get("entry_time", "").startswith(target_date)]
    if not day_trades:
        print(f"No trades found for {target_date}")
        return

    print(f"Generating digest for {target_date} ({len(day_trades)} trades)...")
    digest = generate_digest(day_trades)
    if digest:
        save_digest(digest)
        print(f"\nDigest saved to {DIGESTS_PATH}")
        print(f"Coaching note:\n{digest.get('coaching_note', 'N/A')}")
    else:
        print("Digest generation failed.")


if __name__ == "__main__":
    main()
