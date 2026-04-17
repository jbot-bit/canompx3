#!/usr/bin/env python3
"""AI coaching digest engine — generates session analysis via Claude API.

Reads broker_trades.jsonl, trader_profile.json, and daily_features.
Generates coaching digest + profile update patch.

Stage 3 of claude-api-modernization:
  - Canonical `claude_client` (Opus 4.7 reasoning model)
  - `messages.parse(output_format=DigestResponseSchema)` — no more regex
    markdown-fencing, no raw JSON parsing
  - Adaptive thinking (coaching is reasoning-heavy)
  - Typed Anthropic exceptions (RateLimitError / APIStatusError / APIConnectionError)
    replace the bare `except Exception` silent-failure path

Usage:
    python scripts/tools/coaching_digest.py                    # today's session
    python scripts/tools/coaching_digest.py --date 2026-03-06  # specific date
"""

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

from scripts.tools.coaching_prompts import (  # noqa: E402
    BEHAVIORAL_PATTERNS,
    COACHING_RULES,
    EMOTION_CATEGORIES,
    TENDLER_FRAMEWORK,
    TRADE_GRADING_RUBRIC,
)
from trading_app.ai.claude_client import CLAUDE_REASONING_MODEL, get_client  # noqa: E402

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROFILE_PATH = DATA_DIR / "trader_profile.json"
TRADES_PATH = DATA_DIR / "broker_trades.jsonl"
DIGESTS_PATH = DATA_DIR / "coaching_digests.jsonl"
TRADING_RULES_PATH = PROJECT_ROOT / "TRADING_RULES.md"

SYSTEM_PROMPT = f"""\
You are a trading performance coach grounded in Tendler's Mental Game of Trading framework.

{TENDLER_FRAMEWORK}

{TRADE_GRADING_RUBRIC}

{BEHAVIORAL_PATTERNS}

{EMOTION_CATEGORIES}

{COACHING_RULES}"""


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base model enforcing closed schema — required for Claude structured outputs."""

    model_config = {"extra": "forbid"}


class TradeGrade(_StrictModel):
    trade_id: str
    grade: str  # A | B | C | D | F
    zone: str   # A-game | B-game | C-game
    reason: str


class PatternObserved(_StrictModel):
    name: str
    emotion: str
    severity: int
    evidence: str


class Metrics(_StrictModel):
    trades: int
    win_rate: float
    gross_pnl: float
    fees: float
    net_pnl: float
    a_game_pct: float
    c_game_pct: float


class Digest(_StrictModel):
    summary: str
    trade_grades: list[TradeGrade] = Field(default_factory=list)
    patterns_observed: list[PatternObserved] = Field(default_factory=list)
    coaching_note: str
    metrics: Metrics


class Inchworm(_StrictModel):
    c_game_patterns: list[str] = Field(default_factory=list)
    b_game_patterns: list[str] = Field(default_factory=list)
    a_game_indicators: list[str] = Field(default_factory=list)


class TraitEntry(_StrictModel):
    trait: str
    confidence: float
    evidence_count: int


class BehavioralPatternEntry(_StrictModel):
    pattern: str
    emotion: str
    trigger: str
    frequency: str
    avg_cost_r: float


class ProfilePatch(_StrictModel):
    inchworm: Inchworm = Field(default_factory=Inchworm)
    strengths: list[TraitEntry] = Field(default_factory=list)
    growth_edges: list[TraitEntry] = Field(default_factory=list)
    behavioral_patterns: list[BehavioralPatternEntry] = Field(default_factory=list)


class DigestResponseSchema(_StrictModel):
    digest: Digest
    profile_patch: ProfilePatch = Field(default_factory=ProfilePatch)


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
    parts.append(
        "Generate the session digest and profile patch. "
        "Respond with the structured digest object — no prose wrapper."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Profile patching
# ---------------------------------------------------------------------------


def apply_profile_patch(profile: dict, patch: dict) -> None:
    """Merge a profile patch into the existing profile. Mutates in-place."""
    if not patch:
        return

    changed = False

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

    for dict_field in ("inchworm", "emotional_profile"):
        if dict_field not in patch:
            continue
        existing = profile.setdefault(dict_field, {})
        for key, value in patch[dict_field].items():
            if isinstance(value, list) and isinstance(existing.get(key), list):
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
    """Generate a coaching digest via Claude API.

    Returns the digest dict (already enriched with metadata) or None on
    recoverable failure. All error paths are explicit and logged — no bare
    `except Exception`.
    """
    profile = load_trader_profile(path=profile_path)
    rules_excerpt = ""
    if TRADING_RULES_PATH.exists():
        rules_excerpt = TRADING_RULES_PATH.read_text(encoding="utf-8")[:2000]

    user_prompt = build_digest_prompt(profile, trades, rules_excerpt)

    try:
        client = get_client()
    except ValueError as exc:
        logger.error("ANTHROPIC_API_KEY missing: %s", exc)
        return None

    try:
        response = client.messages.parse(
            model=CLAUDE_REASONING_MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=DigestResponseSchema,
            cache_control={"type": "ephemeral"},
        )
    except anthropic.BadRequestError as exc:
        logger.error("Claude BadRequestError (malformed request or schema): %s", exc)
        return None
    except anthropic.AuthenticationError as exc:
        logger.error("Claude AuthenticationError (bad API key): %s", exc)
        return None
    except anthropic.RateLimitError as exc:
        logger.warning("Claude RateLimitError — digest skipped: %s", exc)
        return None
    except anthropic.APIStatusError as exc:
        logger.error("Claude APIStatusError (status=%s): %s", exc.status_code, exc)
        return None
    except anthropic.APIConnectionError as exc:
        logger.warning("Claude APIConnectionError — digest skipped: %s", exc)
        return None

    parsed: DigestResponseSchema = response.parsed_output
    digest = parsed.digest.model_dump()
    patch = parsed.profile_patch.model_dump()

    version_before = profile.get("version", 1)
    apply_profile_patch(profile, patch)
    save_trader_profile(profile, path=profile_path)

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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate coaching digest")
    parser.add_argument("--date", help="Date to analyze (YYYY-MM-DD), default today")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()

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
