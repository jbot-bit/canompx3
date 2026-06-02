#!/usr/bin/env python3
"""Compact Codex prompt broker for UserPromptSubmit.

Combines risk, bias, intent, and stage context into one bounded
`additionalContext` payload. Each classifier is isolated so one failure does
not suppress the others. State is kept under `.codex/hooks/state/`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / ".codex" / "hooks" / "state"
STAGES_DIR = PROJECT_ROOT / "docs" / "runtime" / "stages"
STAGE_STATE_FILE = PROJECT_ROOT / "docs" / "runtime" / "STAGE_STATE.md"
GLOBAL_CAP = 1500
MAX_STAGE_PREVIEW_CHARS = 180

RISK_COOLDOWN = timedelta(minutes=20)
BIAS_COOLDOWN = timedelta(minutes=20)
INTENT_COOLDOWN = timedelta(minutes=5)
STAGE_COOLDOWN = timedelta(minutes=20)

RISK_CRITICAL_RE = re.compile(
    r"\b("
    r"real capital|live trading|deploy|production|promotion|promote|"
    r"broker|order routing|position sizing|risk limit|kill switch|"
    r"account routing|account safety|capital review|deploy readiness|"
    r"readiness|live path|runtime control|threat model|security review"
    r")\b",
    re.IGNORECASE,
)

RISK_HIGH_RE = re.compile(
    r"\b("
    r"pipeline|check_drift|schema|migration|duckdb|database|timezone|dst|"
    r"session boundary|session time|orb window|concurrency|worktree|hook|"
    r"mutex|review|audit|verify|validation|backtest|research|hypothesis|"
    r"holdout|oos|p.?value|fdr|slippage|cost model|execution engine|"
    r"trading_app/live|refresh_data|outcome_builder"
    r")\b",
    re.IGNORECASE,
)

BIAS_RESEARCH_RE = re.compile(
    r"\b("
    r"research|review|audit|verify|validation|validate|bias|ground|"
    r"source|sources|literature|evidence|proof|prove|claim|claims|"
    r"result|results|deploy|promotion|promote|ready|readiness|"
    r"oos|holdout|backtest|significance|p.?value|fdr|dsr|sharpe"
    r")\b",
    re.IGNORECASE,
)

INTENT_DESIGN_RE = re.compile(
    r"\b(brainstorm|design|plan|approach|proposal|propose)\b",
    re.IGNORECASE,
)
INTENT_COMMIT_RE = re.compile(
    r"\b(commit|push|merge|ship|publish)\b",
    re.IGNORECASE,
)

RISK_CRITICAL_MSG = (
    "[risk:critical] RISK TIER: critical. Keep exploration lean; reserve "
    "high-reasoning or an independent review for final decisions. Require "
    "execution evidence before done."
)
RISK_HIGH_MSG = (
    "[risk:high] RISK TIER: high. Default to normal reasoning for "
    "exploration, then escalate only for review/decision points. Require "
    "targeted tests, drift, and explicit review."
)
BIAS_RESEARCH_MSG = (
    "[bias:research] RESEARCH MODE: Canon only, disconfirm first, tag "
    "MEASURED/INFERRED/UNSUPPORTED, then state edge, issue, next step."
)
INTENT_DESIGN_MSG = (
    "[intent:design] DESIGN MODE: state what/why, files, blast radius, and "
    "approach before edits."
)
INTENT_COMMIT_MSG = (
    "[intent:commit] GIT OPERATION: verify tests, diff, and review before "
    "committing or pushing."
)


@dataclass(frozen=True)
class BrokerFragment:
    """One candidate line plus the dedup state needed to emit it."""

    key: str
    value: str
    cooldown: timedelta
    priority: int
    text: str


def _state_path(key: str) -> Path:
    return STATE_DIR / f"{key}.json"


def _load_state(key: str) -> dict[str, str | None]:
    try:
        payload = json.loads(_state_path(key).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "last_value": payload.get("last_value"),
        "last_at": payload.get("last_at"),
    }


def _save_state(key: str, value: str) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        _state_path(key).write_text(
            json.dumps(
                {
                    "last_value": value,
                    "last_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _parse_when(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _state_allows_emit(key: str, value: str, cooldown: timedelta) -> bool:
    state = _load_state(key)
    if state.get("last_value") != value:
        return True
    last_at = _parse_when(state.get("last_at"))
    if last_at is None:
        return True
    return datetime.now(UTC) - last_at >= cooldown


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _parse_field(text: str, field: str) -> str | None:
    prefix = f"{field}:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix.lower()):
            value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def _parse_blast_radius(text: str) -> str | None:
    if "## Blast Radius" in text:
        section = text.split("## Blast Radius", 1)[1]
        section = section.split("\n## ", 1)[0].split("\n---", 1)[0]
        cleaned = _normalize(section)
        return cleaned or None
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped.lower().startswith("blast_radius:"):
            continue
        value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
        if value:
            return _normalize(value)
        items: list[str] = []
        for follow in text.splitlines()[idx + 1 :]:
            s_follow = follow.strip()
            if s_follow.startswith("- "):
                items.append(s_follow[2:].strip())
                continue
            if s_follow:
                break
        if items:
            return _normalize("; ".join(items))
    return None


def _iter_stage_candidates() -> list[Path]:
    candidates: list[Path] = []
    if STAGES_DIR.is_dir():
        candidates.extend(sorted(p for p in STAGES_DIR.glob("*.md") if p.name != ".gitkeep"))
    if STAGE_STATE_FILE.exists():
        candidates.append(STAGE_STATE_FILE)
    return candidates


def _classify_risk(prompt: str) -> BrokerFragment | None:
    if RISK_CRITICAL_RE.search(prompt):
        value = "critical"
        if not _state_allows_emit("prompt-broker-risk", value, RISK_COOLDOWN):
            return None
        return BrokerFragment(
            key="prompt-broker-risk",
            value=value,
            cooldown=RISK_COOLDOWN,
            priority=0,
            text=RISK_CRITICAL_MSG,
        )
    if RISK_HIGH_RE.search(prompt):
        value = "high"
        if not _state_allows_emit("prompt-broker-risk", value, RISK_COOLDOWN):
            return None
        return BrokerFragment(
            key="prompt-broker-risk",
            value=value,
            cooldown=RISK_COOLDOWN,
            priority=0,
            text=RISK_HIGH_MSG,
        )
    return None


def _classify_bias(prompt: str) -> BrokerFragment | None:
    value = "research"
    if not BIAS_RESEARCH_RE.search(prompt):
        return None
    if not _state_allows_emit("prompt-broker-bias", value, BIAS_COOLDOWN):
        return None
    return BrokerFragment(
        key="prompt-broker-bias",
        value=value,
        cooldown=BIAS_COOLDOWN,
        priority=1,
        text=BIAS_RESEARCH_MSG,
    )


def _classify_intent(prompt: str) -> BrokerFragment | None:
    if INTENT_COMMIT_RE.search(prompt):
        value = "commit"
        if not _state_allows_emit("prompt-broker-intent", value, INTENT_COOLDOWN):
            return None
        return BrokerFragment(
            key="prompt-broker-intent",
            value=value,
            cooldown=INTENT_COOLDOWN,
            priority=2,
            text=INTENT_COMMIT_MSG,
        )
    if INTENT_DESIGN_RE.search(prompt):
        value = "design"
        if not _state_allows_emit("prompt-broker-intent", value, INTENT_COOLDOWN):
            return None
        return BrokerFragment(
            key="prompt-broker-intent",
            value=value,
            cooldown=INTENT_COOLDOWN,
            priority=2,
            text=INTENT_DESIGN_MSG,
        )
    return None


def _classify_stage(_prompt: str) -> BrokerFragment | None:
    preferred: list[tuple[int, Path, str, str]] = []
    for path in _iter_stage_candidates():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        mode = (_parse_field(text, "mode") or "").upper()
        if not mode:
            continue
        task = _parse_field(text, "task") or path.stem
        priority = 0 if mode == "IMPLEMENTATION" else 1 if mode == "DESIGN" else 2
        preferred.append((priority, path, text, task))
    if not preferred:
        return None

    _, path, text, task = sorted(preferred, key=lambda item: (item[0], item[1].name))[0]
    mode = (_parse_field(text, "mode") or path.stem).upper()
    stage = _parse_field(text, "stage")
    stage_of = _parse_field(text, "stage_of")
    blast_radius = _parse_blast_radius(text)
    hash_basis = json.dumps(
        {
            "mode": mode,
            "task": task,
            "stage": stage,
            "stage_of": stage_of,
            "blast_radius": blast_radius,
            "path": str(path),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(hash_basis.encode("utf-8")).hexdigest()
    if not _state_allows_emit("prompt-broker-stage", digest, STAGE_COOLDOWN):
        return None

    parts = [f"[stage:{mode.lower()}] ACTIVE STAGE: {mode}", task]
    if stage and stage_of:
        parts.append(f"({stage}/{stage_of})")
    elif stage:
        parts.append(f"(stage {stage})")
    if blast_radius:
        parts.append(f"blast={_truncate(blast_radius, MAX_STAGE_PREVIEW_CHARS)}")
    text_line = " | ".join(parts)
    return BrokerFragment(
        key="prompt-broker-stage",
        value=digest,
        cooldown=STAGE_COOLDOWN,
        priority=3,
        text=text_line,
    )


def _cap_fragments(fragments: list[BrokerFragment]) -> list[BrokerFragment]:
    selected: list[BrokerFragment] = []
    total = 0
    for fragment in sorted(fragments, key=lambda item: item.priority):
        cost = len(fragment.text) if not selected else len(fragment.text) + 1
        if total + cost > GLOBAL_CAP:
            continue
        selected.append(fragment)
        total += cost
    return selected


def _load_event() -> dict[str, object]:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        raise SystemExit(0) from None
    if not isinstance(payload, dict):
        raise SystemExit(0)
    return payload


def main() -> None:
    event = _load_event()

    hook_event = str(event.get("hook_event_name") or "UserPromptSubmit")
    if hook_event != "UserPromptSubmit":
        raise SystemExit(0)

    prompt = str(event.get("prompt") or "").strip()
    if not prompt:
        raise SystemExit(0)

    fragments: list[BrokerFragment] = []
    for classifier in (_classify_risk, _classify_bias, _classify_intent, _classify_stage):
        try:
            fragment = classifier(prompt)
        except Exception:
            fragment = None
        if fragment is not None:
            fragments.append(fragment)

    selected = _cap_fragments(fragments)
    if not selected:
        raise SystemExit(0)

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(fragment.text for fragment in selected),
        }
    }
    print(json.dumps(payload))

    for fragment in selected:
        _save_state(fragment.key, fragment.value)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
