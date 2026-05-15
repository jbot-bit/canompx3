"""Shared context-budget tier computation for Claude Code hooks.

Reads the active session transcript path from the hook stdin payload, estimates
token count via byte-size / 4 heuristic, and maps to a discrete tier 1-5 that
both `context-gauge.py` (UserPromptSubmit) and `subagent-budget-guard.py`
(PreToolUse:Task) use as their canonical signal.

Fail-open contract: ANY error (missing file, unreadable, malformed payload)
returns (1, 0). The hooks must never block a session they cannot measure.
"""

from __future__ import annotations

import os
from pathlib import Path

# Tier thresholds as % of model context cap. Conservative because the 4-bytes-
# per-token heuristic is ±20%. Hard-block (Layer 2) fires at Tier 4.
_TIER_THRESHOLDS_PCT = (20, 60, 100, 150)  # T2, T3, T4, T5 floors
_DEFAULT_CAP_TOKENS = 200_000  # Sonnet 4.6 default; override via CLAUDE_CTX_CAP


def _cap_tokens() -> int:
    try:
        return int(os.environ.get("CLAUDE_CTX_CAP", _DEFAULT_CAP_TOKENS))
    except (TypeError, ValueError):
        return _DEFAULT_CAP_TOKENS


def estimate_tokens(transcript_path: str | None) -> int | None:
    if not transcript_path:
        return None
    try:
        size = Path(transcript_path).stat().st_size
    except OSError:
        return None
    return size // 4


def current_tier(payload: dict) -> tuple[int, int]:
    """Return (tier, pct_used). Fail-open: any error -> (1, 0)."""
    try:
        tokens = estimate_tokens(payload.get("transcript_path"))
        if tokens is None:
            return (1, 0)
        pct = int((tokens / _cap_tokens()) * 100)
        tier = 1
        for i, floor in enumerate(_TIER_THRESHOLDS_PCT, start=2):
            if pct >= floor:
                tier = i
        return (tier, pct)
    except Exception:
        return (1, 0)
