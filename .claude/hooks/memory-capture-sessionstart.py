#!/usr/bin/env python3
"""Auto-memory-capture: Claude-visible cue on the next session start.

The model-facing half of the 3-event loop — see
`.claude/rules/auto-memory-capture.md`. Runs as a PARALLEL SIBLING to
session-start.py (different channel: stdout-JSON additionalContext vs
session-start.py's stderr text). Reads ONLY its own breadcrumb — no .claude.pid
dependency, so it is immune to any session-lock format change.

Behavior: read the pending breadcrumb written by the prior SessionEnd. If it is
present, NOT yet consumed, younger than 24h, and the signal counts are nonzero,
emit a CLAUDE-VISIBLE `additionalContext` cue (verified to survive /clear via
the Stage-0 probe seq=2 round-trip) prompting a capture JUDGEMENT, then mark the
breadcrumb consumed (one-shot). Otherwise emit nothing (silent clean start).

Fires on all sources (matcher ""), incl. clear/compact/startup/resume. Reads
both `source` (official key, confirmed by the probe) and `session_type` (in-repo
variant) fail-open. Fail-open everywhere: any error -> exit 0, no stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

import _memory_capture as mc


_CUE_TEMPLATE = (
    "Prior session accrued {summary}. Before proceeding, CONSIDER whether any "
    "durable, non-obvious lesson is worth capturing to memory/*.md (dedup "
    "against existing files; taxonomy user/feedback/project/reference). Capture "
    "nothing if it is obvious, transient, or already documented — this is a cue "
    "to JUDGE, not an instruction to write."
)

# Tier-1: git-PROVEN stale "not-merged" claims. Strong cue — fix the baton.
_STALE_TEMPLATE = (
    "STALE BATON (git-proven): {detail}. The named SHA(s) ARE on origin/main, "
    "yet the baton claims unmerged/not-landed. Before relying on these files, "
    "UPDATE them to reflect the integrated state (and their MEMORY.md index "
    "line). This is a factual contradiction, not a judgement call."
)

# Tier-2: recent project resume-batons asserting live status. Advisory nudge.
_LIVE_TEMPLATE = (
    "LIVE PROJECT BATONS ({n}): {files}. These recent `type: project` batons "
    "assert live status (NEXT/RESUME/OPEN/pending). Confirm each is still "
    "current against git/DB before acting on it — finishing work includes "
    "updating the baton that describes it. Advisory: confirm, do not assume stale."
)

# MEMORY.md HOT-tier budget warning. Fires BEFORE the ~24.4KB load cap so the
# index can be trimmed while there is still headroom — past the cap the loader
# silently truncates and the dropped lines vanish from every session. This cues
# Claude to JUDGE what to demote; it never auto-writes (per auto-memory-capture).
_MEMORY_SIZE_TEMPLATE = (
    "MEMORY.md near load cap ({kb:.1f}KB / {lines} lines — soft limit "
    "{warn_kb:.0f}KB/{warn_lines}). The loader truncates the index around 24.4KB "
    "and silently drops trailing lines from every session. Demote the oldest "
    "CLOSED batons (✅ DONE / ARCHIVABLE) to MEMORY_ARCHIVE.md, leaving a 1-line "
    "pointer only if follow-up is owed; collapse any content-carrying line to a "
    "≤200-char pointer. Cold tier stays searchable via recall/pinecone/`ls`."
)


def _counts_nonzero(counts: dict) -> bool:
    if not isinstance(counts, dict):
        return False
    return mc.signal_meets_threshold(counts)


def _should_emit(crumb: dict | None) -> bool:
    if not isinstance(crumb, dict):
        return False
    if crumb.get("consumed"):
        return False
    if not mc.breadcrumb_is_fresh(crumb):
        return False
    return _counts_nonzero(crumb.get("counts", {}))


def _staleness_context() -> str:
    """Tier-1 + tier-2 baton-drift cues. Empty string if nothing to surface.

    Independent of the capture breadcrumb — runs every SessionStart so a baton
    that went stale (its work merged, or its live status drifted) is flagged
    even on a clean start with no new-capture signal. Fail-open per caller.
    """
    parts: list[str] = []
    try:
        stale = mc.scan_stale_batons()
    except BaseException:
        stale = []
    if stale:
        detail = "; ".join(
            f"{f['file']} claims {', '.join(f['shas'])} unmerged" for f in stale
        )
        parts.append(_STALE_TEMPLATE.format(detail=detail))
    try:
        live = mc.scan_live_project_batons()
    except BaseException:
        live = []
    if live:
        parts.append(_LIVE_TEMPLATE.format(n=len(live), files=", ".join(live)))
    try:
        size = mc.check_memory_index_size()
    except BaseException:
        size = None
    if size:
        parts.append(
            _MEMORY_SIZE_TEMPLATE.format(
                kb=size["bytes"] / 1024.0,
                lines=size["lines"],
                warn_kb=mc._MEMORY_WARN_BYTES / 1024.0,
                warn_lines=mc._MEMORY_WARN_LINES,
            )
        )
    return "\n\n".join(parts)


def main() -> None:
    try:
        json.load(sys.stdin)  # drain stdin; source/session_type unused for gating
    except (json.JSONDecodeError, ValueError):
        pass

    sections: list[str] = []

    # Capture cue (breadcrumb-gated, one-shot, consumed when emitted).
    crumb = mc.read_breadcrumb()
    if _should_emit(crumb):
        sections.append(_CUE_TEMPLATE.format(summary=mc.describe_signals(crumb["counts"])))
        mc.mark_breadcrumb_consumed()

    # Baton-drift cues (run every start, independent of the breadcrumb).
    drift = _staleness_context()
    if drift:
        sections.append(drift)

    if not sections:
        sys.exit(0)  # silent clean start

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "\n\n".join(sections),
                }
            }
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # pragma: no cover - fail-open
        sys.exit(0)
