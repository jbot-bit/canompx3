#!/usr/bin/env python3
"""Stop hook — when the operator asked for a plan this turn, verify the reply
carried a rigor section.

Architecture (v2.1, 2026-05-31): this hook does NOT guess "is this prose a
plan?" from keywords — that is unsolvable, because reports and plans share
vocabulary (it false-fired on its own author's completion report). Instead it
keys on KNOWN intent: the UserPromptSubmit router (`targeted-grounding-router.py`)
drops a per-turn breadcrumb when the PLAN route fires. This hook fires the
advisory ONLY when:
  (a) a plan was requested this turn (breadcrumb present + pending), AND
  (b) the reply lacks a rigor section / 2nd pass / evidence (`audit`).
It then clears the breadcrumb so it never fires on a later unrelated Stop.

The ExitPlanMode gate (`plan-rigor-gate.py`) remains the strong layer for plans
formally presented via the tool; this backstop only covers plans the operator
explicitly asked for but that were written in chat.

Advisory only — Stop hooks cannot cleanly block, and a hard stop on chat output
would be hostile. Fail-open on every error path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BREADCRUMB = HERE / "state" / "plan-intent.json"

sys.path.insert(0, str(HERE))
try:
    from _plan_rigor import audit  # noqa: E402
except Exception:
    sys.exit(0)


def _load_event() -> dict:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig", errors="replace").strip()
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _consume_breadcrumb(session_id: str) -> bool:
    """Return True iff a plan was requested this turn for this session, and clear
    the pending flag. Fail-open: any error returns False (no fire)."""
    try:
        if not BREADCRUMB.exists():
            return False
        data = json.loads(BREADCRUMB.read_text(encoding="utf-8"))
        pending = bool(data.get("pending")) and data.get("session_id") == session_id
        if pending:
            data["pending"] = False
            BREADCRUMB.write_text(json.dumps(data), encoding="utf-8")
        return pending
    except Exception:
        return False


def _last_assistant_text(transcript_path: str) -> str:
    """Pull the text of the most recent assistant message from the JSONL transcript."""
    try:
        p = Path(transcript_path)
        if not p.exists():
            return ""
        last = ""
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "assistant":
                continue
            content = rec.get("message", {}).get("content", [])
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if parts:
                last = "\n".join(parts)
        return last
    except Exception:
        return ""


def main() -> int:
    event = _load_event()
    # Avoid loops: if this Stop was itself triggered by our continuation.
    if event.get("stop_hook_active"):
        return 0
    session_id = str(event.get("session_id") or "")
    # Gate on KNOWN intent: only proceed if the operator asked for a plan this turn.
    if not _consume_breadcrumb(session_id):
        return 0

    text = _last_assistant_text(str(event.get("transcript_path") or ""))
    if not text:
        return 0

    verdict = audit(text)
    if verdict["ok"]:
        return 0

    gaps: list[str] = []
    if verdict["missing_pillars"]:
        gaps.append("missing: " + ", ".join(verdict["missing_pillars"]))
    if not verdict["second_pass"]:
        gaps.append("no 2nd pass shown")
    if verdict["performative"]:
        gaps.append("claims rigor without evidence")

    advisory = (
        "PLAN RIGOR BACKSTOP: you asked for a plan this turn and the reply skipped rigor ("
        + "; ".join(gaps)
        + "). Fold it in on the next turn before the operator acts on it."
    )
    print(json.dumps({"decision": "block", "reason": advisory}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
