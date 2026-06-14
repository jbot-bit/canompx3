#!/usr/bin/env python3
"""PreToolUse advisory: verify session-time claims in a prereg YAML against the
canonical `pipeline.dst` resolvers.

WHY THIS EXISTS
---------------
A prereg in 2026-06-14 miscited `LONDON_METALS` as "10-11am Brisbane". The
canonical resolver `london_open_brisbane` returns 17:00 (UK summer/BST) / 18:00
(UK winter/GMT) AEST. The Phase-2 audit
(`scripts/audits/phase_2_infra_config.py`) only checks that the session *name*
is in SESSION_CATALOG — it never checks the Brisbane-*time* claim, so the
miscite passed. This hook guards the NEXT such miscite (a future regression). It
is ADVISORY ONLY — it warns on stderr, never blocks or edits; the operator
decides. The current v3 MGC prereg cites times from the resolver docstrings and
passes this hook clean — the hook does not imply v3 is flawed.

WHAT IT DOES
------------
On a Write/Edit to `docs/audit/hypotheses/*.yaml`, it scans the body for
"<SESSION_KEY> ... <time>" claims, resolves each key against BOTH sample days
(northern-summer + northern-winter) via the real resolver, and emits a
`[SESSION-TIME-VERIFY]` stderr line on a genuine mismatch.

FAIL DIRECTION (deliberate, two-sided)
--------------------------------------
- The RESOLVER CALL is fail-CLOSED: a raising resolver surfaces a DEFECT line
  (a silently-broken resolver would let bad claims through). It is caught per
  session so it never crashes the operator's edit.
- EVERYTHING ELSE is fail-OPEN: unreadable stdin, missing fields, non-prereg
  path, import failure, parse errors -> exit 0, silent. The module-bottom guard
  rewrites any unexpected BaseException to exit 0. We never block an edit we
  cannot fully understand.

FALSE-POSITIVE GUARDS (proven against the real v3 prereg)
---------------------------------------------------------
1. AMBIGUOUS-LINE SKIP: a line with >=2 distinct SESSION keys AND >=2 distinct
   time tokens is a catalog-style note (e.g. v3 line 58: 4 keys + 7 times). We
   cannot bind a time to a key on such a line without cross-pairing, so we treat
   it as too-ambiguous-to-verify -> documented no-op. This is a deliberate skip,
   NOT a silent gap.
2. ADJACENCY BINDING: a time binds to a key only if it appears within a tight
   window AFTER the key (same clause, before the next session key, <= ADJ_WINDOW
   chars). v3 line 369 puts `SINGAPORE_OPEN` near `11:30` (a DROPPED non-catalog
   session); a too-wide window would false-fire. The window + next-key boundary
   keep unrelated tokens out.
3. MINUTE PRECISION: claims compare on (hour, minute), not hour-only —
   COMEX_SETTLE resolves to :30 (03:30/04:30), so an hour-only match would wrongly
   pass "COMEX 3:00". A claim that omits minutes matches on hour and is annotated.
4. SLASH/RANGE PAIRS: `17:00/18:00` and `08:00 summer / 09:00 winter` parse to
   BOTH endpoints; a claim is OK iff EVERY claimed (h,m) matches one of the two
   resolver tuples.

Patterned on `bias-grounding-guard.py` (stdin json + cooldown + stderr emit) and
`autopilot-tier-guard.py` (PROJECT_ROOT + sys.path.insert + SystemExit-aware
fail-open).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = Path(__file__).parent / ".prereg-session-time-verify-state.json"
COOLDOWN_MINUTES = 20

# Two sample days: northern-summer + northern-winter. The resolver carries the
# source-market DST arithmetic (CDT/EDT/BST) and any next-calendar-day roll —
# we NEVER infer a season label or assume "June == first value".
_SAMPLE_SUMMER = date(2024, 6, 15)
_SAMPLE_WINTER = date(2024, 12, 15)

# A time binds to a key only within this many chars after the key (and before
# the next session key). Tuned against the real v3 prereg lines.
ADJ_WINDOW = 40

# Matches the prereg path on either separator, anywhere in the path.
_PREREG_RE = re.compile(r"docs/audit/hypotheses/[^/]+\.yaml$")

# A clock time: 17:00, 5pm, 10am, 08:00, 11. Captures hour, optional minute,
# optional am/pm. Bare integers like "11" are accepted only when an am/pm or a
# colon is present elsewhere in the token group (guarded at parse time).
_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
    re.IGNORECASE,
)


def _load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        state = {}
    state.setdefault("last_key", None)
    state.setdefault("last_at", None)
    return state


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass  # state is best-effort; never break the advisory on a write error


def _should_emit(state: dict, key: str) -> bool:
    if key != state.get("last_key"):
        return True
    last_at = state.get("last_at")
    if not last_at:
        return True
    try:
        age_min = (datetime.now(UTC) - datetime.fromisoformat(last_at)).total_seconds() / 60
    except (TypeError, ValueError):
        return True
    return age_min >= COOLDOWN_MINUTES


def _is_prereg_path(file_path: str) -> bool:
    norm = file_path.replace("\\", "/")
    return bool(_PREREG_RE.search(norm))


def _body_from_event(tool_input: dict, file_path: str) -> str | None:
    """Body source: Write content, Edit new_string, else the existing file.

    Any read error -> None (caller treats as no-op, fail-open)."""
    if tool_input.get("content"):
        return tool_input["content"]
    if tool_input.get("new_string"):
        return tool_input["new_string"]
    try:
        return (PROJECT_ROOT / file_path.replace("\\", "/")).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _parse_times(segment: str) -> set[tuple[int, int | None]]:
    """Extract claimed (hour, minute|None) tuples from a text segment.

    A bare integer (no colon, no am/pm) is NOT a time claim — it is far more
    often a count, a confirm_bars value, a page number, etc. We require either a
    `:MM` or an am/pm marker for the token to count."""
    claims: set[tuple[int, int | None]] = set()
    for m in _TIME_RE.finditer(segment):
        hour_s, min_s, ampm = m.group(1), m.group(2), m.group(3)
        if min_s is None and ampm is None:
            continue  # bare integer — not a time claim
        hour = int(hour_s)
        minute = int(min_s) if min_s is not None else None
        if ampm:
            ap = ampm.lower()
            if ap == "pm" and hour != 12:
                hour += 12
            elif ap == "am" and hour == 12:
                hour = 0
        if 0 <= hour <= 23 and (minute is None or 0 <= minute <= 59):
            claims.add((hour, minute))
    return claims


def _find_session_positions(line: str, keys: list[str]) -> list[tuple[int, str]]:
    """All (start_index, key) occurrences of any SESSION key on the line, sorted."""
    hits: list[tuple[int, str]] = []
    for key in keys:
        for m in re.finditer(re.escape(key), line):
            hits.append((m.start(), key))
    hits.sort()
    return hits


def _resolver_tuples(entry: dict) -> tuple[set[tuple[int, int]], str | None]:
    """Resolve a session on both sample days. fail-CLOSED: a raise -> DEFECT.

    Returns (set_of_(h,m)_tuples, defect_msg|None)."""
    resolver = entry.get("resolver")
    if resolver is None:
        return set(), "no resolver in SESSION_CATALOG entry"
    try:
        summer = resolver(_SAMPLE_SUMMER)
        winter = resolver(_SAMPLE_WINTER)
    except BaseException as exc:  # fail-CLOSED: surface a broken resolver
        return set(), f"resolver raised: {type(exc).__name__}: {exc}"
    out: set[tuple[int, int]] = set()
    for tup in (summer, winter):
        try:
            h, mi = int(tup[0]), int(tup[1])
        except (TypeError, ValueError, IndexError):
            return set(), f"resolver returned non-(h,m): {tup!r}"
        out.add((h, mi))
    return out, None


def _claim_matches(claimed: tuple[int, int | None], resolver_set: set[tuple[int, int]]) -> bool:
    h, mi = claimed
    if mi is None:
        return any(rh == h for rh, _ in resolver_set)  # hour-only claim
    return (h, mi) in resolver_set


def _fmt(resolver_set: set[tuple[int, int]]) -> str:
    return " / ".join(f"{h:02d}:{m:02d}" for h, m in sorted(resolver_set))


def _fmt_claim(c: tuple[int, int | None]) -> str:
    h, mi = c
    return f"{h:02d}:{mi:02d}" if mi is not None else f"{h:02d}:??"


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # fail-open on unreadable input

    tool_input = event.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path", "") or ""
    if not _is_prereg_path(file_path):
        sys.exit(0)  # rule 4: path gate — only prereg YAMLs

    body = _body_from_event(tool_input, file_path)
    if not body:
        sys.exit(0)  # rule 5: read/parse failure stays no-op

    # Canonical import — fail-OPEN if it fails (cannot make a safe decision).
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from pipeline.dst import SESSION_CATALOG
    except BaseException:
        sys.exit(0)

    keys = list(SESSION_CATALOG.keys())
    warnings: list[str] = []

    for line in body.splitlines():
        positions = _find_session_positions(line, keys)
        if not positions:
            continue

        distinct_keys = {k for _, k in positions}
        all_times = _parse_times(line)

        # Guard 1: ambiguous catalog-style line (>=2 keys AND >=2 times) -> skip.
        if len(distinct_keys) >= 2 and len(all_times) >= 2:
            continue

        for idx, (start, key) in enumerate(positions):
            # Guard 2: bind times only in the window AFTER the key, before the
            # next session key on the line.
            seg_start = start + len(key)
            next_start = positions[idx + 1][0] if idx + 1 < len(positions) else len(line)
            seg_end = min(seg_start + ADJ_WINDOW, next_start)
            segment = line[seg_start:seg_end]
            claimed = _parse_times(segment)
            if not claimed:
                continue

            resolver_set, defect = _resolver_tuples(SESSION_CATALOG[key])
            if defect is not None:
                warnings.append(
                    f"[SESSION-TIME-VERIFY] {key}: resolver DEFECT — {defect}. "
                    "Cannot verify the Brisbane-time claim; fix the resolver."
                )
                continue

            mismatched = [c for c in claimed if not _claim_matches(c, resolver_set)]
            if not mismatched:
                continue  # all claimed times match a resolver output

            claim_str = ", ".join(_fmt_claim(c) for c in sorted(mismatched, key=lambda c: (c[0], c[1] or 0)))
            min_note = ""
            if any(mi is None for _, mi in mismatched):
                min_note = " (minutes unspecified in claim)"
            warnings.append(
                f"[SESSION-TIME-VERIFY] {key}: claimed {claim_str} Bris, resolver "
                f"returns {_fmt(resolver_set)} (summer/winter) — MISMATCH{min_note}. "
                "Read the resolver docstring, not the SESSION_CATALOG entry."
            )

    if not warnings:
        sys.exit(0)

    key = " || ".join(warnings)
    state = _load_state()
    if not _should_emit(state, key):
        sys.exit(0)

    for w in warnings:
        print(w, file=sys.stderr)

    state["last_key"] = key
    state["last_at"] = datetime.now(UTC).isoformat()
    _save_state(state)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # main()'s deliberate sys.exit(0) — propagate the intended code.
        raise
    except BaseException:  # pragma: no cover — fail-open on any unexpected error
        sys.exit(0)
