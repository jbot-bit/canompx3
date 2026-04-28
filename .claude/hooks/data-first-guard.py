#!/usr/bin/env python3
"""Data-First Guard — PreToolUse(Read|Bash) only.

Tracks consecutive Read calls without an intervening data query.
Warns at WARN_THRESHOLD, blocks at BLOCK_THRESHOLD when in
investigation mode.

The UserPromptSubmit branch (intent classification + investigation-mode
toggle) was merged into `prompt-broker.py` on 2026-04-27. This file
keeps only the tool-use enforcement path. The broker writes to the
same `state/data-first.json` file, so PreToolUse here observes the
investigation-mode flag and consecutive-reads counter the broker sets.

State file path moved from `.claude/hooks/.data-first-state.json` to
`.claude/hooks/state/data-first.json` as part of the broker merge.
"""

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / ".claude" / "hooks" / "state" / "data-first.json"


# Bash commands that count as "querying data" (resets the read counter)
QUERY_PATTERNS = re.compile(
    r"python\s+-c|python\s+.*\.py|duckdb|sqlite|SELECT\s|"
    r"pipeline[./]|trading_app[./]|scripts[./]",
    re.IGNORECASE,
)

# Pipeline rebuild commands — need FK constraint and ordering reminder
REBUILD_PATTERNS = re.compile(
    r"build_daily_features|build_bars_5m|ingest_dbn|"
    r"outcome_builder|strategy_discovery|strategy_validator|"
    r"run_pipeline|rebuild",
    re.IGNORECASE,
)

REBUILD_DIRECTIVE = (
    "PIPELINE REBUILD DETECTED. Before running, verify:\n"
    "1. FK constraints: orb_outcomes -> daily_features -> bars_5m -> bars_1m. "
    "Cannot DELETE upstream while downstream references it.\n"
    "2. Rebuild order: ingest -> bars_5m -> daily_features -> outcomes -> discovery -> validator -> edge_families\n"
    "3. For daily_features column changes: USE UPDATE (not DELETE+INSERT) to avoid FK violations.\n"
    "4. For full rebuilds: delete DOWNSTREAM first (outcomes), then rebuild upstream -> downstream.\n"
    "5. Lesson 15: init_db BEFORE daily_features when adding sessions."
)

WARN_THRESHOLD = 4    # After N consecutive Reads, warn
BLOCK_THRESHOLD = 7   # After N consecutive Reads, BLOCK

WARN_MESSAGE = (
    "DATA FIRST WARNING: You have read {count} files without running a single query. "
    "If you are investigating a data question, STOP reading code and RUN A QUERY. "
    "The data shows WHAT and HOW BAD. Code only shows WHY."
)

BLOCK_MESSAGE = (
    "DATA FIRST BLOCK: {count} consecutive file reads with zero queries. "
    "This Read is BLOCKED. Run a python -c query or Bash command to check the data first. "
    "If this is purely a code task (not a data question), "
    "run: python -c \"print('not a data investigation')\" to reset the counter."
)


def load_state():
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state.setdefault("investigation_mode", False)
    state.setdefault("consecutive_reads", 0)
    state.setdefault("last_updated", None)
    return state


def save_state(state):
    state["last_updated"] = datetime.now(UTC).isoformat()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _now_utc() -> datetime:
    return datetime.now(UTC)


def handle_pre_tool_use(event):
    """Track Read vs Bash calls. Warn or block on too many consecutive Reads."""
    tool_name = event.get("tool_name", "")
    state = load_state()

    # Auto-expire investigation mode after 30 minutes of no updates
    if state.get("last_updated"):
        try:
            last = datetime.fromisoformat(state["last_updated"])
            age_min = (_now_utc() - last).total_seconds() / 60
            if age_min > 30:
                state["investigation_mode"] = False
                state["consecutive_reads"] = 0
        except (ValueError, TypeError):
            pass

    if tool_name == "Bash":
        command = event.get("tool_input", {}).get("command", "")
        if QUERY_PATTERNS.search(command):
            # Query detected — reset counter, clear investigation urgency
            state["consecutive_reads"] = 0
            save_state(state)
        # Pipeline rebuild warning — inject ordering reminder
        if REBUILD_PATTERNS.search(command):
            print(REBUILD_DIRECTIVE, file=sys.stderr)
        # Don't block Bash calls
        sys.exit(0)

    elif tool_name == "Read":
        state["consecutive_reads"] = state.get("consecutive_reads", 0) + 1
        count = state["consecutive_reads"]
        save_state(state)

        if state.get("investigation_mode"):
            # In investigation mode — stricter thresholds
            if count >= BLOCK_THRESHOLD:
                print(
                    BLOCK_MESSAGE.format(count=count),
                    file=sys.stderr,
                )
                sys.exit(2)  # BLOCK the Read
            elif count >= WARN_THRESHOLD:
                print(
                    WARN_MESSAGE.format(count=count),
                    file=sys.stderr,
                )
                sys.exit(0)  # Warn but allow
        else:
            # Not in investigation mode — just track, don't block
            # But still warn at higher threshold (catches unmarked investigations)
            if count >= BLOCK_THRESHOLD + 3:  # 10 reads without any query
                print(
                    f"REMINDER: {count} consecutive file reads without a query. "
                    f"If this is a data investigation, run a query.",
                    file=sys.stderr,
                )
                sys.exit(0)  # Warn only, don't block outside investigation mode

        sys.exit(0)

    else:
        # Other tools (Grep, Glob, etc.) — don't count, don't reset
        sys.exit(0)


def main():
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    hook_event = event.get("hook_event_name", "")

    # UserPromptSubmit handling moved to prompt-broker.py on 2026-04-27.
    # This script only handles PreToolUse now.
    if hook_event == "PreToolUse":
        handle_pre_tool_use(event)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
