#!/usr/bin/env python3
"""Data-First Guard + Intent Router: enforces querying data before theorizing
and routes user intent to the correct workflow mode.

Modes based on hook event:

1. UserPromptSubmit: detects intent keywords in user message —
   investigation, trading query, design/implement/commit/research/orient/resume —
   injects the appropriate workflow directive so Claude routes correctly.

2. PreToolUse (Read): tracks consecutive Read calls without a Bash/query call.
   After threshold, warns via stderr. After hard limit, BLOCKS the Read.

3. PreToolUse (Bash): resets the consecutive-read counter (query happened).

State persisted to .claude/hooks/.data-first-state.json between invocations.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

STATE_FILE = Path(__file__).parent / ".data-first-state.json"

# Keywords that signal an investigation (user wants data, not code reading)
INVESTIGATION_KEYWORDS = re.compile(
    r"\b("
    r"check|investigate|why is|why are|why does|why did|what.s happening|"
    r"what happened|how many|mismatch|diverge|divergence|wrong|bug|"
    r"real data|actual number|empirical|verify|count|trade count|"
    r"sample size|performance|how bad|magnitude|compare.*actual|"
    r"query first|data first"
    r")\b",
    re.IGNORECASE,
)

# Keywords for trading queries — must use build_live_portfolio(), not validated_setups
TRADING_QUERY_KEYWORDS = re.compile(
    r"\b("
    r"what do i trade|what.s live|my trades|my playbook|my portfolio|"
    r"trade tonight|trading tonight|what.s on tonight|"
    r"morning trades|evening trades|active strategies|"
    r"what am i trading|current positions"
    r")\b",
    re.IGNORECASE,
)

# Keywords for session time queries — must run generate_trade_sheet.py, not mental math
SESSION_TIME_KEYWORDS = re.compile(
    r"\b("
    r"what time|when does|when is|session time|trade time|"
    r"tonight.s session|what.s on at|schedule tonight|"
    r"session start|when.*open|when.*close"
    r")\b",
    re.IGNORECASE,
)

TRADING_QUERY_DIRECTIVE = (
    "TRADING QUERY DETECTED: Use build_live_portfolio() from trading_app.live_config — "
    "NOT validated_setups. Run: python -c \"from trading_app.live_config import LIVE_PORTFOLIO; "
    "print([s.strategy_id for s in LIVE_PORTFOLIO])\" "
    "Lesson 1: validated_setups ≠ live portfolio. Different tables, different purpose."
)

SESSION_TIME_DIRECTIVE = (
    "SESSION TIME QUERY DETECTED: Do NOT compute timezone math manually. Run: "
    "python scripts/tools/generate_trade_sheet.py — it resolves all times correctly via dst.py. "
    "Lesson 10: Manual timezone math has been wrong EVERY time. Brisbane=UTC+10 + EDT=UTC-4 + mental math = WRONG."
)

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
    "1. FK constraints: orb_outcomes → daily_features → bars_5m → bars_1m. "
    "Cannot DELETE upstream while downstream references it.\n"
    "2. Rebuild order: ingest → bars_5m → daily_features → outcomes → discovery → validator → edge_families\n"
    "3. For daily_features column changes: USE UPDATE (not DELETE+INSERT) to avoid FK violations.\n"
    "4. For full rebuilds: delete DOWNSTREAM first (outcomes), then rebuild upstream → downstream.\n"
    "5. Lesson 15: init_db BEFORE daily_features when adding sessions."
)

# ── Intent routing: detect user's MODE from natural language ─────────
DESIGN_KEYWORDS = re.compile(
    r"\b("
    r"plan|design|think about|brainstorm|how would|how should|what if|"
    r"explore|iterate|4t|approach|architecture|consider|strategy for|"
    r"pros and cons|trade.?offs|options for"
    r")\b",
    re.IGNORECASE,
)

IMPLEMENT_KEYWORDS = re.compile(
    r"\b("
    r"build it|do it|implement|go ahead|ship it|make it happen|just do it|"
    r"write the code|code it|execute|deploy|wire it up|hook it up|"
    r"yes|looks good|approved|lgtm"
    r")\b",
    re.IGNORECASE,
)

COMMIT_KEYWORDS = re.compile(
    r"\b("
    r"commit|push|comit|pusdh|vcommit|merge|commit all|push it|"
    r"stage and commit|git push|commit and push"
    r")\b",
    re.IGNORECASE,
)

RESEARCH_KEYWORDS = re.compile(
    r"\b("
    r"hypothesis|test.*edge|research|validate.*signal|stress test|"
    r"is this real|backtest|forward test|null test|significance|"
    r"p.?value|sharpe|fdr|discover|noise floor"
    r")\b",
    re.IGNORECASE,
)

ORIENT_KEYWORDS = re.compile(
    r"\b("
    r"where are we|what.s the status|orient|what.s broken|"
    r"state of|health check|what needs doing|what.s next"
    r")\b",
    re.IGNORECASE,
)

RESUME_KEYWORDS = re.compile(
    r"\b("
    r"resume|pick up where|last conversation|last session|"
    r"where was i|what were we doing|carry on from|"
    r"it closed|conversation closed|got disconnected|session crashed"
    r")\b",
    re.IGNORECASE,
)

DESIGN_DIRECTIVE = (
    "DESIGN MODE: Do NOT write code. Iterate on the plan. "
    "Present options. Wait for explicit 'go'/'build it'/'implement' before editing files."
)

IMPLEMENT_DIRECTIVE = (
    "IMPLEMENT MODE: User wants code NOW. "
    "If non-trivial, write STAGE_STATE.md first (blast_radius + scope_lock + acceptance). "
    "Then execute. Show evidence when done."
)

COMMIT_DIRECTIVE = (
    "GIT OPERATION: Just execute immediately. No explaining, no asking 'are you sure'. "
    "Check git status, stage files, commit with descriptive message. Push if asked."
)

RESEARCH_DIRECTIVE = (
    "RESEARCH MODE: Open docs/STRATEGY_BLUEPRINT.md. Route through test sequence. "
    "All claims need: source layer, N, p-value, K for BH FDR, WFE. "
    "Default to O5 aperture. Per-session, NEVER pooled."
)

ORIENT_DIRECTIVE = (
    "ORIENT: Check HANDOFF.md + git log --oneline -10 + STAGE_STATE.md + pipeline_status.py. "
    "Report current state from commands, not assumptions."
)

RESUME_DIRECTIVE = (
    "RESUME: Check HANDOFF.md for last state, git log --oneline -10 for recent changes, "
    "STAGE_STATE.md for active work. Verify before continuing."
)

WARN_THRESHOLD = 4    # After N consecutive Reads, warn
BLOCK_THRESHOLD = 7   # After N consecutive Reads, BLOCK

INVESTIGATION_DIRECTIVE = (
    "DATA FIRST (ENFORCED): You are investigating a data question. "
    "QUERY the database or run a computation BEFORE reading more code files. "
    "10 minutes of data beats hours of code reading. "
    "Write a python -c query NOW — do not read another file until you have numbers."
)

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
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "investigation_mode": False,
            "consecutive_reads": 0,
            "last_updated": None,
        }


def save_state(state):
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def handle_user_prompt(event):
    """Detect user intent and inject appropriate workflow directive."""
    prompt = event.get("prompt", "")
    state = load_state()

    directives = []

    # ── Data investigation detection (existing) ──────────────────────
    if INVESTIGATION_KEYWORDS.search(prompt):
        state["investigation_mode"] = True
        state["consecutive_reads"] = 0
        directives.append(INVESTIGATION_DIRECTIVE)

    if TRADING_QUERY_KEYWORDS.search(prompt):
        directives.append(TRADING_QUERY_DIRECTIVE)

    if SESSION_TIME_KEYWORDS.search(prompt):
        directives.append(SESSION_TIME_DIRECTIVE)

    # ── Intent routing (new) ─────────────────────────────────────────
    # Priority: commit > implement > design > research > resume > orient
    # (most specific wins — commit is unambiguous, design/implement need priority)
    if COMMIT_KEYWORDS.search(prompt):
        directives.append(COMMIT_DIRECTIVE)
    elif IMPLEMENT_KEYWORDS.search(prompt) and not DESIGN_KEYWORDS.search(prompt):
        # Only inject implement if NOT also design (avoid false positives)
        directives.append(IMPLEMENT_DIRECTIVE)
    elif DESIGN_KEYWORDS.search(prompt):
        directives.append(DESIGN_DIRECTIVE)

    if RESEARCH_KEYWORDS.search(prompt):
        directives.append(RESEARCH_DIRECTIVE)

    if RESUME_KEYWORDS.search(prompt):
        directives.append(RESUME_DIRECTIVE)
    elif ORIENT_KEYWORDS.search(prompt):
        directives.append(ORIENT_DIRECTIVE)

    # ── Emit directives ──────────────────────────────────────────────
    if directives:
        save_state(state)
        print("\n".join(directives), file=sys.stderr)

    sys.exit(0)


def handle_pre_tool_use(event):
    """Track Read vs Bash calls. Warn or block on too many consecutive Reads."""
    tool_name = event.get("tool_name", "")
    state = load_state()

    # Auto-expire investigation mode after 30 minutes of no updates
    if state.get("last_updated"):
        try:
            last = datetime.fromisoformat(state["last_updated"])
            age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60
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

    if hook_event == "UserPromptSubmit":
        handle_user_prompt(event)
    elif hook_event == "PreToolUse":
        handle_pre_tool_use(event)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
