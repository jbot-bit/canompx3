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
from datetime import UTC, datetime
from pathlib import Path

STATE_FILE = Path(__file__).parent / ".data-first-state.json"
DIRECTIVE_COOLDOWN_MINUTES = 15

# Keywords that signal an investigation (user wants data, not code reading)
INVESTIGATION_KEYWORDS = re.compile(
    r"\b("
    r"check|investigate|why is|why are|why does|why did|what.?s happening|"
    r"what happened|how many|mismatch|diverge|divergence|wrong|bug|"
    r"real data|actual number|empirical|verify|count|trade count|"
    r"sample size|performance|how bad|magnitude|compare.*actual|"
    r"query first|data first|doesn.?t add up|looks? wrong|off|"
    r"doesn.?t make sense|weird|something.?s off|numbers"
    r")\b",
    re.IGNORECASE,
)

# Keywords for trading queries — must use build_live_portfolio(), not validated_setups
TRADING_QUERY_KEYWORDS = re.compile(
    r"\b("
    r"what do i trade|what.?s live|my trades|my playbook|my portfolio|"
    r"trade tonight|trading tonight|what.?s on tonight|"
    r"morning trades|evening trades|active strategies|"
    r"what am i trading|current positions|my strats|my book|"
    r"show me my stuff|what.?s deployed|what.?s running"
    r")\b",
    re.IGNORECASE,
)

# Keywords for session time queries — must run generate_trade_sheet.py, not mental math
SESSION_TIME_KEYWORDS = re.compile(
    r"\b("
    r"what time|when does|when is|session time|trade time|"
    r"tonight.?s session|what.?s on at|schedule tonight|"
    r"session start|when.*open|when.*close"
    r")\b",
    re.IGNORECASE,
)

TRADING_QUERY_DIRECTIVE = (
    "TRADING QUERY: Use trading_app.live_config/LIVE_PORTFOLIO, not validated_setups."
)

SESSION_TIME_DIRECTIVE = (
    "SESSION TIME: Use scripts/tools/generate_trade_sheet.py; do not do manual timezone math."
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
    r"where are we|what.?s the status|orient|what.?s broken|"
    r"state of|health check|what needs doing|what.?s next"
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
    "DESIGN MODE: Plan and options only; do not edit until the user explicitly switches to implementation."
)

IMPLEMENT_DIRECTIVE = (
    "IMPLEMENT MODE: Execute now. If non-trivial, require STAGE_STATE blast_radius/scope first."
)

COMMIT_DIRECTIVE = (
    "GIT OPERATION: Execute directly; stage intentionally, commit clearly, push only if asked."
)

RESEARCH_DIRECTIVE = (
    "RESEARCH MODE: Use STRATEGY_BLUEPRINT sequence; every claim needs source layer, N, p-value, K, and WFE."
)

ORIENT_DIRECTIVE = (
    "ORIENT: Reground from HANDOFF.md, recent git history, active stage state, and live status commands."
)

RESUME_DIRECTIVE = (
    "RESUME: Re-read HANDOFF.md, recent git history, and active stage state before continuing."
)

# Skill-specific routing (extension of auto-skill-routing.md table).
# Each (regex, directive) fires additively when a prompt matches — emitted alongside
# the mode directives above. These cover skills not implied by mode alone.
# No overlap with DESIGN/IMPLEMENT/COMMIT/RESEARCH/ORIENT/RESUME routes.
SKILL_ROUTES = [
    (
        re.compile(r"\b(my book|my portfolio|what.?s live|my trades|trade tonight|my playbook|my strats|what am i trading|show me my stuff|what.?s deployed|what.?s running)\b", re.IGNORECASE),
        "ROUTE: /trade-book — deployed portfolio with full strategy details.",
    ),
    (
        re.compile(r"\b(how.?s it going|performing|decay|regime|fitness|healthy|anything dying|portfolio health)\b", re.IGNORECASE),
        "ROUTE: /regime-check — portfolio fitness and regime health.",
    ),
    (
        re.compile(r"\b(didn.?t we test|wasn.?t that dead|what did we find|remind me|history of|no.?go\?|past research)\b", re.IGNORECASE),
        "ROUTE: /pinecone-assistant — project history and prior findings.",
    ),
    (
        re.compile(r"\b(review|check my work|bloomey|seven sins|before i commit|anything wrong|code review)\b", re.IGNORECASE),
        "ROUTE: /code-review — institutional review (seven sins, canonical integrity).",
    ),
    (
        re.compile(r"\b(next|keep going|continue|what now|what.?s next|more)\b", re.IGNORECASE),
        "ROUTE: /next — auto-determine next concrete task from stage/handoff/queue.",
    ),
    (
        re.compile(r"\b(stage done|task done|complete|finished|that.?s it|all done)\b", re.IGNORECASE),
        "ROUTE: /verify done — stage acceptance (lint/types/gates).",
    ),
]

WARN_THRESHOLD = 4    # After N consecutive Reads, warn
BLOCK_THRESHOLD = 12  # After N consecutive Reads, BLOCK (raised from 7 — research sessions legitimately read more)

INVESTIGATION_DIRECTIVE = (
    "DATA FIRST: Query data before reading more code. Get numbers first, then explain."
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
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state.setdefault("investigation_mode", False)
    state.setdefault("consecutive_reads", 0)
    state.setdefault("last_updated", None)
    state.setdefault("last_prompt_directive_key", None)
    state.setdefault("last_prompt_directive_at", None)
    return state


def save_state(state):
    state["last_updated"] = datetime.now(UTC).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _directive_key(directives):
    return " || ".join(directives)


def _should_emit_directives(state, directives):
    if not directives:
        return False
    key = _directive_key(directives)
    if key != state.get("last_prompt_directive_key"):
        return True
    last_at = state.get("last_prompt_directive_at")
    if not last_at:
        return True
    try:
        age_min = (_now_utc() - datetime.fromisoformat(last_at)).total_seconds() / 60
    except (ValueError, TypeError):
        return True
    return age_min >= DIRECTIVE_COOLDOWN_MINUTES


def handle_user_prompt(event):
    """Detect user intent and inject appropriate workflow directive."""
    prompt = event.get("prompt", "")
    state = load_state()

    directives = []
    matched_investigation = bool(INVESTIGATION_KEYWORDS.search(prompt))
    matched_commit = bool(COMMIT_KEYWORDS.search(prompt))
    matched_implement = bool(IMPLEMENT_KEYWORDS.search(prompt) and not DESIGN_KEYWORDS.search(prompt))
    matched_design = bool(DESIGN_KEYWORDS.search(prompt))
    matched_research = bool(RESEARCH_KEYWORDS.search(prompt))
    matched_resume = bool(RESUME_KEYWORDS.search(prompt))
    matched_orient = bool(ORIENT_KEYWORDS.search(prompt))

    # ── Data investigation detection (existing) ──────────────────────
    if matched_investigation:
        state["investigation_mode"] = True
        state["consecutive_reads"] = 0
        directives.append(INVESTIGATION_DIRECTIVE)
    elif matched_commit or matched_implement or matched_design or matched_research or matched_resume or matched_orient:
        state["investigation_mode"] = False
        state["consecutive_reads"] = 0

    if TRADING_QUERY_KEYWORDS.search(prompt):
        directives.append(TRADING_QUERY_DIRECTIVE)

    if SESSION_TIME_KEYWORDS.search(prompt):
        directives.append(SESSION_TIME_DIRECTIVE)

    # ── Intent routing (new) ─────────────────────────────────────────
    # Priority: commit > implement > design > research > resume > orient
    # (most specific wins — commit is unambiguous, design/implement need priority)
    if matched_commit:
        directives.append(COMMIT_DIRECTIVE)
    elif matched_implement:
        directives.append(IMPLEMENT_DIRECTIVE)
    elif matched_design:
        directives.append(DESIGN_DIRECTIVE)

    if matched_research:
        directives.append(RESEARCH_DIRECTIVE)

    if matched_resume:
        directives.append(RESUME_DIRECTIVE)
    elif matched_orient:
        directives.append(ORIENT_DIRECTIVE)

    # ── Skill routing (additive — emits specific /skill nudges) ──────
    for pattern, directive in SKILL_ROUTES:
        if pattern.search(prompt):
            directives.append(directive)

    # ── Emit directives ──────────────────────────────────────────────
    if _should_emit_directives(state, directives):
        state["last_prompt_directive_key"] = _directive_key(directives)
        state["last_prompt_directive_at"] = _now_utc().isoformat()
        save_state(state)
        print("\n".join(directives), file=sys.stderr)
    else:
        save_state(state)

    sys.exit(0)


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
        # PDFs are research assets (literature, papers). Reading them is expected
        # during research and should not count against the code-read budget.
        read_path = event.get("tool_input", {}).get("file_path", "")
        if read_path.lower().endswith(".pdf"):
            sys.exit(0)
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
    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as exc:
        print(f"[data-first-guard] unexpected: {exc}", file=sys.stderr)
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
