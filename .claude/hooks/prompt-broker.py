#!/usr/bin/env python3
"""Single UserPromptSubmit entry point — merges 4 prior guards.

Replaces these standalone hooks (each previously fired its own Python process):
  - risk-tier-guard.py
  - scripts/tools/bias_grounding_guard.py
  - data-first-guard.py (UserPromptSubmit branch only; PreToolUse stays separate)
  - stage-awareness.py

Why merge:
  4 process startups -> 1; 4 separate `additionalContext` envelopes -> 1
  consolidated envelope; per-prompt token leak cut by ~75-90% under typical
  load. See PR #157 audit for the measurement.

Safety guarantees preserved:
  - Each subroutine's classification regex/logic copied verbatim.
  - Per-guard dedup state files in `.claude/hooks/state/<guard>.json` —
    no global silence gate. A risk-tier escalation cannot be swallowed
    by another guard's recent fire.
  - Each subroutine wrapped in try/except BaseException -> isolated
    failure (one bug in one classifier never kills the others).
  - Token budget allocator orders by safety priority
    (risk > bias > intent > stage) — risk-tier line is never dropped
    under cap pressure.
  - Stage-awareness stale-state and missing-blast-radius warnings
    bypass cooldown (safety override, see _summarize_stages).

Output channel: JSON `hookSpecificOutput.additionalContext` — the same
envelope risk-tier-guard.py used at lines 104-110.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / ".claude" / "hooks" / "state"

# ── Token budget (priority-ordered, safety-critical first) ────────────
PER_LINE_CAP = 500
GLOBAL_CAP = 2000

# ── Cooldowns (preserve each source guard's existing window) ──────────
RISK_COOLDOWN_MIN = 20         # risk-tier-guard.py:22
BIAS_COOLDOWN_MIN = 20         # bias_grounding_guard.py:22
INTENT_COOLDOWN_MIN = 15       # data-first-guard.py:26
STAGE_COOLDOWN_MIN = 30        # NEW — content-hash cooldown


# ─────────────────────────── Risk tier ────────────────────────────────
# Verbatim from risk-tier-guard.py:24-43

CRITICAL_RE = re.compile(
    r"\b("
    r"real capital|live trading|deploy|production|promotion|promote|"
    r"broker|order routing|position sizing|risk limit|kill switch|"
    r"account routing|account safety|capital review|deploy readiness|"
    r"readiness|live path|runtime control|threat model|security review"
    r")\b",
    re.IGNORECASE,
)

HIGH_RE = re.compile(
    r"\b("
    r"pipeline|check_drift|schema|migration|duckdb|database|timezone|dst|"
    r"session boundary|session time|orb window|concurrency|worktree|hook|"
    r"mutex|review|audit|verify|validation|backtest|research|hypothesis|"
    r"holdout|oos|p.?value|fdr|slippage|cost model|execution engine|"
    r"trading_app/live|refresh_data|outcome_builder"
    r")\b",
    re.IGNORECASE,
)

RISK_CRITICAL_MSG = (
    "RISK TIER: critical. Keep exploration lean; reserve high-reasoning "
    "or an independent review for final decisions. Require execution "
    "evidence before done."
)
RISK_HIGH_MSG = (
    "RISK TIER: high. Default to normal reasoning for exploration, then "
    "escalate only for review/decision points. Require targeted tests, "
    "drift, and explicit review."
)


# ─────────────────────────── Bias grounding ───────────────────────────
# Verbatim from scripts/tools/bias_grounding_guard.py:24-36

BIAS_TARGET_RE = re.compile(
    r"\b("
    r"research|review|audit|verify|validation|validate|bias|ground|"
    r"source|sources|literature|evidence|proof|prove|claim|claims|"
    r"result|results|deploy|promotion|promote|ready|readiness|"
    r"oos|holdout|backtest|significance|p.?value|fdr|dsr|sharpe"
    r")\b",
    re.IGNORECASE,
)

BIAS_MSG = (
    "RESEARCH MODE: Canon only, disconfirm first, tag MEASURED/INFERRED/"
    "UNSUPPORTED, then state edge, issue, next step."
)


# ─────────────────────────── Data-first / intent ──────────────────────
# Verbatim from data-first-guard.py:29-171

INVESTIGATION_RE = re.compile(
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

TRADING_QUERY_RE = re.compile(
    r"\b("
    r"what do i trade|what.?s live|my trades|my playbook|my portfolio|"
    r"trade tonight|trading tonight|what.?s on tonight|"
    r"morning trades|evening trades|active strategies|"
    r"what am i trading|current positions|my strats|my book|"
    r"show me my stuff|what.?s deployed|what.?s running"
    r")\b",
    re.IGNORECASE,
)

SESSION_TIME_RE = re.compile(
    r"\b("
    r"what time|when does|when is|session time|trade time|"
    r"tonight.?s session|what.?s on at|schedule tonight|"
    r"session start|when.*open|when.*close"
    r")\b",
    re.IGNORECASE,
)

DESIGN_RE = re.compile(
    r"\b("
    r"plan|design|think about|brainstorm|how would|how should|what if|"
    r"explore|iterate|4t|approach|architecture|consider|strategy for|"
    r"pros and cons|trade.?offs|options for"
    r")\b",
    re.IGNORECASE,
)

IMPLEMENT_RE = re.compile(
    r"\b("
    r"build it|do it|implement|go ahead|ship it|make it happen|just do it|"
    r"write the code|code it|execute|deploy|wire it up|hook it up|"
    r"yes|looks good|approved|lgtm"
    r")\b",
    re.IGNORECASE,
)

COMMIT_RE = re.compile(
    r"\b("
    r"commit|push|comit|pusdh|vcommit|merge|commit all|push it|"
    r"stage and commit|git push|commit and push"
    r")\b",
    re.IGNORECASE,
)

RESEARCH_RE = re.compile(
    r"\b("
    r"hypothesis|test.*edge|research|validate.*signal|stress test|"
    r"is this real|backtest|forward test|null test|significance|"
    r"p.?value|sharpe|fdr|discover|noise floor"
    r")\b",
    re.IGNORECASE,
)

ORIENT_RE = re.compile(
    r"\b("
    r"where are we|what.?s the status|orient|what.?s broken|"
    r"state of|health check|what needs doing|what.?s next"
    r")\b",
    re.IGNORECASE,
)

RESUME_RE = re.compile(
    r"\b("
    r"resume|pick up where|last conversation|last session|"
    r"where was i|what were we doing|carry on from|"
    r"it closed|conversation closed|got disconnected|session crashed"
    r")\b",
    re.IGNORECASE,
)

INVESTIGATION_DIRECTIVE = (
    "DATA FIRST: Query data before reading more code. Get numbers first, "
    "then explain."
)
TRADING_QUERY_DIRECTIVE = (
    "TRADING QUERY: Use trading_app.live_config/LIVE_PORTFOLIO, not "
    "validated_setups."
)
SESSION_TIME_DIRECTIVE = (
    "SESSION TIME: Use scripts/tools/generate_trade_sheet.py; do not do "
    "manual timezone math."
)
DESIGN_DIRECTIVE = (
    "DESIGN MODE: Plan and options only; do not edit until the user "
    "explicitly switches to implementation."
)
IMPLEMENT_DIRECTIVE = (
    "IMPLEMENT MODE: Execute now. If non-trivial, require STAGE_STATE "
    "blast_radius/scope first."
)
COMMIT_DIRECTIVE = (
    "GIT OPERATION: Execute directly; stage intentionally, commit clearly, "
    "push only if asked."
)
RESEARCH_DIRECTIVE = (
    "RESEARCH MODE: Use STRATEGY_BLUEPRINT sequence; every claim needs "
    "source layer, N, p-value, K, and WFE."
)
ORIENT_DIRECTIVE = (
    "ORIENT: Reground from HANDOFF.md, recent git history, active stage "
    "state, and live status commands."
)
RESUME_DIRECTIVE = (
    "RESUME: Re-read HANDOFF.md, recent git history, and active stage "
    "state before continuing."
)


# ─────────────────────────── Stage awareness ──────────────────────────
# Verbatim from stage-awareness.py:29-46

STAGE_STATE_FILE = PROJECT_ROOT / "docs" / "runtime" / "STAGE_STATE.md"
STAGES_DIR = PROJECT_ROOT / "docs" / "runtime" / "stages"

NONE_DIRECTIVES = (
    "SELF-CHECK: Simulate happy/edge/failure scenarios before presenting "
    "ANY plan. Show what you found, not just 'looks good.'",
    "PDF GROUNDING: If citing resources/ files, EXTRACT text first. "
    "Never cite from training memory as if you read the file.",
    "COMPLETION: 'Done' = tests pass (show output) + dead code swept "
    "(grep orphans) + drift clean. Not claims.",
    "BLAST RADIUS: List files changing + their tests + downstream "
    "consumers. Write it in STAGE_STATE before editing.",
)

DESIGN_DIRECTIVES = (
    "BEFORE PRESENTING PLAN: Simulate happy/edge/failure paths. Fix what "
    "breaks. Show simulation results. Do NOT present first draft.",
    "SELF-CHECK: Walk through your plan step by step. At each step: what "
    "if NULL? What if missing? What if the interface changed? Fix first, "
    "present second.",
    "EXECUTION PLAN: Every plan MUST include HOW to deploy, not just WHAT "
    "to change. FK constraints, rebuild ordering, data migration steps. "
    "Code without deployment = half a plan.",
)

MAX_STAGE_PREVIEW = 3


# ────────────────────── State file I/O ────────────────────────────────

def _load_state(name: str) -> dict:
    path = STATE_DIR / f"{name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_state(name: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{name}.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


def _within_cooldown(last_at: str | None, minutes: int) -> bool:
    if not last_at:
        return False
    try:
        age_min = (
            datetime.now(UTC) - datetime.fromisoformat(last_at)
        ).total_seconds() / 60
    except (TypeError, ValueError):
        return False
    return age_min < minutes


# ────────────────────── Classifiers ───────────────────────────────────

def _classify_risk_tier(prompt: str) -> tuple[str, str | None]:
    """Returns ('risk:<tier>', message) or ('risk:none', None)."""
    if CRITICAL_RE.search(prompt):
        tier, msg = "critical", RISK_CRITICAL_MSG
    elif HIGH_RE.search(prompt):
        tier, msg = "high", RISK_HIGH_MSG
    else:
        return ("risk:none", None)

    state = _load_state("risk-tier")
    # Emit if tier changed OR cooldown expired
    if state.get("last_tier") == tier and _within_cooldown(
        state.get("last_at"), RISK_COOLDOWN_MIN
    ):
        return (f"risk:{tier}", None)

    state["last_tier"] = tier
    state["last_at"] = datetime.now(UTC).isoformat()
    _save_state("risk-tier", state)
    return (f"risk:{tier}", msg)


def _classify_bias(prompt: str) -> tuple[str, str | None]:
    if not BIAS_TARGET_RE.search(prompt):
        return ("bias:none", None)

    state = _load_state("bias")
    if state.get("last_key") == BIAS_MSG and _within_cooldown(
        state.get("last_at"), BIAS_COOLDOWN_MIN
    ):
        return ("bias:cooldown", None)

    state["last_key"] = BIAS_MSG
    state["last_at"] = datetime.now(UTC).isoformat()
    _save_state("bias", state)
    return ("bias:research", BIAS_MSG)


def _classify_intent(prompt: str) -> tuple[str, str | None]:
    """Replicates data-first-guard.handle_user_prompt directive selection.

    Side effect: also resets data-first investigation/read counters in
    `data-first.json` so the PreToolUse branch sees current intent.
    """
    matched_investigation = bool(INVESTIGATION_RE.search(prompt))
    matched_commit = bool(COMMIT_RE.search(prompt))
    matched_implement = bool(
        IMPLEMENT_RE.search(prompt) and not DESIGN_RE.search(prompt)
    )
    matched_design = bool(DESIGN_RE.search(prompt))
    matched_research = bool(RESEARCH_RE.search(prompt))
    matched_resume = bool(RESUME_RE.search(prompt))
    matched_orient = bool(ORIENT_RE.search(prompt))

    directives: list[str] = []

    state = _load_state("data-first")
    state.setdefault("investigation_mode", False)
    state.setdefault("consecutive_reads", 0)

    if matched_investigation:
        state["investigation_mode"] = True
        state["consecutive_reads"] = 0
        directives.append(INVESTIGATION_DIRECTIVE)
    elif (
        matched_commit
        or matched_implement
        or matched_design
        or matched_research
        or matched_resume
        or matched_orient
    ):
        state["investigation_mode"] = False
        state["consecutive_reads"] = 0

    if TRADING_QUERY_RE.search(prompt):
        directives.append(TRADING_QUERY_DIRECTIVE)
    if SESSION_TIME_RE.search(prompt):
        directives.append(SESSION_TIME_DIRECTIVE)

    # Priority: commit > implement > design (most specific wins)
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

    state["last_updated"] = datetime.now(UTC).isoformat()

    if not directives:
        _save_state("data-first", state)
        return ("intent:none", None)

    key = " || ".join(directives)
    if state.get("last_prompt_directive_key") == key and _within_cooldown(
        state.get("last_prompt_directive_at"), INTENT_COOLDOWN_MIN
    ):
        _save_state("data-first", state)
        return ("intent:cooldown", None)

    state["last_prompt_directive_key"] = key
    state["last_prompt_directive_at"] = datetime.now(UTC).isoformat()
    _save_state("data-first", state)

    # Tag picks the strongest signal for diagnostics
    if matched_investigation:
        tag = "intent:investigation"
    elif matched_commit:
        tag = "intent:commit"
    elif matched_implement:
        tag = "intent:implement"
    elif matched_design:
        tag = "intent:design"
    elif matched_research:
        tag = "intent:research"
    elif matched_resume:
        tag = "intent:resume"
    elif matched_orient:
        tag = "intent:orient"
    else:
        tag = "intent:other"

    return (tag, "\n".join(directives))


# ── stage parsing helpers (verbatim from stage-awareness.py:50-99) ────

def _parse_field(content: str, field: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def _parse_blast_radius(content: str) -> str | None:
    if "## Blast Radius" in content:
        section = (
            content.split("## Blast Radius")[1]
            .split("##")[0]
            .split("---")[0]
        )
        if section.strip():
            return section.strip()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("blast_radius:"):
            value = (
                stripped.split(":", 1)[1].strip().strip('"').strip("'")
            )
            if value:
                return value
            items: list[str] = []
            collecting = False
            for l2 in content.splitlines():
                if l2.strip().startswith("blast_radius:"):
                    collecting = True
                    continue
                if collecting:
                    s2 = l2.strip()
                    if s2.startswith("- "):
                        items.append(s2[2:].strip())
                    elif s2 and not s2.startswith("#"):
                        break
            return "; ".join(items) if items else None
    return None


def _check_stale(content: str) -> bool:
    updated_str = _parse_field(content, "updated")
    if not updated_str:
        return False
    try:
        updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
        age = datetime.now(UTC) - updated
        return age > timedelta(hours=4)
    except (ValueError, TypeError):
        return False


def _stage_files() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    if STAGES_DIR.is_dir():
        for f in sorted(STAGES_DIR.glob("*.md")):
            files.append((f.stem, f))
    if STAGE_STATE_FILE.exists():
        files.append(("legacy", STAGE_STATE_FILE))
    return files


def _stage_content_hash(files: list[tuple[str, Path]]) -> str:
    """Hash sorted (path, mtime) for cooldown — cheap, no file reads."""
    parts: list[str] = []
    for _, fpath in files:
        try:
            parts.append(f"{fpath}:{fpath.stat().st_mtime_ns}")
        except OSError:
            continue
    return "|".join(parts)


def _classify_stage() -> tuple[str, str | None]:
    """Stage-awareness summary. Bypasses cooldown on safety conditions
    (stale STAGE_STATE warning, missing blast_radius on IMPLEMENTATION).
    """
    files = _stage_files()

    if not files:
        # No active stage — rotate "none" directives
        variant = datetime.now().minute % len(NONE_DIRECTIVES)
        msg = (
            "stage: none | non-trivial work requires STAGE_STATE with "
            "blast_radius before edits.\n"
            + NONE_DIRECTIVES[variant]
        )
        return ("stage:none", msg)

    output_lines: list[str] = []
    missing_blast_radius = 0
    stale_count = 0
    has_design = False

    for agent_name, fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        mode = _parse_field(content, "mode")
        task = _parse_field(content, "task")
        stage = _parse_field(content, "stage")
        stage_of = _parse_field(content, "stage_of")
        blast_radius = _parse_blast_radius(content)
        is_stale = _check_stale(content)

        if not mode:
            continue

        parts = [f"{agent_name}:{mode}"]
        if task:
            parts.append(task)
        if stage and stage_of:
            parts.append(f"({stage}/{stage_of})")
        elif stage:
            parts.append(f"(stage {stage})")
        if is_stale:
            parts.append("STALE(>4h)")
            stale_count += 1
        if mode == "IMPLEMENTATION" and (
            not blast_radius or len(blast_radius.strip()) < 30
        ):
            parts.append("MISSING blast_radius")
            missing_blast_radius += 1
        if mode == "DESIGN":
            has_design = True

        output_lines.append(" ".join(parts))

    if not output_lines:
        return ("stage:empty", None)

    # Safety override: stale or missing blast_radius bypasses cooldown
    safety_override = stale_count > 0 or missing_blast_radius > 0

    state = _load_state("stage")
    current_hash = _stage_content_hash(files)
    if (
        not safety_override
        and state.get("last_hash") == current_hash
        and _within_cooldown(state.get("last_at"), STAGE_COOLDOWN_MIN)
    ):
        return ("stage:cooldown", None)

    prefix = "stages" if len(output_lines) > 1 else "stage"
    shown = output_lines[:MAX_STAGE_PREVIEW]
    summary = " ; ".join(shown)
    extra = len(output_lines) - len(shown)
    if extra > 0:
        summary = f"{summary} ; (+{extra} more)"
    suffix_bits: list[str] = []
    if stale_count:
        suffix_bits.append(f"stale={stale_count}")
    if missing_blast_radius:
        suffix_bits.append(f"missing_blast_radius={missing_blast_radius}")
    suffix = f" | {' | '.join(suffix_bits)}" if suffix_bits else ""

    msg = f"{prefix}: {summary}{suffix}"
    if has_design:
        variant = datetime.now().minute % len(DESIGN_DIRECTIVES)
        msg = f"{msg}\n{DESIGN_DIRECTIVES[variant]}"

    state["last_hash"] = current_hash
    state["last_at"] = datetime.now(UTC).isoformat()
    _save_state("stage", state)

    tag = "stage:design" if has_design else "stage:active"
    return (tag, msg)


# ────────────────────── Output assembly ───────────────────────────────

def _safe_call(fn, *args) -> tuple[str, str | None]:
    """Run a classifier with isolated failure — returns (tag, None) on error."""
    try:
        return fn(*args)
    except BaseException:
        # Intentional BaseException catch: a classifier crash must not
        # take down the other 3. Return a sentinel tag so failure is
        # visible in tests / debugging without polluting Claude's context.
        return ("error", None)


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[: cap - 3] + "..."


def _assemble(items: list[tuple[str, str | None]]) -> str | None:
    """Apply per-line and global caps. Drop low-priority items first.
    Items must already be in priority order (highest first).
    """
    lines: list[str] = []
    total = 0
    for tag, msg in items:
        if msg is None:
            continue
        truncated = _truncate(msg, PER_LINE_CAP)
        line = f"[{tag}] {truncated}"
        if total + len(line) + 1 > GLOBAL_CAP:
            # Skip this line — but never drop a risk-tier line. Risk
            # is always first in priority order, so on cap pressure
            # we drop later (lower-priority) lines.
            continue
        lines.append(line)
        total += len(line) + 1
    if not lines:
        return None
    return "\n".join(lines)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)
    except BaseException:
        sys.exit(0)

    if event.get("hook_event_name") and event.get("hook_event_name") != "UserPromptSubmit":
        sys.exit(0)

    prompt = str(event.get("prompt", ""))
    if not prompt.strip():
        sys.exit(0)

    # Priority-ordered: risk > bias > intent > stage
    results = [
        _safe_call(_classify_risk_tier, prompt),
        _safe_call(_classify_bias, prompt),
        _safe_call(_classify_intent, prompt),
        _safe_call(_classify_stage),
    ]

    additional = _assemble(results)
    if additional is None:
        sys.exit(0)

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional,
        }
    }
    print(json.dumps(payload))
    sys.exit(0)


if __name__ == "__main__":
    main()
