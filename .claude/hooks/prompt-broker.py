#!/usr/bin/env python3
"""Low-token UserPromptSubmit broker for Claude Code.

Claude Code only injects normal hook stdout into model context for a few
events, including UserPromptSubmit. Keep all prompt-time reminders in one
capped JSON payload so routine turns do not spawn several hooks or repeat
long state.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / ".claude" / "hooks" / "state"
STAGES_DIR = PROJECT_ROOT / "docs" / "runtime" / "stages"
STAGE_STATE_FILE = PROJECT_ROOT / "docs" / "runtime" / "STAGE_STATE.md"

GLOBAL_CAP = 1400
RISK_COOLDOWN_MINUTES = 20
STAGE_COOLDOWN_MINUTES = 20

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

BIAS_RE = re.compile(
    r"\b("
    r"research|review|audit|verify|validation|validate|bias|ground|"
    r"source|sources|literature|evidence|proof|prove|claim|claims|"
    r"result|results|deploy|promotion|promote|ready|readiness|"
    r"oos|holdout|backtest|significance|p.?value|fdr|dsr|sharpe"
    r")\b",
    re.IGNORECASE,
)

DESIGN_RE = re.compile(r"\b(plan|design|brainstorm|approach|4t)\b", re.IGNORECASE)
COMMIT_RE = re.compile(r"\b(commit|push|merge)\b", re.IGNORECASE)
DATA_INVESTIGATION_RE = re.compile(
    r"\b(what happened|how bad|why .*numbers?|numbers? differ|data question|investigate|oos|holdout|results?)\b",
    re.IGNORECASE,
)
CONTEXT_HYGIENE_RE = re.compile(
    r"\b("
    r"new task|switch tasks?|different task|unrelated|another thing|"
    r"context (is )?(high|large|full)|too many tokens|token use|"
    r"broad audit|review everything|scan logs?|test output"
    r")\b",
    re.IGNORECASE,
)
DISCOVERY_NARRATION_RE = re.compile(
    r"\b("
    r"reading the remaining|reading remaining|let me (read|check) more|"
    r"isolating (the )?(likely )?weak spots|before I patch|before patching|"
    r"reading (more|the rest|the other) (files|adapters|modules)|"
    r"i'?m reading|i am reading|underusing|under-using|i'?ve isolated"
    r")\b",
    re.IGNORECASE,
)
DISCOVERY_OPEN_ENDED_RE = re.compile(
    r"\b(harden(ing)?|future[- ]?proof(ing)?|tighten up|shore up|robustify|make (it )?(more )?robust|audit everything|review everything)\b",
    re.IGNORECASE,
)
DISCOVERY_ESCAPE_RE = re.compile(r"(\bTRIVIAL:|\bREPRO:|\bEXPLORE:|context_resolver\.py)")


def _read_state(name: str) -> dict:
    try:
        state = json.loads((STATE_DIR / name).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    return state if isinstance(state, dict) else {}


def _write_state(name: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / name).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _age_minutes(value: object) -> float | None:
    if not value:
        return None
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(str(value))).total_seconds() / 60
    except (TypeError, ValueError):
        return None


def _cooldown_elapsed(state: dict, key: str, cooldown_minutes: int) -> bool:
    if state.get("last_key") != key:
        return True
    age = _age_minutes(state.get("last_at"))
    return age is None or age >= cooldown_minutes


def _classify_risk(prompt: str) -> str | None:
    if CRITICAL_RE.search(prompt):
        return (
            "[risk:critical] RISK TIER: critical. Keep exploration lean; use high reasoning or an "
            "independent review only for final capital/deploy decisions. Require execution evidence before done."
        )
    if HIGH_RE.search(prompt):
        return (
            "[risk:high] RISK TIER: high. Use normal reasoning for exploration; escalate only at "
            "review/decision points. Require targeted tests, drift, and explicit review."
        )
    return None


def _risk_line(prompt: str) -> str | None:
    line = _classify_risk(prompt)
    if not line:
        return None
    state = _read_state("risk-tier.json")
    key = "critical" if "[risk:critical]" in line else "high"
    if not _cooldown_elapsed(state, key, RISK_COOLDOWN_MINUTES):
        return None
    _write_state("risk-tier.json", {"last_key": key, "last_at": datetime.now(UTC).isoformat()})
    return line


def _classify_bias(prompt: str) -> str | None:
    if not BIAS_RE.search(prompt):
        return None
    return "[bias:research] Canon only; disconfirm first; tag MEASURED/INFERRED/UNSUPPORTED before conclusion."


def _classify_intent(prompt: str) -> str | None:
    if DESIGN_RE.search(prompt):
        return "[intent:design] DESIGN MODE: scope first, present approach/blast radius before edits."
    if COMMIT_RE.search(prompt):
        return "[intent:commit] GIT OPERATION: inspect status, commit only intended files, push when requested."
    return None


def _context_hygiene_line(prompt: str) -> str | None:
    if not CONTEXT_HYGIENE_RE.search(prompt):
        return None
    state = _read_state("context-hygiene.json")
    key = "context-hygiene"
    if not _cooldown_elapsed(state, key, 30):
        return None
    _write_state("context-hygiene.json", {"last_key": key, "last_at": datetime.now(UTC).isoformat()})
    return (
        "[context:hygiene] If this is unrelated to the current thread, use /clear first. "
        "Keep broad scans/logs/tests in a focused subagent or filtered CLI output."
    )


def _discovery_loop_line(prompt: str) -> str | None:
    if DISCOVERY_ESCAPE_RE.search(prompt):
        return None
    if not (DISCOVERY_NARRATION_RE.search(prompt) or DISCOVERY_OPEN_ENDED_RE.search(prompt)):
        return None
    state = _read_state("discovery-loop.json")
    key = "discovery-loop"
    if not _cooldown_elapsed(state, key, 15):
        return None
    _write_state("discovery-loop.json", {"last_key": key, "last_at": datetime.now(UTC).isoformat()})
    return (
        "[discovery-loop] Before production edits, give one of: REPRO, context_resolver output, "
        "or TRIVIAL file list/diff<100. Treat pasted agent status as narration."
    )


def _parse_field(content: str, field: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def _parse_blast_radius(content: str) -> str | None:
    if "## Blast Radius" in content:
        section = content.split("## Blast Radius", 1)[1].split("##", 1)[0].split("---", 1)[0]
        if section.strip():
            return section.strip()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("blast_radius:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'") or None
    return None


def _stage_files() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    if STAGES_DIR.is_dir():
        files.extend((path.stem, path) for path in sorted(STAGES_DIR.glob("*.md")))
    if STAGE_STATE_FILE.exists():
        files.append(("legacy", STAGE_STATE_FILE))
    return files


def _stage_digest(files: list[tuple[str, Path]]) -> str:
    h = hashlib.sha256()
    for name, path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        h.update(f"{name}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))
    return h.hexdigest()


def _stage_line() -> str | None:
    files = _stage_files()
    if not files:
        return "[stage:none] Non-trivial edits need a stage/blast-radius note; trivial edits stay under 100 net lines."

    digest = _stage_digest(files)
    state = _read_state("stage-awareness.json")
    if not _cooldown_elapsed(state, digest, STAGE_COOLDOWN_MINUTES):
        return None

    summaries: list[str] = []
    missing = 0
    for name, path in files[:3]:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        mode = _parse_field(content, "mode")
        if not mode:
            continue
        task = _parse_field(content, "task")
        blast = _parse_blast_radius(content)
        if mode == "IMPLEMENTATION" and (not blast or len(blast.strip()) < 30):
            missing += 1
        bits = [f"{name}:{mode}"]
        if task:
            bits.append(task[:80])
        summaries.append(" ".join(bits))

    if not summaries:
        return None
    extra = len(files) - min(len(files), 3)
    suffix = f"; +{extra} more" if extra else ""
    missing_part = f"; missing_blast_radius={missing}" if missing else ""
    _write_state("stage-awareness.json", {"last_key": digest, "last_at": datetime.now(UTC).isoformat()})
    return f"[stage:{len(files)}] {'; '.join(summaries)}{suffix}{missing_part}"


def _update_data_first(prompt: str) -> str | None:
    if not DATA_INVESTIGATION_RE.search(prompt):
        return None
    state = _read_state("data-first.json")
    state["investigation_mode"] = True
    state["last_updated"] = datetime.now(UTC).isoformat()
    state.setdefault("consecutive_reads", 0)
    _write_state("data-first.json", state)
    return "[data:first] Data question detected: query canonical data before broad code reading."


def _append_line(lines: list[str], func, prompt: str | None = None) -> None:
    try:
        line = func(prompt) if prompt is not None else func()
    except Exception:
        return
    if line:
        lines.append(line)


def _cap_lines(lines: list[str]) -> str:
    kept: list[str] = []
    total = 0
    for line in lines:
        candidate_len = len(line) + (1 if kept else 0)
        if kept and total + candidate_len > GLOBAL_CAP:
            break
        if not kept and len(line) > GLOBAL_CAP:
            return line[: GLOBAL_CAP - 3] + "..."
        kept.append(line)
        total += candidate_len
    return "\n".join(kept)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    if event.get("hook_event_name") != "UserPromptSubmit":
        sys.exit(0)

    prompt = str(event.get("prompt") or "").strip()
    if not prompt:
        sys.exit(0)

    lines: list[str] = []
    _append_line(lines, _risk_line, prompt)
    _append_line(lines, _classify_bias, prompt)
    _append_line(lines, _classify_intent, prompt)
    _append_line(lines, _context_hygiene_line, prompt)
    _append_line(lines, _discovery_loop_line, prompt)
    _append_line(lines, _update_data_first, prompt)
    _append_line(lines, _stage_line)

    if not lines:
        sys.exit(0)

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": _cap_lines(lines),
        }
    }
    print(json.dumps(payload))
    sys.exit(0)


if __name__ == "__main__":
    main()
