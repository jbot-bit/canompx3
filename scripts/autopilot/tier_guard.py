#!/usr/bin/env python3
"""Deterministic risk-tier classifier for the headless autopilot runner.

Single source of truth for "is this action safe to do unattended (Tier A) or
must it be blocked + journalled for the operator (Tier B)?"

The tier definitions are seeded VERBATIM from `.claude/rules/autonomy-contract.md`
§ Tier B (the canonical doctrine). When that doctrine changes, update the tables
here and the tests in `tests/test_autopilot/test_tier_guard.py` together.

Design contract (differs from the advisory prompt-regex hooks):
- **Fail-CLOSED.** An unknown or ambiguous path that touches `pipeline/` or
  `trading_app/` classifies as Tier B (the safe direction — block + flag).
  Only paths we can positively prove are reversible (docs / config /
  scripts/autopilot / tests) classify as Tier A.
- Pure-Python, no third-party deps, importable AND runnable as a CLI so the
  runner, the PreToolUse hook, and the review tool all share ONE verdict.

CLI:
    python scripts/autopilot/tier_guard.py --tool Edit --path X
        -> prints "A <reason>" / "B <reason>", exit 0 (A) / 2 (B)
    python scripts/autopilot/tier_guard.py --tool Bash --command "git push"
        -> classifies a Bash command by its action table
"""

from __future__ import annotations

import argparse
import re
import sys

Tier = str  # "A" | "B"


# ── Tier-B PATH table (from autonomy-contract.md § Tier B) ─────────────────
# Any changed file whose normalized path matches one of these is capital-/
# schema-/canonical-adjacent and must NOT be edited unattended.
#
# Each entry is (substring-or-prefix, human reason). Matched against the
# forward-slash-normalized path with a simple `in` test, which covers both
# directory prefixes ("trading_app/live/") and basename hits
# ("prop_profiles.py") regardless of the repo-relative depth handed in.
TIER_B_PATH_MARKERS: list[tuple[str, str]] = [
    # Capital paths
    ("trading_app/live/", "live trading / capital path"),
    ("trading_app/broker", "broker / execution path"),
    ("trading_app/execution", "execution engine"),
    ("session_orchestrator", "live session orchestrator"),
    ("trading_app/prop_profiles.py", "lane / account profile (capital sizing)"),
    ("trading_app/prop_portfolio.py", "book builder (capital sizing)"),
    ("trading_app/account_survival.py", "account survival / DD budget"),
    ("risk_manager", "risk manager / kill-switch"),
    ("kill_switch", "kill-switch control"),
    ("live_config", "live deployment config"),
    # Schema / canonical-source modules (pipeline canon)
    ("pipeline/dst.py", "canonical session/DST source"),
    ("pipeline/cost_model.py", "canonical cost specs"),
    ("pipeline/asset_configs.py", "canonical instrument configs"),
    ("pipeline/paths.py", "canonical DB path source"),
    ("pipeline/holdout_policy.py", "canonical holdout policy"),
    ("trading_app/holdout_policy.py", "canonical holdout policy"),
    ("pipeline/schema", "DB schema definition"),
    # Sizing / allocation levers
    ("allocator", "capital allocator"),
    ("lane_allocation", "lane capital allocation"),
    # Binary DB / schema artifacts
    (".db", "database file write"),
    (".sql", "schema / SQL definition"),
]

# Paths we can positively prove are reversible -> Tier A. Checked BEFORE the
# fail-closed pipeline/trading_app catch-all so e.g. tests/ under a prod tree
# stay Tier A.
TIER_A_PATH_MARKERS: tuple[str, ...] = (
    "docs/",
    "scripts/autopilot/",
    "tests/",
    ".claude/rules/",
    ".claude/skills/",
    "docs/prompts/",
    "README",
)

# Trees that are fail-closed: an UNKNOWN path inside one of these is Tier B
# (we cannot prove it is reversible, and it sits next to capital/schema canon).
FAIL_CLOSED_TREES: tuple[str, ...] = ("pipeline/", "trading_app/")


# ── Tier-B ACTION table (Bash commands) ────────────────────────────────────
# Matched against the raw command string (case-insensitive). Order matters only
# for which reason is reported; first hit wins.
TIER_B_ACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"--live\b"), "live execution flag"),
    (re.compile(r"--demo\b"), "demo execution flag"),
    (re.compile(r"\bstrict[-_]?zero\b"), "strict-zero arming flag"),
    (re.compile(r"\bplace[_-]?order\b|\bsubmit[_-]?order\b"), "order placement"),
    (re.compile(r"\brun_live_session\b|\barm\b"), "live arming"),
    (re.compile(r"\brefresh_control_state\b"), "control-state DB write"),
    (re.compile(r"git\s+push\b"), "git push (outward-facing)"),
    (re.compile(r"--force\b|--force-with-lease\b|(?<!\w)-f(?!\w)"), "forced git operation"),
    (re.compile(r"git\s+reset\s+--hard"), "destructive git reset"),
    (re.compile(r"git\s+clean\s+-[a-z]*f"), "destructive git clean"),
    (re.compile(r"git\s+merge\b.*\bmain\b|merge.*\borigin/main\b"), "merge to main"),
    (re.compile(r"git\s+checkout\s+main\b|git\s+switch\s+main\b"), "switch to main branch"),
    (re.compile(r"\bDROP\s+TABLE\b|\bTRUNCATE\b|\bDELETE\s+FROM\b", re.IGNORECASE), "destructive DB write"),
    # Destructive filesystem ops (rm -rf / rmdir) — a recursive delete is
    # irreversible regardless of target, so it is Tier B unconditionally.
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r"), "recursive force delete"),
]


# Write-shaped Bash commands: output redirect, in-place edit, or a writer verb.
# Used to decide whether a canon-path mention in a Bash command is a WRITE
# (Tier B) versus a harmless read (Tier A).
_WRITE_CMD_RE = re.compile(r">|>>|\btee\b|sed\s+-i|\bmv\b|\bcp\b|\btruncate\b|\bdd\b|\bchmod\b|\bchown\b")


def _normalize(path: str) -> str:
    """Normalize for matching: backslashes -> /, strip, drop leading ./,
    and LOWERCASE so the path table is case-insensitive (Windows paths and
    accidental capitalization must not bypass a Tier-B marker)."""
    return (path or "").replace("\\", "/").strip().lstrip("./").lower()


def classify_path(path: str) -> tuple[Tier, str]:
    """Classify a single file path. Returns (tier, reason)."""
    norm = _normalize(path)
    if not norm:
        # Empty/unknown path: cannot prove reversible -> fail closed.
        return ("B", "empty/unknown path (fail-closed)")

    # 1. Positive Tier-B path markers always win (capital/schema canon).
    for marker, reason in TIER_B_PATH_MARKERS:
        if marker in norm:
            return ("B", reason)

    # 2. Positively-reversible trees -> Tier A.
    for marker in TIER_A_PATH_MARKERS:
        if norm.startswith(marker) or f"/{marker}" in norm or marker in norm:
            return ("A", "reversible docs/config/tests/autopilot path")

    # 3. Fail-closed: unknown path inside a prod canon tree -> Tier B.
    for tree in FAIL_CLOSED_TREES:
        if norm.startswith(tree) or f"/{tree}" in norm:
            return ("B", f"unknown path under {tree} (fail-closed)")

    # 4. Everything else (top-level scripts, misc) -> Tier A. These are not
    #    capital/schema canon and are reversible on a branch.
    return ("A", "non-canonical reversible path")


def classify_command(command: str) -> tuple[Tier, str]:
    """Classify a Bash command string by the action table."""
    cmd = command or ""
    for pat, reason in TIER_B_ACTION_PATTERNS:
        if pat.search(cmd):
            return ("B", reason)
    return ("A", "no Tier-B action pattern matched")


def classify_action(tool: str, tool_input: dict) -> tuple[Tier, str]:
    """Classify a tool call (the shape a PreToolUse hook receives).

    - Edit/Write: classify by `file_path`.
    - Bash: classify by `command` (action table). If a Bash command embeds a
      Tier-B-looking file path argument we ALSO honor the path table — the
      stricter (B) verdict wins.
    """
    tool = (tool or "").strip()
    tool_input = tool_input or {}

    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        return classify_path(str(tool_input.get("file_path", "")))

    if tool == "Bash":
        command = str(tool_input.get("command", ""))
        action_tier, action_reason = classify_command(command)
        if action_tier == "B":
            return ("B", action_reason)
        # Secondary: a Bash command that WRITES to a canon path is still B.
        # Scope this to write-shaped commands only (redirects, in-place edits,
        # mv/cp/tee targets) so harmless reads (cat/grep/head of a canon file)
        # are not over-blocked.
        if _WRITE_CMD_RE.search(command):
            norm_cmd = command.replace("\\", "/").lower()
            for marker, reason in TIER_B_PATH_MARKERS:
                if marker in norm_cmd:
                    return ("B", f"command writes to {reason}")
        return ("A", action_reason)

    # Read/Grep/Glob and friends are inherently read-only -> Tier A.
    return ("A", f"read-only tool ({tool or 'unknown'})")


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classify an autopilot action as Tier A or B.")
    ap.add_argument("--tool", default="Edit", help="Tool name (Edit/Write/Bash/Read...)")
    ap.add_argument("--path", default=None, help="File path (for Edit/Write)")
    ap.add_argument("--command", default=None, help="Command string (for Bash)")
    args = ap.parse_args(argv)

    tool_input: dict = {}
    if args.path is not None:
        tool_input["file_path"] = args.path
    if args.command is not None:
        tool_input["command"] = args.command

    tier, reason = classify_action(args.tool, tool_input)
    print(f"{tier} {reason}")
    return 0 if tier == "A" else 2


if __name__ == "__main__":
    sys.exit(_main())
