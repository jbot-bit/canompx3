#!/usr/bin/env python3
"""Report context-heavy surfaces that affect Claude/Codex token usage.

This script is intentionally report-only. It does not mutate repo state.
It focuses on durable, inspectable signals rather than pretending to know
exact billed token usage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_DIR = PROJECT_ROOT / ".claude" / "rules"
STAGES_DIR = PROJECT_ROOT / "docs" / "runtime" / "stages"
SETTINGS_PATH = PROJECT_ROOT / ".claude" / "settings.json"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"


@dataclass(frozen=True)
class FileMetric:
    path: Path
    chars: int
    lines: int


@dataclass(frozen=True)
class RuleMetric:
    path: Path
    chars: int
    scoped: bool


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _file_metric(path: Path) -> FileMetric | None:
    if not path.exists():
        return None
    text = _read_text(path)
    return FileMetric(path=path, chars=len(text), lines=text.count("\n") + 1)


def _rule_metric(path: Path) -> RuleMetric:
    text = _read_text(path)
    scoped = bool(re.search(r"(?m)^paths:\s*$", text))
    return RuleMetric(path=path, chars=len(text), scoped=scoped)


def _load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    return json.loads(_read_text(SETTINGS_PATH))


def _has_agent_teams_enabled(settings: dict) -> bool:
    env = settings.get("env") or {}
    return str(env.get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS", "")).strip() == "1"


def _has_risk_tier_hook(settings: dict) -> bool:
    hooks = settings.get("hooks") or {}
    prompt_hooks = hooks.get("UserPromptSubmit") or []
    for entry in prompt_hooks:
        for hook in entry.get("hooks") or []:
            command = str(hook.get("command") or "")
            if "risk-tier-guard.py" in command:
                return True
    return False


def _home_claude_candidates() -> list[Path]:
    candidates = [Path.home() / ".claude" / "CLAUDE.md"]
    windows_candidate = Path("/mnt/c/Users") / Path.home().name / ".claude" / "CLAUDE.md"
    if windows_candidate not in candidates:
        candidates.append(windows_candidate)
    return candidates


def _gitignore_has(pattern: str) -> bool:
    if not GITIGNORE_PATH.exists():
        return False
    return pattern in _read_text(GITIGNORE_PATH)


def _format_chars(chars: int) -> str:
    return f"{chars:,}"


def main() -> int:
    settings = _load_settings()
    effort = settings.get("effortLevel", "unknown")
    always_thinking = settings.get("alwaysThinkingEnabled", "unknown")
    teams_enabled = _has_agent_teams_enabled(settings)
    risk_tier_hook = _has_risk_tier_hook(settings)

    startup_files = [
        _file_metric(PROJECT_ROOT / "CLAUDE.md"),
        _file_metric(PROJECT_ROOT / "CODEX.md"),
        _file_metric(PROJECT_ROOT / "AGENTS.md"),
    ]
    startup_files = [metric for metric in startup_files if metric is not None]

    rule_metrics = sorted(
        (_rule_metric(path) for path in RULES_DIR.glob("*.md")),
        key=lambda metric: metric.chars,
        reverse=True,
    )
    always_on = [metric for metric in rule_metrics if not metric.scoped]
    scoped = [metric for metric in rule_metrics if metric.scoped]

    stage_files = sorted(path for path in STAGES_DIR.glob("*.md") if path.name != ".gitkeep")
    home_candidates = _home_claude_candidates()
    existing_home = [path for path in home_candidates if path.exists()]

    print("TOKEN HYGIENE REPORT")
    print(f"project: {PROJECT_ROOT}")
    print()
    print("Measured")
    print(f"- claude effortLevel: {effort}")
    print(f"- alwaysThinkingEnabled: {always_thinking}")
    print(f"- experimental agent teams enabled: {'yes' if teams_enabled else 'no'}")
    print(f"- risk-tier hook configured: {'yes' if risk_tier_hook else 'no'}")
    print(f"- startup docs measured: {len(startup_files)}")
    for metric in startup_files:
        rel = metric.path.relative_to(PROJECT_ROOT)
        print(f"  - {rel}: {_format_chars(metric.chars)} chars, {metric.lines} lines")
    print(f"- rules: {len(rule_metrics)} total, {len(always_on)} always-on, {len(scoped)} path-scoped")
    if always_on:
        print("  - largest always-on rules:")
        for metric in always_on[:5]:
            rel = metric.path.relative_to(PROJECT_ROOT)
            print(f"    - {rel}: {_format_chars(metric.chars)} chars")
    print(f"- active stage files: {len(stage_files)}")
    if stage_files:
        preview = ", ".join(path.stem for path in stage_files[:3])
        extra = len(stage_files) - min(len(stage_files), 3)
        if extra > 0:
            preview = f"{preview}, +{extra} more"
        print(f"  - preview: {preview}")
    print(
        f"- gitignored private repo files tracked in policy: "
        f"SOUL.md={'yes' if _gitignore_has('SOUL.md') else 'no'}, "
        f"USER.md={'yes' if _gitignore_has('USER.md') else 'no'}, "
        f"memory/={'yes' if _gitignore_has('memory/') else 'no'}"
    )
    print(f"- user-level Claude memory detected: {'yes' if existing_home else 'no'}")
    for path in existing_home:
        print(f"  - {path}")

    print()
    print("Recommendations")
    if effort == "high" or always_thinking is True:
        print(
            "- Keep default reasoning cheaper. Reserve high reasoning for risky final review, not routine exploration."
        )
    else:
        print("- Default reasoning is already in the cheaper tier. Keep escalation reserved for risky work.")
    if teams_enabled:
        print("- Disable agent teams by default. They multiply context even when the task is not naturally parallel.")
    else:
        print("- Agent teams are off by default. Only enable them for bounded parallel work with real payoff.")
    if not risk_tier_hook:
        print(
            "- Add a tiny prompt-tier hook so risky prompts escalate without paying a max-reasoning tax on every turn."
        )
    if len(always_on) > 3:
        print("- Trim or split always-on rules. Move long examples and rationale into appendices or skills.")
    else:
        print("- Always-on rule count is modest. Focus on keeping them concise rather than adding more doctrine.")
    if len(stage_files) > 3:
        print("- Archive or close stale stage files. Large stage sets create noisy orientation and extra reading.")
    else:
        print("- Stage-file count is already compact.")
    if not existing_home:
        print("- Move personal startup preferences into user-level Claude memory so worktrees inherit them cleanly.")
    else:
        print("- User-level Claude memory exists. Keep it small so every session does not pay for biography.")

    print()
    print("Operational Notes")
    print(
        "- Local-only work saves tokens only when it avoids web/MCP/GitHub lookups. It is not a major savings lever by itself."
    )
    print(
        "- Skipping commits usually does not save tokens. Small commits often reduce re-orientation and repeated review."
    )
    print(
        "- The most reliable savings come from fresh sessions, narrower prompts, smaller always-on context, and disciplined subagent use."
    )
    print(
        "- For live measurement, compare fresh-session `/context` before and after touching a high-doctrine path like `pipeline/`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
