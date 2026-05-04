#!/usr/bin/env python3
"""Post-edit hook: runs schema tests AND scans skills for stale SQL after schema edits.

Lesson 11: skill/command SQL rots silently after schema changes.
"""

import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_python() -> str:
    venv_win = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_win.exists():
        return str(venv_win)
    venv_unix = _PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_unix.exists():
        return str(venv_unix)
    return sys.executable


_HOOK_PYTHON = _resolve_python()

# Known-bad column/table names that have caused past failures
STALE_PATTERNS = {
    "strategy_fitness": "TABLE DOES NOT EXIST — fitness is in edge_families.robustness_status",
    "v.symbol": "WRONG COLUMN — validated_setups uses 'instrument', not 'symbol'",
    "avg_r": "WRONG COLUMN — use 'expectancy_r'",
    ".sharpe ": "AMBIGUOUS — use 'sharpe_ann' (annualized) explicitly",
}

SKILL_DIRS = [
    Path(".claude/skills"),
    Path(".claude/commands"),
    Path(".claude/agents"),
]


def scan_skills_for_stale_sql():
    """Grep skills/commands/agents for known-stale SQL patterns."""
    violations = []
    for skill_dir in SKILL_DIRS:
        if not skill_dir.exists():
            continue
        for fpath in skill_dir.rglob("*.md"):
            try:
                content = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
            for pattern, reason in STALE_PATTERNS.items():
                if pattern in content:
                    violations.append(f"  {fpath}: contains '{pattern}' — {reason}")
    return violations


def main():
    input_data = json.load(sys.stdin)
    file_path = input_data.get("tool_input", {}).get("file_path", "")

    # Only run for schema files
    if not (
        file_path.endswith("init_db.py")
        or file_path.endswith("db_manager.py")
        or file_path.endswith("schema.py")
    ):
        sys.exit(0)

    # Run schema tests
    result = subprocess.run(
        [
            _HOOK_PYTHON,
            "-m",
            "pytest",
            "tests/test_pipeline/test_schema.py",
            "tests/test_app_sync.py",
            "-x",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(_PROJECT_ROOT),
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(_PROJECT_ROOT),
        },
    )
    if result.returncode != 0:
        print(f"SCHEMA TESTS FAILED after editing {file_path}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(2)

    # Scan skills for stale SQL patterns
    stale = scan_skills_for_stale_sql()
    if stale:
        print(
            "WARNING: Skills/agents contain stale SQL patterns "
            "(Lesson 11 — skill SQL rots after schema changes):",
            file=sys.stderr,
        )
        for v in stale:
            print(v, file=sys.stderr)
        # Advisory — don't block, just warn

    sys.exit(0)


if __name__ == "__main__":
    main()
