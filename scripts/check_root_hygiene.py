"""Check that the project root contains only allowed files and directories.

Usage:
    python scripts/check_root_hygiene.py  # Exit 1 if unexpected items found
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_DIRS = {
    "pipeline", "trading_app", "tests", "scripts", "research", "docs", "artifacts", "backups",
    "DB", ".git", ".github", ".githooks", ".claude", "__pycache__",
    "llm-code-scanner", "plugins",
    # Virtual environments
    ".venv", "venv",
    # Tool caches
    ".pytest_cache", ".ruff_cache",
    # Gitignored but may exist locally
    ".obsidian", "openclaw", "MNQ db", "local_db",
    # UI
    "ui",
}

ALLOWED_FILES = {
    "CLAUDE.md", "MARKET_PLAYBOOK.md", "ROADMAP.md", "REPO_MAP.md", "README.md",
    "TRADING_RULES.md", "TRADING_PLAN.md",
    "CANONICAL_LOGIC.txt", "CANONICAL_backfill_dbn_mgc_rules.txt",
    "CANONICAL_backfill_dbn_mgc_rules_addon.txt",
    "pyproject.toml", "requirements.txt", "ruff.toml", ".gitignore", ".ENV",
    "gold.db", "dashboard.html", "pipeline-explorer.html",
    "Canompx3.code-workspace", ".mcp.json", "portfolio_report.json",
    # Data archives
    "GOLD_DB_FULLSIZE.zip",
    # Logs
    "backfill_overnight.log",
}


def main() -> None:
    # Windows reserved device names (nul, con, prn, etc.) — undeletable, skip them
    WIN_DEVICES = {"nul", "con", "prn", "aux"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}

    unexpected = []
    for entry in sorted(PROJECT_ROOT.iterdir()):
        name = entry.name
        if name.lower() in WIN_DEVICES and not entry.is_file() and not entry.is_dir():
            continue
        if entry.is_dir() and name in ALLOWED_DIRS:
            continue
        if entry.is_file() and name in ALLOWED_FILES:
            continue
        unexpected.append(name)

    if unexpected:
        print(f"Root hygiene check FAILED: {len(unexpected)} unexpected item(s):")
        for name in unexpected:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("Root hygiene check passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
