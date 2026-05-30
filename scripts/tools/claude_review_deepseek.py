#!/usr/bin/env python3
"""Deterministic OpenCode/DeepSeek coding-agent commit gate.

Exit contract:
- 0 = APPROVE
- 1 = BLOCK
- 2 = REVIEW_UNAVAILABLE

This legacy-named script is called by `.githooks/pre-commit` only when
`OPENCODE_AGENT_ACTIVE=1`. It does not call external LLM APIs. Its v1 job is
to keep OpenCode work out of protected canompx3 truth surfaces unless a human
explicitly moves that work into a reviewed flow.
"""

from __future__ import annotations

import subprocess
import sys

PROTECTED_EXACT = {
    "gold.db",
    "TRADING_RULES.md",
    "RESEARCH_RULES.md",
    "trading_app/config.py",
    "trading_app/live_config.py",
    "trading_app/prop_profiles.py",
    "docs/runtime/action-queue.yaml",
}

PROTECTED_PREFIXES = (
    "pipeline/",
    "research/",
    "trading_app/live/",
    "trading_app/strategy_",
    "docs/audit/results/",
    "docs/audit/hypotheses/",
    "docs/pre-registrations/",
    "docs/runtime/lane_allocation",
)

PROTECTED_SUFFIXES = (
    ".db",
    ".duckdb",
    ".dbn",
    ".parquet",
    ".feather",
)


def normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def protected_reason(path: str) -> str | None:
    rel = normalize_path(path)
    if not rel:
        return None
    if rel in PROTECTED_EXACT:
        return "protected canonical surface"
    if rel.endswith(PROTECTED_SUFFIXES):
        return "database or market-data artifact"
    if any(rel.startswith(prefix) for prefix in PROTECTED_PREFIXES):
        return "protected live/research/data surface"
    return None


def blocked_files(paths: list[str]) -> list[tuple[str, str]]:
    blocked: list[tuple[str, str]] = []
    for raw_path in paths:
        path = normalize_path(raw_path)
        reason = protected_reason(path)
        if reason:
            blocked.append((path, reason))
    return blocked


def staged_files() -> tuple[int, list[str], str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    files = [line for line in result.stdout.splitlines() if line.strip()]
    return result.returncode, files, (result.stderr or "").strip()


def main() -> int:
    rc, files, error = staged_files()
    if rc != 0:
        print(f"[opencode-gate] REVIEW_UNAVAILABLE: git staged-file inspection failed: {error}", file=sys.stderr)
        return 2

    blocked = blocked_files(files)
    if blocked:
        print("[opencode-gate] BLOCKED: OpenCode staged protected canompx3 truth surfaces.", file=sys.stderr)
        for path, reason in blocked:
            print(f"  - {path}: {reason}", file=sys.stderr)
        print(
            "OpenCode may review these paths, but edits require an explicit Claude/Codex reviewed flow.",
            file=sys.stderr,
        )
        return 1

    if files:
        print(f"[opencode-gate] APPROVE: {len(files)} staged file(s) outside protected surfaces.")
    else:
        print("[opencode-gate] APPROVE: no staged files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
