"""CRG usage-log shim — records per-agent CRG calls for adoption telemetry.

Spec: docs/plans/2026-04-29-crg-integration-spec.md M3 (lines 153, 217, 307).
Plan: docs/plans/2026-04-30-crg-maximization-v2.md PR-2.

Closes the gap from PR #177's F1: spec-promised file, never shipped. PR-1 wired
the MCP prompts; PR-2 instruments their use so adoption is measurable.

Public API:
    record_crg_call(agent, tool, query=None, token_estimate=None) -> None

Behavior contract:
    - Append one JSON line to .code-review-graph/usage-log.jsonl in repo root.
    - Fail-silent on any IOError (institutional-rigor §6: telemetry must never
      break agent execution).
    - Auto-create .code-review-graph/ directory if missing.
    - O_APPEND atomicity guarantees line writes <512 bytes don't interleave
      across processes; no explicit lock needed for typical agent invocations.

CLI usage (for markdown agent prompts that can't import Python):
    python .claude/hooks/_crg_usage_log.py \\
        --agent verify-complete --tool review_changes \\
        --query "diff vs HEAD~1" --tokens 1234
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

LOG_DIR_NAME = ".code-review-graph"
LOG_FILE_NAME = "usage-log.jsonl"


def _repo_root() -> Path:
    """Walk upward from this file until we find a .git directory.

    Falls back to the file's grandparent if traversal fails — fail-silent path.
    """
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


def _log_path() -> Path:
    return _repo_root() / LOG_DIR_NAME / LOG_FILE_NAME


def record_crg_call(
    agent: str,
    tool: str,
    query: str | None = None,
    token_estimate: int | None = None,
) -> None:
    """Append one JSON line to the usage log. Fail-silent on any error.

    Args:
        agent: agent or skill name (e.g. "verify-complete", "quant-debug").
        tool: CRG tool/prompt name (e.g. "review_changes", "get_minimal_context_tool").
        query: optional free-text query/target (truncated to 500 chars on write).
        token_estimate: optional input+output token estimate.
    """
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "agent": str(agent)[:64],
            "tool": str(tool)[:64],
            "query": (str(query)[:500] if query is not None else None),
            "token_estimate": (int(token_estimate) if token_estimate is not None else None),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        # O_APPEND guarantees atomicity for writes <PIPE_BUF (Linux 4096, Win typically 512+).
        # Our line is bounded by truncation; safe under concurrent agents.
        fd = os.open(
            str(path),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except (OSError, ValueError, TypeError):
        # Fail-silent: telemetry must NEVER break agent execution.
        # Specifically swallowed: OSError (IO/permissions), ValueError/TypeError
        # (bad token_estimate). Other exceptions propagate by design.
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record one CRG agent-call entry to the usage log.",
        prog="_crg_usage_log",
    )
    parser.add_argument("--agent", required=True, help="Agent or skill name.")
    parser.add_argument("--tool", required=True, help="CRG tool or MCP prompt name.")
    parser.add_argument("--query", default=None, help="Optional free-text query/target.")
    parser.add_argument(
        "--tokens",
        dest="token_estimate",
        type=int,
        default=None,
        help="Optional input+output token estimate.",
    )
    args = parser.parse_args(argv)
    record_crg_call(
        agent=args.agent,
        tool=args.tool,
        query=args.query,
        token_estimate=args.token_estimate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
