#!/usr/bin/env python3
"""Pre-edit guard: block direct edits to gold.db and other protected binary files.

Phase 3 / A1 (advisory): on edits to pipeline/ or trading_app/, emit a CRG
impact-radius advisory to stderr so the editor sees the top-N affected files
before changing anything. Non-blocking, fail-open, 2s timeout.

Refs: docs/plans/2026-04-29-crg-integration-spec.md § Phase 3 / A1
"""

import json
import os
import subprocess
import sys

BLOCKED_PATTERNS = [
    "gold.db",
    "gold.db.wal",
    ".env",
]

ADVISORY_PATH_PREFIXES = (
    "pipeline/",
    "trading_app/",
    "pipeline\\",
    "trading_app\\",
)

CANONICAL_REPO = "C:/Users/joshd/canompx3"
CRG_TIMEOUT_SECONDS = 2


def _emit_impact_advisory(file_path: str) -> None:
    """Best-effort CRG impact-radius advisory. Fail-open; never block.

    Prints up to 5 affected files / symbols to stderr as a notice.
    Silent on any failure (binary missing, timeout, non-zero exit).
    """
    try:
        env = os.environ.copy()
        env.setdefault("CRG_REPO_ROOT", CANONICAL_REPO)
        result = subprocess.run(
            [
                "code-review-graph",
                "impact-radius",
                "--target",
                file_path,
                "--max-depth",
                "2",
                "--repo",
                CANONICAL_REPO,
            ],
            capture_output=True,
            text=True,
            timeout=CRG_TIMEOUT_SECONDS,
            env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return
    if result.returncode != 0:
        return
    head_lines = (result.stdout or "").splitlines()[:5]
    if not head_lines:
        return
    print(
        f"CRG impact-radius advisory for {file_path} (top {len(head_lines)}):",
        file=sys.stderr,
    )
    for line in head_lines:
        print(f"  {line}", file=sys.stderr)


def main():
    input_data = json.load(sys.stdin)
    file_path = input_data.get("tool_input", {}).get("file_path", "")

    for pattern in BLOCKED_PATTERNS:
        if file_path.endswith(pattern):
            print(
                f"BLOCKED: Direct edit to '{pattern}' is not allowed.\n"
                f"  gold.db → use pipeline commands or DuckDB CLI\n"
                f"  .env    → edit manually outside Claude",
                file=sys.stderr,
            )
            sys.exit(2)

    if file_path and any(
        norm in file_path.replace("\\", "/")
        for norm in ("pipeline/", "trading_app/")
    ):
        _emit_impact_advisory(file_path)

    sys.exit(0)


if __name__ == "__main__":
    main()
