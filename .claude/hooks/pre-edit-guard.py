#!/usr/bin/env python3
"""Pre-edit guard: block direct edits to gold.db and other protected binary files."""

import json
import sys

BLOCKED_PATTERNS = [
    "gold.db",
    "gold.db.wal",
    ".env",
]


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as exc:
        print(f"[pre-edit-guard] unexpected: {exc}", file=sys.stderr)
        sys.exit(0)
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

    sys.exit(0)


if __name__ == "__main__":
    main()
