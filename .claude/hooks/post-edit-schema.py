#!/usr/bin/env python3
"""Post-edit hook: runs schema tests after init_db.py or db_manager.py edits."""

import json
import subprocess
import sys


def main():
    input_data = json.load(sys.stdin)
    file_path = input_data.get("tool_input", {}).get("file_path", "")

    # Only run for schema files
    if not (file_path.endswith("init_db.py") or file_path.endswith("db_manager.py") or file_path.endswith("schema.py")):
        sys.exit(0)

    # Run schema tests
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_pipeline/test_schema.py", "tests/test_app_sync.py", "-x", "-q"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"SCHEMA TESTS FAILED after editing {file_path}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
