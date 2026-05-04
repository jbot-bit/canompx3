#!/usr/bin/env python3
"""Profile pipeline/check_drift.py — time each check, sort by duration.

Usage: python -m scripts.tools.profile_check_drift
Reports slow checks (>200ms) and total wall time.
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import check_drift  # noqa: E402


def main() -> int:
    duckdb = check_drift._import_duckdb_or_exit()
    db_path = check_drift._get_db_path()

    shared_con = None
    if db_path.exists():
        try:
            shared_con = duckdb.connect(str(db_path), read_only=True)
        except Exception as exc:
            print(f"Could not open DB ({exc}) — DB checks will skip", file=sys.stderr)

    timings: list[tuple[float, str, bool, str]] = []
    overall_start = time.perf_counter()

    for label, check_fn, _is_advisory, requires_db in check_drift.CHECKS:
        buf = io.StringIO()
        start = time.perf_counter()
        status = "ok"
        try:
            with redirect_stdout(buf):
                if requires_db:
                    check_fn(con=shared_con)
                else:
                    check_fn()
        except Exception as e:
            status = f"error: {type(e).__name__}: {e}"[:80]
        elapsed = time.perf_counter() - start
        timings.append((elapsed, label, requires_db, status))

    overall = time.perf_counter() - overall_start
    if shared_con is not None:
        shared_con.close()

    timings.sort(reverse=True)

    print(f"\nTotal wall time: {overall:.2f}s across {len(timings)} checks\n")
    print(f"{'Time(s)':>8}  {'DB':>3}  Check")
    print("-" * 100)
    for elapsed, label, requires_db, status in timings:
        flag = "DB" if requires_db else ""
        suffix = "" if status == "ok" else f" [{status}]"
        print(f"{elapsed:>8.3f}  {flag:>3}  {label}{suffix}")

    slow = [t for t in timings if t[0] > 0.2]
    print(f"\n{len(slow)} checks exceed 200ms (sum: {sum(t[0] for t in slow):.2f}s)")
    over_1s = [t for t in timings if t[0] > 1.0]
    print(f"{len(over_1s)} checks exceed 1.0s (sum: {sum(t[0] for t in over_1s):.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
