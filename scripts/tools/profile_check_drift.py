#!/usr/bin/env python3
"""Profile pipeline/check_drift.py check durations."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import check_drift  # noqa: E402


def _run_profile() -> dict[str, Any]:
    duckdb = check_drift._import_duckdb_or_exit()
    db_path = check_drift._get_db_path()

    shared_con = None
    if db_path.exists():
        try:
            shared_con = duckdb.connect(str(db_path), read_only=True)
        except Exception as exc:
            print(f"Could not open DB ({exc}) - DB checks will skip", file=sys.stderr)

    timings: list[dict[str, Any]] = []
    overall_start = time.perf_counter()

    try:
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
            timings.append(
                {
                    "duration_seconds": elapsed,
                    "label": label,
                    "requires_db": requires_db,
                    "status": status,
                }
            )
    finally:
        if shared_con is not None:
            shared_con.close()

    overall = time.perf_counter() - overall_start
    timings.sort(key=lambda item: item["duration_seconds"], reverse=True)
    slow = [item for item in timings if item["duration_seconds"] > 0.2]
    over_1s = [item for item in timings if item["duration_seconds"] > 1.0]

    return {
        "total_seconds": overall,
        "check_count": len(timings),
        "slow_threshold_seconds": 0.2,
        "slow_count": len(slow),
        "slow_sum_seconds": sum(item["duration_seconds"] for item in slow),
        "over_1s_count": len(over_1s),
        "over_1s_sum_seconds": sum(item["duration_seconds"] for item in over_1s),
        "checks": timings,
    }


def _format_text(profile: dict[str, Any]) -> str:
    out = io.StringIO()
    print(f"\nTotal wall time: {profile['total_seconds']:.2f}s across {profile['check_count']} checks\n", file=out)
    print(f"{'Time(s)':>8}  {'DB':>3}  Check", file=out)
    print("-" * 100, file=out)
    for item in profile["checks"]:
        flag = "DB" if item["requires_db"] else ""
        suffix = "" if item["status"] == "ok" else f" [{item['status']}]"
        print(f"{item['duration_seconds']:>8.3f}  {flag:>3}  {item['label']}{suffix}", file=out)

    print(
        f"\n{profile['slow_count']} checks exceed 200ms (sum: {profile['slow_sum_seconds']:.2f}s)",
        file=out,
    )
    print(
        f"{profile['over_1s_count']} checks exceed 1.0s (sum: {profile['over_1s_sum_seconds']:.2f}s)",
        file=out,
    )
    return out.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile pipeline/check_drift.py check durations")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable timing data")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.json:
        with redirect_stdout(sys.stderr):
            profile = _run_profile()
        print(json.dumps(profile, indent=2, sort_keys=True))
    else:
        profile = _run_profile()
        print(_format_text(profile), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
