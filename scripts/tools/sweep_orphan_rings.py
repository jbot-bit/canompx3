from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from trading_app.live.bar_ring import RING_DIR

DEFAULT_RING_DIR = RING_DIR

log = logging.getLogger(__name__)

PidStatus = Literal["alive", "dead", "unknown"]


def _probe_pid_status(pid: int) -> PidStatus:
    if pid <= 0:
        return "unknown"
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            log.warning("ring sweep: tasklist probe failed for pid=%s", pid, exc_info=True)
            return "unknown"
        stdout = result.stdout.strip()
        if stdout.startswith("INFO: No tasks are running"):
            return "dead"
        if not stdout:
            return "unknown"
        for row in csv.reader(stdout.splitlines()):
            if len(row) < 2:
                continue
            try:
                row_pid = int(row[1])
            except ValueError:
                continue
            if row_pid == pid:
                return "alive"
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "alive"
    except OSError:
        log.warning("ring sweep: os.kill probe failed for pid=%s", pid, exc_info=True)
        return "unknown"
    return "alive"


def sweep(
    ring_dir: Path,
    *,
    dry_run: bool = False,
    pid_status: Callable[[int], PidStatus] | None = None,
) -> list[Path]:
    deleted: list[Path] = []
    probe = pid_status or _probe_pid_status
    if not ring_dir.exists():
        return deleted

    for path in sorted(ring_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.info("ring sweep: preserving %s (corrupt JSON)", path)
            continue
        if not isinstance(payload, dict):
            log.info("ring sweep: preserving %s (non-object payload)", path)
            continue

        writer_pid = payload.get("writer_pid")
        if writer_pid is None:
            log.info("ring sweep: preserving %s (missing writer_pid)", path)
            continue
        if isinstance(writer_pid, bool) or not isinstance(writer_pid, int):
            log.info("ring sweep: preserving %s (ambiguous writer_pid=%r)", path, writer_pid)
            continue
        if writer_pid <= 0:
            log.info("ring sweep: preserving %s (non-positive writer_pid=%r)", path, writer_pid)
            continue

        status = probe(writer_pid)
        if status == "alive":
            log.info("ring sweep: preserving %s (writer pid %s alive)", path, writer_pid)
            continue
        if status != "dead":
            log.info("ring sweep: preserving %s (writer pid %s not provably dead)", path, writer_pid)
            continue

        deleted.append(path)
        if dry_run:
            log.info("ring sweep: DRY-RUN would delete %s (writer pid %s dead)", path, writer_pid)
            continue
        try:
            path.unlink()
        except OSError:
            log.warning("ring sweep: failed deleting %s", path, exc_info=True)
            deleted.pop()
            continue
        log.info("ring sweep: deleted %s (writer pid %s dead)", path, writer_pid)

    return deleted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete orphaned live bar ring files whose writer PID is provably dead."
    )
    parser.add_argument(
        "--ring-dir", type=Path, default=DEFAULT_RING_DIR, help="Directory containing live bar ring JSON files."
    )
    parser.add_argument("--dry-run", action="store_true", help="Report orphaned ring files without deleting them.")
    args = parser.parse_args(argv)

    deleted = sweep(args.ring_dir, dry_run=args.dry_run)
    for path in deleted:
        if args.dry_run:
            print(f"DRY-RUN delete {path}")
        else:
            print(f"DELETE {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
