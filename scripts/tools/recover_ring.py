#!/usr/bin/env python3
"""Recover a preserved live-bars ring file into bars_1m and clear it.

Operator escape valve for the audit-fix #2 preserve-on-failure state in
`trading_app.live.session_orchestrator.post_session`. When `flush_to_db()`
returns 0 with bars_captured > 0, the ring file is deliberately preserved
(data-loss guard); this script reads those bars off-disk and pushes them
into bars_1m via the canonical persister path.

Usage:
    python scripts/tools/recover_ring.py MNQ
    python scripts/tools/recover_ring.py MNQ --dry-run    # show, do not write
    python scripts/tools/recover_ring.py MNQ --keep-ring  # do not delete on success

Behavior:
    - Reads data/live_bars/<SYMBOL>.json via canonical `bar_ring.read_bar_ring`.
    - Reconstructs Bar objects and delegates to `BarPersister.flush_to_db`
      (NEVER re-encode bars_1m INSERT — institutional-rigor § 4).
    - Uses `pipeline.paths.GOLD_DB_PATH` (canonical DB path — § 10).
    - Deletes the ring file on success unless --keep-ring (default: delete).
    - Fail-closed: any parse or write error aborts with non-zero exit and
      preserves the ring file for re-attempt.

Exit codes:
    0  - bars recovered and ring cleared (or --dry-run preview shown).
    1  - ring file missing or empty (nothing to recover).
    2  - ring parse / reconstruction error.
    3  - flush_to_db returned 0 (DB write failure — ring preserved).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports deferred under sys.path bootstrap so direct invocation works.
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.live import bar_ring  # noqa: E402
from trading_app.live.bar_aggregator import Bar  # noqa: E402
from trading_app.live.bar_persister import BarPersister  # noqa: E402

log = logging.getLogger("recover_ring")


def _bar_from_ring_entry(symbol: str, entry: dict[str, Any]) -> Bar:
    """Reconstruct a Bar from the ring JSON dict.

    Raises ValueError on any missing/invalid field. The ring serializer in
    `bar_ring._serialize_bar` is the canonical inverse — keep this in sync.
    """
    ts_raw = entry.get("ts_utc")
    if not isinstance(ts_raw, str):
        raise ValueError(f"missing/invalid ts_utc: {ts_raw!r}")
    ts = datetime.fromisoformat(ts_raw)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    try:
        return Bar(
            ts_utc=ts,
            open=float(entry["open"]),
            high=float(entry["high"]),
            low=float(entry["low"]),
            close=float(entry["close"]),
            volume=int(entry["volume"]),
            symbol=symbol,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"reconstruct failed for entry {entry!r}: {exc}") from exc


def recover(
    symbol: str,
    *,
    db_path: str | None = None,
    dry_run: bool = False,
    keep_ring: bool = False,
) -> int:
    """Recover bars for ``symbol`` from the ring file. Returns exit code.

    Separated from main() for testing — pass an explicit ``db_path`` in tests.
    """
    snap = bar_ring.read_bar_ring(symbol)
    if snap.is_empty():
        print(f"[recover_ring] {symbol}: ring is empty or missing — nothing to recover.")
        return 1

    print(
        f"[recover_ring] {symbol}: {len(snap.bars)} bar(s) in ring, "
        f"writer_pid={snap.writer_pid}, updated_utc={snap.updated_utc}",
    )

    try:
        bars = [_bar_from_ring_entry(symbol, entry) for entry in snap.bars]
    except ValueError as exc:
        print(f"[recover_ring] {symbol}: ERROR reconstructing bars: {exc}", file=sys.stderr)
        return 2

    ts_min = min(b.ts_utc for b in bars)
    ts_max = max(b.ts_utc for b in bars)
    valid_count = sum(1 for b in bars if b.is_valid())
    print(
        f"[recover_ring] {symbol}: window {ts_min} -> {ts_max}; {valid_count}/{len(bars)} pass is_valid()",
    )

    if dry_run:
        print(f"[recover_ring] {symbol}: --dry-run; not writing to bars_1m.")
        return 0

    # Canonical delegation: never re-encode the bars_1m write path.
    # BarPersister.flush_to_db applies the same DELETE+INSERT idempotent
    # contract used by every live session shutdown.
    target_db = db_path or str(GOLD_DB_PATH)
    bp = BarPersister(symbol, db_path=target_db)
    # Populate the private bar list directly so flush_to_db sees them.
    # Read-only fields _bars / _lock are class-internal; recovery is the one
    # legitimate caller outside of append(). (Alternative — calling append() —
    # would also re-enqueue to the ring writer, which is the wrong direction.)
    with bp._lock:
        bp._bars.extend(bars)
    # flush_to_db catches only (duckdb.Error, OSError) and returns 0; any other
    # exception class would otherwise escape with an uncontrolled traceback,
    # breaking the documented fail-closed exit-code contract (exit 3). Map every
    # write failure to the same operator-facing outcome: ring preserved, exit 3.
    try:
        n_persisted = bp.flush_to_db()
    except Exception as exc:  # noqa: BLE001 — operator escape valve: any write failure → exit 3
        print(
            f"[recover_ring] {symbol}: ERROR flush_to_db raised {type(exc).__name__}: {exc} — "
            "ring file PRESERVED for re-attempt. Check gold.db permissions / "
            "free space and re-run.",
            file=sys.stderr,
        )
        return 3
    if n_persisted == 0:
        print(
            f"[recover_ring] {symbol}: ERROR flush_to_db returned 0 — "
            "ring file PRESERVED for re-attempt. Check gold.db permissions / "
            "free space and re-run.",
            file=sys.stderr,
        )
        return 3
    print(f"[recover_ring] {symbol}: {n_persisted} bar(s) written to bars_1m at {target_db}")

    if keep_ring:
        print(f"[recover_ring] {symbol}: --keep-ring; leaving ring file in place.")
    else:
        bar_ring.clear_ring(symbol)
        print(f"[recover_ring] {symbol}: ring file deleted.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbol", help="Instrument symbol (e.g. MNQ, MES, MGC)")
    parser.add_argument("--dry-run", action="store_true", help="Show bars without writing to bars_1m")
    parser.add_argument("--keep-ring", action="store_true", help="Do not delete the ring file after success")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Override gold.db path (default: pipeline.paths.GOLD_DB_PATH)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    return recover(
        args.symbol,
        db_path=args.db_path,
        dry_run=args.dry_run,
        keep_ring=args.keep_ring,
    )


if __name__ == "__main__":
    sys.exit(main())
