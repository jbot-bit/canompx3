"""Phase D Volume Pilot — Stage D-1 daily forward-shadow append.

Pre-reg: docs/audit/hypotheses/2026-04-17-phase-d-d1-signal-only-shadow.yaml
D-0 locks: P33=1.0529, P67=1.7880 (commit 55b6ba89).

Runs daily from 2026-04-17 through 2026-05-15. Appends one row per NEW
qualifying MNQ COMEX_SETTLE E2 trade day to the shadow log CSV with
window_label = 'forward_shadow'.

APPEND-ONLY. No gate evaluation. No threshold changes. No window changes.

Fail-closed guards
    - Refuses to append outside [2026-04-17, 2026-05-15]
    - Refuses to append a trading_day already in the log (idempotent)
    - Refuses to append rows with trading_day >= today (incomplete day)
    - Prints every action to stdout; nothing is silent

Execute daily:
    uv run python research/phase_d_volume_sizing_pilot_d1_forward_append.py

Exit codes
    0 -> success (even if nothing new to append)
    1 -> error (pre-reg violation, missing data, etc.)
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH

# LOCKED from D-0 pre-reg. Do not change.
P33 = 1.0529
P67 = 1.7880
BUCKET_LOW = 0.5
BUCKET_MID = 1.0
BUCKET_HIGH = 1.5

INSTRUMENT = "MNQ"
SESSION = "COMEX_SETTLE"
ORB_MINUTES = 5
RR_TARGET = 1.5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1

FORWARD_FROM = date(2026, 4, 17)
FORWARD_TO = date(2026, 5, 15)

SHADOW_LOG = Path("docs/audit/results/phase-d-d1-shadow-log.csv")
LOG_COLUMNS = [
    "trading_day",
    "window_label",
    "rel_vol",
    "size_bucket",
    "actual_pnl_r",
    "counterfactual_pnl_r_scaled",
    "outcome",
]


def _size_bucket(rel_vol: float) -> float:
    if rel_vol < P33:
        return BUCKET_LOW
    if rel_vol > P67:
        return BUCKET_HIGH
    return BUCKET_MID


def _existing_trading_days() -> set[date]:
    if not SHADOW_LOG.exists():
        return set()
    days: set[date] = set()
    with SHADOW_LOG.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            td_str = row.get("trading_day", "")
            if td_str:
                days.add(date.fromisoformat(td_str))
    return days


def _load_forward_candidates(existing: set[date]) -> list[tuple]:
    today = date.today()
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        q = f"""
        SELECT
            o.trading_day,
            d.rel_vol_{SESSION} AS rel_vol,
            o.pnl_r,
            o.outcome
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
           AND o.symbol = d.symbol
           AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{INSTRUMENT}'
          AND o.orb_label = '{SESSION}'
          AND o.orb_minutes = {ORB_MINUTES}
          AND o.rr_target = {RR_TARGET}
          AND o.entry_model = '{ENTRY_MODEL}'
          AND o.confirm_bars = {CONFIRM_BARS}
          AND o.trading_day >= '{FORWARD_FROM.isoformat()}'
          AND o.trading_day <= '{FORWARD_TO.isoformat()}'
          AND o.trading_day < '{today.isoformat()}'
          AND o.pnl_r IS NOT NULL
          AND d.rel_vol_{SESSION} IS NOT NULL
        ORDER BY o.trading_day
        """
        rows = con.execute(q).fetchall()
    finally:
        con.close()
    return [r for r in rows if r[0] not in existing]


def _assert_window(rows: list[tuple]) -> None:
    for r in rows:
        td = r[0]
        if td < FORWARD_FROM or td > FORWARD_TO:
            raise RuntimeError(
                f"Forward window breach: trading_day {td} outside "
                f"[{FORWARD_FROM}, {FORWARD_TO}]"
            )
        if td >= date.today():
            raise RuntimeError(
                f"Same-day append breach: trading_day {td} >= today "
                f"{date.today()}. Refusing to append incomplete day."
            )


def _append_rows(rows: list[tuple]) -> None:
    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SHADOW_LOG.exists()
    with SHADOW_LOG.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(LOG_COLUMNS)
        for trading_day, rel_vol, pnl_r, outcome in rows:
            size = _size_bucket(float(rel_vol))
            writer.writerow(
                [
                    trading_day.isoformat(),
                    "forward_shadow",
                    f"{float(rel_vol):.6f}",
                    f"{size:.1f}",
                    f"{float(pnl_r):.6f}",
                    f"{float(pnl_r) * size:.6f}",
                    outcome,
                ]
            )


def main() -> int:
    today = date.today()
    print(f"[{today.isoformat()}] Phase D D-1 forward-shadow append starting.")
    print(f"  Forward window: [{FORWARD_FROM}, {FORWARD_TO}]")
    print(f"  Universe: {INSTRUMENT} {SESSION} O{ORB_MINUTES} RR{RR_TARGET} {ENTRY_MODEL} CB{CONFIRM_BARS}")

    if today < FORWARD_FROM:
        print(f"  Today {today} is before forward window start {FORWARD_FROM}. Nothing to do.")
        return 0
    if today > FORWARD_TO:
        print(
            f"  Today {today} is after forward window end {FORWARD_TO}. "
            f"D-1 forward shadow is closed. Run the review script instead."
        )
        return 0

    existing = _existing_trading_days()
    print(f"  Shadow log has {len(existing)} existing trading_days.")

    new_rows = _load_forward_candidates(existing)
    if not new_rows:
        print("  No new qualifying trading days to append.")
        return 0

    _assert_window(new_rows)
    _append_rows(new_rows)

    print(f"  Appended {len(new_rows)} new rows:")
    for td, rel_vol, pnl_r, outcome in new_rows:
        size = _size_bucket(float(rel_vol))
        print(
            f"    {td.isoformat()}  rel_vol={float(rel_vol):.3f}  "
            f"bucket={size:.1f}x  pnl_r={float(pnl_r):+.3f}  "
            f"scaled={float(pnl_r) * size:+.3f}  outcome={outcome}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
