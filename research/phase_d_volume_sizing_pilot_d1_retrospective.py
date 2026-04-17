"""Phase D Volume Pilot — Stage D-1 kickoff retrospective.

Pre-reg: docs/audit/hypotheses/2026-04-17-phase-d-d1-signal-only-shadow.yaml
Locks: P33=1.0529, P67=1.7880 from D-0 (commit 55b6ba89).

Computes the locked_descriptive_oos_read for the Jan 1 - Apr 16 2026 window.
Runs ONCE at D-1 kickoff. Writes one row per qualifying E2 trade day to
the shadow log CSV with window_label = 'retrospective'.

Forward shadow rows (window_label = 'forward_shadow') are appended
daily by a separate process from 2026-04-17 through 2026-05-15. Final
verdict is produced by phase_d_volume_sizing_pilot_d1_review.py on
2026-05-15 against the combined log.

Execute:
    uv run python research/phase_d_volume_sizing_pilot_d1_retrospective.py

Fail-closed guards
    - Only reads trading_day in [2026-01-01, 2026-04-16]
    - Raises if any row outside the window slips through
    - Raises if shadow log already has retrospective rows (one-shot only)
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH

# LOCKED from D-0. DO NOT modify.
P33_REL_VOL_COMEX_SETTLE = 1.0529
P67_REL_VOL_COMEX_SETTLE = 1.7880
BUCKET_LOW = 0.5
BUCKET_MID = 1.0
BUCKET_HIGH = 1.5

INSTRUMENT = "MNQ"
SESSION = "COMEX_SETTLE"
ORB_MINUTES = 5
RR_TARGET = 1.5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1

RETROSPECTIVE_FROM = date(2026, 1, 1)
RETROSPECTIVE_TO = date(2026, 4, 16)

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
    if rel_vol < P33_REL_VOL_COMEX_SETTLE:
        return BUCKET_LOW
    if rel_vol > P67_REL_VOL_COMEX_SETTLE:
        return BUCKET_HIGH
    return BUCKET_MID


def _load_retrospective_rows() -> list[tuple]:
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
          AND o.trading_day >= '{RETROSPECTIVE_FROM.isoformat()}'
          AND o.trading_day <= '{RETROSPECTIVE_TO.isoformat()}'
          AND o.pnl_r IS NOT NULL
          AND d.rel_vol_{SESSION} IS NOT NULL
        ORDER BY o.trading_day
        """
        return con.execute(q).fetchall()
    finally:
        con.close()


def _assert_window(rows: list[tuple]) -> None:
    for r in rows:
        td = r[0]
        if td < RETROSPECTIVE_FROM or td > RETROSPECTIVE_TO:
            raise RuntimeError(
                f"Retrospective window breach: trading_day {td} outside "
                f"[{RETROSPECTIVE_FROM}, {RETROSPECTIVE_TO}]"
            )


def _assert_one_shot() -> None:
    """Shadow log must not already contain retrospective rows (one-shot)."""
    if not SHADOW_LOG.exists():
        return
    with SHADOW_LOG.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("window_label") == "retrospective":
                raise RuntimeError(
                    f"One-shot violation: shadow log already contains "
                    f"retrospective rows at {SHADOW_LOG}. Delete the file "
                    f"manually and re-run only if you have a pre-reg-compliant "
                    f"reason (you almost certainly do not)."
                )


def _write_rows(rows: list[tuple]) -> None:
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
                    "retrospective",
                    f"{float(rel_vol):.6f}",
                    f"{size:.1f}",
                    f"{float(pnl_r):.6f}",
                    f"{float(pnl_r) * size:.6f}",
                    outcome,
                ]
            )


def _summary(rows: list[tuple]) -> None:
    if not rows:
        print("No qualifying rows found in retrospective window.")
        return
    n = len(rows)
    buckets = {BUCKET_LOW: 0, BUCKET_MID: 0, BUCKET_HIGH: 0}
    sum_actual = 0.0
    sum_scaled = 0.0
    for _, rel_vol, pnl_r, _outcome in rows:
        size = _size_bucket(float(rel_vol))
        buckets[size] += 1
        sum_actual += float(pnl_r)
        sum_scaled += float(pnl_r) * size
    print(f"Retrospective rows written: {n}")
    print(f"  2026-01-01 to {rows[-1][0]}")
    print(f"  Bucket 0.5x: {buckets[BUCKET_LOW]} ({buckets[BUCKET_LOW] / n:.1%})")
    print(f"  Bucket 1.0x: {buckets[BUCKET_MID]} ({buckets[BUCKET_MID] / n:.1%})")
    print(f"  Bucket 1.5x: {buckets[BUCKET_HIGH]} ({buckets[BUCKET_HIGH] / n:.1%})")
    print(f"  Sum pnl_r baseline (1x): {sum_actual:.4f}")
    print(f"  Sum pnl_r scaled:        {sum_scaled:.4f}")
    print(f"  Delta R:                 {sum_scaled - sum_actual:+.4f}")


def main() -> int:
    print("Phase D D-1 retrospective kickoff starting...")
    _assert_one_shot()
    rows = _load_retrospective_rows()
    if not rows:
        print(
            f"WARNING: no rows in retrospective window "
            f"[{RETROSPECTIVE_FROM}, {RETROSPECTIVE_TO}]. "
            f"Shadow log will be created with header only. "
            f"Forward shadow will still accumulate from 2026-04-17 onwards."
        )
        # Still write header so daily forward-append code can open-for-append.
        SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        if not SHADOW_LOG.exists():
            with SHADOW_LOG.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(LOG_COLUMNS)
        return 0
    _assert_window(rows)
    _write_rows(rows)
    _summary(rows)
    print(f"Shadow log at {SHADOW_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
