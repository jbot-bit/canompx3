#!/usr/bin/env python3
"""Shadow recorder for MNQ US_DATA_1000 O5 E2 long NOT_F6_INSIDE_PDR.

Pre-registered at:
  docs/audit/hypotheses/2026-04-20-usdata1000-long-not-f6-shadow-v1.yaml

Zero-capital observational contract. Reads canonical daily_features +
orb_outcomes and appends one row per post-peek trade day to the ledger:
  docs/audit/shadow_ledgers/usdata1000-long-not-f6-shadow-ledger.md

Also refreshes a small status surface at:
  docs/audit/results/usdata1000-long-not-f6-shadow-status.md

Semantics locked by the YAML:
  - fresh_oos_window.start = 2026-04-17
  - signal = NOT_F6_INSIDE_PDR
  - descriptor = F5_BELOW_PDL
  - both RR 1.0 and RR 1.5 outcomes must be complete for a day to log
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

FRESH_OOS_START = date(2026, 4, 17)
PEEKED_THROUGH = date(2026, 4, 16)
INSTRUMENT = "MNQ"
SESSION = "US_DATA_1000"
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGETS = (1.0, 1.5)
DIRECTION = "long"

LEDGER_PATH = PROJECT_ROOT / "docs/audit/shadow_ledgers/usdata1000-long-not-f6-shadow-ledger.md"
STATUS_MD_PATH = PROJECT_ROOT / "docs/audit/results/usdata1000-long-not-f6-shadow-status.md"
YAML_PATH = PROJECT_ROOT / "docs/audit/hypotheses/2026-04-20-usdata1000-long-not-f6-shadow-v1.yaml"

QUERY = """
WITH base AS (
  SELECT
    d.trading_day,
    (d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 AS orb_mid,
    d.prev_day_low,
    d.prev_day_high,
    CASE
      WHEN ((d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 > d.prev_day_low)
       AND ((d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 < d.prev_day_high)
      THEN 1 ELSE 0
    END AS f6_inside_pdr,
    CASE
      WHEN ((d.orb_US_DATA_1000_high + d.orb_US_DATA_1000_low) / 2.0 < d.prev_day_low)
      THEN 1 ELSE 0
    END AS f5_below_pdl
  FROM daily_features d
  WHERE d.symbol = ?
    AND d.orb_minutes = ?
    AND d.trading_day >= ?
    AND d.prev_day_low IS NOT NULL
    AND d.prev_day_high IS NOT NULL
    AND d.orb_US_DATA_1000_high IS NOT NULL
    AND d.orb_US_DATA_1000_low IS NOT NULL
),
trade_days AS (
  SELECT
    o.trading_day,
    MAX(CASE WHEN o.rr_target = 1.0 THEN o.outcome END) AS rr_1p0_outcome,
    MAX(CASE WHEN o.rr_target = 1.0 THEN o.pnl_r END) AS rr_1p0_pnl_r,
    MAX(CASE WHEN o.rr_target = 1.5 THEN o.outcome END) AS rr_1p5_outcome,
    MAX(CASE WHEN o.rr_target = 1.5 THEN o.pnl_r END) AS rr_1p5_pnl_r
  FROM orb_outcomes o
  WHERE o.symbol = ?
    AND o.orb_label = ?
    AND o.orb_minutes = ?
    AND o.entry_model = ?
    AND o.confirm_bars = ?
    AND o.rr_target IN (1.0, 1.5)
    AND o.target_price > o.entry_price
    AND o.outcome NOT IN ('skip_no_break', 'skip_missing_data')
    AND o.trading_day >= ?
  GROUP BY 1
  HAVING COUNT(DISTINCT o.rr_target) = 2
)
SELECT
  b.trading_day,
  b.orb_mid,
  b.prev_day_low,
  b.prev_day_high,
  b.f5_below_pdl,
  t.rr_1p0_outcome,
  t.rr_1p0_pnl_r,
  t.rr_1p5_outcome,
  t.rr_1p5_pnl_r
FROM base b
JOIN trade_days t USING (trading_day)
WHERE b.f6_inside_pdr = 0
ORDER BY b.trading_day
"""

LEDGER_HEADER = """# US_DATA_1000 Long NOT_F6 — Shadow Ledger

**Pre-registration:** [`docs/audit/hypotheses/2026-04-20-usdata1000-long-not-f6-shadow-v1.yaml`](../hypotheses/2026-04-20-usdata1000-long-not-f6-shadow-v1.yaml)

**Canonical predicate:** `NOT_F6_INSIDE_PDR`, where `F6_INSIDE_PDR := orb_mid > prev_day_low AND orb_mid < prev_day_high`

**Descriptor:** `F5_BELOW_PDL := orb_mid < prev_day_low`

**Fresh OOS window:** `trading_day >= 2026-04-17` because candidate-lane validation already consumed OOS through `2026-04-16`.

**Contract:** Zero capital. Observational only. Idempotent append-only ledger.
Per-day row requires both RR 1.0 and RR 1.5 trades complete.

| trading_day | orb_mid | prev_day_low | prev_day_high | descriptor_f5 | RR1.0 outcome | RR1.0 pnl_r | RR1.5 outcome | RR1.5 pnl_r |
|---|---:|---:|---:|---|---|---:|---|---:|
"""


def _existing_trading_days(path: Path) -> set[str]:
    if not path.exists():
        return set()
    days: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        candidate = parts[1]
        if len(candidate) == 10 and candidate[4] == "-" and candidate[7] == "-":
            days.add(candidate)
    return days


def _fmt_float(value: float | None, prec: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{prec}f}"


def _render_status(rows: list[tuple], appended: int, db_max_day: date | None) -> str:
    n_total = len(rows)
    n_f5 = sum(int(r[4]) for r in rows)
    rr1_vals = [float(r[6]) for r in rows]
    rr15_vals = [float(r[8]) for r in rows]
    rr1_expr = sum(rr1_vals) / n_total if n_total else None
    rr15_expr = sum(rr15_vals) / n_total if n_total else None

    lines = [
        "# US_DATA_1000 Long NOT_F6 — Shadow Status",
        "",
        f"**Pre-registration:** `{YAML_PATH.relative_to(PROJECT_ROOT)}`",
        f"**Ledger:** `{LEDGER_PATH.relative_to(PROJECT_ROOT)}`",
        f"**Canonical DB max day in this lane universe:** `{db_max_day}`" if db_max_day else "**Canonical DB max day in this lane universe:** `None`",
        f"**Peeked through:** `{PEEKED_THROUGH}`",
        f"**Fresh OOS start:** `{FRESH_OOS_START}`",
        "",
        "## Current ledger state",
        "",
        f"- Total post-peek trade days logged: **{n_total}**",
        f"- New rows appended on latest run: **{appended}**",
        f"- F5 descriptor fires inside NOT_F6: **{n_f5}**",
        f"- RR1.0 shadow ExpR: `{rr1_expr:+.4f}`" if rr1_expr is not None else "- RR1.0 shadow ExpR: `N/A`",
        f"- RR1.5 shadow ExpR: `{rr15_expr:+.4f}`" if rr15_expr is not None else "- RR1.5 shadow ExpR: `N/A`",
        "",
        "## Gate posture",
        "",
        "- This remains `signal-only` shadow. No promotion, no lane allocation edits, no live-capital action.",
        "- Re-open candidate review only after forward OOS is no longer thin per the locked YAML gate.",
        "",
    ]
    if rows:
        lines.extend(
            [
                "## Logged trade days",
                "",
                "| trading_day | descriptor_f5 | RR1.0 pnl_r | RR1.5 pnl_r |",
                "|---|---|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row[0]} | {'YES' if int(row[4]) == 1 else 'NO'} | "
                f"{float(row[6]):+.4f} | {float(row[8]):+.4f} |"
            )
    else:
        lines.extend(
            [
                "## Logged trade days",
                "",
                f"No eligible post-peek trade days are present yet. Current DB coverage only reaches `{db_max_day}`.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    db = GOLD_DB_PATH
    if not db.exists():
        print(f"FATAL: DB not found at {db}", file=sys.stderr)
        return 2

    if HOLDOUT_SACRED_FROM > FRESH_OOS_START:
        print(
            f"FATAL: HOLDOUT_SACRED_FROM={HOLDOUT_SACRED_FROM} > FRESH_OOS_START={FRESH_OOS_START}. "
            "Shadow invalidated — re-pre-register required.",
            file=sys.stderr,
        )
        return 2

    existing = _existing_trading_days(LEDGER_PATH)

    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            QUERY,
            [
                INSTRUMENT,
                ORB_MINUTES,
                FRESH_OOS_START,
                INSTRUMENT,
                SESSION,
                ORB_MINUTES,
                ENTRY_MODEL,
                CONFIRM_BARS,
                FRESH_OOS_START,
            ],
        ).fetchall()
        db_max_day = con.execute(
            """
            SELECT MAX(trading_day)
            FROM orb_outcomes
            WHERE symbol = ?
              AND orb_label = ?
              AND orb_minutes = ?
              AND entry_model = ?
              AND confirm_bars = ?
              AND rr_target IN (1.0, 1.5)
              AND target_price > entry_price
              AND outcome NOT IN ('skip_no_break', 'skip_missing_data')
            """,
            [INSTRUMENT, SESSION, ORB_MINUTES, ENTRY_MODEL, CONFIRM_BARS],
        ).fetchone()[0]

    if not LEDGER_PATH.exists():
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEDGER_PATH.write_text(LEDGER_HEADER, encoding="utf-8")

    new_rows = [r for r in rows if str(r[0]) not in existing]

    if new_rows:
        with LEDGER_PATH.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(
                    f"| {row[0]} | {_fmt_float(float(row[1]))} | {_fmt_float(float(row[2]))} | "
                    f"{_fmt_float(float(row[3]))} | {'YES' if int(row[4]) == 1 else 'NO'} | "
                    f"{row[5]} | {_fmt_float(float(row[6]))} | {row[7]} | {_fmt_float(float(row[8]))} |\n"
                )

    STATUS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_MD_PATH.write_text(_render_status(rows, len(new_rows), db_max_day), encoding="utf-8")

    print(f"Ledger rows total: {len(rows)}")
    print(f"New rows appended: {len(new_rows)}")
    print(f"Wrote {LEDGER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {STATUS_MD_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
