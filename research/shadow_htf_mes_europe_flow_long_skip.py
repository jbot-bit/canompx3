#!/usr/bin/env python3
"""Shadow recorder for MES EUROPE_FLOW O15 E2 CB1 long HTF skip-rule.

Pre-registered at:
  docs/audit/hypotheses/2026-04-18-htf-mes-europe-flow-long-skip-rule-shadow.yaml

# e2-lookahead-policy: tainted
# Reclassified 2026-04-28 (was: not-predictor) after real-data audit on this exact lane.
# The filter predicate `orb_EUROPE_FLOW_break_dir = 'long'` (line ~71) is post-entry for
# ~42.6% of MES EUROPE_FLOW O15 E2 CB1 RR1.5 IS trades (real-data measurement, N=1719).
# break_bar_volume and rel_vol are also selected for the ledger but NOT used as predicates.
# Per postmortem 2026-04-21-e2-break-bar-lookahead.md § 5.2 and quant-audit-protocol RULE 6.3,
# break_dir as a fire-day SELECTOR for E2 = look-ahead even though VWAPBreakDirectionFilter
# canonical doctrine (config.py:2648) claims "known at entry time." Doctrine is partially
# wrong: range-touch-then-reverse fakeouts make break_dir post-entry on ~42% of E2 fills.
# Blast radius: ZERO CAPITAL — observational shadow ledger only, no live deployment risk.
# Action required: re-pre-register the shadow with a pre-break direction proxy (e.g.,
# orb_high > prev_week_high alone, or pre-break VWAP slope) before any deployment use.
# Registry: docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md row 27

Zero-capital observational contract. Reads canonical tables (daily_features
+ orb_outcomes) and appends one row per HTF fire trading-day to the ledger:
  docs/audit/shadow_ledgers/htf-mes-europe-flow-long-skip-rule-ledger.md

Semantics locked by the YAML:
  - canonical_predicate: orb_EUROPE_FLOW_break_dir='long'
                         AND prev_week_high IS NOT NULL
                         AND orb_EUROPE_FLOW_high > prev_week_high
  - fresh_oos_window.start = 2026-04-18 (post-v1-scan peek boundary)
  - holdout binding: HOLDOUT_SACRED_FROM imported, never hardcoded
  - idempotent: existing ledger rows read, duplicates skipped by trading_day

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/shadow_htf_mes_europe_flow_long_skip.py
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

# =========================================================================
# Locked shadow parameters (from YAML; DO NOT modify without re-prereg)
# =========================================================================

FRESH_OOS_START = date(2026, 4, 18)  # peek_contamination boundary per YAML
INSTRUMENT = "MES"
SESSION = "EUROPE_FLOW"
ORB_MINUTES = 15
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGETS = (1.5, 2.0)
DIRECTION = "long"

LEDGER_PATH = PROJECT_ROOT / "docs/audit/shadow_ledgers/htf-mes-europe-flow-long-skip-rule-ledger.md"
YAML_PATH = PROJECT_ROOT / "docs/audit/hypotheses/2026-04-18-htf-mes-europe-flow-long-skip-rule-shadow.yaml"


QUERY = """
SELECT
  d.trading_day,
  d.orb_EUROPE_FLOW_break_ts,
  d.orb_EUROPE_FLOW_high,
  d.prev_week_high,
  d.orb_EUROPE_FLOW_break_bar_volume,
  d.rel_vol_EUROPE_FLOW,
  MAX(CASE WHEN o.rr_target = 1.5 THEN o.outcome END) AS rr_1p5_outcome,
  MAX(CASE WHEN o.rr_target = 1.5 THEN o.pnl_r  END) AS rr_1p5_pnl_r,
  MAX(CASE WHEN o.rr_target = 2.0 THEN o.outcome END) AS rr_2p0_outcome,
  MAX(CASE WHEN o.rr_target = 2.0 THEN o.pnl_r  END) AS rr_2p0_pnl_r
FROM daily_features d
JOIN orb_outcomes o USING (trading_day, symbol, orb_minutes)
WHERE d.symbol = ?
  AND d.orb_minutes = ?
  AND d.orb_EUROPE_FLOW_break_dir = ?
  AND d.prev_week_high IS NOT NULL
  AND d.orb_EUROPE_FLOW_high > d.prev_week_high
  AND d.trading_day >= ?
  AND o.orb_label = ?
  AND o.orb_minutes = ?
  AND o.entry_model = ?
  AND o.confirm_bars = ?
  AND o.rr_target IN (1.5, 2.0)
  AND o.outcome IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6
HAVING COUNT(DISTINCT o.rr_target) = 2
ORDER BY d.trading_day
"""


LEDGER_HEADER = """# HTF MES EUROPE_FLOW long skip-rule — Shadow Ledger

**Pre-registration:** [`docs/audit/hypotheses/2026-04-18-htf-mes-europe-flow-long-skip-rule-shadow.yaml`](../hypotheses/2026-04-18-htf-mes-europe-flow-long-skip-rule-shadow.yaml)

**Canonical predicate:** `orb_EUROPE_FLOW_break_dir = 'long' AND prev_week_high IS NOT NULL AND orb_EUROPE_FLOW_high > prev_week_high`

**Fresh OOS window:** `trading_day >= 2026-04-18` (post-v1-scan peek boundary).

**Contract:** Zero capital. Observational only. Idempotent append-only ledger.
Per-day row requires both RR 1.5 and RR 2.0 trades complete (outcome NOT NULL).

RULE 3.2 directional-only verdict is an acceptable outcome at calendar cap
(2028-06-30) per YAML `review_verdict_at_cap`.

| trading_day | break_ts | orb_high | prev_week_high | break_bar_vol | rel_vol | RR1.5 outcome | RR1.5 pnl_r | RR2.0 outcome | RR2.0 pnl_r |
|---|---|---|---|---|---|---|---|---|---|
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


def _fmt(value, prec: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{prec}f}"
    return str(value)


def main() -> int:
    db = GOLD_DB_PATH
    if not db.exists():
        print(f"FATAL: DB not found at {db}", file=sys.stderr)
        return 2

    # Anti-lookahead sanity: the canonical holdout must be <= the shadow's fresh
    # window start. If HOLDOUT_SACRED_FROM is ever moved forward past
    # FRESH_OOS_START, the shadow is invalid until the YAML is re-pre-registered.
    if HOLDOUT_SACRED_FROM > FRESH_OOS_START:
        print(
            f"FATAL: HOLDOUT_SACRED_FROM={HOLDOUT_SACRED_FROM} > FRESH_OOS_START="
            f"{FRESH_OOS_START}. Shadow invalidated — re-pre-register required.",
            file=sys.stderr,
        )
        return 2

    print(f"Canonical DB: {db}")
    print(f"Holdout (Mode A): {HOLDOUT_SACRED_FROM} | Fresh OOS: {FRESH_OOS_START}+")
    print(f"YAML: {YAML_PATH}")
    print(f"Ledger: {LEDGER_PATH}")
    print()

    existing = _existing_trading_days(LEDGER_PATH)
    print(f"Existing ledger rows: {len(existing)}")

    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            QUERY,
            [
                INSTRUMENT,
                ORB_MINUTES,
                DIRECTION,
                FRESH_OOS_START,
                SESSION,
                ORB_MINUTES,
                ENTRY_MODEL,
                CONFIRM_BARS,
            ],
        ).fetchall()

    new_rows = [r for r in rows if str(r[0]) not in existing]
    print(f"Candidate fire-days (complete outcomes): {len(rows)}")
    print(f"New rows to append: {len(new_rows)}")

    if not LEDGER_PATH.exists():
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEDGER_PATH.write_text(LEDGER_HEADER, encoding="utf-8")
        print(f"Initialised ledger at {LEDGER_PATH}")

    if not new_rows:
        print("No new fires to record. Ledger unchanged.")
        return 0

    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        for r in new_rows:
            td, bts, ohigh, pwh, bvol, rvol, out15, pnl15, out20, pnl20 = r
            f.write(
                f"| {td} | {bts} | {_fmt(ohigh)} | {_fmt(pwh)} | "
                f"{_fmt(bvol, 0)} | {_fmt(rvol, 3)} | "
                f"{_fmt(out15)} | {_fmt(pnl15)} | {_fmt(out20)} | {_fmt(pnl20)} |\n"
            )
    print(f"Appended {len(new_rows)} row(s) to {LEDGER_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
