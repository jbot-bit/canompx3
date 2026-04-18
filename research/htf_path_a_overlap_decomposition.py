#!/usr/bin/env python3
"""HTF Path A prev-week × prev-month overlap decomposition.

Closes review-finding #1 against docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md
§ "Adversarial-audit addendum — 2026-04-19".

Purpose:
  For the two MES wrong-sign cells identified in the addendum
  (EUROPE_FLOW long RR2.0 and TOKYO_OPEN long RR2.0), decompose the
  prev-month v1 "on-filter" trade population into:
    - OVERLAP   : trading_days where BOTH prev_week v1 AND prev_month v1
                  predicates fire
    - NON_OVERLAP : trading_days where prev_month v1 fires but prev_week
                  v1 does NOT
    - COMBINED  : OVERLAP ∪ NON_OVERLAP  (= full prev-month on-filter set)

Computes N, mean pnl_r, one-sample two-tailed t vs 0, raw p per subset.
Tests the addendum's claim that MES EUROPE_FLOW's significance is overlap-
driven while MES TOKYO_OPEN's is non-overlap-driven.

Reads ONLY canonical tables (daily_features + orb_outcomes). No writes to
validated_setups or experimental_strategies.

Holdout: Mode A. IS = trading_day < HOLDOUT_SACRED_FROM.

Cell axes (IDENTICAL to prev_week_v1 and prev_month_v1 scans by design):
  ENTRY_MODEL=E2, CONFIRM_BARS=1, ORB_MINUTES=15, direction='long', rr=2.0.

Output: docs/audit/results/2026-04-19-htf-path-a-overlap-decomposition.md

Usage:
  DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/htf_path_a_overlap_decomposition.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from scipy import stats as _sstats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# =============================================================================
# CELLS UNDER DECOMPOSITION
# =============================================================================

# Axes identical to the two v1 scans (ENTRY_MODEL=E2, CB=1, O15, direction=long,
# RR=2.0). Only the HTF predicate differs (prev_week_high vs prev_month_high).
ORB_MINUTES = 15
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR = 2.0
DIRECTION = "long"

CELLS = [
    {
        "instrument": "MES",
        "session": "EUROPE_FLOW",
        "addendum_cell_id": "#18",
        "addendum_claim_combined_t": -3.53,
        "addendum_claim_overlap_t": -4.018,
        "addendum_claim_nonoverlap_t": -1.384,
    },
    {
        "instrument": "MES",
        "session": "TOKYO_OPEN",
        "addendum_cell_id": "#14",
        "addendum_claim_combined_t": -3.79,
        "addendum_claim_overlap_t": -1.863,
        "addendum_claim_nonoverlap_t": -3.367,
    },
]

RESULT_PATH = (
    PROJECT_ROOT
    / "docs/audit/results/2026-04-19-htf-path-a-overlap-decomposition.md"
)

# =============================================================================
# SQL PREDICATES (copied from the two v1 scans — same direction='long' branch)
# =============================================================================


def _predicate_pw_long(session: str) -> str:
    return (
        f"d.orb_{session}_break_dir = 'long' "
        f"AND d.prev_week_high IS NOT NULL "
        f"AND d.orb_{session}_high > d.prev_week_high"
    )


def _predicate_pm_long(session: str) -> str:
    return (
        f"d.orb_{session}_break_dir = 'long' "
        f"AND d.prev_month_high IS NOT NULL "
        f"AND d.orb_{session}_high > d.prev_month_high"
    )


# =============================================================================
# LOAD CELL TRADES (on one predicate at a time)
# =============================================================================


def _load_on_filter_trades(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    session: str,
    predicate_sql: str,
    holdout: date,
) -> list[tuple[date, float]]:
    """Return [(trading_day, pnl_r)] for this cell × predicate, IS only."""
    sql = f"""
        SELECT o.trading_day, o.pnl_r
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = ?
          AND o.orb_minutes = ?
          AND o.entry_model = ?
          AND o.confirm_bars = ?
          AND o.rr_target = ?
          AND d.orb_{session}_break_dir = ?
          AND ({predicate_sql})
          AND o.trading_day < ?
          AND o.pnl_r IS NOT NULL
        ORDER BY o.trading_day
    """
    rows = con.execute(
        sql,
        [
            instrument,
            session,
            ORB_MINUTES,
            ENTRY_MODEL,
            CONFIRM_BARS,
            RR,
            DIRECTION,
            holdout,
        ],
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


# =============================================================================
# STATISTICS
# =============================================================================


@dataclass
class SubsetStats:
    n: int
    mean: float | None
    std: float | None
    t: float | None
    raw_p: float | None


def _stats(pnl: list[float]) -> SubsetStats:
    n = len(pnl)
    if n < 2:
        mean = pnl[0] if n == 1 else None
        return SubsetStats(n=n, mean=mean, std=None, t=None, raw_p=None)
    mean = sum(pnl) / n
    var = sum((p - mean) ** 2 for p in pnl) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return SubsetStats(n=n, mean=mean, std=0.0, t=None, raw_p=None)
    se = std / math.sqrt(n)
    t = mean / se
    raw_p = 2.0 * (1.0 - _sstats.t.cdf(abs(t), df=n - 1))
    return SubsetStats(n=n, mean=float(mean), std=float(std), t=float(t), raw_p=float(raw_p))


# =============================================================================
# DECOMPOSITION FOR ONE CELL
# =============================================================================


def decompose_cell(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    session: str,
    holdout: date,
) -> dict[str, Any]:
    pw_trades = _load_on_filter_trades(
        con, instrument, session, _predicate_pw_long(session), holdout
    )
    pm_trades = _load_on_filter_trades(
        con, instrument, session, _predicate_pm_long(session), holdout
    )

    pw_days = {td for td, _ in pw_trades}
    pm_days = {td for td, _ in pm_trades}
    overlap_days = pw_days & pm_days
    pm_only_days = pm_days - pw_days
    pw_only_days = pw_days - pm_days

    overlap_pnl = [p for td, p in pm_trades if td in overlap_days]
    nonoverlap_pnl = [p for td, p in pm_trades if td in pm_only_days]
    combined_pnl = [p for _, p in pm_trades]  # = overlap + non-overlap under PM

    # Sanity: PM = overlap ∪ non_overlap
    assert len(combined_pnl) == len(overlap_pnl) + len(nonoverlap_pnl), (
        f"PM decomposition sanity failed for {instrument} {session}: "
        f"combined={len(combined_pnl)} overlap={len(overlap_pnl)} nonoverlap={len(nonoverlap_pnl)}"
    )

    return {
        "instrument": instrument,
        "session": session,
        "pw_fires": len(pw_days),
        "pm_fires": len(pm_days),
        "overlap_days": len(overlap_days),
        "pm_only_days": len(pm_only_days),
        "pw_only_days": len(pw_only_days),
        "overlap_pct_of_pm": (
            100.0 * len(overlap_days) / len(pm_days) if pm_days else None
        ),
        "overlap": _stats(overlap_pnl),
        "nonoverlap": _stats(nonoverlap_pnl),
        "combined": _stats(combined_pnl),
    }


# =============================================================================
# MARKDOWN RENDER
# =============================================================================


def _fmt(x: float | None, places: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.{places}f}"


def _render(cells: list[dict[str, Any]], holdout: date) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# HTF Path A — prev-week × prev-month overlap decomposition")
    lines.append("")
    lines.append(f"**Generated:** {ts}")
    lines.append(f"**Script:** `research/htf_path_a_overlap_decomposition.py`")
    lines.append(f"**IS window:** `trading_day < {holdout.isoformat()}` (Mode A, from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`)")
    lines.append(f"**Cell axes:** entry_model={ENTRY_MODEL}, confirm_bars={CONFIRM_BARS}, orb_minutes={ORB_MINUTES}, direction={DIRECTION}, rr={RR}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "Reproducible verification of the numbers cited in "
        "`docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md` "
        "§ 'Adversarial-audit addendum — 2026-04-19'. Closes review-finding #1 "
        "(unreproducible quantitative claims) from the 2026-04-19 self-review."
    )
    lines.append("")

    for c in cells:
        inst = c["instrument"]
        sess = c["session"]
        lines.append(f"## {inst} {sess} long RR{RR}")
        lines.append("")
        lines.append(
            f"- prev-week v1 fires (unique trading-days, IS): **{c['pw_fires']}**"
        )
        lines.append(
            f"- prev-month v1 fires (unique trading-days, IS): **{c['pm_fires']}**"
        )
        lines.append(
            f"- OVERLAP days (PM ∧ PW): **{c['overlap_days']}**"
        )
        lines.append(
            f"- PM-only days (PM ∧ ¬PW): **{c['pm_only_days']}**"
        )
        lines.append(
            f"- PW-only days (PW ∧ ¬PM): **{c['pw_only_days']}**"
        )
        ovpct = c["overlap_pct_of_pm"]
        lines.append(
            f"- Overlap as % of PM fires: "
            + (f"**{ovpct:.1f}%**" if ovpct is not None else "—")
        )
        lines.append("")
        lines.append("| Subset | N | mean pnl_r | std | t | raw p |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for label, key in (
            ("OVERLAP (PM ∧ PW)", "overlap"),
            ("NON-OVERLAP (PM ∧ ¬PW)", "nonoverlap"),
            ("COMBINED (PM)", "combined"),
        ):
            s: SubsetStats = c[key]
            lines.append(
                f"| {label} | {s.n} | {_fmt(s.mean)} | {_fmt(s.std)} "
                f"| {_fmt(s.t)} | {_fmt(s.raw_p, 4)} |"
            )
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- Predicates copied verbatim from the long-direction branch of the two v1 scans: "
        "`research/htf_path_a_prev_week_v1_scan.py::_predicate_sql` and "
        "`research/htf_path_a_prev_month_v1_scan.py::_predicate_sql` (direction='long')."
    )
    lines.append(
        "- Trade loader uses the same JOIN and filter shape as the v1 scans' "
        "`_load_cell_trades`, restricted to IS window `trading_day < HOLDOUT_SACRED_FROM`."
    )
    lines.append(
        "- OVERLAP = trading_days present in BOTH predicate fire-sets. "
        "NON-OVERLAP = PM-fire days NOT in PW-fire set. COMBINED = full PM-on set."
    )
    lines.append(
        "- Statistics: one-sample two-tailed t-test vs 0 on per-trade `pnl_r`. "
        "Formula matches `research/htf_path_a_prev_month_v1_scan.py::_t_test` "
        "(t = mean / (std / sqrt(n)); raw p = 2·(1 − scipy.stats.t.cdf(|t|, n−1)))."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(
        "DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db "
        "python research/htf_path_a_overlap_decomposition.py"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "Writes the result markdown idempotently. Re-running on the same DB state "
        "re-creates the same numbers exactly (canonical IS window, no randomness)."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The addendum's claim is: MES EUROPE_FLOW long RR2.0 is OVERLAP-driven "
        "(prev-month v1 is not independent evidence once prev-week v1 fires are "
        "stripped out); MES TOKYO_OPEN long RR2.0 is NON-OVERLAP-driven "
        "(a genuinely new observation vs prev-week v1). See each cell's "
        "| t | row above to verify."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    holdout = HOLDOUT_SACRED_FROM
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        results = [
            decompose_cell(con, c["instrument"], c["session"], holdout) for c in CELLS
        ]
    finally:
        con.close()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(_render(results, holdout), encoding="utf-8")

    # Console summary
    print(f"Wrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    for r in results:
        print(
            f"  {r['instrument']} {r['session']}: PW={r['pw_fires']} PM={r['pm_fires']} "
            f"OVERLAP={r['overlap_days']} ({r['overlap_pct_of_pm']:.1f}% of PM)"
        )
        for label, key in (
            ("  overlap    ", "overlap"),
            ("  nonoverlap ", "nonoverlap"),
            ("  combined   ", "combined"),
        ):
            s = r[key]
            print(
                f"{label} N={s.n:3d} mean={_fmt(s.mean)} "
                f"t={_fmt(s.t)} p={_fmt(s.raw_p, 4)}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
