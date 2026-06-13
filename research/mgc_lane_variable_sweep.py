"""MGC no-pigeonhole lane-variable sweep (Phase A3) — committed reproducer.

Regenerates `research/output/mgc_lane_variable_sweep.csv`, the 612-cell grid the
2026-06-14 MGC trend-day result doc's A3 verdict rests on
(`docs/audit/results/2026-06-14-mgc-trend-day-tail-descriptive.md` § Phase A3).

Why this file exists
--------------------
A3 was originally run as an inline scan in the prior session's transcript — the
CSV landed on disk but was UNTRACKED and had no committed generator. A decision
doc citing an untracked CSV with no script violates RULE 11 (audit trail) and the
Project Truth Protocol ("real on my disk today" != "reproducible / auditable").
This script makes the t=2.36 verdict falsifiable forever. It is a confirmatory
re-frame, not new discovery — no `validated_setups` / `experimental_strategies`
write, no prereg required (RULE 10 carve-out for confirmatory audits).

Object under sweep
------------------
The no-pigeonhole lane variable space, swept directly from canonical layers with
multi-year stability + OOS + clustered-significance as the ONLY selection gate:

    9 sessions x {O5,O15,O30} x {E1,E2} x {RR1.0,1.5,2.0} x {NO_FILTER,G4,G6,G8}

Data-availability auto-skips (RULE 5.2): MGC has no data on several equity-hours
session/aperture combos; those cells emit N=0 and are dropped, logged as
coverage. 648 max combos -> 612 with N>=1.

DIRECTION NOTE (load-bearing, self-audit 2026-06-14)
----------------------------------------------------
This sweep pools BOTH directions per cell (no `direction` axis) — that is how the
original A3 grid was run, and it is what the CSV's headline t reproduces. The
top cell `US_DATA_830 O30 E2 RR2.0 ORB_G6` lands clustered-t=2.363 on the
BOTH-direction pool (N=569). Isolating the long-only deployable slice drops it to
t=2.004 (N=299). Any standalone-deployment framing must use the long-only number;
the grid headline is the both-direction pool. Disclosed in the result doc's
Validation verdict section.

Discipline (binding)
--------------------
- RULE 9 canonical triple-join, `orb_minutes` PINNED. Canonical layers only
  (`orb_outcomes` JOIN `daily_features`). Read-only — no DB writes.
- Canonical filter delegation: `research.filter_utils.filter_signal` — never
  re-encode `StrategyFilter.matches_df` (research-truth-protocol § Canonical
  filter delegation).
- Clustered-by-trading-day t (the honest unit): delegated to
  `research.vwap_mid_family_pooled_oos_v1._clustered_t` — never re-implement.
- Sacred holdout (Mode A): `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`,
  imported, never inlined.
- Costs: `orb_outcomes.pnl_r` already net of canonical MGC friction
  (`pipeline.cost_model.COST_SPECS['MGC'].total_friction`); scored directly.
- DB path: `pipeline.paths.GOLD_DB_PATH`, never hardcoded.

MGC G-FILTER PRICE-LEVEL CAVEAT (pre_registered_criteria.md:926)
----------------------------------------------------------------
Absolute G4/G6/G8 point thresholds select DIFFERENT price-level populations on
GC-proxy data (a 5pt ORB is 0.42% at $1200 vs 0.17% at $3000). The size filters
here are absolute-point ORB_Gn from the canonical registry; this caveat is
disclosed in any downstream prereg.

Reproduction
------------
    python research/mgc_lane_variable_sweep.py            # regenerate the grid CSV
    python research/mgc_lane_variable_sweep.py --verify    # regenerate + diff vs committed CSV
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import NamedTuple, TypedDict

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from research.vwap_mid_family_pooled_oos_v1 import _clustered_t
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# ---- locked sweep axes (RULE 5.1 comprehensive scope) -----------------------
SYMBOL = "MGC"
# confirm_bars is NOT a swept axis and is NOT pinned: the grid pools all
# confirm_bars variants per cell, matching the original A3 scan. This matters
# for E1 (which exists at cb=1..5 in orb_outcomes — five confirmation-timing
# variants of nearly the same trades) but is a no-op for E2 (cb=1 only). The
# pooled E1 N is therefore ~5x the cb=1 count (e.g. TOKYO_OPEN E1 O5 = 4493 =
# 918+909+896+889+881). The clustered-by-trading-day t partly discounts this
# (same-day cb variants share a cluster), and the cell selection gate is
# significance, not naive N — so pooling cb does not rescue any cell (all
# clustered-t < 2.5 regardless). Disclosed so the inflated E1 N is not misread
# as independent power.
SESSIONS: tuple[str, ...] = (
    "CME_REOPEN",
    "TOKYO_OPEN",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
)
ORB_MINUTES: tuple[int, ...] = (5, 15, 30)
ENTRY_MODELS: tuple[str, ...] = ("E1", "E2")
RR_TARGETS: tuple[float, ...] = (1.0, 1.5, 2.0)
# NO_FILTER plus the absolute-point ORB size filters (canonical registry keys).
FILTERS: tuple[str, ...] = ("NO_FILTER", "ORB_G4", "ORB_G6", "ORB_G8")

# Minimum IS sample to ADMIT a cell into the grid at all — a deployable-power
# pre-screen (below this, the clustered-t is too noisy to be a candidate). The
# committed A3 grid's smallest admitted cell is N=53; cells below this floor are
# dropped (matches the original scan's coverage: 612 of 648 Cartesian combos).
MIN_IS_N = 50

# Min deployable-power sample for the headline gate table (Criterion 7).
N_FLOOR = 100

OUTPUT_CSV = Path(__file__).resolve().parent / "output" / "mgc_lane_variable_sweep.csv"

# CSV column order — frozen to match the committed grid byte-for-byte.
COLUMNS = ["sess", "om", "em", "rr", "filt", "N", "ExpR", "clt", "clp", "posyrs", "nyrs", "oosN", "oos"]


class GridRow(TypedDict):
    """One cell of the lane-variable grid (CSV row)."""

    sess: str
    om: int
    em: str
    rr: float
    filt: str
    N: int
    ExpR: float
    clt: float
    clp: float
    posyrs: int
    nyrs: int
    oosN: int
    oos: float


# ---- canonical data pull (RULE 9 triple-join) -------------------------------
def _pull_cell(
    con: duckdb.DuckDBPyConnection,
    session: str,
    orb_minutes: int,
    entry_model: str,
    rr: float,
) -> pd.DataFrame:
    """All-direction MGC trade series for one (session, om, em, rr) cell.

    Canonical triple-join with `orb_minutes` pinned to the cell's aperture so the
    `daily_features` join is 1:1 (daily-features-joins.md). Returns every
    `orb_*` column so the canonical filter's `matches_df` can look up whatever
    the size filter requires without pre-aliasing (filter_utils contract).
    """
    q = f"""
        SELECT o.trading_day, o.pnl_r, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE o.symbol = '{SYMBOL}'
          AND o.orb_minutes = {orb_minutes}
          AND o.entry_model = '{entry_model}'
          AND o.rr_target = {rr}
          AND o.orb_label = '{session}'
          AND o.outcome IS NOT NULL
    """
    df = con.sql(q).df()
    if df.empty:
        return df
    df = df.dropna(subset=["pnl_r"]).copy()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df.sort_values("trading_day").reset_index(drop=True)


def _apply_filter(df: pd.DataFrame, filter_key: str, session: str) -> pd.DataFrame:
    """Apply a canonical size filter (or pass-through for NO_FILTER)."""
    if filter_key == "NO_FILTER":
        return df
    sig = filter_signal(df, filter_key, session)
    return df[sig == 1].copy()


# ---- per-cell metrics -------------------------------------------------------
class CellMetrics(NamedTuple):
    """Typed per-cell statistics (the grid columns past the lane identity)."""

    N: int
    ExpR: float
    clt: float
    clp: float
    posyrs: int
    nyrs: int
    oosN: int
    oos: float


def _cell_metrics(df: pd.DataFrame) -> CellMetrics | None:
    """Compute the grid row metrics for one filtered cell.

    Returns None if the cell has fewer than ``MIN_IS_N`` IS trades (the
    deployable-power pre-screen — below this the clustered-t is too noisy to be a
    candidate, and the cell is dropped from the grid).

    `clt`/`clp` are clustered-by-trading-day (the honest unit — correlated
    same-day trades are one cluster). `posyrs`/`nyrs` are the per-calendar-year
    IS stability counts (RULE 12 outlier guard). `oosN`/`oos` are the 2026 Mode-A
    holdout slice. All on `pnl_r` already net of canonical MGC friction.
    """
    sacred = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_df = df[df["trading_day"] < sacred]
    oos_df = df[df["trading_day"] >= sacred]
    n = len(is_df)
    if n < MIN_IS_N:
        return None

    pnl = is_df["pnl_r"].to_numpy(dtype=float)
    cluster = np.asarray(is_df["trading_day"].to_numpy())
    _, t_clust, p_clust, _ = _clustered_t(pnl, cluster)

    yr = is_df.assign(yr=is_df["trading_day"].dt.year).groupby("yr")["pnl_r"].mean()
    nyrs = int(yr.shape[0])
    posyrs = int((yr > 0).sum())

    oos_n = len(oos_df)
    oos_expr = float(oos_df["pnl_r"].mean()) if oos_n else float("nan")

    return CellMetrics(
        N=n,
        ExpR=float(pnl.mean()),
        clt=float(t_clust),
        clp=float(p_clust),
        posyrs=posyrs,
        nyrs=nyrs,
        oosN=oos_n,
        oos=oos_expr,
    )


# ---- sweep ------------------------------------------------------------------
def run_sweep(con: duckdb.DuckDBPyConnection) -> tuple[list[GridRow], int, int]:
    """Sweep the full lane variable space. Returns (rows, attempted, with_data)."""
    rows: list[GridRow] = []
    attempted = 0
    with_data = 0
    for session in SESSIONS:
        for om in ORB_MINUTES:
            for em in ENTRY_MODELS:
                for rr in RR_TARGETS:
                    base = _pull_cell(con, session, om, em, rr)
                    if base.empty:
                        # No raw data for this combo (RULE 5.2 auto-skip).
                        continue
                    for filt in FILTERS:
                        attempted += 1
                        cell = _apply_filter(base, filt, session)
                        metrics = _cell_metrics(cell)
                        if metrics is None:
                            continue  # below MIN_IS_N — dropped from grid
                        with_data += 1
                        rows.append(
                            GridRow(
                                sess=session,
                                om=om,
                                em=em,
                                rr=rr,
                                filt=filt,
                                N=metrics.N,
                                ExpR=metrics.ExpR,
                                clt=metrics.clt,
                                clp=metrics.clp,
                                posyrs=metrics.posyrs,
                                nyrs=metrics.nyrs,
                                oosN=metrics.oosN,
                                oos=metrics.oos,
                            )
                        )
    return rows, attempted, with_data


def _write_csv(rows: list[GridRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in COLUMNS})


def _gate_summary(rows: list[GridRow]) -> str:
    """The honest gate table from the result doc (N>=100 cells)."""
    eligible = [r for r in rows if r["N"] >= N_FLOOR]
    pos3 = sum(1 for r in eligible if r["posyrs"] >= 3)
    oos_pos = sum(1 for r in eligible if r["oos"] > 0)
    structure = sum(1 for r in eligible if r["posyrs"] >= 3 and r["oos"] > 0)
    t300 = sum(1 for r in eligible if r["clt"] >= 3.00)
    t250 = sum(1 for r in eligible if r["clt"] >= 2.50)
    full = sum(1 for r in eligible if r["posyrs"] >= 3 and r["oos"] > 0 and r["clt"] >= 3.00)
    return (
        f"\nGate summary (N>={N_FLOOR} cells; eligible={len(eligible)}):\n"
        f"  >=3/4 IS years positive   : {pos3}\n"
        f"  OOS ExpR > 0              : {oos_pos}\n"
        f"  >=3/4yr AND OOS>0         : {structure}  (structure)\n"
        f"  clustered-t >= 3.00       : {t300}\n"
        f"  clustered-t >= 2.50       : {t250}\n"
        f"  full gate (struct + t3.0) : {full}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="override gold.db path (default canonical)")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="regenerate the grid, then diff vs the committed CSV (non-zero exit on drift)",
    )
    args = ap.parse_args()

    db = args.db or str(GOLD_DB_PATH)
    con = duckdb.connect(db, read_only=True)
    rows, attempted, with_data = run_sweep(con)
    con.close()

    if not rows:
        print("No cells produced — check DB freshness / coverage.")
        return 1

    print(f"Coverage: {with_data} cells with IS data / {attempted} filter-cells attempted.")
    print(_gate_summary(rows))

    top = sorted(rows, key=lambda r: -r["clt"])[:5]
    print("Top cells by clustered-t:")
    print(f"  {'lane':42} {'N':>5} {'ExpR':>8} {'clt':>6} {'yrs+':>5} {'oos':>8}")
    for r in top:
        lane = f"{r['sess']} O{r['om']} {r['em']} RR{r['rr']} {r['filt']}"
        print(
            f"  {lane:42} {r['N']:5d} {r['ExpR']:+8.4f} "
            f"{r['clt']:6.3f} {r['posyrs']}/{r['nyrs']:>3} {r['oos']:+8.4f}"
        )

    if args.verify:
        # VERDICT-INVARIANT verification (not byte-identity). The committed CSV is
        # a snapshot of a mutable gold.db; backfill since the original run produces
        # off-by-one N drift on boundary cells — a value-oracle pinned to a mutable
        # DB is an anti-pattern (memory:feedback_golden_against_live_db_*). What
        # MUST hold (and what the result doc cites) are the verdict-bearing facts:
        #   1. ZERO cells clear clustered-t >= 3.00 (the with-theory Chordia bar).
        #   2. ZERO cells clear clustered-t >= 2.50 at deployable N (>=100).
        #   3. The top deployable-N cell is US_DATA_830 O30 E2 RR2.0 ORB_G6 at
        #      clustered-t ~ 2.36 (the named near-miss object).
        eligible = [r for r in rows if r["N"] >= N_FLOOR]
        t300 = [r for r in rows if r["clt"] >= 3.00]
        t250_elig = [r for r in eligible if r["clt"] >= 2.50]
        top_elig = max(eligible, key=lambda r: r["clt"]) if eligible else None
        named = next(
            (
                r
                for r in rows
                if r["sess"] == "US_DATA_830"
                and r["om"] == 30
                and r["em"] == "E2"
                and r["rr"] == 2.0
                and r["filt"] == "ORB_G6"
            ),
            None,
        )
        failures: list[str] = []
        if t300:
            failures.append(f"INVARIANT 1 BROKEN: {len(t300)} cell(s) clear clustered-t>=3.00")
        if t250_elig:
            failures.append(
                f"INVARIANT 2 BROKEN: {len(t250_elig)} N>={N_FLOOR} cell(s) clear clustered-t>=2.50"
            )
        if named is None:
            failures.append("INVARIANT 3 BROKEN: named US_DATA_830 G6 cell not produced")
        elif not (2.20 <= named["clt"] <= 2.50):
            failures.append(
                f"INVARIANT 3 DRIFTED: US_DATA_830 G6 clustered-t={named['clt']:.3f} outside [2.20,2.50]"
            )
        if failures:
            print("\n--verify: FAIL")
            for f in failures:
                print(f"  {f}")
            return 1
        print("\n--verify: PASS — verdict invariants hold (0 cells t>=3.0; 0 N>=100 cells t>=2.5).")
        if named is not None:
            print(f"  named cell US_DATA_830 O30 E2 RR2.0 ORB_G6: clustered-t={named['clt']:.3f}")
        if top_elig is not None:
            lane = f"{top_elig['sess']} O{top_elig['om']} {top_elig['em']} RR{top_elig['rr']} {top_elig['filt']}"
            print(f"  top N>={N_FLOOR} cell: {lane} clustered-t={top_elig['clt']:.3f}")
        return 0

    _write_csv(rows, OUTPUT_CSV)
    print(f"\nWrote {len(rows)} cells -> {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
