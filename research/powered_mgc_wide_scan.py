"""Powered comprehensive MGC scan — full RULE 5 grid, canonical triple-join.

Goal: find MGC's BEST powered candidate without pigeonholing on ovn>=80.
MGC = gold = the genuinely uncorrelated diversifier vs a 100%-MNQ (Nasdaq) book.

Discipline:
  - RULE 9 canonical triple-join (orb_minutes PINNED) — the bug that 3x-inflated
    the earlier ad-hoc pass.
  - RULE 1.2 look-ahead gate on overnight_* features (delegated to canonical).
  - RULE 5 full grid: all MGC sessions x {5,15,30} x {1.0,1.5,2.0} x {long,short}.
  - RULE 6.1 trade-time-knowable features ONLY (no break-bar / rel_vol / mae / outcome).
  - Power via canonical research.oos_power (one-sample, trade-fraction OOS).
  - BH-FDR at K_family via canonical bh_fdr.
  - NO DB writes. Read-only. Emits ranked markdown to stdout + optional CSV.

This is a DISCOVERY scan over a single instrument's grid; it is NOT a pooled
universality claim, so the pooled-finding-rule per-cell schema does not bind —
but every cell is reported individually with its own t/power, which satisfies
the spirit of it.
"""

from __future__ import annotations

import argparse
import itertools

import duckdb
import numpy as np
import pandas as pd

from research.oos_power import one_sample_power, power_verdict
from research.comprehensive_deployed_lane_scan import _overnight_lookhead_clean

OOS_FRACTION = 0.30
MIN_N_FULL = 80  # below this a cell is too thin to judge at all

# Trade-time-knowable features (RULE 6.1) + their op/threshold grids.
# overnight_* carry needs_overnight=True -> RULE 1.2 gate applied per session.
FEATURE_GRID: list[tuple[str, str, list[float], bool]] = [
    ("none", "none", [0], False),
    ("overnight_range_pct", ">=", [70, 80, 90], True),
    ("atr_20_pct", ">=", [60, 75], False),
    ("atr_20_pct", "<=", [40, 25], False),
    ("garch_forecast_vol_pct", ">=", [70], False),
    ("garch_forecast_vol_pct", "<=", [30], False),
    ("day_of_week", "==", [0, 1, 2, 3, 4], False),  # Mon..Fri
]

RRS = [1.0, 1.5, 2.0]
APERTURES = [5, 15, 30]
DIRECTIONS = ["long", "short"]


def _t(s: pd.Series) -> tuple[int, float, float]:
    n = len(s)
    if n < 2:
        return n, float("nan"), float("nan")
    m = s.mean()
    sd = s.std(ddof=1)
    t = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return n, float(m), float(t)


def _mgc_sessions(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.sql(
        "SELECT DISTINCT orb_label FROM orb_outcomes WHERE symbol='MGC' ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows]


def _pull_base(con: duckdb.DuckDBPyConnection, session: str, om: int, rr: float) -> pd.DataFrame:
    """Canonical triple-join (orb_minutes PINNED) — all trade-time features at once."""
    q = f"""
        SELECT o.trading_day, o.pnl_r,
               CASE WHEN o.stop_price < o.entry_price THEN 'long' ELSE 'short' END AS dir,
               d.overnight_range_pct, d.atr_20_pct, d.garch_forecast_vol_pct, d.day_of_week
        FROM orb_outcomes o
        JOIN daily_features d
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE o.symbol = 'MGC' AND o.orb_label = '{session}' AND o.orb_minutes = {om}
          AND o.entry_model = 'E2' AND o.rr_target = {rr} AND o.confirm_bars = 1
          AND o.outcome IS NOT NULL
    """
    df = con.sql(q).df()
    if df.empty:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df.sort_values("trading_day").reset_index(drop=True)


def _apply(df: pd.DataFrame, feat: str, op: str, thr: float) -> pd.DataFrame:
    if feat == "none":
        return df
    s = pd.to_numeric(df[feat], errors="coerce")
    if op == ">=":
        return df[s >= thr]
    if op == "<=":
        return df[s <= thr]
    if op == "==":
        return df[s == thr]
    return df


def scan(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sessions = _mgc_sessions(con)
    out: list[dict] = []
    for session, om, rr in itertools.product(sessions, APERTURES, RRS):
        base = _pull_base(con, session, om, rr)
        if base.empty:
            continue
        for direction in DIRECTIONS:
            dsub = base[base["dir"] == direction]
            for feat, op, threshes, needs_ovn in FEATURE_GRID:
                # RULE 1.2: overnight_* invalid for ORB < 17:00 Brisbane.
                if needs_ovn and not _overnight_lookhead_clean(session):
                    continue
                for thr in threshes:
                    cell = _apply(dsub, feat, op, thr).dropna(subset=["pnl_r"])
                    cell = cell.sort_values("trading_day").reset_index(drop=True)
                    n_full = len(cell)
                    if n_full < MIN_N_FULL:
                        continue
                    _, mf, tf = _t(cell["pnl_r"])
                    k = int(n_full * (1 - OOS_FRACTION))
                    is_, oos = cell.iloc[:k]["pnl_r"], cell.iloc[k:]["pnl_r"]
                    ni, mi, ti = _t(is_)
                    no, mo, to = _t(oos)
                    d_is = abs(ti) / np.sqrt(ni) if ni > 1 else 0.0
                    pw = one_sample_power(d_is, no) if no >= 2 else 0.0
                    tier = power_verdict(pw)
                    dir_match = (np.sign(mi) == np.sign(mo)) if no >= 2 else False
                    out.append({
                        "session": session, "om": om, "rr": rr, "dir": direction,
                        "filter": f"{feat}{op}{thr}" if feat != "none" else "none",
                        "n_full": n_full, "t_full": round(tf, 2), "expr_full": round(mf, 4),
                        "is_t": round(ti, 2),
                        "oos_n": no, "oos_expr": round(mo, 4), "oos_t": round(to, 2),
                        "oos_pw": round(pw, 2), "tier": tier, "dir_match": bool(dir_match),
                    })
    res = pd.DataFrame(out)
    if res.empty:
        return res
    # BH-FDR at K_family = the full MGC grid (single instrument family).
    # p two-sided from full-period t. Delegate the FDR step-up to the canonical
    # bh_fdr (comprehensive_deployed_lane_scan) — never re-encode (institutional
    # -rigor §4). The canonical helper keys on a `p_is` column.
    from scipy import stats as _st

    from research.comprehensive_deployed_lane_scan import bh_fdr

    res["p_is"] = res.apply(
        lambda r: float(2 * _st.t.sf(abs(r["t_full"]), max(r["n_full"] - 1, 1))), axis=1
    )
    res = bh_fdr(res, alpha=0.05)
    res = res.rename(columns={"bh_pass": "bh_pass_Kfamily", "p_is": "p_full"})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    if args.db:
        db = args.db
    else:
        from pipeline.paths import GOLD_DB_PATH
        db = str(GOLD_DB_PATH)
    con = duckdb.connect(db, read_only=True)
    res = scan(con)
    con.close()
    if res.empty:
        print("No MGC cells met MIN_N_FULL.")
        return 0
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print("# Powered MGC wide scan (canonical triple-join)\n")
    print(f"cells_tested={len(res)}  OOS_FRACTION={OOS_FRACTION}  MIN_N_FULL={MIN_N_FULL}\n")
    # Top by full t among those with positive full ExpR and dir_match.
    promising = res[(res["expr_full"] > 0) & (res["dir_match"])].copy()
    promising = promising.sort_values("t_full", ascending=False)
    print("## Top 20 promising MGC cells (positive ExpR + dir_match), by full t\n")
    cols = ["session", "om", "rr", "dir", "filter", "n_full", "t_full", "expr_full",
            "is_t", "oos_n", "oos_expr", "oos_t", "oos_pw", "tier", "bh_pass_Kfamily"]
    print(promising[cols].head(20).to_string(index=False))
    print("\n## Any cell reaching DIRECTIONAL_ONLY+ powered OOS:\n")
    powered = res[(res["tier"] != "STATISTICALLY_USELESS") & (res["oos_expr"] > 0)
                  & (res["dir_match"])].sort_values("oos_t", ascending=False)
    print(powered[cols].head(15).to_string(index=False) if not powered.empty
          else "  NONE — every MGC cell is OOS-underpowered (expected: 3.0yr horizon).")
    if args.csv:
        res.to_csv(args.csv, index=False)
        print(f"\nFull grid -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
