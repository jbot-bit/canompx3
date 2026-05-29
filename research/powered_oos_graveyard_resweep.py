"""Powered-OOS re-judge of calendar-OOS-starved graveyard candidates.

Doctrine: feedback_powered_oos_holdout_at_discovery_no_calendar_wait_2026_05_29.
Many graveyard kills failed ONLY on a thin CALENDAR OOS (>=2026-01-01). For a
low-frequency session the calendar slice is too small to ever reach RULE 3.3
power, so the verdict was "wait to accumulate" — banned posture. This script
re-judges each candidate with a TRADE-FRACTION holdout (last 30% of trades, in
temporal order) and reports the canonical RULE 3.3 power tier.

Canonical delegation (institutional-rigor §4 / RULE 9):
  - power math   -> research.oos_power.one_sample_power / power_verdict
  - look-ahead   -> research.comprehensive_deployed_lane_scan gates
  - data layers  -> orb_outcomes JOIN daily_features ONLY (RULE 9)
  - holdout      -> trading_app.holdout_policy.HOLDOUT_SACRED_FROM (calendar contrast)

NO DB writes. Read-only. Emits a markdown table to stdout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from research.oos_power import one_sample_power, one_sample_n_for_power, power_verdict
from research.comprehensive_deployed_lane_scan import _overnight_lookhead_clean
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

# Trade-fraction reserved as OOS (most-recent, temporal order). 0.30 is the
# default per the doctrine; the achieved power is what actually gates the
# verdict, not the fraction.
OOS_FRACTION = 0.30


@dataclass
class Candidate:
    """One graveyard cell to re-judge. feature/op/thresh define the filter."""

    tag: str
    symbol: str
    session: str
    orb_minutes: int
    rr: float
    direction: str  # 'long' | 'short' | 'both'
    feature: str | None  # daily_features column, or None for no filter
    op: str  # '>=', '<=', '==', 'none'
    thresh: float | None
    needs_overnight: bool  # if feature is an overnight_* class -> RULE 1.2 gate


def _pull(con: duckdb.DuckDBPyConnection, c: Candidate) -> pd.DataFrame:
    """Canonical orb_outcomes JOIN daily_features (RULE 9 triple-join safe:
    we join on (trading_day, symbol) and the orb_minutes is pinned in WHERE,
    so daily_features' 3-rows-per-day is deduplicated by the orb_minutes match).
    """
    feat_cols = ""
    if c.feature:
        feat_cols = f", d.{c.feature} AS feat"
    q = f"""
        SELECT o.trading_day, o.pnl_r,
               CASE WHEN o.stop_price < o.entry_price THEN 'long' ELSE 'short' END AS dir
               {feat_cols}
        FROM orb_outcomes o
        JOIN daily_features d
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE o.symbol = '{c.symbol}'
          AND o.orb_label = '{c.session}'
          AND o.orb_minutes = {c.orb_minutes}
          AND o.entry_model = 'E2'
          AND o.rr_target = {c.rr}
          AND o.confirm_bars = 1
          AND o.outcome IS NOT NULL
    """
    df = con.sql(q).df()
    if df.empty:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df = df.sort_values("trading_day").reset_index(drop=True)
    if c.direction != "both":
        df = df[df["dir"] == c.direction]
    if c.feature:
        s = pd.to_numeric(df["feat"], errors="coerce")
        if c.op == ">=":
            df = df[s >= c.thresh]
        elif c.op == "<=":
            df = df[s <= c.thresh]
        elif c.op == "==":
            df = df[s == c.thresh]
    return df.dropna(subset=["pnl_r"]).reset_index(drop=True)


def _t(s: pd.Series) -> tuple[int, float, float]:
    n = len(s)
    if n < 2:
        return n, float("nan"), float("nan")
    m = s.mean()
    sd = s.std(ddof=1)
    t = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return n, m, t


def judge(con: duckdb.DuckDBPyConnection, c: Candidate) -> dict:
    # RULE 1.2 look-ahead gate: overnight_* features only valid for ORB >=17:00.
    if c.needs_overnight and not _overnight_lookhead_clean(c.session):
        return {"tag": c.tag, "verdict": "LOOKAHEAD_BLOCKED",
                "note": f"overnight feature invalid for {c.session} (RULE 1.2)"}
    df = _pull(con, c)
    n_full = len(df)
    if n_full < 60:
        return {"tag": c.tag, "n_full": n_full, "verdict": "TOO_THIN",
                "note": "N<60 total trades"}
    _, mf, tf = _t(df["pnl_r"])

    # CALENDAR holdout (the original, starved cut) for contrast.
    cut = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_cal = df[df.trading_day < cut]["pnl_r"]
    oos_cal = df[df.trading_day >= cut]["pnl_r"]
    ni_c, _, ti_c = _t(is_cal)
    no_c, _, _ = _t(oos_cal)
    d_cal = abs(ti_c) / np.sqrt(ni_c) if ni_c > 1 else 0.0
    pw_cal = one_sample_power(d_cal, no_c) if no_c >= 2 else 0.0

    # FRACTION holdout (the doctrine cut): last OOS_FRACTION of trades.
    k = int(n_full * (1 - OOS_FRACTION))
    is_f = df.iloc[:k]["pnl_r"]
    oos_f = df.iloc[k:]["pnl_r"]
    ni_f, mi_f, ti_f = _t(is_f)
    no_f, mo_f, to_f = _t(oos_f)
    d_f = abs(ti_f) / np.sqrt(ni_f) if ni_f > 1 else 0.0
    pw_f = one_sample_power(d_f, no_f) if no_f >= 2 else 0.0
    tier_f = power_verdict(pw_f)
    n_for_80 = one_sample_n_for_power(d_f) if d_f > 0 else None

    # Verdict logic (RULE 3.3): a binary OOS kill only bites at CAN_REFUTE.
    dir_match = np.sign(mi_f) == np.sign(mo_f) if no_f >= 2 else False
    if tier_f == "STATISTICALLY_USELESS":
        verdict = "UNVERIFIED_INSUFFICIENT_POWER"
    elif tier_f == "DIRECTIONAL_ONLY":
        verdict = "SURVIVES_DIRECTIONAL" if (dir_match and mo_f > 0) else "DECAYING_DIRECTIONAL"
    else:  # CAN_REFUTE
        verdict = "SURVIVES_CONFIRMATORY" if (dir_match and mo_f > 0 and to_f >= 2.0) else "REFUTED_OOS"

    return {
        "tag": c.tag, "symbol": c.symbol, "session": c.session,
        "om": c.orb_minutes, "rr": c.rr, "dir": c.direction,
        "filter": f"{c.feature}{c.op}{c.thresh}" if c.feature else "none",
        "n_full": n_full, "t_full": round(tf, 2), "expr_full": round(mf, 4),
        "cal_oos_n": no_c, "cal_oos_pw": round(pw_cal, 2),
        "cal_oos_tier": power_verdict(pw_cal),
        "is_t": round(ti_f, 2),
        "frac_oos_n": no_f, "frac_oos_expr": round(mo_f, 4),
        "frac_oos_t": round(to_f, 2), "frac_oos_pw": round(pw_f, 2),
        "frac_tier": tier_f, "n_for_80": n_for_80,
        "dir_match": bool(dir_match), "verdict": verdict,
    }


def graveyard_candidates() -> list[Candidate]:
    """The calendar-OOS-starved graveyard cells, by source postmortem."""
    out: list[Candidate] = []
    # --- 2026-04-15 T0-T8 non-volume horizon (H1/H2/H5) ---
    out.append(Candidate("H1_MES_LM_ovn80", "MES", "LONDON_METALS", 30, 1.5, "long",
                         "overnight_range_pct", ">=", 80, True))
    out.append(Candidate("H2_MNQ_COMEX_garch70", "MNQ", "COMEX_SETTLE", 5, 1.0, "long",
                         "garch_forecast_vol_pct", ">=", 70, False))
    out.append(Candidate("H2x_MES_COMEX_garch70", "MES", "COMEX_SETTLE", 5, 1.0, "long",
                         "garch_forecast_vol_pct", ">=", 70, False))
    out.append(Candidate("H5_MES_COMEX_ovnpdh0", "MES", "COMEX_SETTLE", 30, 1.0, "long",
                         "overnight_took_pdh", "==", 0, True))
    # --- 2026-04-15 rel_vol BANNED on E2 (RULE 6.3) — control: NO_FILTER baselines ---
    # rel_vol_{s} is look-ahead on E2; we DO NOT test it. Instead re-judge the
    # garch sibling on its cross-instrument MNQ claim at other RRs.
    out.append(Candidate("H2_MNQ_COMEX_garch70_rr15", "MNQ", "COMEX_SETTLE", 5, 1.5, "long",
                         "garch_forecast_vol_pct", ">=", 70, False))
    # --- DOW filters H3/H4 (is_monday / dow_thu) — calendar features, RULE 6.1 safe ---
    # day_of_week is the canonical column; Monday=0 .. ; we test via == on dow.
    out.append(Candidate("H3_MNQ_COMEX_monday", "MNQ", "COMEX_SETTLE", 5, 1.0, "long",
                         "day_of_week", "==", 0, False))
    out.append(Candidate("H4_MNQ_COMEX_thu", "MNQ", "COMEX_SETTLE", 5, 1.0, "long",
                         "day_of_week", "==", 3, False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="DB path (default canonical).")
    args = ap.parse_args()
    if args.db:
        db = args.db
    else:
        from pipeline.paths import GOLD_DB_PATH
        db = str(GOLD_DB_PATH)
    con = duckdb.connect(db, read_only=True)
    rows = [judge(con, c) for c in graveyard_candidates()]
    con.close()

    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("# Powered-OOS graveyard re-sweep")
    print(f"OOS_FRACTION={OOS_FRACTION}  | calendar cut={HOLDOUT_SACRED_FROM}\n")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
