"""6-lane deployed portfolio scale-stability audit.

.. deprecated:: 2026-04-21
    **BUG**: ``parse_strategy_id`` (line 35) extracts aperture_overlay but
    ``load_lane_universe`` (line 84) hardcodes ``o.orb_minutes = 5``, ignoring
    the parsed ``_O15`` suffix on L2 and L6 DEPLOY lanes. L2 and L6 backtests
    returned wrong 5-minute ORB outcomes; canonical live deployment runs those
    two lanes at 15-minute ORB.

    Fixed in ``research/audit_lane_baseline_decomposition_v2.py`` (PR #57)
    which imports canonical ``trading_app.eligibility.builder.parse_strategy_id``
    and passes the parsed ``orb_minutes`` into SQL. See
    ``docs/audit/results/2026-04-21-correction-aperture-audit-rerun.md``.

    L2 and L6 fire-rate / lift numbers from this script are UNTRUSTWORTHY.
    L1, L3, L4, L5 numbers happen to be correct (no ``_O`` suffix → default 5).

    Do not run this script for new analysis. If you need fire-rate stability,
    delegate via ``trading_app.eligibility.builder.parse_strategy_id`` and
    build SQL from the parsed ``orb_minutes``.

For each of the 6 currently-deployed lanes (per lane_allocation.json),
compute fire-rate-by-year and lift-by-year to detect scale-artifact
signatures like the one found at L4 OVNRNG_50_FAST10 (see correction
file 2026-04-20-nyse-open-ovnrng-fast10-correction.md).

A scale-artifact signature is:
  (a) fire rate drifts monotonically >20pp across IS years, AND
  (b) lift over unfiltered baseline collapses toward zero in recent years.

Canonical truth only: orb_outcomes JOIN daily_features + lane_allocation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")
LANE_ALLOCATION = Path("docs/runtime/lane_allocation.json")


def parse_strategy_id(sid: str) -> dict:
    """Parse MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 into components.

    Uses a permissive parse — validates nothing. Session names and filter
    names may contain underscores. We split on the known-position tokens:
      {inst}_{session_tokens}_{entry}_RR{rr}_CB{cb}_{filter_tokens}
    where session and filter may span multiple tokens.
    """
    parts = sid.split("_")
    inst = parts[0]
    entry_ix = next(i for i, p in enumerate(parts) if p.startswith("E") and p[1:].isdigit())
    rr_ix = next(i for i, p in enumerate(parts) if p.startswith("RR"))
    cb_ix = next(i for i, p in enumerate(parts) if p.startswith("CB"))
    session = "_".join(parts[1:entry_ix])
    entry = parts[entry_ix]
    rr = float(parts[rr_ix][2:])
    cb = int(parts[cb_ix][2:])
    # Filter is everything after CB — may be multi-token composite like
    # "VOL_RV12_N20_O15". Strip trailing _O15 / _O30 aperture-overlay.
    filter_tokens = parts[cb_ix + 1:]
    aperture_overlay = ""
    if filter_tokens and filter_tokens[-1] in ("O15", "O30", "O5"):
        aperture_overlay = filter_tokens[-1]
        filter_tokens = filter_tokens[:-1]
    filter_name = "_".join(filter_tokens) if filter_tokens else ""
    return {
        "instrument": inst,
        "session": session,
        "entry": entry,
        "rr": rr,
        "cb": cb,
        "filter": filter_name,
        "aperture_overlay": aperture_overlay,
    }


def load_lane_universe(instrument: str, session: str, entry: str, rr: float, cb: int) -> pd.DataFrame:
    q = """
    SELECT o.trading_day, o.pnl_r,
           CASE WHEN o.pnl_r > 0 THEN 1 ELSE 0 END AS win,
           o.entry_price,
           d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = ?
      AND o.orb_label = ?
      AND o.orb_minutes = 5
      AND o.entry_model = ?
      AND o.rr_target = ?
      AND o.confirm_bars = ?
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = DB.execute(q, [instrument, session, entry, rr, cb]).fetchdf()
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    return df


def audit_lane(lane: dict) -> None:
    sid = lane["strategy_id"]
    spec = parse_strategy_id(sid)
    print("\n" + "=" * 80)
    print(f"LANE: {sid}")
    print(f"  instrument={spec['instrument']} session={spec['session']} "
          f"entry={spec['entry']} rr={spec['rr']} cb={spec['cb']} "
          f"filter={spec['filter']} aperture_overlay={spec['aperture_overlay']}")
    print("=" * 80)

    df = load_lane_universe(spec["instrument"], spec["session"], spec["entry"],
                            spec["rr"], spec["cb"])
    print(f"Universe n={len(df)}  range {df.trading_day.min()} → {df.trading_day.max()}")

    if not spec["filter"]:
        print("  (no filter on this lane — skip)")
        return

    # Fire the canonical filter
    try:
        sig = filter_signal(df, spec["filter"], spec["session"])
    except KeyError as e:
        print(f"  FILTER NOT REGISTERED: {e}")
        return
    df["fire"] = sig

    # Overall fire rate
    overall_fire = sig.mean()
    print(f"\n  Overall fire rate: {overall_fire:.3f}")

    # Per-year fire rate + lift
    print(f"\n  {'Year':6s} {'N_tot':>6s} {'ExpR_unf':>10s} {'N_fire':>7s} "
          f"{'ExpR_filt':>11s} {'lift':>8s} {'fire%':>7s}")
    year_data = []
    for y, grp in df.groupby("year"):
        filt = grp[grp["fire"] == 1]
        unf_expr = grp["pnl_r"].mean()
        f_expr = filt["pnl_r"].mean() if len(filt) else float("nan")
        lift = f_expr - unf_expr if not np.isnan(f_expr) else float("nan")
        fire_pct = len(filt) / len(grp) * 100
        year_data.append({"year": y, "n_tot": len(grp), "unf_expr": unf_expr,
                          "n_fire": len(filt), "filt_expr": f_expr, "lift": lift,
                          "fire_pct": fire_pct})
        print(f"  {y:6d} {len(grp):>6d} {unf_expr:>10.4f} {len(filt):>7d} "
              f"{f_expr:>11.4f} {lift:>+8.4f} {fire_pct:>6.1f}%")

    # Scale-artifact signature
    ydf = pd.DataFrame(year_data)
    is_years = ydf[ydf.year < 2026]
    if len(is_years) >= 5:
        fire_min = is_years["fire_pct"].min()
        fire_max = is_years["fire_pct"].max()
        fire_range = fire_max - fire_min
        lift_early = is_years.iloc[:len(is_years) // 2]["lift"].mean()
        lift_late = is_years.iloc[len(is_years) // 2:]["lift"].mean()
        print(f"\n  IS-years scale-stability:")
        print(f"    fire% range: {fire_min:.1f}% → {fire_max:.1f}% (Δ {fire_range:+.1f}pp)")
        print(f"    lift early-half: {lift_early:+.4f}")
        print(f"    lift late-half:  {lift_late:+.4f}")
        flag = ""
        if fire_range > 20:
            flag += " SCALE_DRIFT_FIRE_RATE"
        if lift_late < lift_early - 0.05:
            flag += " LIFT_COLLAPSE"
        if not flag:
            flag = " STABLE"
        print(f"    verdict:{flag}")


def main() -> None:
    print("=" * 80)
    print(f"6-LANE SCALE-STABILITY AUDIT  (ran {pd.Timestamp.now('UTC')})")
    print("=" * 80)
    payload = json.loads(LANE_ALLOCATION.read_text(encoding="utf-8"))
    deployed = [lane for lane in payload.get("lanes", []) if lane.get("status") == "DEPLOY"]
    print(f"DEPLOY lanes from {LANE_ALLOCATION}: {len(deployed)}")
    for lane in deployed:
        audit_lane(lane)


if __name__ == "__main__":
    main()
