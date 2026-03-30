#!/usr/bin/env python3
"""Portfolio reality check — honest P&L and DD from raw data only.

NO stale metadata. NO guessing. For each Apex lane:
1. Load daily_features at the correct orb_minutes
2. Inject cross_atr where needed
3. Apply the EXACT filter (matches_row)
4. Apply ORB size cap from prop_profiles
5. Load orb_outcomes for eligible days only
6. Apply S0.75 tight stop via apply_tight_stop
7. Compute dollar P&L using per-trade risk_dollars
8. Build equity curve, compute max DD

Every number comes from gold.db queries. Nothing from validated_setups.
"""

import sys
from collections import defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter, apply_tight_stop


def run():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    cs = get_cost_spec("MNQ")

    # The 5 Apex lanes — exact params from prop_profiles.py
    lanes = [
        {
            "label": "L1 NYSE_CLOSE",
            "instrument": "MNQ",
            "session": "NYSE_CLOSE",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 1.0,
            "confirm_bars": 1,
            "filter_type": "VOL_RV12_N20",
            "max_orb_size_pts": 100.0,
        },
        {
            "label": "L2 SINGAPORE",
            "instrument": "MNQ",
            "session": "SINGAPORE_OPEN",
            "orb_minutes": 15,
            "entry_model": "E2",
            "rr_target": 4.0,
            "confirm_bars": 1,
            "filter_type": "ORB_G8",
            "max_orb_size_pts": 80.0,
        },
        {
            "label": "L3 COMEX",
            "instrument": "MNQ",
            "session": "COMEX_SETTLE",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 1.0,
            "confirm_bars": 1,
            "filter_type": "ATR70_VOL",
            "max_orb_size_pts": 80.0,
        },
        {
            "label": "L4 NYSE_OPEN",
            "instrument": "MNQ",
            "session": "NYSE_OPEN",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 1.0,
            "confirm_bars": 1,
            "filter_type": "X_MES_ATR60",
            "max_orb_size_pts": 150.0,
        },
        {
            "label": "L5 US_DATA",
            "instrument": "MNQ",
            "session": "US_DATA_1000",
            "orb_minutes": 5,
            "entry_model": "E2",
            "rr_target": 1.0,
            "confirm_bars": 1,
            "filter_type": "X_MES_ATR60",
            "max_orb_size_pts": 120.0,
        },
    ]

    # Pre-load cross-asset ATR for X_MES_ATR60
    mes_atr = {}
    for td, atr_pct in con.execute("""
        SELECT trading_day, atr_20_pct FROM daily_features
        WHERE symbol = 'MES' AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
    """).fetchall():
        mes_atr[td] = float(atr_pct)

    print("=" * 80)
    print("  PORTFOLIO REALITY CHECK — RAW DATA, NO METADATA")
    print("=" * 80)
    print()

    all_trades = []

    for lane in lanes:
        filt = ALL_FILTERS[lane["filter_type"]]
        sess = lane["session"]
        om = lane["orb_minutes"]

        # Step 1: Load daily_features
        features = con.execute(
            """
            SELECT * FROM daily_features
            WHERE symbol = ? AND orb_minutes = ? ORDER BY trading_day
        """,
            [lane["instrument"], om],
        ).fetchall()
        fcols = [d[0] for d in con.description]

        # Step 2: Filter eligible days
        eligible_days = []
        for row_tuple in features:
            row = dict(zip(fcols, row_tuple, strict=False))
            td = row["trading_day"]
            if row.get(f"orb_{sess}_break_dir") is None:
                continue
            orb_size = row.get(f"orb_{sess}_size")
            if orb_size is not None and orb_size > lane["max_orb_size_pts"]:
                continue
            if isinstance(filt, CrossAssetATRFilter):
                val = mes_atr.get(td)
                if val is not None:
                    row[f"cross_atr_{filt.source_instrument}_pct"] = val
            if filt.matches_row(row, sess):
                eligible_days.append(td)

        if not eligible_days:
            print(f"{lane['label']}: 0 eligible — SKIP")
            continue

        # Step 3: Load outcomes
        outcomes = con.execute(
            """
            SELECT trading_day, entry_price, stop_price, outcome,
                   pnl_r, mae_r, risk_dollars, pnl_dollars
            FROM orb_outcomes
            WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
              AND entry_model = ? AND rr_target = ? AND confirm_bars = ?
              AND entry_price IS NOT NULL AND pnl_r IS NOT NULL
              AND trading_day = ANY(?)
            ORDER BY trading_day
        """,
            [lane["instrument"], sess, om, lane["entry_model"], lane["rr_target"], lane["confirm_bars"], eligible_days],
        ).fetchall()

        if not outcomes:
            print(f"{lane['label']}: 0 outcomes — SKIP")
            continue

        # Step 4: Apply S0.75
        o_dicts = [
            {
                "trading_day": r[0],
                "entry_price": r[1],
                "stop_price": r[2],
                "outcome": r[3],
                "pnl_r": r[4],
                "mae_r": r[5],
            }
            for r in outcomes
        ]
        tight = apply_tight_stop(o_dicts, 0.75, cs)

        # Step 5: Dollar P&L
        lane_trades = []
        for orig_row, t_dict in zip(outcomes, tight, strict=False):
            td = orig_row[0]
            risk_d = orig_row[6]
            pnl_d = t_dict["pnl_r"] * risk_d
            lane_trades.append((td, pnl_d, risk_d, t_dict["pnl_r"]))
            all_trades.append((td, lane["label"], pnl_d))

        n = len(lane_trades)
        first_yr = lane_trades[0][0].year
        last_yr = lane_trades[-1][0].year
        years = last_yr - first_yr + 1
        tpy = n / years
        total_d = sum(pd for _, pd, _, _ in lane_trades)
        avg_risk = sum(rd for _, _, rd, _ in lane_trades) / n
        avg_pnl_d = total_d / n

        by_year = defaultdict(lambda: [0.0, 0])
        for td, pd, _, _ in lane_trades:
            by_year[td.year][0] += pd
            by_year[td.year][1] += 1

        print(f"--- {lane['label']} ({lane['filter_type']} O{om} RR{lane['rr_target']}) ---")
        print(f"  Trades: {n:,}  Years: {years}  TPY: {tpy:.0f}")
        print(f"  Avg risk: ${avg_risk:.0f}  Avg $/trade: ${avg_pnl_d:+.2f}")
        print(f"  Total: ${total_d:+,.0f}  Annual: ${total_d / years:+,.0f}")
        pos = 0
        for yr in sorted(by_year):
            yd, yn = by_year[yr]
            if yd > 0:
                pos += 1
            print(f"    {yr}: ${yd:>+9,.0f} ({yn:>3} trades) {'+' if yd > 0 else '-'}")
        print(f"  Positive years: {pos}/{years}")
        print()

    # Combined equity curve
    print("=" * 80)
    print("  COMBINED EQUITY CURVE (all 5 lanes)")
    print("=" * 80)
    print()

    all_trades.sort(key=lambda x: x[0])
    daily_pnl = defaultdict(float)
    daily_n = defaultdict(int)
    yearly_pnl = defaultdict(float)
    yearly_n = defaultdict(int)

    for td, _label, pnl_d in all_trades:
        daily_pnl[td] += pnl_d
        daily_n[td] += 1
        yearly_pnl[td.year] += pnl_d
        yearly_n[td.year] += 1

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_date = None
    worst_day = 0.0
    worst_day_date = None

    for td in sorted(daily_pnl):
        equity += daily_pnl[td]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            max_dd_date = td
        if daily_pnl[td] < worst_day:
            worst_day = daily_pnl[td]
            worst_day_date = td

    n_total = len(all_trades)
    first_date = min(daily_pnl)
    last_date = max(daily_pnl)
    years_total = last_date.year - first_date.year + 1

    print(f"Range: {first_date} to {last_date} ({years_total} years)")
    print(f"Trades: {n_total:,} ({n_total / years_total:.0f}/yr)")
    print()
    print(f"Final equity:     ${equity:>+12,.0f}")
    print(f"Annual:           ${equity / years_total:>+12,.0f}")
    print(f"Monthly:          ${equity / years_total / 12:>+12,.0f}")
    print()
    print(f"Max drawdown:     ${max_dd:>12,.0f}  ({max_dd_date})")
    print(f"Worst day:        ${worst_day:>12,.0f}  ({worst_day_date})")
    print()
    for yr in sorted(yearly_pnl):
        print(f"  {yr}: ${yearly_pnl[yr]:>+10,.0f}  ({yearly_n[yr]:>4} trades)")

    print()
    print("=" * 80)
    print("  ACCOUNT SURVIVAL")
    print("=" * 80)
    print()
    print("  Apex DD limit: $2,000")
    print(f"  Max DD:        ${max_dd:,.0f}")
    if max_dd > 2000:
        print(f"  *** BLOWN by ${max_dd - 2000:,.0f} ***")
    elif max_dd > 1500:
        print(f"  *** DANGER — ${2000 - max_dd:,.0f} margin ***")
    else:
        print(f"  SURVIVES — ${2000 - max_dd:,.0f} margin")

    print()
    annual = equity / years_total
    split = annual * 0.75
    print(f"  Gross:       ${annual:+,.0f}/yr = ${annual / 12:+,.0f}/mo")
    print(f"  75% split:   ${split:+,.0f}/yr = ${split / 12:+,.0f}/mo")

    con.close()


if __name__ == "__main__":
    run()
