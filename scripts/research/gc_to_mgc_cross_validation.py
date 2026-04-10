"""GC -> MGC Cross-Validation: Confirm proxy edges transfer to micro data.

Reads validated GC strategies, applies the same filters to MGC orb_outcomes,
and reports whether the edge holds on micro data (2022-2026).

This is CONFIRMATION testing, not discovery. No writes to validated_setups.
Uses MGC cost specs, not GC.

Usage:
    python scripts/research/gc_to_mgc_cross_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS


def run_cross_validation() -> None:
    mgc_cost = get_cost_spec("MGC")
    import time
    for attempt in range(10):
        try:
            con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
            break
        except duckdb.IOException:
            if attempt < 9:
                time.sleep(1)
            else:
                raise

    # Load validated GC strategies
    gc_strats = con.execute("""
        SELECT strategy_id, orb_label, filter_type, filter_params,
               rr_target, confirm_bars, entry_model, orb_minutes,
               expectancy_r, win_rate, sample_size
        FROM validated_setups
        WHERE instrument = 'GC'
        ORDER BY strategy_id
    """).fetchall()
    gc_cols = [d[0] for d in con.description]

    print(f"{'='*100}")
    print(f"GC -> MGC CROSS-VALIDATION REPORT")
    print(f"{'='*100}")
    print(f"GC strategies: {len(gc_strats)}")
    print(f"MGC cost: ${mgc_cost.total_friction:.2f} friction")
    print()

    results = []

    for row in gc_strats:
        strat = dict(zip(gc_cols, row))
        sid = strat["strategy_id"]
        orb_label = strat["orb_label"]
        filter_type = strat["filter_type"]
        rr_target = strat["rr_target"]
        confirm_bars = strat["confirm_bars"]
        entry_model = strat["entry_model"]
        orb_minutes = strat["orb_minutes"]

        # Get filter object
        filt = ALL_FILTERS.get(filter_type)
        if filt is None:
            print(f"SKIP {sid}: unknown filter_type '{filter_type}'")
            continue

        # Build filter SQL condition from the filter's describe() method
        # Instead of calling matches_row per-trade (slow), use SQL conditions
        filter_col = f"orb_{orb_label}_size_points"

        # Map filter types to SQL WHERE clauses on daily_features
        if filter_type == "NO_FILTER":
            filter_sql = "1=1"
        elif filter_type.startswith("ORB_G"):
            threshold_col = f"orb_{orb_label}_g{filter_type.replace('ORB_G', '')}_threshold"
            filter_sql = f"d.{filter_col} >= d.{threshold_col}"
        elif filter_type.startswith("ATR_P"):
            pct = int(filter_type.replace("ATR_P", ""))
            filter_sql = f"d.atr_20_pct >= {pct}"
        elif filter_type.startswith("PDR_R"):
            threshold = filter_type.replace("PDR_R", "")
            ratio = int(threshold) / 100
            filter_sql = f"d.prev_day_range / NULLIF(d.atr_20, 0) >= {ratio}"
        elif filter_type.startswith("OVNRNG_"):
            pct = int(filter_type.replace("OVNRNG_", ""))
            filter_sql = f"d.overnight_range_pct >= {pct}"
        elif filter_type.startswith("COST_LT"):
            threshold = int(filter_type.replace("COST_LT", ""))
            filter_sql = f"({mgc_cost.total_friction} / NULLIF(d.{filter_col} * {mgc_cost.point_value}, 0) * 100) < {threshold}"
        else:
            print(f"SKIP {sid}: unhandled filter_type '{filter_type}' for SQL generation")
            continue

        # Query MGC outcomes with the same filter applied
        query = f"""
            SELECT
                EXTRACT(YEAR FROM o.trading_day) as year,
                COUNT(*) as n,
                AVG(CASE WHEN o.outcome = 'win' THEN 1.0 ELSE 0.0 END) as wr,
                AVG(o.pnl_r) as avg_r,
                STDDEV(o.pnl_r) as std_r
            FROM orb_outcomes o
            JOIN daily_features d
                ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = 'MGC'
                AND o.orb_label = '{orb_label}'
                AND o.entry_model = '{entry_model}'
                AND o.confirm_bars = {confirm_bars}
                AND o.rr_target = {rr_target}
                AND o.orb_minutes = {orb_minutes}
                AND o.trading_day < '2026-01-01'
                AND {filter_sql}
            GROUP BY year
            ORDER BY year
        """

        try:
            yearly = con.execute(query).fetchall()
        except Exception as e:
            print(f"ERROR {sid}: {e}")
            continue

        if not yearly:
            print(f"SKIP {sid}: no MGC data matched")
            continue

        # Compute aggregate
        agg_query = f"""
            SELECT
                COUNT(*) as n,
                AVG(CASE WHEN o.outcome = 'win' THEN 1.0 ELSE 0.0 END) as wr,
                AVG(o.pnl_r) as avg_r,
                STDDEV(o.pnl_r) as std_r
            FROM orb_outcomes o
            JOIN daily_features d
                ON o.trading_day = d.trading_day
                AND o.symbol = d.symbol
                AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = 'MGC'
                AND o.orb_label = '{orb_label}'
                AND o.entry_model = '{entry_model}'
                AND o.confirm_bars = {confirm_bars}
                AND o.rr_target = {rr_target}
                AND o.orb_minutes = {orb_minutes}
                AND o.trading_day < '2026-01-01'
                AND {filter_sql}
        """
        agg = con.execute(agg_query).fetchone()
        if agg is None or agg[0] == 0:
            print(f"SKIP {sid}: no MGC aggregate data")
            continue
        mgc_n, mgc_wr, mgc_expr, mgc_std = agg

        # Sharpe (annualized)
        trades_per_year = mgc_n / max(len(yearly), 1)
        mgc_sharpe = (mgc_expr / mgc_std * (trades_per_year ** 0.5)) if mgc_std and mgc_std > 0 else 0

        # Compare to GC
        gc_expr = strat["expectancy_r"]
        gc_wr = strat["win_rate"]
        gc_n = strat["sample_size"]

        # Verdict
        if mgc_n < 30:
            verdict = "INSUFFICIENT_DATA"
        elif mgc_expr <= 0:
            verdict = "FAILED"
        elif mgc_n < 100:
            verdict = "TENTATIVE"
        else:
            verdict = "CONFIRMED"

        results.append({
            "gc_sid": sid,
            "orb_label": orb_label,
            "filter_type": filter_type,
            "rr_target": rr_target,
            "gc_n": gc_n,
            "gc_wr": gc_wr,
            "gc_expr": gc_expr,
            "mgc_n": mgc_n,
            "mgc_wr": mgc_wr,
            "mgc_expr": mgc_expr,
            "mgc_sharpe": mgc_sharpe,
            "verdict": verdict,
            "yearly": yearly,
        })

    con.close()

    # Print results
    print(f"{'Strategy':<55} {'GC':>20} {'MGC (micro)':>25} {'Verdict':>15}")
    print(f"{'':55} {'N':>5} {'WR':>6} {'ExpR':>7}   {'N':>5} {'WR':>6} {'ExpR':>7} {'Sharpe':>6}")
    print("-" * 130)

    for r in results:
        print(
            f"{r['gc_sid']:<55} "
            f"{r['gc_n']:>5d} {r['gc_wr']:>5.1%} {r['gc_expr']:>7.3f}   "
            f"{r['mgc_n']:>5d} {r['mgc_wr']:>5.1%} {r['mgc_expr']:>7.3f} {r['mgc_sharpe']:>6.2f}  "
            f"{r['verdict']:>12}"
        )

    # Per-year breakdown for confirmed/tentative
    print()
    for r in results:
        if r["verdict"] in ("CONFIRMED", "TENTATIVE"):
            print(f"\n  {r['gc_sid']} — per-year on MGC:")
            for yr in r["yearly"]:
                sign = "+" if yr[3] >= 0 else ""
                print(f"    {int(yr[0])}: N={yr[1]:>4d}  WR={yr[2]:.1%}  ExpR={sign}{yr[3]:.3f}")

    # Summary
    print(f"\n{'='*100}")
    confirmed = [r for r in results if r["verdict"] == "CONFIRMED"]
    tentative = [r for r in results if r["verdict"] == "TENTATIVE"]
    failed = [r for r in results if r["verdict"] == "FAILED"]
    insufficient = [r for r in results if r["verdict"] == "INSUFFICIENT_DATA"]

    print(f"CONFIRMED:         {len(confirmed)}  (ExpR>0, N>=100 — ready for MGC hypothesis file)")
    print(f"TENTATIVE:         {len(tentative)}  (ExpR>0, N<100 — wait for data)")
    print(f"FAILED:            {len(failed)}  (ExpR<=0 — edge does NOT transfer)")
    print(f"INSUFFICIENT_DATA: {len(insufficient)}  (N<30 — cannot judge)")

    if confirmed:
        print(f"\nCONFIRMED strategies for MGC deployment:")
        for r in confirmed:
            mgc_sid = r["gc_sid"].replace("GC_", "MGC_")
            print(f"  {mgc_sid}: ExpR={r['mgc_expr']:.3f} WR={r['mgc_wr']:.1%} N={r['mgc_n']}")


if __name__ == "__main__":
    run_cross_validation()
