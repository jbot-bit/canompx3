"""Trade Book Builder — curated slot portfolio targeting specific max DD.

Builds a practical "plays" book:
1. Rank slots by Sharpe/DD ratio (best risk-adjusted first)
2. Add slots greedily until portfolio DD exceeds target
3. Layer adaptive sizing to hit exact DD target
4. Output the trade book with income projections

Usage:
    python research/research_trade_book.py
    python research/research_trade_book.py --target-dd 5
    python research/research_trade_book.py --target-dd 10
"""

import sys
import argparse
from pathlib import Path
from math import sqrt
from collections import defaultdict

import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, VolumeFilter
from trading_app.strategy_discovery import (
    _build_filter_day_sets,
    _compute_relative_volumes,
    _load_daily_features,
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reports"))
from report_edge_portfolio import session_slots

sys.stdout.reconfigure(line_buffering=True)

TRADING_DAYS_PER_YEAR = 252


def get_slot_dd(con, slots):
    """Get per-slot max DD and compute Sharpe/DD ratio."""
    for slot in slots:
        sid = slot["head_strategy_id"]
        row = con.execute(
            "SELECT max_drawdown_r FROM validated_setups WHERE strategy_id = ?",
            [sid],
        ).fetchone()
        slot["max_dd"] = row[0] if row and row[0] else 999
        sh = slot["head_sharpe_ann"] or 0
        slot["sh_dd_ratio"] = sh / slot["max_dd"] if slot["max_dd"] > 0 else 0
    return slots


def load_slot_trades(con, selected_slots):
    """Load trades for selected slots using canonical filter logic."""
    by_instrument = defaultdict(list)
    for slot in selected_slots:
        by_instrument[slot["instrument"]].append(slot)

    all_trades = []

    for instrument, inst_slots in by_instrument.items():
        slot_params = {}
        filter_types = set()
        orb_labels = set()
        for slot in inst_slots:
            row = con.execute("""
                SELECT instrument, orb_label, orb_minutes, entry_model,
                       rr_target, confirm_bars, filter_type
                FROM validated_setups WHERE strategy_id = ?
            """, [slot["head_strategy_id"]]).fetchone()
            if not row:
                continue
            cols = ["instrument", "orb_label", "orb_minutes", "entry_model",
                    "rr_target", "confirm_bars", "filter_type"]
            params = dict(zip(cols, row))
            slot_params[slot["head_strategy_id"]] = params
            filter_types.add(params["filter_type"])
            orb_labels.add(params["orb_label"])

        if not slot_params:
            continue

        needed_filters = {k: v for k, v in ALL_FILTERS.items() if k in filter_types}
        features = _load_daily_features(con, instrument, 5, None, None)
        has_vol = any(isinstance(f, VolumeFilter) for f in needed_filters.values())
        if has_vol:
            _compute_relative_volumes(con, features, instrument, sorted(orb_labels), needed_filters)
        filter_days = _build_filter_day_sets(features, sorted(orb_labels), needed_filters)

        for slot in inst_slots:
            sid = slot["head_strategy_id"]
            params = slot_params.get(sid)
            if not params:
                continue
            eligible = filter_days.get(
                (params["filter_type"], params["orb_label"]), set()
            )

            rows = con.execute("""
                SELECT trading_day, outcome, pnl_r
                FROM orb_outcomes
                WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
                  AND entry_model = ? AND rr_target = ? AND confirm_bars = ?
                  AND outcome IN ('win', 'loss')
                ORDER BY trading_day
            """, [
                params["instrument"], params["orb_label"], params["orb_minutes"],
                params["entry_model"], params["rr_target"], params["confirm_bars"],
            ]).fetchall()

            slot_label = f"{instrument}_{params['orb_label']}"
            for r in rows:
                if r[0] in eligible:
                    all_trades.append({
                        "trading_day": r[0],
                        "outcome": r[1],
                        "pnl_r": r[2],
                        "instrument": instrument,
                        "session": params["orb_label"],
                        "slot_label": slot_label,
                        "strategy_id": sid,
                    })

    return all_trades


def compute_portfolio_metrics(trades, start_date=None, end_date=None):
    """Full portfolio metrics from a trade list."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "total_r": 0, "sharpe_ann": None, "max_dd": 0,
                "dd_start": None, "dd_end": None, "dd_duration": None,
                "recovery": None, "worst_day": 0, "worst_day_date": None,
                "wr": 0, "exp_r": 0, "max_conc": 0, "avg_conc": 0}

    all_days = [t["trading_day"] for t in trades]
    if start_date is None:
        start_date = min(all_days)
    if end_date is None:
        end_date = max(all_days)

    wins = sum(1 for t in trades if t["outcome"] == "win")
    total_r = sum(t.get("effective_pnl_r", t["pnl_r"]) for t in trades)

    daily_r = defaultdict(float)
    daily_count = defaultdict(int)
    for t in trades:
        daily_r[t["trading_day"]] += t.get("effective_pnl_r", t["pnl_r"])
        daily_count[t["trading_day"]] += 1

    bdays = pd.bdate_range(start=start_date, end=end_date)
    rmap = dict(daily_r)
    full = [rmap.get(d.date(), 0.0) for d in bdays]

    n_days = len(full)
    sharpe_ann = None
    if n_days > 1:
        mean_d = sum(full) / n_days
        var = sum((v - mean_d) ** 2 for v in full) / (n_days - 1)
        std_d = var ** 0.5
        if std_d > 0:
            sharpe_ann = (mean_d / std_d) * sqrt(TRADING_DAYS_PER_YEAR)

    cum = peak = max_dd = 0.0
    dd_start_d = bdays[0].date() if len(bdays) > 0 else None
    max_dd_start = max_dd_end = None
    worst_day = 0.0
    worst_day_date = None

    for d in bdays:
        dd = d.date()
        r = rmap.get(dd, 0.0)
        cum += r
        if r < worst_day:
            worst_day = r
            worst_day_date = dd
        if cum > peak:
            peak = cum
            dd_start_d = dd
        drawdown = peak - cum
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_start = dd_start_d
            max_dd_end = dd

    recovery = None
    if max_dd_end and max_dd > 0:
        trough_idx = None
        for i, d in enumerate(bdays):
            if d.date() == max_dd_end:
                trough_idx = i
                break
        if trough_idx:
            cs = sum(rmap.get(d.date(), 0.0) for d in bdays[:trough_idx + 1])
            target = cs + max_dd
            for d in bdays[trough_idx + 1:]:
                cs += rmap.get(d.date(), 0.0)
                if cs >= target:
                    recovery = (d.date() - max_dd_end).days
                    break

    dd_dur = (max_dd_end - max_dd_start).days if max_dd_start and max_dd_end else None

    max_conc = max(daily_count.values()) if daily_count else 0
    active = [c for c in daily_count.values() if c > 0]
    avg_conc = sum(active) / len(active) if active else 0

    return {
        "n": n, "total_r": round(total_r, 1),
        "exp_r": round(total_r / n, 4) if n > 0 else 0,
        "wr": round(wins / n, 3) if n > 0 else 0,
        "sharpe_ann": round(sharpe_ann, 2) if sharpe_ann else None,
        "max_dd": round(max_dd, 1),
        "dd_start": max_dd_start, "dd_end": max_dd_end,
        "dd_duration": dd_dur, "recovery": recovery,
        "worst_day": round(worst_day, 2), "worst_day_date": worst_day_date,
        "max_conc": max_conc, "avg_conc": round(avg_conc, 1),
    }


def apply_adaptive_sizing(trades, dd_threshold, reduced_scale):
    """Scale down when in drawdown. Process in day order."""
    by_day = defaultdict(list)
    for t in trades:
        by_day[t["trading_day"]].append(t)

    result = []
    cum = peak = 0.0
    scale_days = 0

    for day in sorted(by_day.keys()):
        dd = peak - cum
        current_scale = reduced_scale if dd >= dd_threshold else 1.0
        if current_scale < 1.0:
            scale_days += 1

        for t in by_day[day]:
            tc = dict(t)
            tc["effective_pnl_r"] = t.get("effective_pnl_r", t["pnl_r"]) * current_scale
            result.append(tc)
            cum += tc["effective_pnl_r"]

        if cum > peak:
            peak = cum

    return result, scale_days


def main():
    parser = argparse.ArgumentParser(description="Trade Book Builder")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--target-dd", type=float, default=5.0,
                        help="Target max drawdown in R (default: 5)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    target_dd = args.target_dd

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        slots = session_slots(db_path)
        if not slots:
            print("No session slots found.")
            return

        # Rank by Sharpe/DD ratio
        get_slot_dd(con, slots)
        ranked = sorted(slots, key=lambda s: -s["sh_dd_ratio"])

        print(f"\n{'#' * 90}")
        print(f"#  TRADE BOOK BUILDER")
        print(f"#  Target Max DD: {target_dd:.0f}R")
        print(f"{'#' * 90}\n")

        # =====================================================================
        # PHASE 1: Find optimal slot count
        # =====================================================================
        print("PHASE 1: Slot selection by Sharpe/DD ratio\n")

        print(f"  {'Rank':>4} {'Slot':<25} {'Sharpe':>7} {'SlotDD':>7} "
              f"{'Sh/DD':>7} {'ExpR':>7} {'N':>5}")
        print(f"  {'-'*4} {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")

        for i, s in enumerate(ranked, 1):
            label = f"{s['instrument']}_{s['session']}"
            print(f"  {i:>4} {label:<25} {s['head_sharpe_ann']:>6.2f} "
                  f"{s['max_dd']:>6.1f}R {s['sh_dd_ratio']:>6.3f} "
                  f"{s['head_expectancy_r']:>+6.3f} {s['trade_day_count']:>5}")

        # Test each slot count
        print(f"\n  Portfolio DD by slot count:")
        print(f"  {'Slots':>6} {'TotalR':>9} {'Sharpe':>8} {'MaxDD':>7} {'WR':>6}")
        print(f"  {'-'*6} {'-'*9} {'-'*8} {'-'*7} {'-'*6}")

        best_no_adapt = None
        for top_n in range(2, len(ranked) + 1):
            selected = ranked[:top_n]
            trades = load_slot_trades(con, selected)
            if not trades:
                continue
            m = compute_portfolio_metrics(trades)
            sh = f"{m['sharpe_ann']:.2f}" if m["sharpe_ann"] else "N/A"
            print(f"  {top_n:>6} {m['total_r']:>+8.1f}R {sh:>8} "
                  f"{m['max_dd']:>6.1f}R {m['wr']:>5.1%}")

            if best_no_adapt is None or (m["max_dd"] <= target_dd * 3):
                best_no_adapt = (top_n, m, trades, selected)

        # =====================================================================
        # PHASE 2: Layer adaptive sizing to hit target DD
        # =====================================================================
        print(f"\n\nPHASE 2: Adaptive sizing to hit {target_dd:.0f}R target DD\n")

        # Try different slot counts with adaptive sizing
        results = []
        for top_n in [6, 8, 10, 12, 15, 20]:
            if top_n > len(ranked):
                continue
            selected = ranked[:top_n]
            trades = load_slot_trades(con, selected)
            if not trades:
                continue

            base_m = compute_portfolio_metrics(trades)
            base_dd = base_m["max_dd"]

            # Binary search for the adaptive threshold that hits target DD
            best_config = None
            for scale in [0.5, 0.25, 0.1]:
                for threshold in [2, 3, 4, 5, 6, 8, 10, 12, 15]:
                    adapted, scale_days = apply_adaptive_sizing(trades, threshold, scale)
                    am = compute_portfolio_metrics(adapted)
                    if am["max_dd"] <= target_dd and am["total_r"] > 0:
                        if best_config is None or am["total_r"] > best_config[2]["total_r"]:
                            best_config = (threshold, scale, am, adapted, scale_days)

            if best_config:
                thr, sc, am, adapted, sd = best_config
                total_days = len(set(t["trading_day"] for t in trades))
                results.append({
                    "slots": top_n,
                    "threshold": thr,
                    "scale": sc,
                    "metrics": am,
                    "trades": adapted,
                    "scale_days": sd,
                    "total_days": total_days,
                    "base_dd": base_dd,
                    "selected": selected,
                })

        if not results:
            print(f"  Cannot achieve {target_dd:.0f}R max DD with any configuration.")
            print(f"  Minimum achievable DD with all slots at 0.1x: check output above.")
            return

        # Show all configurations that hit target
        print(f"  Configurations hitting <={target_dd:.0f}R max DD:")
        print(f"  {'Slots':>6} {'Thr':>5} {'Scale':>6} {'TotalR':>9} {'Sharpe':>8} "
              f"{'MaxDD':>7} {'WR':>6} {'ScalePct':>9}")
        print(f"  {'-'*6} {'-'*5} {'-'*6} {'-'*9} {'-'*8} {'-'*7} {'-'*6} {'-'*9}")

        for r in sorted(results, key=lambda x: -x["metrics"]["total_r"]):
            m = r["metrics"]
            sh = f"{m['sharpe_ann']:.2f}" if m["sharpe_ann"] else "N/A"
            sp = r["scale_days"] / r["total_days"] if r["total_days"] > 0 else 0
            print(f"  {r['slots']:>6} {r['threshold']:>4}R {r['scale']:>5.2f} "
                  f"{m['total_r']:>+8.1f}R {sh:>8} {m['max_dd']:>6.1f}R "
                  f"{m['wr']:>5.1%} {sp:>8.1%}")

        # Pick the best (highest total R that hits target)
        best = max(results, key=lambda x: x["metrics"]["total_r"])
        bm = best["metrics"]
        bt = best["trades"]

        # =====================================================================
        # PHASE 3: THE TRADE BOOK
        # =====================================================================
        print(f"\n{'#' * 90}")
        print(f"#  YOUR TRADE BOOK")
        print(f"#  {best['slots']} slots, adaptive {best['scale']}x at {best['threshold']}R DD")
        print(f"{'#' * 90}")

        print(f"\n  PLAYS (ordered by priority):")
        print(f"  {'#':>4} {'Slot':<25} {'Entry':>5} {'RR':>4} {'Filter':<15} "
              f"{'Sharpe':>7} {'DD':>6} {'ExpR':>7}")
        print(f"  {'-'*4} {'-'*25} {'-'*5} {'-'*4} {'-'*15} {'-'*7} {'-'*6} {'-'*7}")

        for i, slot in enumerate(best["selected"], 1):
            label = f"{slot['instrument']}_{slot['session']}"
            # Get strategy details
            row = con.execute("""
                SELECT entry_model, rr_target, filter_type
                FROM validated_setups WHERE strategy_id = ?
            """, [slot["head_strategy_id"]]).fetchone()
            em, rr, ft = row if row else ("?", 0, "?")
            print(f"  {i:>4} {label:<25} {em:>5} {rr:>4} {ft:<15} "
                  f"{slot['head_sharpe_ann']:>6.2f} {slot['max_dd']:>5.1f}R "
                  f"{slot['head_expectancy_r']:>+6.3f}")

        print(f"\n  RISK RULES:")
        print(f"  - Normal: risk 1R per trade")
        print(f"  - When portfolio DD > {best['threshold']}R from peak: "
              f"risk {best['scale']}R per trade")
        print(f"  - Scale back to 1R when equity recovers to peak")

        print(f"\n  PORTFOLIO STATS:")
        print(f"  Trades:          {bm['n']:,}")
        print(f"  Total R:         {bm['total_r']:+.1f}")
        print(f"  ExpR:            {bm['exp_r']:+.4f}")
        print(f"  Win rate:        {bm['wr']:.1%}")
        if bm["sharpe_ann"]:
            print(f"  Sharpe (ann):    {bm['sharpe_ann']:.2f}")
        print(f"  Max drawdown:    {bm['max_dd']:.1f}R")
        if bm["dd_start"] and bm["dd_end"]:
            print(f"    Period:        {bm['dd_start']} to {bm['dd_end']} "
                  f"({bm['dd_duration']} cal days)")
            if bm["recovery"]:
                print(f"    Recovery:      {bm['recovery']} cal days")
        print(f"  Worst day:       {bm['worst_day']:+.2f}R ({bm['worst_day_date']})")
        print(f"  Max concurrent:  {bm['max_conc']}")
        print(f"  Avg concurrent:  {bm['avg_conc']:.1f}")
        sp = best["scale_days"] / best["total_days"] if best["total_days"] > 0 else 0
        print(f"  Scaled-down:     {best['scale_days']}/{best['total_days']} days ({sp:.1%})")

        # Yearly
        yearly = defaultdict(lambda: {"n": 0, "wins": 0, "total_r": 0.0})
        for t in bt:
            y = t["trading_day"].year
            yearly[y]["n"] += 1
            if t["outcome"] == "win":
                yearly[y]["wins"] += 1
            yearly[y]["total_r"] += t.get("effective_pnl_r", t["pnl_r"])

        print(f"\n  {'Year':>6} {'N':>7} {'TotalR':>9} {'ExpR':>10} {'WR':>7}")
        print(f"  {'----':>6} {'-----':>7} {'------':>9} {'------':>10} {'---':>7}")
        all_pos = True
        for year in sorted(yearly.keys()):
            y = yearly[year]
            wr = y["wins"] / y["n"] if y["n"] > 0 else 0
            expr = y["total_r"] / y["n"] if y["n"] > 0 else 0
            marker = " <-- NEGATIVE" if y["total_r"] < 0 else ""
            print(f"  {year:>6} {y['n']:>7} {y['total_r']:>+8.1f}R "
                  f"{expr:>+9.4f} {wr:>6.1%}{marker}")
            if y["total_r"] < 0:
                all_pos = False
        print(f"\n  Every year positive: {'YES' if all_pos else 'NO'}")

        # =====================================================================
        # INCOME PROJECTION
        # =====================================================================
        r_2024 = yearly.get(2024, {}).get("total_r", 0)
        r_2025 = yearly.get(2025, {}).get("total_r", 0)
        recent_years = sum(1 for y in [2024, 2025] if yearly.get(y, {}).get("total_r", 0) > 0)
        avg_recent = (r_2024 + r_2025) / recent_years if recent_years > 0 else bm["total_r"] / 5

        print(f"\n{'=' * 90}")
        print("INCOME PROJECTION")
        print(f"{'=' * 90}")
        print(f"  2024: {r_2024:+.1f}R  |  2025: {r_2025:+.1f}R  |  Avg: {avg_recent:+.1f}R/year")
        if bm["max_dd"] > 0:
            print(f"  Max DD: {bm['max_dd']:.1f}R  |  Return/DD: {avg_recent/bm['max_dd']:.1f}x")
        print()
        print(f"  {'$/R':>8} {'Annual':>12} {'Max Loss':>12} {'Min Acct':>14} {'Monthly':>12}")
        print(f"  {'---':>8} {'------':>12} {'--------':>12} {'--------':>14} {'-------':>12}")
        for r_dollar in [100, 200, 300, 400, 500, 750, 1000]:
            annual = avg_recent * r_dollar
            dd_dollar = bm["max_dd"] * r_dollar
            acct = dd_dollar * 2.5  # 2.5x DD for comfortable buffer
            monthly = annual / 12
            print(f"  ${r_dollar:>7} ${annual:>10,.0f} ${dd_dollar:>10,.0f} "
                  f"${acct:>12,.0f} ${monthly:>10,.0f}")

        print(f"\n  'Min Acct' = 2.5x MaxDD (comfortable trading buffer).")
        print(f"  Start at $100-200/R. Scale up ONLY after 3+ months profitable live.")
        print(f"  Worst case scenario: lose {bm['max_dd']:.0f}R before recovery.")

        # =====================================================================
        # ALSO SHOW: Target 10R and 15R for comparison
        # =====================================================================
        print(f"\n{'=' * 90}")
        print("COMPARISON: Different DD targets")
        print(f"{'=' * 90}")
        print(f"  {'Target':>8} {'Slots':>6} {'Config':>20} {'TotalR':>9} "
              f"{'Sharpe':>8} {'MaxDD':>7} {'Ann R':>9}")
        print(f"  {'-'*8} {'-'*6} {'-'*20} {'-'*9} {'-'*8} {'-'*7} {'-'*9}")

        # Current best
        sh = f"{bm['sharpe_ann']:.2f}" if bm["sharpe_ann"] else "N/A"
        config = f"{best['scale']}x@{best['threshold']}R"
        print(f"  {target_dd:>7.0f}R {best['slots']:>6} {config:>20} "
              f"{bm['total_r']:>+8.1f}R {sh:>8} {bm['max_dd']:>6.1f}R "
              f"{avg_recent:>+8.1f}R")

        # Other targets for comparison
        for other_dd in [10, 15, 20]:
            if other_dd == target_dd:
                continue
            other_best = None
            for r in results:
                # Recompute at different DD target
                pass

            for top_n in [6, 8, 10, 12, 15, 20]:
                if top_n > len(ranked):
                    continue
                sel = ranked[:top_n]
                tr = load_slot_trades(con, sel)
                if not tr:
                    continue

                for scale in [0.5, 0.25]:
                    for threshold in [3, 5, 8, 10, 15]:
                        adapted, sd = apply_adaptive_sizing(tr, threshold, scale)
                        am = compute_portfolio_metrics(adapted)
                        if am["max_dd"] <= other_dd and am["total_r"] > 0:
                            if other_best is None or am["total_r"] > other_best[2]["total_r"]:
                                other_best = (top_n, f"{scale}x@{threshold}R", am, adapted)

                # Also try no adaptive sizing
                bm2 = compute_portfolio_metrics(tr)
                if bm2["max_dd"] <= other_dd and bm2["total_r"] > 0:
                    if other_best is None or bm2["total_r"] > other_best[2]["total_r"]:
                        other_best = (top_n, "none", bm2, tr)

            if other_best:
                om = other_best[2]
                sh2 = f"{om['sharpe_ann']:.2f}" if om["sharpe_ann"] else "N/A"
                # Estimate annual R from 2024-2025
                oy = defaultdict(float)
                for t in other_best[3]:
                    oy[t["trading_day"].year] += t.get("effective_pnl_r", t["pnl_r"])
                oann = (oy.get(2024, 0) + oy.get(2025, 0)) / 2
                print(f"  {other_dd:>7}R {other_best[0]:>6} {other_best[1]:>20} "
                      f"{om['total_r']:>+8.1f}R {sh2:>8} {om['max_dd']:>6.1f}R "
                      f"{oann:>+8.1f}R")

    finally:
        con.close()


if __name__ == "__main__":
    main()
