"""Prop Firm Fit Analysis — Does this system fit inside a trailing DD budget?

Converts R-based portfolio stats to actual DOLLARS per instrument,
then tests whether various portfolio configurations survive trailing DD limits.

Answers:
- "I have a $2-3K trailing DD prop firm. What can I trade?"
- "I have $20K personal capital. How does DD play out?"
- "How many slots, which adaptive config, what income?"

Usage:
    python research/research_prop_firm_fit.py
    python research/research_prop_firm_fit.py --dd-budget 2000
    python research/research_prop_firm_fit.py --dd-budget 3000 --personal-capital 20000
"""

import sys
import argparse
from pathlib import Path
from math import sqrt
from collections import defaultdict
from datetime import date

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

# Dollar per point for micro contracts (from cost_model.py CostSpec.point_value)
DOLLAR_PER_POINT = {
    "MGC": 10.0,   # $10 per point
    "MNQ": 2.0,    # $2 per point
    "MES": 5.0,    # $5 per point
    "M2K": 5.0,    # $5 per point
}

# Total round-trip friction per contract (commission + spread + slippage)
FRICTION = {
    "MGC": 8.40,   # 2.40 + 2.00 + 4.00
    "MNQ": 2.74,   # 1.24 + 0.50 + 1.00
    "MES": 3.74,   # 1.24 + 1.25 + 1.25
    "M2K": 3.24,   # 1.24 + 1.00 + 1.00
}


def load_slot_trades_with_dollars(con, selected_slots):
    """Load trades with actual dollar P&L per trade.

    For each trade, JOINs to daily_features to get the actual ORB size
    on that specific trading day, then computes:
        $/R = ORB_size * dollar_per_point + friction
        pnl_dollars = pnl_r * $/R
    """
    by_instrument = defaultdict(list)
    for slot in selected_slots:
        by_instrument[slot["instrument"]].append(slot)

    all_trades = []
    slot_dollar_per_r = {}  # median $/R per slot (for display)

    for instrument, inst_slots in by_instrument.items():
        dpp = DOLLAR_PER_POINT.get(instrument, 10.0)
        friction = FRICTION.get(instrument, 5.0)

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
            _compute_relative_volumes(con, features, instrument,
                                      sorted(orb_labels), needed_filters)
        filter_days = _build_filter_day_sets(features, sorted(orb_labels),
                                             needed_filters)

        # Build a lookup: (trading_day, orb_label) -> ORB size in points
        orb_size_lookup = {}
        for label in orb_labels:
            col = f"orb_{label}_size"
            try:
                rows = con.execute(f"""
                    SELECT trading_day, "{col}"
                    FROM daily_features
                    WHERE symbol = ? AND orb_minutes = 5
                      AND "{col}" IS NOT NULL AND "{col}" > 0
                """, [instrument]).fetchall()
                for r in rows:
                    orb_size_lookup[(r[0], label)] = r[1]
            except Exception:
                pass

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
            slot_dprs = []

            for r in rows:
                if r[0] not in eligible:
                    continue
                pnl_r = r[2]
                # Get ACTUAL ORB size on this specific trading day
                orb_size = orb_size_lookup.get((r[0], params["orb_label"]))
                if orb_size and orb_size > 0:
                    trade_dpr = orb_size * dpp + friction
                else:
                    # Fallback: use 5 points (conservative estimate)
                    trade_dpr = 5 * dpp + friction

                slot_dprs.append(trade_dpr)
                all_trades.append({
                    "trading_day": r[0],
                    "outcome": r[1],
                    "pnl_r": pnl_r,
                    "pnl_dollars": pnl_r * trade_dpr,
                    "dollar_per_r": trade_dpr,
                    "instrument": instrument,
                    "session": params["orb_label"],
                    "slot_label": slot_label,
                })

            if slot_dprs:
                slot_dollar_per_r[slot_label] = float(np.median(slot_dprs))

    return all_trades, slot_dollar_per_r


def compute_dollar_metrics(trades, adaptive_threshold_r=None, adaptive_scale=None):
    """Compute portfolio metrics in actual dollars with optional adaptive sizing."""
    if not trades:
        return {}

    # Sort and apply adaptive sizing if requested
    by_day = defaultdict(list)
    for t in trades:
        by_day[t["trading_day"]].append(t)

    daily_dollars = {}
    daily_r = {}
    daily_count = {}
    cum_r = peak_r = 0.0
    cum_dollars = peak_dollars = 0.0
    max_dd_dollars = 0.0
    max_dd_r = 0.0
    dd_start = dd_end = None
    dd_start_d = None
    worst_day_dollars = 0.0
    worst_day_date = None
    total_trades = 0
    wins = 0
    total_dollars = 0.0
    total_r_val = 0.0
    scale_days = 0
    total_trade_days = 0

    for day in sorted(by_day.keys()):
        total_trade_days += 1
        dd_current_r = peak_r - cum_r
        scale = 1.0
        if adaptive_threshold_r and adaptive_scale and dd_current_r >= adaptive_threshold_r:
            scale = adaptive_scale
            scale_days += 1

        day_dollars = 0.0
        day_r = 0.0
        day_count = len(by_day[day])
        for t in by_day[day]:
            effective_r = t["pnl_r"] * scale
            effective_dollars = t["pnl_dollars"] * scale
            day_dollars += effective_dollars
            day_r += effective_r
            total_trades += 1
            total_dollars += effective_dollars
            total_r_val += effective_r
            if t["outcome"] == "win":
                wins += 1

        daily_dollars[day] = day_dollars
        daily_r[day] = day_r
        daily_count[day] = day_count

        cum_dollars += day_dollars
        cum_r += day_r

        if day_dollars < worst_day_dollars:
            worst_day_dollars = day_dollars
            worst_day_date = day

        if cum_dollars > peak_dollars:
            peak_dollars = cum_dollars
            dd_start_d = day
        if cum_r > peak_r:
            peak_r = cum_r

        dd_d = peak_dollars - cum_dollars
        dd_r_val = peak_r - cum_r
        if dd_d > max_dd_dollars:
            max_dd_dollars = dd_d
            dd_start = dd_start_d
            dd_end = day
        if dd_r_val > max_dd_r:
            max_dd_r = dd_r_val

    # Sharpe from daily dollar returns
    days = sorted(daily_dollars.keys())
    values = [daily_dollars[d] for d in days]

    # Fill business days for proper Sharpe
    if days:
        bdays = pd.bdate_range(start=min(days), end=max(days))
        full = [daily_dollars.get(d.date(), 0.0) for d in bdays]
        n_days = len(full)
        sharpe = None
        if n_days > 1:
            mean_d = sum(full) / n_days
            var = sum((v - mean_d) ** 2 for v in full) / (n_days - 1)
            std = var ** 0.5
            if std > 0:
                sharpe = (mean_d / std) * sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = None
        n_days = 0

    # DD duration
    dd_dur = (dd_end - dd_start).days if dd_start and dd_end else None

    # Recovery time
    recovery_days = None
    if dd_end and max_dd_dollars > 0:
        # Find trough, then find when cum equity recovers
        cum_check = 0.0
        trough_cum = None
        for d in sorted(daily_dollars.keys()):
            cum_check += daily_dollars[d]
            if d == dd_end:
                trough_cum = cum_check
                break
        if trough_cum is not None:
            target = trough_cum + max_dd_dollars
            for d in sorted(daily_dollars.keys()):
                if d <= dd_end:
                    continue
                cum_check += daily_dollars[d]
                if cum_check >= target:
                    recovery_days = (d - dd_end).days
                    break

    return {
        "total_trades": total_trades,
        "wins": wins,
        "total_dollars": round(total_dollars, 2),
        "total_r": round(total_r_val, 1),
        "exp_r": round(total_r_val / total_trades, 4) if total_trades > 0 else 0,
        "wr": round(wins / total_trades, 3) if total_trades > 0 else 0,
        "sharpe": round(sharpe, 2) if sharpe else None,
        "max_dd_dollars": round(max_dd_dollars, 2),
        "max_dd_r": round(max_dd_r, 1),
        "dd_start": dd_start,
        "dd_end": dd_end,
        "dd_duration_days": dd_dur,
        "recovery_days": recovery_days,
        "worst_day_dollars": round(worst_day_dollars, 2),
        "worst_day_date": worst_day_date,
        "max_concurrent": max(daily_count.values()) if daily_count else 0,
        "avg_concurrent": round(sum(daily_count.values()) / len(daily_count), 1) if daily_count else 0,
        "trade_days": total_trade_days,
        "n_business_days": n_days,
        "scale_days": scale_days,
        "scale_pct": round(scale_days / total_trade_days * 100, 1) if total_trade_days > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Prop Firm Fit Analysis")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--dd-budget", type=float, default=None,
                        help="Trailing DD budget in dollars (e.g. 2000, 2500, 3000)")
    parser.add_argument("--personal-capital", type=float, default=None,
                        help="Personal capital in dollars (e.g. 20000)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    dd_budget = args.dd_budget
    personal_capital = args.personal_capital

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        # Load and rank slots
        all_slots = session_slots(db_path)
        for slot in all_slots:
            sid = slot["head_strategy_id"]
            row = con.execute(
                "SELECT max_drawdown_r, entry_model, rr_target, confirm_bars, filter_type "
                "FROM validated_setups WHERE strategy_id = ?", [sid],
            ).fetchone()
            slot["max_dd"] = row[0] if row and row[0] else 999
            slot["entry_model"] = row[1] if row else "?"
            slot["rr_target"] = row[2] if row else 0
            slot["confirm_bars"] = row[3] if row else 0
            slot["filter_type"] = row[4] if row else "?"
            sh = slot["head_sharpe_ann"] or 0
            slot["sh_dd_ratio"] = sh / slot["max_dd"] if slot["max_dd"] > 0 else 0

        ranked = sorted(all_slots, key=lambda s: -s["sh_dd_ratio"])
        n_total_slots = len(ranked)

        # Load ALL slot trades with dollar amounts
        print("Loading trade data with actual dollar P&L per instrument...")
        all_trades, slot_dpr = load_slot_trades_with_dollars(con, ranked)
        print(f"  Loaded {len(all_trades):,} trades across {len(slot_dpr)} slots\n")

        # =====================================================================
        # SECTION 1: Actual $/R per slot
        # =====================================================================
        print("=" * 90)
        print("ACTUAL $/R PER SLOT (from median ORB sizes of trades taken)")
        print("=" * 90)
        print(f"\n  {'Slot':<25} {'Median$/R':>10} {'Filter':>10} "
              f"{'Min ORB':>8} {'Notes':<25}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*25}")

        for slot in ranked:
            label = f"{slot['instrument']}_{slot['session']}"
            dpr = slot_dpr.get(label, 0)
            filt = slot.get("filter_type", "?")
            # G filter sets floor: G4=4pts, G5=5pts, etc.
            dpp = DOLLAR_PER_POINT.get(slot["instrument"], 10.0)
            friction = FRICTION.get(slot["instrument"], 5.0)
            # Extract min ORB from filter name
            min_orb = "?"
            for g in ["G4", "G5", "G6", "G8"]:
                if g in filt:
                    min_orb = f"{int(g[1:])} pts"
                    break
            print(f"  {label:<25} ${dpr:>8.0f} {filt:>10} "
                  f"{min_orb:>8} 1R loss = ${dpr:.0f}")

        weighted_dpr = sum(
            t["dollar_per_r"] for t in all_trades
        ) / len(all_trades) if all_trades else 80

        print(f"\n  Trade-weighted average $/R: ${weighted_dpr:.0f}")
        print(f"  Range: ${min(t['dollar_per_r'] for t in all_trades):.0f} - "
              f"${max(t['dollar_per_r'] for t in all_trades):.0f}")
        print(f"  (Each trade uses the ACTUAL ORB size that day, not an average)")

        # =====================================================================
        # SECTION 2: Portfolio configs in DOLLARS
        # =====================================================================
        print(f"\n{'=' * 90}")
        print("PORTFOLIO CONFIGURATIONS IN ACTUAL DOLLARS")
        print("(1 micro contract per trade)")
        print(f"{'=' * 90}\n")

        configs = [
            ("6 slots, no adaptive",   6,  None, None),
            ("10 slots, no adaptive",  10, None, None),
            ("15 slots, no adaptive",  15, None, None),
            ("All slots, no adaptive", n_total_slots, None, None),
            ("6 slots, 0.5x @ 4R DD", 6,  4.0, 0.5),
            ("10 slots, 0.25x @ 2R",  10, 2.0, 0.25),
            ("10 slots, 0.5x @ 5R",   10, 5.0, 0.5),
            ("15 slots, 0.25x @ 3R",  15, 3.0, 0.25),
            ("15 slots, 0.5x @ 8R",   15, 8.0, 0.5),
        ]

        results = []
        for label, n_slots, adapt_thresh, adapt_scale in configs:
            selected = ranked[:n_slots]
            selected_labels = {f"{s['instrument']}_{s['session']}" for s in selected}
            trades = [t for t in all_trades if t["slot_label"] in selected_labels]

            m = compute_dollar_metrics(trades, adapt_thresh, adapt_scale)
            m["label"] = label
            m["n_slots"] = n_slots
            results.append(m)

        print(f"  {'Config':<27} {'Slots':>5} {'MaxDD$':>9} {'MaxDD R':>8} "
              f"{'Ann$':>10} {'Sharpe':>7} {'Worst Day':>10} {'Scaled%':>8}")
        print(f"  {'-'*27} {'-'*5} {'-'*9} {'-'*8} "
              f"{'-'*10} {'-'*7} {'-'*10} {'-'*8}")

        for m in results:
            years = m["n_business_days"] / 252 if m["n_business_days"] > 0 else 1
            ann_dollars = m["total_dollars"] / years
            print(f"  {m['label']:<27} {m['n_slots']:>5} "
                  f"${m['max_dd_dollars']:>7,.0f} {m['max_dd_r']:>7.1f}R "
                  f"${ann_dollars:>8,.0f} {m.get('sharpe', 0) or 0:>7.2f} "
                  f"${m['worst_day_dollars']:>8,.0f} {m['scale_pct']:>7.1f}%")

        # =====================================================================
        # SECTION 3: PROP FIRM FIT TABLE
        # =====================================================================
        print(f"\n{'=' * 90}")
        print("PROP FIRM FIT TABLE")
        print("Does each config survive the trailing DD limit?")
        print(f"{'=' * 90}\n")

        dd_budgets = [1500, 2000, 2500, 3000, 4000, 5000]
        if dd_budget and dd_budget not in dd_budgets:
            dd_budgets.append(dd_budget)
            dd_budgets.sort()

        header = f"  {'Config':<27}"
        for b in dd_budgets:
            header += f" {'$' + f'{b:,.0f}':>7}"
        print(header)
        print(f"  {'-'*27}" + f" {'-'*7}" * len(dd_budgets))

        for m in results:
            line = f"  {m['label']:<27}"
            for b in dd_budgets:
                if m["max_dd_dollars"] <= b * 0.80:
                    status = "  SAFE"
                elif m["max_dd_dollars"] <= b:
                    status = " TIGHT"
                else:
                    status = "  FAIL"
                line += f" {status:>7}"
            print(line)

        print()
        print("  SAFE  = Max DD is <80% of budget (comfortable margin)")
        print("  TIGHT = Max DD is 80-100% of budget (works but risky)")
        print("  FAIL  = Max DD exceeds budget (will blow the account)")
        print()
        print("  NOTE: Historical max DD is a LOWER BOUND on future DD.")
        print("  Future DD could be worse. A 20% safety margin is minimum.")

        # =====================================================================
        # SECTION 4: RECOMMENDED CONFIG PER DD BUDGET
        # =====================================================================
        print(f"\n{'=' * 90}")
        print("RECOMMENDED CONFIGURATION PER DD BUDGET")
        print(f"{'=' * 90}\n")

        for budget in dd_budgets:
            # Find best config that fits SAFELY (< 80% of budget)
            safe_configs = [
                m for m in results
                if m["max_dd_dollars"] <= budget * 0.80
            ]
            if safe_configs:
                # Best = highest annual dollars among safe configs
                best = max(safe_configs,
                           key=lambda m: m["total_dollars"] / max(m["n_business_days"] / 252, 0.5))
                years = best["n_business_days"] / 252 if best["n_business_days"] > 0 else 1
                ann = best["total_dollars"] / years
                margin_pct = (1 - best["max_dd_dollars"] / budget) * 100
                print(f"  ${budget:>5,} DD budget -> {best['label']}")
                print(f"    Max DD: ${best['max_dd_dollars']:,.0f} ({margin_pct:.0f}% safety margin)")
                print(f"    Annual income: ${ann:,.0f}")
                print(f"    Monthly income: ${ann/12:,.0f}")
                print(f"    Trades/year: ~{best['total_trades'] / years:.0f}")
                print(f"    Sharpe: {best.get('sharpe', 0) or 0:.2f}")
            else:
                # Find tightest fit
                tight = [m for m in results if m["max_dd_dollars"] <= budget]
                if tight:
                    best = max(tight,
                               key=lambda m: m["total_dollars"] / max(m["n_business_days"] / 252, 0.5))
                    years = best["n_business_days"] / 252 if best["n_business_days"] > 0 else 1
                    ann = best["total_dollars"] / years
                    margin_pct = (1 - best["max_dd_dollars"] / budget) * 100
                    print(f"  ${budget:>5,} DD budget -> {best['label']} (TIGHT)")
                    print(f"    Max DD: ${best['max_dd_dollars']:,.0f} ({margin_pct:.0f}% margin — risky)")
                    print(f"    Annual income: ${ann:,.0f}")
                    print(f"    Monthly income: ${ann/12:,.0f}")
                else:
                    print(f"  ${budget:>5,} DD budget -> NO SAFE CONFIG")
                    print(f"    Need to reduce slots or use aggressive adaptive sizing")
            print()

        # =====================================================================
        # SECTION 5: PROP FIRM SPECIFIC ANALYSIS
        # =====================================================================
        print(f"{'=' * 90}")
        print("PROP FIRM SCENARIOS (Apex / Topstep)")
        print(f"{'=' * 90}\n")

        # Apex $50K PA: trailing DD is $2,500 (then locks at floor)
        # Apex $100K PA: trailing DD = $3,000
        # Apex $150K PA: trailing DD = $4,500
        # Topstep $50K: trailing DD = $2,000
        # Topstep $100K: trailing DD = $3,000
        # Topstep $150K: trailing DD = $4,500

        prop_firms = [
            ("Apex $50K PA", 2500, "Trailing locks at floor after profit. 30% intraday rule."),
            ("Apex $100K PA", 3000, "Same rules, larger DD budget."),
            ("Topstep $50K Express", 2000, "EOD trailing DD (not intrabar). No consistency rule."),
            ("Topstep $100K Express", 3000, "EOD trailing. More forgiving for ORB trades."),
            ("Topstep $150K Express", 4500, "Generous DD. Room for full 15-slot system."),
        ]

        for firm_name, firm_dd, firm_notes in prop_firms:
            safe_configs = [
                m for m in results
                if m["max_dd_dollars"] <= firm_dd * 0.80
            ]
            if safe_configs:
                best = max(safe_configs,
                           key=lambda m: m["total_dollars"] / max(m["n_business_days"] / 252, 0.5))
                years = best["n_business_days"] / 252 if best["n_business_days"] > 0 else 1
                ann = best["total_dollars"] / years
                margin_pct = (1 - best["max_dd_dollars"] / firm_dd) * 100

                # Monthly income math
                monthly = ann / 12
                # How long to pass evaluation (assume need ~$3K-5K profit)
                eval_target = firm_dd * 2  # Rough: need 2x DD as profit target
                days_to_eval = eval_target / (ann / 252) if ann > 0 else 999

                verdict = "YES"
            else:
                tight = [m for m in results if m["max_dd_dollars"] <= firm_dd]
                if tight:
                    best = max(tight,
                               key=lambda m: m["total_dollars"] / max(m["n_business_days"] / 252, 0.5))
                    years = best["n_business_days"] / 252 if best["n_business_days"] > 0 else 1
                    ann = best["total_dollars"] / years
                    margin_pct = (1 - best["max_dd_dollars"] / firm_dd) * 100
                    monthly = ann / 12
                    eval_target = firm_dd * 2
                    days_to_eval = eval_target / (ann / 252) if ann > 0 else 999
                    verdict = "RISKY"
                else:
                    verdict = "NO"
                    best = None

            print(f"  {firm_name} (DD limit: ${firm_dd:,})")
            print(f"  {'-' * 50}")
            if best:
                print(f"    Verdict: {verdict}")
                print(f"    Best config: {best['label']}")
                print(f"    Max DD: ${best['max_dd_dollars']:,.0f} "
                      f"(safety margin: {margin_pct:.0f}%)")
                print(f"    Annual income: ${ann:,.0f}")
                print(f"    Monthly income: ${monthly:,.0f}")
                print(f"    Est. days to pass eval: ~{days_to_eval:.0f} trading days")
                print(f"    {firm_notes}")
            else:
                print(f"    Verdict: {verdict}")
                print(f"    System does not fit this DD budget at 1 micro contract.")
            print()

        # =====================================================================
        # SECTION 6: PERSONAL CAPITAL ANALYSIS
        # =====================================================================
        capital = personal_capital or 20000

        print(f"{'=' * 90}")
        print(f"PERSONAL CAPITAL ANALYSIS (${capital:,.0f})")
        print(f"{'=' * 90}\n")

        # At 1 contract per trade
        best_15 = [m for m in results if m["label"] == "15 slots, no adaptive"][0]
        years = best_15["n_business_days"] / 252
        ann_1c = best_15["total_dollars"] / years
        dd_1c = best_15["max_dd_dollars"]
        dd_pct_1c = dd_1c / capital * 100

        print(f"  AT 1 MICRO CONTRACT PER TRADE (all 15 slots):")
        print(f"    Max DD: ${dd_1c:,.0f} ({dd_pct_1c:.1f}% of account)")
        print(f"    Annual income: ${ann_1c:,.0f}")
        print(f"    Monthly income: ${ann_1c/12:,.0f}")
        print(f"    Return on capital: {ann_1c/capital*100:.1f}%")
        print(f"    Risk of ruin: VERY LOW (DD is {dd_pct_1c:.1f}% of account)")
        print()

        # At 2 contracts
        dd_2c = dd_1c * 2
        ann_2c = ann_1c * 2
        dd_pct_2c = dd_2c / capital * 100
        print(f"  AT 2 MICRO CONTRACTS PER TRADE:")
        print(f"    Max DD: ${dd_2c:,.0f} ({dd_pct_2c:.1f}% of account)")
        print(f"    Annual income: ${ann_2c:,.0f}")
        print(f"    Monthly income: ${ann_2c/12:,.0f}")
        print(f"    Return on capital: {ann_2c/capital*100:.1f}%")
        if dd_pct_2c > 25:
            print(f"    WARNING: DD exceeds 25% of account — aggressive")
        elif dd_pct_2c > 15:
            print(f"    CAUTION: DD is {dd_pct_2c:.0f}% of account — moderate risk")
        else:
            print(f"    Risk of ruin: LOW (DD is {dd_pct_2c:.1f}% of account)")
        print()

        # At 3 contracts
        dd_3c = dd_1c * 3
        ann_3c = ann_1c * 3
        dd_pct_3c = dd_3c / capital * 100
        print(f"  AT 3 MICRO CONTRACTS PER TRADE:")
        print(f"    Max DD: ${dd_3c:,.0f} ({dd_pct_3c:.1f}% of account)")
        print(f"    Annual income: ${ann_3c:,.0f}")
        print(f"    Monthly income: ${ann_3c/12:,.0f}")
        print(f"    Return on capital: {ann_3c/capital*100:.1f}%")
        if dd_pct_3c > 25:
            print(f"    WARNING: DD exceeds 25% of account — aggressive")
        elif dd_pct_3c > 15:
            print(f"    CAUTION: DD is {dd_pct_3c:.0f}% of account — moderate risk")
        else:
            print(f"    Risk of ruin: LOW (DD is {dd_pct_3c:.1f}% of account)")
        print()

        # Recommended for personal capital
        max_dd_allowed = capital * 0.15  # 15% max DD rule of thumb
        max_contracts = int(max_dd_allowed / dd_1c) if dd_1c > 0 else 1
        max_contracts = max(1, max_contracts)

        print(f"  RECOMMENDATION FOR ${capital:,.0f}:")
        print(f"    Conservative rule: max DD <= 15% of account = ${max_dd_allowed:,.0f}")
        print(f"    Max contracts at 15% rule: {max_contracts}")
        print(f"    At {max_contracts} contract(s):")
        ann_rec = ann_1c * max_contracts
        dd_rec = dd_1c * max_contracts
        print(f"      Annual income: ${ann_rec:,.0f}")
        print(f"      Monthly income: ${ann_rec/12:,.0f}")
        print(f"      Max DD: ${dd_rec:,.0f} ({dd_rec/capital*100:.1f}% of account)")
        print()

        # Combined strategy: prop firm + personal capital
        print(f"  COMBINED STRATEGY (prop firm + personal capital):")
        print(f"    1. Start on prop firm ($2-3K trailing DD)")
        print(f"       Trade 1 contract. Build profit track record.")
        print(f"       Income: ~${ann_1c/12:,.0f}/month from funded account")
        print(f"    2. Simultaneously trade personal ${capital:,.0f} account")
        print(f"       Trade {max_contracts} contract(s). Keep withdrawals minimal.")
        print(f"       Income: ~${ann_rec/12:,.0f}/month from personal account")
        print(f"    3. Combined monthly income: ~${(ann_1c + ann_rec)/12:,.0f}")
        print(f"    4. If prop firm evaluation fails (cost: $150-300):")
        print(f"       Personal account continues generating income")
        print(f"       Re-attempt prop firm evaluation next month")

        # =====================================================================
        # SECTION 7: TRAILING DD DEEP DIVE
        # =====================================================================
        print(f"\n{'=' * 90}")
        print("TRAILING DD MECHANICS EXPLAINED")
        print(f"{'=' * 90}\n")

        # Compute the actual equity curve day by day for the best 15-slot config
        selected_labels = {f"{s['instrument']}_{s['session']}" for s in ranked[:15]}
        trades_15 = sorted(
            [t for t in all_trades if t["slot_label"] in selected_labels],
            key=lambda t: t["trading_day"]
        )

        # Simulate trailing DD
        by_day = defaultdict(list)
        for t in trades_15:
            by_day[t["trading_day"]].append(t)

        cum = 0.0
        peak = 0.0
        trailing_dd_history = []

        for day in sorted(by_day.keys()):
            day_pnl = sum(t["pnl_dollars"] for t in by_day[day])
            cum += day_pnl
            if cum > peak:
                peak = cum
            trailing_dd = peak - cum
            trailing_dd_history.append({
                "day": day,
                "pnl": day_pnl,
                "cum": cum,
                "peak": peak,
                "trailing_dd": trailing_dd,
            })

        # Find the 5 worst trailing DD moments
        worst_5 = sorted(trailing_dd_history, key=lambda x: -x["trailing_dd"])[:5]

        print("  HOW TRAILING DD WORKS:")
        print("  - Your DD floor starts at (account balance - DD limit)")
        print("  - Every new equity HIGH raises the floor")
        print("  - Floor NEVER goes down, only up")
        print("  - If equity touches the floor -> account blown")
        print()
        print("  Example with $50K Topstep ($2K trailing DD):")
        print("  - Start: balance=$50K, floor=$48K")
        print("  - Day 5: make $500. New peak=$50.5K, floor=$48.5K")
        print("  - Day 10: make $300 more. Peak=$50.8K, floor=$48.8K")
        print("  - Day 15: lose $600. Balance=$50.2K, floor STAYS $48.8K")
        print("  - Gap between balance and floor: $1,400. Still safe.")
        print()
        print("  THE 5 WORST TRAILING DD MOMENTS IN BACKTEST:")
        print(f"  {'Date':>12} {'Peak$':>10} {'Trough$':>10} {'Trailing DD':>12}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12}")
        for w in worst_5:
            trough = w["cum"]
            print(f"  {str(w['day']):>12} ${w['peak']:>8,.0f} ${trough:>8,.0f} "
                  f"${w['trailing_dd']:>10,.0f}")

        # Monthly P&L for a running table
        print(f"\n  MONTHLY P&L (15 slots, 1 contract):")
        monthly_pnl = defaultdict(float)
        for t in trades_15:
            key = (t["trading_day"].year, t["trading_day"].month)
            monthly_pnl[key] += t["pnl_dollars"]

        # Show last 24 months
        sorted_months = sorted(monthly_pnl.keys())
        recent = sorted_months[-24:] if len(sorted_months) > 24 else sorted_months

        print(f"  {'Month':>8} {'P&L':>10} {'Running':>10}")
        print(f"  {'-'*8} {'-'*10} {'-'*10}")
        running = 0
        negative_months = 0
        for ym in recent:
            pnl = monthly_pnl[ym]
            running += pnl
            marker = " <- LOSS" if pnl < 0 else ""
            if pnl < 0:
                negative_months += 1
            print(f"  {ym[0]}-{ym[1]:02d} ${pnl:>8,.0f} ${running:>8,.0f}{marker}")

        total_shown = len(recent)
        print(f"\n  Loss months: {negative_months}/{total_shown} "
              f"({negative_months/total_shown*100:.0f}%)")

        # =====================================================================
        # SECTION 8: THE BOTTOM LINE
        # =====================================================================
        print(f"\n{'=' * 90}")
        print("THE BOTTOM LINE")
        print(f"{'=' * 90}\n")

        # Find the best safe config for $2K and $3K
        for budget_label, budget_val in [("$2,000", 2000), ("$3,000", 3000)]:
            safe = [m for m in results if m["max_dd_dollars"] <= budget_val * 0.80]
            if safe:
                best = max(safe,
                           key=lambda m: m["total_dollars"] / max(m["n_business_days"] / 252, 0.5))
                yrs = best["n_business_days"] / 252
                ann = best["total_dollars"] / yrs
                print(f"  PROP FIRM ({budget_label} trailing DD):")
                print(f"    -> {best['label']}")
                print(f"    -> ${ann/12:,.0f}/month, ${best['max_dd_dollars']:,.0f} max DD")
            else:
                tight = [m for m in results if m["max_dd_dollars"] <= budget_val]
                if tight:
                    best = max(tight,
                               key=lambda m: m["total_dollars"] / max(m["n_business_days"] / 252, 0.5))
                    yrs = best["n_business_days"] / 252
                    ann = best["total_dollars"] / yrs
                    print(f"  PROP FIRM ({budget_label} trailing DD):")
                    print(f"    -> {best['label']} (TIGHT — less than 20% margin)")
                    print(f"    -> ${ann/12:,.0f}/month, ${best['max_dd_dollars']:,.0f} max DD")
                else:
                    print(f"  PROP FIRM ({budget_label} trailing DD):")
                    print(f"    -> No safe configuration. Need smaller system or larger DD budget.")
            print()

        print(f"  PERSONAL CAPITAL (${capital:,.0f}):")
        print(f"    -> 15 slots, {max_contracts} contract(s)")
        print(f"    -> ${ann_rec/12:,.0f}/month, ${dd_rec:,.0f} max DD "
              f"({dd_rec/capital*100:.1f}% of account)")
        print()
        print(f"  COMBINED PLAY:")
        combined_monthly = ann_1c / 12 + ann_rec / 12
        print(f"    -> Prop firm (1 contract) + Personal ({max_contracts} contracts)")
        print(f"    -> ~${combined_monthly:,.0f}/month total")
        print(f"    -> Prop firm costs nothing if you pass. Personal capital = safety net.")
        print()
        print("  HONEST WARNING:")
        print("  These are BACKTESTED numbers. Live trading will be worse because:")
        print("  1. Slippage on entries/exits (~0.02R per trade)")
        print("  2. Missed trades (not at screen, platform issues)")
        print("  3. Emotional interference (skipping after losses)")
        print("  4. Regime changes the backtest hasn't seen yet")
        print("  5. Apply a 30-40% haircut to income projections for realism")

    finally:
        con.close()


if __name__ == "__main__":
    main()
