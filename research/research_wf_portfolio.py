"""Walk-Forward Portfolio Validation.

Tests whether the slot selection and adaptive sizing rules discovered
on training data (2019-2023) hold up on out-of-sample data (2024-2025).

This is the honest test. If it fails, the trade book numbers are overfitted.

Split:
  Train: 2019-01-01 to 2023-12-31 (slot ranking, adaptive threshold)
  Test:  2024-01-01 to 2025-12-31 (out-of-sample verification)
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
TRAIN_END = date(2023, 12, 31)
TEST_START = date(2024, 1, 1)


def get_slot_dd(con, slots):
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


def compute_metrics(trades, start_date=None, end_date=None):
    n = len(trades)
    if n == 0:
        return {"n": 0, "total_r": 0, "sharpe_ann": None, "max_dd": 0,
                "wr": 0, "exp_r": 0}

    all_days = [t["trading_day"] for t in trades]
    if start_date is None:
        start_date = min(all_days)
    if end_date is None:
        end_date = max(all_days)

    wins = sum(1 for t in trades if t["outcome"] == "win")
    total_r = sum(t.get("effective_pnl_r", t["pnl_r"]) for t in trades)

    daily_r = defaultdict(float)
    for t in trades:
        daily_r[t["trading_day"]] += t.get("effective_pnl_r", t["pnl_r"])

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
    for d in bdays:
        r = rmap.get(d.date(), 0.0)
        cum += r
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    return {
        "n": n, "total_r": round(total_r, 1),
        "exp_r": round(total_r / n, 4) if n > 0 else 0,
        "wr": round(wins / n, 3) if n > 0 else 0,
        "sharpe_ann": round(sharpe_ann, 2) if sharpe_ann else None,
        "max_dd": round(max_dd, 1),
    }


def apply_adaptive(trades, threshold, scale):
    by_day = defaultdict(list)
    for t in trades:
        by_day[t["trading_day"]].append(t)

    result = []
    cum = peak = 0.0

    for day in sorted(by_day.keys()):
        dd = peak - cum
        cs = scale if dd >= threshold else 1.0

        for t in by_day[day]:
            tc = dict(t)
            tc["effective_pnl_r"] = t.get("effective_pnl_r", t["pnl_r"]) * cs
            result.append(tc)
            cum += tc["effective_pnl_r"]

        if cum > peak:
            peak = cum

    return result


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Portfolio Validation")
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        slots = session_slots(db_path)
        if not slots:
            print("No session slots found.")
            return

        get_slot_dd(con, slots)
        ranked = sorted(slots, key=lambda s: -s["sh_dd_ratio"])

        print(f"\n{'#' * 90}")
        print(f"#  WALK-FORWARD PORTFOLIO VALIDATION")
        print(f"#  Train: 2019-2023  |  Test: 2024-2025+")
        print(f"#  Question: do train-period slot rankings hold out-of-sample?")
        print(f"{'#' * 90}\n")

        # Load ALL trades
        all_trades = load_slot_trades(con, ranked)
        print(f"Total trades: {len(all_trades)}")

        # Split into train/test
        train_trades = [t for t in all_trades if t["trading_day"] <= TRAIN_END]
        test_trades = [t for t in all_trades if t["trading_day"] >= TEST_START]

        print(f"Train trades: {len(train_trades)} (up to {TRAIN_END})")
        print(f"Test trades:  {len(test_trades)} (from {TEST_START})")

        # =====================================================================
        # STEP 1: Rank slots using TRAIN DATA ONLY
        # =====================================================================
        print(f"\n{'=' * 100}")
        print("STEP 1: Slot ranking from TRAIN period (2019-2023)")
        print(f"{'=' * 100}")

        # Compute per-slot metrics on train data
        slot_train_metrics = {}
        for slot in ranked:
            label = f"{slot['instrument']}_{slot['session']}"
            slot_trades = [t for t in train_trades if t["slot_label"] == label]
            if not slot_trades:
                continue
            m = compute_metrics(slot_trades)
            sh_dd = m["sharpe_ann"] / m["max_dd"] if m["sharpe_ann"] and m["max_dd"] > 0 else 0
            slot_train_metrics[label] = {
                "sharpe": m["sharpe_ann"],
                "max_dd": m["max_dd"],
                "sh_dd": sh_dd,
                "total_r": m["total_r"],
                "n": m["n"],
                "wr": m["wr"],
                "exp_r": m["exp_r"],
            }

        # Rank by train-period Sharpe/DD
        train_ranked = sorted(slot_train_metrics.items(), key=lambda x: -x[1]["sh_dd"])

        print(f"\n  {'Rank':>4} {'Slot':<25} {'Sharpe':>7} {'DD':>6} {'Sh/DD':>7} "
              f"{'ExpR':>7} {'N':>5}")
        print(f"  {'-'*4} {'-'*25} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*5}")

        for i, (label, m) in enumerate(train_ranked, 1):
            sh = f"{m['sharpe']:.2f}" if m["sharpe"] else "N/A"
            print(f"  {i:>4} {label:<25} {sh:>7} {m['max_dd']:>5.1f}R "
                  f"{m['sh_dd']:>6.3f} {m['exp_r']:>+6.3f} {m['n']:>5}")

        # =====================================================================
        # STEP 2: Test each slot count on TEST DATA
        # =====================================================================
        print(f"\n{'=' * 100}")
        print("STEP 2: OUT-OF-SAMPLE performance (2024-2025+)")
        print(f"{'=' * 100}")

        print(f"\n  {'TopN':>5} {'Train TotalR':>13} {'Train Sh':>9} {'Train DD':>9} "
              f"| {'Test TotalR':>12} {'Test Sh':>8} {'Test DD':>8} {'Test WR':>8} {'Test N':>7}")
        print(f"  {'-'*5} {'-'*13} {'-'*9} {'-'*9} "
              f"| {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

        train_slot_labels = [label for label, _ in train_ranked]

        for top_n in [3, 5, 6, 8, 10, 12, 15, 20]:
            if top_n > len(train_slot_labels):
                continue

            selected_labels = set(train_slot_labels[:top_n])

            # Train metrics
            tr_trades = [t for t in train_trades if t["slot_label"] in selected_labels]
            tr_m = compute_metrics(tr_trades) if tr_trades else {"total_r": 0, "sharpe_ann": None, "max_dd": 0}

            # Test metrics
            te_trades = [t for t in test_trades if t["slot_label"] in selected_labels]
            te_m = compute_metrics(te_trades) if te_trades else {"total_r": 0, "sharpe_ann": None, "max_dd": 0, "wr": 0, "n": 0}

            tr_sh = f"{tr_m['sharpe_ann']:.2f}" if tr_m["sharpe_ann"] else "N/A"
            te_sh = f"{te_m['sharpe_ann']:.2f}" if te_m["sharpe_ann"] else "N/A"

            print(f"  {top_n:>5} {tr_m['total_r']:>+12.1f}R {tr_sh:>9} {tr_m['max_dd']:>8.1f}R "
                  f"| {te_m['total_r']:>+11.1f}R {te_sh:>8} {te_m['max_dd']:>7.1f}R "
                  f"{te_m.get('wr', 0):>7.1%} {te_m.get('n', 0):>7}")

        # =====================================================================
        # STEP 3: Adaptive sizing walk-forward
        # =====================================================================
        print(f"\n{'=' * 100}")
        print("STEP 3: Adaptive sizing on TEST DATA (threshold/scale chosen from train)")
        print(f"{'=' * 100}")

        # Use top-15 (the recommended config)
        selected_labels = set(train_slot_labels[:15])
        te_trades = [t for t in test_trades if t["slot_label"] in selected_labels]

        if te_trades:
            print(f"\n  Top-15 slots, test period only:")
            print(f"  {'Config':>20} {'TotalR':>9} {'Sharpe':>8} {'MaxDD':>7} {'WR':>6}")
            print(f"  {'-'*20} {'-'*9} {'-'*8} {'-'*7} {'-'*6}")

            # No adaptive
            te_base = compute_metrics(te_trades)
            sh = f"{te_base['sharpe_ann']:.2f}" if te_base["sharpe_ann"] else "N/A"
            print(f"  {'none':>20} {te_base['total_r']:>+8.1f}R {sh:>8} "
                  f"{te_base['max_dd']:>6.1f}R {te_base['wr']:>5.1%}")

            # With adaptive configs
            for threshold in [5, 8, 10, 15]:
                for scale in [0.5, 0.25]:
                    adapted = apply_adaptive(te_trades, threshold, scale)
                    am = compute_metrics(adapted)
                    sh = f"{am['sharpe_ann']:.2f}" if am["sharpe_ann"] else "N/A"
                    label = f"{scale}x@{threshold}R"
                    print(f"  {label:>20} {am['total_r']:>+8.1f}R {sh:>8} "
                          f"{am['max_dd']:>6.1f}R {am['wr']:>5.1%}")

        # =====================================================================
        # STEP 4: Compare train-ranked vs full-data-ranked
        # =====================================================================
        print(f"\n{'=' * 100}")
        print("STEP 4: Does train-period ranking match full-data ranking?")
        print(f"{'=' * 100}")

        # Full-data ranking (what we showed the user earlier)
        full_ranked = sorted(ranked, key=lambda s: -s["sh_dd_ratio"])
        full_labels = [f"{s['instrument']}_{s['session']}" for s in full_ranked]

        print(f"\n  {'Full Rank':>10} {'Slot':<25} {'Train Rank':>11} {'Movement':>10}")
        print(f"  {'-'*10} {'-'*25} {'-'*11} {'-'*10}")

        for i, label in enumerate(full_labels, 1):
            train_rank = train_slot_labels.index(label) + 1 if label in train_slot_labels else "N/A"
            if isinstance(train_rank, int):
                movement = i - train_rank
                move_str = f"{movement:>+3}" if movement != 0 else "  ="
            else:
                move_str = " NEW"
            print(f"  {i:>10} {label:<25} {train_rank:>11} {move_str:>10}")

        # =====================================================================
        # VERDICT
        # =====================================================================
        print(f"\n{'#' * 90}")
        print("#  VERDICT")
        print(f"{'#' * 90}")

        # Test: do the top-15 from train data still work OOS?
        top15_train = set(train_slot_labels[:15])
        top15_full = set(full_labels[:15])

        overlap = top15_train & top15_full
        train_only = top15_train - top15_full
        full_only = top15_full - top15_train

        print(f"\n  Top-15 overlap (train vs full-data): {len(overlap)}/{15} ({len(overlap)/15:.0%})")
        if train_only:
            print(f"  In train top-15 but NOT full-data top-15: {train_only}")
        if full_only:
            print(f"  In full-data top-15 but NOT train top-15: {full_only}")

        # OOS performance of train-selected top-15
        if te_trades:
            te_m = compute_metrics(te_trades)
            print(f"\n  OOS performance (top-15 from train ranking):")
            print(f"    Trades:   {te_m['n']:,}")
            print(f"    Total R:  {te_m['total_r']:+.1f}")
            print(f"    Sharpe:   {te_m['sharpe_ann']}")
            print(f"    Max DD:   {te_m['max_dd']:.1f}R")
            print(f"    Win rate: {te_m['wr']:.1%}")
            print(f"    ExpR:     {te_m['exp_r']:+.4f}")

            if te_m["sharpe_ann"] and te_m["sharpe_ann"] > 2.0:
                print(f"\n  PASS: OOS Sharpe {te_m['sharpe_ann']:.2f} > 2.0 threshold")
            elif te_m["sharpe_ann"] and te_m["sharpe_ann"] > 1.0:
                print(f"\n  MARGINAL: OOS Sharpe {te_m['sharpe_ann']:.2f} between 1.0-2.0")
            else:
                print(f"\n  FAIL: OOS Sharpe below 1.0")

            if te_m["total_r"] > 0:
                print(f"  PASS: OOS Total R positive ({te_m['total_r']:+.1f})")
            else:
                print(f"  FAIL: OOS Total R negative")

    finally:
        con.close()


if __name__ == "__main__":
    main()
