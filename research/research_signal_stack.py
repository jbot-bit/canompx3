"""Signal Stacking Research — measures combined effect of all confirmed overlay signals.

Loads validated strategy trades via portfolio assembly infrastructure,
then applies each overlay signal incrementally and measures impact.

Overlay signals (applied on top of existing strategy filters):
1. all_narrow AVOID — cross-instrument dead market (all ORBs below median)
2. ATR contraction AVOID — atr_vel_ratio < 0.95 skip
3. Friday high-vol AVOID — MGC 1000 Fridays with high ATR
4. T80 time-stop — use ts_pnl_r instead of pnl_r where available

Usage:
    python research/research_signal_stack.py
    python research/research_signal_stack.py --db-path C:/db/gold.db
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
from trading_app.config import ALL_FILTERS, VolumeFilter, EARLY_EXIT_MINUTES
from trading_app.strategy_discovery import (
    _build_filter_day_sets,
    _compute_relative_volumes,
    _load_daily_features,
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reports"))
from report_edge_portfolio import session_slots

sys.stdout.reconfigure(line_buffering=True)

TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# TRADE LOADING (reuses portfolio assembly pattern)
# =============================================================================

def _get_strategy_params(con, strategy_id):
    row = con.execute("""
        SELECT instrument, orb_label, orb_minutes, entry_model,
               rr_target, confirm_bars, filter_type
        FROM validated_setups
        WHERE strategy_id = ?
    """, [strategy_id]).fetchone()
    if not row:
        return None
    cols = ["instrument", "orb_label", "orb_minutes", "entry_model",
            "rr_target", "confirm_bars", "filter_type"]
    return dict(zip(cols, row))


def load_trades_with_features(con, slots):
    """Load trades for each slot with daily_features columns for signal checking.

    Returns list of trade dicts, each with:
        trading_day, outcome, pnl_r, instrument, session, strategy_id,
        ts_pnl_r, ts_outcome,
        atr_vel_ratio, atr_vel_regime, compression_tier,
        day_of_week, atr_20, is_friday
    """
    all_trades = []

    by_instrument = defaultdict(list)
    for slot in slots:
        by_instrument[slot["instrument"]].append(slot)

    for instrument, inst_slots in by_instrument.items():
        slot_params = {}
        filter_types = set()
        orb_labels = set()
        for slot in inst_slots:
            params = _get_strategy_params(con, slot["head_strategy_id"])
            if params is None:
                continue
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

        # Build features lookup: trading_day -> feature row
        # _load_daily_features returns list of dicts
        feat_lookup = {}
        for row in features:
            feat_lookup[row["trading_day"]] = row

        for slot in inst_slots:
            sid = slot["head_strategy_id"]
            params = slot_params.get(sid)
            if params is None:
                continue

            eligible = filter_days.get(
                (params["filter_type"], params["orb_label"]), set()
            )

            # Check if ts_pnl_r column exists in orb_outcomes
            has_ts = _check_ts_columns(con)

            ts_cols = ", oo.ts_pnl_r, oo.ts_outcome" if has_ts else ""
            rows = con.execute(f"""
                SELECT oo.trading_day, oo.outcome, oo.pnl_r{ts_cols}
                FROM orb_outcomes oo
                WHERE oo.symbol = ?
                  AND oo.orb_label = ?
                  AND oo.orb_minutes = ?
                  AND oo.entry_model = ?
                  AND oo.rr_target = ?
                  AND oo.confirm_bars = ?
                  AND oo.outcome IN ('win', 'loss')
                ORDER BY oo.trading_day
            """, [
                params["instrument"], params["orb_label"], params["orb_minutes"],
                params["entry_model"], params["rr_target"], params["confirm_bars"],
            ]).fetchall()

            for r in rows:
                td = r[0]
                if td not in eligible:
                    continue

                feat = feat_lookup.get(td)
                sess_label = params["orb_label"]

                trade = {
                    "trading_day": td,
                    "outcome": r[1],
                    "pnl_r": r[2],
                    "ts_pnl_r": r[3] if has_ts and len(r) > 3 else None,
                    "ts_outcome": r[4] if has_ts and len(r) > 4 else None,
                    "instrument": instrument,
                    "session": sess_label,
                    "strategy_id": sid,
                    # Features for signal checking
                    "atr_vel_ratio": feat["atr_vel_ratio"] if feat is not None else None,
                    "atr_vel_regime": feat["atr_vel_regime"] if feat is not None else None,
                    "compression_tier": (
                        feat.get(f"orb_{sess_label}_compression_tier")
                        if feat is not None else None
                    ),
                    "day_of_week": td.weekday() if hasattr(td, "weekday") else None,
                    "atr_20": feat["atr_20"] if feat is not None else None,
                    "is_friday": td.weekday() == 4 if hasattr(td, "weekday") else False,
                }
                all_trades.append(trade)

    return all_trades


_ts_check_cache = {}

def _check_ts_columns(con):
    """Check if orb_outcomes has ts_pnl_r column (may not exist in older DBs)."""
    if "checked" in _ts_check_cache:
        return _ts_check_cache["checked"]
    try:
        con.execute("SELECT ts_pnl_r FROM orb_outcomes LIMIT 1")
        _ts_check_cache["checked"] = True
    except Exception:
        _ts_check_cache["checked"] = False
    return _ts_check_cache["checked"]


# =============================================================================
# SIGNAL FILTERS
# =============================================================================

def compute_all_narrow_days(con):
    """Compute days where all instruments have below-median ORB size at 1000.

    Uses expanding window with shift(1) — no lookahead.
    Returns set of trading_days to AVOID.
    """
    # Get ORB sizes at 1000 for each instrument
    rows = con.execute("""
        SELECT trading_day, symbol, orb_1000_size
        FROM daily_features
        WHERE orb_minutes = 5
          AND symbol IN ('MGC', 'MES', 'MNQ')
          AND orb_1000_size IS NOT NULL
        ORDER BY trading_day
    """).fetchall()

    if not rows:
        return set()

    df = pd.DataFrame(rows, columns=["trading_day", "symbol", "orb_1000_size"])

    avoid_days = set()
    # For each instrument, compute expanding median with shift(1)
    medians = {}
    for sym in ["MGC", "MES", "MNQ"]:
        sym_df = df[df["symbol"] == sym].sort_values("trading_day").copy()
        sym_df["expanding_median"] = (
            sym_df["orb_1000_size"].expanding(min_periods=20).median().shift(1)
        )
        sym_df["below_median"] = sym_df["orb_1000_size"] < sym_df["expanding_median"]
        medians[sym] = dict(zip(sym_df["trading_day"], sym_df["below_median"]))

    # A day is "all_narrow" if ALL three instruments are below their expanding median
    all_days = set()
    for sym_days in medians.values():
        all_days.update(sym_days.keys())

    for day in all_days:
        checks = []
        for sym in ["MGC", "MES", "MNQ"]:
            val = medians[sym].get(day)
            if val is None:
                break  # Missing data for this instrument
            checks.append(val)
        if len(checks) == 3 and all(checks):
            avoid_days.add(day)

    return avoid_days


def is_atr_contraction_avoid(trade):
    """ATR contraction AVOID: atr_vel_ratio < 0.95 AND compression_tier in {Neutral, Compressed}."""
    ratio = trade.get("atr_vel_ratio")
    tier = trade.get("compression_tier")
    if ratio is None or tier is None:
        return False
    return ratio < 0.95 and tier in ("Neutral", "Compressed")


def compute_friday_highvol_threshold(con):
    """Compute P25 ATR threshold for Friday high-vol AVOID (MGC 1000 only).

    Returns: ATR threshold (Fridays with ATR > threshold should be avoided).
    """
    rows = con.execute("""
        SELECT atr_20
        FROM daily_features
        WHERE orb_minutes = 5
          AND symbol = 'MGC'
          AND atr_20 IS NOT NULL
        ORDER BY trading_day
    """).fetchall()

    if not rows:
        return None

    atr_values = [r[0] for r in rows]
    return float(np.percentile(atr_values, 75))  # P75 = top 25%


def is_friday_highvol_avoid(trade, atr_threshold):
    """Friday high-vol AVOID: MGC 1000 Fridays with ATR > P25 (=P75 of distribution)."""
    if trade["instrument"] != "MGC" or trade["session"] != "1000":
        return False
    if not trade.get("is_friday", False):
        return False
    atr = trade.get("atr_20")
    if atr is None or atr_threshold is None:
        return False
    return atr > atr_threshold


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(trades, start_date, end_date):
    """Compute portfolio-level metrics from a list of trades."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "total_r": 0, "exp_r": 0, "wr": 0,
                "sharpe_ann": None, "max_dd": 0}

    n_wins = sum(1 for t in trades if t["outcome"] == "win")
    total_r = sum(t.get("effective_pnl_r", t["pnl_r"]) for t in trades)

    # Build daily returns
    daily_r = defaultdict(float)
    for t in trades:
        daily_r[t["trading_day"]] += t.get("effective_pnl_r", t["pnl_r"])

    daily_returns = sorted(daily_r.items())

    # Honest Sharpe (with zero-return days)
    all_bdays = pd.bdate_range(start=start_date, end=end_date)
    return_map = dict(daily_returns)
    full_series = [return_map.get(day.date(), 0.0) for day in all_bdays]

    n_days = len(full_series)
    sharpe_ann = None
    if n_days > 1:
        mean_d = sum(full_series) / n_days
        var = sum((v - mean_d) ** 2 for v in full_series) / (n_days - 1)
        std_d = var ** 0.5
        if std_d > 0:
            sharpe_ann = (mean_d / std_d) * sqrt(TRADING_DAYS_PER_YEAR)

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for day in all_bdays:
        r = return_map.get(day.date(), 0.0)
        cum += r
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    return {
        "n": n,
        "total_r": round(total_r, 1),
        "exp_r": round(total_r / n, 4),
        "wr": round(n_wins / n, 3),
        "sharpe_ann": round(sharpe_ann, 2) if sharpe_ann else None,
        "max_dd": round(max_dd, 1),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Signal stacking research report")
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        # Load slots and trades
        slots = session_slots(db_path)
        if not slots:
            print("No session slots found.")
            return

        print(f"\n{'#' * 80}")
        print(f"#  SIGNAL STACKING REPORT")
        print(f"#  Measuring incremental impact of each overlay signal")
        print(f"{'#' * 80}\n")

        print(f"Database: {db_path}")
        print(f"Slots: {len(slots)}")
        print()

        # Load all trades with features
        print("Loading trades with features...")
        all_trades = load_trades_with_features(con, slots)
        print(f"  Loaded {len(all_trades)} trades across {len(slots)} slots")

        if not all_trades:
            print("No trades found.")
            return

        # Date range
        all_days = [t["trading_day"] for t in all_trades]
        start_date = min(all_days)
        end_date = max(all_days)
        print(f"  Date range: {start_date} to {end_date}")

        # Pre-compute signal data
        print("\nPre-computing signal data...")
        all_narrow_days = compute_all_narrow_days(con)
        print(f"  all_narrow AVOID days: {len(all_narrow_days)}")

        atr_threshold = compute_friday_highvol_threshold(con)
        print(f"  Friday high-vol ATR threshold (P75): {atr_threshold:.1f}" if atr_threshold else "  No ATR data")

        has_ts = _check_ts_columns(con)
        ts_available = sum(1 for t in all_trades if t.get("ts_pnl_r") is not None)
        print(f"  T80 time-stop data: {'available' if has_ts else 'NOT available'} ({ts_available}/{len(all_trades)} trades)")

        # =================================================================
        # LAYER 0: BASELINE (validated strategies with their filters)
        # =================================================================
        print("\n" + "=" * 90)
        print("SIGNAL STACKING — INCREMENTAL IMPACT")
        print("=" * 90)
        print(f"{'Layer':<45} {'N':>6} {'TotalR':>9} {'ExpR':>8} {'WR':>6} {'ShANN':>7} {'MaxDD':>7} {'Delta':>8}")
        print("-" * 90)

        # Set effective_pnl_r = pnl_r for baseline
        for t in all_trades:
            t["effective_pnl_r"] = t["pnl_r"]

        baseline = compute_metrics(all_trades, start_date, end_date)
        _print_layer("0. Baseline (validated filters)", baseline, None)

        # =================================================================
        # LAYER 1: + all_narrow AVOID
        # =================================================================
        layer1_trades = [
            t for t in all_trades
            if t["trading_day"] not in all_narrow_days
        ]
        layer1 = compute_metrics(layer1_trades, start_date, end_date)
        _print_layer("1. + all_narrow AVOID", layer1, baseline)

        # =================================================================
        # LAYER 2: + ATR contraction AVOID
        # =================================================================
        layer2_trades = [
            t for t in layer1_trades
            if not is_atr_contraction_avoid(t)
        ]
        layer2 = compute_metrics(layer2_trades, start_date, end_date)
        _print_layer("2. + ATR contraction AVOID", layer2, layer1)

        # =================================================================
        # LAYER 3: + Friday high-vol AVOID (MGC 1000 only)
        # =================================================================
        layer3_trades = [
            t for t in layer2_trades
            if not is_friday_highvol_avoid(t, atr_threshold)
        ]
        layer3 = compute_metrics(layer3_trades, start_date, end_date)
        _print_layer("3. + Friday high-vol AVOID (MGC 1000)", layer3, layer2)

        # =================================================================
        # LAYER 4: + T80 time-stop
        # =================================================================
        layer4_trades = []
        for t in layer3_trades:
            t_copy = dict(t)
            ts_pnl = t.get("ts_pnl_r")
            ts_out = t.get("ts_outcome")
            if ts_pnl is not None and ts_out is not None:
                t_copy["effective_pnl_r"] = ts_pnl
                # Update outcome for win-rate calc if time-stopped
                if ts_out == "time_stop":
                    t_copy["outcome"] = "loss"  # time-stop exits are losses
            layer4_trades.append(t_copy)

        layer4 = compute_metrics(layer4_trades, start_date, end_date)
        _print_layer("4. + T80 time-stop", layer4, layer3)

        print("-" * 90)
        _print_layer("COMBINED (all signals)", layer4, baseline)

        # =================================================================
        # PER-SIGNAL STANDALONE IMPACT
        # =================================================================
        print("\n" + "=" * 90)
        print("PER-SIGNAL STANDALONE IMPACT (each signal applied alone vs baseline)")
        print("=" * 90)
        print(f"{'Signal':<45} {'Removed':>7} {'N':>6} {'ExpR delta':>11} {'Sh delta':>9}")
        print("-" * 90)

        # all_narrow alone
        an_trades = [t for t in all_trades if t["trading_day"] not in all_narrow_days]
        an = compute_metrics(an_trades, start_date, end_date)
        _print_standalone("all_narrow AVOID", baseline, an, len(all_trades) - len(an_trades))

        # ATR contraction alone
        ac_trades = [t for t in all_trades if not is_atr_contraction_avoid(t)]
        ac = compute_metrics(ac_trades, start_date, end_date)
        _print_standalone("ATR contraction AVOID", baseline, ac, len(all_trades) - len(ac_trades))

        # Friday high-vol alone
        fhv_trades = [t for t in all_trades if not is_friday_highvol_avoid(t, atr_threshold)]
        fhv = compute_metrics(fhv_trades, start_date, end_date)
        _print_standalone("Friday high-vol AVOID (MGC 1000)", baseline, fhv, len(all_trades) - len(fhv_trades))

        # T80 time-stop alone
        ts_trades = []
        for t in all_trades:
            t_copy = dict(t)
            ts_pnl = t.get("ts_pnl_r")
            ts_out = t.get("ts_outcome")
            if ts_pnl is not None and ts_out is not None:
                t_copy["effective_pnl_r"] = ts_pnl
                if ts_out == "time_stop":
                    t_copy["outcome"] = "loss"
            ts_trades.append(t_copy)
        ts = compute_metrics(ts_trades, start_date, end_date)
        _print_standalone("T80 time-stop", baseline, ts, 0)

        # =================================================================
        # PER-YEAR STACKED EFFECT
        # =================================================================
        print("\n" + "=" * 90)
        print("PER-YEAR BREAKDOWN — BASELINE vs STACKED")
        print("=" * 90)
        print(f"{'Year':>6}  {'Base N':>7} {'Base R':>9} {'Base ExpR':>10}  |  {'Stack N':>7} {'Stack R':>9} {'Stack ExpR':>10} {'Delta R':>9}")
        print("-" * 90)

        yearly_base = _by_year(all_trades)
        yearly_stack = _by_year(layer4_trades)

        for year in sorted(set(list(yearly_base.keys()) + list(yearly_stack.keys()))):
            b = yearly_base.get(year, {"n": 0, "total_r": 0, "exp_r": 0})
            s = yearly_stack.get(year, {"n": 0, "total_r": 0, "exp_r": 0})
            delta_r = s["total_r"] - b["total_r"]
            print(
                f"{year:>6}  {b['n']:>7} {b['total_r']:>+8.1f}R {b['exp_r']:>+9.4f}  |  "
                f"{s['n']:>7} {s['total_r']:>+8.1f}R {s['exp_r']:>+9.4f} {delta_r:>+8.1f}R"
            )

        print()

    finally:
        con.close()


def _print_layer(label, metrics, prev):
    delta_sh = ""
    if prev and metrics["sharpe_ann"] and prev["sharpe_ann"]:
        d = metrics["sharpe_ann"] - prev["sharpe_ann"]
        delta_sh = f"{d:>+7.2f}"
    sh = f"{metrics['sharpe_ann']:>6.2f}" if metrics["sharpe_ann"] else "   N/A"
    print(
        f"{label:<45} {metrics['n']:>6} {metrics['total_r']:>+8.1f}R "
        f"{metrics['exp_r']:>+7.4f} {metrics['wr']:>5.1%} {sh} "
        f"{metrics['max_dd']:>6.1f}R {delta_sh:>8}"
    )


def _print_standalone(label, baseline, signal, n_removed):
    exp_delta = signal["exp_r"] - baseline["exp_r"]
    sh_delta = ""
    if signal["sharpe_ann"] and baseline["sharpe_ann"]:
        sh_delta = f"{signal['sharpe_ann'] - baseline['sharpe_ann']:>+8.2f}"
    print(f"{label:<45} {n_removed:>7} {signal['n']:>6} {exp_delta:>+10.4f} {sh_delta:>9}")


def _by_year(trades):
    yearly = defaultdict(lambda: {"n": 0, "wins": 0, "total_r": 0.0})
    for t in trades:
        td = t["trading_day"]
        year = td.year if hasattr(td, "year") else int(str(td)[:4])
        yearly[year]["n"] += 1
        if t["outcome"] == "win":
            yearly[year]["wins"] += 1
        yearly[year]["total_r"] += t.get("effective_pnl_r", t["pnl_r"])

    result = {}
    for year, y in yearly.items():
        result[year] = {
            "n": y["n"],
            "total_r": round(y["total_r"], 1),
            "exp_r": round(y["total_r"] / y["n"], 4) if y["n"] > 0 else 0,
        }
    return result


if __name__ == "__main__":
    main()
