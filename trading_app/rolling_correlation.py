"""
Rolling and tail-aware correlation metrics for strategy portfolios.

Extends the full-period Pearson correlation in portfolio.py with:
  - Rolling windowed correlation (regime shift detection)
  - Drawdown co-occurrence (tail risk)
  - Co-loss percentage (joint bad days)
  - Summary risk flagging

All functions return plain Python data structures. No DB writes.

Usage:
    python -m trading_app.rolling_correlation --db-path gold.db --instrument MGC
"""

import argparse
import math
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np


# =========================================================================
# PnL loading (from strategy_trade_days + orb_outcomes)
# =========================================================================

def _load_strategy_pnl(
    con: duckdb.DuckDBPyConnection,
    strategy_ids: list[str],
) -> dict[str, dict[date, float]]:
    """
    Load daily PnL for each strategy.

    Returns {strategy_id: {trading_day: pnl_r}}.
    Only includes days with non-NULL pnl_r from orb_outcomes.
    """
    if not strategy_ids:
        return {}

    placeholders = ", ".join(["?"] * len(strategy_ids))

    rows = con.execute(f"""
        SELECT std.strategy_id, std.trading_day, oo.pnl_r
        FROM strategy_trade_days std
        JOIN validated_setups vs ON vs.strategy_id = std.strategy_id
        JOIN orb_outcomes oo
          ON oo.symbol = vs.instrument
          AND oo.orb_label = vs.orb_label
          AND oo.orb_minutes = vs.orb_minutes
          AND oo.entry_model = vs.entry_model
          AND oo.rr_target = vs.rr_target
          AND oo.confirm_bars = vs.confirm_bars
          AND oo.trading_day = std.trading_day
        WHERE std.strategy_id IN ({placeholders})
          AND oo.pnl_r IS NOT NULL
    """, strategy_ids).fetchall()

    result: dict[str, dict[date, float]] = {}
    for sid, td, pnl in rows:
        if isinstance(td, str):
            td = date.fromisoformat(td)
        elif hasattr(td, "date"):
            td = td.date() if callable(td.date) else td.date
        result.setdefault(sid, {})[td] = float(pnl)

    return result


def _pnl_to_arrays(
    pnl_a: dict[date, float],
    pnl_b: dict[date, float],
    day_range: list[date] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[date]]:
    """
    Align two PnL series on shared trading days.

    Returns (array_a, array_b, shared_days) where arrays contain PnL values
    only for days where BOTH strategies traded.
    """
    if day_range is not None:
        shared = [d for d in day_range if d in pnl_a and d in pnl_b]
    else:
        shared = sorted(set(pnl_a.keys()) & set(pnl_b.keys()))

    if not shared:
        return np.array([]), np.array([]), []

    arr_a = np.array([pnl_a[d] for d in shared])
    arr_b = np.array([pnl_b[d] for d in shared])
    return arr_a, arr_b, shared


# =========================================================================
# Function 1: Rolling correlation
# =========================================================================

def compute_rolling_correlation(
    db_path: str,
    strategy_ids: list[str],
    window_days: int = 126,
    min_overlap: int = 60,
    step_days: int = 21,
) -> list[dict]:
    """
    Compute rolling pairwise Pearson correlation across sliding windows.

    Returns list of dicts:
      {"window_end": date, "pair": (id_a, id_b), "correlation": float|None,
       "overlap_days": int}
    """
    if len(strategy_ids) < 2:
        return []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        pnl = _load_strategy_pnl(con, strategy_ids)
    finally:
        con.close()

    # Build unified sorted calendar of all trading days across all strategies
    all_days_set: set[date] = set()
    for daily in pnl.values():
        all_days_set.update(daily.keys())

    if not all_days_set:
        return []

    all_days = sorted(all_days_set)
    pairs = list(combinations(strategy_ids, 2))
    results = []

    # Slide window
    idx = window_days - 1  # first window ends at index window_days-1
    while idx < len(all_days):
        window_end = all_days[idx]
        window_start_idx = idx - window_days + 1
        window_days_list = all_days[window_start_idx:idx + 1]

        for id_a, id_b in pairs:
            pnl_a = pnl.get(id_a, {})
            pnl_b = pnl.get(id_b, {})
            arr_a, arr_b, shared = _pnl_to_arrays(pnl_a, pnl_b, window_days_list)
            overlap = len(shared)

            if overlap < min_overlap:
                corr = None
            else:
                std_a = np.std(arr_a, ddof=1)
                std_b = np.std(arr_b, ddof=1)
                if std_a == 0 or std_b == 0:
                    corr = None
                else:
                    corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
                    if math.isnan(corr):
                        corr = None

            results.append({
                "window_end": window_end,
                "pair": (id_a, id_b),
                "correlation": corr,
                "overlap_days": overlap,
            })

        idx += step_days

    return results


# =========================================================================
# Function 2: Drawdown correlation
# =========================================================================

def compute_drawdown_correlation(
    db_path: str,
    strategy_ids: list[str],
    drawdown_threshold_r: float = -2.0,
) -> dict:
    """
    Compute pairwise drawdown co-occurrence.

    Identifies drawdown periods (cumR > threshold below running peak).
    Returns {(id_a, id_b): {"co_drawdown_pct": float, "a_dd_days": int,
             "b_dd_days": int, "overlap_dd_days": int}}
    """
    if len(strategy_ids) < 2:
        return {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        pnl = _load_strategy_pnl(con, strategy_ids)
    finally:
        con.close()

    # Compute drawdown day sets for each strategy
    dd_days: dict[str, set[date]] = {}
    for sid in strategy_ids:
        daily = pnl.get(sid, {})
        if not daily:
            dd_days[sid] = set()
            continue

        days_sorted = sorted(daily.keys())
        cum_r = 0.0
        peak = 0.0
        in_dd: set[date] = set()

        for d in days_sorted:
            cum_r += daily[d]
            if cum_r > peak:
                peak = cum_r
            if cum_r - peak <= drawdown_threshold_r:
                in_dd.add(d)

        dd_days[sid] = in_dd

    # Pairwise overlap
    results = {}
    for id_a, id_b in combinations(strategy_ids, 2):
        dd_a = dd_days.get(id_a, set())
        dd_b = dd_days.get(id_b, set())
        overlap = dd_a & dd_b
        # Union of drawdown days for denominator
        union = dd_a | dd_b
        co_pct = len(overlap) / len(union) if union else 0.0

        results[(id_a, id_b)] = {
            "co_drawdown_pct": co_pct,
            "a_dd_days": len(dd_a),
            "b_dd_days": len(dd_b),
            "overlap_dd_days": len(overlap),
        }

    return results


# =========================================================================
# Function 3: Co-loss percentage
# =========================================================================

def compute_co_loss_pct(
    db_path: str,
    strategy_ids: list[str],
) -> dict:
    """
    Compute pairwise co-loss percentage.

    On days where BOTH traded, what fraction had both negative PnL?
    Returns {(id_a, id_b): {"co_loss_pct": float, "both_traded_days": int,
             "both_negative_days": int}}
    """
    if len(strategy_ids) < 2:
        return {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        pnl = _load_strategy_pnl(con, strategy_ids)
    finally:
        con.close()

    results = {}
    for id_a, id_b in combinations(strategy_ids, 2):
        pnl_a = pnl.get(id_a, {})
        pnl_b = pnl.get(id_b, {})
        shared_days = set(pnl_a.keys()) & set(pnl_b.keys())
        both_traded = len(shared_days)

        if both_traded == 0:
            results[(id_a, id_b)] = {
                "co_loss_pct": 0.0,
                "both_traded_days": 0,
                "both_negative_days": 0,
            }
            continue

        both_neg = sum(
            1 for d in shared_days
            if pnl_a[d] < 0 and pnl_b[d] < 0
        )
        results[(id_a, id_b)] = {
            "co_loss_pct": both_neg / both_traded,
            "both_traded_days": both_traded,
            "both_negative_days": both_neg,
        }

    return results


# =========================================================================
# Function 4: Summary risk flagging
# =========================================================================

def summarize_correlation_risk(
    rolling: list[dict],
    drawdown: dict,
    co_loss: dict,
    max_acceptable_corr: float = 0.85,
    max_acceptable_co_loss: float = 0.50,
) -> list[dict]:
    """
    Flag risky pairs based on rolling correlation, drawdown overlap, and co-loss.

    Flags pairs where:
      - Peak rolling correlation >= max_acceptable_corr
      - co_loss_pct >= max_acceptable_co_loss
      - co_drawdown_pct >= 0.60

    Returns list of flagged pairs with all metrics attached.
    Advisory output only (no automatic action).
    """
    # Compute peak rolling correlation per pair
    peak_corr: dict[tuple, float] = {}
    for entry in rolling:
        pair = entry["pair"]
        corr = entry["correlation"]
        if corr is not None:
            if pair not in peak_corr or corr > peak_corr[pair]:
                peak_corr[pair] = corr

    # Collect all pairs across all metrics
    all_pairs: set[tuple] = set()
    all_pairs.update(peak_corr.keys())
    all_pairs.update(drawdown.keys())
    all_pairs.update(co_loss.keys())

    flagged = []
    for pair in sorted(all_pairs):
        p_corr = peak_corr.get(pair)
        dd_info = drawdown.get(pair, {})
        cl_info = co_loss.get(pair, {})

        co_dd_pct = dd_info.get("co_drawdown_pct", 0.0)
        co_loss_pct = cl_info.get("co_loss_pct", 0.0)

        is_flagged = (
            (p_corr is not None and p_corr >= max_acceptable_corr)
            or co_loss_pct >= max_acceptable_co_loss
            or co_dd_pct >= 0.60
        )

        if is_flagged:
            flagged.append({
                "pair": pair,
                "peak_rolling_corr": p_corr,
                "co_drawdown_pct": co_dd_pct,
                "a_dd_days": dd_info.get("a_dd_days", 0),
                "b_dd_days": dd_info.get("b_dd_days", 0),
                "overlap_dd_days": dd_info.get("overlap_dd_days", 0),
                "co_loss_pct": co_loss_pct,
                "both_traded_days": cl_info.get("both_traded_days", 0),
                "both_negative_days": cl_info.get("both_negative_days", 0),
            })

    return flagged


# =========================================================================
# CLI entry point
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rolling correlation risk analysis for validated strategy portfolios"
    )
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--window-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=21)
    parser.add_argument("--top-n", type=int, default=20,
                        help="Show top N riskiest pairs")
    args = parser.parse_args()

    db = Path(args.db_path)
    if not db.exists():
        print(f"Error: DB not found: {db}")
        sys.exit(1)

    # Load validated family heads for this instrument
    con = duckdb.connect(str(db), read_only=True)
    try:
        heads = con.execute(
            "SELECT strategy_id FROM validated_setups "
            "WHERE instrument = ? AND is_family_head = TRUE",
            [args.instrument],
        ).fetchall()
    finally:
        con.close()

    strategy_ids = [r[0] for r in heads]
    if len(strategy_ids) < 2:
        print(f"Need >= 2 family heads for correlation analysis, got {len(strategy_ids)}")
        sys.exit(0)

    print(f"Analyzing {len(strategy_ids)} family heads for {args.instrument}")
    print(f"Window: {args.window_days}d, step: {args.step_days}d")
    print()

    # Run all 3 metrics
    print("Computing rolling correlation...")
    rolling = compute_rolling_correlation(
        str(db), strategy_ids,
        window_days=args.window_days,
        step_days=args.step_days,
    )
    print(f"  {len(rolling)} window-pair observations")

    print("Computing drawdown correlation...")
    dd = compute_drawdown_correlation(str(db), strategy_ids)
    print(f"  {len(dd)} pairs")

    print("Computing co-loss percentage...")
    cl = compute_co_loss_pct(str(db), strategy_ids)
    print(f"  {len(cl)} pairs")

    print()
    flagged = summarize_correlation_risk(rolling, dd, cl)
    print(f"Flagged pairs: {len(flagged)}")
    print()

    # Sort by peak rolling correlation descending
    flagged.sort(key=lambda x: x.get("peak_rolling_corr") or 0, reverse=True)

    for entry in flagged[:args.top_n]:
        a, b = entry["pair"]
        print(f"  {a} <-> {b}")
        if entry["peak_rolling_corr"] is not None:
            print(f"    Peak rolling corr: {entry['peak_rolling_corr']:.3f}")
        print(f"    Co-drawdown:      {entry['co_drawdown_pct']:.1%}"
              f" ({entry['overlap_dd_days']}d overlap)")
        print(f"    Co-loss:          {entry['co_loss_pct']:.1%}"
              f" ({entry['both_negative_days']}/{entry['both_traded_days']} shared days)")
        print()
