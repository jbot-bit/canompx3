"""Portfolio Engine — institutional-grade portfolio management.

Combines Kelly sizing, regime gating, correlation-aware risk budgeting,
and adaptive drawdown management into a unified decision framework.

Produces:
1. Per-slot Kelly fractions and risk-parity weights
2. Regime-gated allocation (FIT/WATCH/DECAY/STALE)
3. Correlation-adjusted daily risk budget
4. Backtest with institutional sizing vs naive equal-weight
5. Daily decision matrix showing exactly what to trade and how much

The key difference from research_trade_book.py:
- trade_book picks WHICH slots to include (slot selection)
- This engine decides HOW MUCH to risk on each slot each day (sizing)

Usage:
    python research/research_portfolio_engine.py
    python research/research_portfolio_engine.py --profile conservative
    python research/research_portfolio_engine.py --profile aggressive
    python research/research_portfolio_engine.py --dollars-per-r 400
"""

import sys
import argparse
from pathlib import Path
from math import sqrt
from collections import defaultdict
from datetime import timedelta

import numpy as np
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

# Risk profiles
PROFILES = {
    "conservative": {
        "kelly_fraction": 0.25,    # quarter-Kelly
        "max_daily_r": 3.0,        # max R at risk per day
        "dd_circuit_breaker": 6.0,  # halt at 6R portfolio DD
        "dd_scale_threshold": 3.0,  # start scaling at 3R DD
        "dd_scale_floor": 0.25,     # minimum scale in DD
        "watch_weight": 0.0,        # don't trade WATCH slots
        "corr_penalty": 0.5,        # 50% reduction for correlated pairs
        "max_slots_per_day": 6,
    },
    "moderate": {
        "kelly_fraction": 0.40,    # ~half-Kelly
        "max_daily_r": 5.0,
        "dd_circuit_breaker": 10.0,
        "dd_scale_threshold": 5.0,
        "dd_scale_floor": 0.33,
        "watch_weight": 0.5,
        "corr_penalty": 0.4,
        "max_slots_per_day": 10,
    },
    "aggressive": {
        "kelly_fraction": 0.50,    # half-Kelly
        "max_daily_r": 8.0,
        "dd_circuit_breaker": 15.0,
        "dd_scale_threshold": 8.0,
        "dd_scale_floor": 0.50,
        "watch_weight": 0.75,
        "corr_penalty": 0.3,
        "max_slots_per_day": 15,
    },
}


# ---------------------------------------------------------------------------
# KELLY CRITERION
# ---------------------------------------------------------------------------

def compute_kelly(trades):
    """Compute Kelly fraction from trade results.

    Kelly f* = (p*b - q) / b
    where p=win_rate, q=1-p, b=avg_win/avg_loss

    Returns dict with kelly_full, kelly_half, win_rate, payoff_ratio, edge.
    """
    if len(trades) < 30:
        return {"kelly_full": 0, "kelly_half": 0, "win_rate": 0,
                "payoff_ratio": 0, "edge": 0, "n": len(trades)}

    wins = [t["pnl_r"] for t in trades if t["outcome"] == "win"]
    losses = [abs(t["pnl_r"]) for t in trades if t["outcome"] == "loss"]

    if not wins or not losses:
        return {"kelly_full": 0, "kelly_half": 0, "win_rate": 0,
                "payoff_ratio": 0, "edge": 0, "n": len(trades)}

    p = len(wins) / len(trades)
    q = 1 - p
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    b = avg_win / avg_loss if avg_loss > 0 else 0

    kelly = (p * b - q) / b if b > 0 else 0
    kelly = max(0, kelly)  # never bet negative

    return {
        "kelly_full": round(kelly, 4),
        "kelly_half": round(kelly / 2, 4),
        "win_rate": round(p, 4),
        "payoff_ratio": round(b, 4),
        "edge": round(p * b - q, 4),  # expected growth per unit bet
        "n": len(trades),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
    }


def compute_slot_variance(trades):
    """Compute daily return variance for a slot's trade series."""
    if len(trades) < 10:
        return 1.0  # high variance = low weight

    daily_r = defaultdict(float)
    for t in trades:
        daily_r[t["trading_day"]] += t["pnl_r"]

    values = list(daily_r.values())
    if len(values) < 2:
        return 1.0

    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return max(var, 0.001)  # floor to avoid div/0


# ---------------------------------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------------------------------

def compute_slot_correlations(slot_trades_map, min_overlap=30):
    """Compute pairwise daily R correlation between slots.

    slot_trades_map: {slot_label: [trades]}
    Returns dict of (slot_a, slot_b) -> correlation
    """
    # Build daily R series per slot
    slot_daily = {}
    for label, trades in slot_trades_map.items():
        dr = defaultdict(float)
        for t in trades:
            dr[t["trading_day"]] += t["pnl_r"]
        slot_daily[label] = dr

    labels = sorted(slot_daily.keys())
    corr_map = {}

    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            # Find overlapping days
            days_a = set(slot_daily[a].keys())
            days_b = set(slot_daily[b].keys())
            overlap = sorted(days_a & days_b)

            if len(overlap) < min_overlap:
                corr_map[(a, b)] = 0.0
                corr_map[(b, a)] = 0.0
                continue

            va = [slot_daily[a][d] for d in overlap]
            vb = [slot_daily[b][d] for d in overlap]

            # Pearson correlation
            na = np.array(va)
            nb = np.array(vb)
            if na.std() == 0 or nb.std() == 0:
                corr_map[(a, b)] = 0.0
                corr_map[(b, a)] = 0.0
                continue

            r = np.corrcoef(na, nb)[0, 1]
            corr_map[(a, b)] = round(r, 4)
            corr_map[(b, a)] = round(r, 4)

    return corr_map


# ---------------------------------------------------------------------------
# REGIME GATING
# ---------------------------------------------------------------------------

def compute_rolling_fitness(trades, window_days=365):
    """Compute rolling fitness for a slot.

    Returns: FIT, WATCH, DECAY, or STALE with metrics.
    """
    if not trades:
        return "STALE", {"rolling_exp_r": None, "rolling_n": 0}

    cutoff = max(t["trading_day"] for t in trades) - timedelta(days=window_days)
    recent = [t for t in trades if t["trading_day"] >= cutoff]

    if len(recent) < 10:
        return "STALE", {"rolling_exp_r": None, "rolling_n": len(recent)}

    exp_r = sum(t["pnl_r"] for t in recent) / len(recent)
    wins = sum(1 for t in recent if t["outcome"] == "win")
    wr = wins / len(recent)

    if exp_r < -0.01:
        return "DECAY", {"rolling_exp_r": exp_r, "rolling_wr": wr,
                          "rolling_n": len(recent)}

    # Check recent momentum (last 60 days)
    cutoff_60 = max(t["trading_day"] for t in trades) - timedelta(days=60)
    recent_60 = [t for t in trades if t["trading_day"] >= cutoff_60]
    if len(recent_60) >= 5:
        exp_60 = sum(t["pnl_r"] for t in recent_60) / len(recent_60)
        if exp_60 < -0.05:
            return "WATCH", {"rolling_exp_r": exp_r, "rolling_wr": wr,
                              "rolling_n": len(recent), "recent_exp_r": exp_60}

    return "FIT", {"rolling_exp_r": exp_r, "rolling_wr": wr,
                    "rolling_n": len(recent)}


# ---------------------------------------------------------------------------
# PORTFOLIO ENGINE CORE
# ---------------------------------------------------------------------------

def compute_daily_sizing(
    slot_kelly,
    slot_variance,
    slot_fitness,
    slot_corr,
    active_slots_today,
    portfolio_dd,
    profile,
):
    """Compute R-size for each active slot on a given day.

    This is the core institutional sizing algorithm:
    1. Start with Kelly fraction per slot
    2. Apply risk-parity weighting (inverse variance)
    3. Apply regime gate (FIT=full, WATCH=reduced, DECAY/STALE=zero)
    4. Apply correlation penalty for overlapping correlated slots
    5. Apply drawdown scaling
    6. Cap at daily R budget

    Returns: {slot_label: r_size} where r_size is the fraction of 1R to bet.
    """
    if not active_slots_today:
        return {}

    # Step 1: Base sizes from Kelly
    raw_sizes = {}
    for slot in active_slots_today:
        k = slot_kelly.get(slot, {})
        base = k.get("kelly_full", 0) * profile["kelly_fraction"] * 2
        # Kelly fraction param is what fraction of half-Kelly to use
        # So kelly_fraction=0.5 means half-Kelly (standard institutional)
        raw_sizes[slot] = max(base, 0)

    if not any(v > 0 for v in raw_sizes.values()):
        return {}

    # Step 2: Risk parity adjustment (inverse variance weighting)
    total_inv_var = 0
    inv_vars = {}
    for slot in active_slots_today:
        v = slot_variance.get(slot, 1.0)
        iv = 1.0 / v if v > 0 else 1.0
        inv_vars[slot] = iv
        total_inv_var += iv

    rp_weights = {}
    if total_inv_var > 0:
        for slot in active_slots_today:
            rp_weights[slot] = inv_vars[slot] / total_inv_var
    else:
        eq = 1.0 / len(active_slots_today)
        for slot in active_slots_today:
            rp_weights[slot] = eq

    # Blend Kelly and risk-parity (60% Kelly, 40% risk parity)
    blended = {}
    for slot in active_slots_today:
        kelly_size = raw_sizes[slot]
        rp_size = rp_weights[slot] * sum(raw_sizes.values())
        blended[slot] = 0.6 * kelly_size + 0.4 * rp_size

    # Step 3: Regime gating
    gated = {}
    for slot in active_slots_today:
        fitness = slot_fitness.get(slot, ("STALE", {}))
        status = fitness[0] if isinstance(fitness, tuple) else fitness
        if status == "FIT":
            gated[slot] = blended[slot]
        elif status == "WATCH":
            gated[slot] = blended[slot] * profile["watch_weight"]
        else:  # DECAY, STALE
            gated[slot] = 0

    # Remove zeros
    gated = {k: v for k, v in gated.items() if v > 0}
    if not gated:
        return {}

    # Step 4: Correlation penalty
    slots_list = list(gated.keys())
    for i, a in enumerate(slots_list):
        for b in slots_list[i + 1:]:
            corr = abs(slot_corr.get((a, b), 0))
            if corr > 0.3:
                penalty = 1.0 - (corr * profile["corr_penalty"])
                penalty = max(penalty, 0.3)  # floor
                gated[a] *= penalty
                gated[b] *= penalty

    # Step 5: Drawdown scaling
    if portfolio_dd >= profile["dd_circuit_breaker"]:
        return {}  # hard stop

    if portfolio_dd >= profile["dd_scale_threshold"]:
        # Linear scale between threshold and circuit breaker
        range_r = profile["dd_circuit_breaker"] - profile["dd_scale_threshold"]
        if range_r > 0:
            pct_through = (portfolio_dd - profile["dd_scale_threshold"]) / range_r
            scale = max(profile["dd_scale_floor"],
                        1.0 - pct_through * (1.0 - profile["dd_scale_floor"]))
        else:
            scale = profile["dd_scale_floor"]
        gated = {k: v * scale for k, v in gated.items()}

    # Step 6: Normalize to fit daily R budget
    total_r = sum(gated.values())
    if total_r > profile["max_daily_r"]:
        factor = profile["max_daily_r"] / total_r
        gated = {k: v * factor for k, v in gated.items()}

    # Step 7: Max slots per day
    if len(gated) > profile["max_slots_per_day"]:
        top = sorted(gated.items(), key=lambda x: -x[1])[:profile["max_slots_per_day"]]
        gated = dict(top)

    # Floor: minimum useful size is 0.1R
    gated = {k: round(max(v, 0.1), 3) for k, v in gated.items() if v >= 0.05}

    return gated


# ---------------------------------------------------------------------------
# DATA LOADING (reuse trade_book infrastructure)
# ---------------------------------------------------------------------------

def load_all_slot_trades(con, slots):
    """Load trades for all slots, return dict of {slot_label: [trades]}."""
    by_instrument = defaultdict(list)
    for slot in slots:
        by_instrument[slot["instrument"]].append(slot)

    slot_trades = {}

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
            slot_trades[slot_label] = []
            for r in rows:
                if r[0] in eligible:
                    slot_trades[slot_label].append({
                        "trading_day": r[0],
                        "outcome": r[1],
                        "pnl_r": r[2],
                        "instrument": instrument,
                        "session": params["orb_label"],
                        "slot_label": slot_label,
                        "strategy_id": sid,
                    })

    return slot_trades


def get_slot_details(con, slots):
    """Get per-slot validated_setups details."""
    details = {}
    for slot in slots:
        sid = slot["head_strategy_id"]
        row = con.execute("""
            SELECT entry_model, rr_target, confirm_bars, filter_type,
                   max_drawdown_r, expectancy_r, sharpe_ratio, win_rate
            FROM validated_setups WHERE strategy_id = ?
        """, [sid]).fetchone()
        if row:
            f"{slot['instrument']}_{row[3] if row else 'unknown'}"
            slot_label = f"{slot['instrument']}_{slot['session']}"
            details[slot_label] = {
                "strategy_id": sid,
                "entry_model": row[0],
                "rr_target": row[1],
                "confirm_bars": row[2],
                "filter_type": row[3],
                "max_dd": row[4] or 0,
                "exp_r": row[5] or 0,
                "sharpe": row[6] or 0,
                "win_rate": row[7] or 0,
            }
    return details


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------

def run_backtest(slot_trades, slot_kelly, slot_variance, slot_fitness,
                 slot_corr, profile, start_date=None, end_date=None):
    """Run institutional portfolio backtest.

    Returns dict with equity curve, daily decisions, and metrics.
    """
    # Collect all trading days
    all_days = set()
    for trades in slot_trades.values():
        for t in trades:
            all_days.add(t["trading_day"])
    all_days = sorted(all_days)

    if start_date:
        all_days = [d for d in all_days if d >= start_date]
    if end_date:
        all_days = [d for d in all_days if d <= end_date]

    # Index trades by (slot, day)
    trade_index = {}
    for label, trades in slot_trades.items():
        for t in trades:
            key = (label, t["trading_day"])
            if key not in trade_index:
                trade_index[key] = []
            trade_index[key].append(t)

    # Run day by day
    cum_r = 0.0
    peak_r = 0.0
    max_dd = 0.0
    dd_start = None
    max_dd_start = max_dd_end = None

    equity_curve = []
    daily_decisions = []
    daily_pnl = []

    # Also track naive equal-weight for comparison
    naive_cum = 0.0
    naive_peak = 0.0
    naive_max_dd = 0.0

    for day in all_days:
        # Which slots have trades today?
        active_today = []
        for label in slot_trades:
            if (label, day) in trade_index:
                active_today.append(label)

        # Current portfolio DD
        portfolio_dd = peak_r - cum_r

        # Compute sizing
        sizes = compute_daily_sizing(
            slot_kelly, slot_variance, slot_fitness, slot_corr,
            active_today, portfolio_dd, profile,
        )

        # Execute trades
        day_pnl = 0.0
        naive_day_pnl = 0.0
        day_detail = []

        for label in active_today:
            trades_today = trade_index[(label, day)]
            r_size = sizes.get(label, 0)

            for t in trades_today:
                # Institutional sizing
                sized_pnl = t["pnl_r"] * r_size
                day_pnl += sized_pnl

                # Naive: always 1R
                naive_day_pnl += t["pnl_r"]

                day_detail.append({
                    "slot": label,
                    "outcome": t["outcome"],
                    "raw_pnl": t["pnl_r"],
                    "r_size": r_size,
                    "sized_pnl": round(sized_pnl, 4),
                })

        cum_r += day_pnl
        if cum_r > peak_r:
            peak_r = cum_r
            dd_start = day
        drawdown = peak_r - cum_r
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_start = dd_start
            max_dd_end = day

        naive_cum += naive_day_pnl
        if naive_cum > naive_peak:
            naive_peak = naive_cum
        naive_dd = naive_peak - naive_cum
        naive_max_dd = max(naive_max_dd, naive_dd)

        equity_curve.append({
            "date": day,
            "cum_r": round(cum_r, 4),
            "portfolio_dd": round(drawdown, 4),
            "day_pnl": round(day_pnl, 4),
            "naive_cum": round(naive_cum, 4),
            "slots_active": len(active_today),
            "slots_sized": len(sizes),
            "total_r_risked": round(sum(sizes.values()), 3),
        })

        daily_pnl.append(day_pnl)

        if day_detail:
            daily_decisions.append({
                "date": day,
                "dd_level": round(portfolio_dd, 2),
                "trades": day_detail,
            })

    # Compute overall metrics
    n_days = len(daily_pnl)
    if n_days > 1:
        mean_d = sum(daily_pnl) / n_days
        var = sum((v - mean_d) ** 2 for v in daily_pnl) / (n_days - 1)
        std_d = var ** 0.5
        sharpe = (mean_d / std_d) * sqrt(TRADING_DAYS_PER_YEAR) if std_d > 0 else None
    else:
        sharpe = None

    # Naive metrics
    [e["day_pnl"] for e in equity_curve]
    # Recompute naive daily pnl properly
    naive_pnl_list = []
    for day in all_days:
        npnl = 0.0
        for label in slot_trades:
            if (label, day) in trade_index:
                for t in trade_index[(label, day)]:
                    npnl += t["pnl_r"]
        naive_pnl_list.append(npnl)

    naive_sharpe = None
    if len(naive_pnl_list) > 1:
        nm = sum(naive_pnl_list) / len(naive_pnl_list)
        nv = sum((v - nm) ** 2 for v in naive_pnl_list) / (len(naive_pnl_list) - 1)
        ns = nv ** 0.5
        naive_sharpe = (nm / ns) * sqrt(TRADING_DAYS_PER_YEAR) if ns > 0 else None

    # Yearly breakdown
    yearly = defaultdict(lambda: {"n": 0, "pnl": 0.0, "naive_pnl": 0.0})
    for e in equity_curve:
        y = e["date"].year
        yearly[y]["pnl"] += e["day_pnl"]
        n_idx = all_days.index(e["date"])
        yearly[y]["naive_pnl"] += naive_pnl_list[n_idx]
        if e["day_pnl"] != 0:
            yearly[y]["n"] += 1

    return {
        "equity_curve": equity_curve,
        "daily_decisions": daily_decisions,
        "total_r": round(cum_r, 2),
        "max_dd": round(max_dd, 2),
        "max_dd_start": max_dd_start,
        "max_dd_end": max_dd_end,
        "sharpe_ann": round(sharpe, 2) if sharpe else None,
        "naive_total_r": round(naive_cum, 2),
        "naive_max_dd": round(naive_max_dd, 2),
        "naive_sharpe": round(naive_sharpe, 2) if naive_sharpe else None,
        "trading_days": n_days,
        "yearly": dict(yearly),
    }


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def print_slot_analysis(slot_details, slot_kelly, slot_variance, slot_fitness):
    """Print per-slot institutional analysis."""
    print(f"\n{'=' * 110}")
    print("SLOT-LEVEL ANALYSIS")
    print(f"{'=' * 110}")
    print(f"  {'Slot':<25} {'Entry':>5} {'RR':>4} {'Filter':<16} "
          f"{'WR':>5} {'ExpR':>6} {'Kelly':>6} {'HfK':>5} "
          f"{'Var':>6} {'Regime':>6} {'DD':>6}")
    print(f"  {'-'*25} {'-'*5} {'-'*4} {'-'*16} "
          f"{'-'*5} {'-'*6} {'-'*6} {'-'*5} "
          f"{'-'*6} {'-'*6} {'-'*6}")

    for label in sorted(slot_details.keys()):
        d = slot_details[label]
        k = slot_kelly.get(label, {})
        v = slot_variance.get(label, 0)
        f = slot_fitness.get(label, ("?", {}))
        status = f[0] if isinstance(f, tuple) else f

        # Color-code regime status
        status_str = status

        print(f"  {label:<25} {d['entry_model']:>5} {d['rr_target']:>4} "
              f"{d['filter_type']:<16} "
              f"{d['win_rate']:>4.0%} {d['exp_r']:>+5.3f} "
              f"{k.get('kelly_full', 0):>5.3f} {k.get('kelly_half', 0):>4.3f} "
              f"{v:>5.3f} {status_str:>6} {d['max_dd']:>5.1f}R")


def print_correlation_matrix(slot_corr, labels):
    """Print correlation matrix for active slots."""
    print(f"\n{'=' * 110}")
    print("CROSS-SLOT CORRELATION (daily pnl_r)")
    print(f"{'=' * 110}")

    # Header
    print(f"  {'':>25}", end="")
    for lbl in labels:
        print(f" {lbl[-8:]:>8}", end="")
    print()

    for a in labels:
        print(f"  {a:<25}", end="")
        for b in labels:
            if a == b:
                print(f" {'1.00':>8}", end="")
            else:
                corr = slot_corr.get((a, b), 0)
                marker = "*" if abs(corr) > 0.3 else " "
                print(f" {corr:>7.2f}{marker}", end="")
        print()


def print_backtest_results(results, profile_name, profile):
    """Print backtest comparison."""
    r = results

    print(f"\n{'#' * 110}")
    print(f"#  BACKTEST: {profile_name.upper()} PROFILE")
    print(f"#  Kelly frac: {profile['kelly_fraction']:.0%} | "
          f"Daily cap: {profile['max_daily_r']:.0f}R | "
          f"DD breaker: {profile['dd_circuit_breaker']:.0f}R | "
          f"Max slots: {profile['max_slots_per_day']}")
    print(f"{'#' * 110}")

    print(f"\n  {'Metric':<30} {'Institutional':>15} {'Naive (1R each)':>15} {'Delta':>10}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")

    def row(label, inst, naive, fmt=".1f", suffix=""):
        delta = inst - naive if inst is not None and naive is not None else None
        i_str = f"{inst:{fmt}}{suffix}" if inst is not None else "N/A"
        n_str = f"{naive:{fmt}}{suffix}" if naive is not None else "N/A"
        d_str = f"{delta:+{fmt}}{suffix}" if delta is not None else ""
        print(f"  {label:<30} {i_str:>15} {n_str:>15} {d_str:>10}")

    row("Total R", r["total_r"], r["naive_total_r"], ".1f", "R")
    row("Max Drawdown", r["max_dd"], r["naive_max_dd"], ".1f", "R")
    row("Sharpe (ann)", r["sharpe_ann"], r["naive_sharpe"], ".2f")

    if r["total_r"] and r["max_dd"] and r["max_dd"] > 0:
        inst_ret_dd = r["total_r"] / r["max_dd"]
        naive_ret_dd = r["naive_total_r"] / r["naive_max_dd"] if r["naive_max_dd"] > 0 else 0
        row("Return/DD", inst_ret_dd, naive_ret_dd, ".1f", "x")

    row("Trading Days", r["trading_days"], r["trading_days"], ".0f")

    # DD period
    if r["max_dd_start"] and r["max_dd_end"]:
        dur = (r["max_dd_end"] - r["max_dd_start"]).days
        print(f"  {'DD Period':<30} {r['max_dd_start']} to {r['max_dd_end']} ({dur}d)")

    # Yearly
    print(f"\n  {'Year':>6} {'Inst R':>10} {'Naive R':>10} {'Improvement':>12} {'Trades':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    for year in sorted(r["yearly"].keys()):
        y = r["yearly"][year]
        imp = y["pnl"] - y["naive_pnl"]
        print(f"  {year:>6} {y['pnl']:>+9.1f}R {y['naive_pnl']:>+9.1f}R "
              f"{imp:>+11.1f}R {y['n']:>8}")


def print_daily_playbook(results, slot_details, n_days=5):
    """Print the most recent N days as a playbook example."""
    decisions = results["daily_decisions"]
    if not decisions:
        return

    recent = decisions[-n_days:]

    print(f"\n{'=' * 110}")
    print(f"DAILY PLAYBOOK (last {len(recent)} trading days)")
    print(f"{'=' * 110}")

    for day in recent:
        total_r = sum(t["r_size"] for t in day["trades"])
        print(f"\n  {day['date']}  |  DD level: {day['dd_level']:.1f}R  |  "
              f"Total R at risk: {total_r:.2f}R  |  {len(day['trades'])} trades")
        print(f"  {'Slot':<25} {'Size':>6} {'Outcome':>8} {'Raw':>7} {'Sized':>7}")
        print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*7} {'-'*7}")

        for t in sorted(day["trades"], key=lambda x: -x["r_size"]):
            print(f"  {t['slot']:<25} {t['r_size']:>5.2f}R {t['outcome']:>8} "
                  f"{t['raw_pnl']:>+6.2f}R {t['sized_pnl']:>+6.3f}R")


def print_income_projection(results, profile_name, dollars_per_r=None):
    """Print income projections at various $/R levels."""
    r = results

    # Use 2024-2025 average for projection
    r_2024 = r["yearly"].get(2024, {}).get("pnl", 0)
    r_2025 = r["yearly"].get(2025, {}).get("pnl", 0)
    years_pos = sum(1 for y in [2024, 2025] if r["yearly"].get(y, {}).get("pnl", 0) > 0)
    avg_ann = (r_2024 + r_2025) / years_pos if years_pos > 0 else r["total_r"] / 5

    print(f"\n{'=' * 110}")
    print(f"INCOME PROJECTION ({profile_name.upper()} profile)")
    print(f"{'=' * 110}")
    print(f"  Based on 2024-2025 average: {avg_ann:+.1f}R/year")
    print(f"  Max historical DD: {r['max_dd']:.1f}R")
    if r["max_dd"] > 0:
        print(f"  Return/DD ratio: {avg_ann/r['max_dd']:.1f}x")
    print()

    tiers = [100, 200, 400, 750, 1000, 2000]
    if dollars_per_r and dollars_per_r not in tiers:
        tiers.append(dollars_per_r)
        tiers.sort()

    print(f"  {'$/R':>8} {'Annual':>12} {'Monthly':>10} {'Max Loss':>10} "
          f"{'Min Acct':>10} {'Contracts':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for dpr in tiers:
        annual = avg_ann * dpr
        monthly = annual / 12
        dd_dollar = r["max_dd"] * dpr
        acct = dd_dollar * 2.5
        # Approximate contracts (MGC=$10/pt, ~$100-200/R typical)
        contracts = max(1, round(dpr / 100))
        marker = " <--" if dollars_per_r and dpr == dollars_per_r else ""
        print(f"  ${dpr:>7} ${annual:>10,.0f} ${monthly:>8,.0f} "
              f"${dd_dollar:>8,.0f} ${acct:>8,.0f} "
              f"{'~' + str(contracts):>10}{marker}")

    print("\n  Strategy: Start at $100-200/R (1-2 micro contracts).")
    print("  Scale up only after 3+ months profitable live trading.")
    print("  Never risk more than 2% of account per R.")


def print_profile_comparison(all_results):
    """Compare all three profiles side by side."""
    print(f"\n{'#' * 110}")
    print("#  PROFILE COMPARISON")
    print(f"{'#' * 110}")

    print(f"\n  {'Metric':<25}", end="")
    for name in all_results:
        print(f" {name:>18}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in all_results:
        print(f" {'-'*18}", end="")
    print()

    metrics = [
        ("Total R", "total_r", ".1f", "R"),
        ("Max DD", "max_dd", ".1f", "R"),
        ("Sharpe (ann)", "sharpe_ann", ".2f", ""),
        ("Return/DD", None, ".1f", "x"),
    ]

    for label, key, fmt, suffix in metrics:
        print(f"  {label:<25}", end="")
        for name, r in all_results.items():
            if key is None:  # Return/DD
                val = r["total_r"] / r["max_dd"] if r["max_dd"] > 0 else 0
            else:
                val = r[key]
            if val is not None:
                print(f" {val:{fmt}}{suffix:>10}".rjust(18), end="")
            else:
                print(f" {'N/A':>18}", end="")
        print()

    # Best year / worst year
    for label, fn in [("Best Year", max), ("Worst Year", min)]:
        print(f"  {label:<25}", end="")
        for name, r in all_results.items():
            if r["yearly"]:
                best = fn(r["yearly"].values(), key=lambda y: y["pnl"])
                print(f" {best['pnl']:>+10.1f}R".rjust(18), end="")
            else:
                print(f" {'N/A':>18}", end="")
        print()

    # vs naive
    print(f"\n  {'vs Naive (1R each)':<25}", end="")
    for name, r in all_results.items():
        if r["naive_total_r"]:
            delta_r = r["total_r"] - r["naive_total_r"]
            delta_dd = r["max_dd"] - r["naive_max_dd"]
            print(f" R:{delta_r:+.0f} DD:{delta_dd:+.0f}".rjust(18), end="")
        print()

    # =========================================================================
    # RISK-EQUALIZED COMPARISON
    # This is the institutional insight: lever each profile to the SAME DD
    # budget, then compare total R. Higher Sharpe = more R at same risk.
    # =========================================================================
    naive_dd = list(all_results.values())[0]["naive_max_dd"]
    if naive_dd and naive_dd > 0:
        print(f"\n  {'RISK-EQUALIZED':=^78}")
        print(f"  Lever each profile to {naive_dd:.1f}R DD (same as naive)")
        print("  This shows what happens when you run MORE contracts to fill your risk budget\n")
        print(f"  {'Profile':<18} {'Raw R':>8} {'Raw DD':>8} {'Lever':>6} "
              f"{'Equalized R':>12} {'vs Naive':>10} {'Improvement':>12}")
        print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*6} {'-'*12} {'-'*10} {'-'*12}")

        # Naive baseline
        naive_r = list(all_results.values())[0]["naive_total_r"]
        print(f"  {'Naive (1R each)':<18} {naive_r:>+7.0f}R {naive_dd:>7.1f}R "
              f"{'1.0x':>6} {naive_r:>+11.0f}R {naive_r:>+9.0f}R {'baseline':>12}")

        for name, r in all_results.items():
            if r["max_dd"] and r["max_dd"] > 0:
                lever = naive_dd / r["max_dd"]
                eq_r = r["total_r"] * lever
                vs_naive = eq_r - naive_r
                pct = (vs_naive / naive_r * 100) if naive_r > 0 else 0
                print(f"  {name:<18} {r['total_r']:>+7.0f}R {r['max_dd']:>7.1f}R "
                      f"{lever:>5.1f}x {eq_r:>+11.0f}R {eq_r:>+9.0f}R "
                      f"{pct:>+10.0f}%")

        print("\n  Translation: Run more micro contracts to fill the same risk budget.")
        print("  A 5x lever means 5 micro contracts instead of 1.")
        print(f"  Sharpe stays the same. Total R scales linearly. DD stays at {naive_dd:.0f}R.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Institutional Portfolio Engine")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--profile", default="all",
                        choices=["conservative", "moderate", "aggressive", "all"],
                        help="Risk profile (default: run all three)")
    parser.add_argument("--dollars-per-r", type=int, default=None,
                        help="$/R for income projection highlight")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Use top N slots by Sharpe/DD ratio (default: 15)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        # =====================================================================
        # STEP 1: Load slots and rank by Sharpe/DD
        # =====================================================================
        print("Loading session slots...")
        all_slots = session_slots(db_path)
        if not all_slots:
            print("No session slots found.")
            return

        # Get per-slot max DD for ranking
        for slot in all_slots:
            sid = slot["head_strategy_id"]
            row = con.execute(
                "SELECT max_drawdown_r FROM validated_setups WHERE strategy_id = ?",
                [sid],
            ).fetchone()
            slot["max_dd"] = row[0] if row and row[0] else 999
            sh = slot["head_sharpe_ann"] or 0
            slot["sh_dd_ratio"] = sh / slot["max_dd"] if slot["max_dd"] > 0 else 0

        ranked = sorted(all_slots, key=lambda s: -s["sh_dd_ratio"])
        selected = ranked[:args.top_n]

        print(f"Selected top {len(selected)} slots by Sharpe/DD ratio "
              f"(from {len(all_slots)} total)")

        # =====================================================================
        # STEP 2: Load all trade data
        # =====================================================================
        print("Loading trade data...")
        slot_trades = load_all_slot_trades(con, selected)
        total_trades = sum(len(t) for t in slot_trades.values())
        print(f"Loaded {total_trades:,} trades across {len(slot_trades)} slots")

        # =====================================================================
        # STEP 3: Compute Kelly fractions
        # =====================================================================
        print("Computing Kelly fractions...")
        slot_kelly = {}
        for label, trades in slot_trades.items():
            slot_kelly[label] = compute_kelly(trades)

        # =====================================================================
        # STEP 4: Compute slot variances (for risk parity)
        # =====================================================================
        print("Computing slot variances...")
        slot_variance = {}
        for label, trades in slot_trades.items():
            slot_variance[label] = compute_slot_variance(trades)

        # =====================================================================
        # STEP 5: Compute rolling fitness
        # =====================================================================
        print("Computing rolling fitness...")
        slot_fitness = {}
        for label, trades in slot_trades.items():
            slot_fitness[label] = compute_rolling_fitness(trades)

        # =====================================================================
        # STEP 6: Compute correlation matrix
        # =====================================================================
        print("Computing correlations...")
        slot_corr = compute_slot_correlations(slot_trades)

        # =====================================================================
        # STEP 7: Get slot details
        # =====================================================================
        slot_details = get_slot_details(con, selected)

        # =====================================================================
        # PRINT ANALYSIS
        # =====================================================================
        print_slot_analysis(slot_details, slot_kelly, slot_variance, slot_fitness)

        labels = sorted(slot_trades.keys())
        print_correlation_matrix(slot_corr, labels)

        # =====================================================================
        # RUN BACKTESTS
        # =====================================================================
        profiles_to_run = (
            list(PROFILES.keys()) if args.profile == "all"
            else [args.profile]
        )

        all_results = {}
        for profile_name in profiles_to_run:
            profile = PROFILES[profile_name]
            print(f"\nRunning {profile_name} backtest...")
            results = run_backtest(
                slot_trades, slot_kelly, slot_variance, slot_fitness,
                slot_corr, profile,
            )
            all_results[profile_name] = results
            print_backtest_results(results, profile_name, profile)

        # Profile comparison
        if len(all_results) > 1:
            print_profile_comparison(all_results)

        # Show playbook for moderate profile
        playbook_profile = "moderate" if "moderate" in all_results else profiles_to_run[0]
        print_daily_playbook(all_results[playbook_profile], slot_details)

        # Income projection for each profile
        for name, results in all_results.items():
            print_income_projection(results, name, args.dollars_per_r)

    finally:
        con.close()


if __name__ == "__main__":
    main()
