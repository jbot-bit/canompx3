"""Allocator walk-forward backtest — adaptive vs static vs oracle.

VALIDATION ONLY: all parameters are from literature (Carver Ch.11-12,
Chan Ch.7, Pardo Ch.9). Do NOT tune parameters on this backtest.
Tuning = overfitting the allocator itself.

Strategy pool: fixed (today's deployable validated shelf).
Survival bias acknowledged — absolute returns are inflated.
Relative comparison (adaptive vs static) is valid because both use the same pool.

Usage:
    python scripts/tools/backtest_allocator.py
    python scripts/tools/backtest_allocator.py --start 2023-01-01 --end 2025-06-01
    python scripts/tools/backtest_allocator.py --csv backtest_equity.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.cost_model import COST_SPECS
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS
from trading_app.lane_allocator import build_allocation, compute_lane_scores
from trading_app.prop_profiles import ACCOUNT_PROFILES
from trading_app.validated_shelf import deployable_validated_relation

# ── Helpers ────────────────────────────────────────────────────────


def _next_month(d: date) -> date:
    """First day of the next month."""
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _month_seq(start: date, end: date) -> list[date]:
    """Generate first-of-month dates from start to end inclusive."""
    months = []
    d = date(start.year, start.month, 1)
    end_first = date(end.year, end.month, 1)
    while d <= end_first:
        months.append(d)
        d = _next_month(d)
    return months


def _load_strat_meta(con: duckdb.DuckDBPyConnection) -> dict[str, dict]:
    """Load strategy metadata for forward PnL computation.

    LaneScore omits entry_model — we need it for outcome matching.
    """
    shelf_relation = deployable_validated_relation(con)
    rows = con.execute(f"""
        SELECT strategy_id, instrument, orb_label, entry_model,
               rr_target, confirm_bars, filter_type, stop_multiplier
        FROM {shelf_relation}
    """).fetchall()
    meta = {}
    for sid, inst, orb, em, rr, cb, ft, sm in rows:
        meta[sid] = dict(
            instrument=inst,
            orb_label=orb,
            entry_model=em,
            rr_target=rr,
            confirm_bars=cb,
            filter_type=ft,
            stop_multiplier=sm,
        )
    return meta


# ── Batch Forward PnL ──────────────────────────────────────────────


def _batch_forward_pnl(
    con: duckdb.DuckDBPyConnection,
    month_start: date,
    next_start: date,
    strat_meta: dict[str, dict],
) -> dict[str, tuple[float, int, int]]:
    """Compute forward PnL for ALL strategies in one pass.

    Uses 2 SQL queries (outcomes + features) instead of 420.
    SM adjustment and filter logic mirrors _per_month_expr in lane_allocator.py.
    Duplication acknowledged — see canonical source for any future changes.

    Returns: {strategy_id: (total_r, n_trades, n_wins)}
    """
    outcomes = con.execute(
        """
        SELECT symbol, orb_label, entry_model, rr_target, confirm_bars,
               trading_day, pnl_r, mae_r, outcome, entry_price, stop_price
        FROM orb_outcomes
        WHERE orb_minutes = 5 AND outcome IN ('win', 'loss')
          AND trading_day >= ? AND trading_day < ?
        """,
        [month_start, next_start],
    ).fetchall()

    features = con.execute(
        """
        SELECT * FROM daily_features
        WHERE orb_minutes = 5 AND trading_day >= ? AND trading_day < ?
        """,
        [month_start, next_start],
    ).fetchall()

    # Index features by (symbol, trading_day)
    feat_cols = [desc[0] for desc in con.description]
    feat_by_key: dict[tuple, dict] = {}
    for row in features:
        d = dict(zip(feat_cols, row, strict=False))
        feat_by_key[(d["symbol"], d["trading_day"])] = d

    # Index outcomes by matching key for fast strategy lookup
    outcome_by_key: dict[tuple, list] = defaultdict(list)
    for o in outcomes:
        key = (o[0], o[1], o[2], o[3], o[4])  # sym, orb, em, rr, cb
        outcome_by_key[key].append(o)

    result: dict[str, tuple[float, int, int]] = {}

    for sid, info in strat_meta.items():
        key = (
            info["instrument"],
            info["orb_label"],
            info["entry_model"],
            info["rr_target"],
            info["confirm_bars"],
        )
        strat_outcomes = outcome_by_key.get(key, [])
        if not strat_outcomes:
            result[sid] = (0.0, 0, 0)
            continue

        strat_filter = ALL_FILTERS.get(info["filter_type"])
        if strat_filter is None:
            result[sid] = (0.0, 0, 0)
            continue

        sm = info["stop_multiplier"]
        inst = info["instrument"]
        orb = info["orb_label"]
        cost_spec = COST_SPECS.get(inst)

        total_r = 0.0
        n_trades = 0
        n_wins = 0

        for o in strat_outcomes:
            _, _, _, _, _, td, pnl_r, mae_r, _, entry_p, stop_p = o

            # Filter eligibility (same as _per_month_expr)
            feat = feat_by_key.get((inst, td))
            if feat is None:
                continue
            if feat.get(f"orb_{orb}_break_dir") is None:
                continue
            if not strat_filter.matches_row(feat, orb):
                continue

            # SM adjustment (canonical: see _per_month_expr in lane_allocator.py)
            adj_pnl = pnl_r
            if sm != 1.0 and mae_r is not None and entry_p is not None and stop_p is not None and cost_spec is not None:
                risk_pts = abs(entry_p - stop_p)
                if risk_pts > 0:
                    raw_risk_d = risk_pts * cost_spec.point_value
                    risk_d = raw_risk_d + cost_spec.total_friction
                    max_adv_pts = mae_r * risk_d / cost_spec.point_value
                    if max_adv_pts >= sm * risk_pts:
                        adj_pnl = round(-sm, 4)

            total_r += adj_pnl
            n_trades += 1
            if adj_pnl > 0:
                n_wins += 1

        result[sid] = (round(total_r, 4), n_trades, n_wins)

    return result


# ── Core Backtest ──────────────────────────────────────────────────


def run_backtest(
    start: date,
    end: date,
    profile_id: str,
    db_path: Path,
    csv_path: str | None = None,
) -> list[dict]:
    """Run monthly walk-forward backtest.

    For each month M:
      1. SCORE on trailing 12 months (< M_start)
      2. ALLOCATE with hysteresis (prior_allocation from M-1)
      3. FORWARD PnL in [M_start, M+1_start) — non-overlapping with scoring window
    """
    months = _month_seq(start, end)
    profile = ACCOUNT_PROFILES[profile_id]
    max_slots = profile.max_slots
    # DD limit from canonical ACCOUNT_TIERS (not hardcoded)
    from trading_app.prop_profiles import ACCOUNT_TIERS

    tier = ACCOUNT_TIERS.get((profile.firm, profile.account_size))
    max_dd = tier.max_dd if tier else 3000.0

    # Persistent connection for batch forward PnL
    con = duckdb.connect(str(db_path), read_only=True)
    configure_connection(con)
    strat_meta = _load_strat_meta(con)

    print(f"Backtest: {start} to {end} ({len(months)} months)")
    print(f"Profile: {profile_id} | {max_slots} slots | SM={profile.stop_multiplier}")
    print(f"Strategy pool: {len(strat_meta)} validated (fixed)")

    # ── Static baseline: allocate at start, hold forever ──
    static_scores = compute_lane_scores(rebalance_date=start, db_path=db_path)
    static_alloc = build_allocation(
        static_scores,
        max_slots=max_slots,
        max_dd=max_dd,
        allowed_instruments=profile.allowed_instruments,
        allowed_sessions=profile.allowed_sessions,
        stop_multiplier=profile.stop_multiplier,
    )
    static_ids = {s.strategy_id for s in static_alloc}
    print(f"Static baseline: {len(static_alloc)} lanes from {start}")
    for s in static_alloc:
        print(f"  {s.strategy_id} (annual_r={s.annual_r_estimate:.1f})")

    # ── Monthly loop ──
    results: list[dict] = []
    prev_lane_ids: list[str] | None = None

    for i, month_start in enumerate(months):
        next_start = _next_month(month_start)

        # 1. Score and allocate (adaptive)
        scores = compute_lane_scores(rebalance_date=month_start, db_path=db_path)
        allocation = build_allocation(
            scores,
            max_slots=max_slots,
            max_dd=max_dd,
            allowed_instruments=profile.allowed_instruments,
            allowed_sessions=profile.allowed_sessions,
            stop_multiplier=profile.stop_multiplier,
            prior_allocation=prev_lane_ids,
        )
        adaptive_ids = {s.strategy_id for s in allocation}

        # 2. Forward PnL (batch — 2 queries)
        fwd = _batch_forward_pnl(con, month_start, next_start, strat_meta)

        # Adaptive monthly PnL
        adap_pnl = sum(fwd.get(sid, (0, 0, 0))[0] for sid in adaptive_ids)

        # Static monthly PnL
        stat_pnl = sum(fwd.get(sid, (0, 0, 0))[0] for sid in static_ids)

        # Oracle: best strategy per (instrument, session), top N, positive only
        session_best: dict[tuple[str, str], float] = {}
        for sid, (pnl, _n, _w) in fwd.items():
            info = strat_meta[sid]
            if profile.allowed_instruments and info["instrument"] not in profile.allowed_instruments:
                continue
            if profile.allowed_sessions and info["orb_label"] not in profile.allowed_sessions:
                continue
            key = (info["instrument"], info["orb_label"])
            if key not in session_best or pnl > session_best[key]:
                session_best[key] = pnl

        ranked_sessions = sorted(session_best.values(), reverse=True)[:max_slots]
        oracle_pnl = sum(p for p in ranked_sessions if p > 0)

        # Turnover
        turnover = 0
        if prev_lane_ids is not None:
            prev_set = set(prev_lane_ids)
            turnover = len(adaptive_ids - prev_set) + len(prev_set - adaptive_ids)

        results.append(
            dict(
                month=month_start.isoformat(),
                adaptive_pnl=round(adap_pnl, 2),
                static_pnl=round(stat_pnl, 2),
                oracle_pnl=round(oracle_pnl, 2),
                n_lanes=len(allocation),
                turnover=turnover,
            )
        )

        prev_lane_ids = [s.strategy_id for s in allocation]

        # Progress
        if (i + 1) % 12 == 0 or i == len(months) - 1:
            adap_cumul = sum(r["adaptive_pnl"] for r in results)
            print(f"  [{i + 1}/{len(months)}] {month_start} | Adaptive cumul: {adap_cumul:+.1f}R")

    con.close()

    # ── Report ──
    _print_report(results, profile_id, start, end, len(strat_meta), static_alloc)

    if csv_path:
        _write_csv(results, csv_path)

    return results


# ── Report ─────────────────────────────────────────────────────────


def _sharpe(monthly: list[float]) -> float:
    """Annualized Sharpe from monthly returns."""
    if len(monthly) < 2:
        return 0.0
    mean = sum(monthly) / len(monthly)
    var = sum((x - mean) ** 2 for x in monthly) / (len(monthly) - 1)
    std = math.sqrt(var)
    return (mean / std * math.sqrt(12)) if std > 0 else 0.0


def _max_dd(monthly: list[float]) -> float:
    """Max peak-to-trough drawdown from monthly PnL stream."""
    peak = 0.0
    equity = 0.0
    worst = 0.0
    for pnl in monthly:
        equity += pnl
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def _print_report(
    results: list[dict],
    profile_id: str,
    start: date,
    end: date,
    n_strategies: int,
    static_alloc: list,
) -> None:
    """Print human-readable backtest report."""
    n_months = len(results)

    print(f"\n{'=' * 60}")
    print("ALLOCATOR WALK-FORWARD BACKTEST")
    print(f"Period: {start} to {end} ({n_months} months)")
    print(f"Profile: {profile_id}")
    print(f"Pool: {n_strategies} validated strategies (FIXED)")
    print(f"Static: {len(static_alloc)} lanes from {start}, held entire period")
    print("Bias: Survival bias acknowledged — relative comparison valid")
    print("Params: Literature-derived — NOT tuned on this backtest")
    print(f"{'=' * 60}\n")

    # Per-year breakdown
    years: dict[str, dict] = defaultdict(
        lambda: dict(adaptive=0.0, static=0.0, oracle=0.0, turnover=0, months=0),
    )
    for r in results:
        y = r["month"][:4]
        years[y]["adaptive"] += r["adaptive_pnl"]
        years[y]["static"] += r["static_pnl"]
        years[y]["oracle"] += r["oracle_pnl"]
        years[y]["turnover"] += r["turnover"]
        years[y]["months"] += 1

    print(f"{'Year':<6} {'Adaptive':>10} {'Static':>10} {'Oracle':>10} {'Turnover':>10}")
    print("-" * 52)
    for y in sorted(years):
        yd = years[y]
        avg_turn = yd["turnover"] / yd["months"] if yd["months"] > 0 else 0
        print(
            f"{y:<6} {yd['adaptive']:>10.1f} {yd['static']:>10.1f} {yd['oracle']:>10.1f} {avg_turn:>8.1f}/mo",
        )

    total_adap = sum(r["adaptive_pnl"] for r in results)
    total_stat = sum(r["static_pnl"] for r in results)
    total_orac = sum(r["oracle_pnl"] for r in results)
    avg_turnover = sum(r["turnover"] for r in results) / n_months if n_months else 0

    print("-" * 52)
    print(
        f"{'TOTAL':<6} {total_adap:>10.1f} {total_stat:>10.1f} {total_orac:>10.1f} {avg_turnover:>8.1f}/mo",
    )

    # Risk metrics
    m_adap = [r["adaptive_pnl"] for r in results]
    m_stat = [r["static_pnl"] for r in results]
    m_orac = [r["oracle_pnl"] for r in results]

    print(f"\n{'Metric':<12} {'Adaptive':>10} {'Static':>10} {'Oracle':>10}")
    print("-" * 46)
    print(
        f"{'Sharpe':<12} {_sharpe(m_adap):>10.2f} {_sharpe(m_stat):>10.2f} {_sharpe(m_orac):>10.2f}",
    )
    print(
        f"{'Max DD (R)':<12} {_max_dd(m_adap):>10.1f} {_max_dd(m_stat):>10.1f} {_max_dd(m_orac):>10.1f}",
    )

    if total_orac > 0:
        print(f"\nAdaptive captures {total_adap / total_orac * 100:.1f}% of Oracle ceiling")
    if total_stat != 0:
        print(f"Adaptive / Static ratio: {total_adap / total_stat:.2f}x")

    avg_lanes = sum(r["n_lanes"] for r in results) / n_months if n_months else 0
    print(f"Avg lanes deployed: {avg_lanes:.1f}")


def _write_csv(results: list[dict], csv_path: str) -> None:
    """Write monthly equity curves to CSV."""
    adap_cumul = 0.0
    stat_cumul = 0.0
    orac_cumul = 0.0

    with open(csv_path, "w") as f:
        f.write(
            "month,adaptive_pnl,static_pnl,oracle_pnl,adaptive_cumul,static_cumul,oracle_cumul,n_lanes,turnover\n",
        )
        for r in results:
            adap_cumul += r["adaptive_pnl"]
            stat_cumul += r["static_pnl"]
            orac_cumul += r["oracle_pnl"]
            f.write(
                f"{r['month']},{r['adaptive_pnl']},{r['static_pnl']},{r['oracle_pnl']},"
                f"{adap_cumul:.2f},{stat_cumul:.2f},{orac_cumul:.2f},"
                f"{r['n_lanes']},{r['turnover']}\n",
            )
    print(f"\nCSV saved to: {csv_path}")


# ── CLI ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allocator walk-forward backtest — adaptive vs static vs oracle",
    )
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=date(2022, 1, 1),
        help="Start date (default: 2022-01-01)",
    )
    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=date(2025, 12, 1),
        help="End date (default: 2025-12-01)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile ID (default: first active)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV path for equity curves",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path (default: canonical gold.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH

    if args.profile:
        if args.profile not in ACCOUNT_PROFILES:
            print(f"ERROR: Profile '{args.profile}' not found")
            print(f"Available: {', '.join(ACCOUNT_PROFILES.keys())}")
            sys.exit(1)
        profile_id = args.profile
    else:
        for pid, p in ACCOUNT_PROFILES.items():
            if p.active:
                profile_id = pid
                break
        else:
            print("ERROR: No active profiles")
            sys.exit(1)

    run_backtest(args.start, args.end, profile_id, db_path, args.csv)


if __name__ == "__main__":
    main()
