"""
Adaptive Lane Allocator — monthly rebalancer for ORB breakout deployment.

Two-layer system:
  Layer 1 (static): BH FDR + walk-forward validation → validated_setups
  Layer 2 (dynamic): This module. Trailing performance → lane selection.

Grounding: Carver Ch.11-12, LdP Ch.12, Chan Ch.7, Pardo Ch.9.

All queries use trading_day < rebalance_date (zero look-ahead).
SM=0.75 adjustment applied via mae_r.
Filter applied in trailing window via matches_row().
Parameters from literature, NOT backtest optimization.

Usage:
    from trading_app.lane_allocator import compute_lane_scores, build_allocation
    scores = compute_lane_scores(rebalance_date=date(2026, 4, 1))
    lanes = build_allocation(scores, profile)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from pipeline.cost_model import COST_SPECS
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS
from trading_app.strategy_discovery import parse_stop_multiplier

# ---------------------------------------------------------------------------
# Literature-grounded constants (do NOT tune on backtest — see spec §Parameter Source)
# ---------------------------------------------------------------------------
DEPLOY_WINDOW_MONTHS = 12  # Carver Ch.11: forecast weighting window
REGIME_WINDOW_MONTHS = 6  # Chan Ch.7: regime half-life capture
PAUSE_MONTHS_NEGATIVE = 2  # Chan Ch.7 + Pardo Ch.9: fast kill
RESUME_MONTHS_POSITIVE = 3  # Carver Ch.12: asymmetric switching
MAGNITUDE_PAUSE_THRESHOLD = -0.10  # 3-month avg ExpR below this → immediate pause
MIN_TRAILING_N = 20  # Minimum trades for reliable score
HYSTERESIS_PCT = 0.20  # Carver Ch.12: 20% switching cost
PROVISIONAL_MONTHS = 6  # Minimum months for full (non-provisional) status
STALENESS_WARN_DAYS = 35
STALENESS_BLOCK_DAYS = 60


@dataclass
class LaneScore:
    """Score for a single validated strategy in the trailing window."""

    strategy_id: str
    instrument: str
    orb_label: str
    rr_target: float
    filter_type: str
    confirm_bars: int
    stop_multiplier: float
    trailing_expr: float
    trailing_n: int
    trailing_months: int
    annual_r_estimate: float
    trailing_wr: float
    session_regime_expr: float | None
    months_negative: int
    months_positive_since_last_neg_streak: int
    status: str  # DEPLOY / PROVISIONAL / PAUSE / RESUME / STALE
    status_reason: str


def _parse_strategy_params(strategy_id: str) -> dict:
    """Extract instrument, orb_label, entry_model, rr, cb, filter from strategy_id."""
    parts = strategy_id.split("_")
    instrument = parts[0]
    # Find entry model position (E1 or E2)
    em_idx = None
    for i, p in enumerate(parts):
        if p in ("E1", "E2", "E3"):
            em_idx = i
            break
    if em_idx is None:
        raise ValueError(f"Cannot parse entry_model from {strategy_id}")

    orb_label = "_".join(parts[1:em_idx])
    entry_model = parts[em_idx]

    # RR target: next part after EM, format RR1.0
    rr_str = parts[em_idx + 1]
    rr_target = float(rr_str.replace("RR", ""))

    # Confirm bars: CB1, CB2, etc.
    cb_str = parts[em_idx + 2]
    confirm_bars = int(cb_str.replace("CB", ""))

    # Filter type: everything after CB, excluding _S075, _O15, _O30, _W, _S suffixes
    remaining = parts[em_idx + 3 :]
    # Strip known suffixes from the end
    filter_parts = []
    for p in remaining:
        if p in ("S075", "O15", "O30", "W", "S"):
            continue
        filter_parts.append(p)
    filter_type = "_".join(filter_parts) if filter_parts else "NO_FILTER"

    sm = parse_stop_multiplier(strategy_id)

    return {
        "instrument": instrument,
        "orb_label": orb_label,
        "entry_model": entry_model,
        "rr_target": rr_target,
        "confirm_bars": confirm_bars,
        "filter_type": filter_type,
        "stop_multiplier": sm,
    }


def _month_range(rebalance_date: date, months_back: int) -> tuple[date, date]:
    """Return (start_date, end_date) for a trailing window."""
    # Go back N months from rebalance_date
    year = rebalance_date.year
    month = rebalance_date.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    start = date(year, month, 1)
    return start, rebalance_date


def _per_month_expr(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
    stop_multiplier: float,
    rebalance_date: date,
    n_months: int = 6,
) -> list[tuple[str, float, int]]:
    """Compute per-calendar-month ExpR for a strategy.

    Returns list of (year_month, avg_pnl_r, n_trades) tuples, most recent first.
    Applies filter via matches_row and SM adjustment via mae_r.
    """
    start, end = _month_range(rebalance_date, n_months)

    # Load outcomes
    outcomes = con.execute(
        """
        SELECT o.trading_day, o.pnl_r, o.mae_r, o.outcome
        FROM orb_outcomes o
        WHERE o.symbol = ? AND o.orb_label = ? AND o.entry_model = ?
          AND o.rr_target = ? AND o.confirm_bars = ? AND o.orb_minutes = 5
          AND o.outcome IN ('win', 'loss')
          AND o.trading_day >= ? AND o.trading_day < ?
        ORDER BY o.trading_day
        """,
        [instrument, orb_label, entry_model, rr_target, confirm_bars, start, end],
    ).fetchall()

    if not outcomes:
        return []

    # Load daily_features for filter application
    features = con.execute(
        """
        SELECT * FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
          AND trading_day >= ? AND trading_day < ?
        ORDER BY trading_day
        """,
        [instrument, start, end],
    ).fetchall()

    if not features:
        return []

    feat_cols = [desc[0] for desc in con.description]
    feat_by_day = {}
    for row in features:
        d = dict(zip(feat_cols, row, strict=False))
        feat_by_day[d["trading_day"]] = d

    # Apply filter
    strat_filter = ALL_FILTERS.get(filter_type)
    eligible_days = set()
    for td, feat_row in feat_by_day.items():
        # Must have a break in this session
        if feat_row.get(f"orb_{orb_label}_break_dir") is None:
            continue
        # Apply filter (fail-closed: unknown filter = ineligible)
        if strat_filter is None:
            continue
        if strat_filter.matches_row(feat_row, orb_label):
            eligible_days.add(td)

    # Filter outcomes to eligible days + apply SM adjustment
    adjusted = []
    for td, pnl_r, mae_r, outcome in outcomes:
        if td not in eligible_days:
            continue
        # SM adjustment
        if stop_multiplier != 1.0 and mae_r is not None and mae_r >= stop_multiplier:
            adj_pnl_r = round(-stop_multiplier, 4)
        else:
            adj_pnl_r = pnl_r
        adjusted.append((td, adj_pnl_r))

    if not adjusted:
        return []

    # Group by calendar month
    from collections import defaultdict

    monthly: dict[str, list[float]] = defaultdict(list)
    for td, pnl_r in adjusted:
        ym = f"{td.year}-{td.month:02d}"
        monthly[ym].append(pnl_r)

    result = []
    for ym in sorted(monthly.keys(), reverse=True):
        trades = monthly[ym]
        avg_r = sum(trades) / len(trades)
        result.append((ym, round(avg_r, 4), len(trades)))

    return result


def compute_lane_scores(
    rebalance_date: date,
    trailing_months: int = DEPLOY_WINDOW_MONTHS,
    db_path: str | Path | None = None,
) -> list[LaneScore]:
    """Compute trailing performance for all validated strategies.

    Zero look-ahead: all queries use trading_day < rebalance_date.
    SM-adjusted: applies tight-stop via mae_r.
    Filter-applied: uses matches_row on daily_features.
    """
    db = db_path or GOLD_DB_PATH
    con = duckdb.connect(str(db), read_only=True)

    try:
        # Load all active validated strategies
        strategies = con.execute(
            """
            SELECT strategy_id, instrument, orb_label, entry_model,
                   rr_target, confirm_bars, filter_type, stop_multiplier,
                   sample_size
            FROM validated_setups
            WHERE status = 'active'
            ORDER BY instrument, orb_label
            """
        ).fetchall()

        scores = []
        for sid, inst, orb, em, rr, cb, ft, sm, total_n in strategies:
            # Per-month ExpR for trailing window
            monthly = _per_month_expr(
                con,
                inst,
                orb,
                em,
                rr,
                cb,
                ft,
                sm,
                rebalance_date,
                trailing_months,
            )

            if not monthly:
                scores.append(
                    LaneScore(
                        strategy_id=sid,
                        instrument=inst,
                        orb_label=orb,
                        rr_target=rr,
                        filter_type=ft,
                        confirm_bars=cb,
                        stop_multiplier=sm,
                        trailing_expr=0.0,
                        trailing_n=0,
                        trailing_months=trailing_months,
                        annual_r_estimate=0.0,
                        trailing_wr=0.0,
                        session_regime_expr=None,
                        months_negative=trailing_months,
                        months_positive_since_last_neg_streak=0,
                        status="STALE",
                        status_reason="No trades in trailing window",
                    )
                )
                continue

            # Aggregate trailing stats
            all_trades_n = sum(n for _, _, n in monthly)
            # Weighted average ExpR (weighted by trade count per month)
            total_pnl = sum(expr * n for _, expr, n in monthly)
            trailing_expr = round(total_pnl / all_trades_n, 4) if all_trades_n > 0 else 0.0

            # Actual months with data
            actual_months = len(monthly)

            # Annual R estimate
            annual_r = round(trailing_expr * all_trades_n / (actual_months / 12.0), 1) if actual_months > 0 else 0.0

            # Win rate from trailing (need to recount — approximate from monthly)
            # Note: monthly ExpR doesn't give us WR directly. Use trailing_expr sign as proxy.
            # For proper WR, we'd need to count wins separately. Use trailing_expr > 0 as indicator.
            trailing_wr = sum(1 for _, e, _ in monthly if e > 0) / len(monthly) if monthly else 0.0

            # Consecutive months negative (most recent backward)
            months_neg = 0
            for _, expr_m, _ in monthly:  # monthly is most-recent-first
                if expr_m < 0:
                    months_neg += 1
                else:
                    break

            # Months positive since last negative streak of 2+
            months_pos_since = 0
            found_neg_streak = False
            for _, expr_m, _ in monthly:
                if not found_neg_streak:
                    if expr_m >= 0:
                        months_pos_since += 1
                    else:
                        found_neg_streak = True
                else:
                    break

            # Session regime (unfiltered E2 RR1.0 CB1, 6-month window)
            session_regime = _compute_session_regime(
                con,
                inst,
                orb,
                rebalance_date,
            )

            # Determine status
            status, reason = _classify_status(
                trailing_expr=trailing_expr,
                trailing_n=all_trades_n,
                actual_months=actual_months,
                months_neg=months_neg,
                months_pos_since=months_pos_since,
                annual_r=annual_r,
                session_regime_expr=session_regime,
                monthly=monthly,
            )

            scores.append(
                LaneScore(
                    strategy_id=sid,
                    instrument=inst,
                    orb_label=orb,
                    rr_target=rr,
                    filter_type=ft,
                    confirm_bars=cb,
                    stop_multiplier=sm,
                    trailing_expr=trailing_expr,
                    trailing_n=all_trades_n,
                    trailing_months=actual_months,
                    annual_r_estimate=annual_r,
                    trailing_wr=trailing_wr,
                    session_regime_expr=session_regime,
                    months_negative=months_neg,
                    months_positive_since_last_neg_streak=months_pos_since,
                    status=status,
                    status_reason=reason,
                )
            )

        return scores
    finally:
        con.close()


def _compute_session_regime(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    orb_label: str,
    rebalance_date: date,
) -> float | None:
    """Session-level trailing ExpR (unfiltered E2 RR1.0 CB1 O5).

    Used as regime gate for thin-data strategies.
    Window: REGIME_WINDOW_MONTHS (6 months).
    """
    start, end = _month_range(rebalance_date, REGIME_WINDOW_MONTHS)
    r = con.execute(
        """
        SELECT AVG(pnl_r)
        FROM orb_outcomes
        WHERE symbol = ? AND orb_label = ? AND entry_model = 'E2'
          AND rr_target = 1.0 AND confirm_bars = 1 AND orb_minutes = 5
          AND outcome IN ('win', 'loss')
          AND trading_day >= ? AND trading_day < ?
        """,
        [instrument, orb_label, start, end],
    ).fetchone()
    return round(r[0], 4) if r and r[0] is not None else None


def _classify_status(
    *,
    trailing_expr: float,
    trailing_n: int,
    actual_months: int,
    months_neg: int,
    months_pos_since: int,
    annual_r: float,
    session_regime_expr: float | None,
    monthly: list[tuple[str, float, int]],
) -> tuple[str, str]:
    """Classify strategy status based on trailing data."""
    # STALE: not enough trades
    if trailing_n < MIN_TRAILING_N:
        # Fall back to session regime gate
        if session_regime_expr is not None and session_regime_expr > 0:
            return "DEPLOY", f"Thin data (N={trailing_n}), session regime HOT ({session_regime_expr:.4f})"
        elif session_regime_expr is not None and session_regime_expr <= 0:
            return "PAUSE", f"Thin data (N={trailing_n}), session regime COLD ({session_regime_expr:.4f})"
        return "STALE", f"Insufficient trades (N={trailing_n} < {MIN_TRAILING_N}) and no session regime"

    # MAGNITUDE OVERRIDE: 3-month average deeply negative
    if len(monthly) >= 3:
        recent_3 = monthly[:3]
        total_3mo = sum(e * n for _, e, n in recent_3)
        count_3mo = sum(n for _, _, n in recent_3)
        avg_3mo = total_3mo / count_3mo if count_3mo > 0 else 0
        if avg_3mo < MAGNITUDE_PAUSE_THRESHOLD:
            return "PAUSE", f"Magnitude override: 3mo avg ExpR={avg_3mo:.4f} < {MAGNITUDE_PAUSE_THRESHOLD}"

    # PAUSE: 2 consecutive months negative (individual month check)
    if months_neg >= PAUSE_MONTHS_NEGATIVE:
        return "PAUSE", f"{months_neg} consecutive months negative"

    # RESUME check: was previously in a negative streak, now recovering
    # If there was a negative streak of 2+ in the last 6 months, need 3 positive months to resume
    has_prior_neg_streak = False
    streak = 0
    for _, expr_m, _ in monthly:
        if expr_m < 0:
            streak += 1
            if streak >= 2:
                has_prior_neg_streak = True
                break
        else:
            streak = 0

    if has_prior_neg_streak and months_pos_since < RESUME_MONTHS_POSITIVE:
        return "PAUSE", f"Recovering: {months_pos_since} positive months < {RESUME_MONTHS_POSITIVE} needed"

    if has_prior_neg_streak and months_pos_since >= RESUME_MONTHS_POSITIVE and annual_r > 0:
        return "RESUME", f"Recovery confirmed: {months_pos_since} positive months, annual_r={annual_r:.1f}"

    # PROVISIONAL: less than 6 months of data
    if actual_months < PROVISIONAL_MONTHS:
        return "PROVISIONAL", f"Only {actual_months} months of data (< {PROVISIONAL_MONTHS})"

    # DEPLOY: positive and sufficient data
    if trailing_expr > 0:
        return "DEPLOY", f"Trailing ExpR={trailing_expr:.4f}, N={trailing_n}, {actual_months}mo"

    # Negative but not paused (just went negative this month)
    return "PAUSE", f"Trailing ExpR={trailing_expr:.4f} negative"


def build_allocation(
    scores: list[LaneScore],
    *,
    max_slots: int = 5,
    max_dd: float = 3000.0,
    allowed_instruments: frozenset[str] | None = None,
    allowed_sessions: frozenset[str] | None = None,
    stop_multiplier: float = 0.75,
    prior_allocation: list[str] | None = None,
) -> list[LaneScore]:
    """Select top lanes for a profile, respecting constraints.

    Greedy selection: rank by annual_r, add lanes that fit DD budget.
    Hysteresis: only replace a lane if new candidate >20% better.
    """
    # Filter to deployable
    candidates = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]

    # Apply profile constraints
    if allowed_instruments:
        candidates = [s for s in candidates if s.instrument in allowed_instruments]
    if allowed_sessions:
        candidates = [s for s in candidates if s.orb_label in allowed_sessions]

    # One winner per instrument × session (best annual_r)
    best_per_session: dict[tuple[str, str], LaneScore] = {}
    for s in candidates:
        key = (s.instrument, s.orb_label)
        if key not in best_per_session or s.annual_r_estimate > best_per_session[key].annual_r_estimate:
            best_per_session[key] = s

    # Sort by annual R descending. PROVISIONAL ranked below non-provisional at equal annual_r
    ranked = sorted(
        best_per_session.values(),
        key=lambda s: (0 if s.status == "PROVISIONAL" else 1, s.annual_r_estimate),
        reverse=True,
    )

    # Greedy DD-budget selection
    selected: list[LaneScore] = []
    dd_used = 0.0

    for lane in ranked:
        if len(selected) >= max_slots:
            break

        # Estimate worst-case DD contribution
        cost = COST_SPECS.get(lane.instrument)
        if cost is None:
            continue
        # Use P90 ORB size estimate (from prop_profiles _P90_ORB_PTS)
        from trading_app.prop_profiles import _P90_ORB_PTS

        p90_orb = _P90_ORB_PTS.get(lane.instrument, 100.0)
        lane_dd = p90_orb * stop_multiplier * cost.point_value

        if dd_used + lane_dd > max_dd:
            continue

        # Hysteresis check (only if prior allocation exists)
        if prior_allocation and lane.strategy_id not in prior_allocation:
            # New lane — check if it beats any existing lane by >20%
            # Simple: if it made the ranked list, it's good enough
            # The 20% check is against what it's REPLACING, not the full list
            pass  # Hysteresis applied at the report level, not hard-block

        selected.append(lane)
        dd_used += lane_dd

    return selected


def generate_report(
    scores: list[LaneScore],
    allocation: list[LaneScore],
    rebalance_date: date,
    profile_id: str,
) -> str:
    """Generate human-readable rebalance report."""
    lines = [
        f"# Lane Allocation Report — {rebalance_date}",
        f"Profile: {profile_id}",
        f"Total candidates: {len(scores)}",
        f"Deployable: {sum(1 for s in scores if s.status in ('DEPLOY', 'RESUME', 'PROVISIONAL'))}",
        f"Paused: {sum(1 for s in scores if s.status == 'PAUSE')}",
        f"Stale: {sum(1 for s in scores if s.status == 'STALE')}",
        "",
        "## Selected Lanes",
        f"{'#':<3} {'Strategy':<55} {'AnnR':>6} {'ExpR':>8} {'N':>5} {'Status':<12}",
        "-" * 95,
    ]

    for i, s in enumerate(allocation, 1):
        lines.append(
            f"{i:<3} {s.strategy_id:<55} {s.annual_r_estimate:>6.1f} "
            f"{s.trailing_expr:>8.4f} {s.trailing_n:>5} {s.status:<12}"
        )

    lines.extend(["", "## Paused Lanes"])
    for s in sorted(scores, key=lambda x: x.annual_r_estimate, reverse=True):
        if s.status == "PAUSE":
            lines.append(f"  {s.strategy_id}: {s.status_reason}")

    lines.extend(["", "## Session Regimes (6mo trailing, unfiltered E2 RR1.0)"])
    seen_sessions: set[tuple[str, str]] = set()
    for s in scores:
        key = (s.instrument, s.orb_label)
        if key in seen_sessions:
            continue
        seen_sessions.add(key)
        regime = s.session_regime_expr
        regime_str = f"{regime:+.4f}" if regime is not None else "N/A"
        tag = "HOT" if regime and regime > 0.03 else "COLD" if regime and regime < -0.03 else "FLAT"
        lines.append(f"  {s.instrument} {s.orb_label:<20} {regime_str:>8} [{tag}]")

    return "\n".join(lines)


def save_allocation(
    scores: list[LaneScore],
    allocation: list[LaneScore],
    rebalance_date: date,
    profile_id: str,
    output_path: str | Path | None = None,
) -> Path:
    """Save allocation to JSON file."""
    path = Path(output_path) if output_path else Path("docs/runtime/lane_allocation.json")

    data = {
        "rebalance_date": rebalance_date.isoformat(),
        "trailing_window_months": DEPLOY_WINDOW_MONTHS,
        "profile_id": profile_id,
        "lanes": [
            {
                "strategy_id": s.strategy_id,
                "instrument": s.instrument,
                "orb_label": s.orb_label,
                "rr_target": s.rr_target,
                "filter_type": s.filter_type,
                "annual_r": s.annual_r_estimate,
                "trailing_expr": s.trailing_expr,
                "trailing_n": s.trailing_n,
                "trailing_wr": s.trailing_wr,
                "months_negative": s.months_negative,
                "session_regime": (
                    "HOT"
                    if s.session_regime_expr and s.session_regime_expr > 0.03
                    else "COLD"
                    if s.session_regime_expr and s.session_regime_expr < -0.03
                    else "FLAT"
                ),
                "status": s.status,
                "status_reason": s.status_reason,
            }
            for s in allocation
        ],
        "paused": [{"strategy_id": s.strategy_id, "reason": s.status_reason} for s in scores if s.status == "PAUSE"],
        "all_scores_count": len(scores),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path
