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
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.chordia import (
    CHORDIA_T_WITH_THEORY,
    CHORDIA_T_WITHOUT_THEORY,
    ChordiaAuditLog,
    chordia_verdict_allows_deploy,
    load_chordia_audit_log,
)
from trading_app.config import ALL_FILTERS
from trading_app.lane_correlation import RHO_REJECT_THRESHOLD, _load_lane_daily_pnl, _pearson
from trading_app.strategy_discovery import _inject_cross_asset_atrs
from trading_app.validated_shelf import deployable_validated_relation


def _normalize_writable_path(path: Path) -> Path:
    text = str(path)
    if text.startswith("/mnt/c/Users/"):
        return Path(text.replace("/mnt/c/Users/", "/mnt/c/users/", 1))
    return path


_REPO_ROOT = _normalize_writable_path(Path(__file__).resolve().parents[1])
DEFAULT_LANE_ALLOCATION_PATH = _REPO_ROOT / "docs" / "runtime" / "lane_allocation.json"

# ---------------------------------------------------------------------------
# Literature-grounded constants (do NOT tune on backtest — see spec §Parameter Source)
# ---------------------------------------------------------------------------
DEPLOY_WINDOW_MONTHS = 12  # Carver Ch.11: forecast weighting window
REGIME_WINDOW_MONTHS = 6  # Chan Ch.7: regime half-life capture
# Individual strategy pause/resume constants REMOVED (Apr 2026).
# Backtest 2022-2025 proved regime-only gating outperforms:
#   Current (top 5, individual pause): -799R, negative every year
#   Regime gate (top 9, session-level): +630R, positive every year
# Individual month streaks are noise. Session regime is structural.
MIN_TRAILING_N = 20  # Minimum trades for reliable score
HYSTERESIS_PCT = 0.20  # Carver Ch.12: 20% switching cost
PROVISIONAL_MONTHS = 6  # Minimum months for full (non-provisional) status


@dataclass
class LaneScore:
    """Score for a single validated strategy in the trailing window."""

    strategy_id: str
    instrument: str
    orb_label: str
    orb_minutes: int
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
    # Liveness fields (populated post-scoring)
    recent_3mo_expr: float | None = None  # 3-month trailing ExpR (decay signal)
    sr_status: str = "UNKNOWN"  # ALARM / CONTINUE / NO_DATA / UNKNOWN
    # Chordia gate fields (populated by compute_lane_scores from
    # chordia_audit_log.yaml strict-replay rows). Defaults are None so
    # existing _make_score() test factories with 20+ call sites do not need
    # updating.
    chordia_verdict: str | None = None  # PASS_CHORDIA/PASS_PROTOCOL_A/FAIL_CHORDIA/FAIL_BOTH/MISSING
    chordia_audit_age_days: int | None = None  # days since audit_date in doctrine YAML


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
    orb_minutes: int,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
    stop_multiplier: float,
    rebalance_date: date,
    n_months: int = 6,
) -> tuple[list[tuple[str, float, int]], int, int]:
    """Compute per-calendar-month ExpR for a strategy.

    Returns list of (year_month, avg_pnl_r, n_trades) tuples, most recent first.
    Applies filter via matches_row and SM adjustment via mae_r.
    """
    start, end = _month_range(rebalance_date, n_months)

    # Load outcomes (include entry/stop for canonical SM adjustment)
    outcomes = con.execute(
        """
        SELECT o.trading_day, o.pnl_r, o.mae_r, o.outcome,
               o.entry_price, o.stop_price
        FROM orb_outcomes o
        WHERE o.symbol = ? AND o.orb_label = ? AND o.entry_model = ?
          AND o.rr_target = ? AND o.confirm_bars = ? AND o.orb_minutes = ?
          AND o.outcome IN ('win', 'loss')
          AND o.trading_day >= ? AND o.trading_day < ?
        ORDER BY o.trading_day
        """,
        [instrument, orb_label, entry_model, rr_target, confirm_bars, orb_minutes, start, end],
    ).fetchall()

    if not outcomes:
        return [], 0, 0

    # Load daily_features for filter application
    features = con.execute(
        """
        SELECT * FROM daily_features
        WHERE symbol = ? AND orb_minutes = ?
          AND trading_day >= ? AND trading_day < ?
        ORDER BY trading_day
        """,
        [instrument, orb_minutes, start, end],
    ).fetchall()

    if not features:
        return [], 0, 0

    feat_cols = [desc[0] for desc in con.description]
    feat_by_day = {}
    for row in features:
        d = dict(zip(feat_cols, row, strict=False))
        feat_by_day[d["trading_day"]] = d

    # Inject cross-asset ATR enrichment (cross_atr_{source}_pct) for any
    # CrossAssetATRFilter in ALL_FILTERS. Without this the filter fails-closed
    # on every day and the lane shows STALE ("No trades in trailing window")
    # despite an active validated_setups cohort. Canonical helper from
    # strategy_discovery; live entry path injects identically. See
    # trading_app/config.py:982-984 for the schema-not-stored doctrine note.
    _inject_cross_asset_atrs(con, list(feat_by_day.values()), instrument, ALL_FILTERS)

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

    # Filter outcomes to eligible days + apply canonical SM adjustment
    cost_spec = COST_SPECS.get(instrument)
    adjusted = []  # (trading_day, adjusted_pnl_r, is_win)
    for td, pnl_r, mae_r, _outcome, entry_price, stop_price in outcomes:
        if td not in eligible_days:
            continue
        # Canonical SM adjustment (matches apply_tight_stop in config.py):
        # max_adv_pts = mae_r * risk_d / point_value
        # killed = max_adv_pts >= stop_multiplier * risk_pts
        adj_pnl_r = pnl_r
        if (
            stop_multiplier != 1.0
            and mae_r is not None
            and entry_price is not None
            and stop_price is not None
            and cost_spec is not None
        ):
            risk_pts = abs(entry_price - stop_price)
            if risk_pts > 0:
                raw_risk_d = risk_pts * cost_spec.point_value
                risk_d = raw_risk_d + cost_spec.total_friction
                max_adv_pts = mae_r * risk_d / cost_spec.point_value
                if max_adv_pts >= stop_multiplier * risk_pts:
                    adj_pnl_r = round(-stop_multiplier, 4)
        is_win = adj_pnl_r > 0
        adjusted.append((td, adj_pnl_r, is_win))

    if not adjusted:
        return [], 0, 0

    # Group by calendar month
    from collections import defaultdict

    monthly: dict[str, list[float]] = defaultdict(list)
    total_wins = 0
    total_trades = 0
    for td, pnl_r, is_win in adjusted:
        ym = f"{td.year}-{td.month:02d}"
        monthly[ym].append(pnl_r)
        total_trades += 1
        if is_win:
            total_wins += 1

    result = []
    for ym in sorted(monthly.keys(), reverse=True):
        trades = monthly[ym]
        avg_r = sum(trades) / len(trades)
        result.append((ym, round(avg_r, 4), len(trades)))

    return result, total_wins, total_trades


def compute_lane_scores(
    rebalance_date: date,
    trailing_months: int = DEPLOY_WINDOW_MONTHS,
    db_path: str | Path | None = None,
    audit_log: ChordiaAuditLog | None = None,
) -> list[LaneScore]:
    """Compute trailing performance for all validated strategies.

    Zero look-ahead: all queries use trading_day < rebalance_date.
    SM-adjusted: applies tight-stop via mae_r.
    Filter-applied: uses matches_row on daily_features.

    Each LaneScore is populated with chordia_verdict + chordia_audit_age_days
    from doctrine audit YAML only. The build_allocation() gate refuses DEPLOY
    for verdicts other than PASS_CHORDIA / PASS_PROTOCOL_A, for missing strict
    replay audit rows, and for audits older than the doctrine's freshness
    threshold.
    """
    db = db_path or GOLD_DB_PATH
    con = duckdb.connect(str(db), read_only=True)
    configure_connection(con)

    if audit_log is None:
        audit_log = load_chordia_audit_log()
    assert audit_log is not None  # for type narrowing inside the try block

    try:
        # Load all active validated strategies. The live Chordia gate does NOT
        # derive verdicts from validated_setups; strict replay audit rows in
        # docs/runtime/chordia_audit_log.yaml are the gate truth.
        shelf_relation = deployable_validated_relation(con, alias="vs")
        strategies = con.execute(
            f"""
            SELECT strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   rr_target, confirm_bars, filter_type, stop_multiplier
            FROM {shelf_relation}
            ORDER BY instrument, orb_label, orb_minutes
            """
        ).fetchall()

        scores = []
        for sid, inst, orb, om, em, rr, cb, ft, sm in strategies:
            chordia_verdict_value = audit_log.verdict(sid) or "MISSING"
            chordia_age = audit_log.audit_age_days(sid, rebalance_date)
            # Per-month ExpR for trailing window
            monthly, total_wins, total_trades = _per_month_expr(
                con,
                inst,
                orb,
                om,
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
                        orb_minutes=om,
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
                        chordia_verdict=chordia_verdict_value,
                        chordia_audit_age_days=chordia_age,
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

            # Trade-level win rate (from SM-adjusted outcomes, not monthly averages)
            trailing_wr = round(total_wins / total_trades, 3) if total_trades > 0 else 0.0

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

            # Recent 3-month trailing ExpR (decay signal)
            recent_3mo = None
            if len(monthly) >= 3:
                recent_entries = monthly[:3]  # most-recent-first
                r3_n = sum(n for _, _, n in recent_entries)
                r3_pnl = sum(expr * n for _, expr, n in recent_entries)
                recent_3mo = round(r3_pnl / r3_n, 4) if r3_n > 0 else None

            scores.append(
                LaneScore(
                    strategy_id=sid,
                    instrument=inst,
                    orb_label=orb,
                    orb_minutes=om,
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
                    recent_3mo_expr=recent_3mo,
                    chordia_verdict=chordia_verdict_value,
                    chordia_audit_age_days=chordia_age,
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

    NOTE: orb_minutes=5 is a deliberate fixed reference aperture for the
    session-health regime signal — NOT the strategy's own aperture. Every
    strategy in a given session sees the same regime number; it acts as a
    shared "is this session structurally hot or cold right now?" gate.
    O5 is the canonical reference because it's the densest cohort and
    most stable signal across instruments. Do not parameterize this on
    strategy.orb_minutes — that would make the regime gate per-strategy,
    breaking the cross-aperture comparison and the original 2025 backtest
    design (+630R regime-only vs -799R per-strategy pause).
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
    """Classify strategy status — regime-only gating.

    Backtest 2022-2025 proved session regime is the only gate that matters:
      - Individual strategy month streaks are NOISE (small N per month)
      - Session regime is STRUCTURAL (pooled across all strategies)
      - Regime gate: +630R, positive all 4 years
      - Old individual pause: -799R, negative all 4 years

    Rules:
      1. STALE if not enough trades AND no regime data
      2. PAUSE if session regime is COLD (6mo trailing <= 0)
      3. DEPLOY if session regime is HOT (6mo trailing > 0)
    """
    # STALE: no regime data available
    if session_regime_expr is None:
        if trailing_n < MIN_TRAILING_N:
            return "STALE", f"No regime data, thin trades (N={trailing_n})"
        # No regime but has trades — deploy if trailing is positive
        if trailing_expr > 0:
            return "DEPLOY", f"No regime data, trailing positive (ExpR={trailing_expr:.4f}, N={trailing_n})"
        return "PAUSE", f"No regime data, trailing negative (ExpR={trailing_expr:.4f})"

    # SESSION REGIME GATE — the only gate that matters
    if session_regime_expr <= 0:
        return "PAUSE", f"Session regime COLD ({session_regime_expr:+.4f})"

    # Session is HOT — deploy regardless of individual strategy streaks
    if trailing_n < MIN_TRAILING_N:
        return "DEPLOY", f"Thin data (N={trailing_n}), session regime HOT ({session_regime_expr:+.4f})"

    return "DEPLOY", f"Session HOT ({session_regime_expr:+.4f}), ExpR={trailing_expr:+.4f}, N={trailing_n}"


def load_sr_state() -> dict[str, str]:
    """Load persisted SR monitor state.

    Returns {strategy_id: "ALARM" | "CONTINUE" | "NO_DATA"}.
    Fail-open: returns empty dict if file missing/corrupt/stale.
    """
    state_path = Path(__file__).resolve().parents[1] / "data" / "state" / "sr_state.json"
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text())
        # Validate envelope freshness (max 7 days — SR state older than that is stale)
        freshness = data.get("freshness", {})
        as_of = freshness.get("as_of_date")
        if as_of:
            from datetime import date as _date

            state_date = _date.fromisoformat(as_of)
            if (date.today() - state_date).days > 7:
                return {}  # stale — fail-open with empty
        payload = data.get("payload", {})
        results = payload.get("results", [])
        return {r["strategy_id"]: r["status"] for r in results if "strategy_id" in r and "status" in r}
    except (json.JSONDecodeError, KeyError, OSError, ValueError):
        return {}


def enrich_scores_with_liveness(scores: list[LaneScore]) -> None:
    """Populate sr_status on scored lanes from persisted SR state.

    Mutates scores in place. Called after compute_lane_scores().
    """
    sr_state = load_sr_state()
    for s in scores:
        s.sr_status = sr_state.get(s.strategy_id, "UNKNOWN")


# ---------------------------------------------------------------------------
# Liveness-aware ranking
# ---------------------------------------------------------------------------
# Ranking discount factors (Carver Ch.12 forecast decay principle):
#   ALARM: halve expected return — drift detected, but ARL~60 means ~1 false
#          alarm per 60 trades, so hard-block is too aggressive.
#   3mo-negative: 0.75x — recent momentum is against the lane, but 12mo
#                 trailing is still positive. Discount, don't kill.
SR_ALARM_DISCOUNT = 0.50
RECENT_DECAY_DISCOUNT = 0.75


def _effective_annual_r(s: LaneScore) -> float:
    """Liveness-adjusted annual R for ranking purposes.

    Applies multiplicative discounts for SR alarm and recent decay.
    The raw annual_r_estimate is preserved on the LaneScore; this
    function is used ONLY for sorting during selection.
    """
    adj = s.annual_r_estimate
    if s.sr_status == "ALARM":
        adj *= SR_ALARM_DISCOUNT
    if s.recent_3mo_expr is not None and s.recent_3mo_expr < 0 and s.trailing_expr > 0:
        # 12mo positive but recent 3mo negative = decaying
        adj *= RECENT_DECAY_DISCOUNT
    return adj


def compute_pairwise_correlation(
    candidates: list[LaneScore],
    db_path: str | Path | None = None,
) -> dict[tuple[str, str], float]:
    """Compute pairwise Pearson rho on filtered daily P&L for candidate strategies.

    Uses the same trailing window as the allocator (all available data through
    rebalance_date). Returns {(sid_a, sid_b): rho} with canonical key ordering.
    """
    db = db_path or GOLD_DB_PATH
    con = duckdb.connect(str(db), read_only=True)
    configure_connection(con)

    try:
        pnl_series: dict[str, dict] = {}
        for s in candidates:
            lane = {
                "instrument": s.instrument,
                "orb_label": s.orb_label,
                "orb_minutes": s.orb_minutes,
                "entry_model": "E2",
                "rr_target": s.rr_target,
                "confirm_bars": s.confirm_bars,
                "filter_type": s.filter_type,
            }
            pnl_series[s.strategy_id] = _load_lane_daily_pnl(con, lane)

        sids = [s.strategy_id for s in candidates]
        pairs: dict[tuple[str, str], float] = {}
        for i, a in enumerate(sids):
            for j, b in enumerate(sids):
                if j <= i:
                    continue
                key = (a, b) if a < b else (b, a)
                shared = sorted(set(pnl_series[a]) & set(pnl_series[b]))
                if len(shared) >= 5:
                    xs = [pnl_series[a][d] for d in shared]
                    ys = [pnl_series[b][d] for d in shared]
                    pairs[key] = _pearson(xs, ys)
                else:
                    pairs[key] = 0.0
        return pairs
    finally:
        con.close()


def apply_chordia_gate(
    scores: list[LaneScore],
    *,
    audit_log: ChordiaAuditLog | None = None,
) -> list[LaneScore]:
    """Refuse DEPLOY for strategies failing the Chordia criterion.

    For each input score, returns a new LaneScore with status mutated to
    "PAUSE" and status_reason describing the chordia failure when:
      - chordia_verdict is None or in (FAIL_BOTH, FAIL_CHORDIA, MISSING)
      - chordia_audit_age_days is None (no doctrine audit) — treated as MISSING
      - chordia_audit_age_days > audit_log.audit_freshness_days (default 90)

    Strategies that already have status STALE or PAUSE are left unchanged
    (chordia gate does not "rescue" a stale strategy or change an existing
    PAUSE reason). Strategies passing the gate are returned unchanged.

    The gate runs BEFORE build_allocation's ranking. This keeps chordia
    logic out of the DD/correlation/hysteresis machinery — build_allocation's
    existing filter `status in ("DEPLOY","RESUME","PROVISIONAL")` then
    naturally excludes refused strategies.
    """
    log = audit_log if audit_log is not None else load_chordia_audit_log()
    freshness = log.audit_freshness_days

    out: list[LaneScore] = []
    for s in scores:
        # Don't override an existing PAUSE/STALE — chordia gate is additive.
        if s.status in ("PAUSE", "STALE"):
            out.append(s)
            continue

        verdict = s.chordia_verdict
        age = s.chordia_audit_age_days

        reason: str | None = None
        if verdict is None or verdict == "MISSING":
            reason = "chordia gate: missing strict replay audit verdict"
        elif verdict == "PARK":
            reason = "chordia gate: PARK (strict replay not deployment-cleared)"
        elif verdict == "FAIL_BOTH":
            reason = f"chordia gate: FAIL_BOTH (t<{CHORDIA_T_WITH_THEORY})"
        elif verdict == "FAIL_CHORDIA":
            reason = f"chordia gate: FAIL_CHORDIA (t<{CHORDIA_T_WITHOUT_THEORY}, no theory grant)"
        elif age is None:
            reason = "chordia gate: no audit_date in doctrine YAML"
        elif age > freshness:
            reason = f"chordia gate: audit stale ({age}d > {freshness}d)"
        elif not chordia_verdict_allows_deploy(verdict):
            # Defensive: any future verdict label not in the deploy-allow set.
            reason = f"chordia gate: verdict {verdict} does not permit DEPLOY"

        if reason is None:
            out.append(s)
            continue

        # Demote to PAUSE with structured reason.
        out.append(
            LaneScore(
                strategy_id=s.strategy_id,
                instrument=s.instrument,
                orb_label=s.orb_label,
                orb_minutes=s.orb_minutes,
                rr_target=s.rr_target,
                filter_type=s.filter_type,
                confirm_bars=s.confirm_bars,
                stop_multiplier=s.stop_multiplier,
                trailing_expr=s.trailing_expr,
                trailing_n=s.trailing_n,
                trailing_months=s.trailing_months,
                annual_r_estimate=s.annual_r_estimate,
                trailing_wr=s.trailing_wr,
                session_regime_expr=s.session_regime_expr,
                months_negative=s.months_negative,
                months_positive_since_last_neg_streak=s.months_positive_since_last_neg_streak,
                status="PAUSE",
                status_reason=reason,
                recent_3mo_expr=s.recent_3mo_expr,
                sr_status=s.sr_status,
                chordia_verdict=verdict,
                chordia_audit_age_days=age,
            )
        )
    return out


def build_allocation(
    scores: list[LaneScore],
    *,
    max_slots: int = 5,
    max_dd: float = 3000.0,
    allowed_instruments: frozenset[str] | None = None,
    allowed_sessions: frozenset[str] | None = None,
    stop_multiplier: float = 0.75,
    prior_allocation: list[str] | None = None,
    orb_size_stats: dict[tuple[str, str, int], tuple[float, float]] | None = None,
    correlation_matrix: dict[tuple[str, str], float] | None = None,
) -> list[LaneScore]:
    """Select top lanes for a profile, respecting constraints.

    Correlation-aware greedy selection: rank by liveness-adjusted annual_r,
    accept candidate only if pairwise rho < 0.70 with all already-selected
    lanes. This replaces the old 1-per-session heuristic and naturally
    deduplicates same-session strategies (rho ≈ 1.0) while also catching
    cross-session redundancies.

    When correlation_matrix is None, falls back to the 1-per-session
    heuristic (for unit tests and environments without DB access).

    DD estimation uses per-session P90 from orb_size_stats when available.
    Hysteresis: only replace a lane if new candidate >20% better.

    Chordia gate is applied in-line: any score whose chordia_verdict / age
    fails policy is demoted to PAUSE before the deployable filter, so the
    refused strategy cannot leak into the JSON. Callers may also call
    apply_chordia_gate() upstream — running the gate twice is idempotent.
    """
    # Apply Chordia gate first — refuse FAIL_BOTH / FAIL_CHORDIA / MISSING /
    # stale-audit strategies before any ranking. Idempotent.
    scores = apply_chordia_gate(scores)

    # Filter to deployable
    candidates = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]

    # Apply profile constraints
    if allowed_instruments:
        candidates = [s for s in candidates if s.instrument in allowed_instruments]
    if allowed_sessions:
        candidates = [s for s in candidates if s.orb_label in allowed_sessions]

    if correlation_matrix is not None:
        # Correlation-aware: all candidates enter ranking, greedy rho gate
        ranked = sorted(
            candidates,
            key=lambda s: (0 if s.status == "PROVISIONAL" else 1, _effective_annual_r(s)),
            reverse=True,
        )
    else:
        # Fallback: 1 winner per instrument × session (legacy behavior)
        best_per_session: dict[tuple[str, str], LaneScore] = {}
        for s in candidates:
            key = (s.instrument, s.orb_label)
            if key not in best_per_session or _effective_annual_r(s) > _effective_annual_r(best_per_session[key]):
                best_per_session[key] = s
        ranked = sorted(
            best_per_session.values(),
            key=lambda s: (0 if s.status == "PROVISIONAL" else 1, _effective_annual_r(s)),
            reverse=True,
        )

    # Greedy DD-budget + correlation selection
    selected: list[LaneScore] = []
    dd_used = 0.0

    for lane in ranked:
        if len(selected) >= max_slots:
            break

        # Correlation gate: reject if rho > threshold with any selected lane
        if correlation_matrix is not None:
            corr_reject = False
            for sel in selected:
                key = (
                    (lane.strategy_id, sel.strategy_id)
                    if lane.strategy_id < sel.strategy_id
                    else (sel.strategy_id, lane.strategy_id)
                )
                rho = correlation_matrix.get(key, 0.0)
                if rho > RHO_REJECT_THRESHOLD:
                    corr_reject = True
                    break
            if corr_reject:
                continue

        # Estimate worst-case DD contribution using per-session P90
        cost = COST_SPECS.get(lane.instrument)
        if cost is None:
            continue
        if orb_size_stats:
            orb_key = (lane.instrument, lane.orb_label, lane.orb_minutes)
            _, p90_orb = orb_size_stats.get(orb_key, (100.0, 100.0))
        else:
            from trading_app.prop_profiles import _P90_ORB_PTS

            p90_orb = _P90_ORB_PTS.get(lane.instrument, 100.0)
        lane_dd = p90_orb * stop_multiplier * cost.point_value

        if dd_used + lane_dd > max_dd:
            continue

        # Hysteresis: only replace a prior lane if new candidate is >20% better
        if prior_allocation and lane.strategy_id not in prior_allocation:
            session_key = (lane.instrument, lane.orb_label)
            prior_in_session = [
                s for s in scores if s.strategy_id in prior_allocation and (s.instrument, s.orb_label) == session_key
            ]
            if prior_in_session:
                best_prior = max(prior_in_session, key=lambda s: s.annual_r_estimate)
                if best_prior.annual_r_estimate > 0:
                    improvement = (lane.annual_r_estimate - best_prior.annual_r_estimate) / best_prior.annual_r_estimate
                    if improvement < HYSTERESIS_PCT:
                        if best_prior.status in ("DEPLOY", "RESUME", "PROVISIONAL"):
                            selected.append(best_prior)
                            dd_used += lane_dd
                        continue

        selected.append(lane)
        dd_used += lane_dd

    return selected


def generate_report(
    scores: list[LaneScore],
    allocation: list[LaneScore],
    rebalance_date: date,
    profile_id: str,
) -> str:
    """Generate human-readable rebalance report with diff and collapsed paused list."""
    from collections import Counter

    gated_scores = apply_chordia_gate(scores)

    lines = [
        f"# Lane Allocation Report — {rebalance_date}",
        f"Profile: {profile_id}",
        f"Total candidates: {len(gated_scores)}",
        f"Deployable: {sum(1 for s in gated_scores if s.status in ('DEPLOY', 'RESUME', 'PROVISIONAL'))}",
        f"Paused: {sum(1 for s in gated_scores if s.status == 'PAUSE')}",
        f"Stale: {sum(1 for s in gated_scores if s.status == 'STALE')}",
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

    # Diff vs current prop_profiles lanes
    try:
        from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes

        profile = ACCOUNT_PROFILES.get(profile_id)
        current_lanes = effective_daily_lanes(profile) if profile else ()
        if current_lanes:
            current_ids = {spec.strategy_id for spec in current_lanes}
            recommended_ids = {s.strategy_id for s in allocation}
            added = recommended_ids - current_ids
            removed = current_ids - recommended_ids
            kept = current_ids & recommended_ids

            lines.extend(["", "## Changes vs Current Lanes"])
            if not added and not removed:
                lines.append("  No changes — current lanes match recommendation.")
            else:
                for sid in sorted(kept):
                    lines.append(f"  [KEEP]   {sid}")
                for sid in sorted(added):
                    s_obj = next((x for x in allocation if x.strategy_id == sid), None)
                    ann = f" (annual_r={s_obj.annual_r_estimate:.1f})" if s_obj else ""
                    lines.append(f"  [NEW]    {sid}{ann}")
                for sid in sorted(removed):
                    s_obj = next((x for x in gated_scores if x.strategy_id == sid), None)
                    reason = f" ({s_obj.status}: {s_obj.status_reason})" if s_obj else ""
                    lines.append(f"  [DROP]   {sid}{reason}")
                lines.extend(
                    [
                        "",
                        "  ACTION: Update prop_profiles.py daily_lanes if you accept these changes.",
                    ]
                )
    except ImportError as e:
        lines.append(f"\n  [WARNING] Could not load prop_profiles for diff: {e}")

    # Paused lanes — collapsed by reason category (not 115 individual lines)
    paused = [s for s in gated_scores if s.status == "PAUSE"]
    if paused:
        reason_groups: dict[str, int] = Counter()
        for s in paused:
            if "Recovering" in s.status_reason:
                reason_groups["Recovering (need 3+ positive months)"] += 1
            elif "chordia gate:" in s.status_reason:
                reason_groups["Chordia gate"] += 1
            elif "Magnitude override" in s.status_reason:
                reason_groups["Magnitude override (3mo avg < -0.10)"] += 1
            elif "consecutive months negative" in s.status_reason:
                reason_groups["Consecutive months negative"] += 1
            else:
                reason_groups["Other"] += 1
        lines.extend(["", f"## Paused ({len(paused)} strategies)"])
        for reason, count in sorted(reason_groups.items(), key=lambda x: -x[1]):
            lines.append(f"  {count:3d} x {reason}")

    # Session regimes
    lines.extend(["", "## Session Regimes (6mo trailing, unfiltered E2 RR1.0)"])
    seen_sessions: set[tuple[str, str]] = set()
    for s in gated_scores:
        key = (s.instrument, s.orb_label)
        if key in seen_sessions:
            continue
        seen_sessions.add(key)
        regime = s.session_regime_expr
        regime_str = f"{regime:+.4f}" if regime is not None else "N/A"
        tag = "HOT" if regime and regime > 0.03 else "COLD" if regime and regime < -0.03 else "FLAT"
        lines.append(f"  {s.instrument} {s.orb_label:<20} {regime_str:>8} [{tag}]")

    return "\n".join(lines)


def compute_orb_size_stats(
    rebalance_date: date,
    db_path: str | Path | None = None,
) -> dict[tuple[str, str, int], tuple[float, float]]:
    """Compute trailing ORB size stats per (instrument, session, aperture).

    Returns {(instrument, orb_label, orb_minutes): (avg_orb_pts, p90_orb_pts)}.
    Uses 12-month trailing window with zero look-ahead.
    ORB size = risk_dollars / point_value (at SM=1.0, this equals the ORB range).

    Per-aperture grouping: O5 / O15 / O30 ORB ranges differ materially
    (O15 is ~50-60% larger than O5 in MNQ). DD budgets must use the
    strategy's actual aperture, not a fixed reference.
    """
    db = db_path or GOLD_DB_PATH
    con = duckdb.connect(str(db), read_only=True)
    try:
        start, end = _month_range(rebalance_date, DEPLOY_WINDOW_MONTHS)
        rows = con.execute(
            """
            SELECT symbol, orb_label, orb_minutes,
                   AVG(risk_dollars) as avg_risk_d,
                   PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY risk_dollars) as p90_risk_d
            FROM orb_outcomes
            WHERE entry_model = 'E2'
              AND rr_target = 1.0 AND confirm_bars = 1
              AND outcome IN ('win', 'loss')
              AND trading_day >= ? AND trading_day < ?
            GROUP BY symbol, orb_label, orb_minutes
            """,
            [start, end],
        ).fetchall()

        result: dict[tuple[str, str, int], tuple[float, float]] = {}
        for sym, orb, om, avg_d, p90_d in rows:
            pv = COST_SPECS[sym].point_value if sym in COST_SPECS else 1.0
            result[(sym, orb, om)] = (round(avg_d / pv, 1), round(p90_d / pv, 1))
        return result
    finally:
        con.close()


def save_allocation(
    scores: list[LaneScore],
    allocation: list[LaneScore],
    rebalance_date: date,
    profile_id: str,
    output_path: str | Path | None = None,
    orb_size_stats: dict[tuple[str, str, int], tuple[float, float]] | None = None,
) -> Path:
    """Save allocation to JSON file.

    orb_size_stats: {(instrument, orb_label, orb_minutes): (avg_orb_pts, p90_orb_pts)}.
    If None, ORB size fields are omitted from the output.
    """
    gated_scores = apply_chordia_gate(scores)
    path = Path(output_path) if output_path else DEFAULT_LANE_ALLOCATION_PATH
    path = _normalize_writable_path(path)

    def _blocked_entry(s: LaneScore) -> dict[str, object]:
        return {
            "strategy_id": s.strategy_id,
            "status": s.status,
            "reason": s.status_reason,
            "chordia_verdict": s.chordia_verdict,
            "chordia_audit_age_days": s.chordia_audit_age_days,
        }

    lanes_data = []
    for s in allocation:
        lane = {
            "strategy_id": s.strategy_id,
            "instrument": s.instrument,
            "orb_label": s.orb_label,
            "orb_minutes": s.orb_minutes,
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
            # Chordia gate state — read by drift check and downstream auditors.
            # Drift check refuses lanes[] whose verdict is not PASS_* or whose
            # audit_age_days exceeds the doctrine freshness threshold.
            "chordia_verdict": s.chordia_verdict,
            "chordia_audit_age_days": s.chordia_audit_age_days,
        }
        if orb_size_stats:
            orb_key = (s.instrument, s.orb_label, s.orb_minutes)
            if orb_key in orb_size_stats:
                avg_pts, p90_pts = orb_size_stats[orb_key]
                lane["avg_orb_pts"] = avg_pts
                lane["p90_orb_pts"] = p90_pts
        lanes_data.append(lane)

    data = {
        "rebalance_date": rebalance_date.isoformat(),
        "trailing_window_months": DEPLOY_WINDOW_MONTHS,
        "profile_id": profile_id,
        "lanes": lanes_data,
        "paused": [_blocked_entry(s) for s in gated_scores if s.status == "PAUSE"],
        "stale": [_blocked_entry(s) for s in gated_scores if s.status == "STALE"],
        "all_scores_count": len(gated_scores),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------
STALENESS_WARNING_DAYS = 35  # Pre-session check: log warning, continue trading
STALENESS_BLOCK_DAYS = 60  # Pre-session check: refuse to trade until rebalance


def check_allocation_staleness(
    allocation_path: str | Path | None = None,
    today: date | None = None,
) -> tuple[str, int]:
    """Check if lane allocation is stale.

    Returns (status, days_old):
      - "OK": allocation is fresh
      - "WARNING": >35 days, should rebalance soon
      - "BLOCK": >60 days, must rebalance before trading
    days_old = -1 if file not found.
    """
    if allocation_path:
        path = _normalize_writable_path(Path(allocation_path))
    else:
        path = DEFAULT_LANE_ALLOCATION_PATH
    if not path.exists():
        return "BLOCK", -1

    data = json.loads(path.read_text())
    rebalance_date = date.fromisoformat(data["rebalance_date"])
    check_date = today or date.today()
    days_old = (check_date - rebalance_date).days

    if days_old > STALENESS_BLOCK_DAYS:
        return "BLOCK", days_old
    if days_old > STALENESS_WARNING_DAYS:
        return "WARNING", days_old
    return "OK", days_old
