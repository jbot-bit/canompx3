"""
Prop firm portfolio selection.

Selects optimal strategy subset from validated universe for a specific
prop firm account profile. Enforces DD budget, contract caps, cognitive
load limits, and consistency rules.

Zero modification to existing live_config.py or portfolio.py.

Usage:
    python -m trading_app.prop_portfolio --profile topstep_50k
    python -m trading_app.prop_portfolio --all
    python -m trading_app.prop_portfolio --summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

import duckdb  # noqa: F401 — used at runtime in _load_daily_snapshot

from pipeline.cost_model import get_cost_spec  # noqa: F401
from pipeline.db_config import configure_connection  # noqa: F401
from pipeline.dst import SESSION_CATALOG
from trading_app.portfolio import PortfolioStrategy
from trading_app.prop_profiles import (
    ACCOUNT_PROFILES,
    AccountProfile,
    DailyLaneSpec,  # noqa: F401
    ExcludedEntry,
    TradingBook,
    TradingBookEntry,
    compute_profit_split_factor,
    get_account_tier,
    effective_daily_lanes,
    get_firm_spec,
)
from trading_app.strategy_fitness import compute_fitness  # noqa: F401
from trading_app.validated_shelf import deployable_validated_relation

# =========================================================================
# DD estimation — per-lane from DB, with fallback constants
# =========================================================================

# Fallback constants when median_risk_points is unavailable.
# Updated Apr 2026 from actual median_risk_points across validated strategies.
# Old values (935/1350) were from a stale Monte Carlo — 23x too high.
DD_PER_CONTRACT_075X = 100.0  # P95 of actual per-lane DD at S0.75
DD_PER_CONTRACT_10X = 120.0  # P95 of actual per-lane DD at S1.0
# Intraday trailing is stricter: unrealized PnL moves floor in real-time.
INTRADAY_TRAILING_FACTOR = 1.4


def _compute_dd_per_contract(
    stop_multiplier: float,
    dd_type: str,
    median_risk_points: float | None = None,
    point_value: float | None = None,
    friction: float | None = None,
) -> float:
    """DD contribution per contract per lane.

    If median_risk_points is provided (from experimental_strategies), computes
    actual risk: median_risk_pts * stop_mult * point_value + friction.
    Otherwise falls back to conservative P95 estimate.
    """
    if median_risk_points is not None and point_value is not None:
        base = median_risk_points * stop_multiplier * point_value + (friction or 0)
    elif stop_multiplier <= 0.75:
        base = DD_PER_CONTRACT_075X
    else:
        base = DD_PER_CONTRACT_10X

    if dd_type == "intraday_trailing":
        return base * INTRADAY_TRAILING_FACTOR
    return base


def _apply_instrument_bans(
    strategies: list[PortfolioStrategy],
    banned: frozenset[str],
) -> tuple[list[PortfolioStrategy], list[ExcludedEntry]]:
    """Remove strategies on banned instruments."""
    if not banned:
        return strategies, []
    kept = []
    excluded = []
    for s in strategies:
        if s.instrument in banned:
            excluded.append(
                ExcludedEntry(
                    s.strategy_id,
                    s.instrument,
                    s.orb_label,
                    f"Instrument banned by firm ({s.instrument})",
                )
            )
        else:
            kept.append(s)
    return kept, excluded


def _deduplicate_sessions(
    strategies: list[PortfolioStrategy],
) -> tuple[list[PortfolioStrategy], list[ExcludedEntry]]:
    """Keep best strategy per (session x instrument). Best = highest ExpR."""
    best: dict[tuple[str, str], PortfolioStrategy] = {}
    for s in strategies:
        key = (s.orb_label, s.instrument)
        if key not in best or s.expectancy_r > best[key].expectancy_r:
            best[key] = s

    kept_ids = {s.strategy_id for s in best.values()}
    excluded = [
        ExcludedEntry(
            s.strategy_id,
            s.instrument,
            s.orb_label,
            f"Session conflict: better strategy exists for {s.orb_label} x {s.instrument}",
        )
        for s in strategies
        if s.strategy_id not in kept_ids
    ]
    return list(best.values()), excluded


@dataclass
class _RankedStrategy:
    """Strategy with computed ranking score."""

    strategy: PortfolioStrategy
    effective_expr: float
    expr_dd_ratio: float  # ExpR/DD — project rules: sort by ExpR, never Sharpe alone


def _rank_strategies(
    strategies: list[PortfolioStrategy],
    split_factor: float,
) -> list[_RankedStrategy]:
    """Rank strategies by ExpR/DD ratio (per project statistical rules).

    ExpR is adjusted by profit split factor for effective ranking.
    """
    ranked = []
    for s in strategies:
        effective_expr = s.expectancy_r * split_factor
        dd = s.max_drawdown_r or 999.0
        ratio = effective_expr / dd if dd > 0 else 0.0
        ranked.append(_RankedStrategy(s, effective_expr, ratio))

    ranked.sort(key=lambda r: r.expr_dd_ratio, reverse=True)
    return ranked


def _get_session_time_brisbane(orb_label: str, trading_day: date | None = None) -> str:
    """Look up session time in Brisbane timezone from SESSION_CATALOG."""
    entry = SESSION_CATALOG.get(orb_label)
    if entry is None:
        return "unknown"
    resolver = entry.get("resolver")
    if resolver is None:
        return "unknown"
    h, m = resolver(trading_day or date.today())
    return f"{h:02d}:{m:02d}"


# =========================================================================
# Daily lane resolver (pinned strategy IDs for manual profiles)
# =========================================================================


@dataclass(frozen=True)
class DailyExecutionLane:
    """Resolved daily lane with TRADE/HOLD/REVIEW/SKIP status."""

    strategy_id: str
    instrument: str
    orb_label: str
    session_time_brisbane: str
    status: str
    reason: str
    entry_model: str | None = None
    rr_target: float | None = None
    confirm_bars: int | None = None
    filter_type: str | None = None
    orb_minutes: int | None = None
    strategy_stop: float | None = None
    planned_stop: float | None = None
    fitness_status: str = "UNKNOWN"
    expectancy_r: float | None = None
    exp_dollars: float | None = None
    sample_size: int | None = None
    execution_notes: str = ""


def _calendar_gate_reason(filter_type: str | None, trading_day: date) -> str | None:
    """Skip reason for weekday filters (NOMON/NOTUE/NOFRI)."""
    if not filter_type:
        return None
    wd = trading_day.weekday()
    if "NOMON" in filter_type and wd == 0:
        return "Calendar: skip Monday"
    if "NOTUE" in filter_type and wd == 1:
        return "Calendar: skip Tuesday"
    if "NOFRI" in filter_type and wd == 4:
        return "Calendar: skip Friday"
    return None


def _load_daily_snapshot(db_path: Path, strategy_id: str) -> dict | None:
    """Load validated row for a pinned daily lane."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        configure_connection(con, writing=False)
        row = con.execute(
            """
            SELECT vs.strategy_id, vs.instrument, vs.orb_label,
                   vs.entry_model, vs.rr_target, vs.confirm_bars,
                   vs.filter_type, vs.orb_minutes,
                   COALESCE(vs.stop_multiplier, 1.0) AS stop_multiplier,
                   vs.expectancy_r, vs.win_rate, vs.sample_size,
                   LOWER(vs.status) AS status,
                   es.median_risk_points
            FROM validated_setups vs
            LEFT JOIN experimental_strategies es ON vs.strategy_id = es.strategy_id
            WHERE vs.strategy_id = ?
            LIMIT 1
            """,
            [strategy_id],
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in con.description]
        return dict(zip(cols, row, strict=False))
    finally:
        con.close()


def _resolve_daily_lane(
    profile: AccountProfile, lane: DailyLaneSpec, db_path: Path, trading_day: date
) -> DailyExecutionLane:
    """Resolve one pinned lane against DB + fitness."""
    session_time = _get_session_time_brisbane(lane.orb_label, trading_day)
    planned_stop = lane.planned_stop_multiplier if lane.planned_stop_multiplier is not None else profile.stop_multiplier
    snap = _load_daily_snapshot(db_path, lane.strategy_id)
    if snap is None:
        return DailyExecutionLane(
            strategy_id=lane.strategy_id,
            instrument=lane.instrument,
            orb_label=lane.orb_label,
            session_time_brisbane=session_time,
            status="HOLD",
            reason="Strategy missing from validated_setups.",
            planned_stop=planned_stop,
            execution_notes=lane.execution_notes,
        )
    fitness_status = "UNKNOWN"
    try:
        f = compute_fitness(snap["strategy_id"], db_path=db_path)
        fitness_status = f.fitness_status
    except (ValueError, duckdb.Error):
        pass
    exp_dollars = None
    mrp = snap.get("median_risk_points")
    if snap["expectancy_r"] is not None and mrp is not None:
        try:
            exp_dollars = snap["expectancy_r"] * mrp * get_cost_spec(snap["instrument"]).point_value
        except (KeyError, ValueError):
            pass
    status, reason = "TRADE", "Ready."
    firm_spec = get_firm_spec(profile.firm)
    strat_stop = snap["stop_multiplier"]
    if snap["status"] != "active":
        status, reason = "HOLD", f"Strategy status: {snap['status']}"
    elif snap["instrument"] in firm_spec.banned_instruments:
        status, reason = "HOLD", f"Instrument banned ({snap['instrument']})"
    else:
        cal = _calendar_gate_reason(snap["filter_type"], trading_day)
        if cal:
            status, reason = "SKIP", cal
        elif fitness_status not in lane.required_fitness:
            status, reason = "HOLD", f"Fitness: {fitness_status}"
        elif abs(strat_stop - planned_stop) > 1e-9:
            status, reason = "REVIEW", f"Stop mismatch: plan {planned_stop:.2f}x vs validated {strat_stop:.2f}x"
    return DailyExecutionLane(
        strategy_id=snap["strategy_id"],
        instrument=snap["instrument"],
        orb_label=snap["orb_label"],
        session_time_brisbane=session_time,
        status=status,
        reason=reason,
        entry_model=snap["entry_model"],
        rr_target=snap["rr_target"],
        confirm_bars=snap["confirm_bars"],
        filter_type=snap["filter_type"],
        orb_minutes=snap["orb_minutes"],
        strategy_stop=strat_stop,
        planned_stop=planned_stop,
        fitness_status=fitness_status,
        expectancy_r=snap["expectancy_r"],
        exp_dollars=exp_dollars,
        sample_size=snap["sample_size"],
        execution_notes=lane.execution_notes,
    )


def _lane_stop(lane: DailyExecutionLane, profile: AccountProfile) -> float:
    """Resolved stop multiplier for a lane (lane override or profile default)."""
    return lane.planned_stop if lane.planned_stop is not None else profile.stop_multiplier


def check_daily_lanes_dd_budget(
    profile: AccountProfile,
    lanes: list[DailyExecutionLane],
) -> tuple[float, float, float, bool]:
    """Check DD budget for manually pinned daily lanes.

    Returns (max_dd_per_lane, total_dd_exposure, dd_limit, is_over_budget).

    Uses each lane's resolved planned_stop (which may differ from
    profile.stop_multiplier when DailyLaneSpec.planned_stop_multiplier
    is set). Only TRADE-status lanes count toward DD exposure.
    """
    firm_spec = get_firm_spec(profile.firm)
    tier = get_account_tier(profile.firm, profile.account_size)
    tradeable = [la for la in lanes if la.status == "TRADE"]
    if not tradeable:
        return 0.0, 0.0, tier.max_dd, False
    # Per-lane DD using each lane's actual stop multiplier
    lane_dds = [_compute_dd_per_contract(_lane_stop(la, profile), firm_spec.dd_type) for la in tradeable]
    total_dd = sum(lane_dds)
    max_dd_per_lane = max(lane_dds)
    return max_dd_per_lane, total_dd, tier.max_dd, total_dd > tier.max_dd


def resolve_daily_lanes(profile: AccountProfile, db_path: Path, trading_day: date) -> list[DailyExecutionLane]:
    """Resolve all pinned daily lanes for a profile, sorted by time.

    When profile.daily_lanes is empty, loads lanes from lane_allocation.json
    via load_allocation_lanes(). Applies lane overrides (pause/resume) from
    data/state/ if present.
    """
    lane_specs = profile.daily_lanes
    if not lane_specs:
        from trading_app.prop_profiles import load_allocation_lanes

        lane_specs = load_allocation_lanes(profile.profile_id)

    lanes = [_resolve_daily_lane(profile, lane, db_path, trading_day) for lane in lane_specs]

    # Apply lane overrides (pause/resume state file)
    try:
        from trading_app.lane_ctl import get_lane_override

        for i, la in enumerate(lanes):
            override = get_lane_override(profile.profile_id, la.strategy_id)
            if override is not None and la.status == "TRADE":
                reason = override.get("reason", "paused")
                since = override.get("since", "?")
                lanes[i] = DailyExecutionLane(
                    strategy_id=la.strategy_id,
                    instrument=la.instrument,
                    orb_label=la.orb_label,
                    session_time_brisbane=la.session_time_brisbane,
                    status="PAUSED",
                    reason=f"Paused since {since}: {reason}",
                    entry_model=la.entry_model,
                    rr_target=la.rr_target,
                    confirm_bars=la.confirm_bars,
                    filter_type=la.filter_type,
                    orb_minutes=la.orb_minutes,
                    strategy_stop=la.strategy_stop,
                    planned_stop=la.planned_stop,
                    fitness_status=la.fitness_status,
                    expectancy_r=la.expectancy_r,
                    exp_dollars=la.exp_dollars,
                    sample_size=la.sample_size,
                    execution_notes=la.execution_notes,
                )
    except ImportError:
        pass  # lane_ctl not available — no overrides

    return sorted(lanes, key=lambda la: _time_sort_key(la.session_time_brisbane))


def _query_paper_pnl(db_path: Path, strategy_id: str, lookback_days: int = 30) -> dict | None:
    """Query recent paper trade performance for a strategy. Returns None if no data."""
    try:
        with duckdb.connect(str(db_path), read_only=True) as con:
            configure_connection(con)
            row = con.execute(
                """SELECT
                    COUNT(*) FILTER (WHERE pnl_r > 0) as wins,
                    COUNT(*) FILTER (WHERE pnl_r <= 0) as losses,
                    COALESCE(SUM(pnl_r), 0.0) as cum_r,
                    MAX(trading_day) as last_trade
                FROM paper_trades
                WHERE strategy_id = ?
                  AND trading_day >= CURRENT_DATE - INTERVAL ? DAY""",
                [strategy_id, lookback_days],
            ).fetchone()
            if row is None or row[0] + row[1] == 0:
                return None
            return {"wins": row[0], "losses": row[1], "cum_r": row[2], "last_trade": row[3]}
    except Exception:
        logger.debug("_query_paper_pnl failed for %s: %s", strategy_id, exc_info=True)
        return None


def print_daily_lanes(
    profile: AccountProfile,
    lanes: list[DailyExecutionLane],
    trading_day: date,
    db_path: Path | None = None,
) -> None:
    """Print manual daily execution sheet for one profile."""
    firm_spec = get_firm_spec(profile.firm)
    day_str = trading_day.strftime("%A %d %b %Y")
    print(f"\n{'=' * 100}")
    print(f"  MANUAL DAILY SHEET — {firm_spec.display_name} — {profile.profile_id}")
    print(f"  {day_str} | Brisbane | Source: daily_lanes + validated_setups + fitness")
    print(f"{'=' * 100}")
    print(
        f"\n  {'Status':<8} {'Time':<10} {'Session':<16} {'Inst':<5} {'ORB':<4} "
        f"{'RR':<4} {'Stop':<8} {'Fit':<8} {'Strategy'}"
    )
    print(f"  {'-' * 8} {'-' * 10} {'-' * 16} {'-' * 5} {'-' * 4} {'-' * 4} {'-' * 8} {'-' * 8} {'-' * 44}")
    for la in lanes:
        rr = f"{la.rr_target:.1f}" if la.rr_target is not None else "-"
        orb = f"{la.orb_minutes}m" if la.orb_minutes is not None else "-"
        stop = f"{la.planned_stop:.2f}x" if la.planned_stop is not None else "-"
        print(
            f"  {la.status:<8} {_format_time_ampm(la.session_time_brisbane):<10} "
            f"{la.orb_label:<16} {la.instrument:<5} {orb:<4} {rr:<4} {stop:<8} "
            f"{la.fitness_status:<8} {la.strategy_id[:44]}"
        )
        details = []
        if la.filter_type:
            details.append(f"filter={la.filter_type}")
        if la.expectancy_r is not None:
            details.append(f"ExpR={la.expectancy_r:+.3f}")
        if la.exp_dollars is not None:
            details.append(f"Exp$={la.exp_dollars:+.2f}")
        if la.sample_size is not None:
            details.append(f"N={la.sample_size}")
        if la.entry_model and la.confirm_bars is not None:
            details.append(f"{la.entry_model} CB{la.confirm_bars}")
        if details:
            print(f"    {' | '.join(details)}")
        # Paper PnL (if db_path available)
        if db_path is not None:
            pnl = _query_paper_pnl(db_path, la.strategy_id)
            if pnl is not None:
                last = pnl["last_trade"]
                last_str = str(last) if last else "?"
                print(f"    paper 30d: {pnl['wins']}W {pnl['losses']}L | cumR: {pnl['cum_r']:+.2f} | last: {last_str}")
            else:
                print("    paper 30d: no data (run: python -m trading_app.paper_trade_logger --sync)")
        print(f"    -> {la.reason}")
        if la.execution_notes:
            print(f"    note: {la.execution_notes}")

    # DD budget summary (per-lane breakdown)
    _, total_dd, dd_limit, over_budget = check_daily_lanes_dd_budget(profile, lanes)
    tradeable = [la for la in lanes if la.status == "TRADE"]
    print(f"\n  {'─' * 60}")
    print(f"  DD BUDGET BREAKDOWN ({len(tradeable)} tradeable lanes)")
    for la in tradeable:
        stop = _lane_stop(la, profile)
        lane_dd = _compute_dd_per_contract(stop, firm_spec.dd_type)
        print(f"    {la.orb_label:<20} {stop:.2f}x stop  ${lane_dd:,.0f} DD")
    print(f"  {'─' * 40}")
    print(f"  TOTAL EXPOSURE: ${total_dd:,.0f}")
    print(f"  DD LIMIT:       ${dd_limit:,.0f} ({firm_spec.display_name} {profile.account_size // 1000}K)")
    if dd_limit > 0:
        pct = total_dd / dd_limit * 100
        if over_budget:
            print(f"  *** OVER-COMMITTED: {pct:.0f}% of DD limit. Max simultaneous losses WILL breach account. ***")
            print("    Mitigations: intraday DD halt, reduced sizing, skip marginal lanes.")
        else:
            print(f"  Within DD budget ({pct:.0f}%)")
    print()


def select_for_profile(
    profile: AccountProfile,
    strategies: list[PortfolioStrategy],
) -> TradingBook:
    """Select optimal strategy subset for an account profile.

    Algorithm:
    1. Filter banned instruments
    2. Deduplicate session x instrument (keep best ExpR)
    3. Adjust ExpR for profit split
    4. Rank by Sharpe/DD ratio
    5. Greedy fill: DD budget -> contract cap -> slot cap -> consistency
    """
    if not strategies:
        return TradingBook(profile.profile_id, [], [])

    firm_spec = get_firm_spec(profile.firm)
    tier = get_account_tier(profile.firm, profile.account_size)
    all_excluded: list[ExcludedEntry] = []

    # 1. Instrument bans (firm-level)
    candidates, banned_excluded = _apply_instrument_bans(strategies, firm_spec.banned_instruments)
    all_excluded.extend(banned_excluded)

    # 1b. Session routing (profile-level — from playbook account grid)
    if profile.allowed_sessions is not None:
        routed = []
        for s in candidates:
            if s.orb_label in profile.allowed_sessions:
                routed.append(s)
            else:
                all_excluded.append(
                    ExcludedEntry(
                        s.strategy_id,
                        s.instrument,
                        s.orb_label,
                        f"Session not assigned to this profile ({s.orb_label})",
                    )
                )
        candidates = routed

    # 1c. Instrument routing (profile-level — e.g. Tradeify=MNQ, TopStep=MGC)
    if profile.allowed_instruments is not None:
        routed = []
        for s in candidates:
            if s.instrument in profile.allowed_instruments:
                routed.append(s)
            else:
                all_excluded.append(
                    ExcludedEntry(
                        s.strategy_id,
                        s.instrument,
                        s.orb_label,
                        f"Instrument not assigned to this profile ({s.instrument})",
                    )
                )
        candidates = routed

    # 2. Deduplicate sessions
    candidates, dedup_excluded = _deduplicate_sessions(candidates)
    all_excluded.extend(dedup_excluded)

    # 3. Rank
    split_factor = compute_profit_split_factor(firm_spec)
    ranked = _rank_strategies(candidates, split_factor)

    # 4. Greedy fill
    dd_budget = tier.max_dd
    dd_per_slot = _compute_dd_per_contract(profile.stop_multiplier, firm_spec.dd_type)
    contract_budget = tier.max_contracts_micro  # All our instruments are micro
    slot_budget = profile.max_slots

    entries: list[TradingBookEntry] = []
    dd_used = 0.0
    contracts_used = 0
    slots_used = 0
    total_effective_expr = 0.0

    for rs in ranked:
        s = rs.strategy
        contracts = 1  # 1 micro per slot (conservative default)

        # DD budget check
        slot_dd = dd_per_slot * contracts
        if dd_used + slot_dd > dd_budget:
            all_excluded.append(
                ExcludedEntry(
                    s.strategy_id,
                    s.instrument,
                    s.orb_label,
                    f"DD budget exhausted (${dd_used:.0f} + ${slot_dd:.0f} > ${dd_budget:.0f})",
                )
            )
            continue

        # Contract cap check
        if contracts_used + contracts > contract_budget:
            all_excluded.append(
                ExcludedEntry(
                    s.strategy_id,
                    s.instrument,
                    s.orb_label,
                    f"Contract cap reached ({contracts_used}/{contract_budget} micro)",
                )
            )
            continue

        # Slot cap check
        if slots_used >= slot_budget:
            all_excluded.append(
                ExcludedEntry(
                    s.strategy_id,
                    s.instrument,
                    s.orb_label,
                    f"Cognitive cap reached ({slots_used}/{slot_budget} slots)",
                )
            )
            continue

        # NOTE: consistency_rule (e.g. TopStep 40%) is a PAYOUT gate on realized
        # daily P&L, not a portfolio construction constraint. It's checked when
        # requesting a payout: "best single day < X% of total profit." Not
        # relevant to strategy selection — left in PropFirmSpec for reference only.

        # Minimum effective ExpR check (split kills the edge?)
        if rs.effective_expr < 0.05:
            all_excluded.append(
                ExcludedEntry(
                    s.strategy_id,
                    s.instrument,
                    s.orb_label,
                    f"Edge too thin after profit split (eff_ExpR={rs.effective_expr:.3f})",
                )
            )
            continue

        # --- ACCEPTED ---
        session_time = _get_session_time_brisbane(s.orb_label)

        entries.append(
            TradingBookEntry(
                strategy_id=s.strategy_id,
                instrument=s.instrument,
                orb_label=s.orb_label,
                session_time_brisbane=session_time,
                entry_model=s.entry_model,
                rr_target=s.rr_target,
                confirm_bars=s.confirm_bars,
                filter_type=s.filter_type,
                direction="both",  # Resolved at trade time from ORB break direction
                contracts=contracts,
                stop_multiplier=profile.stop_multiplier,
                effective_expr=rs.effective_expr,
                sharpe_dd_ratio=rs.expr_dd_ratio,
                dd_contribution=slot_dd,
            )
        )

        dd_used += slot_dd
        contracts_used += contracts
        slots_used += 1
        total_effective_expr += rs.effective_expr

    return TradingBook(profile.profile_id, entries, all_excluded)


def build_all_books(
    strategies_by_instrument: dict[str, list[PortfolioStrategy]],
    profiles: dict[str, AccountProfile] | None = None,
) -> dict[str, TradingBook]:
    """Build trading books for all active profiles.

    strategies_by_instrument: output of build_live_portfolio() per instrument.
    Returns dict of profile_id -> TradingBook.
    """
    if profiles is None:
        profiles = ACCOUNT_PROFILES

    # Pool all strategies cross-instrument
    all_strategies = []
    for instrument_strats in strategies_by_instrument.values():
        all_strategies.extend(instrument_strats)

    books = {}
    for pid, profile in profiles.items():
        if not profile.active:
            continue
        books[pid] = select_for_profile(profile, all_strategies)

    return books


def print_trading_book(book: TradingBook, profile: AccountProfile, verbose: bool = False) -> None:
    """Pretty-print a trading book."""
    firm_spec = get_firm_spec(profile.firm)
    tier = get_account_tier(profile.firm, profile.account_size)

    print(f"\n{'=' * 70}")
    print(f"  {firm_spec.display_name} ${tier.account_size:,} — {profile.profile_id}")
    if profile.copies > 1:
        print(f"  Copies: {profile.copies} (identical accounts)")
    print(f"  DD budget: ${tier.max_dd:,.0f} ({firm_spec.dd_type})")
    print(f"  Stop multiplier: {profile.stop_multiplier}x")
    print(f"  Slot cap: {profile.max_slots}")
    print(f"{'=' * 70}")

    if not book.entries:
        print("  NO STRATEGIES SELECTED")
    else:
        print(
            f"\n  {'Strategy':<45} {'Inst':<5} {'Session':<18} {'Time':<6} "
            f"{'EM':<3} {'RR':<4} {'CB':<3} {'Filter':<16} "
            f"{'EffExpR':<8} {'S/DD':<6} {'DD$':<7}"
        )
        print(
            f"  {'-' * 45} {'-' * 5} {'-' * 18} {'-' * 6} "
            f"{'-' * 3} {'-' * 4} {'-' * 3} {'-' * 16} "
            f"{'-' * 8} {'-' * 6} {'-' * 7}"
        )
        for e in book.entries:
            sid_short = e.strategy_id[:45]
            print(
                f"  {sid_short:<45} {e.instrument:<5} {e.orb_label:<18} "
                f"{e.session_time_brisbane:<6} {e.entry_model:<3} "
                f"{e.rr_target:<4.1f} {e.confirm_bars:<3} {e.filter_type:<16} "
                f"{e.effective_expr:<8.3f} {e.sharpe_dd_ratio:<6.2f} "
                f"${e.dd_contribution:<6.0f}"
            )

    dd_pct = (book.total_dd_used / tier.max_dd * 100) if tier.max_dd > 0 else 0
    print(
        f"\n  SUMMARY: {book.total_slots} slots | "
        f"${book.total_dd_used:,.0f} DD used of ${tier.max_dd:,.0f} "
        f"({dd_pct:.0f}%) | "
        f"{book.total_contracts} contracts"
    )

    if book.excluded:
        if verbose:
            print(f"\n  EXCLUDED ({len(book.excluded)}):")
            for ex in book.excluded:
                print(f"    x {ex.strategy_id[:40]:<40} {ex.instrument:<5} {ex.orb_label:<18} -- {ex.reason}")
        else:
            print(f"  ({len(book.excluded)} excluded — use --verbose to see reasons)")
    print()


def _time_sort_key(time_str: str) -> int:
    """Sort Brisbane times starting from 08:00 (trading day start)."""
    try:
        h, m = int(time_str[:2]), int(time_str[3:5])
    except (ValueError, IndexError):
        return 9999
    return (h * 60 + m - 8 * 60) % (24 * 60)


def _format_time_ampm(time_str: str) -> str:
    """Convert HH:MM to h:MM AM/PM."""
    try:
        h, m = int(time_str[:2]), int(time_str[3:5])
    except (ValueError, IndexError):
        return time_str
    period = "AM" if h < 12 else "PM"
    display_h = h % 12 or 12
    return f"{display_h}:{m:02d} {period}"


def print_daily(
    books: dict[str, TradingBook],
    trading_day: date,
    fitness_results: dict[str, str] | None = None,
) -> None:
    """Print glanceable daily execution card — one line per trade, sorted by time."""
    day_name = trading_day.strftime("%A")
    date_str = trading_day.strftime("%d %b %Y")

    # Collect all trades with firm labels
    all_trades: list[tuple[str, TradingBookEntry, str]] = []  # (time_sort, entry, firm_label)
    for pid, book in books.items():
        profile = ACCOUNT_PROFILES[pid]
        firm_spec = get_firm_spec(profile.firm)
        copies_str = f" x{profile.copies}" if profile.copies > 1 else ""
        short_name = firm_spec.display_name.split()[0]
        label = f"{short_name}{copies_str}"
        if firm_spec.auto_trading == "none":
            label += " manual"
        for entry in book.entries:
            all_trades.append((entry.session_time_brisbane, entry, label))

    all_trades.sort(key=lambda t: _time_sort_key(t[0]))

    print(f"\n  TODAY: {day_name} {date_str}")
    print(f"  {'=' * 68}")

    if not all_trades:
        print("  No trades selected across any profile.")
        print()
        return

    # Header
    print(
        f"  {'Time':<10} {'Session':<16} {'Inst':<5} {'Filter':<16} "
        f"{'RR':<5} {'Entry':<8} {'Account':<18} {'Fitness':<8}"
    )
    print(f"  {'-' * 10} {'-' * 16} {'-' * 5} {'-' * 16} {'-' * 5} {'-' * 8} {'-' * 18} {'-' * 8}")

    warnings = []
    for time_str, entry, firm_label in all_trades:
        time_display = _format_time_ampm(time_str)

        # Fitness lookup
        fitness = "—"
        if fitness_results and entry.strategy_id in fitness_results:
            fitness = fitness_results[entry.strategy_id]
        if fitness in ("DECAY", "UNKNOWN"):
            warnings.append(f"  !! {entry.strategy_id}: {fitness}")

        print(
            f"  {time_display:<10} {entry.orb_label:<16} {entry.instrument:<5} "
            f"{entry.filter_type:<16} {entry.rr_target:<5.1f} "
            f"{entry.entry_model} CB{entry.confirm_bars:<4} "
            f"{firm_label:<18} {fitness:<8}"
        )

    # Summary
    total_slots = sum(b.total_slots for b in books.values())
    total_copies = sum(ACCOUNT_PROFILES[pid].copies for pid in books)
    aggregate_dd = sum(books[pid].total_dd_used * ACCOUNT_PROFILES[pid].copies for pid in books)
    firms_used = len(books)
    print(f"  {'=' * 68}")
    print(f"  {total_slots} trades | {firms_used} firms | {total_copies} accounts | ${aggregate_dd:,.0f} aggregate DD")

    if warnings:
        print()
        for w in warnings:
            print(w)

    print()


def _load_strategies_and_build_books(db_path: Path) -> tuple[dict[str, TradingBook], list[PortfolioStrategy]]:
    """Load active validated strategies from DB and build books for all profiles."""
    import duckdb

    from pipeline.asset_configs import get_active_instruments
    from pipeline.db_config import configure_connection

    print("Loading validated strategies...")
    con = duckdb.connect(str(db_path), read_only=True)
    configure_connection(con)
    try:
        active_instruments = get_active_instruments()
        shelf_relation = deployable_validated_relation(con, alias="vs")
        rows = con.execute(
            f"SELECT * FROM {shelf_relation} WHERE instrument = ANY($1)",
            [list(active_instruments)],
        ).fetchall()
        columns = [desc[0] for desc in con.description]
    finally:
        con.close()

    all_strategies: list[PortfolioStrategy] = []
    counts: dict[str, int] = {}
    for row in rows:
        r = dict(zip(columns, row, strict=False))
        s = PortfolioStrategy(
            strategy_id=r["strategy_id"],
            instrument=r["instrument"],
            orb_label=r["orb_label"],
            entry_model=r["entry_model"],
            rr_target=r["rr_target"],
            confirm_bars=r["confirm_bars"],
            filter_type=r["filter_type"],
            orb_minutes=r.get("orb_minutes", 5),
            expectancy_r=r["expectancy_r"],
            win_rate=r["win_rate"],
            sample_size=r["sample_size"],
            sharpe_ratio=r.get("sharpe_ratio"),
            max_drawdown_r=r.get("max_drawdown_r"),
            median_risk_points=r.get("median_risk_points"),
            stop_multiplier=r.get("stop_multiplier", 1.0),
        )
        all_strategies.append(s)
        counts[s.instrument] = counts.get(s.instrument, 0) + 1

    for inst in sorted(counts):
        print(f"  {inst}: {counts[inst]} active")
    print(f"  Total pool: {len(all_strategies)} strategies across {len(counts)} instruments\n")

    strategies_by_instrument: dict[str, list[PortfolioStrategy]] = {}
    for s in all_strategies:
        strategies_by_instrument.setdefault(s.instrument, []).append(s)
    books = build_all_books(strategies_by_instrument)
    return books, all_strategies


def main() -> None:
    """CLI entry point."""
    import argparse

    from pipeline.paths import GOLD_DB_PATH
    from trading_app.prop_profiles import get_profile

    parser = argparse.ArgumentParser(description="Prop firm portfolio — what to trade, where, when")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=f"Single profile. Available: {', '.join(ACCOUNT_PROFILES.keys())}",
    )
    parser.add_argument("--all", action="store_true", help="Show all active profiles (default)")
    parser.add_argument(
        "--daily", action="store_true", help="Daily execution card — one line per trade, sorted by time"
    )
    parser.add_argument("--summary", action="store_true", help="Cross-account summary")
    parser.add_argument("--verbose", action="store_true", help="Show excluded strategies and reasons")
    parser.add_argument("--fitness", action="store_true", help="Check fitness status per strategy (slower)")
    parser.add_argument("--date", type=str, default=None, help="Trading date YYYY-MM-DD (default: today)")
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    if not args.profile and not args.all and not args.daily:
        args.daily = True  # Default to the useful view

    from pathlib import Path

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    trading_day = date.fromisoformat(args.date) if args.date else date.today()

    # Fitness check (optional — adds ~2s)
    fitness_results: dict[str, str] | None = None
    if args.fitness or args.daily:
        fitness_results = {}

    if args.daily:
        if args.profile:
            # Single profile with pinned or JSON-sourced daily lanes
            profile = get_profile(args.profile)
            if effective_daily_lanes(profile):
                lanes = resolve_daily_lanes(profile, db_path=db_path, trading_day=trading_day)
                print_daily_lanes(profile, lanes, trading_day, db_path=db_path)
                return
            # Fallback: profile has no pinned lanes, use dynamic
        # Cross-account daily view (all firms, sorted by time)
        books, _ = _load_strategies_and_build_books(db_path)
        if fitness_results is not None:
            all_sids = [e.strategy_id for b in books.values() for e in b.entries]
            for sid in all_sids:
                try:
                    f = compute_fitness(sid, db_path=db_path)
                    fitness_results[sid] = f.fitness_status
                except (ValueError, duckdb.Error):
                    fitness_results[sid] = "UNKNOWN"
        print_daily(books, trading_day, fitness_results)
        return

    if args.profile:
        books, all_strategies = _load_strategies_and_build_books(db_path)
        profile = get_profile(args.profile)
        book = select_for_profile(profile, all_strategies)
        print_trading_book(book, profile, verbose=args.verbose)
    else:
        books, _ = _load_strategies_and_build_books(db_path)
        for pid, book in books.items():
            print_trading_book(book, ACCOUNT_PROFILES[pid], verbose=args.verbose)

        if args.summary:
            print(f"\n{'=' * 70}")
            print("  CROSS-ACCOUNT SUMMARY")
            print(f"{'=' * 70}")
            total_slots = sum(b.total_slots for b in books.values())
            total_dd = sum(b.total_dd_used for b in books.values())
            all_instruments: set[str] = set()
            all_sessions: set[str] = set()
            for b in books.values():
                all_instruments.update(b.instruments_used)
                all_sessions.update(b.sessions_used)
            print(f"  Active profiles: {len(books)}")
            print(f"  Total slots: {total_slots}")
            print(f"  Total DD exposure: ${total_dd:,.0f}")
            print(f"  Instruments: {', '.join(sorted(all_instruments))}")
            print(f"  Sessions: {', '.join(sorted(all_sessions))}")
            total_copies = sum(ACCOUNT_PROFILES[pid].copies for pid in books)
            if total_copies > len(books):
                aggregate_dd = sum(books[pid].total_dd_used * ACCOUNT_PROFILES[pid].copies for pid in books)
                print(f"  Account copies: {total_copies} (${aggregate_dd:,.0f} aggregate DD)")
            print()


if __name__ == "__main__":
    main()
