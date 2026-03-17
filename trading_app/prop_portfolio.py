"""
Prop firm portfolio selection.

Selects optimal strategy subset from validated universe for a specific
prop firm account profile. Enforces DD budget, contract caps, cognitive
load limits, and consistency rules.

Also provides a narrow daily execution view for manual profiles. The daily
view resolves exact profile-owned lanes from validated_setups and fitness,
instead of inferring "best available" rows from the broader live universe.

Usage:
    python -m trading_app.prop_portfolio --profile topstep_50k
    python -m trading_app.prop_portfolio --profile apex_50k_manual --daily
    python -m trading_app.prop_portfolio --all
    python -m trading_app.prop_portfolio --summary
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb

from pipeline.cost_model import get_cost_spec
from pipeline.db_config import configure_connection
from pipeline.dst import SESSION_CATALOG
from trading_app.portfolio import PortfolioStrategy
from trading_app.prop_profiles import (
    ACCOUNT_PROFILES,
    AccountProfile,
    DailyLaneSpec,
    ExcludedEntry,
    TradingBook,
    TradingBookEntry,
    compute_profit_split_factor,
    get_account_tier,
    get_firm_spec,
)
from trading_app.strategy_fitness import compute_fitness

# =========================================================================
# DD estimation constants (from Monte Carlo sim — trading_plan_sim.md)
# =========================================================================

# Median max DD per contract at 0.75x stop (EOD trailing)
DD_PER_CONTRACT_075X = 935.0
# Median max DD per contract at 1.0x stop (EOD trailing)
DD_PER_CONTRACT_10X = 1350.0
# Intraday trailing is stricter: unrealized PnL moves floor in real-time.
# Factor applied to EOD estimate. Conservative 1.4x (40% more DD risk).
INTRADAY_TRAILING_FACTOR = 1.4


def _compute_dd_per_contract(stop_multiplier: float, dd_type: str) -> float:
    """Estimated median max DD contribution per contract.

    Based on Monte Carlo simulation results (see trading_plan_sim.md).
    """
    if stop_multiplier <= 0.75:
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


@dataclass(frozen=True)
class DailyExecutionLane:
    """Resolved daily execution row for a profile-owned manual lane."""

    strategy_id: str
    instrument: str
    orb_label: str
    session_time_brisbane: str
    status: str  # TRADE | HOLD | REVIEW | SKIP
    reason: str
    entry_model: str | None = None
    rr_target: float | None = None
    confirm_bars: int | None = None
    filter_type: str | None = None
    orb_minutes: int | None = None
    strategy_stop_multiplier: float | None = None
    planned_stop_multiplier: float | None = None
    fitness_status: str = "UNKNOWN"
    expectancy_r: float | None = None
    exp_dollars: float | None = None
    sample_size: int | None = None
    execution_notes: str = ""


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


def _parse_trading_day(raw: str | None) -> date:
    """Parse CLI trading day. Defaults to today."""
    if raw is None:
        return date.today()
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _calendar_gate_reason(filter_type: str | None, trading_day: date) -> str | None:
    """Return skip reason for simple weekday filters."""
    if not filter_type:
        return None
    weekday = trading_day.weekday()
    if "NOMON" in filter_type and weekday == 0:
        return "Calendar filter active: skip Monday."
    if "NOTUE" in filter_type and weekday == 1:
        return "Calendar filter active: skip Tuesday."
    if "NOFRI" in filter_type and weekday == 4:
        return "Calendar filter active: skip Friday."
    return None


def _compute_expectancy_dollars(
    instrument: str,
    expectancy_r: float | None,
    median_risk_points: float | None,
) -> float | None:
    """Convert ExpR into dollars per trade when risk is available."""
    if expectancy_r is None or median_risk_points is None:
        return None
    try:
        cost_spec = get_cost_spec(instrument)
    except Exception:
        return None
    return expectancy_r * median_risk_points * cost_spec.point_value


def _load_daily_strategy_snapshot(db_path: Path, strategy_id: str) -> dict | None:
    """Load the exact validated row for a manual daily lane."""
    con = duckdb.connect(str(db_path), read_only=True)
    configure_connection(con, writing=False)
    try:
        row = con.execute(
            """
            SELECT vs.strategy_id,
                   vs.instrument,
                   vs.orb_label,
                   vs.entry_model,
                   vs.rr_target,
                   vs.confirm_bars,
                   vs.filter_type,
                   vs.orb_minutes,
                   COALESCE(vs.stop_multiplier, 1.0) AS stop_multiplier,
                   vs.expectancy_r,
                   vs.win_rate,
                   vs.sample_size,
                   LOWER(vs.status) AS status,
                   es.median_risk_points
            FROM validated_setups vs
            LEFT JOIN experimental_strategies es
              ON vs.strategy_id = es.strategy_id
            WHERE vs.strategy_id = ?
            LIMIT 1
        """,
            [strategy_id],
        ).fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in con.description]
        return dict(zip(cols, row, strict=False))
    finally:
        con.close()


def _resolve_daily_lane(
    profile: AccountProfile,
    lane: DailyLaneSpec,
    db_path: Path,
    trading_day: date,
) -> DailyExecutionLane:
    """Resolve one exact daily lane against current DB + fitness state."""
    session_time = _get_session_time_brisbane(lane.orb_label, trading_day)
    planned_stop = lane.planned_stop_multiplier if lane.planned_stop_multiplier is not None else profile.stop_multiplier
    snapshot = _load_daily_strategy_snapshot(db_path, lane.strategy_id)
    if snapshot is None:
        return DailyExecutionLane(
            strategy_id=lane.strategy_id,
            instrument=lane.instrument,
            orb_label=lane.orb_label,
            session_time_brisbane=session_time,
            status="HOLD",
            reason="Strategy missing from validated_setups.",
            planned_stop_multiplier=planned_stop,
            execution_notes=lane.execution_notes,
        )

    fitness_status = "UNKNOWN"
    fitness_note = ""
    try:
        fitness = compute_fitness(snapshot["strategy_id"], db_path=db_path)
        fitness_status = fitness.fitness_status
        fitness_note = fitness.fitness_notes
    except Exception as exc:
        fitness_note = f"Fitness lookup failed: {type(exc).__name__}: {exc}"

    status = "TRADE"
    reason = "Manual lane is ready."
    firm_spec = get_firm_spec(profile.firm)
    strategy_stop = snapshot["stop_multiplier"]

    if snapshot["status"] != "active":
        status = "HOLD"
        reason = f"Strategy status is {snapshot['status']}."
    elif snapshot["instrument"] in firm_spec.banned_instruments:
        status = "HOLD"
        reason = f"Instrument banned by firm ({snapshot['instrument']})."
    elif profile.allowed_sessions is not None and snapshot["orb_label"] not in profile.allowed_sessions:
        status = "HOLD"
        reason = f"Session not assigned to profile ({snapshot['orb_label']})."
    elif profile.allowed_instruments is not None and snapshot["instrument"] not in profile.allowed_instruments:
        status = "HOLD"
        reason = f"Instrument not assigned to profile ({snapshot['instrument']})."
    elif snapshot["instrument"] != lane.instrument or snapshot["orb_label"] != lane.orb_label:
        status = "REVIEW"
        reason = "Manual lane spec does not match the resolved validated row."
    else:
        calendar_reason = _calendar_gate_reason(snapshot["filter_type"], trading_day)
        if calendar_reason is not None:
            status = "SKIP"
            reason = calendar_reason
        elif fitness_note.startswith("Fitness lookup failed:"):
            status = "REVIEW"
            reason = fitness_note
        elif fitness_status not in lane.required_fitness:
            status = "HOLD"
            reason = f"Fitness {fitness_status}: {fitness_note}"
        elif abs(strategy_stop - planned_stop) > 1e-9:
            status = "REVIEW"
            reason = f"Manual plan stop {planned_stop:.2f}x differs from validated row {strategy_stop:.2f}x."

    return DailyExecutionLane(
        strategy_id=snapshot["strategy_id"],
        instrument=snapshot["instrument"],
        orb_label=snapshot["orb_label"],
        session_time_brisbane=session_time,
        status=status,
        reason=reason,
        entry_model=snapshot["entry_model"],
        rr_target=snapshot["rr_target"],
        confirm_bars=snapshot["confirm_bars"],
        filter_type=snapshot["filter_type"],
        orb_minutes=snapshot["orb_minutes"],
        strategy_stop_multiplier=strategy_stop,
        planned_stop_multiplier=planned_stop,
        fitness_status=fitness_status,
        expectancy_r=snapshot["expectancy_r"],
        exp_dollars=_compute_expectancy_dollars(
            snapshot["instrument"],
            snapshot["expectancy_r"],
            snapshot["median_risk_points"],
        ),
        sample_size=snapshot["sample_size"],
        execution_notes=lane.execution_notes,
    )


def _session_sort_key(session_time: str) -> int:
    """Sort HH:MM strings using an 08:00 Brisbane trading-day start."""
    if session_time == "unknown":
        return 10_000
    hour, minute = [int(part) for part in session_time.split(":")]
    return (hour * 60 + minute - 8 * 60) % (24 * 60)


def resolve_daily_execution_lanes(
    profile: AccountProfile,
    db_path: Path,
    trading_day: date,
) -> list[DailyExecutionLane]:
    """Resolve all exact daily lanes for a manual-facing profile."""
    if not profile.daily_lanes:
        raise ValueError(f"Profile {profile.profile_id} has no daily lanes configured.")
    lanes = [_resolve_daily_lane(profile, lane, db_path, trading_day) for lane in profile.daily_lanes]
    return sorted(lanes, key=lambda lane: _session_sort_key(lane.session_time_brisbane))


def print_daily_execution_sheet(
    profile: AccountProfile,
    lanes: list[DailyExecutionLane],
    trading_day: date,
) -> None:
    """Pretty-print the manual daily execution surface."""
    firm_spec = get_firm_spec(profile.firm)
    print(f"\n{'=' * 118}")
    print("  MANUAL DAILY EXECUTION SHEET [manual canonical]")
    print(
        f"  Profile: {profile.profile_id} | Firm: {firm_spec.display_name} | "
        f"Date: {trading_day.isoformat()} | Brisbane timezone"
    )
    print("  Source: prop_profiles.py daily_lanes + validated_setups + strategy_fitness")
    print(f"{'=' * 118}")

    print(
        f"\n  {'Status':<8} {'Time':<6} {'Session':<18} {'Inst':<5} {'ORB':<4} "
        f"{'RR':<4} {'Stop':<9} {'Fit':<8} {'Strategy':<44}"
    )
    print(f"  {'-' * 8} {'-' * 6} {'-' * 18} {'-' * 5} {'-' * 4} {'-' * 4} {'-' * 9} {'-' * 8} {'-' * 44}")

    for lane in lanes:
        rr_label = f"{lane.rr_target:.1f}" if lane.rr_target is not None else "-"
        orb_label = f"{lane.orb_minutes}m" if lane.orb_minutes is not None else "-"
        stop_label = f"{lane.planned_stop_multiplier:.2f}x" if lane.planned_stop_multiplier is not None else "-"
        print(
            f"  {lane.status:<8} {lane.session_time_brisbane:<6} {lane.orb_label:<18} {lane.instrument:<5} "
            f"{orb_label:<4} {rr_label:<4} {stop_label:<9} {lane.fitness_status:<8} {lane.strategy_id[:44]:<44}"
        )

        details = []
        if lane.filter_type:
            details.append(f"filter={lane.filter_type}")
        if lane.expectancy_r is not None:
            details.append(f"ExpR={lane.expectancy_r:+.3f}")
        if lane.exp_dollars is not None:
            details.append(f"Exp$={lane.exp_dollars:+.2f}")
        if lane.sample_size is not None:
            details.append(f"N={lane.sample_size}")
        if lane.entry_model is not None and lane.confirm_bars is not None:
            details.append(f"{lane.entry_model} CB{lane.confirm_bars}")
        if lane.strategy_stop_multiplier is not None:
            details.append(f"validated_stop={lane.strategy_stop_multiplier:.2f}x")
        if details:
            print(f"    {' | '.join(details)}")
        print(f"    action: {lane.reason}")
        if lane.execution_notes:
            print(f"    notes: {lane.execution_notes}")
    print()


def select_for_profile(
    profile: AccountProfile,
    strategies: list[PortfolioStrategy],
) -> TradingBook:
    """Select optimal strategy subset for an account profile.

    Algorithm:
    1. Filter banned instruments + session/instrument routing
    2. Deduplicate session x instrument (keep best ExpR)
    3. Adjust ExpR for profit split
    4. Rank by ExpR/DD ratio
    5. Greedy fill: DD budget -> contract cap -> slot cap
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


def print_trading_book(book: TradingBook, profile: AccountProfile) -> None:
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
        print(f"\n  EXCLUDED ({len(book.excluded)}):")
        for ex in book.excluded:
            print(f"    x {ex.strategy_id[:40]:<40} {ex.instrument:<5} {ex.orb_label:<18} -- {ex.reason}")
    print()


def main() -> None:
    """CLI entry point."""
    import argparse

    from pipeline.asset_configs import get_active_instruments
    from pipeline.paths import GOLD_DB_PATH
    from trading_app.live_config import build_live_portfolio
    from trading_app.prop_profiles import get_profile

    parser = argparse.ArgumentParser(description="Build prop firm trading books from validated strategies")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=f"Profile ID. Available: {', '.join(ACCOUNT_PROFILES.keys())}",
    )
    parser.add_argument("--all", action="store_true", help="Build all active profiles")
    parser.add_argument("--summary", action="store_true", help="Cross-account summary")
    parser.add_argument("--daily", action="store_true", help="Show the manual daily execution sheet")
    parser.add_argument("--date", type=str, default=None, help="Trading day in YYYY-MM-DD (default: today)")
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    if args.daily and args.all:
        raise SystemExit("--daily cannot be combined with --all")
    if args.daily and args.summary:
        raise SystemExit("--daily cannot be combined with --summary")
    if args.daily and not args.profile:
        args.profile = "apex_50k_manual"
    if not args.profile and not args.all:
        args.all = True  # Default: show all

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    trading_day = _parse_trading_day(args.date)

    if args.daily:
        profile = get_profile(args.profile)
        lanes = resolve_daily_execution_lanes(profile, db_path=db_path, trading_day=trading_day)
        print_daily_execution_sheet(profile, lanes, trading_day)
        return

    # Build eligible strategies for each instrument
    print("Loading validated strategies...")
    all_strategies: list[PortfolioStrategy] = []
    for instrument in get_active_instruments():
        portfolio, _notes = build_live_portfolio(db_path=db_path, instrument=instrument)
        print(f"  {instrument}: {len(portfolio.strategies)} eligible")
        all_strategies.extend(portfolio.strategies)
    print(f"  Total pool: {len(all_strategies)} strategies across all instruments")

    if args.profile:
        profile = get_profile(args.profile)
        book = select_for_profile(profile, all_strategies)
        print_trading_book(book, profile)
    else:
        # Use build_all_books to avoid duplicating profile iteration logic
        strategies_by_instrument: dict[str, list[PortfolioStrategy]] = {}
        # Re-key by instrument from the flat list
        for s in all_strategies:
            strategies_by_instrument.setdefault(s.instrument, []).append(s)
        books = build_all_books(strategies_by_instrument)
        for pid, book in books.items():
            print_trading_book(book, ACCOUNT_PROFILES[pid])

        if args.summary and books:
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
