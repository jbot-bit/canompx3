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

from dataclasses import dataclass
from datetime import date

from pipeline.dst import SESSION_CATALOG
from trading_app.portfolio import PortfolioStrategy
from trading_app.prop_profiles import (
    ACCOUNT_PROFILES,
    AccountProfile,
    ExcludedEntry,
    TradingBook,
    TradingBookEntry,
    compute_profit_split_factor,
    get_account_tier,
    get_firm_spec,
)


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


def _compute_dd_per_contract(
    stop_multiplier: float, dd_type: str
) -> float:
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
            excluded.append(ExcludedEntry(
                s.strategy_id, s.instrument, s.orb_label,
                f"Instrument banned by firm ({s.instrument})",
            ))
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
            s.strategy_id, s.instrument, s.orb_label,
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
    sharpe_dd_ratio: float


def _rank_strategies(
    strategies: list[PortfolioStrategy],
    split_factor: float,
) -> list[_RankedStrategy]:
    """Rank strategies by Sharpe/DD ratio.

    ExpR is adjusted by profit split factor for effective ranking.
    """
    ranked = []
    for s in strategies:
        effective_expr = s.expectancy_r * split_factor
        sharpe = s.sharpe_ratio or 0.0
        dd = s.max_drawdown_r or 999.0
        ratio = sharpe / dd if dd > 0 else 0.0
        ranked.append(_RankedStrategy(s, effective_expr, ratio))

    ranked.sort(key=lambda r: r.sharpe_dd_ratio, reverse=True)
    return ranked


def _get_session_time_brisbane(orb_label: str) -> str:
    """Look up session time in Brisbane timezone from SESSION_CATALOG."""
    entry = SESSION_CATALOG.get(orb_label)
    if entry is None:
        return "unknown"
    resolver = entry.get("resolver")
    if resolver is None:
        return "unknown"
    h, m = resolver(date.today())
    return f"{h:02d}:{m:02d}"


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

    # 1. Instrument bans
    candidates, banned_excluded = _apply_instrument_bans(
        strategies, firm_spec.banned_instruments
    )
    all_excluded.extend(banned_excluded)

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
            all_excluded.append(ExcludedEntry(
                s.strategy_id, s.instrument, s.orb_label,
                f"DD budget exhausted (${dd_used:.0f} + ${slot_dd:.0f} > ${dd_budget:.0f})",
            ))
            continue

        # Contract cap check
        if contracts_used + contracts > contract_budget:
            all_excluded.append(ExcludedEntry(
                s.strategy_id, s.instrument, s.orb_label,
                f"Contract cap reached ({contracts_used}/{contract_budget} micro)",
            ))
            continue

        # Slot cap check
        if slots_used >= slot_budget:
            all_excluded.append(ExcludedEntry(
                s.strategy_id, s.instrument, s.orb_label,
                f"Cognitive cap reached ({slots_used}/{slot_budget} slots)",
            ))
            continue

        # Minimum effective ExpR check (split kills the edge?)
        if rs.effective_expr < 0.05:
            all_excluded.append(ExcludedEntry(
                s.strategy_id, s.instrument, s.orb_label,
                f"Edge too thin after profit split (eff_ExpR={rs.effective_expr:.3f})",
            ))
            continue

        # --- ACCEPTED ---
        session_time = _get_session_time_brisbane(s.orb_label)

        entries.append(TradingBookEntry(
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
            sharpe_dd_ratio=rs.sharpe_dd_ratio,
            dd_contribution=slot_dd,
        ))

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

    print(
        f"\n  SUMMARY: {book.total_slots} slots | "
        f"${book.total_dd_used:,.0f} DD used of ${tier.max_dd:,.0f} "
        f"({book.total_dd_used / tier.max_dd * 100:.0f}%) | "
        f"{book.total_contracts} contracts"
    )

    if book.excluded:
        print(f"\n  EXCLUDED ({len(book.excluded)}):")
        for ex in book.excluded:
            print(
                f"    x {ex.strategy_id[:40]:<40} {ex.instrument:<5} "
                f"{ex.orb_label:<18} -- {ex.reason}"
            )
    print()


def main() -> None:
    """CLI entry point."""
    import argparse

    from pipeline.asset_configs import get_active_instruments
    from pipeline.paths import GOLD_DB_PATH
    from trading_app.live_config import build_live_portfolio
    from trading_app.prop_profiles import get_profile

    parser = argparse.ArgumentParser(
        description="Build prop firm trading books from validated strategies"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=f"Profile ID. Available: {', '.join(ACCOUNT_PROFILES.keys())}",
    )
    parser.add_argument("--all", action="store_true", help="Build all active profiles")
    parser.add_argument(
        "--summary", action="store_true", help="Cross-account summary"
    )
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    if not args.profile and not args.all:
        args.all = True  # Default: show all

    from pathlib import Path

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH

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
        books = {}
        for pid, profile in ACCOUNT_PROFILES.items():
            if not profile.active:
                continue
            book = select_for_profile(profile, all_strategies)
            books[pid] = book
            print_trading_book(book, profile)

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
                print(
                    f"  Account copies: {total_copies} "
                    f"(${total_dd * total_copies / len(books):,.0f} aggregate DD)"
                )
            print()


if __name__ == "__main__":
    main()
