"""
Prop firm portfolio profiles — configuration and data structures.

Three layers:
1. PROP_FIRM_SPECS: static firm rules (verified from firm websites, Mar 2026)
2. ACCOUNT_TIERS: account size -> DD/contract limits
3. ACCOUNT_PROFILES: user's actual accounts (editable)

Pattern follows COST_SPECS in pipeline/cost_model.py — canonical source of truth,
imported everywhere, easy to edit.
"""

from __future__ import annotations

from dataclasses import dataclass

# =========================================================================
# Data structures
# =========================================================================


@dataclass(frozen=True)
class PropFirmSpec:
    """Static rules for a prop firm. Changes rarely (maybe yearly)."""

    name: str
    display_name: str
    dd_type: str  # "eod_trailing" | "intraday_trailing" | "none"
    # Profit split tiers: list of (cumulative_profit_threshold, trader_pct).
    # Evaluated in order — first tier where cumulative < threshold applies.
    # Example: [(5000, 0.50), (float("inf"), 0.90)] = 50% up to $5K, 90% after.
    profit_split_tiers: tuple[tuple[float, float], ...]
    consistency_rule: float | None  # None or max pct (0.40 = best day <= 40%)
    news_restriction: bool
    close_time_et: str  # "16:00" etc.
    platform: str  # "topstepx" | "tradovate" | "ninjatrader" | "any"
    min_hold_seconds: int | None  # Tradeify: 10s microscalp rule
    banned_instruments: frozenset[str]  # e.g. frozenset({"MGC", "GC"})
    auto_trading: str  # "full" | "semi" | "none"
    notes: str = ""


@dataclass(frozen=True)
class PropFirmAccount:
    """Account tier at a specific firm. One per (firm, account_size)."""

    firm: str
    account_size: int
    max_dd: float
    max_contracts_mini: int
    max_contracts_micro: int
    daily_loss_limit: float | None = None  # None = no DLL


@dataclass(frozen=True)
class AccountProfile:
    """User's actual trading account. Editable."""

    profile_id: str
    firm: str
    account_size: int
    copies: int = 1  # Number of identical accounts (e.g. x3 TopStep)
    stop_multiplier: float = 0.75  # 0.75 for prop, 1.0 for self-funded
    max_slots: int = 6  # Cognitive cap
    active: bool = True
    notes: str = ""


@dataclass(frozen=True)
class TradingBookEntry:
    """A strategy selected for a specific account profile."""

    strategy_id: str
    instrument: str
    orb_label: str
    session_time_brisbane: str
    entry_model: str
    rr_target: float
    confirm_bars: int
    filter_type: str
    direction: str
    contracts: int
    stop_multiplier: float
    effective_expr: float  # ExpR after profit-split adjustment
    sharpe_dd_ratio: float  # Ranking score
    dd_contribution: float  # Estimated DD this slot adds ($)


@dataclass(frozen=True)
class ExcludedEntry:
    """A strategy excluded from a profile, with reason."""

    strategy_id: str
    instrument: str
    orb_label: str
    reason: str  # Human-readable reason


@dataclass
class TradingBook:
    """Complete trading book for one account profile."""

    profile_id: str
    entries: list[TradingBookEntry]
    excluded: list[ExcludedEntry]

    @property
    def total_slots(self) -> int:
        return len(self.entries)

    @property
    def total_dd_used(self) -> float:
        return sum(e.dd_contribution for e in self.entries)

    @property
    def total_contracts(self) -> int:
        return sum(e.contracts for e in self.entries)

    @property
    def instruments_used(self) -> set[str]:
        return {e.instrument for e in self.entries}

    @property
    def sessions_used(self) -> set[str]:
        return {e.orb_label for e in self.entries}


# =========================================================================
# Verified firm specs (March 2026)
# Sources in docs/plans/2026-03-15-prop-portfolio-design.md
# =========================================================================

PROP_FIRM_SPECS: dict[str, PropFirmSpec] = {
    "topstep": PropFirmSpec(
        name="topstep",
        display_name="TopStep",
        dd_type="eod_trailing",
        profit_split_tiers=((5_000, 0.50), (float("inf"), 0.90)),
        consistency_rule=0.40,
        news_restriction=False,
        close_time_et="16:00",
        platform="topstepx",
        min_hold_seconds=None,
        banned_instruments=frozenset(),
        auto_trading="full",
        notes="TopstepX API for automation. 5 min winning days @$200+ for payout.",
    ),
    "mffu": PropFirmSpec(
        name="mffu",
        display_name="MyFundedFutures",
        dd_type="eod_trailing",  # Core/Pro are EOD; Rapid is intraday (handled at tier level)
        profit_split_tiers=((float("inf"), 0.80),),  # Core/Pro default; Rapid overrides at tier
        consistency_rule=0.40,  # Core funded; Rapid/Pro have none
        news_restriction=True,  # Flat 2 min before/after Tier 1
        close_time_et="16:10",
        platform="tradovate",
        min_hold_seconds=None,
        banned_instruments=frozenset(),
        auto_trading="semi",  # Own strategies with active monitoring
        notes="News restriction: flat 2min before/after FOMC/CPI/NFP. Contract limits TBD.",
    ),
    "tradeify": PropFirmSpec(
        name="tradeify",
        display_name="Tradeify",
        dd_type="eod_trailing",  # EOD trailing -> static lock at starting balance
        profit_split_tiers=((float("inf"), 0.90),),  # Select Flex: flat 90/10
        consistency_rule=None,  # Select Flex: no consistency rule when funded
        news_restriction=False,
        close_time_et="16:00",
        platform="tradovate",
        min_hold_seconds=10,  # 50% of trades held 10+ seconds
        banned_instruments=frozenset(),
        auto_trading="full",  # Own bots allowed on Select
        notes="DD locks to static at starting_balance+$100. 10s hold rule. No weekend holds.",
    ),
    "apex": PropFirmSpec(
        name="apex",
        display_name="Apex Trader Funding",
        dd_type="eod_trailing",
        profit_split_tiers=((float("inf"), 1.00),),  # 100% split on recent plans
        consistency_rule=0.50,
        news_restriction=False,
        close_time_et="16:59",
        platform="tradovate",
        min_hold_seconds=None,
        banned_instruments=frozenset({"MGC", "GC", "SI", "SIL", "HG", "PL", "PA"}),
        auto_trading="semi",  # Copy trading only, no independent bots
        notes="ALL METALS SUSPENDED as of Mar 2026. No return date.",
    ),
    "self_funded": PropFirmSpec(
        name="self_funded",
        display_name="Self-Funded",
        dd_type="none",
        profit_split_tiers=((float("inf"), 1.00),),
        consistency_rule=None,
        news_restriction=False,
        close_time_et="none",
        platform="any",
        min_hold_seconds=None,
        banned_instruments=frozenset(),
        auto_trading="full",
        notes="Your own capital. DD is temporary, not fatal.",
    ),
}


ACCOUNT_TIERS: dict[tuple[str, int], PropFirmAccount] = {
    # TopStep
    ("topstep", 50_000): PropFirmAccount("topstep", 50_000, 2_000, 5, 50),
    ("topstep", 100_000): PropFirmAccount("topstep", 100_000, 3_000, 10, 100),
    ("topstep", 150_000): PropFirmAccount("topstep", 150_000, 4_500, 15, 150),
    # MFFU Core (EOD, 3% DD, 80/20)
    ("mffu", 50_000): PropFirmAccount("mffu", 50_000, 1_500, 5, 50),
    ("mffu", 100_000): PropFirmAccount("mffu", 100_000, 3_000, 8, 80),
    ("mffu", 150_000): PropFirmAccount("mffu", 150_000, 4_500, 12, 120),
    # Tradeify Select
    ("tradeify", 50_000): PropFirmAccount("tradeify", 50_000, 2_000, 4, 40),
    ("tradeify", 100_000): PropFirmAccount("tradeify", 100_000, 4_000, 8, 80),
    ("tradeify", 150_000): PropFirmAccount("tradeify", 150_000, 6_000, 12, 120),
    # Apex (metals banned — included for completeness)
    ("apex", 50_000): PropFirmAccount("apex", 50_000, 1_500, 4, 40),
    ("apex", 100_000): PropFirmAccount("apex", 100_000, 3_000, 6, 60),
    ("apex", 150_000): PropFirmAccount("apex", 150_000, 4_500, 9, 90),
    # Self-funded
    ("self_funded", 50_000): PropFirmAccount("self_funded", 50_000, 5_000, 50, 500),
}


# =========================================================================
# User account profiles (EDITABLE)
# =========================================================================

ACCOUNT_PROFILES: dict[str, AccountProfile] = {
    "topstep_50k": AccountProfile(
        profile_id="topstep_50k",
        firm="topstep",
        account_size=50_000,
        copies=1,
        stop_multiplier=0.75,
        max_slots=6,
        notes="Phase 1: single account, auto via TopstepX API",
    ),
    "topstep_150k": AccountProfile(
        profile_id="topstep_150k",
        firm="topstep",
        account_size=150_000,
        copies=1,
        stop_multiplier=0.75,
        max_slots=8,
        notes="Phase 2: scale-up",
    ),
    "tradeify_50k": AccountProfile(
        profile_id="tradeify_50k",
        firm="tradeify",
        account_size=50_000,
        copies=1,
        stop_multiplier=0.75,
        max_slots=6,
        notes="Alternative: 90/10 flat split, DD locks to static",
    ),
    "mffu_50k": AccountProfile(
        profile_id="mffu_50k",
        firm="mffu",
        account_size=50_000,
        copies=1,
        stop_multiplier=0.75,
        max_slots=5,
        notes="Tightest DD ($1,500). News restriction active.",
    ),
    "self_funded_50k": AccountProfile(
        profile_id="self_funded_50k",
        firm="self_funded",
        account_size=50_000,
        copies=1,
        stop_multiplier=1.0,
        max_slots=10,
        notes="Own capital. DD=temporary. 1.0x stops.",
    ),
}


# =========================================================================
# Lookup helpers
# =========================================================================


def get_firm_spec(firm: str) -> PropFirmSpec:
    """Look up firm spec by name. Raises KeyError if not found."""
    return PROP_FIRM_SPECS[firm]


def get_account_tier(firm: str, account_size: int) -> PropFirmAccount:
    """Look up account tier. Raises KeyError if not found."""
    return ACCOUNT_TIERS[(firm, account_size)]


def get_profile(profile_id: str) -> AccountProfile:
    """Look up account profile. Raises KeyError if not found."""
    return ACCOUNT_PROFILES[profile_id]


def compute_profit_split_factor(firm_spec: PropFirmSpec, cumulative_profit: float = 0.0) -> float:
    """Return the trader's effective split percentage (0.0-1.0).

    Evaluates profit_split_tiers in order. First tier where
    cumulative_profit < threshold applies.
    """
    for threshold, pct in firm_spec.profit_split_tiers:
        if cumulative_profit < threshold:
            return pct
    # Fallback: last tier
    return firm_spec.profit_split_tiers[-1][1]
