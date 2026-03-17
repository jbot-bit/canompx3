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
class DailyLaneSpec:
    """Exact daily execution lane for a manual profile."""

    strategy_id: str
    instrument: str
    orb_label: str
    execution_notes: str = ""
    planned_stop_multiplier: float | None = None
    required_fitness: tuple[str, ...] = ("FIT",)


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
    # Session→firm routing (from playbook account grid).
    # None = all sessions allowed. Set = only these sessions eligible.
    allowed_sessions: frozenset[str] | None = None
    # Instrument routing. None = all allowed (firm bans still apply).
    allowed_instruments: frozenset[str] | None = None
    # Exact daily lanes for manual profiles. Empty = use dynamic selection.
    daily_lanes: tuple[DailyLaneSpec, ...] = ()
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
        # UNRESOLVED per playbook — 50/90 tiered believed correct but not re-verified. Conservative for ranking.
        profit_split_tiers=((5_000, 0.50), (float("inf"), 0.90)),
        consistency_rule=0.40,
        news_restriction=False,
        close_time_et="16:00",
        platform="topstepx",
        min_hold_seconds=None,
        banned_instruments=frozenset(),
        auto_trading="full",
        notes="MGC morning lane. 5 Express + 1 Live (stay Express). ProjectX API. Copier on Express only.",
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
        auto_trading="full",  # Own bots allowed on Select (exclusive ownership, no cross-firm)
        notes="PRIMARY MNQ scaling lane. 5 accts, Tradovate API. Bot must be exclusive (no cross-firm). Group Trading broken for brackets — use API.",
    ),
    "apex": PropFirmSpec(
        name="apex",
        display_name="Apex Trader Funding",
        dd_type="eod_trailing",
        profit_split_tiers=((float("inf"), 1.00),),  # 100% split on recent plans
        consistency_rule=0.30,  # Underwrite to stricter 30% (compliance page) over 50% (EOD payout page) per playbook
        news_restriction=False,
        close_time_et="16:59",
        platform="tradovate",
        min_hold_seconds=None,
        banned_instruments=frozenset({"MGC", "GC", "SI", "SIL", "HG", "PL", "PA"}),
        auto_trading="none",  # PROHIBITED — PA Compliance: no bots, no copy trading, manual only
        notes="Manual proof only (1 account). Automation AND copy trading PROHIBITED on PA/Live. Metals suspended.",
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
    ("apex", 50_000): PropFirmAccount("apex", 50_000, 2_000, 4, 40),  # Official: $2K DD per rules doc
    ("apex", 100_000): PropFirmAccount("apex", 100_000, 3_000, 6, 60),
    ("apex", 150_000): PropFirmAccount("apex", 150_000, 4_000, 10, 100),  # Official: $4K DD, 10 mini
    # Self-funded
    ("self_funded", 50_000): PropFirmAccount("self_funded", 50_000, 5_000, 50, 500),
}


# =========================================================================
# User account profiles (EDITABLE)
# =========================================================================

ACCOUNT_PROFILES: dict[str, AccountProfile] = {
    # =========================================================================
    # Phase 1: Manual proof (1 account, prove the edge)
    # =========================================================================
    "apex_50k_manual": AccountProfile(
        profile_id="apex_50k_manual",
        firm="apex",
        account_size=50_000,
        copies=1,
        stop_multiplier=0.75,
        max_slots=4,
        # Phase 1 manual: playbook 5-session plan (lines 193-227)
        # TOKYO_OPEN, SINGAPORE_OPEN, EUROPE_FLOW, LONDON_METALS, NYSE_OPEN
        # CME_REOPEN is TopStep-only (playbook line 600). BRISBANE_1025 not in Phase 1.
        allowed_sessions=frozenset(
            {
                "TOKYO_OPEN",
                "SINGAPORE_OPEN",
                "EUROPE_FLOW",
                "LONDON_METALS",
                "NYSE_OPEN",  # Night block — "last thing before bed" per playbook
            }
        ),
        daily_lanes=(
            DailyLaneSpec("MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G5_CONT_S075", "MNQ", "TOKYO_OPEN"),
            DailyLaneSpec("MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_ORB_G8_O15_S075", "MNQ", "SINGAPORE_OPEN"),
            DailyLaneSpec("MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G8_S075", "MNQ", "EUROPE_FLOW"),
            DailyLaneSpec("MNQ_LONDON_METALS_E2_RR1.0_CB1_ORB_G6_NOMON_S075", "MNQ", "LONDON_METALS"),
            DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4_S075", "MNQ", "NYSE_OPEN"),
        ),
        notes="Phase 1 manual proof. No automation, no copy trading. Waking hours only.",
    ),
    # =========================================================================
    # Phase 2: Automation scaling (Tradeify MNQ + TopStep MGC)
    # =========================================================================
    "tradeify_50k": AccountProfile(
        profile_id="tradeify_50k",
        firm="tradeify",
        account_size=50_000,
        copies=5,  # 5 identical accounts — PRIMARY MNQ scaling lane
        stop_multiplier=0.75,
        max_slots=6,
        # Overnight sessions from playbook account grid
        allowed_sessions=frozenset(
            {
                "CME_PRECLOSE",
                "COMEX_SETTLE",
                "NYSE_CLOSE",
                "NYSE_OPEN",
            }
        ),
        allowed_instruments=frozenset({"MNQ"}),
        notes="MNQ overnight auto. Tradovate API. Bot exclusive (no cross-firm).",
    ),
    "topstep_50k": AccountProfile(
        profile_id="topstep_50k",
        firm="topstep",
        account_size=50_000,
        copies=5,  # 5 Express accounts — MGC morning lane
        stop_multiplier=0.75,
        max_slots=4,
        # Morning sessions from playbook account grid
        allowed_sessions=frozenset({"CME_REOPEN", "TOKYO_OPEN"}),
        allowed_instruments=frozenset({"MGC"}),
        daily_lanes=(
            DailyLaneSpec(
                "MGC_CME_REOPEN_E2_RR1.5_CB1_ORB_G4_FAST10_S075",
                "MGC",
                "CME_REOPEN",
                execution_notes="Skip if CME_REOPEN ORB exceeds 26 points.",
            ),
        ),
        notes="MGC morning auto. ProjectX API. Stay Express (don't accept Live).",
    ),
    # =========================================================================
    # Phase 3: Self-funded (after prop proof, $100K/year target)
    # =========================================================================
    "self_funded_50k": AccountProfile(
        profile_id="self_funded_50k",
        firm="self_funded",
        account_size=50_000,
        copies=1,
        stop_multiplier=1.0,
        max_slots=10,
        active=False,  # Phase 3 — not active until prop proof complete
        # None = all sessions, all instruments
        notes="Own capital. ALL 9 sessions. 5-10c. DD=temporary. IBKR (not built yet).",
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
