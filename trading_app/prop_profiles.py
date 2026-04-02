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
    max_orb_size_pts: float | None = None  # Max risk in pts — skip if risk_points >= this (includes stop_mult)


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
# Verified firm specs (April 2026)
# Sources: help.topstep.com, help.tradeify.co, support.apextraderfunding.com
# =========================================================================

PROP_FIRM_SPECS: dict[str, PropFirmSpec] = {
    "topstep": PropFirmSpec(
        name="topstep",
        display_name="TopStep",
        dd_type="eod_trailing",
        # Flat 90/10 since Jan 12, 2026. First 4 Express payouts capped at 50% of available balance.
        profit_split_tiers=((float("inf"), 0.90),),
        consistency_rule=0.40,
        news_restriction=False,
        close_time_et="16:10",  # 3:10 PM CT = 4:10 PM ET. Verified 2026-04-01.
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
        close_time_et="16:59",  # 4:59 PM ET. Verified 2026-04-01.
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
        profit_split_tiers=((float("inf"), 1.00),),  # 100% split on EOD PA plans
        consistency_rule=0.50,  # 50% since Apex 4.0 (Mar 2026). Was 30% legacy. Verified 2026-04-01.
        news_restriction=False,
        close_time_et="16:59",
        platform="tradovate",
        min_hold_seconds=None,
        banned_instruments=frozenset({"MGC", "GC", "SI", "SIL", "HG", "PL", "PA"}),
        auto_trading="none",  # PROHIBITED — PA Compliance: no bots, no copy trading, manual only
        # Official rules (resources/prop-firm-official-rules.md, fetched 2026-03-16):
        # - Automation/copy trading → immediate account closure + forfeiture
        # - 5:1 max RR ratio (stop ≤ 5× target) — all our RR1-4 strategies comply
        # - 30% per-trade loss rule: open unrealized loss ≤ 30% of start-of-day profit
        # - Stop losses REQUIRED on every trade (mental stops OK unless on Probation)
        # - Contract scaling: half max until trailing threshold reached
        # - Safety net: first 3 payouts require balance > DD + $100
        # - 8 trading day eval period, min $50 profit on 5 different days
        # - Metals SUSPENDED (not permanent ban — check periodically)
        notes=(
            "Manual proof only (1 account). Automation AND copy trading PROHIBITED on PA/Live. "
            "Metals suspended. 5:1 RR max. 30% per-trade loss rule. 30% windfall consistency. "
            "Contract scaling: half max until trailing threshold ($52.6K on 50K). "
            "Safety net first 3 payouts: balance > DD + $100."
        ),
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
    # TopStep — DLL removed on TopStepX since Aug 25, 2024.
    # DLL only applies on NinjaTrader/Tradovate/TradingView platforms.
    # We trade TopStepX (ProjectX API) → no DLL.
    ("topstep", 50_000): PropFirmAccount("topstep", 50_000, 2_000, 5, 50),
    ("topstep", 100_000): PropFirmAccount("topstep", 100_000, 3_000, 10, 100),
    ("topstep", 150_000): PropFirmAccount("topstep", 150_000, 4_500, 15, 150),
    # MFFU Core (EOD, 80/20). 50K DD confirmed $2K (not $1.5K).
    ("mffu", 50_000): PropFirmAccount("mffu", 50_000, 2_000, 5, 50),
    ("mffu", 100_000): PropFirmAccount("mffu", 100_000, 3_000, 8, 80),
    ("mffu", 150_000): PropFirmAccount("mffu", 150_000, 4_500, 12, 120),
    # Tradeify Select: verified 2026-04-01 via saveonpropfirms.com/blog/tradeify-select-guide
    # Prior values ($4K/$6K on 100K/150K) were from old Growth plan. Select = $2K/$3K/$4.5K.
    ("tradeify", 50_000): PropFirmAccount("tradeify", 50_000, 2_000, 4, 40),
    ("tradeify", 100_000): PropFirmAccount("tradeify", 100_000, 3_000, 8, 80),
    ("tradeify", 150_000): PropFirmAccount("tradeify", 150_000, 4_500, 12, 120),
    # Apex 4.0 EOD PA (March 2026) — metals banned, DLL introduced.
    # DLL is soft: pauses trading for the day, does NOT fail the account.
    ("apex", 50_000): PropFirmAccount("apex", 50_000, 2_000, 4, 40, daily_loss_limit=1_000),
    ("apex", 100_000): PropFirmAccount("apex", 100_000, 3_000, 6, 60, daily_loss_limit=1_500),
    ("apex", 150_000): PropFirmAccount("apex", 150_000, 4_000, 9, 90, daily_loss_limit=2_000),
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
        max_slots=5,
        active=False,  # Superseded by apex_100k_manual ($3K DD vs $2K)
        # Phase 1 manual: 5 validated MNQ lanes.
        # Score-driven rebuild 2026-03-31: composite score = ExpR * sharpe_adj * ayp *
        # n_confidence * fitness * rr_adj * prop_sm. 20% switching threshold (Carver Ch 12).
        # @research-source score-driven lane selection 2026-03-31
        # @revalidated-for E2 event-based sessions, post-confluence filters (2026-03-31)
        allowed_sessions=frozenset(
            {
                "CME_PRECLOSE",  # 5:45 AM Brisbane — score #1 (1.396)
                "NYSE_CLOSE",  # 6:00 AM Brisbane — score #2 (0.833)
                "COMEX_SETTLE",  # 3:30/4:30 AM Brisbane — score #3 (0.753)
                "US_DATA_1000",  # 00:00/01:00 AM Brisbane — score #4 (0.625)
                "TOKYO_OPEN",  # 10:00 AM Brisbane — score #5 (0.587)
            }
        ),
        daily_lanes=(
            # Lane selection via composite score applied uniformly to all 586 MNQ
            # candidates (non-PURGED, N>=100, E2 CB1 O5). Switching threshold 20%.
            # ORB caps from adversarial audit P90 data (2026-03-29).
            # L1: CME_PRECLOSE — NEW. Score 1.396 (#1). AYP=True, Sharpe 2.41.
            # Was 0 survivors in old validation; confluence filters unlocked 73 strategies.
            DailyLaneSpec(
                "MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VOL_RV20_N20_S075",
                "MNQ",
                "CME_PRECLOSE",
                max_orb_size_pts=120.0,
            ),
            # L2: NYSE_CLOSE — UPGRADED from VOL_RV12_N20 (+67.7% score).
            # VOL_RV20_N20 is strict superset filter (fewer, higher-edge trades).
            DailyLaneSpec(
                "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV20_N20_S075",
                "MNQ",
                "NYSE_CLOSE",
                max_orb_size_pts=100.0,
            ),
            # L3: COMEX_SETTLE — KEPT (ATR70_VOL tied with OVNRNG_100 at +0.4%).
            DailyLaneSpec(
                "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR70_VOL_S075",
                "MNQ",
                "COMEX_SETTLE",
                max_orb_size_pts=80.0,
            ),
            # L4: US_DATA_1000 — KEPT (already score-optimal). AYP=True, Sharpe 1.80.
            DailyLaneSpec(
                "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075",
                "MNQ",
                "US_DATA_1000",
                max_orb_size_pts=120.0,
            ),
            # L5: TOKYO_OPEN MNQ — NEW. Score 0.587 (#5). WHITELISTED fitness.
            # Replaces NYSE_OPEN (score 0.530, #8). RR2.5 = wider target.
            DailyLaneSpec(
                "MNQ_TOKYO_OPEN_E2_RR2.5_CB1_VOL_RV30_N20_S075",
                "MNQ",
                "TOKYO_OPEN",
                max_orb_size_pts=80.0,
            ),
        ),
        notes=(
            "Phase 1 manual. 5 MNQ lanes rebuilt 2026-03-31 via composite score. "
            "S075 aligned 2026-04-01 (strategy_ids reference _S075 validated variants). "
            "DD budget: $750 / $2K = 37.5%. "
            "Filter diversity: 4 filter families across 5 lanes."
        ),
    ),
    # =========================================================================
    # UPGRADE PATH: $100K Apex EOD PA. DD limit $3,000 (vs $2K on $50K).
    # Same 5 lanes + ORB caps from adversarial audit. Activate when ready.
    # =========================================================================
    "apex_100k_manual": AccountProfile(
        profile_id="apex_100k_manual",
        firm="apex",
        account_size=100_000,
        copies=1,
        stop_multiplier=0.75,
        max_slots=7,
        active=True,  # Upgraded from 50K — $3K DD gives $2,250 margin
        allowed_sessions=frozenset(
            {
                "CME_PRECLOSE",
                "COMEX_SETTLE",
                "EUROPE_FLOW",
                "SINGAPORE_OPEN",
                "TOKYO_OPEN",
                "NYSE_OPEN",
                "NYSE_CLOSE",
            }
        ),
        daily_lanes=(
            # HONEST DEPLOYMENT 2026-04-03. Adversarial audit findings:
            # - All RR targets from family_rr_locks (no RR snooping)
            # - COST_LT filters preferred (stable across price levels)
            # - ORB_G8 at NYSE_CLOSE only (88% pass — acceptable with monitoring)
            # - No vacuous filters (OVNRNG_50=100% pass, ORB_G6=96% for MNQ)
            # - DD: $296 worst-case / $3000 (10%)
            # - All 7 verified in validated_setups + family_rr_locks
            DailyLaneSpec(
                "MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08",
                "MNQ",
                "CME_PRECLOSE",
                max_orb_size_pts=120.0,
            ),
            DailyLaneSpec(
                "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT08",
                "MNQ",
                "EUROPE_FLOW",
                max_orb_size_pts=100.0,
            ),
            DailyLaneSpec(
                "MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_COST_LT08",
                "MNQ",
                "SINGAPORE_OPEN",
                max_orb_size_pts=100.0,
            ),
            DailyLaneSpec(
                "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12",
                "MNQ",
                "COMEX_SETTLE",
                max_orb_size_pts=80.0,
            ),
            DailyLaneSpec(
                "MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10",
                "MNQ",
                "TOKYO_OPEN",
                max_orb_size_pts=80.0,
            ),
            DailyLaneSpec(
                "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08",
                "MNQ",
                "NYSE_OPEN",
                max_orb_size_pts=200.0,
            ),
            DailyLaneSpec(
                "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8",
                "MNQ",
                "NYSE_CLOSE",
                max_orb_size_pts=80.0,
            ),
        ),
        notes=(
            "$100K Apex. 7 lanes (all MNQ). RR-locked, COST_LT preferred. "
            "Adversarial audit 2026-04-03: vacuous filters removed, RR snoop fixed. "
            "DD $296/$3000 (10%). MGC CME_REOPEN on trade sheet as MANUAL only."
        ),
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
        allowed_sessions=frozenset({"CME_PRECLOSE", "NYSE_CLOSE", "COMEX_SETTLE", "US_DATA_1000", "TOKYO_OPEN"}),
        allowed_instruments=frozenset({"MNQ"}),
        active=False,  # Activate when Tradovate API bot is ready for per-account execution
        # Score-driven rebuild 2026-03-31 — mirrors Apex sessions.
        # CME_PRECLOSE uses ATR70_VOL (not VOL_RV20_N20 like Apex) = cross-firm filter diversity.
        # Execution: Tradovate API per-account (Group Trading broken for brackets).
        # Bot must be exclusive to Tradeify (official rule — no cross-firm sharing).
        # 10s microscalp rule: no issue for ORB trades (hold 27-100+ minutes).
        daily_lanes=(
            # CME_PRECLOSE — ATR70_VOL (different filter from Apex = diversification)
            DailyLaneSpec(
                "MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR70_VOL",
                "MNQ",
                "CME_PRECLOSE",
                max_orb_size_pts=120.0,
            ),
            DailyLaneSpec(
                "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV20_N20_S075",
                "MNQ",
                "NYSE_CLOSE",
                max_orb_size_pts=100.0,
            ),
            DailyLaneSpec(
                "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR70_VOL_S075",
                "MNQ",
                "COMEX_SETTLE",
                max_orb_size_pts=80.0,
            ),
            DailyLaneSpec(
                "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075",
                "MNQ",
                "US_DATA_1000",
                max_orb_size_pts=120.0,
            ),
            DailyLaneSpec(
                "MNQ_TOKYO_OPEN_E2_RR2.5_CB1_VOL_RV30_N20_S075",
                "MNQ",
                "TOKYO_OPEN",
                max_orb_size_pts=80.0,
            ),
        ),
        notes=(
            "Phase 2 MNQ auto. 5 copies x 5 lanes via Tradovate API. "
            "CME_PRECLOSE ATR70_VOL still SM=1.0 ID (S075 variant not yet validated). "
            "Other 4 lanes S075 aligned 2026-04-01. DD $2K, budget $750 (37.5%)."
        ),
    ),
    "topstep_50k": AccountProfile(
        profile_id="topstep_50k",
        firm="topstep",
        account_size=50_000,
        copies=5,  # 5 Express accounts — MGC morning lane
        stop_multiplier=0.75,
        max_slots=4,
        # CONDITIONAL — per-session null P95=0.153 cleared, P99=0.364 not cleared
        # N=125 trades. Reduce size: 1 contract only until N=250.
        # Invalidation: 3 consecutive losing months OR forward ExpR < 0.10
        # Evidence: commits c236c57 (pooled null), 3850efa (per-session null)
        # Pooled P95=0.305 killed this; per-session P95=0.153 revived it.
        # Per Harvey & Liu (2015): per-session floor is methodologically correct.
        # @research-source C1 null rerun per-session TOKYO_OPEN 2026-03-24
        allowed_sessions=frozenset({"TOKYO_OPEN"}),
        allowed_instruments=frozenset({"MGC"}),
        daily_lanes=(
            DailyLaneSpec(
                "MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075",
                "MGC",
                "TOKYO_OPEN",
                execution_notes="1 contract only until N=250.",
                max_orb_size_pts=26.0,  # P90=20.4, was execution_notes only. Now code-enforced.
            ),
        ),
        notes=(
            "MGC CONDITIONAL — per-session P95 cleared, P99 not. 1 contract max. "
            "Invalidate at 3 losing months or ExpR<0.10. "
            "ERA_DEPENDENT: 77.6% of trades from 2025 (gold vol regime). "
            "G4 filter = de facto ATR regime selector (0.4% pass rate in low-vol vs 40.8% in high-vol). "
            "Shadow-trade only — no position size increase until N=250 with temporal diversity. "
            "Remediation audit 2026-03-25."
        ),
    ),
    # =========================================================================
    # Phase 2b: TopStep MNQ automation (ProjectX API)
    # First automated trade — single lane, prove the loop works, then scale.
    # =========================================================================
    "topstep_50k_mnq_auto": AccountProfile(
        profile_id="topstep_50k_mnq_auto",
        firm="topstep",
        account_size=50_000,
        copies=1,  # Start with 1 Express, scale to 5 after proving loop
        stop_multiplier=0.75,
        max_slots=1,  # Single lane — prove loop, then add lanes
        active=True,
        allowed_sessions=frozenset({"COMEX_SETTLE"}),
        allowed_instruments=frozenset({"MNQ"}),
        # COMEX_SETTLE = bot-only session (03:30 AM Brisbane, never manually tradeable).
        # Adds genuine portfolio diversification vs Apex manual lanes.
        # Scorer output 2026-03-31: score 0.225, rank #3 of deployed lanes.
        # Higher marginal value than duplicating CME_PRECLOSE (already on Apex).
        # ROBUST family (7 members, PBO=0.000, FDR adj_p=0.0000).
        # 2025 forward: +25.7R (N=63). 50 trades/yr = fastest loop proof.
        # Risk $29/trade = 1.5% DD, 2.9% DLL. 34 consecutive losers to DLL.
        # @research-source score_lanes.py composite score 2026-03-31
        # @revalidated-for E2 event-based sessions, holdout-clean re-discovery (2026-03-31)
        daily_lanes=(
            DailyLaneSpec(
                "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR70_VOL_S075",
                "MNQ",
                "COMEX_SETTLE",
                max_orb_size_pts=80.0,
            ),
        ),
        notes=(
            "Phase 2b: First MNQ auto lane via ProjectX API. "
            "COMEX_SETTLE 03:30 Brisbane (bot-only session). "
            "ATR70_VOL_S075 = stats match live 0.75x stops. "
            "Risk $29/trade = 1.5% DD. Scale to 5 Express after loop proof."
        ),
    ),
    # =========================================================================
    # Phase 2c: Full auto-scaling — TYPE-A (TopStep) and TYPE-B (Tradeify)
    #
    # TYPE-A sessions: US_DATA_1000, COMEX_SETTLE, CME_PRECLOSE, CME_REOPEN,
    #                  TOKYO_OPEN, LONDON_METALS, US_DATA_830, NYSE_OPEN
    # TYPE-B sessions: US_DATA_1000, COMEX_SETTLE, NYSE_CLOSE, CME_REOPEN,
    #                  SINGAPORE_OPEN, EUROPE_FLOW, US_DATA_830, NYSE_OPEN
    # Shared: US_DATA_1000, COMEX_SETTLE, CME_REOPEN, US_DATA_830, NYSE_OPEN
    # Fork:   TYPE-A gets CME_PRECLOSE/TOKYO_OPEN/LONDON_METALS
    #         TYPE-B gets NYSE_CLOSE/SINGAPORE_OPEN/EUROPE_FLOW
    #
    # Lane selection: best per session x instrument from validated_setups
    # (E2 CB1 O5, N>=100, CORE, ranked by ExpR). DB query 2026-04-01.
    #
    # ORB caps: set per lane to control DD risk. P90 caps for cheap sessions,
    # P75 caps for expensive sessions (MNQ NYSE_OPEN, US_DATA_830, US_DATA_1000).
    # Worst-day all-lose at 1ct: TYPE-A=$1,384, TYPE-B=$1,391.
    #
    # DD reality at 1ct: 50K=69%, 100K=46%, 150K=31%. Start 1ct, ramp with buffer.
    # =========================================================================
    # --- TYPE-A: TopStep 50K (5 Express accounts via ProjectX) ---
    "topstep_50k_type_a": AccountProfile(
        profile_id="topstep_50k_type_a",
        firm="topstep",
        account_size=50_000,
        copies=5,
        stop_multiplier=0.75,  # Default; per-lane SM encoded in strategy_id
        max_slots=16,
        active=False,  # Activate after proving loop on topstep_50k_mnq_auto
        allowed_sessions=frozenset(
            {
                "US_DATA_1000",
                "COMEX_SETTLE",
                "CME_PRECLOSE",
                "CME_REOPEN",
                "TOKYO_OPEN",
                "LONDON_METALS",
                "US_DATA_830",
                "NYSE_OPEN",
            }
        ),
        allowed_instruments=frozenset({"MNQ", "MGC", "MES"}),
        daily_lanes=(
            # --- US_DATA_1000 (00:00 Brisbane) — 3 instruments ---
            DailyLaneSpec(
                "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR70_S075", "MNQ", "US_DATA_1000", max_orb_size_pts=65.0
            ),  # P90=89.7, capped ~P75
            DailyLaneSpec(
                "MGC_US_DATA_1000_E2_RR1.0_CB1_ORB_G6", "MGC", "US_DATA_1000", max_orb_size_pts=15.0
            ),  # P90=13.0
            DailyLaneSpec(
                "MES_US_DATA_1000_E2_RR1.0_CB1_VOL_RV15_N20_S075", "MES", "US_DATA_1000", max_orb_size_pts=20.0
            ),  # P90=18.5
            # --- COMEX_SETTLE (03:30 Brisbane) — 2 instruments ---
            DailyLaneSpec(
                "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K", "MNQ", "COMEX_SETTLE", max_orb_size_pts=50.0
            ),  # P90=46.0
            DailyLaneSpec(
                "MES_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_S075", "MES", "COMEX_SETTLE", max_orb_size_pts=12.0
            ),  # P90=10.3
            # --- CME_PRECLOSE (05:45 Brisbane) — 2 instruments ---
            DailyLaneSpec(
                "MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VOL_RV20_N20", "MNQ", "CME_PRECLOSE", max_orb_size_pts=50.0
            ),  # P90=48.5
            DailyLaneSpec(
                "MES_CME_PRECLOSE_E2_RR1.0_CB1_VOL_RV20_N20_S075", "MES", "CME_PRECLOSE", max_orb_size_pts=12.0
            ),  # P90=11.3
            # --- CME_REOPEN (08:00 Brisbane) — 1 instrument ---
            DailyLaneSpec(
                "MNQ_CME_REOPEN_E2_RR1.0_CB1_VOL_RV30_N20", "MNQ", "CME_REOPEN", max_orb_size_pts=50.0
            ),  # P90=65.2, capped ~P75
            # --- TOKYO_OPEN (10:00 Brisbane) — 2 instruments ---
            DailyLaneSpec(
                "MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075", "MGC", "TOKYO_OPEN", max_orb_size_pts=10.0
            ),  # P90=8.4
            DailyLaneSpec(
                "MNQ_TOKYO_OPEN_E2_RR3.0_CB1_VOL_RV30_N20_S075", "MNQ", "TOKYO_OPEN", max_orb_size_pts=40.0
            ),  # P90=36.7
            # --- LONDON_METALS (17:00 Brisbane) — 2 instruments ---
            DailyLaneSpec(
                "MES_LONDON_METALS_E2_RR3.0_CB1_VOL_RV25_N20_S075", "MES", "LONDON_METALS", max_orb_size_pts=10.0
            ),  # P90=8.3
            DailyLaneSpec(
                "MNQ_LONDON_METALS_E2_RR1.5_CB1_ATR70_VOL_S075", "MNQ", "LONDON_METALS", max_orb_size_pts=40.0
            ),  # P90=36.5
            # --- US_DATA_830 (22:30 Brisbane) — 2 instruments ---
            DailyLaneSpec(
                "MNQ_US_DATA_830_E2_RR1.0_CB1_COST_LT12", "MNQ", "US_DATA_830", max_orb_size_pts=65.0
            ),  # P90=94.9, capped ~P70
            DailyLaneSpec(
                "MES_US_DATA_830_E2_RR1.0_CB1_VOL_RV20_N20_S075", "MES", "US_DATA_830", max_orb_size_pts=25.0
            ),  # P90=23.1
            # --- NYSE_OPEN (23:30 Brisbane) — 2 instruments ---
            DailyLaneSpec(
                "MES_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_25_S075", "MES", "NYSE_OPEN", max_orb_size_pts=20.0
            ),  # P90=18.8
            DailyLaneSpec(
                "MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR70_VOL", "MNQ", "NYSE_OPEN", max_orb_size_pts=70.0
            ),  # P90=104.1, capped ~P70
        ),
        notes=(
            "TYPE-A auto. 8 sessions, 16 lanes, 3 instruments. TopStepX/ProjectX. "
            "DD budget: $1,384 worst-day at 1ct = 69% of $2K DD. Start 1ct, ramp with buffer. "
            "Fork sessions: CME_PRECLOSE, TOKYO_OPEN, LONDON_METALS. "
            "All lanes DB-validated 2026-04-01 (CORE, E2 CB1 O5, N>=100)."
        ),
    ),
    # --- TYPE-A: TopStep 100K (5 Express accounts via ProjectX) ---
    "topstep_100k_type_a": AccountProfile(
        profile_id="topstep_100k_type_a",
        firm="topstep",
        account_size=100_000,
        copies=5,
        stop_multiplier=0.75,
        max_slots=16,
        active=False,  # Activate when upgrading from 50K tier
        allowed_sessions=frozenset(
            {
                "US_DATA_1000",
                "COMEX_SETTLE",
                "CME_PRECLOSE",
                "CME_REOPEN",
                "TOKYO_OPEN",
                "LONDON_METALS",
                "US_DATA_830",
                "NYSE_OPEN",
            }
        ),
        allowed_instruments=frozenset({"MNQ", "MGC", "MES"}),
        # Same lanes as 50K TYPE-A — tier only affects DD budget
        daily_lanes=(
            DailyLaneSpec(
                "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR70_S075", "MNQ", "US_DATA_1000", max_orb_size_pts=65.0
            ),
            DailyLaneSpec("MGC_US_DATA_1000_E2_RR1.0_CB1_ORB_G6", "MGC", "US_DATA_1000", max_orb_size_pts=15.0),
            DailyLaneSpec(
                "MES_US_DATA_1000_E2_RR1.0_CB1_VOL_RV15_N20_S075", "MES", "US_DATA_1000", max_orb_size_pts=20.0
            ),
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K", "MNQ", "COMEX_SETTLE", max_orb_size_pts=50.0),
            DailyLaneSpec("MES_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_S075", "MES", "COMEX_SETTLE", max_orb_size_pts=12.0),
            DailyLaneSpec("MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VOL_RV20_N20", "MNQ", "CME_PRECLOSE", max_orb_size_pts=50.0),
            DailyLaneSpec(
                "MES_CME_PRECLOSE_E2_RR1.0_CB1_VOL_RV20_N20_S075", "MES", "CME_PRECLOSE", max_orb_size_pts=12.0
            ),
            DailyLaneSpec("MNQ_CME_REOPEN_E2_RR1.0_CB1_VOL_RV30_N20", "MNQ", "CME_REOPEN", max_orb_size_pts=50.0),
            DailyLaneSpec("MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075", "MGC", "TOKYO_OPEN", max_orb_size_pts=10.0),
            DailyLaneSpec("MNQ_TOKYO_OPEN_E2_RR3.0_CB1_VOL_RV30_N20_S075", "MNQ", "TOKYO_OPEN", max_orb_size_pts=40.0),
            DailyLaneSpec(
                "MES_LONDON_METALS_E2_RR3.0_CB1_VOL_RV25_N20_S075", "MES", "LONDON_METALS", max_orb_size_pts=10.0
            ),
            DailyLaneSpec(
                "MNQ_LONDON_METALS_E2_RR1.5_CB1_ATR70_VOL_S075", "MNQ", "LONDON_METALS", max_orb_size_pts=40.0
            ),
            DailyLaneSpec("MNQ_US_DATA_830_E2_RR1.0_CB1_COST_LT12", "MNQ", "US_DATA_830", max_orb_size_pts=65.0),
            DailyLaneSpec(
                "MES_US_DATA_830_E2_RR1.0_CB1_VOL_RV20_N20_S075", "MES", "US_DATA_830", max_orb_size_pts=25.0
            ),
            DailyLaneSpec("MES_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_25_S075", "MES", "NYSE_OPEN", max_orb_size_pts=20.0),
            DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR70_VOL", "MNQ", "NYSE_OPEN", max_orb_size_pts=70.0),
        ),
        notes=(
            "TYPE-A auto 100K. Same 16 lanes as 50K. $3K DD = 46% at 1ct. "
            "TopStepX no DLL. AGGRO ceiling: 1ct. YOLO: 2ct. "
            "Upgrade from 50K when payouts flowing."
        ),
    ),
    # --- TYPE-B: Tradeify 50K (5 accounts via Tradovate API) ---
    "tradeify_50k_type_b": AccountProfile(
        profile_id="tradeify_50k_type_b",
        firm="tradeify",
        account_size=50_000,
        copies=5,
        stop_multiplier=0.75,
        max_slots=15,
        active=False,  # Blocked: Tradovate auth broken
        allowed_sessions=frozenset(
            {
                "US_DATA_1000",
                "COMEX_SETTLE",
                "NYSE_CLOSE",
                "CME_REOPEN",
                "SINGAPORE_OPEN",
                "EUROPE_FLOW",
                "US_DATA_830",
                "NYSE_OPEN",
            }
        ),
        allowed_instruments=frozenset({"MNQ", "MGC", "MES"}),
        daily_lanes=(
            # --- US_DATA_1000 (00:00 Brisbane) — 3 instruments ---
            DailyLaneSpec(
                "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR70_S075", "MNQ", "US_DATA_1000", max_orb_size_pts=65.0
            ),
            DailyLaneSpec("MGC_US_DATA_1000_E2_RR1.0_CB1_ORB_G6", "MGC", "US_DATA_1000", max_orb_size_pts=15.0),
            DailyLaneSpec(
                "MES_US_DATA_1000_E2_RR1.0_CB1_VOL_RV15_N20_S075", "MES", "US_DATA_1000", max_orb_size_pts=20.0
            ),
            # --- COMEX_SETTLE (03:30 Brisbane) — 2 instruments ---
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K", "MNQ", "COMEX_SETTLE", max_orb_size_pts=50.0),
            DailyLaneSpec("MES_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_S075", "MES", "COMEX_SETTLE", max_orb_size_pts=12.0),
            # --- NYSE_CLOSE (06:00 Brisbane) — 2 instruments ---
            DailyLaneSpec(
                "MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV25_N20", "MNQ", "NYSE_CLOSE", max_orb_size_pts=50.0
            ),  # P90=60.3, capped ~P75
            DailyLaneSpec(
                "MES_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT10", "MES", "NYSE_CLOSE", max_orb_size_pts=13.0
            ),  # P90=12.3
            # --- CME_REOPEN (08:00 Brisbane) — 1 instrument ---
            DailyLaneSpec("MNQ_CME_REOPEN_E2_RR1.0_CB1_VOL_RV30_N20", "MNQ", "CME_REOPEN", max_orb_size_pts=50.0),
            # --- SINGAPORE_OPEN (11:00 Brisbane) — 1 instrument ---
            DailyLaneSpec(
                "MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K", "MNQ", "SINGAPORE_OPEN", max_orb_size_pts=35.0
            ),  # P90=30.8
            # --- EUROPE_FLOW (18:00 Brisbane) — 2 instruments ---
            DailyLaneSpec("MGC_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G4", "MGC", "EUROPE_FLOW", max_orb_size_pts=8.0),  # P90=6.4
            DailyLaneSpec(
                "MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100", "MNQ", "EUROPE_FLOW", max_orb_size_pts=35.0
            ),  # P90=32.8
            # --- US_DATA_830 (22:30 Brisbane) — 2 instruments ---
            DailyLaneSpec("MNQ_US_DATA_830_E2_RR1.0_CB1_COST_LT12", "MNQ", "US_DATA_830", max_orb_size_pts=65.0),
            DailyLaneSpec(
                "MES_US_DATA_830_E2_RR1.0_CB1_VOL_RV20_N20_S075", "MES", "US_DATA_830", max_orb_size_pts=25.0
            ),
            # --- NYSE_OPEN (23:30 Brisbane) — 2 instruments ---
            DailyLaneSpec("MES_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_25_S075", "MES", "NYSE_OPEN", max_orb_size_pts=20.0),
            DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR70_VOL", "MNQ", "NYSE_OPEN", max_orb_size_pts=70.0),
        ),
        notes=(
            "TYPE-B auto. 8 sessions, 15 lanes, 3 instruments. Tradovate API (auth broken). "
            "DD budget: $1,391 worst-day at 1ct = 70% of $2K DD. Start 1ct. "
            "Fork sessions: NYSE_CLOSE, SINGAPORE_OPEN, EUROPE_FLOW. "
            "Tradeify: 90% split, no DLL, no consistency rule, 10s min hold (N/A for ORB). "
            "Bot must be exclusive to Tradeify (no cross-firm sharing)."
        ),
    ),
    # --- TYPE-B: Tradeify 100K (5 accounts via Tradovate API) ---
    "tradeify_100k_type_b": AccountProfile(
        profile_id="tradeify_100k_type_b",
        firm="tradeify",
        account_size=100_000,
        copies=5,
        stop_multiplier=0.75,
        max_slots=15,
        active=False,  # Blocked: Tradovate auth broken
        allowed_sessions=frozenset(
            {
                "US_DATA_1000",
                "COMEX_SETTLE",
                "NYSE_CLOSE",
                "CME_REOPEN",
                "SINGAPORE_OPEN",
                "EUROPE_FLOW",
                "US_DATA_830",
                "NYSE_OPEN",
            }
        ),
        allowed_instruments=frozenset({"MNQ", "MGC", "MES"}),
        # Same lanes as 50K TYPE-B — tier only affects DD budget
        daily_lanes=(
            DailyLaneSpec(
                "MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR70_S075", "MNQ", "US_DATA_1000", max_orb_size_pts=65.0
            ),
            DailyLaneSpec("MGC_US_DATA_1000_E2_RR1.0_CB1_ORB_G6", "MGC", "US_DATA_1000", max_orb_size_pts=15.0),
            DailyLaneSpec(
                "MES_US_DATA_1000_E2_RR1.0_CB1_VOL_RV15_N20_S075", "MES", "US_DATA_1000", max_orb_size_pts=20.0
            ),
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K", "MNQ", "COMEX_SETTLE", max_orb_size_pts=50.0),
            DailyLaneSpec("MES_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_S075", "MES", "COMEX_SETTLE", max_orb_size_pts=12.0),
            DailyLaneSpec("MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV25_N20", "MNQ", "NYSE_CLOSE", max_orb_size_pts=50.0),
            DailyLaneSpec("MES_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT10", "MES", "NYSE_CLOSE", max_orb_size_pts=13.0),
            DailyLaneSpec("MNQ_CME_REOPEN_E2_RR1.0_CB1_VOL_RV30_N20", "MNQ", "CME_REOPEN", max_orb_size_pts=50.0),
            DailyLaneSpec("MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K", "MNQ", "SINGAPORE_OPEN", max_orb_size_pts=35.0),
            DailyLaneSpec("MGC_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G4", "MGC", "EUROPE_FLOW", max_orb_size_pts=8.0),
            DailyLaneSpec("MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100", "MNQ", "EUROPE_FLOW", max_orb_size_pts=35.0),
            DailyLaneSpec("MNQ_US_DATA_830_E2_RR1.0_CB1_COST_LT12", "MNQ", "US_DATA_830", max_orb_size_pts=65.0),
            DailyLaneSpec(
                "MES_US_DATA_830_E2_RR1.0_CB1_VOL_RV20_N20_S075", "MES", "US_DATA_830", max_orb_size_pts=25.0
            ),
            DailyLaneSpec("MES_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_25_S075", "MES", "NYSE_OPEN", max_orb_size_pts=20.0),
            DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR70_VOL", "MNQ", "NYSE_OPEN", max_orb_size_pts=70.0),
        ),
        notes=(
            "TYPE-B auto 100K. Same 15 lanes as 50K. $3K DD = 46% at 1ct. "
            "No DLL. AGGRO: 1ct. YOLO: 2ct. "
            "Upgrade from 50K when payouts flowing."
        ),
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


def parse_strategy_id(strategy_id: str) -> dict:
    """Parse strategy parameters from canonical strategy_id string.

    Format: {INSTR}_{SESSION}_{ENTRY}_{RR}_{CB}_{FILTER}[_O{MIN}][_S{MULT}]
    Returns dict with: entry_model, rr_target, confirm_bars, filter_type, orb_minutes.
    """
    parts = strategy_id.split("_")
    result: dict = {
        "entry_model": "E2",
        "rr_target": 1.0,
        "confirm_bars": 1,
        "filter_type": "NO_FILTER",
        "orb_minutes": 5,
    }
    for p in parts:
        if p in ("E1", "E2", "E3"):
            result["entry_model"] = p
        elif p.startswith("RR"):
            result["rr_target"] = float(p[2:])
        elif p.startswith("CB"):
            result["confirm_bars"] = int(p[2:])
        elif p.startswith("O") and p[1:].isdigit():
            result["orb_minutes"] = int(p[1:])
    # filter_type: everything between CB and O/S suffix (or end)
    # Reconstruct from parts after CB until we hit O{digits} or S{digits} or end
    cb_idx = None
    for i, p in enumerate(parts):
        if p.startswith("CB") and p[2:].isdigit():
            cb_idx = i
            break
    if cb_idx is not None:
        filter_parts = []
        for p in parts[cb_idx + 1 :]:
            if (p.startswith("O") and p[1:].isdigit()) or (p.startswith("S") and p[1:].replace(".", "").isdigit()):
                break
            filter_parts.append(p)
        if filter_parts:
            result["filter_type"] = "_".join(filter_parts)
        else:
            result["filter_type"] = "NO_FILTER"
    return result


# Abbreviated lane names for paper_trades.lane_name column.
# Must match existing DB values to preserve continuity.
_LANE_NAMES: dict[str, str] = {
    "NYSE_CLOSE": "NYSE_CLOSE_VOL",
    "SINGAPORE_OPEN": "SING_G8",
    "CME_PRECLOSE": "CME_PRE",
    "COMEX_SETTLE": "COMEX_G8",
    "US_DATA_1000": "US_DATA_XMES",
    # TOKYO_OPEN: MNQ lane (Apex primary) takes priority over MGC shadow (TopStep).
    # MGC shadow is suppressed in get_lane_registry when MNQ TOKYO_OPEN is present.
    "TOKYO_OPEN": "TOKYO_VOL",
}


def find_active_manual_profile() -> str:
    """Find the active Apex manual profile (highest account_size wins)."""
    best = None
    for pid, p in ACCOUNT_PROFILES.items():
        if not p.active or not p.daily_lanes or p.firm != "apex":
            continue
        if best is None or p.account_size > ACCOUNT_PROFILES[best].account_size:
            best = pid
    return best or "apex_50k_manual"  # fallback if none active


def get_lane_registry(profile_id: str | None = None) -> dict[str, dict]:
    """Build a lane registry from a profile's daily_lanes.

    If profile_id is None, auto-selects the active Apex manual profile.
    Merges TopStep lanes (as shadow) from all active TopStep profiles.

    Returns {orb_label: {strategy_id, instrument, orb_label, entry_model,
    rr_target, confirm_bars, filter_type, orb_minutes, lane_name,
    is_half_size, shadow_only, execution_notes, stop_multiplier}}.

    This is the SINGLE SOURCE OF TRUTH for lane definitions.
    All consumer scripts (pre_session_check, log_trade, forward_monitor,
    slippage_scenario, sprt_monitor) must import from here.
    """
    if profile_id is None:
        profile_id = find_active_manual_profile()
    profile = ACCOUNT_PROFILES[profile_id]
    registry: dict[str, dict] = {}

    for lane in profile.daily_lanes:
        parsed = parse_strategy_id(lane.strategy_id)
        is_half = lane.planned_stop_multiplier is not None or "0.5x" in lane.execution_notes
        shadow = lane.instrument == "MGC" and lane.orb_label == "TOKYO_OPEN"  # MGC is shadow-only

        registry[lane.orb_label] = {
            "strategy_id": lane.strategy_id,
            "instrument": lane.instrument,
            "orb_label": lane.orb_label,
            "entry_model": parsed["entry_model"],
            "rr_target": parsed["rr_target"],
            "confirm_bars": parsed["confirm_bars"],
            "filter_type": parsed["filter_type"],
            "orb_minutes": parsed["orb_minutes"],
            "lane_name": _LANE_NAMES.get(lane.orb_label, lane.orb_label),
            "stop_multiplier": profile.stop_multiplier,
            "is_half_size": is_half,
            "shadow_only": shadow,
            "execution_notes": lane.execution_notes,
            "max_orb_size_pts": lane.max_orb_size_pts,
        }

    # Also include TopStep shadow lanes for any Apex manual profile.
    # Note: if Apex has an MNQ lane at the same session (e.g. TOKYO_OPEN),
    # the Apex lane takes priority and the MGC shadow is suppressed.
    # This is intentional — MNQ TOKYO_OPEN is the primary trade.
    # NOTE: Only merges topstep_50k (MGC shadow). topstep_50k_mnq_auto is
    # a separate bot-only profile — accessed via its own get_lane_registry() call,
    # not merged into the Apex manual registry.
    if profile.firm == "apex":
        ts_profile = ACCOUNT_PROFILES.get("topstep_50k")
        if ts_profile:
            for lane in ts_profile.daily_lanes:
                if lane.orb_label not in registry:
                    parsed = parse_strategy_id(lane.strategy_id)
                    registry[lane.orb_label] = {
                        "strategy_id": lane.strategy_id,
                        "instrument": lane.instrument,
                        "orb_label": lane.orb_label,
                        "entry_model": parsed["entry_model"],
                        "rr_target": parsed["rr_target"],
                        "confirm_bars": parsed["confirm_bars"],
                        "filter_type": parsed["filter_type"],
                        "orb_minutes": parsed["orb_minutes"],
                        "lane_name": _LANE_NAMES.get(lane.orb_label, lane.orb_label),
                        "stop_multiplier": ts_profile.stop_multiplier,
                        "is_half_size": False,
                        "shadow_only": True,  # TopStep MGC is shadow-trade
                        "execution_notes": lane.execution_notes,
                        "max_orb_size_pts": lane.max_orb_size_pts,
                    }

    return registry


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


# =========================================================================
# DD Budget Validation (CRITICAL-3 from adversarial audit 2026-03-29)
#
# Validates that worst-case simultaneous stop losses across all daily_lanes
# do not exceed the firm's max_dd. Uses a FIXED P90 ORB estimate per
# instrument (not DB-dependent — must be updated when ORB regimes shift).
# =========================================================================

# P90 ORB sizes in points (from adversarial audit 2026-03-29, gold.db, last 6 months).
# These are empirical — update when ORB regime shifts materially.
_P90_ORB_PTS: dict[str, float] = {
    "MNQ": 120.0,  # Across sessions: NYSE_CLOSE=66, SING=59, COMEX=52, NYSE_OPEN=212, US_DATA=101
    "MGC": 20.0,  # TOKYO_OPEN=20.4
    "MES": 30.0,  # Estimated (not in active lanes)
}

# Point values (must match pipeline.cost_model but inlined to avoid circular import)
_PV: dict[str, float] = {"MNQ": 2.0, "MGC": 10.0, "MES": 5.0}


def validate_dd_budget(profile_id: str | None = None) -> list[str]:
    """Validate DD budget for one or all active profiles.

    Returns list of violation strings. Empty = all clear.
    Does NOT raise — caller decides severity (pre_session_check gates on this).
    """
    # Verify inlined _PV matches canonical COST_SPECS (avoid silent drift)
    from pipeline.cost_model import COST_SPECS

    for inst, pv in _PV.items():
        if inst in COST_SPECS and COST_SPECS[inst].point_value != pv:
            raise RuntimeError(
                f"_PV[{inst}]={pv} != COST_SPECS.point_value={COST_SPECS[inst].point_value} — update _PV"
            )
    violations: list[str] = []
    profiles = (
        {profile_id: ACCOUNT_PROFILES[profile_id]}
        if profile_id
        else {pid: p for pid, p in ACCOUNT_PROFILES.items() if p.active}
    )

    for pid, prof in profiles.items():
        if not prof.daily_lanes:
            continue
        tier = ACCOUNT_TIERS.get((prof.firm, prof.account_size))
        if tier is None:
            violations.append(f"{pid}: no ACCOUNT_TIER for ({prof.firm}, {prof.account_size})")
            continue

        max_dd = tier.max_dd
        total_worst_case = 0.0
        lane_risks: list[tuple[str, float]] = []

        for lane in prof.daily_lanes:
            inst = lane.instrument
            p90_orb = _P90_ORB_PTS.get(inst, 100.0)
            pv = _PV.get(inst, 2.0)
            sm = lane.planned_stop_multiplier or prof.stop_multiplier

            # Worst case: use cap if set (max ORB we'd actually trade), else P90
            if lane.max_orb_size_pts is not None:
                effective_orb = lane.max_orb_size_pts
            else:
                effective_orb = p90_orb

            worst_stop = effective_orb * sm * pv
            total_worst_case += worst_stop
            lane_risks.append((lane.orb_label, worst_stop))

        if total_worst_case > max_dd:
            detail = ", ".join(f"{s}=${v:.0f}" for s, v in lane_risks)
            violations.append(
                f"{pid}: worst-case total ${total_worst_case:.0f} > DD limit ${max_dd:.0f} "
                f"({total_worst_case / max_dd:.0%}). Lanes: {detail}"
            )

    return violations


# Import-time check: warn (do not crash) if any profile is over budget.
import logging as _logging
import sys as _sys

_budget_violations = validate_dd_budget()
if _budget_violations:
    _log = _logging.getLogger(__name__)
    for _v in _budget_violations:
        _log.warning("DD BUDGET VIOLATION: %s", _v)
    # Also print to stderr for scripts that don't configure logging
    if not _logging.root.handlers:
        for _v in _budget_violations:
            print(f"[DD_BUDGET WARNING] {_v}", file=_sys.stderr)
