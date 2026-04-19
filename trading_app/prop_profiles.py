"""
Prop firm portfolio profiles — configuration and data structures.

Three layers:
1. PROP_FIRM_SPECS: static firm rules (verified from firm websites, Mar 2026)
2. ACCOUNT_TIERS: account size -> DD/contract limits
3. ACCOUNT_PROFILES: user's actual accounts (editable)

Pattern follows COST_SPECS in pipeline/cost_model.py — canonical source of truth,
imported everywhere, easy to edit.

Scaling constraint (see docs/audit/2026-04-15-topstep-scaling-reality-audit.md):
TopStep XFA and LFA are MUTUALLY EXCLUSIVE — LFA promotion destroys all XFAs
(topstep_live_funded_parameters.md:280). Any revenue projection or scaling
design that treats them as additive (e.g. "5 XFA + 1 LFA concurrent") is wrong.
50K account sizing confirmed over 100K/150K for current bot at 1 MNQ/lane:
economics identical across sizes at that contract count, LFA tier advantage
swamped by <1% LFA long-term survival (external data). Real multi-account
scale lives across firms (TS 5 + Bulenox 11 + Apex 20 = 36 accounts), not
Topstep-alone promotion climb.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading_app.config import ENTRY_MODELS

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
    """Exact daily execution lane for an account profile."""

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
    # Exact daily lanes for an account profile. Empty = use dynamic selection.
    daily_lanes: tuple[DailyLaneSpec, ...] = ()
    # Canonical payout path for this account (e.g. topstep_express_standard).
    # None = payout mechanics are not modeled for this profile.
    payout_policy_id: str | None = None
    max_risk_per_trade: float | None = None  # Dollar cap per trade. None = no limit.
    # @canonical-source docs/research-input/topstep/topstep_mll_article.md
    # @audit-finding F-5 (HWM freeze formula must differentiate XFA vs TC starting balance).
    # XFA accounts start at $0 broker equity; TC accounts start at account_size.
    # Default True because the active deployment topstep_50k_mnq_auto is XFA-shaped.
    # Set to False only for Trading Combine practice profiles.
    is_express_funded: bool = True
    # @canonical-source docs/research-input/topstep/topstep_live_funded_parameters.md
    # @audit-finding F-3 (DEFERRED — LFA DLL = MLL with $10K low-balance override).
    # Reserved for the LFA-promotion path. Not yet wired into AccountHWMTracker.
    # Stage 4 reserves the slot; LFA DLL semantics will be wired in a future stage.
    is_live_funded: bool = False
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
        consistency_rule=None,  # No universal firm-level consistency rule. Standard vs Consistency path handled in payout policies.
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
        dd_type="intraday_trailing",  # Rapid sim = intraday trailing. Live = EOD. (official: article 13134709)
        profit_split_tiers=((float("inf"), 0.90),),  # Rapid = 90/10 since Jan 12 2026 (official: article 13745661)
        consistency_rule=None,  # Rapid sim funded = none (official: article 13134709). Eval = 50%.
        news_restriction=True,  # No T1 news in sim funded (official: article 13134709)
        close_time_et="16:10",
        platform="tradovate",
        min_hold_seconds=None,
        banned_instruments=frozenset(),
        auto_trading="full",  # "Traders may make use of automated trading strategies" (official: article 8444599)
        notes=(
            "RAPID PLAN ONLY. Sim: intraday DD $2K locks at $100, daily payouts, $2100 buffer, 90/10. "
            "Live: EOD DD $2K floor at $0, 4mini/40micro (reduced from 5/50). "
            "$5K reserve held on Live transition. 21-day cooldown after Live blow. "
            "No T1 news in sim. $10K single day = forced Live transition. "
            "Source: help.myfundedfutures.com articles 13134709, 13134718, 13286746, 13745661."
        ),
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
    "self_funded": PropFirmSpec(
        name="self_funded",
        display_name="Self-Funded",
        dd_type="eod_trailing",  # Bot-enforced trailing DD (no external firm rule)
        profit_split_tiers=((float("inf"), 1.00),),
        consistency_rule=None,
        news_restriction=False,
        close_time_et="16:00",  # Force-flatten 10min before CME hard close (overnight margin guard)
        platform="any",
        min_hold_seconds=None,
        banned_instruments=frozenset(),
        auto_trading="full",
        notes=(
            "Your own capital on NinjaTrader/IBKR. Bot enforces DD limits, "
            "daily loss, and 24h cooling after equity halt. 0.75x stops for "
            "survival. Force-flatten at 16:00 ET to avoid overnight margin."
        ),
    ),
    "bulenox": PropFirmSpec(
        name="bulenox",
        display_name="Bulenox",
        dd_type="eod_trailing",  # Trader chooses trailing or EOD at qualification (official: bulenox.com/help/master-account)
        profit_split_tiers=((10_000, 1.00), (float("inf"), 0.90)),  # 100% first $10K, then 90/10
        consistency_rule=0.40,  # 40% rule: best day < 40% of total profit (official: master account rules)
        news_restriction=False,
        close_time_et="16:00",  # Not explicitly published — using CME default
        platform="rithmic",
        min_hold_seconds=None,
        banned_instruments=frozenset(),
        auto_trading="full",  # "Traders can use third-party algorithms" (official: bulenox.com/help)
        notes=(
            "DD locks at starting+$100. Safety reserve LOCKED ($2.6K on 50K). "
            "First 3 payouts capped ($1.5K on 50K). After 3rd: UNCAPPED. "
            "Weekly Wednesday payouts. 10 trading days before first withdrawal. "
            "Min $1K payout. Up to 3 simultaneous Master accounts (11 total). "
            "One-time activation fee, no monthly. Rithmic platform. "
            "Source: bulenox.com/help/master-account (scraped Apr 3 2026)."
        ),
    ),
}


ACCOUNT_TIERS: dict[tuple[str, int], PropFirmAccount] = {
    # ─── TopStep ────────────────────────────────────────────────────
    # DLL is OPT-IN on TopstepX for TC/XFA, MANDATORY on Live Funded.
    # @canonical-source docs/research-input/topstep/topstep_dll_article.md  (article 10490293, scraped 2026-04-08)
    # @verbatim "The Daily Loss Limit should be viewed as a safety net. It's a risk
    #            feature that can be turned on and off in your Trading Combine or
    #            Express Funded Account, but will automatically be applied to all
    #            Live Funded Accounts."
    # @verbatim "by removing the Daily Loss Limit on TopstepX™, we're giving traders
    #            more freedom and flexibility to trade their way."
    # @audit-finding F-9 (verified TRUE — code claim is correct, see docs/audit/2026-04-08-topstep-canonical-audit.md)
    # @bot-platform We trade TopstepX (ProjectX API) → DLL exempt for TC/XFA.
    #
    # Maximum Loss Limit (MLL) values for TC and XFA:
    # @canonical-source docs/research-input/topstep/topstep_mll_article.md  (article 8284204, scraped 2026-04-08)
    # @verbatim "$50K account: $2,000 / $100K account: $3,000 / $150K account: $4,500"
    #
    # IMPORTANT: max_contracts_mini/max_contracts_micro below are TOP-OF-LADDER
    # (max scaling tier reached after sufficient EOD profit). They are NOT day-1 limits.
    # Day-1 enforcement is pending — see audit finding F-1 (BLOCKER).
    # @canonical-source docs/research-input/topstep/topstep_scaling_plan_article.md  (article 8284223, scraped 2026-04-08)
    # @canonical-image docs/research-input/topstep/images/xfa_scaling_chart.png  (visually parsed canonical ladder)
    # @audit-finding F-1 (Stage 7 fix pending — trading_app/topstep_scaling_plan.py)
    #
    # @lfa-only For LIVE FUNDED accounts (not XFA), DLL is MANDATORY = same as MLL
    # ($2K/$3K/$4.5K) with $10K-low-balance override (DLL drops to $2K).
    # @canonical-source docs/research-input/topstep/topstep_live_funded_parameters.md  (article 10657969, scraped 2026-04-08)
    # @verbatim "Live Funded Accounts begin with a Daily Loss Limit based on account
    #            size: $2,000 for $50K accounts, $3,000 for $100K accounts, $4,500 for
    #            $150K accounts. Regardless of account size, if the tradable balance
    #            reaches $10,000 or below, the Daily Loss Limit will be set to $2,000."
    # @audit-finding F-3 DEFERRED (no LFA today; blocker at LFA-promotion time)
    ("topstep", 50_000): PropFirmAccount("topstep", 50_000, 2_000, 5, 50),
    ("topstep", 100_000): PropFirmAccount("topstep", 100_000, 3_000, 10, 100),
    ("topstep", 150_000): PropFirmAccount("topstep", 150_000, 4_500, 15, 150),
    # MFFU Rapid (intraday trailing sim, EOD live). 90/10 split. Official: article 13134709.
    # Live contracts REDUCED: 4/40, 6/60, 8/80 (official: article 13134718).
    ("mffu", 50_000): PropFirmAccount("mffu", 50_000, 2_000, 5, 50),
    ("mffu", 100_000): PropFirmAccount("mffu", 100_000, 3_000, 6, 60),
    ("mffu", 150_000): PropFirmAccount("mffu", 150_000, 4_500, 8, 80),
    # Tradeify Select: verified 2026-04-01 via saveonpropfirms.com/blog/tradeify-select-guide
    # Prior values ($4K/$6K on 100K/150K) were from old Growth plan. Select = $2K/$3K/$4.5K.
    ("tradeify", 50_000): PropFirmAccount("tradeify", 50_000, 2_000, 4, 40),
    ("tradeify", 100_000): PropFirmAccount("tradeify", 100_000, 3_000, 8, 80),
    ("tradeify", 150_000): PropFirmAccount("tradeify", 150_000, 4_500, 12, 120),
    # Bulenox Master (official: bulenox.com/help/master-account, scraped Apr 3 2026)
    # DD locks at starting+$100. Safety reserve locked. One-time activation fee.
    ("bulenox", 25_000): PropFirmAccount("bulenox", 25_000, 1_500, 2, 20),
    ("bulenox", 50_000): PropFirmAccount("bulenox", 50_000, 2_500, 5, 50),
    ("bulenox", 100_000): PropFirmAccount("bulenox", 100_000, 3_000, 10, 100),
    ("bulenox", 150_000): PropFirmAccount("bulenox", 150_000, 4_500, 15, 150),
    ("bulenox", 250_000): PropFirmAccount("bulenox", 250_000, 5_500, 25, 250),
    # ─── Self-Funded ──────────────────────────────────────────────────
    # Bot-enforced DD limits. No external firm rules.
    # Small accounts (<$10K): 15% max DD (tighter for survival)
    # Larger accounts (>=$10K): 20% max DD
    # Daily loss limit: 5% of account across all tiers
    # Max contracts: 1 MNQ per $2,500, 1 NQ per $25,000
    # PropFirmAccount(firm, account_size, max_dd, max_mini, max_micro, daily_loss_limit)
    ("self_funded", 2_500): PropFirmAccount("self_funded", 2_500, 375, 0, 1, 125),
    ("self_funded", 5_000): PropFirmAccount("self_funded", 5_000, 750, 0, 2, 250),
    ("self_funded", 10_000): PropFirmAccount("self_funded", 10_000, 1_500, 0, 4, 500),
    ("self_funded", 25_000): PropFirmAccount("self_funded", 25_000, 5_000, 1, 10, 1_250),
    ("self_funded", 30_000): PropFirmAccount("self_funded", 30_000, 6_000, 1, 12, 1_500),
    ("self_funded", 50_000): PropFirmAccount("self_funded", 50_000, 10_000, 2, 20, 2_500),
}


# =========================================================================
# User account profiles (EDITABLE)
# =========================================================================

ACCOUNT_PROFILES: dict[str, AccountProfile] = {
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
        # Rebuilt from current allocator-backed deployable shelf on 2026-04-19.
        # Prior config had 4 ghost lanes and 1 valid incumbent displaced by the
        # current liveness-aware allocator. Keep inactive until explicit account
        # activation review.
        # Execution: Tradovate API per-account (Group Trading broken for brackets).
        # Bot must be exclusive to Tradeify (official rule — no cross-firm sharing).
        # 10s microscalp rule: no issue for ORB trades (hold 27-100+ minutes).
        daily_lanes=(
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "MNQ", "COMEX_SETTLE", max_orb_size_pts=52.8),
            DailyLaneSpec("MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12", "MNQ", "TOKYO_OPEN", max_orb_size_pts=45.6),
            DailyLaneSpec("MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15", "MNQ", "US_DATA_1000", max_orb_size_pts=94.9),
        ),
        payout_policy_id="tradeify_select_funded",
        notes=(
            "Phase 2 MNQ auto. 5 copies x 3 current lanes via Tradovate API. "
            "Inactive profile rebuilt 2026-04-19 from current allocator-backed shelf. "
            "Current recommendation = 3 lanes after removing stale ghosts and "
            "dropping SR-alarmed COMEX incumbent substitute. "
            "DD $2K, budget $750 (37.5%)."
        ),
    ),
    "topstep_50k": AccountProfile(
        profile_id="topstep_50k",
        firm="topstep",
        account_size=50_000,
        copies=5,  # 5 Express accounts — MGC morning lane
        stop_multiplier=0.75,
        max_slots=4,
        active=False,  # Conditional shadow lane. Keep explicit, but do not treat as project primary.
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
        payout_policy_id="topstep_express_standard",
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
        copies=2,  # Start with 1-2 Express, scale to 5 after proving loop
        stop_multiplier=0.75,
        max_slots=7,
        active=True,
        allowed_sessions=frozenset(
            {
                "EUROPE_FLOW",
                "TOKYO_OPEN",
                "NYSE_OPEN",
                "COMEX_SETTLE",
                "CME_PRECLOSE",
                "SINGAPORE_OPEN",
                "US_DATA_1000",
            }
        ),
        allowed_instruments=frozenset({"MNQ"}),
        # DYNAMIC LANES — loaded from docs/runtime/lane_allocation.json
        # via load_allocation_lanes() at resolve_daily_lanes() time.
        #
        # Refresh: python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto
        #
        # Enforced by drift check 94 (pipeline/check_drift.py) — every lane
        # in the allocation JSON must exist in validated_setups with
        # status='active'. Staleness enforced by check_allocation_staleness()
        # in pre_session_check.py (BLOCK after 60 days, WARNING after 35).
        #
        # Prior hardcoded lanes (2026-04-13 reconstruction) moved to
        # allocator output. History: 6 unique sessions, max pairwise
        # rho=0.060. Now 7 sessions with US_DATA_1000 VWAP addition.
        daily_lanes=(),
        payout_policy_id="topstep_express_standard",
        notes=(
            "7-lane MNQ auto profile — DYNAMIC (allocator-managed). "
            "Lanes loaded from lane_allocation.json at runtime. "
            "Prior 6-lane profile reconstructed 2026-04-13 via "
            "correlation audit (max rho=0.060, +41% ExpR). "
            "7th lane: US_DATA_1000 VWAP (ExpR=+0.210, N=701). "
            "Each lane is the best-ExpR strategy from its session's "
            "independent correlation family. Audit script: "
            "scripts/research/portfolio_correlation_audit.py. "
            "All lanes validated via BH FDR + walk-forward + Criterion 11 "
            "Monte Carlo. Enforced by drift check 94."
        ),
    ),
    # =========================================================================
    # Phase 2b-MES: TopStep MES automation — instrument diversification
    #
    # Single MES lane on a separate TopStep Express account. MES on
    # CME_PRECLOSE provides session diversification from the MNQ lanes
    # (no overlapping sessions). MES/MNQ daily returns are correlated
    # (~0.90 equity indices) but trade-level correlation is lower due
    # to non-overlapping session times — NOT measured yet.
    #
    # Pre-flight audit (2026-04-12):
    #   - SR pre-flight: CONTINUE (max_SR=11.46, threshold=31.96, N=11 OOS)
    #   - COST_LT08 is strict subset of ORB_G8 (orb > 9.0 vs >= 8.0) —
    #     deploy ORB_G8 only, NOT both
    #   - 2026 OOS: N=11, mean +0.012R, WR 54.5%, directionally positive
    #   - Year-by-year: 1 negative year (2023, N=12, -0.074 avg_R)
    #   - Declining avg_R trend 2020-2025: monitor, insufficient N to diagnose
    #
    # Why ORB_G8 over COST_LT08:
    #   - More trades (50/yr vs 34/yr), more total R (+51.4 vs +38.5)
    #   - Better for single-lane profile (need frequency)
    #   - WF 5/5 vs 4/4 (more walk-forward windows tested)
    #   - Higher Sharpe (1.34 vs 1.25)
    # =========================================================================
    "topstep_50k_mes_auto": AccountProfile(
        profile_id="topstep_50k_mes_auto",
        firm="topstep",
        account_size=50_000,
        copies=1,  # Start with 1 Express — prove MES loop, then scale
        stop_multiplier=0.75,
        max_slots=2,  # 1 lane now, room for 1 expansion
        active=False,  # User activates when TopStep Express account is ready
        allowed_sessions=frozenset({"CME_PRECLOSE"}),
        allowed_instruments=frozenset({"MES"}),
        daily_lanes=(
            # MES CME_PRECLOSE ORB_G8: ExpR 0.173, Sharpe 1.34, N=287,
            # p=0.00123, WF 5/5 passed, FDR significant.
            # MES P90 ORB size = 11.2pts at CME_PRECLOSE. Cap at 20.0
            # (well above P95=14.5, catches only extreme outliers).
            # Dollar risk at 0.75x stop on avg qualifying ORB (11.1pts):
            # 11.1 * 0.75 * $5.0 = $41.63 per trade.
            DailyLaneSpec(
                "MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8",
                "MES",
                "CME_PRECLOSE",
                max_orb_size_pts=20.0,
            ),
        ),
        payout_policy_id="topstep_express_standard",
        notes=(
            "1-lane MES-only auto profile — instrument diversification from "
            "MNQ. CME_PRECLOSE is the highest-ExpR untapped session (0.173). "
            "ORB_G8 chosen over COST_LT08 because COST_LT08 is a strict "
            "trade-level subset (orb > 9.0 vs >= 8.0, 100% containment). "
            "ORB_G8 has more trades (50/yr vs 34/yr) and higher Sharpe "
            "(1.34 vs 1.25). SR pre-flight CONTINUE at N=11 OOS "
            "(max_SR=11.46, threshold=31.96). 2026 OOS directionally "
            "positive (+0.012 avg R). Declining avg_R trend from 2020 "
            "peak to monitor — retire if SR fires post-N=50 OOS."
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
        # Rebuilt from current allocator-backed deployable shelf on 2026-04-19.
        # Prior config had 7 ghost lanes and 1 valid incumbent displaced by the
        # current liveness-aware allocator. Keep inactive until explicit account
        # activation review.
        daily_lanes=(
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "MNQ", "COMEX_SETTLE", max_orb_size_pts=52.8),
            DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "MNQ", "NYSE_OPEN", max_orb_size_pts=117.8),
            DailyLaneSpec("MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12", "MNQ", "TOKYO_OPEN", max_orb_size_pts=45.6),
            DailyLaneSpec("MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15", "MNQ", "US_DATA_1000", max_orb_size_pts=94.9),
        ),
        payout_policy_id="topstep_express_standard",
        notes=(
            "TYPE-A auto inactive profile rebuilt 2026-04-19 from current allocator-backed shelf. "
            "Current recommendation = 4 lanes, all MNQ-led; stale ghosts removed and "
            "SR-alarmed COMEX incumbent displaced by current liveness-aware substitute. "
            "TopStepX/ProjectX. Keep inactive pending explicit activation review."
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
        # Rebuilt from current allocator-backed deployable shelf on 2026-04-19.
        # Same inactive recommendation as 50K TYPE-A under current shelf truth.
        daily_lanes=(
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "MNQ", "COMEX_SETTLE", max_orb_size_pts=52.8),
            DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "MNQ", "NYSE_OPEN", max_orb_size_pts=117.8),
            DailyLaneSpec("MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12", "MNQ", "TOKYO_OPEN", max_orb_size_pts=45.6),
            DailyLaneSpec("MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15", "MNQ", "US_DATA_1000", max_orb_size_pts=94.9),
        ),
        payout_policy_id="topstep_express_standard",
        notes=(
            "TYPE-A auto 100K inactive profile rebuilt 2026-04-19 from current allocator-backed shelf. "
            "Same 4-lane MNQ-led recommendation as 50K TYPE-A under current truth. "
            "$3K DD = 46% at 1ct. Keep inactive pending explicit activation review."
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
        # Rebuilt from current allocator-backed deployable shelf on 2026-04-19.
        # Prior config had 8 ghost lanes and 1 valid incumbent displaced by the
        # current liveness-aware allocator. Keep inactive until explicit account
        # activation review.
        daily_lanes=(
            DailyLaneSpec("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5", "MNQ", "EUROPE_FLOW", max_orb_size_pts=39.0),
            DailyLaneSpec("MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15", "MNQ", "SINGAPORE_OPEN", max_orb_size_pts=37.8),
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "MNQ", "COMEX_SETTLE", max_orb_size_pts=52.8),
            DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "MNQ", "NYSE_OPEN", max_orb_size_pts=117.8),
            DailyLaneSpec("MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15", "MNQ", "US_DATA_1000", max_orb_size_pts=94.9),
        ),
        payout_policy_id="tradeify_select_funded",
        notes=(
            "TYPE-B auto inactive profile rebuilt 2026-04-19 from current allocator-backed shelf. "
            "Current recommendation = 5 lanes, MNQ-led. Tradovate API (auth broken). "
            "Bot must be exclusive to Tradeify (no cross-firm sharing). "
            "Keep inactive pending explicit activation review."
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
        # Rebuilt from current allocator-backed deployable shelf on 2026-04-19.
        # Same inactive recommendation as 50K TYPE-B under current shelf truth.
        daily_lanes=(
            DailyLaneSpec("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5", "MNQ", "EUROPE_FLOW", max_orb_size_pts=39.0),
            DailyLaneSpec("MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15", "MNQ", "SINGAPORE_OPEN", max_orb_size_pts=37.8),
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "MNQ", "COMEX_SETTLE", max_orb_size_pts=52.8),
            DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "MNQ", "NYSE_OPEN", max_orb_size_pts=117.8),
            DailyLaneSpec("MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15", "MNQ", "US_DATA_1000", max_orb_size_pts=94.9),
        ),
        payout_policy_id="tradeify_select_funded",
        notes=(
            "TYPE-B auto 100K inactive profile rebuilt 2026-04-19 from current allocator-backed shelf. "
            "Same 5-lane MNQ-led recommendation as 50K TYPE-B under current truth. "
            "$3K DD = 46% at 1ct. Keep inactive pending explicit activation review."
        ),
    ),
    # =========================================================================
    # Phase 2d: Rithmic scaling (Bulenox — durable, no forced conversion)
    # =========================================================================
    "bulenox_50k": AccountProfile(
        profile_id="bulenox_50k",
        firm="bulenox",
        account_size=50_000,
        copies=3,  # Max 3 simultaneous Master accounts (official: bulenox.com/help/master-account)
        stop_multiplier=0.75,
        max_slots=5,
        active=False,  # Activate after Rithmic API conformance + paper trading validation
        allowed_sessions=frozenset(
            {
                "CME_REOPEN",
                "SINGAPORE_OPEN",
                "COMEX_SETTLE",
                "EUROPE_FLOW",
                "TOKYO_OPEN",
            }
        ),
        allowed_instruments=frozenset({"MNQ", "MGC"}),
        # Rebuilt from current allocator-backed deployable shelf on 2026-04-19.
        # Prior config had 4 ghost lanes and 1 valid incumbent displaced by the
        # current liveness-aware allocator. Keep inactive until explicit account
        # activation review.
        daily_lanes=(
            DailyLaneSpec("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5", "MNQ", "EUROPE_FLOW", max_orb_size_pts=39.0),
            DailyLaneSpec("MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15", "MNQ", "SINGAPORE_OPEN", max_orb_size_pts=37.8),
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "MNQ", "COMEX_SETTLE", max_orb_size_pts=52.8),
            DailyLaneSpec("MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12", "MNQ", "TOKYO_OPEN", max_orb_size_pts=45.6),
        ),
        notes=(
            "Bulenox 50K via Rithmic API. 3 copies (max simultaneous). "
            "No forced conversion. 100% first $10K then 90/10. "
            "40% consistency rule. DD locks at starting+$100. "
            "Inactive profile rebuilt 2026-04-19 from current allocator-backed shelf. "
            "Source: bulenox.com/help/master-account (scraped Apr 3 2026)."
        ),
    ),
    # =========================================================================
    # Phase 3: Self-funded (after prop proof, $100K/year target)
    # =========================================================================
    "self_funded_tradovate": AccountProfile(
        profile_id="self_funded_tradovate",
        firm="self_funded",
        account_size=30_000,
        copies=1,
        stop_multiplier=0.75,  # Same as prop — validated, don't change without re-testing
        max_slots=10,
        max_risk_per_trade=300.0,  # $300 cap = 1% of $30K per trade
        active=False,  # Activate after opening Tradovate personal account + API test
        allowed_sessions=frozenset(
            {
                "CME_REOPEN",
                "SINGAPORE_OPEN",
                "COMEX_SETTLE",
                "EUROPE_FLOW",
                "TOKYO_OPEN",
                "NYSE_OPEN",
                "US_DATA_1000",
                "CME_PRECLOSE",
            }
        ),
        allowed_instruments=frozenset({"MNQ", "MGC", "MES"}),
        # Rebuilt from current allocator-backed deployable shelf on 2026-04-19.
        # Prior config had 9 ghost lanes and 1 valid incumbent displaced by the
        # current liveness-aware allocator. ORB caps now use allocator-backed
        # session P90 limits rather than the older $300-budget-derived translation.
        daily_lanes=(
            DailyLaneSpec("MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5", "MNQ", "EUROPE_FLOW", max_orb_size_pts=39.0),
            DailyLaneSpec("MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15", "MNQ", "SINGAPORE_OPEN", max_orb_size_pts=37.8),
            DailyLaneSpec("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5", "MNQ", "COMEX_SETTLE", max_orb_size_pts=52.8),
            DailyLaneSpec("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "MNQ", "NYSE_OPEN", max_orb_size_pts=117.8),
            DailyLaneSpec("MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12", "MNQ", "TOKYO_OPEN", max_orb_size_pts=45.6),
            DailyLaneSpec("MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15", "MNQ", "US_DATA_1000", max_orb_size_pts=94.9),
        ),
        payout_policy_id="self_funded",
        notes=(
            "Self-funded Tradovate inactive profile rebuilt 2026-04-19 from current allocator-backed shelf. "
            "Current recommendation = 6 MNQ-led lanes after removing stale ghosts and "
            "dropping the SR-alarmed COMEX incumbent substitute. Current ORB caps are allocator-backed "
            "session P90 limits, not the older $300-budget-derived translation. Profile SM=0.75 overrides "
            "at runtime. Margin: $50/contract intraday (Tradovate). Commission: $1.22/RT. "
            "Self-imposed limits: daily -$600, weekly -$1,500, DD halt -$3,000."
        ),
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


def get_active_profile_ids(
    *,
    require_daily_lanes: bool = True,
    exclude_self_funded: bool = True,
) -> list[str]:
    """Return active profile ids that match the requested constraints."""
    active: list[str] = []
    for pid, profile in ACCOUNT_PROFILES.items():
        if not profile.active:
            continue
        if require_daily_lanes and not profile.daily_lanes and not load_allocation_lanes(pid):
            continue
        if exclude_self_funded and profile.firm == "self_funded":
            continue
        active.append(pid)
    return active


def resolve_profile_id(
    profile_id: str | None = None,
    *,
    require_daily_lanes: bool = True,
    active_only: bool = True,
    exclude_self_funded: bool = True,
) -> str:
    """Resolve one profile id, failing closed on ambiguity."""
    if profile_id is not None:
        profile = get_profile(profile_id)
        if active_only and not profile.active:
            raise ValueError(f"Profile {profile_id!r} is not active")
        if require_daily_lanes and not profile.daily_lanes and not load_allocation_lanes(profile_id):
            raise ValueError(f"Profile {profile_id!r} has no daily_lanes configured and no allocation JSON")
        if exclude_self_funded and profile.firm == "self_funded":
            raise ValueError(f"Profile {profile_id!r} is self-funded and not valid for this command")
        return profile_id

    active = get_active_profile_ids(require_daily_lanes=require_daily_lanes, exclude_self_funded=exclude_self_funded)
    if not active:
        raise ValueError("No active execution profile with daily_lanes is configured")
    if len(active) > 1:
        raise ValueError(
            "Multiple active execution profiles found: " + ", ".join(sorted(active)) + ". Pass an explicit profile_id."
        )
    return active[0]


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
        if p in ENTRY_MODELS:
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
    "TOKYO_OPEN": "TOKYO_VOL",
}


def find_active_primary_profile() -> str:
    """Backward-compatible wrapper for the single active execution profile."""
    return resolve_profile_id()


def find_active_manual_profile() -> str:
    """Backward-compatible alias for the active primary execution profile."""
    return find_active_primary_profile()


def get_profile_lane_definitions(profile_id: str | None = None) -> list[dict]:
    """Return the canonical lane definitions for one execution profile.

    The returned list preserves profile order and keeps full lane identity via
    strategy_id so consumers do not collapse duplicate sessions.

    This is the SINGLE SOURCE OF TRUTH for lane definitions.
    All consumer scripts (pre_session_check, log_trade, forward_monitor,
    slippage_scenario, sprt_monitor) must import from here.
    """
    profile_id = resolve_profile_id(profile_id, active_only=profile_id is None)
    profile = ACCOUNT_PROFILES[profile_id]
    lane_defs: list[dict] = []

    lane_specs = profile.daily_lanes
    if not lane_specs:
        lane_specs = load_allocation_lanes(profile.profile_id)

    for lane in lane_specs:
        parsed = parse_strategy_id(lane.strategy_id)
        is_half = lane.planned_stop_multiplier is not None or "0.5x" in lane.execution_notes
        shadow = False

        lane_defs.append(
            {
                "profile_id": profile.profile_id,
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
        )

    return lane_defs


def get_lane_registry(profile_id: str | None = None) -> dict[str, dict]:
    """Return a session-keyed lane map of session-level attributes.

    Multi-RR profiles have multiple lanes per session. The ORB cap
    (``max_orb_size_pts``) is a session-level attribute — a function of the
    session's volatility profile, not of the specific strategy — so every
    lane on the same session must share the same cap. This function returns
    one representative lane dict per session and fails closed if lanes on
    the same session disagree on ``max_orb_size_pts``.

    Consumers (``live/session_orchestrator.py`` ORB cap loader,
    ``scripts/tools/forward_monitor.py`` baseline loader) only read
    session-level fields; they do not care which specific lane represents
    the session.
    """
    registry: dict[str, dict] = {}
    cap_conflicts: dict[str, set[float | None]] = {}
    for lane in get_profile_lane_definitions(profile_id):
        label = lane["orb_label"]
        cap = lane.get("max_orb_size_pts")
        if label not in registry:
            registry[label] = lane
            continue
        existing_cap = registry[label].get("max_orb_size_pts")
        if cap != existing_cap:
            cap_conflicts.setdefault(label, set()).update([existing_cap, cap])

    if cap_conflicts:
        details = ", ".join(
            f"{label}={sorted(caps, key=lambda c: (c is None, c))}" for label, caps in sorted(cap_conflicts.items())
        )
        raise ValueError(
            "Profile has inconsistent max_orb_size_pts across lanes on the "
            f"same session: {details}. The ORB cap is a session-level "
            "attribute and must be identical for every lane on a given "
            "session. Reconcile the DailyLaneSpec entries in prop_profiles.py."
        )

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
# Dynamic lane loading from allocator JSON
# =========================================================================


def load_allocation_lanes(
    profile_id: str,
    allocation_path: str | Path | None = None,
) -> tuple[DailyLaneSpec, ...]:
    """Load allocated lanes for a profile from lane_allocation.json.

    Fail-closed: returns empty tuple on missing file, profile mismatch,
    corrupt JSON, or any parse error. This ensures resolve_daily_lanes()
    returns an empty list, which check_allocation_staleness() then blocks.
    """
    import json

    if allocation_path:
        path = Path(allocation_path)
    else:
        path = Path(__file__).resolve().parents[1] / "docs" / "runtime" / "lane_allocation.json"

    if not path.exists():
        return ()

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return ()

    # Profile must match — fail-closed on mismatch
    if data.get("profile_id") != profile_id:
        return ()

    lanes_data = data.get("lanes")
    if not isinstance(lanes_data, list):
        return ()

    specs: list[DailyLaneSpec] = []
    for entry in lanes_data:
        # Only deploy DEPLOY/PROVISIONAL lanes (not PAUSE/STALE)
        status = entry.get("status", "")
        if status not in ("DEPLOY", "PROVISIONAL"):
            continue

        sid = entry.get("strategy_id")
        inst = entry.get("instrument")
        orb = entry.get("orb_label")
        if not all((sid, inst, orb)):
            continue

        # Use per-session P90 from JSON if available, else fallback to flat _P90_ORB_PTS
        max_orb = entry.get("p90_orb_pts")
        if max_orb is None:
            max_orb = _P90_ORB_PTS.get(inst)

        specs.append(
            DailyLaneSpec(
                strategy_id=sid,
                instrument=inst,
                orb_label=orb,
                max_orb_size_pts=max_orb,
            )
        )

    return tuple(specs)


def effective_daily_lanes(profile: AccountProfile) -> tuple[DailyLaneSpec, ...]:
    """Return effective lanes for a profile — hardcoded or JSON-sourced.

    This is the SINGLE CANONICAL HELPER for resolving which DailyLaneSpec
    tuples apply to a profile. All callers that previously read
    profile.daily_lanes directly should use this instead.

    Returns profile.daily_lanes if populated, otherwise loads from
    lane_allocation.json. Returns empty tuple if neither source has data.
    """
    if profile.daily_lanes:
        return profile.daily_lanes
    return load_allocation_lanes(profile.profile_id)


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
        lanes = effective_daily_lanes(prof)
        if not lanes:
            continue
        tier = ACCOUNT_TIERS.get((prof.firm, prof.account_size))
        if tier is None:
            violations.append(f"{pid}: no ACCOUNT_TIER for ({prof.firm}, {prof.account_size})")
            continue

        max_dd = tier.max_dd
        total_worst_case = 0.0
        lane_risks: list[tuple[str, float]] = []

        for lane in lanes:
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
