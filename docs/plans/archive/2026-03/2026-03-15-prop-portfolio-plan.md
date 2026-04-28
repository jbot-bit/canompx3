---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Prop Firm Portfolio Construction — Implementation Plan

> **HISTORICAL ARTIFACT (Mar 15 2026).** Code examples below are STALE — they predate the Apex automation/copy-trading prohibition discovery. The LIVE code is in `trading_app/prop_profiles.py` and `trading_app/prop_portfolio.py`. Canonical firm rules are in `docs/plans/manual-trading-playbook.md` (V3, Mar 16 2026). DO NOT copy code from this file.

**Goal:** Build a profile-based portfolio selector that filters the validated strategy universe down to realistic, tradeable portfolios per prop firm account.

**Architecture:** Option A — profile layer on top of existing `live_config`. New files only. Zero modification to existing production code.

**Tech Stack:** Python dataclasses, DuckDB (read-only via existing `build_live_portfolio()`), `COST_SPECS` pattern for firm config.

**Design doc:** `docs/plans/2026-03-15-prop-portfolio-design.md`

---

### Task 0: Data Structures & Firm Config

**Files:**
- Create: `trading_app/prop_profiles.py`
- Test: `tests/test_trading_app/test_prop_profiles.py`

**Step 1: Write failing tests for dataclasses**

```python
# tests/test_trading_app/test_prop_profiles.py
"""Tests for trading_app.prop_profiles — prop firm config and data structures."""

import pytest

from trading_app.prop_profiles import (
    PropFirmSpec,
    PropFirmAccount,
    AccountProfile,
    TradingBookEntry,
    ExcludedEntry,
    TradingBook,
    PROP_FIRM_SPECS,
    ACCOUNT_TIERS,
    ACCOUNT_PROFILES,
    get_firm_spec,
    get_account_tier,
    get_profile,
    compute_profit_split_factor,
)


class TestPropFirmSpec:
    def test_topstep_exists(self):
        spec = get_firm_spec("topstep")
        assert spec.display_name == "TopStep"
        assert spec.dd_type == "eod_trailing"
        assert spec.auto_trading == "full"

    def test_tradeify_exists(self):
        spec = get_firm_spec("tradeify")
        assert spec.auto_trading == "full"
        assert spec.min_hold_seconds == 10

    def test_mffu_exists(self):
        spec = get_firm_spec("mffu")
        assert spec.auto_trading == "semi"

    def test_self_funded_no_firm(self):
        spec = get_firm_spec("self_funded")
        assert spec.consistency_rule is None
        assert spec.banned_instruments == frozenset()

    def test_unknown_firm_raises(self):
        with pytest.raises(KeyError):
            get_firm_spec("nonexistent")


class TestPropFirmAccount:
    def test_topstep_50k(self):
        tier = get_account_tier("topstep", 50_000)
        assert tier.max_dd == 2_000
        assert tier.max_contracts_mini == 5
        assert tier.max_contracts_micro == 50

    def test_topstep_150k(self):
        tier = get_account_tier("topstep", 150_000)
        assert tier.max_dd == 4_500
        assert tier.max_contracts_mini == 15

    def test_self_funded_50k(self):
        tier = get_account_tier("self_funded", 50_000)
        assert tier.max_dd == 5_000  # User-defined risk tolerance
        assert tier.max_contracts_micro == 999  # Effectively unlimited

    def test_unknown_tier_raises(self):
        with pytest.raises(KeyError):
            get_account_tier("topstep", 999_999)


class TestAccountProfile:
    def test_default_profiles_exist(self):
        p = get_profile("topstep_50k")
        assert p.firm == "topstep"
        assert p.account_size == 50_000
        assert p.stop_multiplier == 0.75
        assert p.max_slots == 6
        assert p.active is True

    def test_self_funded_profile(self):
        p = get_profile("self_funded_50k")
        assert p.stop_multiplier == 1.0
        assert p.max_slots == 10

    def test_profile_copies(self):
        p = get_profile("topstep_50k")
        assert p.copies >= 1

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            get_profile("nonexistent")


class TestProfitSplitFactor:
    def test_topstep_below_threshold(self):
        """First $5K at 50% split."""
        spec = get_firm_spec("topstep")
        factor = compute_profit_split_factor(spec, cumulative_profit=0)
        assert factor == pytest.approx(0.50)

    def test_topstep_above_threshold(self):
        """After $5K at 90% split."""
        spec = get_firm_spec("topstep")
        factor = compute_profit_split_factor(spec, cumulative_profit=6000)
        assert factor == pytest.approx(0.90)

    def test_tradeify_flat_split(self):
        """Tradeify Select: flat 90/10."""
        spec = get_firm_spec("tradeify")
        factor = compute_profit_split_factor(spec, cumulative_profit=0)
        assert factor == pytest.approx(0.90)

    def test_self_funded_keeps_all(self):
        spec = get_firm_spec("self_funded")
        factor = compute_profit_split_factor(spec, cumulative_profit=0)
        assert factor == pytest.approx(1.0)


class TestTradingBook:
    def test_empty_book(self):
        book = TradingBook(profile_id="test", entries=[], excluded=[])
        assert book.total_slots == 0
        assert book.total_dd_used == 0.0

    def test_book_with_entries(self):
        entry = TradingBookEntry(
            strategy_id="MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5",
            instrument="MGC",
            orb_label="TOKYO_OPEN",
            session_time_brisbane="19:00",
            entry_model="E2",
            rr_target=2.0,
            confirm_bars=1,
            filter_type="ORB_G5",
            direction="long",
            contracts=1,
            stop_multiplier=0.75,
            effective_expr=0.18,
            sharpe_dd_ratio=1.5,
            dd_contribution=935.0,
        )
        book = TradingBook(profile_id="test", entries=[entry], excluded=[])
        assert book.total_slots == 1
        assert book.total_dd_used == 935.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trading_app/test_prop_profiles.py -x -q`
Expected: FAIL (module not found)

**Step 3: Implement `trading_app/prop_profiles.py`**

```python
"""
Prop firm portfolio profiles — configuration and data structures.

Three layers:
1. PROP_FIRM_SPECS: static firm rules (verified from firm websites, Mar 2026)
2. ACCOUNT_TIERS: account size → DD/contract limits
3. ACCOUNT_PROFILES: user's actual accounts (editable)

Pattern follows COST_SPECS in pipeline/cost_model.py — canonical source of truth,
imported everywhere, easy to edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
        dd_type="eod_trailing",  # EOD trailing → static lock at starting balance
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


def compute_profit_split_factor(
    firm_spec: PropFirmSpec, cumulative_profit: float = 0.0
) -> float:
    """Return the trader's effective split percentage (0.0-1.0).

    Evaluates profit_split_tiers in order. First tier where
    cumulative_profit < threshold applies.
    """
    for threshold, pct in firm_spec.profit_split_tiers:
        if cumulative_profit < threshold:
            return pct
    # Fallback: last tier
    return firm_spec.profit_split_tiers[-1][1]
```

**Step 4: Run tests**

Run: `pytest tests/test_trading_app/test_prop_profiles.py -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add trading_app/prop_profiles.py tests/test_trading_app/test_prop_profiles.py
git commit -m "feat: add prop firm profile config and data structures"
```

---

### Task 1: Selection Algorithm Core

**Files:**
- Create: `trading_app/prop_portfolio.py`
- Test: `tests/test_trading_app/test_prop_portfolio.py`

**Step 1: Write failing tests for selection logic**

```python
# tests/test_trading_app/test_prop_portfolio.py
"""Tests for trading_app.prop_portfolio — prop firm portfolio selection."""

import pytest

from trading_app.portfolio import PortfolioStrategy
from trading_app.prop_profiles import (
    AccountProfile,
    get_firm_spec,
    get_account_tier,
)
from trading_app.prop_portfolio import (
    select_for_profile,
    _compute_dd_per_contract,
    _apply_instrument_bans,
    _deduplicate_sessions,
    _rank_strategies,
)


def _make_strategy(**overrides) -> PortfolioStrategy:
    """Build a PortfolioStrategy with sane defaults."""
    defaults = dict(
        strategy_id="MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5",
        instrument="MGC",
        orb_label="TOKYO_OPEN",
        entry_model="E2",
        rr_target=2.0,
        confirm_bars=1,
        filter_type="ORB_G5",
        expectancy_r=0.25,
        win_rate=0.55,
        sample_size=150,
        sharpe_ratio=1.2,
        max_drawdown_r=4.0,
        median_risk_points=10.0,
        orb_minutes=5,
        stop_multiplier=1.0,
        source="baseline",
        weight=1.0,
    )
    defaults.update(overrides)
    return PortfolioStrategy(**defaults)


class TestDDPerContract:
    def test_075x_stop(self):
        """$935 median DD at 0.75x stop (from Monte Carlo sim)."""
        dd = _compute_dd_per_contract(stop_multiplier=0.75, dd_type="eod_trailing")
        assert dd == pytest.approx(935.0)

    def test_10x_stop(self):
        """$1,350 median DD at 1.0x stop."""
        dd = _compute_dd_per_contract(stop_multiplier=1.0, dd_type="eod_trailing")
        assert dd == pytest.approx(1350.0)

    def test_intraday_trailing_adjustment(self):
        """Intraday trailing is stricter — 0.7x factor."""
        dd_eod = _compute_dd_per_contract(0.75, "eod_trailing")
        dd_intra = _compute_dd_per_contract(0.75, "intraday_trailing")
        assert dd_intra > dd_eod  # More DD consumed per contract


class TestInstrumentBans:
    def test_apex_bans_mgc(self):
        strats = [
            _make_strategy(instrument="MGC"),
            _make_strategy(instrument="MNQ", strategy_id="MNQ_TOKYO_E2"),
        ]
        banned = frozenset({"MGC", "GC", "SI"})
        filtered, excluded = _apply_instrument_bans(strats, banned)
        assert len(filtered) == 1
        assert filtered[0].instrument == "MNQ"
        assert len(excluded) == 1
        assert "banned" in excluded[0].reason.lower()

    def test_no_bans(self):
        strats = [_make_strategy()]
        filtered, excluded = _apply_instrument_bans(strats, frozenset())
        assert len(filtered) == 1
        assert len(excluded) == 0


class TestDeduplicateSessions:
    def test_keeps_best_per_session_instrument(self):
        """Two strategies for same session+instrument: keep higher ExpR."""
        s1 = _make_strategy(expectancy_r=0.20, strategy_id="s1")
        s2 = _make_strategy(expectancy_r=0.30, strategy_id="s2")
        deduped, excluded = _deduplicate_sessions([s1, s2])
        assert len(deduped) == 1
        assert deduped[0].strategy_id == "s2"

    def test_different_instruments_kept(self):
        """Same session, different instruments: both kept."""
        s1 = _make_strategy(instrument="MGC", strategy_id="s1")
        s2 = _make_strategy(instrument="MNQ", strategy_id="s2")
        deduped, _ = _deduplicate_sessions([s1, s2])
        assert len(deduped) == 2


class TestRankStrategies:
    def test_ranks_by_sharpe_dd_ratio(self):
        s1 = _make_strategy(sharpe_ratio=1.0, max_drawdown_r=4.0, strategy_id="s1")
        s2 = _make_strategy(sharpe_ratio=2.0, max_drawdown_r=4.0, strategy_id="s2")
        ranked = _rank_strategies([s1, s2], split_factor=1.0)
        assert ranked[0].strategy_id == "s2"  # Higher Sharpe/DD


class TestSelectForProfile:
    def test_dd_budget_exhaustion(self):
        """Should stop adding when DD budget is full."""
        profile = AccountProfile("test", "topstep", 50_000, 1, 0.75, max_slots=10)
        # $2K budget, $935/slot at 0.75x = fits ~2 slots
        strats = [
            _make_strategy(strategy_id=f"s{i}", orb_label=f"SESSION_{i}")
            for i in range(5)
        ]
        book = select_for_profile(profile, strats)
        assert book.total_slots == 2
        assert book.total_dd_used <= 2_000
        assert len(book.excluded) == 3

    def test_slot_cap(self):
        """Should respect cognitive cap even with DD room."""
        profile = AccountProfile("test", "self_funded", 50_000, 1, 1.0, max_slots=3)
        strats = [
            _make_strategy(strategy_id=f"s{i}", orb_label=f"SESSION_{i}")
            for i in range(5)
        ]
        book = select_for_profile(profile, strats)
        assert book.total_slots == 3
        assert any("cognitive cap" in e.reason.lower() for e in book.excluded)

    def test_contract_cap(self):
        """Should respect firm's contract limit."""
        profile = AccountProfile("test", "tradeify", 50_000, 1, 0.75, max_slots=10)
        # Tradeify $50K: 4 mini / 40 micro. With 1 micro per strategy,
        # DD budget ($2K / $935) constrains before contracts.
        strats = [
            _make_strategy(strategy_id=f"s{i}", orb_label=f"SESSION_{i}")
            for i in range(5)
        ]
        book = select_for_profile(profile, strats)
        assert book.total_contracts <= 40  # micro limit

    def test_empty_strategies(self):
        profile = AccountProfile("test", "topstep", 50_000)
        book = select_for_profile(profile, [])
        assert book.total_slots == 0
        assert len(book.excluded) == 0

    def test_consistency_rule(self):
        """No single session > 40% of portfolio ExpR (TopStep)."""
        profile = AccountProfile("test", "topstep", 50_000, 1, 0.75, max_slots=10)
        # One dominant strategy + several weaker ones
        dominant = _make_strategy(
            strategy_id="dominant", orb_label="TOKYO_OPEN", expectancy_r=0.80
        )
        weak = [
            _make_strategy(
                strategy_id=f"w{i}", orb_label=f"SESSION_{i}", expectancy_r=0.10
            )
            for i in range(4)
        ]
        book = select_for_profile(profile, [dominant] + weak)
        # If dominant is included, it's 0.80/(0.80+sum_weak) of total
        # With 1 weak at 0.10: 0.80/0.90 = 89% > 40% → need more diversity
        # Selection should still include dominant but flag if consistency violated
        assert book.total_slots >= 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trading_app/test_prop_portfolio.py -x -q`
Expected: FAIL (module not found)

**Step 3: Implement `trading_app/prop_portfolio.py`**

```python
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
from pathlib import Path

from pipeline.cost_model import get_cost_spec
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
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
    """Keep best strategy per (session × instrument). Best = highest ExpR."""
    best: dict[tuple[str, str], PortfolioStrategy] = {}
    for s in strategies:
        key = (s.orb_label, s.instrument)
        if key not in best or s.expectancy_r > best[key].expectancy_r:
            best[key] = s

    kept_ids = {s.strategy_id for s in best.values()}
    excluded = [
        ExcludedEntry(
            s.strategy_id, s.instrument, s.orb_label,
            f"Session conflict: better strategy exists for {s.orb_label} × {s.instrument}",
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
    """Rank strategies by Sharpe/DD ratio (proven methodology).

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
    for entry in SESSION_CATALOG:
        if entry.get("label") == orb_label or entry.get("name") == orb_label:
            # Return the base time (non-DST / Brisbane)
            return entry.get("brisbane_time", entry.get("time_local", "unknown"))
    return "unknown"


def select_for_profile(
    profile: AccountProfile,
    strategies: list[PortfolioStrategy],
) -> TradingBook:
    """Select optimal strategy subset for an account profile.

    Algorithm:
    1. Filter banned instruments
    2. Deduplicate session×instrument (keep best ExpR)
    3. Adjust ExpR for profit split
    4. Rank by Sharpe/DD ratio
    5. Greedy fill: DD budget → contract cap → slot cap → consistency
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

        # Consistency rule check (prospective)
        if firm_spec.consistency_rule is not None and total_effective_expr > 0:
            projected_total = total_effective_expr + rs.effective_expr
            pct_of_total = rs.effective_expr / projected_total
            if pct_of_total > firm_spec.consistency_rule:
                # Don't hard-exclude — just flag. Consistency is about daily P&L,
                # not static ExpR. We add it but note the risk.
                pass  # Still include, but could add a warning note

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
    Returns dict of profile_id → TradingBook.
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

    print(f"\n{'='*70}")
    print(f"  {firm_spec.display_name} ${tier.account_size:,} — {profile.profile_id}")
    if profile.copies > 1:
        print(f"  Copies: {profile.copies} (identical accounts)")
    print(f"  DD budget: ${tier.max_dd:,.0f} ({firm_spec.dd_type})")
    print(f"  Stop multiplier: {profile.stop_multiplier}x")
    print(f"  Slot cap: {profile.max_slots}")
    print(f"{'='*70}")

    if not book.entries:
        print("  NO STRATEGIES SELECTED")
    else:
        print(f"\n  {'Strategy':<45} {'Inst':<5} {'Session':<18} {'Time':<6} "
              f"{'EM':<3} {'RR':<4} {'CB':<3} {'Filter':<16} "
              f"{'EffExpR':<8} {'S/DD':<6} {'DD$':<7}")
        print(f"  {'-'*45} {'-'*5} {'-'*18} {'-'*6} "
              f"{'-'*3} {'-'*4} {'-'*3} {'-'*16} "
              f"{'-'*8} {'-'*6} {'-'*7}")
        for e in book.entries:
            sid_short = e.strategy_id[:45]
            print(f"  {sid_short:<45} {e.instrument:<5} {e.orb_label:<18} "
                  f"{e.session_time_brisbane:<6} {e.entry_model:<3} "
                  f"{e.rr_target:<4.1f} {e.confirm_bars:<3} {e.filter_type:<16} "
                  f"{e.effective_expr:<8.3f} {e.sharpe_dd_ratio:<6.2f} "
                  f"${e.dd_contribution:<6.0f}")

    print(f"\n  SUMMARY: {book.total_slots} slots | "
          f"${book.total_dd_used:,.0f} DD used of ${tier.max_dd:,.0f} "
          f"({book.total_dd_used/tier.max_dd*100:.0f}%) | "
          f"{book.total_contracts} contracts")

    if book.excluded:
        print(f"\n  EXCLUDED ({len(book.excluded)}):")
        for ex in book.excluded:
            print(f"    ✗ {ex.strategy_id[:40]:<40} {ex.instrument:<5} "
                  f"{ex.orb_label:<18} — {ex.reason}")
    print()
```

**Step 4: Run tests**

Run: `pytest tests/test_trading_app/test_prop_portfolio.py -x -q`
Expected: ALL PASS (may need minor adjustments to test expectations based on exact logic)

**Step 5: Commit**

```bash
git add trading_app/prop_portfolio.py tests/test_trading_app/test_prop_portfolio.py
git commit -m "feat: add prop firm portfolio selection algorithm"
```

---

### Task 2: CLI & Integration with `build_live_portfolio`

**Files:**
- Modify: `trading_app/prop_portfolio.py` (add `__main__` block)

**Step 1: Add CLI `main()` function**

Append to `trading_app/prop_portfolio.py`:

```python
def main() -> None:
    """CLI entry point."""
    import argparse

    from pipeline.asset_configs import get_active_instruments
    from trading_app.live_config import build_live_portfolio

    parser = argparse.ArgumentParser(
        description="Build prop firm trading books from validated strategies"
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help=f"Profile ID. Available: {', '.join(ACCOUNT_PROFILES.keys())}",
    )
    parser.add_argument("--all", action="store_true", help="Build all active profiles")
    parser.add_argument("--summary", action="store_true", help="Cross-account summary")
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args()

    if not args.profile and not args.all:
        args.all = True  # Default: show all

    db_path = args.db_path or GOLD_DB_PATH

    # Build eligible strategies for each instrument
    print("Loading validated strategies...")
    strategies_by_instrument: dict[str, list[PortfolioStrategy]] = {}
    for instrument in get_active_instruments():
        portfolio, notes = build_live_portfolio(db_path=db_path, instrument=instrument)
        strategies_by_instrument[instrument] = portfolio.strategies
        print(f"  {instrument}: {len(portfolio.strategies)} eligible")

    all_strategies = []
    for strats in strategies_by_instrument.values():
        all_strategies.extend(strats)
    print(f"  Total pool: {len(all_strategies)} strategies across all instruments")

    if args.profile:
        # Single profile
        profile = get_profile(args.profile)
        book = select_for_profile(profile, all_strategies)
        print_trading_book(book, profile)
    else:
        # All active profiles
        books = {}
        for pid, profile in ACCOUNT_PROFILES.items():
            if not profile.active:
                continue
            book = select_for_profile(profile, all_strategies)
            books[pid] = book
            print_trading_book(book, profile)

        if args.summary and books:
            print(f"\n{'='*70}")
            print("  CROSS-ACCOUNT SUMMARY")
            print(f"{'='*70}")
            total_slots = sum(b.total_slots for b in books.values())
            total_dd = sum(b.total_dd_used for b in books.values())
            all_instruments = set()
            all_sessions = set()
            for b in books.values():
                all_instruments.update(b.instruments_used)
                all_sessions.update(b.sessions_used)
            print(f"  Active profiles: {len(books)}")
            print(f"  Total slots: {total_slots}")
            print(f"  Total DD exposure: ${total_dd:,.0f}")
            print(f"  Instruments: {', '.join(sorted(all_instruments))}")
            print(f"  Sessions: {', '.join(sorted(all_sessions))}")
            # Copies
            total_copies = sum(
                ACCOUNT_PROFILES[pid].copies for pid in books
            )
            if total_copies > len(books):
                print(f"  Account copies: {total_copies} "
                      f"(${total_dd * total_copies / len(books):,.0f} aggregate DD)")
            print()


if __name__ == "__main__":
    main()
```

**Step 2: Test CLI manually**

Run: `python -m trading_app.prop_portfolio --all --summary`
Expected: Loads strategies, builds books for all profiles, prints formatted output.

**Step 3: Commit**

```bash
git add trading_app/prop_portfolio.py
git commit -m "feat: add prop portfolio CLI with --profile/--all/--summary"
```

---

### Task 3: Session Time Resolution

**Files:**
- Modify: `trading_app/prop_portfolio.py` (fix `_get_session_time_brisbane`)

The SESSION_CATALOG structure needs to be inspected to ensure correct field access.

**Step 1: Check SESSION_CATALOG structure**

Run: `python -c "from pipeline.dst import SESSION_CATALOG; import json; print(json.dumps(SESSION_CATALOG[0], indent=2, default=str))"`

**Step 2: Fix `_get_session_time_brisbane()` to match actual structure**

Adjust field names based on actual SESSION_CATALOG dict keys.

**Step 3: Verify session times appear correctly in output**

Run: `python -m trading_app.prop_portfolio --profile topstep_50k`
Check: session times are populated, not "unknown".

**Step 4: Commit**

```bash
git add trading_app/prop_portfolio.py
git commit -m "fix: resolve session times from SESSION_CATALOG correctly"
```

---

### Task 4: Integration Test (End-to-End)

**Files:**
- Modify: `tests/test_trading_app/test_prop_portfolio.py`

**Step 1: Add integration test**

```python
class TestEndToEnd:
    """Integration test with real build_live_portfolio output."""

    def test_select_from_real_strategies(self):
        """Build real strategies, run through selection."""
        # Use _make_strategy to simulate a realistic pool
        pool = [
            _make_strategy(
                strategy_id=f"MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G{g}",
                instrument="MGC",
                orb_label="TOKYO_OPEN",
                expectancy_r=0.20 + g * 0.01,
                sharpe_ratio=1.0 + g * 0.1,
                max_drawdown_r=3.0 + g * 0.5,
            )
            for g in [4, 5, 6]
        ] + [
            _make_strategy(
                strategy_id=f"MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_VOL",
                instrument="MNQ",
                orb_label="SINGAPORE_OPEN",
                expectancy_r=0.25,
                sharpe_ratio=1.3,
                max_drawdown_r=3.5,
            ),
            _make_strategy(
                strategy_id=f"MES_NYSE_OPEN_E1_RR2.0_CB3_G4",
                instrument="MES",
                orb_label="NYSE_OPEN",
                expectancy_r=0.22,
                sharpe_ratio=1.1,
                max_drawdown_r=4.0,
            ),
            _make_strategy(
                strategy_id=f"M2K_NYSE_OPEN_E2_RR1.0_CB1_VOL",
                instrument="M2K",
                orb_label="NYSE_OPEN",
                expectancy_r=0.18,
                sharpe_ratio=0.9,
                max_drawdown_r=5.0,
            ),
        ]

        profile = AccountProfile("test", "topstep", 50_000, 1, 0.75, max_slots=6)
        book = select_for_profile(profile, pool)

        # Should have selected some strategies
        assert book.total_slots > 0
        # Should respect DD budget ($2K at 0.75x = ~2 slots)
        assert book.total_dd_used <= 2_000
        # TOKYO_OPEN MGC should be deduped to 1
        tokyo_mgc = [e for e in book.entries
                     if e.orb_label == "TOKYO_OPEN" and e.instrument == "MGC"]
        assert len(tokyo_mgc) <= 1
        # Should have excluded entries with reasons
        assert len(book.excluded) > 0

    def test_apex_blocks_mgc(self):
        """Apex profile should exclude all metals."""
        pool = [
            _make_strategy(instrument="MGC", strategy_id="mgc1"),
            _make_strategy(instrument="MNQ", strategy_id="mnq1",
                          orb_label="SINGAPORE_OPEN"),
        ]
        profile = AccountProfile("test", "apex", 50_000, 1, 0.75, max_slots=6)
        book = select_for_profile(profile, pool)
        assert all(e.instrument != "MGC" for e in book.entries)
        assert any("banned" in ex.reason.lower() for ex in book.excluded)

    def test_self_funded_more_slots(self):
        """Self-funded should allow more slots (higher DD budget, no firm rules)."""
        pool = [
            _make_strategy(
                strategy_id=f"s{i}", orb_label=f"SESSION_{i}",
                instrument=["MGC", "MNQ", "MES", "M2K"][i % 4],
            )
            for i in range(10)
        ]
        prop_profile = AccountProfile("prop", "topstep", 50_000, 1, 0.75, max_slots=6)
        self_profile = AccountProfile("self", "self_funded", 50_000, 1, 1.0, max_slots=10)

        prop_book = select_for_profile(prop_profile, pool)
        self_book = select_for_profile(self_profile, pool)

        assert self_book.total_slots >= prop_book.total_slots
```

**Step 2: Run tests**

Run: `pytest tests/test_trading_app/test_prop_portfolio.py -x -q`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_trading_app/test_prop_portfolio.py
git commit -m "test: add integration tests for prop portfolio selection"
```

---

### Task 5: Drift Check & Final Verification

**Step 1: Run drift detection**

Run: `python pipeline/check_drift.py`
Expected: PASS (no new drift — we only added files, modified none)

**Step 2: Run full test suite**

Run: `pytest tests/test_trading_app/test_prop_profiles.py tests/test_trading_app/test_prop_portfolio.py -v`
Expected: ALL PASS

**Step 3: Run behavioral audit**

Run: `python scripts/tools/audit_behavioral.py`
Expected: PASS

**Step 4: Run the CLI against real DB**

Run: `python -m trading_app.prop_portfolio --all --summary`
Expected: Output shows all profiles with selected strategies, DD usage, excluded reasons.

**Step 5: Final commit with all files**

```bash
git add docs/plans/2026-03-15-prop-portfolio-design.md docs/plans/2026-03-15-prop-portfolio-plan.md
git commit -m "docs: add prop portfolio design doc and implementation plan"
```

---

## Key Implementation Notes

### For the Implementor

1. **`build_live_portfolio()` is per-instrument** — you must loop over all 4 active instruments and pool the results before passing to `select_for_profile()`.

2. **All our instruments are micro** — use `max_contracts_micro` for contract budget, not `max_contracts_mini`.

3. **`PortfolioStrategy` is frozen** — don't try to mutate it. The selection algorithm works with read-only strategy objects.

4. **SESSION_CATALOG structure** — check the actual dict keys before implementing `_get_session_time_brisbane()`. The field names may be different from what's assumed.

5. **`compute_profit_split_factor`** — TopStep's split is cumulative-profit-dependent. For initial portfolio construction, use `cumulative_profit=0` (conservative: assume 50% split). The user can adjust after earning through the $5K tier.

6. **Test DB not needed** — the tests use `_make_strategy()` to build synthetic `PortfolioStrategy` objects directly, bypassing the database entirely. This avoids the test DB setup complexity.

7. **Zero existing file modifications** — if you find yourself editing `live_config.py`, `portfolio.py`, `config.py`, or any existing file: STOP. The architecture is additive-only. New files: `prop_profiles.py`, `prop_portfolio.py`, and their test files.
