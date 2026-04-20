#!/usr/bin/env python3
"""
Cost model for futures instruments.

CANONICAL source of truth for all friction and R-multiple calculations.
See CANONICAL_LOGIC.txt sections 2B and 10.

MGC (Micro Gold):
  Point value: $10/point
  Commission RT: $1.74
  Spread (doubled): $2.00
  Slippage: $2.00
  Total friction: $5.74 RT

FAIL-CLOSED: Unknown instruments raise ValueError.

Usage:
    from pipeline.cost_model import get_cost_spec, to_r_multiple, realized_rr

    spec = get_cost_spec("MGC")
    r = to_r_multiple(spec, entry=2350.0, stop=2340.0, pnl_points=5.0)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostSpec:
    """Immutable cost specification for a futures instrument."""

    instrument: str
    point_value: float  # Dollars per point of price movement
    commission_rt: float  # Round-trip commission ($)
    spread_doubled: float  # Spread cost doubled for entry+exit ($)
    slippage: float  # Expected slippage both sides ($)
    tick_size: float = 0.10  # Minimum price increment (points)
    min_ticks_floor: int = 10  # Minimum risk in ticks for stress test

    def __post_init__(self):
        if self.point_value <= 0:
            raise ValueError(f"point_value must be positive, got {self.point_value}")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")

    @property
    def total_friction(self) -> float:
        """Total round-trip friction in dollars."""
        return self.commission_rt + self.spread_doubled + self.slippage

    @property
    def friction_in_points(self) -> float:
        """Total friction converted to price points."""
        return self.total_friction / self.point_value

    @property
    def min_risk_floor_points(self) -> float:
        """Minimum risk floor in points (tick-based)."""
        return self.min_ticks_floor * self.tick_size

    @property
    def min_risk_floor_dollars(self) -> float:
        """Minimum risk floor in dollars (tick-based)."""
        return self.min_risk_floor_points * self.point_value


# =============================================================================
# INSTRUMENT COST SPECS
# =============================================================================

# ─── TopStep canonical commissions (round-trip $) ─────────────────────
# @canonical-source docs/research-input/topstep/topstep_xfa_commissions.md  (article 8284213, scraped 2026-04-08)
# @verbatim-section "Tradovate (Tradovate/TradingView)" and "Rithmic" tables
# @verbatim "Fees updated as of May 12th, 2025"
# @audit-finding F-4 (HIGH — initial code values 8-13% LOW for MNQ/MES vs canonical)
#
# Canonical TopStep RT rates by clearing platform:
#                    Tradovate    Rithmic
#   MGC   (Gold)     $1.64        $1.72
#   MNQ   (NQ)       $1.34        $1.42
#   MES   (ES)       $1.34        $1.42
#   MCL   (Crude)    $1.64        $1.72
#   SIL   (Silver)   $2.64        $2.72
#   M6E   (EUR/USD)  $1.12        $1.20
#   M2K   (Russell)  $1.34        $1.42
#   MBT   (Bitcoin)  $5.64        $5.72
#
# We use the higher (Rithmic) value as the conservative baseline because:
#   1. The user's TopStep XFA via ProjectX uses TopstepX's clearing path which is
#      not separately listed in the article (the dedicated TopstepX commissions
#      article 14363528 returned 404 on 2026-04-08).
#   2. Rithmic-based prop firms (Bulenox, Elite Trader Funding) use these rates
#      directly, so a Rithmic baseline is correct for those deployments too.
#   3. Conservative cost modeling cannot UNDER-estimate friction → backtest
#      results are floor-bounded vs reality.
# Per-firm overrides should be added when Tradeify (Tradovate rates) goes live.

COST_SPECS = {
    # @canonical-source docs/research-input/topstep/topstep_xfa_commissions.md
    # @verbatim "Micro Gold (MGC) Rithmic $1.72"
    # Code value $1.74 is $0.02 above Rithmic (within rounding); keeping for
    # historical backtest stability and conservative bias. F-4 verified.
    "MGC": CostSpec(
        instrument="MGC",
        point_value=10.0,
        commission_rt=1.74,  # canonical Rithmic = $1.72 (within rounding); keep for stability
        spread_doubled=2.00,
        slippage=2.00,
        tick_size=0.10,
        min_ticks_floor=10,
    ),
    # GC = full-size Gold (100 troy oz, $100/pt). Parent contract of MGC.
    # Added for Amendment 3.1 (GC proxy for MGC price-based discovery).
    # Cost specs are 10x MGC by contract multiplier. Commission estimated
    # proportionally — exact broker commission varies but does NOT affect
    # pnl_r (R-multiples are price-based). COST_LT thresholds are identical
    # in points because both numerator and denominator scale by 10x.
    # @research-source: Amendment 3.1 (pre_registered_criteria.md, 2026-04-10)
    "GC": CostSpec(
        instrument="GC",
        point_value=100.0,  # 100 troy oz * $1/oz per point
        commission_rt=17.40,  # 10x MGC ($1.74 * 10). Not canonical — GC not traded live
        spread_doubled=20.00,  # 2 ticks * $10/tick = $20 (10x MGC's $2). Dollar terms!
        slippage=20.00,  # 2 ticks * $10/tick = $20 (10x MGC's $2). Dollar terms!
        tick_size=0.10,  # $0.10/oz = $10/tick (vs MGC $1/tick)
        min_ticks_floor=10,  # 1.0pt = $100 minimum risk
    ),
    # NQ = E-mini Nasdaq 100 (full-size). 10x MNQ by contract multiplier.
    # Same price, same tick size, same sessions — only the point value differs.
    # Commission is a flat per-contract fee, so friction ratio drops ~10x vs MNQ.
    # Spread/slippage scale linearly with point value (same number of ticks).
    # @canonical-source docs/research-input/topstep/topstep_xfa_commissions.md
    # @verbatim "E-mini NASDAQ 100 (NQ) Tradovate $3.22 / Rithmic $4.10"
    # NQ commission is higher than MNQ ($4.10 vs $1.42) but per-point cost is
    # $4.10 / $20pt = $0.205/pt vs MNQ $1.42 / $2pt = $0.71/pt — 3.5x cheaper.
    "NQ": CostSpec(
        instrument="NQ",
        point_value=20.0,  # $20 per index point (10x MNQ)
        commission_rt=4.10,  # canonical TopStep Rithmic (higher than MNQ flat rate)
        spread_doubled=5.00,  # 10x MNQ's $0.50 (same 1-tick spread, 10x $/tick)
        slippage=10.00,  # 10x MNQ's $1.00 (same tick-based slippage model)
        tick_size=0.25,  # Same tick size as MNQ
        min_ticks_floor=10,  # 10 ticks = 2.5pt = $50 minimum risk
    ),
    # MNQ slippage model: 1 tick round-trip ($1.00 = 2 ticks × $0.50/tick).
    #
    # MNQ TBBO pilot (2026-04-20, N=114, 2021-02-10 to 2026-02-12,
    # 6 sessions: CME_PRECLOSE, LONDON_METALS, NYSE_OPEN, SINGAPORE_OPEN,
    # TOKYO_OPEN, US_DATA_830): MEDIAN=0 ticks, p95=0.35 ticks, MAX=+2 ticks,
    # 100% of days ≤ 2 ticks. Modeled slippage is CONSERVATIVE vs measured
    # on routine days. Deployed-lane subset (NYSE_OPEN / SINGAPORE_OPEN /
    # TOKYO_OPEN, N=56): median=0, max=+1 tick, 100% ≤ 1 tick. Mean=-0.93
    # is dominated by 4 BBO-staleness rows (spread ≥ 3 ticks at trigger);
    # floor-at-zero conservative read gives mean ≈ +0.1 tick.
    # Full doc: docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v1.md
    #
    # MGC TBBO pilot earlier (research/output/mgc_e2_slippage_analysis.json):
    # N=40, MEDIAN=0, mean=6.75 dominated by ONE day (2018-01-18 gap-open
    # event, 263 ticks). Trimmed mean ≈0.18 ticks. Honest central tendency
    # is "both instruments fill at modeled routinely."
    #
    # STILL OPEN:
    # - MNQ sample missing EUROPE_FLOW / COMEX_SETTLE / US_DATA_1000 (3 of 5
    #   deployed sessions absent from cache)
    # - Event-day tail NOT measured for MNQ (sample had no MGC-2018-type gap)
    # - MES TBBO pilot has NOT been run. Book-wide event-day tail unquantified.
    # - Phase D MNQ COMEX_SETTLE pilot gate (2026-05-15) benefits from a
    #   targeted COMEX_SETTLE TBBO pull before evaluation.
    #
    # Break-even analysis (scripts/tools/slippage_scenario.py) for REFERENCE:
    #   COMEX_SETTLE: 4.9 extra ticks to zero (FRAGILE)
    #   SINGAPORE_OPEN: 6.0 extra ticks to zero
    #   NYSE_CLOSE: 15.4 extra ticks (robust)
    #   NYSE_OPEN: 17.7 extra ticks (robust)
    # SESSION_SLIPPAGE_MULT exists for live but is NOT used in backtests.
    # @canonical-source docs/research-input/topstep/topstep_xfa_commissions.md
    # @verbatim "Micro E-mini NASDAQ 100 (MNQ) Tradovate $1.34 / Rithmic $1.42"
    # @audit-finding F-4 — bumped from $1.24 to $1.42 (canonical TopStep Rithmic).
    # Backtest impact: +$0.18/RT × N trades. For active topstep_50k_mnq_auto with
    # 4 MNQ lanes × ~150 trades/yr ≈ 600 RT/yr → ~$108/yr more friction modeled.
    "MNQ": CostSpec(
        instrument="MNQ",
        point_value=2.0,
        commission_rt=1.42,  # canonical TopStep Rithmic (was $1.24 — F-4 fix)
        spread_doubled=0.50,
        slippage=1.00,
        tick_size=0.25,
        min_ticks_floor=10,
    ),
    # @canonical-source docs/research-input/topstep/topstep_xfa_commissions.md
    # @verbatim "Micro E-mini S&P (MES) Tradovate $1.34 / Rithmic $1.42"
    # @audit-finding F-4 — bumped from $1.24 to $1.42 (canonical TopStep Rithmic).
    "MES": CostSpec(
        instrument="MES",
        point_value=5.0,  # $5 per index point
        commission_rt=1.42,  # canonical TopStep Rithmic (was $1.24 — F-4 fix)
        spread_doubled=1.25,  # 0.25pt spread * $5 * 2 sides (tick = 0.25)
        slippage=1.25,  # 0.25pt slippage * $5 * 2 sides
        tick_size=0.25,  # $0.25/point minimum increment
        min_ticks_floor=10,  # 2.5pt = $12.50 minimum risk
    ),
    "MCL": CostSpec(
        instrument="MCL",
        point_value=100.0,  # 100 barrels * $1/barrel per point
        commission_rt=1.24,  # Micro contract RT commission
        spread_doubled=2.00,  # ~$0.01 spread * 100 barrels * 2 sides
        slippage=2.00,  # ~$0.01 slippage * 100 barrels * 2 sides
        tick_size=0.01,  # $0.01/barrel minimum increment
        min_ticks_floor=10,  # $0.10 = $10 minimum risk
    ),
    "SIL": CostSpec(
        instrument="SIL",
        point_value=1000.0,  # 1000 oz * $1/oz per point
        commission_rt=1.24,  # Micro contract RT commission
        spread_doubled=10.00,  # ~$0.005 spread * 1000 oz * 2 sides
        slippage=10.00,  # ~$0.005 slippage * 1000 oz * 2 sides
        tick_size=0.005,  # $0.005/oz minimum increment
        min_ticks_floor=10,  # $0.05 = $50 minimum risk
    ),
    "M6E": CostSpec(
        instrument="M6E",
        point_value=12500.0,  # Micro EUR/USD: 12,500 EUR × $1 per EUR per point
        commission_rt=1.24,  # IB micro futures RT commission
        spread_doubled=1.25,  # ~1 tick ($0.625) each side (tick = 0.00005)
        slippage=1.25,  # ~1 tick ($0.625) each side (measured from actual data)
        tick_size=0.00005,  # Minimum price increment (half-pip for EUR/USD)
        min_ticks_floor=10,  # 10 ticks = 0.0005 = 5 pips minimum risk
    ),
    "M2K": CostSpec(
        instrument="M2K",
        point_value=5.0,  # Micro E-mini Russell 2000: $5 per index point
        commission_rt=1.24,  # IB micro futures RT commission
        spread_doubled=1.00,  # ~1 tick ($0.50) each side (tick = 0.10pt = $0.50)
        slippage=1.00,  # ~1 tick ($0.50) each side
        tick_size=0.10,  # Minimum price increment (0.10 index points)
        min_ticks_floor=10,  # 10 ticks = 1.0pt = $5 minimum risk
    ),
    "MBT": CostSpec(
        instrument="MBT",
        point_value=0.10,  # Micro Bitcoin: 0.1 BTC per contract, $0.10 per $1 of BTC price
        commission_rt=2.50,  # MBT higher commission than other micros (~$1.25/side at IB)
        spread_doubled=2.00,  # ~2 ticks ($1.00) per side × 2 (tick = $5 BTC = $0.50 MBT)
        slippage=2.00,  # ~2 ticks ($1.00) per side × 2 (E2 stop-market fill)
        tick_size=5.0,  # $5 minimum price increment in BTC price
        min_ticks_floor=10,  # 10 ticks = $50 BTC price = $5.00 MBT minimum risk
    ),
}


# =============================================================================
# COST MODEL LIMITATIONS (adversarial review 2026-03-18)
# =============================================================================
# Backtest uses FLAT slippage per instrument. Real slippage likely correlates
# with ORB size and session liquidity — bigger breakouts = more competition
# at the entry level = more slippage. The 1.5x stress test partially addresses
# this but is a fixed multiplier, not a function of trade characteristics.
#
# Until paper trading fills provide actual TCA data (Apex Phase 1), the cost
# model is structurally optimistic for ORB entries. The magnitude is unknown —
# the adversarial review suggested 2.0-2.5x but provided no evidence.
# Paper trading kill criteria: if actual slippage > 2x modeled → cost model
# is wrong and all backtest results are overstated.
#
# SESSION_SLIPPAGE_MULT below applies to LIVE execution only, not backtests.
# =============================================================================

# =============================================================================
# SESSION-AWARE SLIPPAGE (live execution only — backtest uses flat costs)
# =============================================================================

# Per-instrument session slippage multipliers (live execution only)
# Keys: instrument -> orb_label -> multiplier
# Fallback: instrument not found -> use 1.0 for all sessions
SESSION_SLIPPAGE_MULT = {
    "MGC": {
        "CME_REOPEN": 1.3,  # 23:00 UTC -- thin Asian session
        "TOKYO_OPEN": 1.2,  # 00:00 UTC -- thin early Asian
        "BRISBANE_1025": 1.2,  # 00:25 UTC -- adjacent to TOKYO
        "SINGAPORE_OPEN": 1.0,  # 01:00 UTC -- moderate
        "EUROPE_FLOW": 0.9,  # ~08:00 UTC -- pre-London (adjacent to LONDON_METALS)
        "LONDON_METALS": 0.9,  # 08:00 UTC -- pre-London, decent liquidity
        "US_DATA_830": 0.8,  # 13:00 UTC -- NY session, best liquidity
        "NYSE_OPEN": 1.1,  # 14:30 UTC -- moderate NY
        "US_DATA_1000": 0.9,  # ~14:00 UTC -- post-equity-open flow
        "COMEX_SETTLE": 0.9,  # ~18:25 UTC -- settlement window
        "CME_PRECLOSE": 1.0,  # ~19:45 UTC -- closing session
        "NYSE_CLOSE": 1.0,  # ~20:00 UTC -- NYSE close
    },
    "MNQ": {
        "CME_REOPEN": 1.0,  # 23:00 UTC -- NQ liquid even in Asian
        "TOKYO_OPEN": 1.0,  # 00:00 UTC -- early Asian
        "BRISBANE_1025": 1.0,  # 00:25 UTC -- adjacent to TOKYO
        "SINGAPORE_OPEN": 0.9,  # 01:00 UTC -- moderate
        "EUROPE_FLOW": 0.9,  # ~08:00 UTC -- pre-London (adjacent to LONDON_METALS)
        "LONDON_METALS": 0.9,  # 08:00 UTC -- pre-London
        "US_DATA_830": 0.8,  # 13:00 UTC -- NY session, best liquidity
        "NYSE_OPEN": 0.9,  # 14:30 UTC -- moderate NY
        "US_DATA_1000": 0.9,  # ~14:00 UTC -- post-equity-open flow
        "COMEX_SETTLE": 0.9,  # ~18:25 UTC -- settlement window
        "CME_PRECLOSE": 1.0,  # ~19:45 UTC -- closing session
        "NYSE_CLOSE": 0.9,  # ~20:00 UTC -- NYSE close
    },
    # NQ = full-size Nasdaq. Same sessions as MNQ, same or better liquidity.
    # Using MNQ multipliers as conservative baseline (NQ is the primary contract).
    "NQ": {
        "CME_REOPEN": 1.0,
        "TOKYO_OPEN": 1.0,
        "BRISBANE_1025": 1.0,
        "SINGAPORE_OPEN": 0.9,
        "EUROPE_FLOW": 0.9,
        "LONDON_METALS": 0.9,
        "US_DATA_830": 0.8,
        "NYSE_OPEN": 0.9,
        "US_DATA_1000": 0.9,
        "COMEX_SETTLE": 0.9,
        "CME_PRECLOSE": 1.0,
        "NYSE_CLOSE": 0.9,
    },
    "MES": {
        "CME_REOPEN": 1.0,  # 23:00 UTC -- ES/MES liquid 24h
        "TOKYO_OPEN": 1.0,  # 00:00 UTC -- early Asian
        "BRISBANE_1025": 1.0,  # 00:25 UTC -- adjacent to TOKYO
        "SINGAPORE_OPEN": 0.9,  # 01:00 UTC -- moderate
        "EUROPE_FLOW": 0.9,  # ~08:00 UTC -- pre-London (adjacent to LONDON_METALS)
        "LONDON_METALS": 0.9,  # 08:00 UTC -- pre-London
        "US_DATA_830": 0.8,  # 13:00 UTC -- NY session, best liquidity
        "NYSE_OPEN": 0.9,  # 14:30 UTC -- moderate NY
        "US_DATA_1000": 0.9,  # ~14:00 UTC -- post-equity-open flow
        "COMEX_SETTLE": 0.9,  # ~18:25 UTC -- settlement window (MES has COMEX-adjacent flow)
        "CME_PRECLOSE": 1.0,  # ~19:45 UTC -- closing session
        "NYSE_CLOSE": 0.9,  # ~20:00 UTC -- NYSE close
    },
    "MCL": {
        "CME_REOPEN": 1.2,  # 23:00 UTC -- thin Asian session for crude
        "TOKYO_OPEN": 1.1,  # 00:00 UTC -- early Asian
        "SINGAPORE_OPEN": 1.0,  # 01:00 UTC -- moderate
        "LONDON_METALS": 0.9,  # 08:00 UTC -- pre-London, decent crude liquidity
        "US_DATA_830": 0.8,  # 13:00 UTC -- NY/NYMEX session, best liquidity
        "NYSE_OPEN": 1.0,  # 14:30 UTC -- moderate NY
    },
    "SIL": {
        # Defaults — update after volume analysis determines actual session list
        "CME_REOPEN": 1.3,  # 23:00 UTC -- thin Asian session for silver
        "TOKYO_OPEN": 1.2,  # 00:00 UTC -- early Asian
        "SINGAPORE_OPEN": 1.0,  # 01:00 UTC -- moderate
        "LONDON_METALS": 0.9,  # 08:00 UTC -- pre-London, decent liquidity
        "US_DATA_830": 0.8,  # 13:00 UTC -- NY/COMEX session, best liquidity
        "NYSE_OPEN": 1.0,  # 14:30 UTC -- moderate NY
        "COMEX_SETTLE": 0.9,  # ~18:25 UTC -- settlement window
    },
    "M6E": {
        # EUR/USD liquidity peaks at London open and tapers through US session
        "TOKYO_OPEN": 1.3,  # 00:00 UTC -- thin Asian FX; EUR/USD less active pre-London
        "SINGAPORE_OPEN": 1.2,  # 01:00 UTC -- early Asian; slight improvement
        "LONDON_METALS": 0.8,  # 08:00 UTC -- London open; best EUR/USD liquidity
        "NYSE_OPEN": 1.0,  # 14:30 UTC -- US equity open; decent EUR/USD flow
        "US_DATA_830": 1.5,  # 08:30 ET data release; wide spreads during news
        "US_DATA_1000": 1.0,  # 10:00 ET; moderate
    },
    "M2K": {
        # Russell 2000 micro — same equity session structure as MES/MNQ
        "CME_REOPEN": 1.1,  # 23:00 UTC -- thin Asian (RTY less liquid than ES overnight)
        "TOKYO_OPEN": 1.1,  # 00:00 UTC -- early Asian
        "SINGAPORE_OPEN": 1.0,  # 01:00 UTC -- moderate
        "LONDON_METALS": 1.0,  # 08:00 UTC -- pre-NY, picking up
        "NYSE_OPEN": 0.9,  # 14:30 UTC -- US equity open, peak Russell liquidity
        "US_DATA_830": 1.3,  # 08:30 ET data; wider spreads during release
        "US_DATA_1000": 0.9,  # 10:00 ET; solid
        "CME_PRECLOSE": 1.0,  # 2:45 PM ET; moderate
    },
    # MBT (Micro Bitcoin) intentionally absent: dead instrument (0 validated ORB
    # strategies, no live trading). get_session_cost_spec() falls back to mult=1.0
    # which is correct — BTC's 24/7 trading means session multipliers don't apply.
}


def get_session_cost_spec(instrument: str, orb_label: str) -> CostSpec:
    """CostSpec with session-adjusted slippage for live execution.

    Applies a multiplier to the base slippage based on the session's
    typical liquidity. Backtest should NOT use this -- use get_cost_spec()
    with 1.5x stress test instead.
    """
    base = get_cost_spec(instrument)
    inst_mults = SESSION_SLIPPAGE_MULT.get(instrument.upper(), {})
    mult = inst_mults.get(orb_label, 1.0)
    if mult == 1.0:
        return base
    return CostSpec(
        instrument=base.instrument,
        point_value=base.point_value,
        commission_rt=base.commission_rt,
        spread_doubled=base.spread_doubled,
        slippage=round(base.slippage * mult, 2),
        tick_size=base.tick_size,
        min_ticks_floor=base.min_ticks_floor,
    )


def get_cost_spec(instrument: str) -> CostSpec:
    """
    Return the CostSpec for the given instrument.

    FAIL-CLOSED: Raises ValueError for unvalidated instruments.
    """
    instrument = instrument.upper()
    if instrument not in COST_SPECS:
        raise ValueError(
            f"No cost model for instrument '{instrument}'. "
            f"Validated instruments: {sorted(COST_SPECS.keys())}. "
            f"Add broker specs before using."
        )
    return COST_SPECS[instrument]


def list_validated_instruments() -> list[str]:
    """Return sorted list of instruments with validated cost models."""
    return sorted(COST_SPECS.keys())


# =============================================================================
# R-MULTIPLE CALCULATIONS
# =============================================================================


def risk_in_dollars(spec: CostSpec, entry: float, stop: float) -> float:
    """
    Realized risk in dollars (costs INCREASE risk).

    Realized_Risk_$ = |entry - stop| * point_value + total_friction
    """
    raw_risk = abs(entry - stop) * spec.point_value
    return raw_risk + spec.total_friction


def reward_in_dollars(spec: CostSpec, entry: float, target: float) -> float:
    """
    Realized reward in dollars (costs REDUCE reward).

    Realized_Reward_$ = |target - entry| * point_value - total_friction
    """
    raw_reward = abs(target - entry) * spec.point_value
    return raw_reward - spec.total_friction


def realized_rr(spec: CostSpec, entry: float, stop: float, target: float) -> float:
    """
    Realized risk-reward ratio after costs.

    Realized_RR = Realized_Reward_$ / Realized_Risk_$
    """
    risk = risk_in_dollars(spec, entry, stop)
    rew = reward_in_dollars(spec, entry, target)

    if risk <= 0:
        raise ValueError(f"Risk must be positive, got {risk}")

    return rew / risk


def to_r_multiple(spec: CostSpec, entry: float, stop: float, pnl_points: float) -> float:
    """
    Convert a P&L in price points to an R-multiple.

    R = (pnl_points * point_value - total_friction) / risk_in_dollars

    This is used for MAE/MFE calculations where we know the
    adverse/favorable excursion in points and want it in R.
    """
    risk = risk_in_dollars(spec, entry, stop)
    if risk <= 0:
        raise ValueError(f"Risk must be positive, got {risk}")

    pnl_dollars = pnl_points * spec.point_value - spec.total_friction
    return pnl_dollars / risk


def pnl_points_to_r(spec: CostSpec, entry: float, stop: float, pnl_points: float) -> float:
    """
    Convert raw P&L points to R-multiple WITHOUT deducting friction from P&L.

    Used for MAE/MFE where the excursion is measured from entry,
    and friction is only in the denominator (risk).

    R = (pnl_points * point_value) / risk_in_dollars
    """
    risk = risk_in_dollars(spec, entry, stop)
    if risk <= 0:
        raise ValueError(f"Risk must be positive, got {risk}")

    return (pnl_points * spec.point_value) / risk


def stress_test_costs(spec: CostSpec, multiplier: float = 1.5) -> CostSpec:
    """
    Return a CostSpec with friction increased by the given multiplier.

    Default: +50% costs per CANONICAL_LOGIC.txt section 9.
    """
    return CostSpec(
        instrument=spec.instrument,
        point_value=spec.point_value,
        commission_rt=spec.commission_rt * multiplier,
        spread_doubled=spec.spread_doubled * multiplier,
        slippage=spec.slippage * multiplier,
        tick_size=spec.tick_size,
        min_ticks_floor=spec.min_ticks_floor,
    )
