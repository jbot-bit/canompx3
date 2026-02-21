#!/usr/bin/env python3
"""
Cost model for futures instruments.

CANONICAL source of truth for all friction and R-multiple calculations.
See CANONICAL_LOGIC.txt sections 2B and 10.

MGC (Micro Gold):
  Point value: $10/point
  Commission RT: $2.40
  Spread (doubled): $2.00
  Slippage: $4.00
  Total friction: $8.40 RT

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
    point_value: float        # Dollars per point of price movement
    commission_rt: float      # Round-trip commission ($)
    spread_doubled: float     # Spread cost doubled for entry+exit ($)
    slippage: float           # Expected slippage both sides ($)
    tick_size: float = 0.10   # Minimum price increment (points)
    min_ticks_floor: int = 10 # Minimum risk in ticks for stress test

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

COST_SPECS = {
    "MGC": CostSpec(
        instrument="MGC",
        point_value=10.0,
        commission_rt=2.40,
        spread_doubled=2.00,
        slippage=4.00,
        tick_size=0.10,
        min_ticks_floor=10,
    ),
    "MNQ": CostSpec(
        instrument="MNQ",
        point_value=2.0,
        commission_rt=1.24,
        spread_doubled=0.50,
        slippage=1.00,
        tick_size=0.25,
        min_ticks_floor=10,
    ),
    "MES": CostSpec(
        instrument="MES",
        point_value=5.0,         # $5 per index point
        commission_rt=1.24,      # Micro contract RT commission
        spread_doubled=1.25,     # 0.25pt spread * $5 * 2 sides (tick = 0.25)
        slippage=1.25,           # 0.25pt slippage * $5 * 2 sides
        tick_size=0.25,          # $0.25/point minimum increment
        min_ticks_floor=10,      # 2.5pt = $12.50 minimum risk
    ),
    "MCL": CostSpec(
        instrument="MCL",
        point_value=100.0,       # 100 barrels * $1/barrel per point
        commission_rt=1.24,      # Micro contract RT commission
        spread_doubled=2.00,     # ~$0.01 spread * 100 barrels * 2 sides
        slippage=2.00,           # ~$0.01 slippage * 100 barrels * 2 sides
        tick_size=0.01,          # $0.01/barrel minimum increment
        min_ticks_floor=10,      # $0.10 = $10 minimum risk
    ),
    "SIL": CostSpec(
        instrument="SIL",
        point_value=1000.0,      # 1000 oz * $1/oz per point
        commission_rt=1.24,      # Micro contract RT commission
        spread_doubled=10.00,    # ~$0.005 spread * 1000 oz * 2 sides
        slippage=10.00,          # ~$0.005 slippage * 1000 oz * 2 sides
        tick_size=0.005,         # $0.005/oz minimum increment
        min_ticks_floor=10,      # $0.05 = $50 minimum risk
    ),
    "M6E": CostSpec(
        instrument="M6E",
        point_value=12500.0,     # Micro EUR/USD: 12,500 EUR × $1 per EUR per point
        commission_rt=1.24,      # IB micro futures RT commission
        spread_doubled=1.25,     # ~1 tick ($0.625) each side (tick = 0.00005)
        slippage=1.25,           # ~1 tick ($0.625) each side (measured from actual data)
        tick_size=0.00005,       # Minimum price increment (half-pip for EUR/USD)
        min_ticks_floor=10,      # 10 ticks = 0.0005 = 5 pips minimum risk
    ),
    "M2K": CostSpec(
        instrument="M2K",
        point_value=5.0,         # Micro E-mini Russell 2000: $5 per index point
        commission_rt=1.24,      # IB micro futures RT commission
        spread_doubled=1.00,     # ~1 tick ($0.50) each side (tick = 0.10pt = $0.50)
        slippage=1.00,           # ~1 tick ($0.50) each side
        tick_size=0.10,          # Minimum price increment (0.10 index points)
        min_ticks_floor=10,      # 10 ticks = 1.0pt = $5 minimum risk
    ),
}


# =============================================================================
# SESSION-AWARE SLIPPAGE (live execution only — backtest uses flat costs)
# =============================================================================

# Per-instrument session slippage multipliers (live execution only)
# Keys: instrument -> orb_label -> multiplier
# Fallback: instrument not found -> use 1.0 for all sessions
SESSION_SLIPPAGE_MULT = {
    "MGC": {
        "0900": 1.3,   # 23:00 UTC -- thin Asian session
        "1000": 1.2,   # 00:00 UTC -- thin early Asian
        "1100": 1.0,   # 01:00 UTC -- moderate
        "1800": 0.9,   # 08:00 UTC -- pre-London, decent liquidity
        "2300": 0.8,   # 13:00 UTC -- NY session, best liquidity
        "0030": 1.1,   # 14:30 UTC -- moderate NY
    },
    "MNQ": {
        "0900": 1.0,   # 23:00 UTC -- NQ liquid even in Asian
        "1000": 1.0,   # 00:00 UTC -- early Asian
        "1100": 0.9,   # 01:00 UTC -- moderate
        "1800": 0.9,   # 08:00 UTC -- pre-London
        "2300": 0.8,   # 13:00 UTC -- NY session, best liquidity
        "0030": 0.9,   # 14:30 UTC -- moderate NY
    },
    "MES": {
        "0900": 1.0,   # 23:00 UTC -- ES/MES liquid 24h
        "1000": 1.0,   # 00:00 UTC -- early Asian
        "1100": 0.9,   # 01:00 UTC -- moderate
        "1800": 0.9,   # 08:00 UTC -- pre-London
        "2300": 0.8,   # 13:00 UTC -- NY session, best liquidity
        "0030": 0.9,   # 14:30 UTC -- moderate NY
    },
    "MCL": {
        "0900": 1.2,   # 23:00 UTC -- thin Asian session for crude
        "1000": 1.1,   # 00:00 UTC -- early Asian
        "1100": 1.0,   # 01:00 UTC -- moderate
        "1800": 0.9,   # 08:00 UTC -- pre-London, decent crude liquidity
        "2300": 0.8,   # 13:00 UTC -- NY/NYMEX session, best liquidity
        "0030": 1.0,   # 14:30 UTC -- moderate NY
    },
    "SIL": {
        # Defaults — update after volume analysis determines actual session list
        "0900": 1.3,   # 23:00 UTC -- thin Asian session for silver
        "1000": 1.2,   # 00:00 UTC -- early Asian
        "1100": 1.0,   # 01:00 UTC -- moderate
        "1800": 0.9,   # 08:00 UTC -- pre-London, decent liquidity
        "2300": 0.8,   # 13:00 UTC -- NY/COMEX session, best liquidity
        "0030": 1.0,   # 14:30 UTC -- moderate NY
    },
    "M6E": {
        # EUR/USD liquidity peaks at London open and tapers through US session
        "1000": 1.3,          # 00:00 UTC -- thin Asian FX; EUR/USD less active pre-London
        "1100": 1.2,          # 01:00 UTC -- early Asian; slight improvement
        "1800": 0.8,          # 08:00 UTC -- London open; best EUR/USD liquidity
        "0030": 1.0,          # 14:30 UTC -- US equity open; decent EUR/USD flow
        "LONDON_OPEN": 0.8,   # Dynamic London open; peak FX liquidity
        "US_DATA_OPEN": 1.5,  # 08:30 ET data release; wide spreads during news
        "US_EQUITY_OPEN": 1.0,  # 09:30 ET; solid EUR/USD depth
        "US_POST_EQUITY": 1.0,  # 10:00 ET; moderate
    },
    "M2K": {
        # Russell 2000 micro — same equity session structure as MES/MNQ
        "0900": 1.1,   # 23:00 UTC -- thin Asian (RTY less liquid than ES overnight)
        "1000": 1.1,   # 00:00 UTC -- early Asian
        "1100": 1.0,   # 01:00 UTC -- moderate
        "1800": 1.0,   # 08:00 UTC -- pre-NY, picking up
        "0030": 0.9,   # 14:30 UTC -- US equity open, peak Russell liquidity
        "US_EQUITY_OPEN": 0.9,   # Dynamic 9:30 ET; best M2K liquidity
        "US_DATA_OPEN": 1.3,     # 08:30 ET data; wider spreads during release
        "US_POST_EQUITY": 0.9,   # 10:00 ET; solid
        "CME_CLOSE": 1.0,        # 2:45 PM ET; moderate
    },
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


def to_r_multiple(spec: CostSpec, entry: float, stop: float,
                   pnl_points: float) -> float:
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


def pnl_points_to_r(spec: CostSpec, entry: float, stop: float,
                     pnl_points: float) -> float:
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
