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

    @property
    def total_friction(self) -> float:
        """Total round-trip friction in dollars."""
        return self.commission_rt + self.spread_doubled + self.slippage

    @property
    def friction_in_points(self) -> float:
        """Total friction converted to price points."""
        return self.total_friction / self.point_value


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
    ),
}


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
    )
