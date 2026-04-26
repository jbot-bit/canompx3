"""Pure no-lookahead research functions for the medium-frequency futures kernel."""

from __future__ import annotations

import math
from collections.abc import Sequence

TRADING_DAYS_PER_YEAR = 252.0


def _ema(values: Sequence[float], span: int) -> float:
    """Return the final EMA value over a backward-looking series."""
    if span <= 0:
        raise ValueError("span must be positive")
    if not values:
        raise ValueError("values must be non-empty")

    alpha = 2.0 / (span + 1.0)
    ema = float(values[0])
    for value in values[1:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema
    return ema


def ewmac_forecast(
    prices: Sequence[float],
    *,
    fast_span: int,
    slow_span: int,
    annualized_vol: float,
    forecast_cap: float = 20.0,
) -> float:
    """Compute a bounded EWMAC forecast from a backward-looking price series.

    The caller is responsible for ensuring `prices` contains only observations
    known at the decision timestamp.
    """
    if slow_span <= fast_span:
        raise ValueError("slow_span must be greater than fast_span")
    if annualized_vol <= 0:
        raise ValueError("annualized_vol must be positive")

    fast = _ema(prices, fast_span)
    slow = _ema(prices, slow_span)
    raw_signal = fast - slow

    # Convert annualized return volatility to daily volatility so that the
    # signal is dimensionless and comparable across instruments.
    daily_vol = annualized_vol / math.sqrt(TRADING_DAYS_PER_YEAR)
    normalized = raw_signal / daily_vol
    return max(-forecast_cap, min(forecast_cap, normalized))


def annualized_carry_from_curve(
    *,
    front_price: float,
    next_price: float,
    days_between_expiries: int,
    forecast_cap: float = 20.0,
) -> float:
    """Compute bounded annualized futures carry from the front/next curve.

    Positive values imply positive long carry (for example, backwardation where
    front > next). The function uses same-day observable curve points only.
    """
    if front_price <= 0 or next_price <= 0:
        raise ValueError("curve prices must be positive")
    if days_between_expiries <= 0:
        raise ValueError("days_between_expiries must be positive")

    annualized = math.log(front_price / next_price) * (365.0 / float(days_between_expiries))
    return max(-forecast_cap, min(forecast_cap, annualized))


def bounded_weighted_forecast(
    components: Sequence[tuple[float, float]],
    *,
    target_abs_forecast: float = 10.0,
    forecast_cap: float = 20.0,
) -> float:
    """Combine weighted components into a bounded forecast.

    Components are `(forecast_value, weight)` pairs. The output is rescaled so
    that a unit-weight average near 1.0 maps near `target_abs_forecast`.
    """
    if not components:
        raise ValueError("components must be non-empty")

    total_weight = sum(weight for _, weight in components)
    if total_weight <= 0:
        raise ValueError("total weight must be positive")

    weighted_average = sum(value * weight for value, weight in components) / total_weight
    scaled = weighted_average * target_abs_forecast
    return max(-forecast_cap, min(forecast_cap, scaled))


def target_contracts_from_vol(
    *,
    capital_usd: float,
    annualized_vol_target: float,
    contract_notional_usd: float,
    annualized_instrument_vol: float,
    combined_forecast: float,
    instrument_weight: float = 1.0,
    diversification_multiplier: float = 1.0,
    forecast_target: float = 10.0,
) -> int:
    """Compute an integer target position using a Carver-style volatility budget.

    The function is research-only and assumes the caller already chose a
    tradable contract and computed a same-day contract notional.
    """
    if capital_usd <= 0:
        raise ValueError("capital_usd must be positive")
    if annualized_vol_target <= 0:
        raise ValueError("annualized_vol_target must be positive")
    if contract_notional_usd <= 0:
        raise ValueError("contract_notional_usd must be positive")
    if annualized_instrument_vol <= 0:
        raise ValueError("annualized_instrument_vol must be positive")
    if forecast_target <= 0:
        raise ValueError("forecast_target must be positive")
    if instrument_weight < 0 or diversification_multiplier < 0:
        raise ValueError("weights must be non-negative")

    daily_cash_vol_target = capital_usd * annualized_vol_target / math.sqrt(TRADING_DAYS_PER_YEAR)
    daily_contract_cash_vol = contract_notional_usd * annualized_instrument_vol / math.sqrt(TRADING_DAYS_PER_YEAR)
    if daily_contract_cash_vol <= 0:
        return 0

    normalized_forecast = combined_forecast / forecast_target
    raw_contracts = (
        daily_cash_vol_target
        / daily_contract_cash_vol
        * normalized_forecast
        * instrument_weight
        * diversification_multiplier
    )
    return int(round(raw_contracts))


def apply_inertia_band(*, current_contracts: int, target_contracts: int, inertia_band_pct: float = 0.10) -> int:
    """Keep the current target if the new target is within the inertia band."""
    if inertia_band_pct < 0:
        raise ValueError("inertia_band_pct must be non-negative")
    if current_contracts == 0:
        return target_contracts

    threshold = max(1.0, abs(current_contracts) * inertia_band_pct)
    if abs(target_contracts - current_contracts) <= threshold:
        return current_contracts
    return target_contracts
