"""Core schemas for the medium-frequency futures kernel."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class InstrumentConfig:
    """Static metadata for one research/live instrument mapping."""

    research_symbol: str
    live_symbol: str
    asset_class: str
    sector: str
    research_price_symbol: str | None = None
    stats_symbol: str | None = None
    contract_multiplier: float | None = None


@dataclass(frozen=True)
class TrendSleeveConfig:
    """Fixed EWMAC parameter set for one sleeve."""

    fast_span: int
    slow_span: int
    weight: float


@dataclass(frozen=True)
class ForecastCombiner:
    """Combine and cap multiple forecast components."""

    target_abs_forecast: float = 10.0
    forecast_cap: float = 20.0


@dataclass(frozen=True)
class PortfolioPolicy:
    """Research-only portfolio sizing policy."""

    annualized_vol_target: float
    diversification_multiplier_cap: float = 2.5
    inertia_band_pct: float = 0.10


@dataclass(frozen=True)
class CombinedForecast:
    """Final bounded daily forecast for an instrument."""

    trading_day: date
    symbol: str
    trend_forecast: float
    carry_forecast: float
    combined_forecast: float


@dataclass(frozen=True)
class TargetPosition:
    """Research-only target position in contracts."""

    trading_day: date
    symbol: str
    target_contracts: int
    current_contracts: int
    effective_contracts: int
    contract_notional_usd: float
    annualized_vol: float
    combined_forecast: float


@dataclass(frozen=True)
class ExecutionIntent:
    """Publish-only intent derived from current vs target state."""

    trading_day: date
    symbol: str
    current_contracts: int
    target_contracts: int
    delta_contracts: int
    action: str


@dataclass(frozen=True)
class DailyMarketSnapshot:
    """One no-lookahead daily snapshot for the medium-frequency kernel."""

    trading_day: date
    symbol: str
    research_symbol: str
    stats_symbol: str | None
    research_close: float
    contract_price: float
    front_contract: str | None
    front_settlement: float | None
    open_interest: int | None
    cleared_volume: int | None
    carry_available: bool
    coverage_note: str | None = None


@dataclass(frozen=True)
class ContractObservation:
    """Same-day per-contract observation from canonical statistics sources."""

    trading_day: date
    symbol: str
    contract_symbol: str
    contract_year: int
    contract_month: int
    settlement: float | None
    cleared_volume: int | None
    open_interest: int | None
    expiry_date: date | None = None
    source: str = "raw_statistics"


@dataclass(frozen=True)
class FrontNextPair:
    """Deterministic front/next pairing for one trading day."""

    trading_day: date
    symbol: str
    front: ContractObservation | None
    next_contract: ContractObservation | None
    selection_metric: str | None
    contract_gap_months: int | None
    calendar_gap_days: int | None
    carry_available: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class CarryInputSlice:
    """Honest carry inputs for one trading day, or explicit unavailability."""

    trading_day: date
    symbol: str
    front_contract: str | None
    next_contract: str | None
    front_price: float | None
    next_price: float | None
    price_spread: float | None
    price_ratio: float | None
    selection_metric: str | None
    contract_gap_months: int | None
    calendar_gap_days: int | None
    annualized_carry: float | None
    carry_available: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class KernelInputSlice:
    """Thin adapter payload from assembled snapshots into kernel inputs."""

    trading_day: date
    symbol: str
    price_history: tuple[float, ...]
    annualized_vol: float
    contract_price: float
    contract_notional_usd: float | None
    front_contract: str | None
    front_price: float | None
    next_price: float | None = None
    days_between_expiries: int | None = None
    carry_available: bool = False
    carry_reason_code: str | None = None
