"""Research-only medium-frequency futures kernel.

This package is deliberately separate from the ORB runtime.
It owns daily-signal alpha logic only.
"""

from .carry import build_carry_input_slice
from .config import DEFAULT_COMBINER, DEFAULT_PORTFOLIO_POLICY, DEFAULT_TREND_SLEEVE
from .contracts import (
    build_front_next_pair,
    discover_statistics_files,
    load_contract_observations,
    pair_observations_by_day,
)
from .expiry import compute_expiry_date, supported_expiry_rule
from .kernel import (
    annualized_carry_from_curve,
    apply_inertia_band,
    bounded_weighted_forecast,
    ewmac_forecast,
    target_contracts_from_vol,
)
from .models import (
    CarryInputSlice,
    CombinedForecast,
    ContractObservation,
    DailyMarketSnapshot,
    ExecutionIntent,
    ForecastCombiner,
    FrontNextPair,
    InstrumentConfig,
    KernelInputSlice,
    PortfolioPolicy,
    TargetPosition,
    TrendSleeveConfig,
)
from .research import (
    SUPPORTED_PHASE4_LIVE_SYMBOLS,
    ResearchCostModel,
    ResearchDayResult,
    WalkForwardReport,
    WalkForwardWindow,
    is_phase4_supported,
    simulate_research_path,
    summarize_walk_forward,
)
from .snapshot import (
    annualized_realized_vol,
    build_kernel_input_slice,
    load_carry_input_slices,
    load_daily_market_snapshots,
)

__all__ = [
    "DEFAULT_COMBINER",
    "DEFAULT_PORTFOLIO_POLICY",
    "DEFAULT_TREND_SLEEVE",
    "build_carry_input_slice",
    "build_front_next_pair",
    "discover_statistics_files",
    "load_contract_observations",
    "pair_observations_by_day",
    "compute_expiry_date",
    "supported_expiry_rule",
    "annualized_carry_from_curve",
    "apply_inertia_band",
    "bounded_weighted_forecast",
    "ewmac_forecast",
    "target_contracts_from_vol",
    "ResearchCostModel",
    "ResearchDayResult",
    "SUPPORTED_PHASE4_LIVE_SYMBOLS",
    "WalkForwardReport",
    "WalkForwardWindow",
    "is_phase4_supported",
    "simulate_research_path",
    "summarize_walk_forward",
    "annualized_realized_vol",
    "build_kernel_input_slice",
    "load_carry_input_slices",
    "load_daily_market_snapshots",
    "CombinedForecast",
    "ContractObservation",
    "CarryInputSlice",
    "DailyMarketSnapshot",
    "ExecutionIntent",
    "ForecastCombiner",
    "FrontNextPair",
    "InstrumentConfig",
    "KernelInputSlice",
    "PortfolioPolicy",
    "TargetPosition",
    "TrendSleeveConfig",
]
