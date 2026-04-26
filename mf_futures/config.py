"""Static defaults for the medium-frequency futures kernel."""

from __future__ import annotations

from .models import ForecastCombiner, InstrumentConfig, PortfolioPolicy, TrendSleeveConfig

RESEARCH_UNIVERSE: tuple[InstrumentConfig, ...] = (
    InstrumentConfig(
        "ES",
        "MES",
        "equity_index",
        "equities",
        research_price_symbol="MES",
        stats_symbol="MES",
        contract_multiplier=5.0,
    ),
    InstrumentConfig(
        "NQ",
        "MNQ",
        "equity_index",
        "equities",
        research_price_symbol="MNQ",
        stats_symbol="MNQ",
        contract_multiplier=2.0,
    ),
    InstrumentConfig("ZN", "ZN", "rates", "rates"),
    InstrumentConfig("ZB", "ZB", "rates", "rates"),
    InstrumentConfig(
        "GC",
        "MGC",
        "metals",
        "metals",
        research_price_symbol="GC",
        stats_symbol="MGC",
        contract_multiplier=10.0,
    ),
    InstrumentConfig(
        "6E",
        "M6E",
        "fx",
        "fx",
        research_price_symbol="M6E",
        stats_symbol="M6E",
        contract_multiplier=12_500.0,
    ),
    InstrumentConfig("6J", "6J", "fx", "fx"),
)


DEFAULT_TREND_SLEEVE: tuple[TrendSleeveConfig, ...] = (
    TrendSleeveConfig(16, 64, 1.0),
    TrendSleeveConfig(32, 128, 1.0),
    TrendSleeveConfig(64, 256, 1.0),
)


DEFAULT_COMBINER = ForecastCombiner(
    target_abs_forecast=10.0,
    forecast_cap=20.0,
)


DEFAULT_PORTFOLIO_POLICY = PortfolioPolicy(
    annualized_vol_target=0.08,
    diversification_multiplier_cap=2.5,
    inertia_band_pct=0.10,
)
