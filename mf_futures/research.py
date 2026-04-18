"""Research-only walk-forward harness for supported `mf_futures` symbols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import DEFAULT_COMBINER, DEFAULT_PORTFOLIO_POLICY, DEFAULT_TREND_SLEEVE
from .kernel import apply_inertia_band, ewmac_forecast, target_contracts_from_vol
from .models import CarryInputSlice, DailyMarketSnapshot, ExecutionIntent, InstrumentConfig, TargetPosition
from .snapshot import build_kernel_input_slice

SUPPORTED_PHASE4_LIVE_SYMBOLS = frozenset({"MES", "MNQ", "MGC"})


@dataclass(frozen=True)
class ResearchCostModel:
    """Simple fixed-cost model for research turnover accounting."""

    round_turn_cost_usd: float


@dataclass(frozen=True)
class ResearchDayResult:
    """One daily research-only simulation row."""

    trading_day: object
    symbol: str
    carry_available: bool
    trend_forecast: float
    carry_forecast: float
    combined_forecast: float
    target_position: TargetPosition
    execution_intent: ExecutionIntent
    turnover_contracts: int
    gross_pnl_usd: float
    cost_usd: float
    net_pnl_usd: float
    return_basis: Literal["research_close_to_close"] = "research_close_to_close"


@dataclass(frozen=True)
class WalkForwardWindow:
    """Anchored non-overlapping test window summary."""

    train_start: object
    train_end: object
    test_start: object
    test_end: object
    n_test_days: int
    gross_pnl_usd: float
    net_pnl_usd: float
    total_turnover_contracts: int
    average_turnover_contracts: float
    positive_net: bool


@dataclass(frozen=True)
class WalkForwardReport:
    """Aggregated walk-forward summary for one supported instrument."""

    symbol: str
    n_total_windows: int
    n_positive_windows: int
    pct_positive_windows: float
    agg_gross_pnl_usd: float
    agg_net_pnl_usd: float
    total_turnover_contracts: int
    windows: tuple[WalkForwardWindow, ...]


def is_phase4_supported(instrument: InstrumentConfig) -> bool:
    """True only for the narrow supported Phase 4 surface."""
    return instrument.live_symbol in SUPPORTED_PHASE4_LIVE_SYMBOLS


def simulate_research_path(
    instrument: InstrumentConfig,
    snapshots: list[DailyMarketSnapshot],
    *,
    carry_inputs: list[CarryInputSlice] | None = None,
    capital_usd: float,
    cost_model: ResearchCostModel,
    current_contracts: int = 0,
    instrument_weight: float = 1.0,
    diversification_multiplier: float = 1.0,
    carry_weight: float = 1.0,
    trend_sleeves: tuple = DEFAULT_TREND_SLEEVE,
    portfolio_policy=DEFAULT_PORTFOLIO_POLICY,
    combiner=DEFAULT_COMBINER,
    vol_lookback: int = 63,
    min_history: int = 64,
) -> list[ResearchDayResult]:
    """Simulate a daily research path on the supported medium-frequency surface."""
    if not is_phase4_supported(instrument):
        raise ValueError(f"{instrument.live_symbol} is outside the supported Phase 4 surface")
    if instrument.contract_multiplier is None or instrument.contract_multiplier <= 0:
        raise ValueError("instrument.contract_multiplier must be positive")
    if capital_usd <= 0:
        raise ValueError("capital_usd must be positive")
    if cost_model.round_turn_cost_usd < 0:
        raise ValueError("round_turn_cost_usd must be non-negative")
    if carry_weight < 0:
        raise ValueError("carry_weight must be non-negative")
    if len(snapshots) <= min_history:
        raise ValueError("need more snapshots than min_history to simulate forward returns")

    ordered = sorted(snapshots, key=lambda snapshot: snapshot.trading_day)
    results: list[ResearchDayResult] = []
    live_contracts = current_contracts
    carry_by_day = {carry_input.trading_day: carry_input for carry_input in carry_inputs or []}

    for idx in range(min_history - 1, len(ordered) - 1):
        current = ordered[idx]
        next_snapshot = ordered[idx + 1]
        kernel_slice = build_kernel_input_slice(
            ordered,
            instrument,
            as_of=current.trading_day,
            carry_inputs=carry_inputs,
            vol_lookback=vol_lookback,
            min_history=min_history,
        )
        trend_forecast = _trend_forecast_from_slice(kernel_slice, trend_sleeves=trend_sleeves, forecast_cap=combiner.forecast_cap)
        carry_input = carry_by_day.get(current.trading_day)
        carry_forecast = carry_input.annualized_carry if carry_input is not None and carry_input.carry_available and carry_input.annualized_carry is not None else 0.0
        combined_forecast = _combine_forecasts(
            trend_forecast=trend_forecast,
            carry_forecast=carry_forecast,
            carry_weight=carry_weight,
            forecast_cap=combiner.forecast_cap,
        )

        target_contracts = target_contracts_from_vol(
            capital_usd=capital_usd,
            annualized_vol_target=portfolio_policy.annualized_vol_target,
            contract_notional_usd=kernel_slice.contract_notional_usd or 0.0,
            annualized_instrument_vol=kernel_slice.annualized_vol,
            combined_forecast=combined_forecast,
            instrument_weight=instrument_weight,
            diversification_multiplier=min(diversification_multiplier, portfolio_policy.diversification_multiplier_cap),
        )
        effective_contracts = apply_inertia_band(
            current_contracts=live_contracts,
            target_contracts=target_contracts,
            inertia_band_pct=portfolio_policy.inertia_band_pct,
        )
        target_position = TargetPosition(
            trading_day=current.trading_day,
            symbol=instrument.research_symbol,
            target_contracts=target_contracts,
            current_contracts=live_contracts,
            effective_contracts=effective_contracts,
            contract_notional_usd=kernel_slice.contract_notional_usd or 0.0,
            annualized_vol=kernel_slice.annualized_vol,
            combined_forecast=combined_forecast,
        )
        execution_intent = ExecutionIntent(
            trading_day=current.trading_day,
            symbol=instrument.research_symbol,
            current_contracts=live_contracts,
            target_contracts=effective_contracts,
            delta_contracts=effective_contracts - live_contracts,
            action=_intent_action(live_contracts, effective_contracts),
        )

        turnover_contracts = abs(effective_contracts - live_contracts)
        gross_pnl_usd = effective_contracts * instrument.contract_multiplier * (next_snapshot.research_close - current.research_close)
        cost_usd = turnover_contracts * cost_model.round_turn_cost_usd
        net_pnl_usd = gross_pnl_usd - cost_usd

        results.append(
            ResearchDayResult(
                trading_day=current.trading_day,
                symbol=instrument.research_symbol,
                carry_available=bool(carry_input.carry_available) if carry_input is not None else False,
                trend_forecast=trend_forecast,
                carry_forecast=carry_forecast,
                combined_forecast=combined_forecast,
                target_position=target_position,
                execution_intent=execution_intent,
                turnover_contracts=turnover_contracts,
                gross_pnl_usd=gross_pnl_usd,
                cost_usd=cost_usd,
                net_pnl_usd=net_pnl_usd,
            )
        )
        live_contracts = effective_contracts

    return results


def summarize_walk_forward(
    rows: list[ResearchDayResult],
    *,
    min_train_days: int = 63,
    test_window_days: int = 21,
) -> WalkForwardReport:
    """Summarize anchored non-overlapping walk-forward windows over research rows."""
    if min_train_days <= 0 or test_window_days <= 0:
        raise ValueError("min_train_days and test_window_days must be positive")
    if not rows:
        raise ValueError("rows must be non-empty")

    ordered = sorted(rows, key=lambda row: row.trading_day)
    if len(ordered) <= min_train_days:
        raise ValueError("not enough rows for one test window")

    windows: list[WalkForwardWindow] = []
    test_start_idx = min_train_days
    while test_start_idx + test_window_days <= len(ordered):
        test_slice = ordered[test_start_idx : test_start_idx + test_window_days]
        train_slice = ordered[:test_start_idx]
        gross_pnl_usd = sum(row.gross_pnl_usd for row in test_slice)
        net_pnl_usd = sum(row.net_pnl_usd for row in test_slice)
        total_turnover = sum(row.turnover_contracts for row in test_slice)
        windows.append(
            WalkForwardWindow(
                train_start=train_slice[0].trading_day,
                train_end=train_slice[-1].trading_day,
                test_start=test_slice[0].trading_day,
                test_end=test_slice[-1].trading_day,
                n_test_days=len(test_slice),
                gross_pnl_usd=gross_pnl_usd,
                net_pnl_usd=net_pnl_usd,
                total_turnover_contracts=total_turnover,
                average_turnover_contracts=total_turnover / len(test_slice),
                positive_net=net_pnl_usd > 0,
            )
        )
        test_start_idx += test_window_days

    if not windows:
        raise ValueError("not enough rows for a full non-overlapping test window")

    n_positive = sum(1 for window in windows if window.positive_net)
    agg_gross = sum(window.gross_pnl_usd for window in windows)
    agg_net = sum(window.net_pnl_usd for window in windows)
    total_turnover = sum(window.total_turnover_contracts for window in windows)
    return WalkForwardReport(
        symbol=ordered[0].symbol,
        n_total_windows=len(windows),
        n_positive_windows=n_positive,
        pct_positive_windows=n_positive / len(windows),
        agg_gross_pnl_usd=agg_gross,
        agg_net_pnl_usd=agg_net,
        total_turnover_contracts=total_turnover,
        windows=tuple(windows),
    )


def _trend_forecast_from_slice(kernel_slice, *, trend_sleeves: tuple, forecast_cap: float) -> float:
    weighted_total = 0.0
    total_weight = 0.0
    for sleeve in trend_sleeves:
        forecast = ewmac_forecast(
            kernel_slice.price_history,
            fast_span=sleeve.fast_span,
            slow_span=sleeve.slow_span,
            annualized_vol=kernel_slice.annualized_vol,
            forecast_cap=forecast_cap,
        )
        weighted_total += forecast * sleeve.weight
        total_weight += sleeve.weight
    if total_weight <= 0:
        raise ValueError("trend sleeve weight sum must be positive")
    return max(-forecast_cap, min(forecast_cap, weighted_total / total_weight))


def _combine_forecasts(*, trend_forecast: float, carry_forecast: float, carry_weight: float, forecast_cap: float) -> float:
    total_weight = 1.0 + carry_weight
    combined = (trend_forecast + carry_forecast * carry_weight) / total_weight
    return max(-forecast_cap, min(forecast_cap, combined))


def _intent_action(current_contracts: int, target_contracts: int) -> str:
    delta = target_contracts - current_contracts
    if delta > 0:
        return "INCREASE"
    if delta < 0:
        return "DECREASE"
    return "HOLD"
