"""Phase 1 daily snapshot assembly for the medium-frequency futures kernel."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH

from .carry import build_carry_input_slice
from .contracts import load_contract_observations, pair_observations_by_day
from .models import CarryInputSlice, DailyMarketSnapshot, InstrumentConfig, KernelInputSlice


def load_daily_market_snapshots(
    instrument: InstrumentConfig,
    *,
    start: date | None = None,
    end: date | None = None,
    db_path: Path | None = None,
    orb_minutes: int = 5,
) -> list[DailyMarketSnapshot]:
    """Load daily research snapshots from canonical repo surfaces.

    Phase 1 uses:
    - `daily_features` as the research-close surface
    - `exchange_statistics` as the front-contract stats surface when available

    Carry remains explicitly unavailable until the repo has a next-contract
    surface with same-day observed prices.
    """
    if orb_minutes <= 0:
        raise ValueError("orb_minutes must be positive")

    db_path = db_path or GOLD_DB_PATH
    research_symbol = instrument.research_price_symbol or instrument.live_symbol
    stats_symbol = instrument.stats_symbol or instrument.live_symbol

    where_parts = ["symbol = ?", "orb_minutes = ?", "daily_close IS NOT NULL"]
    params: list[object] = [research_symbol, orb_minutes]
    if start is not None:
        where_parts.append("trading_day >= ?")
        params.append(start)
    if end is not None:
        where_parts.append("trading_day <= ?")
        params.append(end)

    with duckdb.connect(str(db_path), read_only=True) as con:
        research_rows = con.execute(
            f"""
            SELECT trading_day, daily_close
            FROM daily_features
            WHERE {' AND '.join(where_parts)}
            ORDER BY trading_day
            """,
            params,
        ).fetchall()

        stats_rows = con.execute(
            """
            SELECT cal_date, settlement, front_contract, open_interest, cleared_volume
            FROM exchange_statistics
            WHERE symbol = ?
              AND (? IS NULL OR cal_date >= ?)
              AND (? IS NULL OR cal_date <= ?)
            ORDER BY cal_date
            """,
            [stats_symbol, start, start, end, end],
        ).fetchall()

    if not research_rows:
        raise ValueError(
            f"no daily_features coverage for {instrument.research_symbol} "
            f"(research source {research_symbol})"
        )

    stats_by_day = {
        row[0]: {
            "settlement": float(row[1]) if row[1] is not None else None,
            "front_contract": row[2],
            "open_interest": int(row[3]) if row[3] is not None else None,
            "cleared_volume": int(row[4]) if row[4] is not None else None,
        }
        for row in stats_rows
    }

    snapshots: list[DailyMarketSnapshot] = []
    for trading_day, daily_close in research_rows:
        if daily_close is None:
            continue

        stats_row = stats_by_day.get(trading_day)
        front_settlement = stats_row["settlement"] if stats_row else None
        coverage_note = None
        if stats_row is None:
            coverage_note = "missing_exchange_statistics"
        elif front_settlement is None:
            coverage_note = "missing_front_settlement"
        else:
            coverage_note = "front_only_no_next_contract_surface"

        contract_price = float(front_settlement) if front_settlement is not None else float(daily_close)
        snapshots.append(
            DailyMarketSnapshot(
                trading_day=trading_day,
                symbol=instrument.research_symbol,
                research_symbol=research_symbol,
                stats_symbol=stats_symbol,
                research_close=float(daily_close),
                contract_price=contract_price,
                front_contract=stats_row["front_contract"] if stats_row else None,
                front_settlement=front_settlement,
                open_interest=stats_row["open_interest"] if stats_row else None,
                cleared_volume=stats_row["cleared_volume"] if stats_row else None,
                carry_available=False,
                coverage_note=coverage_note,
            )
        )

    return snapshots


def annualized_realized_vol(prices: Sequence[float], *, lookback: int = 63) -> float:
    """Compute backward-looking annualized realized volatility from closes."""
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1")
    if len(prices) < 2:
        raise ValueError("need at least two prices")

    trimmed = [float(price) for price in prices[-(lookback + 1) :]]
    if any(price <= 0 for price in trimmed):
        raise ValueError("prices must be positive")

    returns = [math.log(curr / prev) for prev, curr in zip(trimmed, trimmed[1:], strict=False)]
    if len(returns) < 2:
        raise ValueError("need at least two returns")

    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(252.0)


def load_carry_input_slices(
    instrument: InstrumentConfig,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[CarryInputSlice]:
    """Load honest carry inputs from canonical raw statistics, if available."""
    stats_symbol = instrument.stats_symbol or instrument.live_symbol
    observations = load_contract_observations(stats_symbol, start=start, end=end)
    if not observations:
        sentinel_day = end or start or date.min
        return [
            CarryInputSlice(
                trading_day=sentinel_day,
                symbol=stats_symbol,
                front_contract=None,
                next_contract=None,
                front_price=None,
                next_price=None,
                price_spread=None,
                price_ratio=None,
                selection_metric=None,
                contract_gap_months=None,
                calendar_gap_days=None,
                annualized_carry=None,
                carry_available=False,
                unavailable_reason="missing_raw_statistics",
            )
        ]

    return [
        build_carry_input_slice(pair)
        for pair in pair_observations_by_day(stats_symbol, observations)
    ]


def build_kernel_input_slice(
    snapshots: Sequence[DailyMarketSnapshot],
    instrument: InstrumentConfig,
    *,
    as_of: date,
    carry_inputs: Sequence[CarryInputSlice] | None = None,
    vol_lookback: int = 63,
    min_history: int = 20,
) -> KernelInputSlice:
    """Adapt assembled snapshots into the existing kernel input shape."""
    usable = [snapshot for snapshot in snapshots if snapshot.trading_day <= as_of]
    if len(usable) < min_history:
        raise ValueError(
            f"insufficient history for {instrument.research_symbol}: "
            f"{len(usable)} rows < min_history {min_history}"
        )

    prices = tuple(snapshot.research_close for snapshot in usable)
    annualized_vol = annualized_realized_vol(prices, lookback=vol_lookback)
    latest = usable[-1]
    carry_by_day = {carry_input.trading_day: carry_input for carry_input in carry_inputs or ()}
    carry_input = carry_by_day.get(latest.trading_day)

    contract_notional = None
    if instrument.contract_multiplier is not None:
        contract_notional = latest.contract_price * instrument.contract_multiplier

    return KernelInputSlice(
        trading_day=latest.trading_day,
        symbol=instrument.research_symbol,
        price_history=prices,
        annualized_vol=annualized_vol,
        contract_price=latest.contract_price,
        contract_notional_usd=contract_notional,
        front_contract=latest.front_contract,
        front_price=latest.front_settlement,
        next_price=carry_input.next_price if carry_input is not None else None,
        days_between_expiries=carry_input.calendar_gap_days if carry_input is not None else None,
        carry_available=carry_input.carry_available if carry_input is not None else latest.carry_available,
        carry_reason_code=carry_input.unavailable_reason if carry_input is not None else latest.coverage_note,
    )
