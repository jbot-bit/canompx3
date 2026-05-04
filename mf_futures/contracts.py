"""Canonical contract observation and front/next pairing for `mf_futures`."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import databento as db
import pandas as pd

from pipeline.asset_configs import PROJECT_ROOT, get_asset_config
from pipeline.ingest_dbn_mgc import choose_front_contract, parse_expiry

from .expiry import compute_expiry_date
from .models import ContractObservation, FrontNextPair

STATISTICS_ROOT = PROJECT_ROOT / "data" / "raw" / "databento" / "statistics"
_FILE_SPAN_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})")
_REQUIRED_STAT_TYPES = (3, 6, 9)


def _parse_statistics_file_span(path: Path) -> tuple[date, date] | None:
    match = _FILE_SPAN_RE.search(path.name)
    if not match:
        return None
    return date.fromisoformat(match.group(1)), date.fromisoformat(match.group(2))


def discover_statistics_files(
    stats_symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[Path]:
    """Return raw statistics files whose filename spans overlap the request."""
    directory = STATISTICS_ROOT / stats_symbol
    if not directory.exists():
        return []

    candidates: list[Path] = []
    for path in sorted(directory.glob("*.dbn.zst")):
        span = _parse_statistics_file_span(path)
        if span is None:
            candidates.append(path)
            continue

        file_start, file_end = span
        if start is not None and file_end < start:
            continue
        if end is not None and file_start > end:
            continue
        candidates.append(path)
    return candidates


def load_contract_observations(
    stats_symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[ContractObservation]:
    """Load same-day per-contract observations from canonical raw statistics."""
    files = discover_statistics_files(stats_symbol, start=start, end=end)
    if not files:
        return []

    asset_config = get_asset_config(stats_symbol)
    outright_pattern = asset_config["outright_pattern"]
    prefix_len = int(asset_config["prefix_len"])

    frames: list[pd.DataFrame] = []
    for path in files:
        store = db.DBNStore.from_file(str(path))
        df = store.to_df().reset_index()
        if df.empty:
            continue

        df["symbol"] = df["symbol"].astype(str)
        df = df[df["symbol"].str.match(outright_pattern.pattern)]
        if df.empty:
            continue

        df = df[df["stat_type"].isin(_REQUIRED_STAT_TYPES)]
        if df.empty:
            continue

        df["cal_date"] = pd.to_datetime(df["ts_event"], utc=True).dt.date
        if start is not None:
            df = df[df["cal_date"] >= start]
        if end is not None:
            df = df[df["cal_date"] <= end]
        if df.empty:
            continue

        frames.append(df[["cal_date", "symbol", "ts_event", "ts_recv", "sequence", "stat_type", "price", "quantity"]])

    if not frames:
        return []

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["cal_date", "symbol", "stat_type", "ts_event", "ts_recv", "sequence"])
    latest = merged.groupby(["cal_date", "symbol", "stat_type"], as_index=False).last()

    observations: list[ContractObservation] = []
    for (trading_day, contract_symbol), group in latest.groupby(["cal_date", "symbol"], sort=True):
        contract_year, contract_month = parse_expiry(contract_symbol, prefix_len)
        group_by_type = {int(row.stat_type): row for row in group.itertuples(index=False)}

        settlement = group_by_type.get(3)
        cleared_volume = group_by_type.get(6)
        open_interest = group_by_type.get(9)

        observations.append(
            ContractObservation(
                trading_day=trading_day,
                symbol=stats_symbol,
                contract_symbol=contract_symbol,
                contract_year=contract_year,
                contract_month=contract_month,
                settlement=float(settlement.price) if settlement is not None and pd.notna(settlement.price) else None,
                cleared_volume=(
                    int(cleared_volume.quantity)
                    if cleared_volume is not None and pd.notna(cleared_volume.quantity)
                    else None
                ),
                open_interest=(
                    int(open_interest.quantity)
                    if open_interest is not None and pd.notna(open_interest.quantity)
                    else None
                ),
                expiry_date=compute_expiry_date(
                    stats_symbol,
                    contract_year=contract_year,
                    contract_month=contract_month,
                ),
            )
        )

    return observations


def _month_index(observation: ContractObservation) -> int:
    return observation.contract_year * 12 + observation.contract_month


def _nearest_later_contract(
    observations: list[ContractObservation],
    *,
    front_month_index: int,
) -> ContractObservation | None:
    later = [observation for observation in observations if _month_index(observation) > front_month_index]
    if not later:
        return None
    return min(later, key=lambda observation: (_month_index(observation), observation.contract_symbol))


def _select_metric_map(
    observations: list[ContractObservation],
) -> tuple[str | None, dict[str, int]]:
    volume_map = {
        observation.contract_symbol: observation.cleared_volume
        for observation in observations
        if observation.cleared_volume is not None and observation.cleared_volume > 0
    }
    if len(volume_map) >= 2:
        return "cleared_volume", volume_map

    oi_map = {
        observation.contract_symbol: observation.open_interest
        for observation in observations
        if observation.open_interest is not None and observation.open_interest > 0
    }
    if len(oi_map) >= 2:
        return "open_interest", oi_map

    return None, {}


def build_front_next_pair(
    stats_symbol: str,
    observations: list[ContractObservation],
) -> FrontNextPair:
    """Build a deterministic same-day front/next pair from contract observations."""
    if not observations:
        return FrontNextPair(
            trading_day=date.min,
            symbol=stats_symbol,
            front=None,
            next_contract=None,
            selection_metric=None,
            contract_gap_months=None,
            calendar_gap_days=None,
            carry_available=False,
            unavailable_reason="missing_contract_observations",
        )

    trading_days = {observation.trading_day for observation in observations}
    if len(trading_days) != 1:
        raise ValueError("build_front_next_pair expects observations for exactly one trading day")

    trading_day = next(iter(trading_days))
    asset_config = get_asset_config(stats_symbol)
    selection_metric, metric_map = _select_metric_map(observations)
    if not metric_map:
        return FrontNextPair(
            trading_day=trading_day,
            symbol=stats_symbol,
            front=None,
            next_contract=None,
            selection_metric=None,
            contract_gap_months=None,
            calendar_gap_days=None,
            carry_available=False,
            unavailable_reason="missing_liquidity_rank_input",
        )

    prefix_len = int(asset_config["prefix_len"])
    outright_pattern = asset_config["outright_pattern"]
    front_symbol = choose_front_contract(metric_map, outright_pattern=outright_pattern, prefix_len=prefix_len)
    observation_by_symbol = {observation.contract_symbol: observation for observation in observations}
    front = observation_by_symbol.get(front_symbol) if front_symbol is not None else None
    if front is None:
        return FrontNextPair(
            trading_day=trading_day,
            symbol=stats_symbol,
            front=None,
            next_contract=None,
            selection_metric=selection_metric,
            contract_gap_months=None,
            calendar_gap_days=None,
            carry_available=False,
            unavailable_reason="front_contract_not_found",
        )

    front_month_index = _month_index(front)
    next_contract = _nearest_later_contract(observations, front_month_index=front_month_index)
    if next_contract is None:
        return FrontNextPair(
            trading_day=trading_day,
            symbol=stats_symbol,
            front=front,
            next_contract=None,
            selection_metric=selection_metric,
            contract_gap_months=None,
            calendar_gap_days=None,
            carry_available=False,
            unavailable_reason="no_next_contract",
        )

    contract_gap_months = _month_index(next_contract) - front_month_index
    calendar_gap_days = None
    if front.expiry_date is not None and next_contract.expiry_date is not None:
        calendar_gap_days = (next_contract.expiry_date - front.expiry_date).days

    carry_available = (
        front.settlement is not None
        and next_contract.settlement is not None
        and calendar_gap_days is not None
        and calendar_gap_days > 0
    )
    unavailable_reason = None
    if front.settlement is None:
        unavailable_reason = "missing_front_settlement"
    elif next_contract.settlement is None:
        unavailable_reason = "missing_next_settlement"
    elif calendar_gap_days is None:
        unavailable_reason = "missing_expiry_date"
    elif calendar_gap_days <= 0:
        unavailable_reason = "invalid_expiry_gap"

    return FrontNextPair(
        trading_day=trading_day,
        symbol=stats_symbol,
        front=front,
        next_contract=next_contract,
        selection_metric=selection_metric,
        contract_gap_months=contract_gap_months,
        calendar_gap_days=calendar_gap_days,
        carry_available=carry_available,
        unavailable_reason=unavailable_reason,
    )


def pair_observations_by_day(
    stats_symbol: str,
    observations: list[ContractObservation],
) -> list[FrontNextPair]:
    """Pair daily observations into deterministic front/next contracts."""
    grouped: dict[date, list[ContractObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.trading_day, []).append(observation)

    return [build_front_next_pair(stats_symbol, grouped[trading_day]) for trading_day in sorted(grouped)]
