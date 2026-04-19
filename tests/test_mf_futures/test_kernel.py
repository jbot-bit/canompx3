from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from mf_futures import expiry as expiry_module
from mf_futures.carry import build_carry_input_slice
from mf_futures.contracts import build_front_next_pair
from mf_futures.expiry import compute_expiry_date
from mf_futures.kernel import (
    annualized_carry_from_curve,
    apply_inertia_band,
    bounded_weighted_forecast,
    ewmac_forecast,
    target_contracts_from_vol,
)
from mf_futures.models import (
    CombinedForecast,
    ContractObservation,
    DailyMarketSnapshot,
    ExecutionIntent,
    InstrumentConfig,
    TargetPosition,
)
from mf_futures.research import ResearchCostModel, is_phase4_supported, simulate_research_path, summarize_walk_forward
from mf_futures.snapshot import (
    annualized_realized_vol,
    build_kernel_input_slice,
    load_carry_input_slices,
    load_daily_market_snapshots,
)


def test_ewmac_forecast_positive_for_rising_series() -> None:
    prices = [100 + i for i in range(300)]
    forecast = ewmac_forecast(prices, fast_span=16, slow_span=64, annualized_vol=0.20)
    assert forecast > 0


def test_annualized_carry_positive_when_front_above_next() -> None:
    carry = annualized_carry_from_curve(front_price=105.0, next_price=100.0, days_between_expiries=90)
    assert carry > 0


def test_bounded_weighted_forecast_scales_and_caps() -> None:
    forecast = bounded_weighted_forecast([(2.0, 1.0), (1.0, 1.0)], target_abs_forecast=10.0, forecast_cap=12.0)
    assert forecast == 12.0


def test_target_contracts_from_vol_respects_forecast_sign() -> None:
    contracts = target_contracts_from_vol(
        capital_usd=100_000.0,
        annualized_vol_target=0.10,
        contract_notional_usd=50_000.0,
        annualized_instrument_vol=0.20,
        combined_forecast=10.0,
    )
    assert contracts > 0

    short_contracts = target_contracts_from_vol(
        capital_usd=100_000.0,
        annualized_vol_target=0.10,
        contract_notional_usd=50_000.0,
        annualized_instrument_vol=0.20,
        combined_forecast=-10.0,
    )
    assert short_contracts < 0


def test_apply_inertia_band_keeps_small_change() -> None:
    assert apply_inertia_band(current_contracts=10, target_contracts=11, inertia_band_pct=0.10) == 10
    assert apply_inertia_band(current_contracts=10, target_contracts=13, inertia_band_pct=0.10) == 13


def test_models_support_research_publication_shapes() -> None:
    combined = CombinedForecast(
        trading_day=date(2026, 4, 18),
        symbol="ES",
        trend_forecast=8.0,
        carry_forecast=2.0,
        combined_forecast=10.0,
    )
    target = TargetPosition(
        trading_day=date(2026, 4, 18),
        symbol="ES",
        target_contracts=2,
        current_contracts=1,
        effective_contracts=2,
        contract_notional_usd=250_000.0,
        annualized_vol=0.18,
        combined_forecast=combined.combined_forecast,
    )
    intent = ExecutionIntent(
        trading_day=date(2026, 4, 18),
        symbol="ES",
        current_contracts=target.current_contracts,
        target_contracts=target.effective_contracts,
        delta_contracts=target.effective_contracts - target.current_contracts,
        action="INCREASE",
    )

    assert combined.combined_forecast == 10.0
    assert intent.delta_contracts == 1


def _build_snapshot_test_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "mf_phase1.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE daily_features (
                trading_day DATE,
                symbol VARCHAR,
                orb_minutes INTEGER,
                daily_close DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE exchange_statistics (
                cal_date DATE,
                symbol VARCHAR,
                settlement DOUBLE,
                front_contract VARCHAR,
                open_interest BIGINT,
                cleared_volume BIGINT
            )
            """
        )
        con.executemany(
            "INSERT INTO daily_features VALUES (?, ?, ?, ?)",
            [
                (date(2026, 4, 1), "GC", 5, 3000.0),
                (date(2026, 4, 2), "GC", 5, 3030.0),
                (date(2026, 4, 3), "GC", 5, 3060.0),
                (date(2026, 4, 4), "GC", 5, 3090.0),
                (date(2026, 4, 5), "GC", 5, 3120.0),
                (date(2026, 4, 6), "GC", 5, 3150.0),
            ],
        )
        con.executemany(
            "INSERT INTO exchange_statistics VALUES (?, ?, ?, ?, ?, ?)",
            [
                (date(2026, 4, 3), "MGC", 3055.0, "MGCM6", 40000, 500000),
                (date(2026, 4, 4), "MGC", 3085.0, "MGCM6", 41000, 510000),
                (date(2026, 4, 5), "MGC", 3115.0, "MGCQ6", 42000, 520000),
            ],
        )
    finally:
        con.close()
    return db_path


def test_load_daily_market_snapshots_uses_research_and_stats_symbols(tmp_path: Path) -> None:
    db_path = _build_snapshot_test_db(tmp_path)
    instrument = InstrumentConfig(
        "GC",
        "MGC",
        "metals",
        "metals",
        research_price_symbol="GC",
        stats_symbol="MGC",
        contract_multiplier=10.0,
    )

    snapshots = load_daily_market_snapshots(instrument, db_path=db_path)

    assert [snapshot.trading_day for snapshot in snapshots] == [
        date(2026, 4, 1),
        date(2026, 4, 2),
        date(2026, 4, 3),
        date(2026, 4, 4),
        date(2026, 4, 5),
        date(2026, 4, 6),
    ]
    assert snapshots[0].coverage_note == "missing_exchange_statistics"
    assert snapshots[2].front_contract == "MGCM6"
    assert snapshots[2].contract_price == 3055.0
    assert snapshots[2].carry_available is False
    assert snapshots[4].front_contract == "MGCQ6"


def test_build_kernel_input_slice_is_backward_looking_and_sizes_notional(tmp_path: Path) -> None:
    db_path = _build_snapshot_test_db(tmp_path)
    instrument = InstrumentConfig(
        "GC",
        "MGC",
        "metals",
        "metals",
        research_price_symbol="GC",
        stats_symbol="MGC",
        contract_multiplier=10.0,
    )
    snapshots = load_daily_market_snapshots(instrument, db_path=db_path)

    input_slice = build_kernel_input_slice(
        snapshots,
        instrument,
        as_of=date(2026, 4, 5),
        vol_lookback=4,
        min_history=4,
    )

    assert input_slice.trading_day == date(2026, 4, 5)
    assert input_slice.price_history == (3000.0, 3030.0, 3060.0, 3090.0, 3120.0)
    assert input_slice.contract_price == 3115.0
    assert input_slice.contract_notional_usd == 31150.0
    assert input_slice.front_contract == "MGCQ6"
    assert input_slice.carry_available is False


def test_annualized_realized_vol_requires_backward_history() -> None:
    vol = annualized_realized_vol([100.0, 101.0, 102.0, 101.5, 103.0], lookback=3)
    assert vol > 0


def test_build_front_next_pair_uses_same_day_liquidity_only() -> None:
    day_one = [
        ContractObservation(date(2026, 4, 1), "MGC", "MGCM6", 2026, 6, 3000.0, 10_000, 20_000),
        ContractObservation(date(2026, 4, 1), "MGC", "MGCQ6", 2026, 8, 3010.0, 3_000, 10_000),
        ContractObservation(date(2026, 4, 1), "MGC", "MGCV6", 2026, 10, 3020.0, 1_000, 5_000),
    ]
    day_two = [
        ContractObservation(date(2026, 4, 2), "MGC", "MGCM6", 2026, 6, 3005.0, 500, 10_000),
        ContractObservation(date(2026, 4, 2), "MGC", "MGCQ6", 2026, 8, 3015.0, 12_000, 30_000),
        ContractObservation(date(2026, 4, 2), "MGC", "MGCV6", 2026, 10, 3025.0, 2_000, 6_000),
    ]

    pair_one = build_front_next_pair("MGC", day_one)
    pair_two = build_front_next_pair("MGC", day_two)

    assert pair_one.front is not None and pair_one.front.contract_symbol == "MGCM6"
    assert pair_one.next_contract is not None and pair_one.next_contract.contract_symbol == "MGCQ6"
    assert pair_two.front is not None and pair_two.front.contract_symbol == "MGCQ6"
    assert pair_two.next_contract is not None and pair_two.next_contract.contract_symbol == "MGCV6"


def test_build_front_next_pair_does_not_skip_nearest_later_contract() -> None:
    observations = [
        ContractObservation(
            date(2026, 4, 1),
            "MNQ",
            "MNQM6",
            2026,
            6,
            23_915.0,
            100_000,
            180_000,
            expiry_date=date(2026, 6, 19),
        ),
        ContractObservation(
            date(2026, 4, 1),
            "MNQ",
            "MNQU6",
            2026,
            9,
            None,
            500,
            2_000,
            expiry_date=date(2026, 9, 18),
        ),
        ContractObservation(
            date(2026, 4, 1),
            "MNQ",
            "MNQZ6",
            2026,
            12,
            24_500.0,
            90_000,
            160_000,
            expiry_date=date(2026, 12, 18),
        ),
    ]

    pair = build_front_next_pair("MNQ", observations)
    carry = build_carry_input_slice(pair)

    assert pair.front is not None and pair.front.contract_symbol == "MNQM6"
    assert pair.next_contract is not None and pair.next_contract.contract_symbol == "MNQU6"
    assert pair.unavailable_reason == "missing_next_settlement"
    assert carry.carry_available is False
    assert carry.unavailable_reason == "missing_next_settlement"
    assert carry.next_contract == "MNQU6"


def test_build_front_next_pair_computes_expiry_gap_when_available() -> None:
    observations = [
        ContractObservation(
            date(2026, 4, 1),
            "MGC",
            "MGCM6",
            2026,
            6,
            3000.0,
            10_000,
            20_000,
            expiry_date=date(2026, 6, 25),
        ),
        ContractObservation(
            date(2026, 4, 1),
            "MGC",
            "MGCQ6",
            2026,
            8,
            3010.0,
            3_000,
            10_000,
            expiry_date=date(2026, 8, 27),
        ),
    ]

    pair = build_front_next_pair("MGC", observations)
    carry = build_carry_input_slice(pair)

    assert pair.contract_gap_months == 2
    assert pair.calendar_gap_days == 63
    assert carry.carry_available is True
    assert carry.annualized_carry is not None


def test_compute_expiry_date_supports_micro_equity_quarterlies() -> None:
    assert compute_expiry_date("MES", contract_year=2026, contract_month=6) == date(2026, 6, 19)
    assert compute_expiry_date("MNQ", contract_year=2026, contract_month=9) == date(2026, 9, 18)


def test_compute_expiry_date_supports_micro_gold() -> None:
    assert compute_expiry_date("MGC", contract_year=2026, contract_month=6) == date(2026, 6, 26)


def test_compute_expiry_date_fails_closed_when_third_friday_is_not_session(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_session_dates(*_args: object, **_kwargs: object) -> list[date]:
        return [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]

    monkeypatch.setattr(expiry_module, "_session_dates", _fake_session_dates)

    assert compute_expiry_date("MES", contract_year=2026, contract_month=6) is None


def test_compute_expiry_date_fails_closed_for_unsupported_symbol_or_cycle() -> None:
    assert compute_expiry_date("M6E", contract_year=2026, contract_month=6) is None
    assert compute_expiry_date("MES", contract_year=2026, contract_month=4) is None


def test_build_carry_input_slice_fails_closed_without_expiry_dates() -> None:
    observations = [
        ContractObservation(date(2026, 4, 1), "MGC", "MGCM6", 2026, 6, 3000.0, 10_000, 20_000),
        ContractObservation(date(2026, 4, 1), "MGC", "MGCQ6", 2026, 8, 3010.0, 3_000, 10_000),
    ]

    pair = build_front_next_pair("MGC", observations)
    carry = build_carry_input_slice(pair)

    assert pair.unavailable_reason == "missing_expiry_date"
    assert carry.carry_available is False
    assert carry.unavailable_reason == "missing_expiry_date"
    assert carry.front_price == 3000.0
    assert carry.next_price == 3010.0


def test_build_front_next_pair_fails_closed_without_liquidity_metric() -> None:
    observations = [
        ContractObservation(date(2026, 4, 1), "MGC", "MGCM6", 2026, 6, 3000.0, None, None),
        ContractObservation(date(2026, 4, 1), "MGC", "MGCQ6", 2026, 8, 3010.0, None, None),
    ]

    pair = build_front_next_pair("MGC", observations)

    assert pair.carry_available is False
    assert pair.unavailable_reason == "missing_liquidity_rank_input"


def test_load_carry_input_slices_unlocks_annualized_carry_when_expiry_is_supported() -> None:
    instrument = InstrumentConfig(
        "GC",
        "MGC",
        "metals",
        "metals",
        research_price_symbol="GC",
        stats_symbol="MGC",
        contract_multiplier=10.0,
    )

    slices = load_carry_input_slices(
        instrument,
        start=date(2026, 3, 31),
        end=date(2026, 3, 31),
    )

    assert len(slices) == 1
    assert slices[0].carry_available is True
    assert slices[0].annualized_carry is not None
    assert slices[0].unavailable_reason is None


def test_build_front_next_pair_unlocks_mnq_annualized_carry_with_attached_expiries() -> None:
    observations = [
        ContractObservation(
            date(2026, 3, 31),
            "MNQ",
            "MNQM6",
            2026,
            6,
            23_915.0,
            100_000,
            180_000,
            expiry_date=compute_expiry_date("MNQ", contract_year=2026, contract_month=6),
        ),
        ContractObservation(
            date(2026, 3, 31),
            "MNQ",
            "MNQU6",
            2026,
            9,
            24_125.0,
            80_000,
            150_000,
            expiry_date=compute_expiry_date("MNQ", contract_year=2026, contract_month=9),
        ),
    ]

    pair = build_front_next_pair("MNQ", observations)
    carry = build_carry_input_slice(pair)

    assert pair.front is not None and pair.front.contract_symbol == "MNQM6"
    assert pair.next_contract is not None and pair.next_contract.contract_symbol == "MNQU6"
    assert pair.calendar_gap_days == 91
    assert carry.carry_available is True
    assert carry.annualized_carry is not None
    assert carry.unavailable_reason is None


def test_load_carry_input_slices_returns_unavailable_when_raw_stats_missing() -> None:
    instrument = InstrumentConfig(
        "6E",
        "M6E",
        "fx",
        "fx",
        research_price_symbol="M6E",
        stats_symbol="M6E",
        contract_multiplier=12_500.0,
    )

    slices = load_carry_input_slices(
        instrument,
        start=date(2026, 2, 20),
        end=date(2026, 2, 20),
    )

    assert len(slices) == 1
    assert slices[0].carry_available is False
    assert slices[0].unavailable_reason == "missing_raw_statistics"


def test_is_phase4_supported_is_narrow() -> None:
    assert is_phase4_supported(InstrumentConfig("ES", "MES", "equity_index", "equities")) is True
    assert is_phase4_supported(InstrumentConfig("6E", "M6E", "fx", "fx")) is False


def test_simulate_research_path_accounts_for_turnover_costs() -> None:
    instrument = InstrumentConfig(
        "GC",
        "MGC",
        "metals",
        "metals",
        research_price_symbol="GC",
        stats_symbol="MGC",
        contract_multiplier=10.0,
    )
    snapshots = [
        DailyMarketSnapshot(date(2026, 1, 1), "GC", "GC", "MGC", 100.0, 100.0, "MGCG6", 100.0, 1000, 5000, True),
        DailyMarketSnapshot(date(2026, 1, 2), "GC", "GC", "MGC", 101.0, 101.0, "MGCG6", 101.0, 1000, 5000, True),
        DailyMarketSnapshot(date(2026, 1, 3), "GC", "GC", "MGC", 102.0, 102.0, "MGCG6", 102.0, 1000, 5000, True),
        DailyMarketSnapshot(date(2026, 1, 4), "GC", "GC", "MGC", 103.0, 103.0, "MGCG6", 103.0, 1000, 5000, True),
        DailyMarketSnapshot(date(2026, 1, 5), "GC", "GC", "MGC", 104.0, 104.0, "MGCG6", 104.0, 1000, 5000, True),
        DailyMarketSnapshot(date(2026, 1, 6), "GC", "GC", "MGC", 105.0, 105.0, "MGCG6", 105.0, 1000, 5000, True),
    ]
    carry_inputs = [
        build_carry_input_slice(
            build_front_next_pair(
                "MGC",
                [
                    ContractObservation(
                        date(2026, 1, day),
                        "MGC",
                        "MGCG6",
                        2026,
                        2,
                        100.0 + day - 1,
                        5000,
                        1000,
                        expiry_date=date(2026, 2, 24),
                    ),
                    ContractObservation(
                        date(2026, 1, day),
                        "MGC",
                        "MGCJ6",
                        2026,
                        4,
                        99.0 + day - 1,
                        3000,
                        900,
                        expiry_date=date(2026, 4, 28),
                    ),
                ],
            )
        )
        for day in range(1, 7)
    ]

    rows = simulate_research_path(
        instrument,
        snapshots,
        carry_inputs=carry_inputs,
        capital_usd=100_000.0,
        cost_model=ResearchCostModel(round_turn_cost_usd=5.0),
        vol_lookback=3,
        min_history=4,
    )

    assert len(rows) == 2
    assert rows[0].execution_intent.action == "INCREASE"
    assert rows[0].turnover_contracts >= 1
    assert rows[0].cost_usd == rows[0].turnover_contracts * 5.0
    assert rows[0].net_pnl_usd == rows[0].gross_pnl_usd - rows[0].cost_usd
    assert rows[0].return_basis == "research_close_to_close"


def test_summarize_walk_forward_builds_non_overlapping_windows() -> None:
    rows = [
        type(
            "Row",
            (),
            {
                "trading_day": date(2026, 1, idx + 1),
                "symbol": "GC",
                "gross_pnl_usd": float(idx + 1),
                "net_pnl_usd": float(idx if idx % 2 == 0 else -(idx + 1)),
                "turnover_contracts": 1,
            },
        )()
        for idx in range(8)
    ]

    report = summarize_walk_forward(rows, min_train_days=4, test_window_days=2)

    assert report.symbol == "GC"
    assert report.n_total_windows == 2
    assert len(report.windows) == 2
    assert report.windows[0].train_start == date(2026, 1, 1)
    assert report.windows[0].train_end == date(2026, 1, 4)
    assert report.windows[0].test_start == date(2026, 1, 5)
    assert report.windows[0].test_end == date(2026, 1, 6)
