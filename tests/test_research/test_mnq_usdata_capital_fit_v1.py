from __future__ import annotations

import numpy as np
import pandas as pd

from research.mnq_usdata_capital_fit_v1 import (
    MAX_CONTRACTS_PER_LEG,
    _book_daily,
    _max_drawdown,
    _simulate_survival,
)


def test_max_drawdown_uses_cumulative_equity_curve() -> None:
    values = np.array([10.0, -3.0, -9.0, 5.0])

    assert _max_drawdown(values) == 12.0


def test_book_daily_fills_non_eligible_days_with_zero() -> None:
    calendar = pd.Series(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    leg = pd.DataFrame(
        {
            "trading_day": [pd.Timestamp("2025-01-02").date()],
            "pnl_r": [1.0],
            "pnl_dollars_1ct": [25.0],
            "risk_dollars": [25.0],
        }
    )

    book = _book_daily(calendar, [leg])

    assert book["pnl_dollars_1ct"].tolist() == [25.0, 0.0]
    assert book["active_trades"].tolist() == [1, 0]


def test_survival_is_deterministic_for_same_contract_size() -> None:
    values = np.array([20.0, -10.0, 30.0, -15.0])

    first = _simulate_survival(
        values,
        contracts_per_leg=1,
        dd_limit=2_000.0,
        daily_loss_limit=450.0,
        freeze_at_balance=2_100.0,
    )
    second = _simulate_survival(
        values,
        contracts_per_leg=1,
        dd_limit=2_000.0,
        daily_loss_limit=450.0,
        freeze_at_balance=2_100.0,
    )

    assert first == second
    assert first["operational_survival"] == 1.0


def test_contract_curve_has_bounded_declared_size() -> None:
    assert MAX_CONTRACTS_PER_LEG == 10
