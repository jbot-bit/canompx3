import math

import pandas as pd

from research.mnq_open_book_validation_v1 import (
    _add_book_statistics,
    _leg_correlation,
    _objective_score,
    filter_diagnostics,
    select_candidate_pool,
)


def _cell(
    strategy: str,
    session: str,
    orb_minutes: int,
    rr: float,
    filter_name: str,
    annual_r: float,
    dd: float,
    *,
    t: float = 4.0,
    q: float = 0.001,
    wfe: float = 0.8,
    era_ok: bool = True,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "session": session,
        "orb_minutes": orb_minutes,
        "rr": rr,
        "filter": filter_name,
        "n_is": 250,
        "annual_r": annual_r,
        "mean_is": annual_r / 250.0,
        "dd": dd,
        "t": t,
        "q_global": q,
        "wfe": wfe,
        "era_ok": era_ok,
        "mean_2026": -99.0,
    }


def test_objective_score_prefers_drawdown_adjusted_return() -> None:
    high_return_high_dd = {"annual_r": 80.0, "dd": 40.0}
    lower_return_low_dd = {"annual_r": 60.0, "dd": 15.0}

    assert _objective_score(lower_return_low_dd) > _objective_score(high_return_high_dd)


def test_candidate_pool_is_pre2026_metric_capped_not_2026_selected() -> None:
    rows = []
    for idx in range(6):
        rows.append(_cell(f"A{idx}", "NYSE_OPEN", 15, 2.0 + idx, "NO_FILTER", 50.0 + idx, 10.0))
    rows.extend(
        [
            _cell("B0", "US_DATA_1000", 15, 2.0, "NO_FILTER", 48.0, 9.0),
            _cell("B1", "US_DATA_1000", 30, 2.0, "COST_LT10", 46.0, 8.0),
            _cell("C0", "CME_PRECLOSE", 5, 1.5, "NO_FILTER", 45.0, 7.0),
            _cell("D_FAIL_Q", "TOKYO_OPEN", 5, 1.5, "NO_FILTER", 200.0, 5.0, q=0.20),
        ]
    )
    cells = pd.DataFrame(rows)

    pool = select_candidate_pool(cells, pool_size=5)

    assert "D_FAIL_Q" not in set(pool["strategy"])
    assert (pool["session"] == "NYSE_OPEN").sum() <= 2
    assert pool["mean_2026"].eq(-99.0).all()


def test_leg_correlation_aligns_days_and_fills_missing_zero() -> None:
    leg_a = pd.DataFrame({"trading_day": pd.to_datetime(["2025-01-02", "2025-01-03"]).date, "pnl_r": [1.0, -1.0]})
    leg_b = pd.DataFrame({"trading_day": pd.to_datetime(["2025-01-03", "2025-01-06"]).date, "pnl_r": [-1.0, 1.0]})

    corr = _leg_correlation([leg_a, leg_b])

    assert math.isfinite(corr)
    assert corr > 0.0


def test_book_statistics_emit_family_and_inherited_dsr_fields() -> None:
    books = pd.DataFrame(
        [
            {
                "strategy": "A",
                "family": "current_two_book",
                "n_is": 120,
                "p": 0.001,
                "t": 3.5,
                "sharpe": 0.35,
                "skewness": 0.0,
                "kurtosis_excess": 0.0,
                "wfe": 0.8,
                "era_ok": True,
            },
            {
                "strategy": "B",
                "family": "current_two_book",
                "n_is": 120,
                "p": 0.02,
                "t": 2.2,
                "sharpe": 0.10,
                "skewness": 0.0,
                "kurtosis_excess": 0.0,
                "wfe": 0.8,
                "era_ok": True,
            },
        ]
    )

    scored = _add_book_statistics(books)

    assert {"q_family", "sr0_family", "dsr_family", "sr0_inherited", "dsr_inherited", "verdict"} <= set(scored.columns)
    assert scored.loc[0, "q_family"] <= scored.loc[1, "q_family"]


def test_filter_diagnostics_compares_filter_to_same_no_filter_parent() -> None:
    cells = pd.DataFrame(
        [
            _cell("BASE", "NYSE_OPEN", 15, 2.0, "NO_FILTER", 40.0, 20.0),
            _cell("FILTERED", "NYSE_OPEN", 15, 2.0, "COST_LT10", 45.0, 18.0),
            _cell("OTHER_BASE", "US_DATA_1000", 15, 2.0, "NO_FILTER", 30.0, 15.0),
            _cell("OTHER_FILTERED", "US_DATA_1000", 15, 2.0, "COST_LT10", 28.0, 14.0),
        ]
    )

    diag = filter_diagnostics(cells)
    row = diag[diag["filter"] == "COST_LT10"].iloc[0]

    assert row["comparisons"] == 2
    assert row["best_delta_annual_r"] == 5.0
    assert row["helped_risk_adjusted_count"] == 1
