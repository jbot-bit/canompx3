import pandas as pd

from research.mnq_usdata_rr_leg_choice_v1 import (
    DECLARED_K,
    US_DATA_FILTERS,
    US_DATA_RRS,
    _book_name,
    _portfolio_series,
    _verdict,
)


def test_declared_k_matches_rr_filter_grid() -> None:
    assert len(US_DATA_RRS) * len(US_DATA_FILTERS) == DECLARED_K
    assert DECLARED_K == 15


def test_book_name_encodes_rr_and_filter() -> None:
    assert _book_name(1.0, "COST_LT12") == "NYOPEN_USDATA_RR1_COST_LT12"
    assert _book_name(1.5, "COST_LT10") == "NYOPEN_USDATA_RR1_5_COST_LT10"


def test_portfolio_series_aligns_missing_days_to_zero() -> None:
    leg_a = pd.DataFrame({"trading_day": pd.to_datetime(["2025-01-02", "2025-01-03"]).date, "pnl_r": [1.0, -1.0]})
    leg_b = pd.DataFrame({"trading_day": pd.to_datetime(["2025-01-03", "2025-01-06"]).date, "pnl_r": [2.0, 3.0]})

    book = _portfolio_series([leg_a, leg_b]).sort_values("trading_day").reset_index(drop=True)

    assert book["pnl_r"].tolist() == [1.0, 1.0, 3.0]


def test_verdict_requires_family_and_inherited_dsr_for_continue() -> None:
    base = pd.Series(
        {
            "q_family": 0.001,
            "t": 4.0,
            "dsr_family": 0.99,
            "dsr_inherited": 0.1,
            "wfe": 0.8,
            "era_ok": True,
        }
    )

    assert _verdict(base) == "NARROW"

    clear = base.copy()
    clear["dsr_inherited"] = 0.99

    assert _verdict(clear) == "CONTINUE"


def test_verdict_kills_failed_no_theory_t_gate() -> None:
    row = pd.Series(
        {
            "q_family": 0.001,
            "t": 3.5,
            "dsr_family": 0.99,
            "dsr_inherited": 0.99,
            "wfe": 0.8,
            "era_ok": True,
        }
    )

    assert _verdict(row) == "KILL"
