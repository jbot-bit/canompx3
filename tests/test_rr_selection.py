"""Tests for JK-MaxExpR family RR lock selection logic."""

import numpy as np
import pandas as pd
import pytest

from scripts.tools.select_family_rr import (
    FAMILY_COLS,
    _jobson_korkie_p,
    select_rr_for_family,
)


# --- Unit tests for Jobson-Korkie ---


def test_jk_identical_sharpe_returns_p1():
    """Identical Sharpes -> p = 1.0 (cannot reject equality)."""
    p = _jobson_korkie_p(1.5, 1.5, 200, 200, 0.7)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_jk_very_different_sharpe_returns_small_p():
    """Large Sharpe difference with large N -> small p."""
    p = _jobson_korkie_p(2.0, 0.5, 500, 500, 0.7)
    assert p < 0.01


def test_jk_small_n_returns_large_p():
    """Small samples -> can't distinguish moderate differences."""
    p = _jobson_korkie_p(1.5, 1.0, 10, 10, 0.7)
    assert p > 0.05


def test_jk_very_small_n_returns_1():
    """N < 5 -> treat as equal."""
    p = _jobson_korkie_p(2.0, 0.5, 3, 3, 0.7)
    assert p == 1.0


def test_jk_symmetric():
    """JK test should be symmetric (absolute difference)."""
    p1 = _jobson_korkie_p(1.5, 1.0, 200, 200, 0.7)
    p2 = _jobson_korkie_p(1.0, 1.5, 200, 200, 0.7)
    assert p1 == pytest.approx(p2, abs=1e-10)


# --- Unit tests for select_rr_for_family ---


def _make_family_row(rr, sharpe, maxdd, n=200, expr=0.10, tpy=50.0):
    """Helper to create a single strategy row."""
    return {
        "instrument": "MGC",
        "orb_label": "TOKYO_OPEN",
        "filter_type": "NO_FILTER",
        "entry_model": "E2",
        "orb_minutes": 5,
        "confirm_bars": 1,
        "rr_target": rr,
        "sharpe_ratio": sharpe,
        "max_drawdown_r": maxdd,
        "sample_size": n,
        "expectancy_r": expr,
        "trades_per_year": tpy,
        "win_rate": 0.55,
    }


def test_single_rr_returns_only_rr():
    """Family with one RR level -> method=ONLY_RR."""
    df = pd.DataFrame([_make_family_row(1.0, 1.2, 10.0)])
    result = select_rr_for_family(df)
    assert result["method"] == "ONLY_RR"
    assert result["locked_rr"] == 1.0


def test_best_sharpe_with_best_expr_returns_max_sharpe():
    """Best Sharpe also has best ExpR -> method=MAX_SHARPE."""
    df = pd.DataFrame(
        [
            _make_family_row(1.0, 1.5, 8.0, expr=0.20),  # best Sharpe AND best ExpR
            _make_family_row(2.0, 1.3, 15.0, expr=0.15),
        ]
    )
    result = select_rr_for_family(df)
    assert result["method"] == "MAX_SHARPE"
    assert result["locked_rr"] == 1.0


def test_max_expr_picks_highest_expr_from_equal_sharpe():
    """Equal Sharpes, different ExpR -> picks highest ExpR (MAX_EXPR)."""
    # Sharpes close enough that JK won't reject, but best Sharpe != best ExpR
    df = pd.DataFrame(
        [
            _make_family_row(1.0, 1.35, 8.0, n=200, expr=0.08),  # best Sharpe, low ExpR
            _make_family_row(2.0, 1.30, 20.0, n=200, expr=0.18),  # lower Sharpe, high ExpR
        ]
    )
    result = select_rr_for_family(df)
    assert result["method"] == "MAX_EXPR"
    assert result["locked_rr"] == 2.0  # picked RR2.0 for its higher ExpR
    assert result["expr_at_rr"] == pytest.approx(0.18)


def test_significantly_better_sharpe_wins_despite_lower_expr():
    """If one RR has SIGNIFICANTLY better Sharpe, it wins even with lower ExpR."""
    # Massive Sharpe difference with large N -> JK rejects equality
    df = pd.DataFrame(
        [
            _make_family_row(1.0, 0.3, 5.0, n=500, expr=0.25),  # terrible Sharpe, high ExpR
            _make_family_row(2.0, 2.5, 25.0, n=500, expr=0.15),  # amazing Sharpe, lower ExpR
        ]
    )
    result = select_rr_for_family(df)
    # RR1.0 should be excluded (JK p < 0.05 vs best), so RR2.0 wins
    assert result["locked_rr"] == 2.0


def test_multi_rr_all_equal_sharpe_picks_highest_expr():
    """4 RR levels with similar Sharpes -> picks highest ExpR."""
    df = pd.DataFrame(
        [
            _make_family_row(1.0, 1.20, 10.0, n=200, expr=0.08),
            _make_family_row(1.5, 1.25, 14.0, n=200, expr=0.12),
            _make_family_row(2.0, 1.22, 18.0, n=200, expr=0.18),
            _make_family_row(2.5, 1.18, 22.0, n=200, expr=0.22),
        ]
    )
    result = select_rr_for_family(df)
    assert result["locked_rr"] == 2.5  # highest ExpR among JK-equal Sharpes
    assert result["expr_at_rr"] == pytest.approx(0.22)


def test_family_key_columns_preserved():
    """Result includes all 6 family key columns."""
    df = pd.DataFrame([_make_family_row(1.5, 1.0, 12.0)])
    result = select_rr_for_family(df)
    for col in FAMILY_COLS:
        assert col in result, f"Missing family key column: {col}"
    assert result["instrument"] == "MGC"
    assert result["orb_minutes"] == 5
    assert result["confirm_bars"] == 1


def test_metrics_match_selected_rr():
    """Metrics in result must match the selected RR row, not some other row."""
    df = pd.DataFrame(
        [
            _make_family_row(1.0, 1.30, 8.0, n=150, expr=0.08, tpy=40),
            _make_family_row(2.0, 1.32, 20.0, n=180, expr=0.15, tpy=45),
        ]
    )
    result = select_rr_for_family(df)
    # Should pick RR2.0 (MAX_EXPR — higher ExpR from equal Sharpe)
    assert result["locked_rr"] == 2.0
    assert result["n_at_rr"] == 180
    assert result["expr_at_rr"] == pytest.approx(0.15)
    assert result["tpy_at_rr"] == pytest.approx(45.0)
    assert result["sharpe_at_rr"] == pytest.approx(1.32)
