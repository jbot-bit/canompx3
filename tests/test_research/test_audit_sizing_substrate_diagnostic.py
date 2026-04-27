# tests/test_research/test_audit_sizing_substrate_diagnostic.py
"""Tests for research/audit_sizing_substrate_diagnostic.py.

Per spec v0.2 §5.2-§5.4a: every gate must be testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.audit_sizing_substrate_diagnostic import (
    apply_bh_fdr,
    bootstrap_sized_vs_flat_ci,
    check_forecast_stability,
    classify_cell,
    compute_quintile_lift,
    is_holdout_clean,
    null_coverage_mark,
    power_floor_mark,
    sign_match_split_half,
)


# -- Holdout guard --------------------------------------------------------------


def test_holdout_clean_returns_true_when_all_pre_2026():
    df = pd.DataFrame({"trading_day": pd.to_datetime(["2025-12-30", "2025-12-31"])})
    assert is_holdout_clean(df, holdout="2026-01-01") is True


def test_holdout_clean_raises_when_any_row_in_holdout():
    df = pd.DataFrame({"trading_day": pd.to_datetime(["2025-12-31", "2026-01-02"])})
    with pytest.raises(RuntimeError, match="holdout row"):
        is_holdout_clean(df, holdout="2026-01-01")


# -- NULL coverage --------------------------------------------------------------


def test_null_coverage_pass_when_drop_under_20pct():
    f = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])  # 1/5 = 20% — boundary inclusive
    status, drop_frac = null_coverage_mark(f, threshold=0.20)
    assert status == "OK"
    assert drop_frac == pytest.approx(0.20)


def test_null_coverage_marks_invalid_when_drop_over_20pct():
    f = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0])  # 2/5 = 40%
    status, drop_frac = null_coverage_mark(f, threshold=0.20)
    assert status == "INVALID"
    assert drop_frac == pytest.approx(0.40)


# -- Power floor ----------------------------------------------------------------


def test_power_floor_pass_at_902():
    assert power_floor_mark(902, min_n=902) == "OK"


def test_power_floor_fails_at_901():
    assert power_floor_mark(901, min_n=902) == "UNDERPOWERED"


# -- Quintile lift --------------------------------------------------------------


def test_quintile_lift_monotonic_increasing():
    rng = np.random.default_rng(7)
    f = np.linspace(0, 10, 1000)
    pnl = f * 0.05 + rng.normal(0, 0.5, 1000)  # signal + noise, sign POS
    df = pd.DataFrame({"f": f, "pnl_r": pnl})
    res = compute_quintile_lift(df, feature_col="f", outcome_col="pnl_r")
    assert res["monotonic"] is True
    assert res["q5_mean_r"] > res["q1_mean_r"]


def test_quintile_lift_non_monotonic_flagged():
    rng = np.random.default_rng(7)
    f = np.arange(1000)
    pnl = np.sin(f / 50) + rng.normal(0, 0.1, 1000)  # oscillating
    df = pd.DataFrame({"f": f, "pnl_r": pnl})
    res = compute_quintile_lift(df, feature_col="f", outcome_col="pnl_r")
    assert res["monotonic"] is False


# -- Bootstrap sized-vs-flat ----------------------------------------------------


def test_bootstrap_sized_vs_flat_ci_positive_when_real_edge():
    rng = np.random.default_rng(7)
    n = 2000
    f = rng.uniform(0, 1, n)
    pnl = f * 0.5 + rng.normal(0, 0.4, n)  # strong positive edge
    df = pd.DataFrame({"f": f, "pnl_r": pnl})
    ci = bootstrap_sized_vs_flat_ci(
        df,
        feature_col="f",
        outcome_col="pnl_r",
        weights=(0.6, 0.8, 1.0, 1.2, 1.4),
        predicted_sign="+",
        B=10000,
        seed=42,
    )
    assert ci["lo"] > 0
    assert ci["hi"] > ci["lo"]


def test_bootstrap_seed_reproducibility():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"f": rng.uniform(0, 1, 500), "pnl_r": rng.normal(0, 0.5, 500)})
    a = bootstrap_sized_vs_flat_ci(df, "f", "pnl_r", (0.6, 0.8, 1.0, 1.2, 1.4), "+", B=1000, seed=42)
    b = bootstrap_sized_vs_flat_ci(df, "f", "pnl_r", (0.6, 0.8, 1.0, 1.2, 1.4), "+", B=1000, seed=42)
    assert a == b


# -- BH-FDR ---------------------------------------------------------------------


def test_bh_fdr_passes_strong_signal():
    pvals = [0.0001, 0.001, 0.5, 0.6, 0.7]
    out = apply_bh_fdr(pvals, q=0.05)
    assert out[0] is True  # very small p must survive
    assert out[1] is True
    assert out[3] is False
    assert out[4] is False


def test_bh_fdr_no_survivors_under_uniform_pvals():
    pvals = [0.30, 0.40, 0.50, 0.60, 0.70]
    out = apply_bh_fdr(pvals, q=0.05)
    assert all(v is False for v in out)


# -- Forecast stability ---------------------------------------------------------


def test_forecast_stability_stable_over_time():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "trading_day": pd.date_range("2010-01-01", periods=2000, freq="D"),
            "f": rng.normal(10, 1.0, 2000),  # stable SD
        }
    )
    assert check_forecast_stability(df, feature_col="f", window=252, max_rel_var=0.50) == "STABLE"


def test_forecast_stability_unstable_when_sd_drifts():
    rng = np.random.default_rng(7)
    early = rng.normal(10, 0.2, 1000)  # tight
    late = rng.normal(10, 5.0, 1000)  # loose
    df = pd.DataFrame(
        {
            "trading_day": pd.date_range("2010-01-01", periods=2000, freq="D"),
            "f": np.concatenate([early, late]),
        }
    )
    assert check_forecast_stability(df, feature_col="f", window=252, max_rel_var=0.50) == "UNSTABLE"


# -- Split-half sign stability --------------------------------------------------


def test_sign_match_split_half_passes_when_signs_match():
    df = pd.DataFrame(
        {
            "trading_day": pd.date_range("2010-01-01", periods=1000, freq="D"),
            "f": np.linspace(0, 1, 1000),
            "pnl_r": np.linspace(-0.5, 0.5, 1000),  # both halves positive correlation
        }
    )
    assert sign_match_split_half(df, feature_col="f", outcome_col="pnl_r") is True


def test_sign_match_split_half_fails_when_signs_flip():
    df = pd.DataFrame(
        {
            "trading_day": pd.date_range("2010-01-01", periods=1000, freq="D"),
            "f": np.concatenate([np.linspace(0, 1, 500), np.linspace(0, 1, 500)]),
            "pnl_r": np.concatenate([np.linspace(0, 1, 500), np.linspace(1, 0, 500)]),  # flip
        }
    )
    assert sign_match_split_half(df, feature_col="f", outcome_col="pnl_r") is False


# -- Final classifier (compose) -------------------------------------------------


def test_classify_cell_pass_when_all_gates_pass():
    cell = {
        "null_status": "OK",
        "power_status": "OK",
        "rho": 0.15,
        "rho_p": 0.0001,
        "bh_fdr_pass": True,
        "monotonic": True,
        "q5_minus_q1": 0.30,
        "sized_flat_delta_lo": 0.02,
        "sized_flat_delta_hi": 0.08,
        "split_half_rho_match": True,
        "split_half_delta_match": True,
        "predicted_sign": "+",
        "realized_sign": "+",
        "stability_status": "STABLE",
    }
    assert classify_cell(cell) == "PASS"


def test_classify_cell_fail_when_underpowered():
    cell = {"null_status": "OK", "power_status": "UNDERPOWERED", "rho": 0.30}
    assert classify_cell(cell) == "FAIL"


def test_classify_cell_fail_when_prediction_flipped():
    cell = {
        "null_status": "OK",
        "power_status": "OK",
        "rho": 0.15,
        "rho_p": 0.0001,
        "bh_fdr_pass": True,
        "monotonic": True,
        "q5_minus_q1": 0.30,
        "sized_flat_delta_lo": 0.02,
        "sized_flat_delta_hi": 0.08,
        "split_half_rho_match": True,
        "split_half_delta_match": True,
        "predicted_sign": "+",
        "realized_sign": "-",  # flipped
        "stability_status": "STABLE",
    }
    assert classify_cell(cell) == "FAIL"


def test_classify_cell_invalid_when_null_heavy():
    cell = {"null_status": "INVALID"}
    assert classify_cell(cell) == "INVALID"
