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
    derive_features,
    feature_temporal_validity,
    resolve_substrate_column,
    stage2_eligible_flag,
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


# -- Feature temporal validity (RULE 1.2 lookahead gate) -----------------------


def test_feature_validity_overnight_invalid_on_tokyo_open():
    lane = {"orb_label": "TOKYO_OPEN"}
    status, reason = feature_temporal_validity(lane, "overnight_range_pct")
    assert status == "INVALID"
    assert "TOKYO_OPEN" in reason


def test_feature_validity_overnight_invalid_on_singapore_open():
    lane = {"orb_label": "SINGAPORE_OPEN"}
    status, _ = feature_temporal_validity(lane, "overnight_range_pct")
    assert status == "INVALID"


def test_feature_validity_overnight_ok_on_europe_flow():
    lane = {"orb_label": "EUROPE_FLOW"}  # 18:00 Brisbane >= 17:00 cutoff
    status, _ = feature_temporal_validity(lane, "overnight_range_pct")
    assert status == "OK"


def test_feature_validity_overnight_ok_on_late_us_sessions():
    for sess in ("NYSE_OPEN", "US_DATA_1000", "COMEX_SETTLE", "CME_PRECLOSE"):
        lane = {"orb_label": sess}
        status, _ = feature_temporal_validity(lane, "overnight_range_pct")
        assert status == "OK", f"{sess} should accept overnight_*"


def test_feature_validity_non_overnight_features_unaffected():
    lane = {"orb_label": "TOKYO_OPEN"}
    for col in ("rel_vol_TOKYO_OPEN", "atr_vel_ratio", "garch_forecast_vol_pct", "pit_range_atr"):
        status, _ = feature_temporal_validity(lane, col)
        assert status == "OK", f"{col} should not be gated"


def test_feature_validity_none_column_passes():
    lane = {"orb_label": "TOKYO_OPEN"}
    status, _ = feature_temporal_validity(lane, None)
    assert status == "OK"


# -- Stage-2 eligibility (UNSTABLE PASS cells barred) --------------------------


def test_stage2_eligible_true_for_stable_pass():
    cell = {"status": "PASS", "stability_status": "STABLE"}
    assert stage2_eligible_flag(cell) is True


def test_stage2_eligible_false_for_unstable_pass():
    cell = {"status": "PASS", "stability_status": "UNSTABLE"}
    assert stage2_eligible_flag(cell) is False


def test_stage2_eligible_false_for_non_pass():
    for status in ("FAIL", "INVALID"):
        cell = {"status": status, "stability_status": "STABLE"}
        assert stage2_eligible_flag(cell) is False


# -- Audit findings 2026-04-27 (PASS_WITH_RISKS) --------------------------------
# Companion tests for findings A, B, D from institutional code+quant audit on
# this diagnostic. Finding C (pooled-finding YAML front-matter) is a doc-only
# fix on the result MD; finding E was verified-correct (no test needed beyond
# existing monotonicity coverage).


# RULE 13 pressure-test (Finding D, MED — mandatory rule).
# .claude/rules/backtesting-methodology.md RULE 13:
#   "deliberately introduce a known-bad feature ... and confirm the script
#    flags or filters it. If the pressure test passes through silently, fix
#    the guard before trusting the scan."


def test_rule13_pressure_test_double_break_blocked():
    """RULE 1.1 hard look-ahead: double_break must be rejected at every session."""
    for session in ("EUROPE_FLOW", "TOKYO_OPEN", "NYSE_OPEN"):
        lane = {"orb_label": session, "entry_model": "E2"}
        status, reason = feature_temporal_validity(lane, "double_break")
        assert status == "INVALID", f"double_break should be blocked on {session}"
        assert "RULE 1.1" in reason or "look-ahead" in reason.lower()


def test_rule13_pressure_test_post_trade_columns_blocked():
    """Hard-banned post-trade columns (mae_r, mfe_r, outcome, pnl_r-as-predictor)
    must be rejected regardless of session or entry_model."""
    lane = {"orb_label": "EUROPE_FLOW", "entry_model": "E2"}
    for banned in ("mae_r", "mfe_r", "outcome", "pnl_r"):
        status, reason = feature_temporal_validity(lane, banned)
        assert status == "INVALID", f"{banned} should be hard-banned"
        assert banned in reason


def test_rule13_pressure_test_e2_break_bar_features_blocked():
    """RULE 6.3 E2 look-ahead: break-bar columns are post-entry on ~41% of E2
    trades because E2 fires on range-touch but daily_features defines break
    by close-outside-ORB. Canonical authority: trading_app/config.py:3540-3568."""
    lane_e2 = {"orb_label": "EUROPE_FLOW", "entry_model": "E2"}
    for sfx in ("_break_ts", "_break_delay_min", "_break_bar_volume", "_break_bar_continues", "_break_dir"):
        col = f"orb_EUROPE_FLOW{sfx}"
        status, reason = feature_temporal_validity(lane_e2, col)
        assert status == "INVALID", f"{col} should be E2-blocked"
        assert "RULE 6.3" in reason or "E2" in reason


def test_rule13_pressure_test_break_bar_features_allowed_on_non_e2():
    """E2 break-bar lookahead applies only to E2; E1/E3 use these legitimately."""
    lane_e1 = {"orb_label": "EUROPE_FLOW", "entry_model": "E1"}
    status, _ = feature_temporal_validity(lane_e1, "orb_EUROPE_FLOW_break_delay_min")
    assert status == "OK", "E1 may use break_delay_min — gate is E2-specific"


# Finding A documentation: ATR_P50 vol_norm and raw resolve to the same column.
# Effective unique cells = 42, not the K=48 declared in pre-reg. This test makes
# the identity explicit so a future "fix" cannot accidentally diverge them
# without a conscious design decision.


def test_atr_p50_vol_norm_raw_identity_documented():
    lane = {"deployed_filter": "ATR_P50", "orb_label": "SINGAPORE_OPEN"}
    raw = resolve_substrate_column(lane, "raw")
    vol_norm = resolve_substrate_column(lane, "vol_norm")
    assert raw == vol_norm == "atr_20_pct", (
        f"ATR_P50 raw and vol_norm intentionally resolve to the same column "
        f"per pre-reg design (atr_20_pct is already volatility-normalized). "
        f"got raw={raw}, vol_norm={vol_norm}. Effective unique cells=42, K=48 declared."
    )


# Finding B anti-drift: derive_features cost-ratio formula must equal canonical
# CostRatioFilter formula in trading_app/config.py:602-613. Run-time equivalent
# at this commit; this test guards against future drift.


def test_cost_lt12_formula_equivalence_with_canonical():
    from pipeline.cost_model import get_cost_spec

    cs = get_cost_spec("MNQ")
    orb_size = 12.0  # arbitrary positive
    raw_risk_canonical = orb_size * cs.point_value
    canonical_cost_ratio_pct = 100.0 * cs.total_friction / (raw_risk_canonical + cs.total_friction)

    lane = {
        "instrument": "MNQ",
        "orb_label": "EUROPE_FLOW",
        "deployed_filter": "COST_LT12",
    }
    df = pd.DataFrame(
        {
            "orb_EUROPE_FLOW_size": [orb_size],
            "atr_20_pct": [50.0],
        }
    )
    out = derive_features(df, lane)
    diagnostic_cost_ratio_pct = float(out["_cost_ratio"].iloc[0])

    assert abs(diagnostic_cost_ratio_pct - canonical_cost_ratio_pct) < 1e-9, (
        f"derive_features cost_ratio {diagnostic_cost_ratio_pct} != canonical "
        f"CostRatioFilter formula {canonical_cost_ratio_pct} for MNQ orb_size=12. "
        f"Drift detected — re-align with trading_app/config.py:602-613."
    )
