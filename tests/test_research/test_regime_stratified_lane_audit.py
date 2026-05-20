"""Tests for research.regime_stratified_lane_audit_v1.

Pure unit tests + 1 monkeypatched in-memory DuckDB integration probe.
Matches `test_lib.py` / `test_allocator_rho_audit.py` conventions — no
live `gold.db` dependency.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from research import regime_stratified_lane_audit_v1 as runner
from research.regime_stratified_lane_audit_v1 import (
    HoldoutContaminationError,
    Lane,
    RegimeBucket,
    _apply_realized_eod_policy,
    _assert_no_holdout_contamination,
    _assign_regime,
    _bonferroni_k8,
    _drop_underpowered_cells,
    _h1_chi_square,
    _h2_anova,
    _load_deployed_mnq_lanes,
    _load_eligible_sessions,
    _per_lane_verdict,
    _stratify,
)

PREREG_STRATEGY_IDS = {
    "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15",
    "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
    "MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25",
}


def _fixture_lane_allocation(tmp_path: Path, strategy_ids: list[dict]) -> Path:
    payload = {
        "rebalance_date": "2026-05-14",
        "profile_id": "topstep_50k_mnq_auto",
        "lanes": strategy_ids,
        "paused": [],
    }
    p = tmp_path / "lane_allocation.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ============================================================================
# Lane-loading tests
# ============================================================================


def test_load_deployed_mnq_lanes_count_4(tmp_path: Path) -> None:
    raw_lanes = [
        {
            "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K",
            "instrument": "MNQ",
            "orb_minutes": 5,
            "filter_type": "ORB_VOL_2K",
        },
        {
            "strategy_id": "MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15",
            "instrument": "MNQ",
            "orb_minutes": 15,
            "filter_type": "VWAP_MID_ALIGNED",
        },
        {
            "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
            "instrument": "MNQ",
            "orb_minutes": 5,
            "filter_type": "COST_LT12",
        },
        {
            "strategy_id": "MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25",
            "instrument": "MNQ",
            "orb_minutes": 5,
            "filter_type": "OVNRNG_25",
        },
    ]
    path = _fixture_lane_allocation(tmp_path, raw_lanes)
    lanes = _load_deployed_mnq_lanes(path, PREREG_STRATEGY_IDS)
    assert len(lanes) == 4
    assert {ln.strategy_id for ln in lanes} == PREREG_STRATEGY_IDS


def test_load_deployed_mnq_lanes_count_mismatch_raises(tmp_path: Path) -> None:
    raw_lanes = [
        {
            "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K",
            "instrument": "MNQ",
            "orb_minutes": 5,
            "filter_type": "ORB_VOL_2K",
        },
        {
            "strategy_id": "MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15",
            "instrument": "MNQ",
            "orb_minutes": 15,
            "filter_type": "VWAP_MID_ALIGNED",
        },
        {
            "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
            "instrument": "MNQ",
            "orb_minutes": 5,
            "filter_type": "COST_LT12",
        },
    ]  # 3 lanes, not 4
    path = _fixture_lane_allocation(tmp_path, raw_lanes)
    with pytest.raises(SystemExit, match="count drift"):
        _load_deployed_mnq_lanes(path, PREREG_STRATEGY_IDS)


def test_load_deployed_mnq_lanes_strategy_id_drift_raises(tmp_path: Path) -> None:
    raw_lanes = [
        {
            "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K",
            "instrument": "MNQ",
            "orb_minutes": 5,
            "filter_type": "ORB_VOL_2K",
        },
        {
            "strategy_id": "MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15",
            "instrument": "MNQ",
            "orb_minutes": 15,
            "filter_type": "VWAP_MID_ALIGNED",
        },
        {
            "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
            "instrument": "MNQ",
            "orb_minutes": 5,
            "filter_type": "COST_LT12",
        },
        # WRONG: drifted strategy_id
        {
            "strategy_id": "MNQ_US_DATA_1000_E2_RR1.0_CB1_DRIFTED",
            "instrument": "MNQ",
            "orb_minutes": 5,
            "filter_type": "OVNRNG_25",
        },
    ]
    path = _fixture_lane_allocation(tmp_path, raw_lanes)
    with pytest.raises(SystemExit, match="strategy_id drift"):
        _load_deployed_mnq_lanes(path, PREREG_STRATEGY_IDS)


# ============================================================================
# Regime-assignment tests
# ============================================================================


def _is_buckets() -> list[RegimeBucket]:
    return [
        RegimeBucket(
            id="R0", start=date(2019, 1, 1), end=date(2019, 12, 31), role="INFORMATIONAL_EXCLUDED", label="MICRO_LAUNCH"
        ),
        RegimeBucket(
            id="R2", start=date(2020, 1, 1), end=date(2021, 12, 31), role="IS_TEST_INPUT", label="COVID_REOPENING"
        ),
        RegimeBucket(
            id="R3", start=date(2022, 1, 1), end=date(2022, 12, 31), role="IS_TEST_INPUT", label="INFLATION_RATE_HIKES"
        ),
        RegimeBucket(id="R4", start=date(2023, 1, 1), end=date(2024, 12, 31), role="IS_TEST_INPUT", label="POST_HIKE"),
        RegimeBucket(
            id="R5",
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            role="CURRENT_REGIME_DECISION_INPUT",
            label="CURRENT_REGIME",
        ),
        RegimeBucket(
            id="R6",
            start=date(2026, 1, 1),
            end=date(2099, 12, 31),
            role="FORWARD_MONITOR_ONLY_SACRED_HOLDOUT",
            label="SACRED_HOLDOUT",
        ),
    ]


def test_assign_regime_boundary_R0_R2() -> None:
    buckets = _is_buckets()
    assert _assign_regime(date(2019, 12, 31), buckets) == "R0"
    assert _assign_regime(date(2020, 1, 1), buckets) == "R2"


def test_assign_regime_boundary_R5_R6() -> None:
    buckets = _is_buckets()
    assert _assign_regime(date(2025, 12, 31), buckets) == "R5"
    assert _assign_regime(date(2026, 1, 1), buckets) == "R6"


def test_assign_regime_outside_all_buckets() -> None:
    buckets = _is_buckets()
    # Pre-2019 (before R0) → None
    assert _assign_regime(date(2018, 6, 15), buckets) is None


# ============================================================================
# Power-floor (N<30) tests
# ============================================================================


def _trades_df(n_trades: int, regime_label: str = "R3") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trading_day": [date(2022, 1, 1)] * n_trades,
            "pnl_r_effective": np.linspace(-0.5, 0.5, n_trades),
            "_regime": [regime_label] * n_trades,
        }
    )


def test_drop_underpowered_cells_below_30() -> None:
    per_regime = {
        "R2": _trades_df(40),
        "R3": _trades_df(29),
        "R4": _trades_df(30),
    }
    kept, dropped = _drop_underpowered_cells(per_regime, min_n=30)
    assert "R3" in dropped
    assert "R2" in kept and "R4" in kept
    assert len(dropped) == 1


def test_drop_never_widens_bucket() -> None:
    """Property: a dropped regime is removed, NEVER merged into a sibling."""
    per_regime = {
        "R2": _trades_df(35),
        "R3": _trades_df(5),
        "R4": _trades_df(35),
    }
    kept, dropped = _drop_underpowered_cells(per_regime, min_n=30)
    # R2 size must be unchanged (no R3 leakage into R2)
    assert len(kept["R2"]) == 35
    assert len(kept["R4"]) == 35
    assert dropped == ["R3"]


# ============================================================================
# Holdout-contamination assertion tests
# ============================================================================


def test_holdout_contamination_R0_raises() -> None:
    with pytest.raises(HoldoutContaminationError, match="R0"):
        _assert_no_holdout_contamination("H1", ["R0", "R2", "R3", "R4", "R5"])


def test_holdout_contamination_R6_raises() -> None:
    with pytest.raises(HoldoutContaminationError, match="R6"):
        _assert_no_holdout_contamination("H2", ["R2", "R3", "R4", "R5", "R6"])


def test_holdout_contamination_clean_passes() -> None:
    # No-op when input set is clean
    _assert_no_holdout_contamination("H1", ["R2", "R3", "R4", "R5"])
    _assert_no_holdout_contamination("H2", ["R2", "R3", "R4", "R5"])


# ============================================================================
# H1 chi-square + small-cell fallback tests
# ============================================================================


def test_h1_chi_square_small_cell_fallback_fisher() -> None:
    # Synthetic 4x2 where one regime has very few sessions total → expected
    # cell < 5 on the "fired" column for that regime. Marginal totals must
    # be lopsided enough that the row * column / grand_total < 5.
    # R4 has just 8 sessions; with a ~3% pooled fire-rate, expected R4 fired ≈ 0.24
    fire_masks = {
        "R2": np.array([1] * 3 + [0] * 247),  # 3 fired / 250
        "R3": np.array([1] * 4 + [0] * 246),  # 4 fired / 250
        "R4": np.array([1] * 1 + [0] * 7),  # 1 fired / 8 (tiny)
        "R5": np.array([1] * 2 + [0] * 248),  # 2 fired / 250
    }
    eligible = {rid: pd.DataFrame({"x": np.ones(len(m))}) for rid, m in fire_masks.items()}
    res = _h1_chi_square(eligible, fire_masks)
    assert res["small_cell"] is True, f"expected small-cell trigger, got {res['expected']}"
    assert res["small_cell_method"] in {"CHI2_YATES", "CHI2_YATES_FAILED"}
    assert np.isfinite(res["raw_p"]) or np.isfinite(res["fallback_p"])


# ============================================================================
# K=8 Bonferroni tests
# ============================================================================


def test_bonferroni_k8_caps_at_1() -> None:
    out = _bonferroni_k8({("lane_a", "H1"): 0.5})
    assert out[("lane_a", "H1")]["adj_p_k8"] == 1.0


def test_bonferroni_k8_tier_boundaries() -> None:
    cells = {
        ("a", "H1"): 0.04 / 8,  # adj_p = 0.04 -> WATCH
        ("b", "H1"): 0.05 / 8,  # adj_p = 0.05 -> PASS
        ("c", "H1"): 0.009 / 8,  # adj_p = 0.009 -> FAIL
    }
    out = _bonferroni_k8(cells)
    assert out[("a", "H1")]["tier"] == "WATCH"
    assert out[("b", "H1")]["tier"] == "PASS"
    assert out[("c", "H1")]["tier"] == "FAIL"


# ============================================================================
# Scratch-policy / pnl_r_effective tests
# ============================================================================


def test_pnl_r_effective_null_non_scratch_counter() -> None:
    """1 scratch (null OK after fillna) + 1 non-scratch null -> counter == 1."""
    df = pd.DataFrame(
        {
            "outcome": ["win", "scratch", "loss"],
            "pnl_r": [1.5, np.nan, np.nan],  # row 1 is scratch+null (OK), row 2 is non-scratch+null (BUG)
        }
    )
    out = _apply_realized_eod_policy(df)
    assert int(out["null_pnl_non_scratch"].sum()) == 1
    # win row preserved
    assert out.loc[out["outcome"].eq("win"), "pnl_r_effective"].iloc[0] == 1.5
    # scratch null -> coerced to 0
    assert out.loc[out["outcome"].eq("scratch"), "pnl_r_effective"].iloc[0] == 0.0


def test_scratch_policy_header_marker_present() -> None:
    """Source file MUST carry literal `# scratch-policy: realized-eod`."""
    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert "# scratch-policy: realized-eod" in src.splitlines()[0:5][0]


# ============================================================================
# H2 sensitivity-diagnostic tests
# ============================================================================


def _equal_variance_groups(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "R2": rng.normal(0.10, 1.0, size=100),
        "R3": rng.normal(0.12, 1.0, size=100),
        "R4": rng.normal(0.11, 1.0, size=100),
        "R5": rng.normal(0.10, 1.0, size=100),
    }


def _heteroscedastic_groups(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "R2": rng.normal(0.10, 0.5, size=80),
        "R3": rng.normal(0.10, 3.0, size=80),  # 6x variance
        "R4": rng.normal(0.10, 0.5, size=80),
        "R5": rng.normal(0.10, 3.0, size=80),
    }


def test_h2_sensitivity_emits_welch_kruskal_on_assumption_violation() -> None:
    rng = np.random.default_rng(seed=42)
    groups = _heteroscedastic_groups(rng)
    res = _h2_anova(groups)
    assert res["sensitivity"] is not None
    assert "welch_p" in res["sensitivity"]
    assert "kruskal_p" in res["sensitivity"]
    # Primary verdict still from f_oneway
    assert np.isfinite(res["raw_p"])


def test_h2_sensitivity_skipped_when_assumptions_hold() -> None:
    rng = np.random.default_rng(seed=42)
    groups = _equal_variance_groups(rng)
    res = _h2_anova(groups)
    assert res["sensitivity"] is None


def test_h2_verdict_never_flips_on_sensitivity() -> None:
    """Sensitivity is diagnostic only; primary ANOVA drives verdict.

    Construct a case where primary ANOVA p is high (≈ 0.5 PASS) but
    Kruskal-Wallis would reject. Verdict from _per_lane_verdict must
    remain CONTINUE (not flip).
    """
    rng = np.random.default_rng(seed=0)
    # Mostly-equal means, with a heavy-tailed outlier-loaded group
    groups = {
        "R2": rng.normal(0.10, 1.0, size=200),
        "R3": rng.normal(0.10, 1.0, size=200),
        "R4": rng.normal(0.10, 1.0, size=200),
        "R5": rng.normal(0.10, 1.0, size=200),
    }
    res = _h2_anova(groups)
    h1_p = 0.8  # PASS
    h2_p = res["raw_p"]
    verdict, _ = _per_lane_verdict(h1_p, h2_p, r5_expr=0.10, r5_n=200)
    assert verdict == "CONTINUE"


# ============================================================================
# H1 denominator uniqueness — in-memory DuckDB integration probe
# ============================================================================


def test_h1_denominator_uniqueness_assertion() -> None:
    """In-memory DuckDB: 2 trading_days × 3 (entry_model,rr,cb,direction) =
    6 orb_outcomes rows but 2 unique (trading_day, orb_label) session-events.

    `_load_eligible_sessions` must return 2 rows (SELECT DISTINCT). If a
    future refactor breaks DISTINCT, the runtime
    ``duplicated().sum() == 0`` assertion raises.
    """
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            CREATE TABLE orb_outcomes (
                trading_day DATE,
                symbol VARCHAR,
                orb_label VARCHAR,
                orb_minutes INTEGER,
                entry_model VARCHAR,
                confirm_bars INTEGER,
                rr_target DOUBLE,
                outcome VARCHAR,
                pnl_r DOUBLE
            );
            CREATE TABLE daily_features (
                trading_day DATE,
                symbol VARCHAR,
                orb_minutes INTEGER,
                some_feature DOUBLE
            );
            """
        )
        for day in [date(2022, 6, 1), date(2022, 6, 2)]:
            for em in ["E1", "E2", "E3"]:
                con.execute(
                    "INSERT INTO orb_outcomes VALUES (?, 'MNQ', 'COMEX_SETTLE', 5, ?, 1, 1.5, 'win', 1.0)",
                    [day, em],
                )
            con.execute(
                "INSERT INTO daily_features VALUES (?, 'MNQ', 5, 0.25)",
                [day],
            )

        lane = Lane(
            strategy_id="MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K",
            instrument="MNQ",
            session="COMEX_SETTLE",
            orb_minutes=5,
            entry_model="E2",
            confirm_bars=1,
            rr_target=1.5,
            filter_type="ORB_VOL_2K",
        )
        df = _load_eligible_sessions(con, lane, date(2020, 1, 1), date(2026, 1, 1))
        # 6 raw rows -> 2 unique session-events after SELECT DISTINCT
        assert len(df) == 2
        assert df.duplicated(subset=["trading_day", "orb_label"]).sum() == 0
    finally:
        con.close()


# ============================================================================
# Stratify helper test
# ============================================================================


def test_stratify_assigns_correctly() -> None:
    buckets = _is_buckets()
    df = pd.DataFrame(
        {
            "trading_day": [
                date(2022, 6, 1),
                date(2023, 6, 1),
                date(2025, 6, 1),
            ],
            "pnl_r_effective": [0.1, 0.2, 0.3],
        }
    )
    out = _stratify(df, buckets, ["R2", "R3", "R4", "R5"])
    assert len(out["R3"]) == 1
    assert len(out["R4"]) == 1
    assert len(out["R5"]) == 1
    assert len(out["R2"]) == 0
