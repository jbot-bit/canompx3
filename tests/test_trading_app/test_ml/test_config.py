"""Tests for trading_app.ml.config — canonical ML configuration."""

import pathlib

import pytest

from trading_app.ml.config import (
    ACTIVE_INSTRUMENTS,
    ATR_NORMALIZE,
    CATEGORICAL_FEATURES,
    CPCV_EMBARGO_DAYS,
    CPCV_K_TEST,
    CPCV_N_GROUPS,
    CPCV_PURGE_DAYS,
    GLOBAL_FEATURES,
    LOOKAHEAD_BLACKLIST,
    MIN_SAMPLES_TRAIN,
    MODEL_DIR,
    REL_VOL_SESSIONS,
    RF_PARAMS,
    SESSION_FEATURE_SUFFIXES,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    TRADE_CONFIG_FEATURES,
)


class TestFeatureListIntegrity:
    """Feature lists are internally consistent."""

    def test_no_overlap_global_features_and_blacklist(self):
        overlap = set(GLOBAL_FEATURES) & LOOKAHEAD_BLACKLIST
        assert overlap == set(), f"Features in both GLOBAL and BLACKLIST: {overlap}"

    def test_no_overlap_trade_config_and_blacklist(self):
        overlap = set(TRADE_CONFIG_FEATURES) & LOOKAHEAD_BLACKLIST
        assert overlap == set(), f"Features in both TRADE_CONFIG and BLACKLIST: {overlap}"

    def test_no_overlap_session_suffixes_and_blacklist(self):
        overlap = set(SESSION_FEATURE_SUFFIXES) & LOOKAHEAD_BLACKLIST
        assert overlap == set(), f"Suffixes in both SESSION and BLACKLIST: {overlap}"

    def test_no_duplicates_in_global_features(self):
        assert len(GLOBAL_FEATURES) == len(set(GLOBAL_FEATURES))

    def test_no_duplicates_in_categorical_features(self):
        assert len(CATEGORICAL_FEATURES) == len(set(CATEGORICAL_FEATURES))

    def test_no_duplicates_in_session_suffixes(self):
        assert len(SESSION_FEATURE_SUFFIXES) == len(set(SESSION_FEATURE_SUFFIXES))

    def test_all_features_are_strings(self):
        for feat in GLOBAL_FEATURES:
            assert isinstance(feat, str), f"Non-string in GLOBAL_FEATURES: {feat}"
        for feat in CATEGORICAL_FEATURES:
            assert isinstance(feat, str), f"Non-string in CATEGORICAL_FEATURES: {feat}"
        for feat in SESSION_FEATURE_SUFFIXES:
            assert isinstance(feat, str), f"Non-string in SESSION_FEATURE_SUFFIXES: {feat}"

    def test_atr_normalize_targets_exist(self):
        """ATR normalization targets should be plausible feature names."""
        for col in ATR_NORMALIZE:
            assert isinstance(col, str) and len(col) > 0


class TestLookaheadBlacklist:
    """Lookahead guard prevents target leakage."""

    def test_outcome_in_blacklist(self):
        assert "outcome" in LOOKAHEAD_BLACKLIST

    def test_pnl_r_in_blacklist(self):
        assert "pnl_r" in LOOKAHEAD_BLACKLIST

    def test_pnl_dollars_in_blacklist(self):
        assert "pnl_dollars" in LOOKAHEAD_BLACKLIST

    def test_mae_mfe_in_blacklist(self):
        assert "mae_r" in LOOKAHEAD_BLACKLIST
        assert "mfe_r" in LOOKAHEAD_BLACKLIST

    def test_double_break_in_blacklist(self):
        assert "double_break" in LOOKAHEAD_BLACKLIST

    def test_exit_fields_in_blacklist(self):
        assert "exit_ts" in LOOKAHEAD_BLACKLIST
        assert "exit_price" in LOOKAHEAD_BLACKLIST

    def test_time_stop_fields_in_blacklist(self):
        assert "ts_outcome" in LOOKAHEAD_BLACKLIST
        assert "ts_pnl_r" in LOOKAHEAD_BLACKLIST
        assert "ts_exit_ts" in LOOKAHEAD_BLACKLIST

    def test_minimum_blacklist_size(self):
        """Blacklist should never shrink below its founding set."""
        assert len(LOOKAHEAD_BLACKLIST) >= 15


class TestRFParams:
    """Random Forest hyperparameters are valid sklearn kwargs."""

    def test_n_estimators_positive(self):
        assert RF_PARAMS["n_estimators"] > 0

    def test_max_depth_positive(self):
        assert RF_PARAMS["max_depth"] > 0

    def test_min_samples_leaf_positive(self):
        assert RF_PARAMS["min_samples_leaf"] > 0

    def test_class_weight_balanced(self):
        assert RF_PARAMS["class_weight"] == "balanced"

    def test_random_state_deterministic(self):
        assert RF_PARAMS["random_state"] == 42

    def test_valid_sklearn_keys(self):
        valid_keys = {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "random_state",
            "n_jobs",
        }
        assert set(RF_PARAMS.keys()) == valid_keys


class TestCPCVConfig:
    """CPCV configuration yields valid split counts."""

    def test_n_groups_positive(self):
        assert CPCV_N_GROUPS > 0

    def test_k_test_less_than_n_groups(self):
        assert CPCV_K_TEST < CPCV_N_GROUPS

    def test_purge_days_non_negative(self):
        assert CPCV_PURGE_DAYS >= 0

    def test_embargo_days_non_negative(self):
        assert CPCV_EMBARGO_DAYS >= 0

    def test_expected_splits_count(self):
        """C(10, 2) = 45 splits."""
        from math import comb

        expected = comb(CPCV_N_GROUPS, CPCV_K_TEST)
        assert expected == 45


class TestThresholdConfig:
    """Threshold search range is valid."""

    def test_min_less_than_max(self):
        assert THRESHOLD_MIN < THRESHOLD_MAX

    def test_step_positive(self):
        assert THRESHOLD_STEP > 0

    def test_range_within_probabilities(self):
        assert 0.0 <= THRESHOLD_MIN <= 1.0
        assert 0.0 <= THRESHOLD_MAX <= 1.0


class TestActiveInstruments:
    """Active instruments match pipeline canonical source."""

    def test_active_instrument_count(self):
        assert len(ACTIVE_INSTRUMENTS) == 3

    def test_canonical_instruments_present(self):
        expected = {"MGC", "MNQ", "MES"}
        assert set(ACTIVE_INSTRUMENTS) == expected

    def test_subset_of_pipeline_source(self):
        """ML instruments must be a subset of pipeline instruments
        (excludes instruments with no validated strategies, e.g. MBT)."""
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        extra = set(ACTIVE_INSTRUMENTS) - set(ACTIVE_ORB_INSTRUMENTS)
        assert extra == set(), f"ML has instruments not in pipeline: {extra}"


class TestRelVolSessions:
    """REL_VOL_SESSIONS matches session catalog."""

    def test_sessions_match_catalog(self):
        from pipeline.dst import SESSION_CATALOG

        catalog_dynamic = {name for name, cfg in SESSION_CATALOG.items() if cfg.get("type") == "dynamic"}
        rel_vol_set = set(REL_VOL_SESSIONS)
        assert rel_vol_set == catalog_dynamic, (
            f"REL_VOL_SESSIONS mismatch: extra={rel_vol_set - catalog_dynamic}, missing={catalog_dynamic - rel_vol_set}"
        )


class TestModelDirectory:
    """Model persistence directory configured correctly."""

    def test_model_dir_is_path(self):
        assert isinstance(MODEL_DIR, pathlib.Path)

    def test_model_dir_ends_with_models_ml(self):
        parts = MODEL_DIR.parts
        assert parts[-2] == "models" and parts[-1] == "ml"

    def test_min_samples_train_positive(self):
        assert MIN_SAMPLES_TRAIN > 0
