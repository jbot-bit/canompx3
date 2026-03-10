"""Tests for trading_app.ml.features — feature extraction pipeline."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from trading_app.ml.config import (
    CATEGORICAL_FEATURES,
    GLOBAL_FEATURES,
    LOOKAHEAD_BLACKLIST,
    TRADE_CONFIG_FEATURES,
)
from trading_app.ml.features import (
    _backfill_global_features,
    _encode_categoricals,
    _extract_session_features,
    _normalize_features,
)


class TestExtractSessionFeatures:
    """Session-specific feature extraction from wide to generic columns."""

    @pytest.fixture
    def mock_df(self):
        """DataFrame mimicking daily_features + orb_outcomes join."""
        return pd.DataFrame(
            {
                "orb_label": ["CME_REOPEN", "TOKYO_OPEN", "CME_REOPEN"],
                "orb_CME_REOPEN_size": [5.0, np.nan, 8.0],
                "orb_TOKYO_OPEN_size": [np.nan, 3.0, np.nan],
                "orb_CME_REOPEN_volume": [1000, np.nan, 2000],
                "orb_TOKYO_OPEN_volume": [np.nan, 500, np.nan],
                "orb_CME_REOPEN_break_bar_volume": [100, np.nan, 200],
                "orb_TOKYO_OPEN_break_bar_volume": [np.nan, 50, np.nan],
                "orb_CME_REOPEN_break_delay_min": [2.0, np.nan, 5.0],
                "orb_TOKYO_OPEN_break_delay_min": [np.nan, 1.0, np.nan],
                "orb_CME_REOPEN_break_bar_continues": [True, np.nan, False],
                "orb_TOKYO_OPEN_break_bar_continues": [np.nan, True, np.nan],
                "orb_CME_REOPEN_break_dir": ["LONG", np.nan, "SHORT"],
                "orb_TOKYO_OPEN_break_dir": [np.nan, "LONG", np.nan],
                "rel_vol_CME_REOPEN": [1.1, np.nan, 0.9],
                "rel_vol_TOKYO_OPEN": [np.nan, 1.3, np.nan],
            }
        )

    def test_extracts_correct_session_values(self, mock_df):
        result = _extract_session_features(mock_df)
        # Row 0: CME_REOPEN → size = 5.0
        assert result.loc[0, "orb_size"] == 5.0
        # Row 1: TOKYO_OPEN → size = 3.0
        assert result.loc[1, "orb_size"] == 3.0
        # Row 2: CME_REOPEN → size = 8.0
        assert result.loc[2, "orb_size"] == 8.0

    def test_extracts_rel_vol(self, mock_df):
        result = _extract_session_features(mock_df)
        assert result.loc[0, "rel_vol"] == 1.1
        assert result.loc[1, "rel_vol"] == 1.3

    def test_output_has_generic_columns(self, mock_df):
        result = _extract_session_features(mock_df)
        # Pre-break features only: size, volume, rel_vol
        # break_delay_min, break_bar_volume, break_bar_continues, break_dir
        # removed Mar 4 2026 (pre-break ML architecture + theory audit)
        expected_cols = {"orb_size", "orb_volume", "rel_vol"}
        assert expected_cols == set(result.columns)


class TestNormalizeFeatures:
    """ATR normalization creates _norm columns."""

    def test_creates_norm_columns(self):
        df = pd.DataFrame(
            {
                "atr_20": [10.0, 20.0],
                "gap_open_points": [5.0, 10.0],
                "orb_size": [3.0, 6.0],
            }
        )
        result = _normalize_features(df)
        assert "gap_open_points_norm" in result.columns
        assert "orb_size_norm" in result.columns

    def test_norm_values_correct(self):
        df = pd.DataFrame(
            {
                "atr_20": [10.0, 20.0],
                "gap_open_points": [5.0, 10.0],
            }
        )
        result = _normalize_features(df)
        assert result.loc[0, "gap_open_points_norm"] == 0.5
        assert result.loc[1, "gap_open_points_norm"] == 0.5

    def test_zero_atr_becomes_nan(self):
        df = pd.DataFrame(
            {
                "atr_20": [0.0],
                "gap_open_points": [5.0],
            }
        )
        result = _normalize_features(df)
        assert pd.isna(result.loc[0, "gap_open_points_norm"])

    def test_original_columns_preserved(self):
        df = pd.DataFrame(
            {
                "atr_20": [10.0],
                "gap_open_points": [5.0],
            }
        )
        result = _normalize_features(df)
        assert "gap_open_points" in result.columns
        assert result.loc[0, "gap_open_points"] == 5.0


class TestEncodeCategoricals:
    """One-hot encoding handles NaN and string categories."""

    def test_one_hot_creates_columns(self):
        df = pd.DataFrame(
            {
                "orb_label": ["CME_REOPEN", "TOKYO_OPEN", "CME_REOPEN"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        result = _encode_categoricals(df)
        assert "orb_label_CME_REOPEN" in result.columns
        assert "orb_label_TOKYO_OPEN" in result.columns
        assert "orb_label" not in result.columns

    def test_nan_becomes_unknown(self):
        df = pd.DataFrame(
            {
                "orb_label": ["CME_REOPEN", np.nan],
                "value": [1.0, 2.0],
            }
        )
        result = _encode_categoricals(df)
        assert "orb_label_UNKNOWN" in result.columns

    def test_non_categorical_columns_preserved(self):
        df = pd.DataFrame(
            {
                "orb_label": ["CME_REOPEN"],
                "value": [1.0],
            }
        )
        result = _encode_categoricals(df)
        assert "value" in result.columns

    def test_no_categoricals_returns_unchanged(self):
        df = pd.DataFrame({"value": [1.0, 2.0]})
        result = _encode_categoricals(df)
        assert list(result.columns) == ["value"]


class TestBackfillGlobalFeatures:
    """_backfill_global_features fills NULL globals from orb_minutes=5 rows."""

    def _make_mock_con(self, g5_rows: pd.DataFrame | None = None):
        """Create a mock DuckDB connection that returns g5_rows from fetchdf()."""
        con = MagicMock()
        result = MagicMock()
        if g5_rows is not None:
            result.fetchdf.return_value = g5_rows
        else:
            result.fetchdf.return_value = pd.DataFrame(columns=["trading_day"] + GLOBAL_FEATURES)
        con.execute.return_value = result
        return con

    def test_no_missing_skips_db_query(self):
        """When all global features are present, no DB query should fire."""
        df = pd.DataFrame(
            {
                "trading_day": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                **{col: [1.0, 2.0] for col in GLOBAL_FEATURES},
            }
        )
        con = MagicMock()
        result = _backfill_global_features(df, con, "MGC")
        # No DB query should have been made
        con.execute.assert_not_called()
        assert len(result) == 2

    def test_fills_null_from_o5(self):
        """NULL global features are filled from orb_minutes=5 data."""
        days = pd.to_datetime(["2025-01-01", "2025-01-02"])
        df = pd.DataFrame(
            {
                "trading_day": days,
                "atr_20": [10.0, 20.0],  # populated
                "atr_vel_ratio": [np.nan, np.nan],  # NULL
                "gap_open_points": [1.0, 2.0],  # populated
                "prev_day_range": [np.nan, np.nan],  # NULL
                "overnight_range": [np.nan, np.nan],  # NULL
            }
        )
        g5 = pd.DataFrame(
            {
                "trading_day": days,
                "atr_20": [10.0, 20.0],
                "atr_vel_ratio": [1.1, 1.2],
                "gap_open_points": [1.0, 2.0],
                "prev_day_range": [30.0, 40.0],
                "overnight_range": [15.0, 25.0],
            }
        )
        con = self._make_mock_con(g5)
        result = _backfill_global_features(df, con, "MGC")

        assert result["atr_vel_ratio"].tolist() == [1.1, 1.2]
        assert result["prev_day_range"].tolist() == [30.0, 40.0]
        assert result["overnight_range"].tolist() == [15.0, 25.0]

    def test_does_not_overwrite_existing_values(self):
        """Backfill only fills NULL — never overwrites existing data."""
        days = pd.to_datetime(["2025-01-01"])
        df = pd.DataFrame(
            {
                "trading_day": days,
                "atr_20": [10.0],
                "atr_vel_ratio": [999.0],  # existing — must NOT be overwritten
                "gap_open_points": [1.0],
                "prev_day_range": [np.nan],  # NULL — should fill
                "overnight_range": [np.nan],
            }
        )
        g5 = pd.DataFrame(
            {
                "trading_day": days,
                "atr_20": [10.0],
                "atr_vel_ratio": [1.1],  # different from existing 999.0
                "gap_open_points": [1.0],
                "prev_day_range": [30.0],
                "overnight_range": [15.0],
            }
        )
        con = self._make_mock_con(g5)
        result = _backfill_global_features(df, con, "MGC")

        # Existing value preserved
        assert result["atr_vel_ratio"].iloc[0] == 999.0
        # NULL values filled
        assert result["prev_day_range"].iloc[0] == 30.0
        assert result["overnight_range"].iloc[0] == 15.0

    def test_missing_o5_rows_stay_null(self):
        """If no orb_minutes=5 row exists for a day, features stay NULL."""
        days = pd.to_datetime(["2025-01-01", "2025-01-02"])
        df = pd.DataFrame(
            {
                "trading_day": days,
                "atr_20": [10.0, 20.0],
                "atr_vel_ratio": [np.nan, np.nan],
                "gap_open_points": [1.0, 2.0],
                "prev_day_range": [np.nan, np.nan],
                "overnight_range": [np.nan, np.nan],
            }
        )
        # Only one day in g5
        g5 = pd.DataFrame(
            {
                "trading_day": pd.to_datetime(["2025-01-01"]),
                "atr_20": [10.0],
                "atr_vel_ratio": [1.1],
                "gap_open_points": [1.0],
                "prev_day_range": [30.0],
                "overnight_range": [15.0],
            }
        )
        con = self._make_mock_con(g5)
        result = _backfill_global_features(df, con, "MGC")

        # Day 1: filled
        assert result["atr_vel_ratio"].iloc[0] == 1.1
        # Day 2: still NaN (no O5 data)
        assert pd.isna(result["atr_vel_ratio"].iloc[1])

    def test_preserves_non_global_columns(self):
        """Columns not in GLOBAL_FEATURES are untouched."""
        days = pd.to_datetime(["2025-01-01"])
        df = pd.DataFrame(
            {
                "trading_day": days,
                "my_custom_col": [42.0],
                **{col: [np.nan] for col in GLOBAL_FEATURES},
            }
        )
        g5 = pd.DataFrame(
            {
                "trading_day": days,
                **{col: [1.0] for col in GLOBAL_FEATURES},
            }
        )
        con = self._make_mock_con(g5)
        result = _backfill_global_features(df, con, "MGC")

        assert result["my_custom_col"].iloc[0] == 42.0
        assert "my_custom_col_g5" not in result.columns

    def test_empty_g5_query_no_crash(self):
        """Empty O5 table returns df unchanged (except possibly NaN still)."""
        days = pd.to_datetime(["2025-01-01"])
        df = pd.DataFrame(
            {
                "trading_day": days,
                **{col: [np.nan] for col in GLOBAL_FEATURES},
            }
        )
        con = self._make_mock_con(None)  # Empty result
        result = _backfill_global_features(df, con, "MGC")
        assert len(result) == 1
        # Still NaN since no O5 data
        assert pd.isna(result[GLOBAL_FEATURES[0]].iloc[0])


class TestFillMissingFeatures:
    """_fill_missing_features uses correct fill values per column type."""

    def test_onehot_filled_with_zero(self):
        from trading_app.ml.evaluate import _fill_missing_features

        X = pd.DataFrame({"existing": [1.0, 2.0]})
        feature_names = ["existing", "orb_label_CME_REOPEN", "entry_model_E2"]
        result = _fill_missing_features(X, feature_names)
        assert result["orb_label_CME_REOPEN"].tolist() == [0.0, 0.0]
        assert result["entry_model_E2"].tolist() == [0.0, 0.0]

    def test_numeric_filled_with_negative_999(self):
        from trading_app.ml.evaluate import _fill_missing_features

        X = pd.DataFrame({"existing": [1.0, 2.0]})
        feature_names = ["existing", "some_numeric_feature"]
        result = _fill_missing_features(X, feature_names)
        assert result["some_numeric_feature"].tolist() == [-999.0, -999.0]

    def test_column_order_matches_feature_names(self):
        from trading_app.ml.evaluate import _fill_missing_features

        X = pd.DataFrame({"b": [1.0], "a": [2.0]})
        feature_names = ["a", "b", "orb_label_X"]
        result = _fill_missing_features(X, feature_names)
        assert list(result.columns) == ["a", "b", "orb_label_X"]


class TestLookaheadSafety:
    """Verify no look-ahead features can leak into the feature matrix."""

    def test_blacklist_has_all_targets(self):
        targets = {"outcome", "pnl_r", "pnl_dollars", "mae_r", "mfe_r"}
        assert targets.issubset(LOOKAHEAD_BLACKLIST)

    def test_trade_config_not_in_blacklist(self):
        """rr_target, confirm_bars, orb_minutes are KNOWN at entry time."""
        for feat in TRADE_CONFIG_FEATURES:
            assert feat not in LOOKAHEAD_BLACKLIST, f"{feat} is in TRADE_CONFIG but also in BLACKLIST"

    def test_blacklist_catches_session_prefixed_columns(self):
        """Substring matching should catch orb_CME_REOPEN_mae_r etc."""
        test_cols = ["orb_CME_REOPEN_mae_r", "orb_TOKYO_OPEN_mfe_r", "orb_NYSE_OPEN_pnl_r", "safe_feature"]
        caught = [
            col for col in test_cols if col in LOOKAHEAD_BLACKLIST or any(bl in col for bl in LOOKAHEAD_BLACKLIST)
        ]
        assert "safe_feature" not in caught
        assert len(caught) == 3


class TestLoadFeatureMatrixIntegration:
    """Integration test for load_feature_matrix against real DB."""

    @pytest.fixture
    def db_path(self):
        """Path to gold.db — skips if not available."""
        import os

        path = os.environ.get("DUCKDB_PATH", "gold.db")
        if not os.path.exists(path):
            pytest.skip("gold.db not available")
        return path

    def test_returns_correct_shapes(self, db_path):
        """X, y, meta have consistent row counts and expected structure."""
        from trading_app.ml.features import load_feature_matrix

        X, y, meta = load_feature_matrix(db_path, "MGC", min_date="2024-06-01", max_date="2024-12-31")

        assert len(X) == len(y) == len(meta)
        assert len(X) > 0

    def test_x_is_all_numeric(self, db_path):
        """Feature matrix must be all-numeric after processing."""
        from trading_app.ml.features import load_feature_matrix

        X, _, _ = load_feature_matrix(db_path, "MGC", min_date="2024-06-01", max_date="2024-12-31")

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        assert non_numeric == [], f"Non-numeric columns in X: {non_numeric}"

    def test_no_lookahead_columns(self, db_path):
        """No blacklisted column names or substrings should survive."""
        from trading_app.ml.features import load_feature_matrix

        X, _, _ = load_feature_matrix(db_path, "MGC", min_date="2024-06-01", max_date="2024-12-31")

        for col in X.columns:
            for bl in LOOKAHEAD_BLACKLIST:
                assert bl not in col, f"Lookahead column leaked: {col} (contains {bl})"

    def test_meta_has_required_columns(self, db_path):
        """Meta should have trading_day, pnl_r, orb_label at minimum."""
        from trading_app.ml.features import load_feature_matrix

        _, _, meta = load_feature_matrix(db_path, "MGC", min_date="2024-06-01", max_date="2024-12-31")

        required = {"trading_day", "pnl_r", "orb_label", "symbol"}
        assert required.issubset(set(meta.columns)), f"Missing meta columns: {required - set(meta.columns)}"

    def test_y_is_binary(self, db_path):
        """Target must be 0/1 only."""
        from trading_app.ml.features import load_feature_matrix

        _, y, _ = load_feature_matrix(db_path, "MGC", min_date="2024-06-01", max_date="2024-12-31")

        assert set(y.unique()).issubset({0, 1})
