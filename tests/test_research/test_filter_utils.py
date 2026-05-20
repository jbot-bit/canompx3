"""Tests for research.filter_utils — canonical filter signal wrappers.

Verifies that `filter_signal(df, key, orb_label)` is equivalent to calling
`ALL_FILTERS[key].matches_df(df, orb_label).fillna(False).to_numpy().astype(int)`
directly. The whole point of the wrapper is that it is NOT re-encoding
logic — these tests prove the wrapper is transparent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.filter_utils import filter_signal, filter_signals
from trading_app.config import ALL_FILTERS


# ─────────────────────────────────────────────────────────────────────────
# Test data — synthetic rows that exercise the filter branches without
# needing a DuckDB fixture. Feature columns match the canonical
# `orb_{orb_label}_*` naming that matches_df looks up.
# ─────────────────────────────────────────────────────────────────────────


def _vwap_df() -> pd.DataFrame:
    """5-row DataFrame exercising VWAP filter: aligned long, aligned short,
    misaligned long, misaligned short, missing vwap."""
    return pd.DataFrame(
        {
            "orb_US_DATA_1000_high": [100.0, 90.0, 100.0, 90.0, 100.0],
            "orb_US_DATA_1000_low": [99.0, 89.0, 99.0, 89.0, 99.0],
            "orb_US_DATA_1000_vwap": [98.0, 95.0, 101.0, 88.0, float("nan")],
            "orb_US_DATA_1000_break_dir": ["long", "short", "long", "short", "long"],
        }
    )


def _orb_size_df() -> pd.DataFrame:
    """4-row DataFrame exercising OrbSizeFilter (ORB_G5 => size >= 5.0)."""
    return pd.DataFrame(
        {
            "orb_COMEX_SETTLE_size": [3.0, 5.0, 7.5, float("nan")],
            # OrbSizeFilter also reads orb_{label}_break_dir for some paths
            "orb_COMEX_SETTLE_break_dir": ["long", "long", "short", "long"],
            "orb_COMEX_SETTLE_high": [100.0, 105.0, 107.5, 100.0],
            "orb_COMEX_SETTLE_low": [97.0, 100.0, 100.0, 99.0],
        }
    )


# ─────────────────────────────────────────────────────────────────────────
# filter_signal — API contract
# ─────────────────────────────────────────────────────────────────────────


class TestFilterSignalContract:
    def test_returns_numpy_int_array(self):
        df = _vwap_df()
        sig = filter_signal(df, "VWAP_MID_ALIGNED", orb_label="US_DATA_1000")
        assert isinstance(sig, np.ndarray)
        assert sig.dtype.kind == "i"
        assert sig.shape == (len(df),)
        # Values in {0, 1}
        assert set(sig.tolist()).issubset({0, 1})

    def test_empty_df_returns_empty_array(self):
        df = pd.DataFrame(
            {
                "orb_US_DATA_1000_high": pd.Series([], dtype=float),
                "orb_US_DATA_1000_low": pd.Series([], dtype=float),
                "orb_US_DATA_1000_vwap": pd.Series([], dtype=float),
                "orb_US_DATA_1000_break_dir": pd.Series([], dtype=str),
            }
        )
        sig = filter_signal(df, "VWAP_MID_ALIGNED", orb_label="US_DATA_1000")
        assert sig.shape == (0,)
        assert sig.dtype.kind == "i"

    def test_unknown_key_raises_keyerror(self):
        df = _vwap_df()
        with pytest.raises(KeyError, match="NOT_A_REAL_FILTER"):
            filter_signal(df, "NOT_A_REAL_FILTER", orb_label="US_DATA_1000")


# ─────────────────────────────────────────────────────────────────────────
# filter_signal — equivalence with canonical matches_df
# This is the load-bearing test: proves filter_utils is a transparent
# wrapper, not a re-encoding.
# ─────────────────────────────────────────────────────────────────────────


class TestFilterSignalEquivalence:
    def _assert_equivalent(self, df: pd.DataFrame, key: str, orb_label: str) -> None:
        """Wrapper output == direct canonical-filter output, element-wise."""
        via_wrapper = filter_signal(df, key, orb_label)
        canonical = ALL_FILTERS[key].matches_df(df, orb_label)
        via_direct = canonical.fillna(False).to_numpy().astype(int)
        np.testing.assert_array_equal(
            via_wrapper,
            via_direct,
            err_msg=f"wrapper diverged from canonical matches_df for {key!r}",
        )

    def test_vwap_mid_aligned_matches_canonical(self):
        df = _vwap_df()
        self._assert_equivalent(df, "VWAP_MID_ALIGNED", "US_DATA_1000")

    def test_vwap_bp_aligned_matches_canonical(self):
        df = _vwap_df()
        self._assert_equivalent(df, "VWAP_BP_ALIGNED", "US_DATA_1000")

    def test_orb_g5_matches_canonical(self):
        df = _orb_size_df()
        self._assert_equivalent(df, "ORB_G5", "COMEX_SETTLE")

    def test_orb_g8_matches_canonical(self):
        df = _orb_size_df()
        self._assert_equivalent(df, "ORB_G8", "COMEX_SETTLE")

    def test_ovnrng_100_matches_canonical(self):
        df = pd.DataFrame(
            {
                "overnight_range": [5.0, 20.0, 40.0, float("nan")],
                "atr_20": [10.0, 20.0, 20.0, 15.0],
            }
        )
        self._assert_equivalent(df, "OVNRNG_100", "COMEX_SETTLE")

    def test_atr_p50_matches_canonical(self):
        df = pd.DataFrame(
            {
                "atr_20_pct": [10.0, 49.9, 50.0, 75.0, float("nan")],
            }
        )
        self._assert_equivalent(df, "ATR_P50", "COMEX_SETTLE")

    def test_cost_lt12_matches_canonical_with_symbol(self):
        # COST_LT12 needs `symbol` for per-instrument cost lookup. With
        # symbol present, wrapper must equal canonical matches_df.
        df = pd.DataFrame(
            {
                "orb_NYSE_OPEN_size": [10.0, 50.0, 100.0, 200.0, float("nan")],
                "symbol": ["MNQ", "MNQ", "MNQ", "MNQ", "MNQ"],
            }
        )
        self._assert_equivalent(df, "COST_LT12", "NYSE_OPEN")


# ─────────────────────────────────────────────────────────────────────────
# filter_signal — required_columns contract & silent-fail-closed prevention
# Origin: 2026-05-20 wrapper-parity bug. CostRatioFilter.matches_df silently
# returned all-False on a DataFrame missing `symbol`, producing 0/1718 fires
# vs the inline reference's 1695/1718. The wrapper now pre-validates and
# raises ValueError instead of propagating the silent failure.
# ─────────────────────────────────────────────────────────────────────────


class TestRequiredColumnsContract:
    def test_cost_filter_raises_on_missing_symbol(self):
        # The original CHECK 3 repro from the prior session — DataFrame has
        # orb size but no `symbol` column. Must raise ValueError naming the
        # missing column, not silently return [0, 0, 0].
        df = pd.DataFrame(
            {
                "orb_NYSE_OPEN_size": [50.0, 100.0, 200.0],
            }
        )
        with pytest.raises(ValueError, match="symbol"):
            filter_signal(df, "COST_LT12", "NYSE_OPEN")

    def test_cost_filter_raises_on_missing_orb_size(self):
        # Equally a missing prerequisite — orb size column absent.
        df = pd.DataFrame(
            {
                "symbol": ["MNQ", "MNQ", "MNQ"],
            }
        )
        with pytest.raises(ValueError, match="orb_NYSE_OPEN_size"):
            filter_signal(df, "COST_LT12", "NYSE_OPEN")

    def test_cost_filter_error_message_mentions_silent_fail(self):
        # The diagnostic must explain WHY this raises now (silent-fail
        # context) so future readers don't loosen the guard.
        df = pd.DataFrame({"orb_NYSE_OPEN_size": [50.0]})
        with pytest.raises(ValueError, match="silent"):
            filter_signal(df, "COST_LT12", "NYSE_OPEN")

    def test_volume_filter_raises_on_missing_rel_vol(self):
        # VolumeFilter needs rel_vol_{orb_label} injected at discovery time.
        # Find a VOL_RV* key dynamically — registry may rename.
        from trading_app.config import ALL_FILTERS

        vol_keys = [k for k in ALL_FILTERS if k.startswith("VOL_RV")]
        if not vol_keys:
            pytest.skip("No VOL_RV* filter in registry — VolumeFilter retired?")
        df = pd.DataFrame({"orb_NYSE_OPEN_size": [50.0, 100.0]})
        with pytest.raises(ValueError, match="rel_vol_NYSE_OPEN"):
            filter_signal(df, vol_keys[0], "NYSE_OPEN")

    def test_cross_asset_atr_raises_on_missing_injection(self):
        # CrossAssetATRFilter needs cross_atr_{source}_pct injected by
        # _inject_cross_asset_atrs() at discovery/fitness time.
        from trading_app.config import ALL_FILTERS

        x_keys = [k for k in ALL_FILTERS if k.startswith("X_MES_ATR")]
        if not x_keys:
            pytest.skip("No X_MES_ATR* filter in registry")
        df = pd.DataFrame({"orb_NYSE_OPEN_size": [50.0, 100.0]})
        with pytest.raises(ValueError, match="cross_atr_MES_pct"):
            filter_signal(df, x_keys[0], "NYSE_OPEN")

    def test_default_filter_no_required_columns(self):
        # Filters without explicit required_columns override (e.g.,
        # VWAP_MID_ALIGNED, ORB_G5) must continue to accept lean DataFrames
        # without raising — backward-compatibility for the 47 existing
        # wrapper callers.
        df = pd.DataFrame(
            {
                "orb_US_DATA_1000_high": [100.0],
                "orb_US_DATA_1000_low": [99.0],
                "orb_US_DATA_1000_vwap": [98.0],
                "orb_US_DATA_1000_break_dir": ["long"],
            }
        )
        # Must not raise.
        sig = filter_signal(df, "VWAP_MID_ALIGNED", "US_DATA_1000")
        assert sig.shape == (1,)


# ─────────────────────────────────────────────────────────────────────────
# filter_signal — VWAP_MID_ALIGNED behavioral correctness
# Sanity checks that the wrapper correctly propagates the canonical filter's
# fire pattern. Redundant with equivalence tests but catches any regression
# in the canonical filter itself.
# ─────────────────────────────────────────────────────────────────────────


class TestVwapMidAlignedBehavior:
    def test_fires_on_aligned_long(self):
        # orb_mid=99.5 > vwap=98 => long aligned => fire
        df = pd.DataFrame(
            {
                "orb_US_DATA_1000_high": [100.0],
                "orb_US_DATA_1000_low": [99.0],
                "orb_US_DATA_1000_vwap": [98.0],
                "orb_US_DATA_1000_break_dir": ["long"],
            }
        )
        sig = filter_signal(df, "VWAP_MID_ALIGNED", orb_label="US_DATA_1000")
        assert sig[0] == 1

    def test_fails_on_misaligned_long(self):
        # orb_mid=99.5 < vwap=101 => long misaligned => no fire
        df = pd.DataFrame(
            {
                "orb_US_DATA_1000_high": [100.0],
                "orb_US_DATA_1000_low": [99.0],
                "orb_US_DATA_1000_vwap": [101.0],
                "orb_US_DATA_1000_break_dir": ["long"],
            }
        )
        sig = filter_signal(df, "VWAP_MID_ALIGNED", orb_label="US_DATA_1000")
        assert sig[0] == 0

    def test_fails_on_missing_vwap(self):
        df = pd.DataFrame(
            {
                "orb_US_DATA_1000_high": [100.0],
                "orb_US_DATA_1000_low": [99.0],
                "orb_US_DATA_1000_vwap": [float("nan")],
                "orb_US_DATA_1000_break_dir": ["long"],
            }
        )
        sig = filter_signal(df, "VWAP_MID_ALIGNED", orb_label="US_DATA_1000")
        assert sig[0] == 0


# ─────────────────────────────────────────────────────────────────────────
# filter_signals — batch helper
# ─────────────────────────────────────────────────────────────────────────


class TestFilterSignalsBatch:
    def test_returns_dict_per_key(self):
        df = _vwap_df()
        out = filter_signals(df, ["VWAP_MID_ALIGNED", "VWAP_BP_ALIGNED"], orb_label="US_DATA_1000")
        assert set(out.keys()) == {"VWAP_MID_ALIGNED", "VWAP_BP_ALIGNED"}
        assert all(isinstance(v, np.ndarray) for v in out.values())
        assert all(v.shape == (len(df),) for v in out.values())

    def test_empty_keys_list_returns_empty_dict(self):
        df = _vwap_df()
        out = filter_signals(df, [], orb_label="US_DATA_1000")
        assert out == {}

    def test_unknown_key_raises(self):
        df = _vwap_df()
        with pytest.raises(KeyError):
            filter_signals(df, ["VWAP_MID_ALIGNED", "FAKE"], orb_label="US_DATA_1000")
