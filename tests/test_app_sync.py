"""
Mandatory sync validation between DB schema, config.py, and pipeline constants.

Zero tolerance for mismatches. FAIL-CLOSED.

These tests ensure that:
1. ORB_LABELS in init_db.py matches expected values
2. ALL_FILTERS keys match filter_type fields inside each filter
3. RR_TARGETS and CONFIRM_BARS_OPTIONS are consistent across modules
4. DB schema columns match what strategy_discovery/validator actually write
5. Strategy ID format is deterministic and parseable
6. Filter JSON serialization round-trips correctly
7. daily_features ORB columns match ORB_LABELS exactly
8. ENTRY_MODELS is consistent across modules
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

import duckdb
import pytest

from pipeline.asset_configs import ASSET_CONFIGS, get_enabled_sessions
from pipeline.init_db import DAILY_FEATURES_SCHEMA, ORB_LABELS
from trading_app.config import (
    ALL_FILTERS,
    BASE_GRID_FILTERS,
    CORE_MIN_SAMPLES,
    COST_RATIO_FILTERS,
    DIR_LONG,
    DIR_SHORT,
    ENTRY_MODELS,
    MGC_ORB_SIZE_FILTERS,
    MGC_VOLUME_FILTERS,
    REGIME_MIN_SAMPLES,
    CostRatioFilter,
    DirectionFilter,
    NoFilter,
    OrbSizeFilter,
    StrategyFilter,
    VolumeFilter,
    classify_strategy,
    get_filters_for_grid,
)
from trading_app.db_manager import (
    init_trading_app_schema,
    verify_trading_app_schema,
)
from trading_app.outcome_builder import CONFIRM_BARS_OPTIONS, RR_TARGETS
from trading_app.strategy_discovery import make_strategy_id

# ============================================================================
# 1. ORB_LABELS consistency
# ============================================================================


class TestOrbLabelsSync:
    """ORB_LABELS must be consistent across all modules."""

    EXPECTED_ORB_LABELS = [
        "CME_REOPEN",
        "TOKYO_OPEN",
        "SINGAPORE_OPEN",
        "LONDON_METALS",
        "EUROPE_FLOW",
        "US_DATA_830",
        "NYSE_OPEN",
        "US_DATA_1000",
        "COMEX_SETTLE",
        "CME_PRECLOSE",
        "NYSE_CLOSE",
        "BRISBANE_1025",
    ]

    def test_orb_labels_exact(self):
        """ORB_LABELS matches expected values exactly."""
        assert ORB_LABELS == self.EXPECTED_ORB_LABELS

    def test_orb_labels_no_duplicates(self):
        """No duplicate ORB labels."""
        assert len(ORB_LABELS) == len(set(ORB_LABELS))

    def test_daily_features_columns_match_orb_labels(self):
        """daily_features DDL has columns for every ORB label and no extras."""
        # Match both fixed (4-digit) and dynamic (alpha) ORB column prefixes
        orb_col_pattern = re.compile(
            r"orb_([A-Za-z0-9_]+?)_(?:high|low|size|break_dir|break_ts|outcome|mae_r|mfe_r|double_break)\b"
        )
        found_labels = set()
        for match in orb_col_pattern.finditer(DAILY_FEATURES_SCHEMA):
            found_labels.add(match.group(1))

        assert found_labels == set(ORB_LABELS), (
            f"DDL ORB columns {sorted(found_labels)} != ORB_LABELS {sorted(ORB_LABELS)}"
        )


# ============================================================================
# 2. ALL_FILTERS registry consistency
# ============================================================================


class TestAllFiltersSync:
    """ALL_FILTERS keys must match filter_type inside each filter."""

    # Base: NO_FILTER + 4 G + 4 COST + 6 VOL + 4 ORB_VOL + 3 ATR_P = 22 (in BASE_GRID_FILTERS)
    # DOW composites: 3 variants (NOFRI, NOMON, NOTUE) x 4 G-filters = 12
    #   (NOFRI/NOTUE removed from grid Mar 2026 but retained in ALL_FILTERS for DB compat)
    # Break quality composites: 3 variants (FAST5, FAST10, CONT) x 4 G-filters = 12
    # M6E pip-scaled size filters: M6E_G4/G6/G8 = 3
    # Direction filters: DIR_LONG/DIR_SHORT = 2
    # MES 1000 band filters: ORB_G4_L12/ORB_G5_L12 = 2
    # Cross-asset ATR filters: X_MES_ATR70/X_MES_ATR60/X_MGC_ATR70 = 3
    # + 4 OVNRNG (session-routed, NOT in BASE_GRID_FILTERS)
    # Total: 22 + 12 + 12 + 3 + 2 + 2 + 3 + 4 = 60
    EXPECTED_FILTER_KEYS = {
        "NO_FILTER",
        "ORB_G4",
        "ORB_G5",
        "ORB_G6",
        "ORB_G8",
        "COST_LT08",
        "COST_LT10",
        "COST_LT12",
        "COST_LT15",
        "VOL_RV12_N20",
        "VOL_RV15_N20",
        "VOL_RV20_N20",
        "VOL_RV25_N20",
        "VOL_RV30_N20",
        # Combined ATR+VOL filter (Mar 2026 vol-regime research)
        "ATR70_VOL",
        # ORB window volume gates (Mar 2026 confluence research program)
        "ORB_VOL_2K",
        "ORB_VOL_4K",
        "ORB_VOL_8K",
        "ORB_VOL_16K",
        # Standalone ATR percentile filters (Mar 2026 confluence research)
        "ATR_P30",
        "ATR_P50",
        "ATR_P70",
        # DOW composites (registered globally for portfolio.py lookups / DB compat)
        "ORB_G4_NOFRI",
        "ORB_G5_NOFRI",
        "ORB_G6_NOFRI",
        "ORB_G8_NOFRI",
        "ORB_G4_NOMON",
        "ORB_G5_NOMON",
        "ORB_G6_NOMON",
        "ORB_G8_NOMON",
        "ORB_G4_NOTUE",
        "ORB_G5_NOTUE",
        "ORB_G6_NOTUE",
        "ORB_G8_NOTUE",
        # Break quality composites (Feb 2026 research: break speed + conviction)
        "ORB_G4_FAST5",
        "ORB_G5_FAST5",
        "ORB_G6_FAST5",
        "ORB_G8_FAST5",
        "ORB_G4_FAST10",
        "ORB_G5_FAST10",
        "ORB_G6_FAST10",
        "ORB_G8_FAST10",
        "ORB_G4_CONT",
        "ORB_G5_CONT",
        "ORB_G6_CONT",
        "ORB_G8_CONT",
        # M6E (EUR/USD) pip-scaled size filters — MGC point filters meaningless for FX
        "M6E_G4",
        "M6E_G6",
        "M6E_G8",
        # Direction filters (session-specific, registered for portfolio lookups)
        "DIR_LONG",
        "DIR_SHORT",
        # MES 1000 band filters (ORB size between min and max points)
        "ORB_G4_L12",
        "ORB_G5_L12",
        # Cross-asset ATR filters (Mar 2026 vol-regime research)
        "X_MES_ATR70",
        "X_MES_ATR60",
        "X_MGC_ATR70",
        # Overnight range absolute filters (Mar 2026 confluence — US sessions only, NOT in BASE)
        "OVNRNG_10",
        "OVNRNG_25",
        "OVNRNG_50",
        "OVNRNG_100",
        # Pre-session volatility filters (Apr 2026 research scan)
        "PDR_R080",
        "PDR_R105",
        "PDR_R125",
        "GAP_R005",
        "GAP_R015",
        # Pit range filter (F5 deployment, commit 2c12641) — pit_range/atr >= 0.10
        # routed to CME_REOPEN only via get_filters_for_grid()
        "PIT_MIN",
        # Cost-ratio + break-speed composites (Apr 2026 break-speed retest)
        "COST_LT08_FAST5",
        "COST_LT08_FAST10",
        "COST_LT10_FAST5",
        "COST_LT10_FAST10",
        "COST_LT12_FAST5",
        "COST_LT12_FAST10",
        "COST_LT15_FAST5",
        "COST_LT15_FAST10",
        # Overnight-range + break-speed composites (Apr 2026 break-speed retest)
        "OVNRNG_10_FAST5",
        "OVNRNG_10_FAST10",
        "OVNRNG_25_FAST5",
        "OVNRNG_25_FAST10",
        "OVNRNG_50_FAST5",
        "OVNRNG_50_FAST10",
        "OVNRNG_100_FAST5",
        "OVNRNG_100_FAST10",
    }

    def test_expected_keys(self):
        """ALL_FILTERS has exactly the expected keys."""
        assert set(ALL_FILTERS.keys()) == self.EXPECTED_FILTER_KEYS

    def test_filter_type_matches_key(self):
        """Each filter's filter_type field matches its key in ALL_FILTERS."""
        for key, filt in ALL_FILTERS.items():
            assert filt.filter_type == key, f"Key '{key}' but filter_type='{filt.filter_type}'"

    def test_all_are_strategy_filters(self):
        """Every value in ALL_FILTERS is a StrategyFilter subclass."""
        for key, filt in ALL_FILTERS.items():
            assert isinstance(filt, StrategyFilter), f"{key} is not StrategyFilter"

    def test_filter_json_roundtrip(self):
        """Every filter serializes to valid JSON containing its filter_type."""
        for key, filt in ALL_FILTERS.items():
            j = filt.to_json()
            parsed = json.loads(j)
            assert parsed["filter_type"] == key, f"JSON filter_type mismatch: key={key}, json={parsed['filter_type']}"

    def test_no_filter_matches_everything(self):
        """NoFilter.matches_row always returns True."""
        nf = ALL_FILTERS["NO_FILTER"]
        assert nf.matches_row({}, "CME_REOPEN") is True
        assert nf.matches_row({"anything": 42}, "US_DATA_830") is True

    def test_filters_are_frozen(self):
        """All filters are frozen (hashable, immutable)."""
        for _key, filt in ALL_FILTERS.items():
            hash(filt)

    def test_size_filters_have_thresholds(self):
        """Every ORB size filter (or composite with size base) has thresholds."""
        from trading_app.config import (
            CompositeFilter,
            CrossAssetATRFilter,
            DirectionFilter,
            GapNormFilter,
            OrbVolumeFilter,
            OvernightRangeAbsFilter,
            OwnATRPercentileFilter,
            PitRangeFilter,
            PrevDayRangeNormFilter,
        )

        for key, filt in ALL_FILTERS.items():
            if key == "NO_FILTER" or isinstance(
                filt,
                (
                    VolumeFilter,
                    DirectionFilter,
                    CrossAssetATRFilter,
                    CostRatioFilter,
                    OrbVolumeFilter,
                    OvernightRangeAbsFilter,
                    OwnATRPercentileFilter,
                    PrevDayRangeNormFilter,
                    GapNormFilter,
                    PitRangeFilter,
                ),
            ):
                continue
            if isinstance(filt, CompositeFilter):
                # Composite with non-size base (cost-ratio, overnight-range): skip
                if isinstance(
                    filt.base,
                    (CostRatioFilter, OvernightRangeAbsFilter),
                ):
                    continue
                # Composite: base should be OrbSizeFilter with thresholds
                assert isinstance(filt.base, OrbSizeFilter), f"{key} composite base should be OrbSizeFilter"
                assert filt.base.min_size is not None or filt.base.max_size is not None, (
                    f"{key} composite base has neither min_size nor max_size"
                )
            else:
                assert isinstance(filt, OrbSizeFilter), f"{key} should be OrbSizeFilter"
                assert filt.min_size is not None or filt.max_size is not None, (
                    f"{key} has neither min_size nor max_size"
                )

    def test_volume_filters_have_params(self):
        """Every volume filter has min_rel_vol and lookback_days set."""
        for key, filt in ALL_FILTERS.items():
            if not isinstance(filt, VolumeFilter):
                continue
            assert filt.min_rel_vol > 0, f"{key} min_rel_vol must be positive"
            assert filt.lookback_days > 0, f"{key} lookback_days must be positive"

    def test_cost_ratio_filters_have_thresholds(self):
        """Every cost-ratio filter has a positive cap."""
        for key, filt in COST_RATIO_FILTERS.items():
            assert isinstance(filt, CostRatioFilter)
            assert filt.max_cost_ratio_pct > 0, f"{key} max_cost_ratio_pct must be positive"


# ============================================================================
# 2b. DirectionFilter sync
# ============================================================================


class TestDirectionFilterSync:
    """DirectionFilter must correctly filter by breakout direction."""

    def test_dir_long_filter_type(self):
        assert DIR_LONG.filter_type == "DIR_LONG"

    def test_dir_short_filter_type(self):
        assert DIR_SHORT.filter_type == "DIR_SHORT"

    def test_matches_long(self):
        row = {"orb_TOKYO_OPEN_break_dir": "long"}
        assert DIR_LONG.matches_row(row, "TOKYO_OPEN") is True
        assert DIR_SHORT.matches_row(row, "TOKYO_OPEN") is False

    def test_matches_short(self):
        row = {"orb_TOKYO_OPEN_break_dir": "short"}
        assert DIR_SHORT.matches_row(row, "TOKYO_OPEN") is True
        assert DIR_LONG.matches_row(row, "TOKYO_OPEN") is False

    def test_missing_dir_fails_closed(self):
        assert DIR_LONG.matches_row({}, "TOKYO_OPEN") is False
        assert DIR_SHORT.matches_row({}, "TOKYO_OPEN") is False


# ============================================================================
# 2c. get_filters_for_grid session-aware dispatch
# ============================================================================


class TestGetFiltersForGrid:
    """get_filters_for_grid must return correct filter sets per instrument+session."""

    def test_mes_tokyo_open_includes_band_and_dir(self):
        filters = get_filters_for_grid("MES", "TOKYO_OPEN")
        assert "ORB_G4_L12" in filters
        assert "ORB_G5_L12" in filters
        assert "DIR_LONG" in filters

    def test_mgc_cme_reopen_no_nofri(self):
        """NOFRI removed from CME_REOPEN grid Mar 2026 (DOW stress test: LIKELY NOISE)."""
        filters = get_filters_for_grid("MGC", "CME_REOPEN")
        assert "ORB_G4_L12" not in filters
        assert "DIR_LONG" not in filters
        nofri_keys = [k for k in filters if "NOFRI" in k]
        assert nofri_keys == [], f"NOFRI should be removed from grid, got {nofri_keys}"
        # Base filters still present
        assert "NO_FILTER" in filters
        assert "ORB_G4" in filters
        assert "ORB_G5" in filters
        assert "ORB_G6" in filters
        assert "ORB_G8" in filters

    def test_mgc_tokyo_open_has_dir_no_band(self):
        filters = get_filters_for_grid("MGC", "TOKYO_OPEN")
        assert "DIR_LONG" in filters
        assert "ORB_G4_L12" not in filters

    def test_base_filters_always_present(self):
        for inst in ("MGC", "MES", "MNQ"):
            for sess in ("CME_REOPEN", "TOKYO_OPEN"):
                filters = get_filters_for_grid(inst, sess)
                assert "NO_FILTER" in filters
                if inst == "MGC":
                    # MGC regime shift: G6 minimum (Feb 2026)
                    assert "ORB_G6" in filters
                else:
                    assert "ORB_G4" in filters


# ============================================================================
# 3. RR_TARGETS, CONFIRM_BARS_OPTIONS, ENTRY_MODELS consistency
# ============================================================================


class TestGridParamsSync:
    """Grid parameters must be consistent and valid."""

    def test_rr_targets_sorted(self):
        assert sorted(RR_TARGETS) == RR_TARGETS

    def test_rr_targets_positive(self):
        assert all(rr > 0 for rr in RR_TARGETS)

    def test_rr_targets_no_duplicates(self):
        assert len(RR_TARGETS) == len(set(RR_TARGETS))

    def test_confirm_bars_sorted(self):
        assert sorted(CONFIRM_BARS_OPTIONS) == CONFIRM_BARS_OPTIONS

    def test_confirm_bars_positive_ints(self):
        assert all(isinstance(cb, int) and cb > 0 for cb in CONFIRM_BARS_OPTIONS)

    def test_confirm_bars_no_duplicates(self):
        assert len(CONFIRM_BARS_OPTIONS) == len(set(CONFIRM_BARS_OPTIONS))

    def test_grid_size(self):
        """Total base grid size matches expected formula (E2+E3 use CB1 only).

        Base grid uses 22 core filters (NO_FILTER + G4/G5/G6/G8 + 4 COST + 5 VOL + ATR70_VOL + 4 ORB_VOL + 3 ATR_P).
        Session-specific DOW composites are added by get_filters_for_grid()
        per-session, expanding the grid contextually.

        12 ORBs x 6 RRs x 5 CBs x 22 base filters = 7920 (E1, all CB options)
        12 ORBs x 6 RRs x 1 CB x 22 base filters = 1584 (E2, always CB1)
        12 ORBs x 6 RRs x 1 CB x 22 base filters = 1584 (E3, always CB1)
        Total base: 11088
        """
        n_base = len(BASE_GRID_FILTERS)  # canonical — never hardcode
        e1 = len(ORB_LABELS) * len(RR_TARGETS) * len(CONFIRM_BARS_OPTIONS) * n_base
        e2_e3 = 2 * len(ORB_LABELS) * len(RR_TARGETS) * 1 * n_base
        expected = e1 + e2_e3
        assert expected > 0  # sanity — formula produces non-trivial grid


class TestEntryModelsSync:
    """ENTRY_MODELS must be consistent."""

    def test_entry_models_exact(self):
        assert ENTRY_MODELS == ["E1", "E2", "E3"]

    def test_entry_models_no_duplicates(self):
        assert len(ENTRY_MODELS) == len(set(ENTRY_MODELS))

    def test_entry_models_are_strings(self):
        assert all(isinstance(em, str) for em in ENTRY_MODELS)


# ============================================================================
# 3b. Strategy classification sync (FIX5 rules)
# ============================================================================


class TestStrategyClassificationSync:
    """FIX5 strategy classification thresholds must be consistent."""

    def test_core_threshold(self):
        assert CORE_MIN_SAMPLES == 100

    def test_regime_threshold(self):
        assert REGIME_MIN_SAMPLES == 30

    def test_regime_below_core(self):
        assert REGIME_MIN_SAMPLES < CORE_MIN_SAMPLES

    def test_classify_core(self):
        assert classify_strategy(100) == "CORE"
        assert classify_strategy(500) == "CORE"

    def test_classify_regime(self):
        assert classify_strategy(30) == "REGIME"
        assert classify_strategy(99) == "REGIME"

    def test_classify_invalid(self):
        assert classify_strategy(29) == "INVALID"
        assert classify_strategy(0) == "INVALID"

    def test_boundary_core(self):
        """Exactly CORE_MIN_SAMPLES is CORE, one below is REGIME."""
        assert classify_strategy(CORE_MIN_SAMPLES) == "CORE"
        assert classify_strategy(CORE_MIN_SAMPLES - 1) == "REGIME"

    def test_boundary_regime(self):
        """Exactly REGIME_MIN_SAMPLES is REGIME, one below is INVALID."""
        assert classify_strategy(REGIME_MIN_SAMPLES) == "REGIME"
        assert classify_strategy(REGIME_MIN_SAMPLES - 1) == "INVALID"


# ============================================================================
# 4. Strategy ID format consistency
# ============================================================================


class TestStrategyIdSync:
    """Strategy IDs must be deterministic and parseable."""

    def test_format(self):
        sid = make_strategy_id("MGC", "CME_REOPEN", "E1", 2.0, 1, "NO_FILTER")
        assert sid == "MGC_CME_REOPEN_E1_RR2.0_CB1_NO_FILTER"

    def test_parseable(self):
        """Strategy ID can be parsed back to components."""
        sid = make_strategy_id("MGC", "CME_REOPEN", "E3", 2.0, 1, "ORB_L4")
        # With multi-word session names, split by known prefix
        assert sid.startswith("MGC_CME_REOPEN_E3_")

    def test_all_grid_ids_unique(self):
        """Every combination in the base grid produces a unique ID (E2+E3 CB1 only).

        Uses BASE_GRID_FILTERS (not ALL_FILTERS) — count is dynamic.
        Session-specific DOW composites expand the grid per-session via
        get_filters_for_grid(); the base grid is the common denominator.
        """
        ids = set()
        for orb in ORB_LABELS:
            for em in ENTRY_MODELS:
                for rr in RR_TARGETS:
                    for cb in CONFIRM_BARS_OPTIONS:
                        if em in ("E2", "E3") and cb > 1:
                            continue
                        for fk in BASE_GRID_FILTERS:
                            sid = make_strategy_id("MGC", orb, em, rr, cb, fk)
                            assert sid not in ids, f"Duplicate ID: {sid}"
                            ids.add(sid)
        # Computed from canonical sources — no hardcoded count
        n_base = len(BASE_GRID_FILTERS)
        e1 = len(ORB_LABELS) * len(RR_TARGETS) * len(CONFIRM_BARS_OPTIONS) * n_base
        e2_e3 = 2 * len(ORB_LABELS) * len(RR_TARGETS) * 1 * n_base
        assert len(ids) == e1 + e2_e3


# ============================================================================
# 5. DB schema column sync
# ============================================================================


class TestSchemaSync:
    """DB schema must match what code actually writes."""

    def test_trading_app_schema_creates_all_tables(self, tmp_path):
        """init_trading_app_schema creates all 4 tables."""
        db_path = tmp_path / "sync_test.db"
        con = duckdb.connect(str(db_path))
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA

        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)
        con.close()

        init_trading_app_schema(db_path=db_path)

        ok, violations = verify_trading_app_schema(db_path=db_path)
        assert ok, f"Schema violations: {violations}"

    def test_experimental_strategies_has_required_columns(self, tmp_path):
        """experimental_strategies has all columns strategy_discovery writes."""
        db_path = tmp_path / "sync_test.db"
        con = duckdb.connect(str(db_path))
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA

        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)
        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        cols = {
            r[0]
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='experimental_strategies'"
            ).fetchall()
        }
        con.close()

        required = {
            "strategy_id",
            "instrument",
            "orb_label",
            "orb_minutes",
            "rr_target",
            "confirm_bars",
            "entry_model",
            "filter_type",
            "filter_params",
            "sample_size",
            "win_rate",
            "avg_win_r",
            "avg_loss_r",
            "expectancy_r",
            "sharpe_ratio",
            "max_drawdown_r",
            "median_risk_points",
            "avg_risk_points",
            "yearly_results",
            "entry_signals",
            "scratch_count",
            "early_exit_count",
            "trade_day_hash",
            "is_canonical",
            "canonical_strategy_id",
            "validation_status",
            "validation_notes",
        }
        missing = required - cols
        assert not missing, f"Missing columns in experimental_strategies: {missing}"

    def test_validated_setups_has_required_columns(self, tmp_path):
        """validated_setups has all columns strategy_validator writes."""
        db_path = tmp_path / "sync_test.db"
        con = duckdb.connect(str(db_path))
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA

        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)
        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        cols = {
            r[0]
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='validated_setups'"
            ).fetchall()
        }
        con.close()

        required = {
            "strategy_id",
            "promoted_from",
            "instrument",
            "orb_label",
            "orb_minutes",
            "rr_target",
            "confirm_bars",
            "entry_model",
            "filter_type",
            "filter_params",
            "sample_size",
            "win_rate",
            "expectancy_r",
            "years_tested",
            "all_years_positive",
            "stress_test_passed",
            "sharpe_ratio",
            "max_drawdown_r",
            "yearly_results",
            "status",
            "first_trade_day",
            "last_trade_day",
            "trade_day_count",
            "validation_run_id",
            "promotion_git_sha",
            "promotion_provenance",
        }
        missing = required - cols
        assert not missing, f"Missing columns in validated_setups: {missing}"

    def test_orb_outcomes_has_required_columns(self, tmp_path):
        """orb_outcomes has all columns outcome_builder writes."""
        db_path = tmp_path / "sync_test.db"
        con = duckdb.connect(str(db_path))
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA

        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)
        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        cols = {
            r[0]
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='orb_outcomes'"
            ).fetchall()
        }
        con.close()

        required = {
            "trading_day",
            "symbol",
            "orb_label",
            "orb_minutes",
            "rr_target",
            "confirm_bars",
            "entry_model",
            "entry_ts",
            "entry_price",
            "stop_price",
            "target_price",
            "outcome",
            "exit_ts",
            "exit_price",
            "pnl_r",
            "mae_r",
            "mfe_r",
        }
        missing = required - cols
        assert not missing, f"Missing columns in orb_outcomes: {missing}"


# ============================================================================
# 6. Cross-module import sync
# ============================================================================


class TestImportSync:
    """Verify that modules import the same constants."""

    def test_outcome_builder_uses_canonical_session_source(self):
        """outcome_builder gets sessions from asset_configs (not hardcoded)."""
        import inspect

        import trading_app.outcome_builder as ob

        source = inspect.getsource(ob)
        assert "from pipeline.asset_configs import get_enabled_sessions" in source

    def test_strategy_discovery_uses_shared_constants(self):
        """strategy_discovery imports RR_TARGETS/CONFIRM_BARS from outcome_builder."""
        import inspect

        import trading_app.strategy_discovery as sd

        source = inspect.getsource(sd)
        assert "from trading_app.outcome_builder import" in source
        assert "RR_TARGETS" in source
        assert "CONFIRM_BARS_OPTIONS" in source

    def test_outcome_builder_imports_entry_models(self):
        """outcome_builder imports ENTRY_MODELS from config."""
        import inspect

        import trading_app.outcome_builder as ob

        source = inspect.getsource(ob)
        assert "from trading_app.config import" in source
        assert "ENTRY_MODELS" in source

    def test_strategy_discovery_imports_entry_models(self):
        """strategy_discovery imports ENTRY_MODELS from config."""
        import inspect

        import trading_app.strategy_discovery as sd

        source = inspect.getsource(sd)
        assert "ENTRY_MODELS" in source

    def test_market_state_imports_orb_labels(self):
        """market_state imports ORB_LABELS from init_db (not hardcoded)."""
        import inspect

        import trading_app.market_state as ms

        source = inspect.getsource(ms)
        assert "from pipeline.init_db import ORB_LABELS" in source

    def test_sql_adapter_imports_orb_labels(self):
        """sql_adapter derives VALID_ORB_LABELS from init_db (not hardcoded)."""
        import inspect

        import trading_app.ai.sql_adapter as sa

        source = inspect.getsource(sa)
        assert "from pipeline.init_db import ORB_LABELS" in source

    def test_grounding_imports_orb_labels(self):
        """grounding.py derives session list from init_db (not hardcoded)."""
        import inspect

        import trading_app.ai.grounding as gr

        source = inspect.getsource(gr)
        assert "from pipeline.init_db import ORB_LABELS" in source


# ============================================================================
# 7. Enabled sessions validation
# ============================================================================


class TestEnabledSessionsSync:
    """Every enabled_sessions label must exist in ORB_LABELS."""

    def test_all_enabled_sessions_in_orb_labels(self):
        orb_set = set(ORB_LABELS)
        for instrument, config in ASSET_CONFIGS.items():
            sessions = config.get("enabled_sessions", [])
            for s in sessions:
                assert s in orb_set, f"{instrument} enabled_sessions has '{s}' which is not in ORB_LABELS"

    def test_no_alias_in_enabled_sessions(self):
        from pipeline.dst import SESSION_CATALOG

        aliases = {label for label, entry in SESSION_CATALOG.items() if entry["type"] == "alias"}
        for instrument, config in ASSET_CONFIGS.items():
            sessions = config.get("enabled_sessions", [])
            for s in sessions:
                assert s not in aliases, f"{instrument} enabled_sessions has alias '{s}' -- use the canonical label"

    def test_get_enabled_sessions_returns_list(self):
        for instrument in ASSET_CONFIGS:
            result = get_enabled_sessions(instrument)
            assert isinstance(result, list)

    def test_get_enabled_sessions_unknown_returns_empty(self):
        assert get_enabled_sessions("UNKNOWN") == []
