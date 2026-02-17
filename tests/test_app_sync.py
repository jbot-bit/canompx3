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

import sys
import json
import re
from pathlib import Path
from datetime import date

import pytest
import duckdb

from pipeline.init_db import ORB_LABELS, DAILY_FEATURES_SCHEMA
from trading_app.config import (
    ALL_FILTERS,
    ENTRY_MODELS,
    CORE_MIN_SAMPLES,
    REGIME_MIN_SAMPLES,
    classify_strategy,
    NoFilter,
    OrbSizeFilter,
    VolumeFilter,
    DirectionFilter,
    get_filters_for_grid,
    DIR_LONG,
    DIR_SHORT,
    MGC_ORB_SIZE_FILTERS,
    MGC_VOLUME_FILTERS,
    StrategyFilter,
)
from trading_app.outcome_builder import RR_TARGETS, CONFIRM_BARS_OPTIONS
from trading_app.strategy_discovery import make_strategy_id
from pipeline.asset_configs import ASSET_CONFIGS, get_enabled_sessions
from trading_app.db_manager import (
    init_trading_app_schema,
    verify_trading_app_schema,
)

# ============================================================================
# 1. ORB_LABELS consistency
# ============================================================================

class TestOrbLabelsSync:
    """ORB_LABELS must be consistent across all modules."""

    EXPECTED_ORB_LABELS = [
        "0900", "1000", "1100", "1130", "1800", "2300", "0030",
        "CME_OPEN", "US_EQUITY_OPEN", "US_DATA_OPEN", "LONDON_OPEN",
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
        orb_col_pattern = re.compile(r'orb_([A-Za-z0-9_]+?)_(?:high|low|size|break_dir|break_ts|outcome|mae_r|mfe_r|double_break)\b')
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

    # L-filters removed from grid (negative ExpR, 0/1024 validated). Classes retained for reference.
    # G2/G3 removed (99%+ pass rate on most sessions = cosmetic, not real filtering)
    EXPECTED_FILTER_KEYS = {
        "NO_FILTER",
        "ORB_G4", "ORB_G5", "ORB_G6", "ORB_G8",
        "VOL_RV12_N20",
    }

    def test_expected_keys(self):
        """ALL_FILTERS has exactly the expected keys."""
        assert set(ALL_FILTERS.keys()) == self.EXPECTED_FILTER_KEYS

    def test_filter_type_matches_key(self):
        """Each filter's filter_type field matches its key in ALL_FILTERS."""
        for key, filt in ALL_FILTERS.items():
            assert filt.filter_type == key, (
                f"Key '{key}' but filter_type='{filt.filter_type}'"
            )

    def test_all_are_strategy_filters(self):
        """Every value in ALL_FILTERS is a StrategyFilter subclass."""
        for key, filt in ALL_FILTERS.items():
            assert isinstance(filt, StrategyFilter), f"{key} is not StrategyFilter"

    def test_filter_json_roundtrip(self):
        """Every filter serializes to valid JSON containing its filter_type."""
        for key, filt in ALL_FILTERS.items():
            j = filt.to_json()
            parsed = json.loads(j)
            assert parsed["filter_type"] == key, (
                f"JSON filter_type mismatch: key={key}, json={parsed['filter_type']}"
            )

    def test_no_filter_matches_everything(self):
        """NoFilter.matches_row always returns True."""
        nf = ALL_FILTERS["NO_FILTER"]
        assert nf.matches_row({}, "0900") is True
        assert nf.matches_row({"anything": 42}, "2300") is True

    def test_filters_are_frozen(self):
        """All filters are frozen (hashable, immutable)."""
        for key, filt in ALL_FILTERS.items():
            hash(filt)

    def test_size_filters_have_thresholds(self):
        """Every ORB size filter has at least min_size or max_size set."""
        for key, filt in ALL_FILTERS.items():
            if key == "NO_FILTER" or isinstance(filt, VolumeFilter):
                continue
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
        row = {"orb_1000_break_dir": "long"}
        assert DIR_LONG.matches_row(row, "1000") is True
        assert DIR_SHORT.matches_row(row, "1000") is False

    def test_matches_short(self):
        row = {"orb_1000_break_dir": "short"}
        assert DIR_SHORT.matches_row(row, "1000") is True
        assert DIR_LONG.matches_row(row, "1000") is False

    def test_missing_dir_fails_closed(self):
        assert DIR_LONG.matches_row({}, "1000") is False
        assert DIR_SHORT.matches_row({}, "1000") is False


# ============================================================================
# 2c. get_filters_for_grid session-aware dispatch
# ============================================================================

class TestGetFiltersForGrid:
    """get_filters_for_grid must return correct filter sets per instrument+session."""

    def test_mes_1000_includes_band_and_dir(self):
        filters = get_filters_for_grid("MES", "1000")
        assert "ORB_G4_L12" in filters
        assert "ORB_G5_L12" in filters
        assert "DIR_LONG" in filters

    def test_mgc_0900_excludes_extras(self):
        filters = get_filters_for_grid("MGC", "0900")
        assert "ORB_G4_L12" not in filters
        assert "DIR_LONG" not in filters
        assert filters == ALL_FILTERS

    def test_mgc_1000_has_dir_no_band(self):
        filters = get_filters_for_grid("MGC", "1000")
        assert "DIR_LONG" in filters
        assert "ORB_G4_L12" not in filters

    def test_base_filters_always_present(self):
        for inst in ("MGC", "MES", "MNQ"):
            for sess in ("0900", "1000"):
                filters = get_filters_for_grid(inst, sess)
                assert "NO_FILTER" in filters
                assert "ORB_G4" in filters


# ============================================================================
# 3. RR_TARGETS, CONFIRM_BARS_OPTIONS, ENTRY_MODELS consistency
# ============================================================================

class TestGridParamsSync:
    """Grid parameters must be consistent and valid."""

    def test_rr_targets_sorted(self):
        assert RR_TARGETS == sorted(RR_TARGETS)

    def test_rr_targets_positive(self):
        assert all(rr > 0 for rr in RR_TARGETS)

    def test_rr_targets_no_duplicates(self):
        assert len(RR_TARGETS) == len(set(RR_TARGETS))

    def test_confirm_bars_sorted(self):
        assert CONFIRM_BARS_OPTIONS == sorted(CONFIRM_BARS_OPTIONS)

    def test_confirm_bars_positive_ints(self):
        assert all(isinstance(cb, int) and cb > 0 for cb in CONFIRM_BARS_OPTIONS)

    def test_confirm_bars_no_duplicates(self):
        assert len(CONFIRM_BARS_OPTIONS) == len(set(CONFIRM_BARS_OPTIONS))

    def test_grid_size(self):
        """Total grid size matches expected formula (E3 uses CB1 only).

        11 ORBs (7 fixed + 4 dynamic) x 6 RRs x 5 CBs x 6 filters x 2 EMs
        E1: 11 x 6 x 5 x 6 = 1980
        E3: 11 x 6 x 1 x 6 = 396  (E3 always CB1)
        Total: 2376
        """
        e1 = len(ORB_LABELS) * len(RR_TARGETS) * len(CONFIRM_BARS_OPTIONS) * len(ALL_FILTERS)
        e3 = len(ORB_LABELS) * len(RR_TARGETS) * 1 * len(ALL_FILTERS)
        expected = e1 + e3
        assert expected == 2376

class TestEntryModelsSync:
    """ENTRY_MODELS must be consistent."""

    def test_entry_models_exact(self):
        assert ENTRY_MODELS == ["E1", "E3"]

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
        sid = make_strategy_id("MGC", "0900", "E1", 2.0, 1, "NO_FILTER")
        assert sid == "MGC_0900_E1_RR2.0_CB1_NO_FILTER"

    def test_parseable(self):
        """Strategy ID can be parsed back to components."""
        sid = make_strategy_id("MGC", "0900", "E3", 2.0, 1, "ORB_L4")
        parts = sid.split("_", 3)  # MGC, 0900, E3, RR2.0_CB1_ORB_L4
        assert parts[0] == "MGC"
        assert parts[1] == "0900"
        assert parts[2] == "E3"

    def test_all_grid_ids_unique(self):
        """Every combination in the full grid produces a unique ID (E3 CB1 only)."""
        ids = set()
        for orb in ORB_LABELS:
            for em in ENTRY_MODELS:
                for rr in RR_TARGETS:
                    for cb in CONFIRM_BARS_OPTIONS:
                        if em == "E3" and cb > 1:
                            continue
                        for fk in ALL_FILTERS:
                            sid = make_strategy_id("MGC", orb, em, rr, cb, fk)
                            assert sid not in ids, f"Duplicate ID: {sid}"
                            ids.add(sid)
        assert len(ids) == 2376

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
        cols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='experimental_strategies'"
        ).fetchall()}
        con.close()

        required = {
            "strategy_id", "instrument", "orb_label", "orb_minutes",
            "rr_target", "confirm_bars", "entry_model", "filter_type",
            "filter_params", "sample_size", "win_rate", "avg_win_r",
            "avg_loss_r", "expectancy_r", "sharpe_ratio", "max_drawdown_r",
            "median_risk_points", "avg_risk_points",
            "yearly_results",
            "entry_signals", "scratch_count", "early_exit_count",
            "trade_day_hash", "is_canonical", "canonical_strategy_id",
            "validation_status", "validation_notes",
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
        cols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='validated_setups'"
        ).fetchall()}
        con.close()

        required = {
            "strategy_id", "promoted_from", "instrument", "orb_label",
            "orb_minutes", "rr_target", "confirm_bars", "entry_model",
            "filter_type", "filter_params", "sample_size", "win_rate",
            "expectancy_r", "years_tested", "all_years_positive",
            "stress_test_passed", "sharpe_ratio", "max_drawdown_r",
            "yearly_results", "status",
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
        cols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='orb_outcomes'"
        ).fetchall()}
        con.close()

        required = {
            "trading_day", "symbol", "orb_label", "orb_minutes",
            "rr_target", "confirm_bars", "entry_model", "entry_ts",
            "entry_price", "stop_price", "target_price", "outcome",
            "exit_ts", "exit_price", "pnl_r", "mae_r", "mfe_r",
        }
        missing = required - cols
        assert not missing, f"Missing columns in orb_outcomes: {missing}"

# ============================================================================
# 6. Cross-module import sync
# ============================================================================

class TestImportSync:
    """Verify that modules import the same constants."""

    def test_outcome_builder_uses_init_db_orb_labels(self):
        """outcome_builder imports ORB_LABELS from init_db (not hardcoded)."""
        import trading_app.outcome_builder as ob
        import inspect
        source = inspect.getsource(ob)
        assert 'from pipeline.init_db import ORB_LABELS' in source

    def test_strategy_discovery_uses_shared_constants(self):
        """strategy_discovery imports RR_TARGETS/CONFIRM_BARS from outcome_builder."""
        import inspect
        import trading_app.strategy_discovery as sd
        source = inspect.getsource(sd)
        assert 'from trading_app.outcome_builder import RR_TARGETS' in source
        assert 'CONFIRM_BARS_OPTIONS' in source

    def test_outcome_builder_imports_entry_models(self):
        """outcome_builder imports ENTRY_MODELS from config."""
        import inspect
        import trading_app.outcome_builder as ob
        source = inspect.getsource(ob)
        assert 'from trading_app.config import ENTRY_MODELS' in source

    def test_strategy_discovery_imports_entry_models(self):
        """strategy_discovery imports ENTRY_MODELS from config."""
        import inspect
        import trading_app.strategy_discovery as sd
        source = inspect.getsource(sd)
        assert 'ENTRY_MODELS' in source

    def test_market_state_imports_orb_labels(self):
        """market_state imports ORB_LABELS from init_db (not hardcoded)."""
        import inspect
        import trading_app.market_state as ms
        source = inspect.getsource(ms)
        assert 'from pipeline.init_db import ORB_LABELS' in source

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
                assert s in orb_set, (
                    f"{instrument} enabled_sessions has '{s}' which is not in ORB_LABELS"
                )

    def test_no_alias_in_enabled_sessions(self):
        from pipeline.dst import SESSION_CATALOG
        aliases = {
            label for label, entry in SESSION_CATALOG.items()
            if entry["type"] == "alias"
        }
        for instrument, config in ASSET_CONFIGS.items():
            sessions = config.get("enabled_sessions", [])
            for s in sessions:
                assert s not in aliases, (
                    f"{instrument} enabled_sessions has alias '{s}' -- use the canonical label"
                )

    def test_get_enabled_sessions_returns_list(self):
        for instrument in ASSET_CONFIGS:
            result = get_enabled_sessions(instrument)
            assert isinstance(result, list)

    def test_get_enabled_sessions_unknown_returns_empty(self):
        assert get_enabled_sessions("UNKNOWN") == []
