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

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.init_db import ORB_LABELS, DAILY_FEATURES_SCHEMA
from trading_app.config import (
    ALL_FILTERS,
    ENTRY_MODELS,
    NoFilter,
    OrbSizeFilter,
    MGC_ORB_SIZE_FILTERS,
    StrategyFilter,
)
from trading_app.outcome_builder import RR_TARGETS, CONFIRM_BARS_OPTIONS
from trading_app.strategy_discovery import make_strategy_id
from trading_app.db_manager import (
    init_trading_app_schema,
    verify_trading_app_schema,
)


# ============================================================================
# 1. ORB_LABELS consistency
# ============================================================================

class TestOrbLabelsSync:
    """ORB_LABELS must be consistent across all modules."""

    EXPECTED_ORB_LABELS = ["0900", "1000", "1100", "1800", "2300", "0030"]

    def test_orb_labels_exact(self):
        """ORB_LABELS matches expected values exactly."""
        assert ORB_LABELS == self.EXPECTED_ORB_LABELS

    def test_orb_labels_no_duplicates(self):
        """No duplicate ORB labels."""
        assert len(ORB_LABELS) == len(set(ORB_LABELS))

    def test_daily_features_columns_match_orb_labels(self):
        """daily_features DDL has columns for every ORB label and no extras."""
        orb_col_pattern = re.compile(r'orb_(\d{4})_\w+')
        found_labels = set()
        for match in orb_col_pattern.finditer(DAILY_FEATURES_SCHEMA):
            found_labels.add(match.group(1))

        assert found_labels == set(ORB_LABELS), (
            f"DDL ORB columns {sorted(found_labels)} != ORB_LABELS {ORB_LABELS}"
        )


# ============================================================================
# 2. ALL_FILTERS registry consistency
# ============================================================================

class TestAllFiltersSync:
    """ALL_FILTERS keys must match filter_type inside each filter."""

    EXPECTED_FILTER_KEYS = {
        "NO_FILTER",
        "ORB_L2", "ORB_L3", "ORB_L4", "ORB_L6", "ORB_L8",
        "ORB_G2", "ORB_G3", "ORB_G4", "ORB_G5", "ORB_G6", "ORB_G8",
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
            if key == "NO_FILTER":
                continue
            assert isinstance(filt, OrbSizeFilter), f"{key} should be OrbSizeFilter"
            assert filt.min_size is not None or filt.max_size is not None, (
                f"{key} has neither min_size nor max_size"
            )


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
        """Total grid size matches expected formula."""
        expected = len(ORB_LABELS) * len(RR_TARGETS) * len(CONFIRM_BARS_OPTIONS) * len(ALL_FILTERS) * len(ENTRY_MODELS)
        # 6 ORBs * 6 RRs * 5 CBs * 12 filters * 3 models = 6480
        assert expected == 6 * 6 * 5 * 12 * 3


class TestEntryModelsSync:
    """ENTRY_MODELS must be consistent."""

    def test_entry_models_exact(self):
        assert ENTRY_MODELS == ["E1", "E2", "E3"]

    def test_entry_models_no_duplicates(self):
        assert len(ENTRY_MODELS) == len(set(ENTRY_MODELS))

    def test_entry_models_are_strings(self):
        assert all(isinstance(em, str) for em in ENTRY_MODELS)


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
        sid = make_strategy_id("MGC", "0900", "E2", 2.0, 1, "ORB_L4")
        parts = sid.split("_", 3)  # MGC, 0900, E2, RR2.0_CB1_ORB_L4
        assert parts[0] == "MGC"
        assert parts[1] == "0900"
        assert parts[2] == "E2"

    def test_all_grid_ids_unique(self):
        """Every combination in the full grid produces a unique ID."""
        ids = set()
        for orb in ORB_LABELS:
            for em in ENTRY_MODELS:
                for rr in RR_TARGETS:
                    for cb in CONFIRM_BARS_OPTIONS:
                        for fk in ALL_FILTERS:
                            sid = make_strategy_id("MGC", orb, em, rr, cb, fk)
                            assert sid not in ids, f"Duplicate ID: {sid}"
                            ids.add(sid)
        assert len(ids) == 6480


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
            "yearly_results", "validation_status", "validation_notes",
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
