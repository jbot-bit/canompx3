"""Tests for trading_app.ai.sql_adapter."""

import pytest
import pandas as pd

from trading_app.ai.sql_adapter import (
    QueryTemplate,
    QueryIntent,
    SQLAdapter,
    MAX_RESULT_ROWS,
    VALID_ORB_LABELS,
    VALID_ENTRY_MODELS,
    VALID_RR_TARGETS,
    VALID_CONFIRM_BARS,
    _validate_orb_label,
    _validate_entry_model,
    _validate_filter_type,
    _validate_rr_target,
    _validate_confirm_bars,
    _orb_size_filter_sql,
    _compute_group_stats,
    _DST_SESSION_MAP,
    _TEMPLATES,
)


class TestQueryTemplate:
    """Test template enum."""

    def test_all_templates_have_sql(self):
        for t in QueryTemplate:
            assert t in _TEMPLATES, f"Template {t} missing SQL"

    def test_no_write_keywords_in_templates(self):
        """CRITICAL: No INSERT/UPDATE/DELETE/DROP in any template."""
        write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP"]
        for t, sql in _TEMPLATES.items():
            sql_upper = sql.upper()
            for kw in write_keywords:
                assert kw not in sql_upper, (
                    f"Template {t.value} contains forbidden keyword: {kw}"
                )

    def test_all_templates_are_select(self):
        """All templates must be SELECT queries."""
        for t, sql in _TEMPLATES.items():
            # TABLE_COUNTS uses a different pattern
            if t == QueryTemplate.TABLE_COUNTS:
                assert "COUNT(*)" in sql
                continue
            assert "SELECT" in sql.upper(), f"Template {t.value} is not a SELECT"

    def test_template_count(self):
        assert len(QueryTemplate) == 18


class TestParameterValidation:
    """Test parameter validation functions."""

    def test_valid_orb_labels(self):
        for label in VALID_ORB_LABELS:
            assert _validate_orb_label(label) == label

    def test_invalid_orb_label_raises(self):
        with pytest.raises(ValueError, match="Invalid ORB label"):
            _validate_orb_label("0800")

    def test_valid_entry_models(self):
        for em in VALID_ENTRY_MODELS:
            assert _validate_entry_model(em) == em

    def test_invalid_entry_model_raises(self):
        with pytest.raises(ValueError, match="Invalid entry model"):
            _validate_entry_model("E4")

    def test_valid_filter_type(self):
        assert _validate_filter_type("ORB_G4") == "ORB_G4"
        assert _validate_filter_type("NO_FILTER") == "NO_FILTER"
        assert _validate_filter_type("VOL_RV12_N20") == "VOL_RV12_N20"

    def test_invalid_filter_type_raises(self):
        with pytest.raises(ValueError, match="Invalid filter_type"):
            _validate_filter_type("HACKED_FILTER")

    def test_sql_injection_in_orb_label_blocked(self):
        """Injection via orb_label is blocked by allowlist."""
        with pytest.raises(ValueError):
            _validate_orb_label("0900; DROP TABLE --")

    def test_sql_injection_in_entry_model_blocked(self):
        with pytest.raises(ValueError):
            _validate_entry_model("E1' OR '1'='1")

    def test_sql_injection_in_filter_type_blocked(self):
        with pytest.raises(ValueError):
            _validate_filter_type("'; DROP TABLE validated_setups; --")


class TestSQLAdapterBuildQuery:
    """Test query building without DB access."""

    def test_build_query_strategy_lookup(self):
        adapter = SQLAdapter.__new__(SQLAdapter)
        adapter.db_path = "dummy.db"
        sql, params = adapter._build_query(
            QueryTemplate.STRATEGY_LOOKUP,
            {"orb_label": "0900", "limit": 10},
        )
        assert "SELECT" in sql
        assert "validated_setups" in sql
        assert params == ["0900", 10]

    def test_build_query_with_multiple_params(self):
        adapter = SQLAdapter.__new__(SQLAdapter)
        adapter.db_path = "dummy.db"
        sql, params = adapter._build_query(
            QueryTemplate.STRATEGY_LOOKUP,
            {"orb_label": "1800", "entry_model": "E3", "filter_type": "ORB_G4", "limit": 20},
        )
        assert "0900" not in str(params)
        assert "1800" in params
        assert "E3" in params
        assert "ORB_G4" in params

    def test_limit_capped_at_max(self):
        adapter = SQLAdapter.__new__(SQLAdapter)
        adapter.db_path = "dummy.db"
        _, params = adapter._build_query(
            QueryTemplate.STRATEGY_LOOKUP,
            {"limit": 9999},
        )
        assert params[-1] == MAX_RESULT_ROWS

    def test_default_limit(self):
        adapter = SQLAdapter.__new__(SQLAdapter)
        adapter.db_path = "dummy.db"
        _, params = adapter._build_query(
            QueryTemplate.STRATEGY_LOOKUP,
            {},
        )
        assert params[-1] == 50


class TestAvailableTemplates:
    def test_returns_all(self):
        templates = SQLAdapter.available_templates()
        assert len(templates) == len(QueryTemplate)
        for t in templates:
            assert "template" in t
            assert "description" in t


class TestNewParameterValidation:
    """Test rr_target and confirm_bars validation."""

    def test_valid_rr_targets(self):
        for rr in VALID_RR_TARGETS:
            assert _validate_rr_target(rr) == rr

    def test_rr_target_from_string(self):
        assert _validate_rr_target("2.0") == 2.0

    def test_invalid_rr_target_raises(self):
        with pytest.raises(ValueError, match="Invalid rr_target"):
            _validate_rr_target(0.5)

    def test_valid_confirm_bars(self):
        for cb in VALID_CONFIRM_BARS:
            assert _validate_confirm_bars(cb) == cb

    def test_confirm_bars_from_string(self):
        assert _validate_confirm_bars("2") == 2

    def test_invalid_confirm_bars_raises(self):
        with pytest.raises(ValueError, match="Invalid confirm_bars"):
            _validate_confirm_bars(5)


class TestOrbSizeFilterSQL:
    """Test ORB size filter SQL generation."""

    def test_no_filter_returns_none(self):
        assert _orb_size_filter_sql("NO_FILTER", "0900") is None

    def test_none_returns_none(self):
        assert _orb_size_filter_sql(None, "0900") is None

    def test_orb_g4(self):
        result = _orb_size_filter_sql("ORB_G4", "1000")
        assert result == "d.orb_1000_size >= 4"

    def test_orb_g6(self):
        result = _orb_size_filter_sql("ORB_G6", "0900")
        assert result == "d.orb_0900_size >= 6"

    def test_orb_l8(self):
        result = _orb_size_filter_sql("ORB_L8", "1000")
        assert result == "d.orb_1000_size < 8"

    def test_vol_filter_returns_none(self):
        """VOL_ filters aren't translatable to SQL — silently skipped."""
        assert _orb_size_filter_sql("VOL_RV12_N20", "0900") is None

    def test_invalid_prefix_raises(self):
        with pytest.raises(ValueError, match="Invalid filter_type"):
            _orb_size_filter_sql("HACKED", "0900")


class TestComputeGroupStats:
    """Test stats computation helper."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["pnl_r", "outcome"])
        stats = _compute_group_stats(df)
        assert stats["N"] == 0
        assert stats["win_rate"] is None

    def test_all_wins(self):
        df = pd.DataFrame({
            "pnl_r": [1.0, 1.0, 1.0],
            "outcome": ["win", "win", "win"],
        })
        stats = _compute_group_stats(df)
        assert stats["N"] == 3
        assert stats["win_rate"] == 100.0
        assert stats["avg_pnl_r"] == 1.0

    def test_mixed_outcomes(self):
        df = pd.DataFrame({
            "pnl_r": [2.0, -1.0, 2.0, -1.0],
            "outcome": ["win", "loss", "win", "loss"],
        })
        stats = _compute_group_stats(df)
        assert stats["N"] == 4
        assert stats["win_rate"] == 50.0
        assert stats["avg_pnl_r"] == 0.5
        assert stats["sharpe"] is not None

    def test_constant_pnl_sharpe_none(self):
        """Zero std dev → sharpe is None."""
        df = pd.DataFrame({
            "pnl_r": [1.0, 1.0, 1.0],
            "outcome": ["win", "win", "win"],
        })
        stats = _compute_group_stats(df)
        # std of constant series is 0 → sharpe None
        assert stats["sharpe"] is None


class TestDSTSessionMap:
    """Test DST session mapping."""

    def test_dst_sensitive_sessions(self):
        assert _DST_SESSION_MAP["0900"] == "us_dst"
        assert _DST_SESSION_MAP["0030"] == "us_dst"
        assert _DST_SESSION_MAP["2300"] == "us_dst"
        assert _DST_SESSION_MAP["1800"] == "uk_dst"

    def test_non_dst_sessions_absent(self):
        assert "1000" not in _DST_SESSION_MAP
        assert "1100" not in _DST_SESSION_MAP


class TestBuildOutcomesBase:
    """Test SAFE_JOIN query builder (no DB needed)."""

    def _make_adapter(self):
        adapter = SQLAdapter.__new__(SQLAdapter)
        adapter.db_path = "dummy.db"
        return adapter

    def test_basic_query(self):
        adapter = self._make_adapter()
        sql, bind = adapter._build_outcomes_base(
            {"instrument": "MGC", "orb_label": "0900"}
        )
        assert "orb_outcomes o" in sql
        assert "daily_features d" in sql
        assert "o.orb_minutes = d.orb_minutes" in sql
        assert bind == ["MGC", "0900"]

    def test_all_params(self):
        adapter = self._make_adapter()
        sql, bind = adapter._build_outcomes_base({
            "instrument": "MNQ", "orb_label": "1000",
            "entry_model": "E0", "rr_target": 2.0, "confirm_bars": 1,
            "filter_type": "ORB_G4",
        })
        assert "o.entry_model = ?" in sql
        assert "o.rr_target = ?" in sql
        assert "o.confirm_bars = ?" in sql
        assert "d.orb_1000_size >= 4" in sql
        assert bind == ["MNQ", "1000", "E0", 2.0, 1]

    def test_missing_orb_label_raises(self):
        adapter = self._make_adapter()
        with pytest.raises(ValueError, match="orb_label is required"):
            adapter._build_outcomes_base({"instrument": "MGC"})

    def test_extra_cols(self):
        adapter = self._make_adapter()
        sql, _ = adapter._build_outcomes_base(
            {"orb_label": "0900"}, extra_cols="o.entry_model"
        )
        assert "o.entry_model" in sql


class TestNewTemplatesSafeJoin:
    """Verify all new templates use the SAFE_JOIN pattern."""

    def test_outcomes_templates_have_safe_join(self):
        """All 5 new templates must join on trading_day + symbol + orb_minutes."""
        new_templates = [
            QueryTemplate.OUTCOMES_STATS,
            QueryTemplate.ENTRY_MODEL_COMPARE,
            QueryTemplate.DOW_BREAKDOWN,
            QueryTemplate.DST_SPLIT,
            QueryTemplate.FILTER_COMPARE,
        ]
        for t in new_templates:
            sql = _TEMPLATES[t]
            assert "o.trading_day = d.trading_day" in sql, f"{t.value}: missing trading_day join"
            assert "o.symbol = d.symbol" in sql, f"{t.value}: missing symbol join"
            assert "o.orb_minutes = d.orb_minutes" in sql, f"{t.value}: missing orb_minutes join"
