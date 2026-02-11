"""Tests for trading_app.ai.sql_adapter."""

import pytest

from trading_app.ai.sql_adapter import (
    QueryTemplate,
    QueryIntent,
    SQLAdapter,
    MAX_RESULT_ROWS,
    VALID_ORB_LABELS,
    VALID_ENTRY_MODELS,
    _validate_orb_label,
    _validate_entry_model,
    _validate_filter_type,
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
        assert len(QueryTemplate) == 13


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
