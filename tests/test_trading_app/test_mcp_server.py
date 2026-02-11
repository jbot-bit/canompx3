"""Tests for trading_app.mcp_server (no DB required)."""

import pytest
from unittest.mock import patch, MagicMock

import pandas as pd

from trading_app.mcp_server import (
    _list_available_queries,
    _query_trading_db,
    _generate_warnings,
    MAX_MCP_ROWS,
    _ALLOWED_PARAMS,
)


class TestListAvailableQueries:
    def test_returns_all_templates(self):
        result = _list_available_queries()
        assert len(result) == 12
        names = {t["template"] for t in result}
        assert "strategy_lookup" in names
        assert "table_counts" in names

    def test_each_has_description(self):
        for t in _list_available_queries():
            assert t["description"], f"Missing description for {t['template']}"


class TestQueryTradingDb:
    def test_invalid_template_returns_error(self):
        result = _query_trading_db(template="nonexistent")
        assert "error" in result
        assert "nonexistent" in result["error"]

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_valid_query_returns_data(self, mock_cls):
        mock_adapter = MagicMock()
        mock_adapter.execute.return_value = pd.DataFrame(
            {"orb_label": ["0900"], "expectancy_r": [0.15]}
        )
        mock_cls.return_value = mock_adapter

        result = _query_trading_db(template="validated_summary")
        assert result["row_count"] == 1
        assert result["columns"] == ["orb_label", "expectancy_r"]
        assert len(result["rows"]) == 1

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_adapter_exception_returns_error(self, mock_cls):
        mock_adapter = MagicMock()
        mock_adapter.execute.side_effect = ValueError("bad param")
        mock_cls.return_value = mock_adapter

        result = _query_trading_db(template="strategy_lookup", orb_label="0900")
        assert "error" in result
        assert "bad param" in result["error"]

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_params_forwarded(self, mock_cls):
        mock_adapter = MagicMock()
        mock_adapter.execute.return_value = pd.DataFrame()
        mock_cls.return_value = mock_adapter

        _query_trading_db(
            template="strategy_lookup",
            orb_label="1800",
            entry_model="E3",
            filter_type="ORB_G6",
            min_sample_size=50,
            limit=10,
        )

        call_args = mock_adapter.execute.call_args[0][0]
        assert call_args.template.value == "strategy_lookup"
        assert call_args.parameters["orb_label"] == "1800"
        assert call_args.parameters["entry_model"] == "E3"
        assert call_args.parameters["filter_type"] == "ORB_G6"
        assert call_args.parameters["min_sample_size"] == 50
        assert call_args.parameters["limit"] == 10


class TestGuardrails:
    """Guardrail enforcement tests."""

    def test_raw_sql_rejected(self):
        """No raw SQL -- only enum templates accepted."""
        result = _query_trading_db(template="SELECT * FROM bars_1m")
        assert "error" in result

    def test_sql_injection_template_rejected(self):
        result = _query_trading_db(template="strategy_lookup; DROP TABLE --")
        assert "error" in result

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_limit_capped_at_max_mcp_rows(self, mock_cls):
        """Limit > MAX_MCP_ROWS gets clamped server-side."""
        mock_adapter = MagicMock()
        mock_adapter.execute.return_value = pd.DataFrame()
        mock_cls.return_value = mock_adapter

        _query_trading_db(template="validated_summary", limit=999999)

        call_args = mock_adapter.execute.call_args[0][0]
        assert call_args.parameters["limit"] <= MAX_MCP_ROWS

    @patch("trading_app.mcp_server.SQLAdapter")
    def test_result_truncated_beyond_cap(self, mock_cls):
        """Even if adapter returns too many rows, MCP truncates."""
        mock_adapter = MagicMock()
        # Return more rows than the cap
        mock_adapter.execute.return_value = pd.DataFrame(
            {"x": range(MAX_MCP_ROWS + 100)}
        )
        mock_cls.return_value = mock_adapter

        result = _query_trading_db(template="validated_summary")
        assert result["row_count"] <= MAX_MCP_ROWS

    def test_allowed_params_whitelist(self):
        """Only known parameter keys are in the allowlist."""
        assert _ALLOWED_PARAMS == {"orb_label", "entry_model", "filter_type", "min_sample_size", "limit"}


class TestGenerateWarnings:
    def test_no_filter_warning(self):
        df = pd.DataFrame({"filter_type": ["NO_FILTER"], "sample_size": [200]})
        w = _generate_warnings(df)
        assert any("NO_FILTER" in x for x in w)

    def test_small_sample_warning(self):
        df = pd.DataFrame({"sample_size": [10, 50, 200]})
        w = _generate_warnings(df)
        assert any("INVALID" in x for x in w)

    def test_regime_sample_warning(self):
        df = pd.DataFrame({"sample_size": [50]})
        w = _generate_warnings(df)
        assert any("REGIME" in x for x in w)

    def test_empty_df_no_warnings(self):
        assert _generate_warnings(pd.DataFrame()) == []

    def test_none_no_warnings(self):
        assert _generate_warnings(None) == []
