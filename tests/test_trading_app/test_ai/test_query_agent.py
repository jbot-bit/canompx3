"""Tests for trading_app.ai.query_agent."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_app.ai.query_agent import (
    QueryResult,
    _generate_warnings,
)
from trading_app.config import CORE_MIN_SAMPLES as _CORE_MIN, REGIME_MIN_SAMPLES as _REGIME_MIN


class TestGenerateWarnings:
    def test_no_filter_warning(self):
        df = pd.DataFrame({"filter_type": ["NO_FILTER"], "sample_size": [200]})
        warnings = _generate_warnings(df)
        assert any("NO_FILTER" in w for w in warnings)

    def test_l_filter_warning(self):
        df = pd.DataFrame({"filter_type": ["ORB_L4"], "sample_size": [200]})
        warnings = _generate_warnings(df)
        assert any("L-filter" in w for w in warnings)

    def test_small_sample_invalid(self):
        df = pd.DataFrame({"filter_type": ["ORB_G4"], "sample_size": [15]})
        warnings = _generate_warnings(df)
        assert any("INVALID" in w for w in warnings)

    def test_regime_sample_warning(self):
        df = pd.DataFrame({"filter_type": ["ORB_G4"], "sample_size": [50]})
        warnings = _generate_warnings(df)
        assert any("REGIME" in w for w in warnings)

    def test_core_sample_no_warning(self):
        df = pd.DataFrame({"filter_type": ["ORB_G4"], "sample_size": [150]})
        warnings = _generate_warnings(df)
        # No sample-size warnings for CORE strategies
        assert not any("INVALID" in w for w in warnings)
        assert not any("REGIME" in w for w in warnings)

    def test_empty_df(self):
        assert _generate_warnings(pd.DataFrame()) == []

    def test_none_df(self):
        assert _generate_warnings(None) == []

    def test_no_filter_type_column(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        warnings = _generate_warnings(df)
        assert not any("NO_FILTER" in w for w in warnings)


class TestQueryResult:
    def test_defaults(self):
        qr = QueryResult(query="test question")
        assert qr.query == "test question"
        assert qr.intent is None
        assert qr.data is None
        assert qr.explanation == ""
        assert qr.warnings == []
        assert qr.grounding_refs == []


class TestQueryAgentInit:
    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                from trading_app.ai.query_agent import QueryAgent
                QueryAgent(db_path="dummy.db", api_key=None)


class TestQueryAgentIntentExtraction:
    """Test intent extraction with mocked Anthropic client."""

    @pytest.fixture
    def mock_agent(self):
        """Create agent with mocked dependencies (no real API calls)."""
        from trading_app.ai.query_agent import QueryAgent

        agent = QueryAgent.__new__(QueryAgent)
        agent.db_path = "dummy.db"
        agent.api_key = "test-key"
        agent.corpus = {
            "CONFIG": "test", "COST_MODEL": "test",
            "CANONICAL_LOGIC": "test", "TRADE_MANAGEMENT_RULES": "test",
        }
        agent.adapter = MagicMock()
        agent.schema_summary = "bars_1m: ts_utc, symbol"
        agent.db_stats = "bars_1m: 1000 rows"
        agent.client = MagicMock()

        return agent

    def test_extract_intent_valid_json(self, mock_agent):
        """Mock Claude returning valid JSON intent."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"template": "strategy_lookup", "parameters": {"orb_label": "0900"}, "explanation": "test"}'
        )]
        mock_agent.client.messages.create.return_value = mock_response

        intent = mock_agent._extract_intent("show 0900 strategies")
        assert intent is not None
        assert intent.template.value == "strategy_lookup"
        assert intent.parameters["orb_label"] == "0900"

    def test_extract_intent_null_template(self, mock_agent):
        """Mock Claude saying no template fits."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"template": null, "parameters": {}, "explanation": "cannot answer"}'
        )]
        mock_agent.client.messages.create.return_value = mock_response

        intent = mock_agent._extract_intent("what is the meaning of life?")
        assert intent is None

    def test_extract_intent_code_block(self, mock_agent):
        """Mock Claude wrapping JSON in code block."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='```json\n{"template": "table_counts", "parameters": {}, "explanation": "row counts"}\n```'
        )]
        mock_agent.client.messages.create.return_value = mock_response

        intent = mock_agent._extract_intent("how many rows?")
        assert intent is not None
        assert intent.template.value == "table_counts"

    def test_full_query_flow(self, mock_agent):
        """Test full query with mocked API and adapter."""
        # Mock intent extraction
        intent_response = MagicMock()
        intent_response.content = [MagicMock(
            text='{"template": "validated_summary", "parameters": {}, "explanation": "summary"}'
        )]

        # Mock interpretation
        interp_response = MagicMock()
        interp_response.content = [MagicMock(
            text="There are 312 validated strategies across 4 sessions."
        )]

        mock_agent.client.messages.create.side_effect = [intent_response, interp_response]

        # Mock adapter execution
        mock_agent.adapter.execute.return_value = pd.DataFrame({
            "orb_label": ["0900", "1000", "1800", "2300"],
            "count": [134, 75, 85, 18],
        })

        result = mock_agent.query("How many validated strategies per session?")
        assert result.intent is not None
        assert result.data is not None
        assert len(result.data) == 4
        assert "312" in result.explanation
