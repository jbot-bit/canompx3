"""Tests for trading_app.ai.query_agent.

Stage 3 of claude-api-modernization: migrated to canonical `claude_client`,
`messages.parse()` for Pass 1 intent extraction, adaptive thinking for Pass 2
interpretation, prompt caching on the grounding prompt.
"""

import importlib.util
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_app.ai.query_agent import (
    QueryIntentSchema,
    QueryResult,
    _generate_warnings,
)

ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None


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
        """QueryAgent delegates API-key check to claude_client.get_client()."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                from trading_app.ai.query_agent import QueryAgent

                QueryAgent(db_path="dummy.db", api_key=None)


def _make_mock_agent():
    """Construct QueryAgent with mocked I/O, no real API calls."""
    from trading_app.ai.query_agent import QueryAgent

    agent = QueryAgent.__new__(QueryAgent)
    agent.db_path = "dummy.db"
    agent.corpus = {
        "CONFIG": "test",
        "COST_MODEL": "test",
        "TRADING_RULES": "test",
        "TRADE_MANAGEMENT_RULES": "test",
        "RESEARCH_RULES": "test",
        "CLAUDE_MD": "test",
        "PRE_REGISTERED_CRITERIA": "test",
        "MECHANISM_PRIORS": "test",
    }
    agent.adapter = MagicMock()
    agent.schema_summary = "bars_1m: ts_utc, symbol"
    agent.db_stats = "bars_1m: 1000 rows"
    agent.client = MagicMock()
    return agent


class TestQueryAgentIntentExtraction:
    """Test Pass 1 (intent extraction) via mocked messages.parse()."""

    @pytest.fixture
    def mock_agent(self):
        return _make_mock_agent()

    def _mock_parse_response(self, *, template, parameters=None, explanation=""):
        """Build a response matching the shape returned by messages.parse()."""
        from trading_app.ai.query_agent import QueryIntentSchema, QueryParameters

        mock_response = MagicMock()
        mock_response.parsed_output = QueryIntentSchema(
            template=template,
            parameters=QueryParameters(**(parameters or {})),
            explanation=explanation,
        )
        return mock_response

    def test_extract_intent_valid_schema(self, mock_agent):
        mock_agent.client.messages.parse.return_value = self._mock_parse_response(
            template="strategy_lookup",
            parameters={"orb_label": "CME_REOPEN"},
            explanation="lookup",
        )
        intent = mock_agent._extract_intent("show CME_REOPEN strategies")
        assert intent is not None
        assert intent.template.value == "strategy_lookup"
        assert intent.parameters["orb_label"] == "CME_REOPEN"

    def test_extract_intent_null_template_returns_none(self, mock_agent):
        mock_agent.client.messages.parse.return_value = self._mock_parse_response(
            template=None, explanation="cannot answer"
        )
        assert mock_agent._extract_intent("what is the meaning of life?") is None

    def test_extract_intent_uses_structured_model(self, mock_agent):
        """Pass 1 pins to Sonnet 4.6 (CLAUDE_STRUCTURED_MODEL)."""
        from trading_app.ai.claude_client import CLAUDE_STRUCTURED_MODEL

        mock_agent.client.messages.parse.return_value = self._mock_parse_response(template="table_counts")
        mock_agent._extract_intent("how many rows?")

        call_kwargs = mock_agent.client.messages.parse.call_args.kwargs
        assert call_kwargs["model"] == CLAUDE_STRUCTURED_MODEL

    def test_extract_intent_applies_cache_control(self, mock_agent):
        """Grounding system prompt is a stable, large prefix — must be cached.

        `messages.parse()` rejects top-level `cache_control` kwarg (strict SDK
        signature on anthropic 0.96.0). Caching must be expressed via a
        TextBlockParam on the `system` list.
        """
        mock_agent.client.messages.parse.return_value = self._mock_parse_response(template="table_counts")
        mock_agent._extract_intent("how many rows?")

        call_kwargs = mock_agent.client.messages.parse.call_args.kwargs

        # cache_control must NOT be a top-level kwarg (would raise TypeError
        # against the real SDK; the MagicMock previously hid this).
        assert "cache_control" not in call_kwargs, (
            "cache_control as top-level kwarg is invalid on messages.parse() — "
            "the real SDK raises TypeError. Use a system TextBlockParam instead."
        )

        # system must be a list of blocks with cache_control on the grounding block.
        assert isinstance(call_kwargs["system"], list)
        assert len(call_kwargs["system"]) >= 1
        grounding_block = call_kwargs["system"][0]
        assert grounding_block["type"] == "text"
        assert grounding_block["cache_control"] == {"type": "ephemeral"}

    def test_extract_intent_passes_no_temperature(self, mock_agent):
        """`temperature` is removed from Opus 4.7; Sonnet 4.6 is forward-compatible.

        We never pass temperature — structured outputs give us determinism via
        schema validation rather than sampling knobs.
        """
        mock_agent.client.messages.parse.return_value = self._mock_parse_response(template="table_counts")
        mock_agent._extract_intent("how many rows?")

        call_kwargs = mock_agent.client.messages.parse.call_args.kwargs
        assert "temperature" not in call_kwargs

    def test_extract_intent_invalid_template_returns_none(self, mock_agent):
        """If Claude returns a template not in QueryTemplate enum, fail-soft None."""
        mock_agent.client.messages.parse.return_value = self._mock_parse_response(template="nonexistent_template_name")
        assert mock_agent._extract_intent("garbled") is None


class TestSDKSurfaceGuards:
    """Regression guards against SDK-surface bugs that MagicMock tests hide.

    MagicMock agent.client.messages.parse accepts any kwargs, so tests green-
    light calls that would TypeError against the real SDK. These tests exercise
    the REAL anthropic.resources.messages.Messages signature via inspect — no
    network, no mock — to validate our call shape.
    """

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_messages_parse_exists_with_required_params(self):
        """Every kwarg _extract_intent passes must exist on the real SDK."""
        import inspect

        from anthropic.resources.messages import Messages

        sig = inspect.signature(Messages.parse)
        required_for_our_call = {
            "model",
            "max_tokens",
            "system",
            "messages",
            "output_format",
        }
        missing = required_for_our_call - set(sig.parameters)
        assert not missing, f"SDK surface changed: parse() missing {missing}"

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_messages_parse_rejects_cache_control_kwarg(self):
        """Regression guard: cache_control is NOT a top-level kwarg on parse().

        If this ever flips (SDK adds the param), we can simplify the system-
        block workaround in _extract_intent. Until then, caching MUST be on a
        TextBlockParam inside the system list.
        """
        import inspect

        from anthropic.resources.messages import Messages

        sig = inspect.signature(Messages.parse)
        assert "cache_control" not in sig.parameters, (
            "SDK now accepts cache_control on parse() — can simplify _extract_intent to pass it at top level."
        )

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_extract_intent_call_shape_binds_to_real_sdk(self):
        """Validate the exact kwargs our code passes against the REAL SDK sig.

        Uses inspect.signature.bind() — zero network, zero mock. Catches what
        MagicMock hides: passing an unexpected kwarg raises TypeError here,
        same as the real call would.

        If _extract_intent is ever updated to pass a new/removed kwarg, update
        the dict below so this guard stays in sync with production code.
        """
        import inspect

        from anthropic.resources.messages import Messages

        from trading_app.ai.claude_client import CLAUDE_STRUCTURED_MODEL

        our_kwargs = dict(
            model=CLAUDE_STRUCTURED_MODEL,
            max_tokens=500,
            system=[
                {
                    "type": "text",
                    "text": "sentinel-grounding",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": "sentinel-question"}],
            output_format=QueryIntentSchema,
        )

        sig = inspect.signature(Messages.parse)
        try:
            sig.bind(None, **our_kwargs)  # None = self, ignored for kwarg validation
        except TypeError as exc:
            pytest.fail(f"_extract_intent call shape invalid against anthropic SDK: {exc}")

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed")
    def test_messages_create_thinking_adaptive_shape(self):
        """Validate Pass-2 (_interpret_results) kwargs against real SDK sig.

        Symmetric guard to test_extract_intent_call_shape_binds_to_real_sdk —
        covers client.messages.create() in _interpret_results (query_agent.py
        lines ~196-202) so a future kwarg change on Pass-2 can't regress the
        same MagicMock-hides-TypeError failure mode.
        """
        import inspect

        from anthropic.resources.messages import Messages
        from anthropic.types.thinking_config_adaptive_param import (
            ThinkingConfigAdaptiveParam,
        )

        from trading_app.ai.claude_client import CLAUDE_REASONING_MODEL

        # The exact kwargs _interpret_results passes to messages.create().
        # `thinking_param` is pulled out so its type stays a plain dict[str,str]
        # for the TypedDict-conformance assertion below (pyright can't narrow
        # heterogenous dict values after packing into a single kwargs dict).
        thinking_param: dict[str, str] = {"type": "adaptive"}
        our_kwargs = dict(
            model=CLAUDE_REASONING_MODEL,
            max_tokens=2000,
            thinking=thinking_param,
            messages=[{"role": "user", "content": "sentinel-interpret"}],
        )

        sig = inspect.signature(Messages.create)
        try:
            sig.bind(None, **our_kwargs)
        except TypeError as exc:
            pytest.fail(f"_interpret_results call shape invalid against anthropic SDK: {exc}")

        # Confirm the thinking shape satisfies the TypedDict contract —
        # the 'type' literal must be 'adaptive', which is a Required field.
        required_keys = set(ThinkingConfigAdaptiveParam.__required_keys__)
        missing = required_keys - set(thinking_param.keys())
        assert not missing, f"thinking dict missing required keys: {missing}"


class TestQueryAgentInterpretation:
    """Test Pass 2 (result interpretation) via mocked messages.create()."""

    @pytest.fixture
    def mock_agent(self):
        return _make_mock_agent()

    def _mock_create_response(self, text: str):
        mock_response = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = text
        mock_response.content = [block]
        return mock_response

    def test_interpret_uses_reasoning_model(self, mock_agent):
        """Pass 2 pins to Opus 4.7 (CLAUDE_REASONING_MODEL)."""
        from trading_app.ai.claude_client import CLAUDE_REASONING_MODEL

        mock_agent.client.messages.create.return_value = self._mock_create_response("The data shows 3 CORE strategies.")
        df = pd.DataFrame({"x": [1, 2, 3]})
        mock_agent._interpret_results("what does it say?", df)

        call_kwargs = mock_agent.client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == CLAUDE_REASONING_MODEL

    def test_interpret_uses_adaptive_thinking(self, mock_agent):
        """Interpretation is reasoning-heavy; adaptive thinking is required."""
        mock_agent.client.messages.create.return_value = self._mock_create_response("...")
        df = pd.DataFrame({"x": [1, 2, 3]})
        mock_agent._interpret_results("q", df)

        call_kwargs = mock_agent.client.messages.create.call_args.kwargs
        assert call_kwargs.get("thinking") == {"type": "adaptive"}

    def test_interpret_passes_no_temperature(self, mock_agent):
        """Opus 4.7 rejects `temperature`."""
        mock_agent.client.messages.create.return_value = self._mock_create_response("...")
        df = pd.DataFrame({"x": [1, 2, 3]})
        mock_agent._interpret_results("q", df)

        call_kwargs = mock_agent.client.messages.create.call_args.kwargs
        assert "temperature" not in call_kwargs

    def test_interpret_extracts_text_from_content_blocks(self, mock_agent):
        """With adaptive thinking, content may be [ThinkingBlock, ..., TextBlock].

        The method must pick the text block, not rely on content[0].
        """
        mock_response = MagicMock()
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "reasoning..."
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The answer is 42."
        mock_response.content = [thinking_block, text_block]
        mock_agent.client.messages.create.return_value = mock_response

        df = pd.DataFrame({"x": [1]})
        result = mock_agent._interpret_results("q", df)
        assert result == "The answer is 42."

    def test_interpret_empty_df_short_circuits(self, mock_agent):
        result = mock_agent._interpret_results("q", pd.DataFrame())
        assert "No results" in result
        mock_agent.client.messages.create.assert_not_called()


class TestFullQueryFlow:
    """End-to-end flow through parse → execute → create."""

    @pytest.fixture
    def mock_agent(self):
        return _make_mock_agent()

    def test_full_flow_happy_path(self, mock_agent):
        from trading_app.ai.query_agent import QueryIntentSchema, QueryParameters

        # Pass 1: intent via messages.parse
        parse_response = MagicMock()
        parse_response.parsed_output = QueryIntentSchema(
            template="validated_summary",
            parameters=QueryParameters(),
            explanation="summary",
        )
        mock_agent.client.messages.parse.return_value = parse_response

        # Pass 2: interpretation via messages.create
        create_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "There are 312 validated strategies across 4 sessions."
        create_response.content = [text_block]
        mock_agent.client.messages.create.return_value = create_response

        mock_agent.adapter.execute.return_value = pd.DataFrame(
            {
                "orb_label": ["CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS", "US_DATA_830"],
                "count": [134, 75, 85, 18],
            }
        )

        result = mock_agent.query("How many validated strategies per session?")
        assert result.intent is not None
        assert result.data is not None
        assert len(result.data) == 4
        assert "312" in result.explanation
