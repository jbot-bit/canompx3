"""Tests for trading_app.ai.claude_client — canonical Anthropic client module.

Stage 1 of claude-api-modernization. See docs/runtime/stages/claude-api-modernization.md.
"""

import os
from unittest.mock import patch

import pytest


# Retired/deprecated model IDs from shared/models.md (cached 2026-04-15).
# Any constant matching these indicates the module is stale.
RETIRED_MODEL_IDS = {
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-sonnet-20240229",
    "claude-2.1",
    "claude-2.0",
}
DEPRECATED_MODEL_IDS = {
    "claude-3-haiku-20240307",  # retires 2026-04-19
}
STALE_MODEL_IDS = RETIRED_MODEL_IDS | DEPRECATED_MODEL_IDS | {
    # Pre-4.6-era models still alive but superseded — project must not pin these
    "claude-sonnet-4-20250514",  # Sonnet 4.0
    "claude-sonnet-4-5-20250929",  # Sonnet 4.5
    "claude-opus-4-20250514",  # Opus 4.0
    "claude-opus-4-1-20250805",  # Opus 4.1
}


class TestModelConstants:
    def test_structured_model_exported(self):
        from trading_app.ai.claude_client import CLAUDE_STRUCTURED_MODEL

        assert isinstance(CLAUDE_STRUCTURED_MODEL, str)
        assert CLAUDE_STRUCTURED_MODEL.startswith("claude-")

    def test_reasoning_model_exported(self):
        from trading_app.ai.claude_client import CLAUDE_REASONING_MODEL

        assert isinstance(CLAUDE_REASONING_MODEL, str)
        assert CLAUDE_REASONING_MODEL.startswith("claude-")

    def test_structured_model_not_stale(self):
        from trading_app.ai.claude_client import CLAUDE_STRUCTURED_MODEL

        assert CLAUDE_STRUCTURED_MODEL not in STALE_MODEL_IDS, (
            f"CLAUDE_STRUCTURED_MODEL is stale: {CLAUDE_STRUCTURED_MODEL}"
        )

    def test_reasoning_model_not_stale(self):
        from trading_app.ai.claude_client import CLAUDE_REASONING_MODEL

        assert CLAUDE_REASONING_MODEL not in STALE_MODEL_IDS, (
            f"CLAUDE_REASONING_MODEL is stale: {CLAUDE_REASONING_MODEL}"
        )

    def test_structured_model_is_sonnet_46(self):
        """Pin to Sonnet 4.6 for structured-output / deterministic extraction.

        Rationale: keeps `temperature=0.0` semantics available via adaptive thinking;
        cheaper than Opus for high-volume Pass-1 intent extraction.
        """
        from trading_app.ai.claude_client import CLAUDE_STRUCTURED_MODEL

        assert CLAUDE_STRUCTURED_MODEL == "claude-sonnet-4-6"

    def test_reasoning_model_is_opus_47(self):
        """Pin to Opus 4.7 for interpretation / coaching / research reasoning.

        Rationale: strongest reasoning model, adaptive thinking, 1M context at
        standard pricing (no long-context premium). Required for research-grade
        interpretation of quant results.
        """
        from trading_app.ai.claude_client import CLAUDE_REASONING_MODEL

        assert CLAUDE_REASONING_MODEL == "claude-opus-4-7"


class TestGetClient:
    def test_get_client_requires_api_key(self):
        """Without ANTHROPIC_API_KEY, raise ValueError with actionable message."""
        from trading_app.ai.claude_client import get_client

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                get_client()

    def test_get_client_returns_anthropic_client(self):
        """With ANTHROPIC_API_KEY set, return a configured anthropic.Anthropic."""
        import anthropic

        from trading_app.ai.claude_client import get_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-fake"}, clear=True):
            client = get_client()

        assert isinstance(client, anthropic.Anthropic)

    def test_get_client_accepts_explicit_key(self):
        """Explicit api_key arg overrides env var (useful for tests / multi-tenant)."""
        import anthropic

        from trading_app.ai.claude_client import get_client

        with patch.dict(os.environ, {}, clear=True):
            client = get_client(api_key="sk-explicit-fake")

        assert isinstance(client, anthropic.Anthropic)
