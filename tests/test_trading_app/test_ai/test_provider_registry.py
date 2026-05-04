"""Tests for trading_app.ai.provider_registry."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trading_app.ai.provider_registry import (
    assert_openrouter_research_profile,
    get_openrouter_request_defaults,
    get_profile,
    list_openrouter_research_profiles,
    list_profiles,
)


class TestProviderRegistry:
    def test_lists_expected_profiles(self) -> None:
        profiles = set(list_profiles())
        assert "claude_structured" in profiles
        assert "claude_reasoning" in profiles
        assert "deepseek_planning" in profiles
        assert "deepseek_research_long_context" in profiles
        assert "deepseek_structured_extraction" in profiles

    def test_openrouter_profile_list_is_narrow(self) -> None:
        assert list_openrouter_research_profiles() == [
            "deepseek_planning",
            "deepseek_research_long_context",
            "deepseek_structured_extraction",
        ]

    def test_claude_profile_keeps_canonical_model(self) -> None:
        profile = get_profile("claude_structured")
        assert profile.provider == "anthropic"
        assert profile.model == "claude-sonnet-4-6"
        assert profile.api_key_env == "ANTHROPIC_API_KEY"

    def test_deepseek_planning_defaults_are_conservative(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
            },
            clear=False,
        ):
            profile = assert_openrouter_research_profile("deepseek_planning")
        assert profile.provider == "openrouter"
        assert profile.base_url == "https://openrouter.ai/api/v1"
        assert profile.mutation_allowed is False
        assert profile.live_control_allowed is False
        assert profile.router is not None
        assert profile.router.allow_fallbacks is False
        assert profile.router.require_parameters is True
        assert profile.router.data_collection == "deny"
        assert profile.host_tools == ("get_context_view", "get_canonical_context", "query_trading_db")

    def test_missing_env_fails_closed(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="missing required env"):
                assert_openrouter_research_profile("deepseek_planning")

    def test_env_overrides_model_and_provider_order_only(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
                "CANOMPX3_AI_DEEPSEEK_PLANNING_PROVIDER_ORDER": "deepseek,fireworks",
                "CANOMPX3_AI_DEEPSEEK_PLANNING_ALLOW_FALLBACKS": "true",
            },
            clear=False,
        ):
            profile = assert_openrouter_research_profile("deepseek_planning")
        assert profile.model == "deepseek/deepseek-v4-pro"
        assert profile.router is not None
        assert profile.router.order == ("deepseek", "fireworks")
        assert profile.router.allow_fallbacks is False

    def test_openrouter_request_defaults_emit_provider_contract(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "CANOMPX3_AI_DEEPSEEK_STRUCTURED_EXTRACTION_MODEL": "deepseek/deepseek-v4-flash",
            },
            clear=False,
        ):
            payload = get_openrouter_request_defaults("deepseek_structured_extraction")
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["require_parameters"] is True
        assert payload["provider"]["data_collection"] == "deny"

    def test_claude_profile_rejected_by_openrouter_assertion(self) -> None:
        with pytest.raises(ValueError, match="OpenRouter research profile"):
            assert_openrouter_research_profile("claude_reasoning")
