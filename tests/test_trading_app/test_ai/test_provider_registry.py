"""Tests for trading_app.ai.provider_registry."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from trading_app.ai.provider_registry import (
    PROFILE_REGISTRY,
    AIProfile,
    ProviderRouting,
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


class TestDeepseekCodingProfile:
    """Phase 1 of DeepSeek Coding Agent v4: profile is registered, fail-closed."""

    def test_deepseek_coding_profile_registered(self) -> None:
        assert "deepseek_coding" in PROFILE_REGISTRY
        profile = PROFILE_REGISTRY["deepseek_coding"]
        assert profile.profile_id == "deepseek_coding"
        assert profile.provider == "openrouter"

    def test_deepseek_coding_runtime_class_is_interactive_editor(self) -> None:
        profile = PROFILE_REGISTRY["deepseek_coding"]
        assert profile.runtime_class == "interactive_editor"

    def test_deepseek_coding_mutation_allowed_true(self) -> None:
        # Mutation authority is the whole point of the editing profile.
        profile = PROFILE_REGISTRY["deepseek_coding"]
        assert profile.mutation_allowed is True

    def test_deepseek_coding_router_is_fail_closed_zdr(self) -> None:
        profile = PROFILE_REGISTRY["deepseek_coding"]
        assert profile.router is not None
        assert profile.router.allow_fallbacks is False
        assert profile.router.zdr is True
        assert profile.router.data_collection == "deny"
        assert profile.router.require_parameters is True

    def test_deepseek_coding_assert_ready_fails_when_model_none(self) -> None:
        # Phase 1 ships model=None by design; Phase 2.5 picks the winner.
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
            profile = get_profile("deepseek_coding")
            errors = profile.validation_errors()
            assert any("model not configured" in e and "DEEPSEEK_CODING_MODEL" in e for e in errors), errors
            with pytest.raises(ValueError, match="model not configured"):
                profile.assert_ready()

    def test_deepseek_coding_assert_ready_succeeds_with_model_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "CANOMPX3_AI_DEEPSEEK_CODING_MODEL": "deepseek/deepseek-v3.2-exp",
            },
            clear=True,
        ):
            profile = get_profile("deepseek_coding")
            assert profile.validation_errors() == []
            assert profile.assert_ready().model == "deepseek/deepseek-v3.2-exp"

    def test_deepseek_coding_rejected_by_openrouter_research_assertion(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "CANOMPX3_AI_DEEPSEEK_CODING_MODEL": "deepseek/deepseek-v3.2-exp",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="not an OpenRouter research profile"):
                assert_openrouter_research_profile("deepseek_coding")

    def test_deepseek_coding_excluded_from_research_profile_list(self) -> None:
        # The narrow research-profile list must NOT include the coding profile.
        assert "deepseek_coding" not in list_openrouter_research_profiles()
        # ...but it IS in the full profile listing.
        assert "deepseek_coding" in list_profiles()

    def test_existing_research_profile_still_rejects_handset_mutation(self) -> None:
        """Regression guard for the validation_errors() runtime_class refactor.

        The refactor gates the mutation/live-control rejection on
        ``runtime_class != "interactive_editor"``. A research profile (which
        always has runtime_class != "interactive_editor") with
        ``mutation_allowed=True`` set by hand must STILL be rejected.
        Without this regression guard, the gate could silently widen.
        """
        base = PROFILE_REGISTRY["deepseek_planning"]
        bad = replace(base, mutation_allowed=True, live_control_allowed=True)
        errors = bad.validation_errors()
        assert any("mutation authority is not allowed" in e for e in errors), errors
        assert any("live-control authority is not allowed" in e for e in errors), errors

    def test_handset_interactive_editor_research_profile_allows_mutation(self) -> None:
        """Sister of the regression guard: the gate hinges on runtime_class.

        A profile with runtime_class="interactive_editor" must NOT raise the
        mutation/live-control rejections (the new escape hatch is exactly this).
        Provider/router rules still apply unchanged.
        """
        editor = AIProfile(
            profile_id="test_editor",
            provider="openrouter",
            use_case="test fixture",
            model="test/model-x",
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            runtime_class="interactive_editor",
            mutation_allowed=True,
            live_control_allowed=True,
            router=ProviderRouting(
                order=("deepseek",),
                allow_fallbacks=False,
                require_parameters=True,
                data_collection="deny",
                zdr=True,
            ),
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
            errors = editor.validation_errors()
        assert not any("mutation authority" in e for e in errors), errors
        assert not any("live-control authority" in e for e in errors), errors
