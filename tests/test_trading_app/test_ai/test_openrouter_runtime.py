"""Tests for trading_app.ai.openrouter_runtime."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from trading_app.ai.openrouter_runtime import (
    OpenRouterModelMetadata,
    run_openrouter_task,
    validate_profile_capabilities,
    validate_runtime_environment,
)
from trading_app.ai.provider_registry import assert_openrouter_research_profile


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, posts):
        self._posts = list(posts)

    def get(self, _url):
        return _FakeResponse(
            {
                "data": [
                    {
                        "id": "deepseek/deepseek-v4-pro",
                        "supported_parameters": ["reasoning", "tools", "response_format", "structured_outputs"],
                        "context_length": 1048576,
                    },
                    {
                        "id": "deepseek/deepseek-v4-flash",
                        "supported_parameters": ["response_format", "structured_outputs"],
                        "context_length": 1048576,
                    },
                ]
            }
        )

    def post(self, _url, headers=None, json=None):
        return _FakeResponse(self._posts.pop(0))

    def close(self) -> None:
        return None


def _minimal_packet(profile_id: str, model: str) -> dict:
    return {
        "task": {"text": "Plan repo research", "route_id": "research_discovery", "title": "Research Discovery"},
        "profile": {"provider": "openrouter", "profile_id": profile_id, "model": model},
        "packet_contract": {"mode": "research_planning_read_only"},
        "system_brief": {},
        "context_route": {},
        "required_reads": ["RESEARCH_RULES.md"],
        "context_views": {},
        "grounding": {"local_literature_refs": []},
        "system_prompt_seed": {"content": "grounding"},
        "openrouter_request_defaults": {"provider": {"allow_fallbacks": False}},
    }


class TestOpenRouterRuntime:
    def test_validate_profile_capabilities_rejects_missing_tools(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "sk-or-test",
                "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
            },
            clear=False,
        ):
            profile = assert_openrouter_research_profile("deepseek_planning")
        metadata = OpenRouterModelMetadata(model_id="deepseek/deepseek-v4-pro", supported_parameters=("reasoning",))
        with pytest.raises(ValueError, match="required parameters: tools"):
            validate_profile_capabilities(profile, metadata)

    def test_tool_loop_environment_requires_worktree(self) -> None:
        with patch(
            "trading_app.ai.openrouter_runtime.build_system_context",
            return_value=SimpleNamespace(git=SimpleNamespace(in_linked_worktree=False)),
        ):
            with pytest.raises(ValueError, match="managed worktree"):
                validate_runtime_environment(Path.cwd(), "read_only_tool_loop")

    def test_dry_run_builds_tool_request(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENROUTER_API_KEY": "sk-or-test",
                    "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
                },
                clear=False,
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_system_context",
                return_value=SimpleNamespace(git=SimpleNamespace(in_linked_worktree=True)),
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_research_packet",
                return_value=_minimal_packet("deepseek_planning", "deepseek/deepseek-v4-pro"),
            ),
        ):
            result = run_openrouter_task(
                task_text="Plan repo research",
                profile_id="deepseek_planning",
                root=Path.cwd(),
                http_client=_FakeClient(posts=[]),
            )
        assert result["status"] == "dry_run"
        assert result["request"]["tool_choice"] == "auto"
        assert any(tool["function"]["name"] == "get_context_view" for tool in result["request"]["tools"])

    def test_structured_extraction_adds_response_format(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENROUTER_API_KEY": "sk-or-test",
                    "CANOMPX3_AI_DEEPSEEK_STRUCTURED_EXTRACTION_MODEL": "deepseek/deepseek-v4-flash",
                },
                clear=False,
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_research_packet",
                return_value=_minimal_packet("deepseek_structured_extraction", "deepseek/deepseek-v4-flash"),
            ),
        ):
            result = run_openrouter_task(
                task_text="Summarize repo research",
                profile_id="deepseek_structured_extraction",
                schema_name="grounded_research_assessment",
                root=Path.cwd(),
                http_client=_FakeClient(posts=[]),
            )
        assert result["request"]["response_format"]["type"] == "json_schema"

    def test_execute_runs_tool_loop_and_returns_result(self) -> None:
        posts = [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "tool-1",
                                    "type": "function",
                                    "function": {"name": "get_canonical_context", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [{"message": {"content": "Grounded answer", "tool_calls": []}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ]
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENROUTER_API_KEY": "sk-or-test",
                    "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
                },
                clear=False,
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_system_context",
                return_value=SimpleNamespace(git=SimpleNamespace(in_linked_worktree=True)),
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_research_packet",
                return_value=_minimal_packet("deepseek_planning", "deepseek/deepseek-v4-pro"),
            ),
            patch(
                "trading_app.ai.openrouter_runtime.load_corpus",
                return_value={"RESEARCH_RULES": "grounded"},
            ),
        ):
            result = run_openrouter_task(
                task_text="Plan repo research",
                profile_id="deepseek_planning",
                root=Path.cwd(),
                http_client=_FakeClient(posts=posts),
                execute=True,
            )
        assert result["status"] == "completed"
        assert result["result"] == "Grounded answer"
        assert result["tool_history"][0]["name"] == "get_canonical_context"
