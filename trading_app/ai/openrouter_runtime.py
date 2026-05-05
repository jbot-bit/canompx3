"""Bounded OpenRouter runtime for read-only research tasks."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from pipeline.paths import GOLD_DB_PATH
from pipeline.system_context import build_system_context
from scripts.tools.context_views import build_view
from trading_app.ai.corpus import load_corpus
from trading_app.ai.provider_registry import AIProfile, assert_openrouter_research_profile
from trading_app.ai.research_packet import build_openrouter_request, build_research_packet
from trading_app.ai.schema_registry import get_schema
from trading_app.mcp_server import _query_trading_db

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class OpenRouterModelMetadata:
    model_id: str
    supported_parameters: tuple[str, ...]
    context_length: int | None = None


def _auth_headers(profile: AIProfile) -> dict[str, str]:
    resolved_profile = profile.assert_ready()
    missing = resolved_profile.missing_env()
    if missing:
        raise ValueError("missing required env: " + ", ".join(missing))
    resolved_key = os.environ.get(resolved_profile.api_key_env)
    if not resolved_key:
        raise ValueError(f"{resolved_profile.api_key_env} required for OpenRouter runtime")
    return {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }


def fetch_model_metadata(
    model_id: str,
    *,
    http_client: httpx.Client | None = None,
    url: str = OPENROUTER_MODELS_URL,
) -> OpenRouterModelMetadata:
    own_client = http_client is None
    client = http_client or httpx.Client(timeout=15.0)
    try:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
    finally:
        if own_client:
            client.close()

    for item in payload.get("data", []):
        if item.get("id") == model_id:
            return OpenRouterModelMetadata(
                model_id=model_id,
                supported_parameters=tuple(item.get("supported_parameters", [])),
                context_length=item.get("context_length"),
            )
    raise LookupError(f"OpenRouter model metadata not found for {model_id}")


def validate_runtime_environment(root: Path, runtime_class: str) -> None:
    if runtime_class != "read_only_tool_loop":
        return
    snapshot = build_system_context(
        root,
        context_name="codex-wsl",
        active_tool="ai_openrouter_runtime",
        active_mode="read-only",
    )
    if not snapshot.git.in_linked_worktree:
        raise ValueError("read-only tool-loop runtime requires a managed worktree checkout")


def validate_profile_capabilities(
    profile: AIProfile,
    metadata: OpenRouterModelMetadata,
    *,
    schema_name: str | None = None,
) -> None:
    supported = set(metadata.supported_parameters)
    required = set(profile.required_parameters)
    if schema_name is not None:
        required.update({"response_format", "structured_outputs"})
    missing = sorted(required.difference(supported))
    if missing:
        raise ValueError(
            f"Configured model {metadata.model_id} does not support required parameters: {', '.join(missing)}"
        )


def _tool_specs(profile: AIProfile) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if "get_context_view" in profile.host_tools:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "get_context_view",
                    "description": "Load a canonical generated context view.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "view": {
                                "type": "string",
                                "enum": ["research", "recent_performance", "trading", "verification"],
                            }
                        },
                        "required": ["view"],
                    },
                },
            }
        )
    if "get_canonical_context" in profile.host_tools:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "get_canonical_context",
                    "description": "Load canonical grounding documents for AI context.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {},
                    },
                },
            }
        )
    if "query_trading_db" in profile.host_tools:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "query_trading_db",
                    "description": "Run a pre-approved read-only trading query.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "template": {"type": "string"},
                            "orb_label": {"type": "string"},
                            "entry_model": {"type": "string"},
                            "filter_type": {"type": "string"},
                            "min_sample_size": {"type": "integer"},
                            "instrument": {"type": "string"},
                            "limit": {"type": "integer"},
                            "rr_target": {"type": "number"},
                            "confirm_bars": {"type": "integer"},
                        },
                        "required": ["template"],
                    },
                },
            }
        )
    return specs


def _execute_host_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    root: Path,
    db_path: Path,
) -> dict[str, Any]:
    if name == "get_context_view":
        return build_view(arguments["view"], root, db_path)
    if name == "get_canonical_context":
        return load_corpus()
    if name == "query_trading_db":
        return _query_trading_db(**arguments)
    raise ValueError(f"Unknown host tool: {name}")


def _extend_request_for_runtime(
    packet: dict[str, Any],
    profile: AIProfile,
    *,
    schema_name: str | None,
) -> dict[str, Any]:
    request = build_openrouter_request(packet)
    if schema_name is not None:
        request["response_format"] = get_schema(schema_name).response_format()
    if profile.host_tools:
        request["tools"] = _tool_specs(profile)
        request["tool_choice"] = "auto"
    return request


def run_openrouter_task(
    *,
    task_text: str,
    profile_id: str,
    root: Path = PROJECT_ROOT,
    db_path: Path = GOLD_DB_PATH,
    schema_name: str | None = None,
    max_turns: int = 4,
    execute: bool = False,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    profile = assert_openrouter_research_profile(profile_id)
    validate_runtime_environment(root, profile.runtime_class)
    packet = build_research_packet(task_text=task_text, profile_id=profile_id, root=root, db_path=db_path)
    metadata = fetch_model_metadata(profile.model or "", http_client=http_client)
    validate_profile_capabilities(profile, metadata, schema_name=schema_name)
    request = _extend_request_for_runtime(packet, profile, schema_name=schema_name)

    envelope: dict[str, Any] = {
        "status": "dry_run",
        "task": task_text,
        "profile": profile.profile_id,
        "model": profile.model,
        "runtime_class": profile.runtime_class,
        "evidence_refs": packet["required_reads"],
        "tool_history": [],
        "request": request,
        "capabilities": {
            "supported_parameters": list(metadata.supported_parameters),
            "context_length": metadata.context_length,
        },
    }
    if not execute:
        return envelope

    own_client = http_client is None
    client = http_client or httpx.Client(timeout=60.0)
    url = f"{profile.base_url}/chat/completions"
    messages = list(request["messages"])
    static_payload = {key: value for key, value in request.items() if key != "messages"}
    started = time.perf_counter()

    try:
        for turn in range(1, max_turns + 1):
            response = client.post(url, headers=_auth_headers(profile), json={**static_payload, "messages": messages})
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]["message"]
            content = choice.get("content", "")
            tool_calls = choice.get("tool_calls") or []
            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            if not tool_calls:
                envelope.update(
                    {
                        "status": "completed",
                        "result": content,
                        "usage": data.get("usage", {}),
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "turns": turn,
                    }
                )
                return envelope

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"].get("arguments") or "{}")
                tool_result = _execute_host_tool(name, arguments, root=root, db_path=db_path)
                envelope["tool_history"].append({"name": name, "arguments": arguments})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result, sort_keys=True),
                    }
                )

        envelope.update(
            {
                "status": "max_turns_exceeded",
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "turns": max_turns,
            }
        )
        return envelope
    finally:
        if own_client:
            client.close()
