"""Generated OpenRouter research packets built from canonical repo surfaces."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from context.registry import resolve_from_text, resolve_task
from pipeline.paths import GOLD_DB_PATH
from pipeline.system_brief import build_system_brief
from scripts.tools.context_views import VIEW_BUILDERS, build_view
from trading_app.ai.corpus import CORPUS_FILES, get_corpus_file_paths
from trading_app.ai.provider_registry import (
    AIProfile,
    assert_openrouter_research_profile,
    get_openrouter_request_defaults,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_CONTEXT_PATH = Path("docs/ai-context/LOCAL_MODEL_CONTEXT.md")
CHATGPT_BUNDLE_INDEX_PATH = Path("chatgpt_bundle/00_INDEX.md")
_REFERENCE_PATTERN = re.compile(
    r"(docs/institutional/literature/[A-Za-z0-9_./-]+\.md|resources/[A-Za-z0-9_ ./()-]+\.pdf)"
)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _read_repo_text(relative_path: str | Path, root: Path) -> str:
    path = root / Path(relative_path)
    return path.read_text(encoding="utf-8", errors="replace")


def _resolve_route(task_text: str):
    route, _candidates = resolve_from_text(task_text)
    if route is None:
        return resolve_task("system_orientation")
    return route


def _corpus_inventory() -> dict[str, dict[str, str]]:
    inventory: dict[str, dict[str, str]] = {}
    for key, info in CORPUS_FILES.items():
        inventory[key] = {
            "path": info["path"],
            "priority": info["priority"],
            "description": info["description"],
        }
    return inventory


def _collect_literature_metadata(root: Path, source_paths: list[str]) -> list[dict[str, Any]]:
    refs: list[str] = []
    for relative_path in source_paths:
        path = root / relative_path
        if not path.exists() or path.suffix.lower() != ".md":
            continue
        refs.extend(_REFERENCE_PATTERN.findall(path.read_text(encoding="utf-8", errors="replace")))

    literature_refs = [ref for ref in _dedupe(refs) if ref.startswith("docs/institutional/literature/")]
    resource_refs = [ref for ref in _dedupe(refs) if ref.startswith("resources/")]

    payload: list[dict[str, Any]] = []
    for relative_path in literature_refs:
        path = root / relative_path
        title = path.stem
        source = None
        excerpt_lines: list[str] = []
        if path.exists():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[:20]:
                if line.startswith("# "):
                    title = line[2:].strip()
                if line.startswith("**Source:**"):
                    source = line.split(":", 1)[1].strip().strip("`")
                if len(excerpt_lines) < 3 and line.strip():
                    excerpt_lines.append(line.strip())
        payload.append(
            {
                "path": relative_path,
                "title": title,
                "source_pdf": source,
                "excerpt_head": excerpt_lines,
            }
        )

    for relative_path in resource_refs:
        payload.append(
            {
                "path": relative_path,
                "title": Path(relative_path).name,
                "source_pdf": relative_path,
                "excerpt_head": [],
            }
        )
    return payload


def _build_packet_contract(profile: AIProfile) -> dict[str, Any]:
    return {
        "mode": "research_planning_read_only",
        "runtime_class": profile.runtime_class,
        "host_tool_loop_enabled": bool(profile.host_tools),
        "mutation_allowed": profile.mutation_allowed,
        "live_control_allowed": profile.live_control_allowed,
        "allowed_host_tools": list(profile.host_tools),
        "disallowed_tools": [
            "shell_mutation",
            "file_writes",
            "raw_sql_writes",
            "lane_pause_unpause",
            "broker_actions",
            "autonomous_code_mutation",
        ],
        "insufficient_evidence_rule": "If information is not found in retrieved files or database outputs, say INSUFFICIENT EVIDENCE.",
        "citation_rule": "Ground all claims in canonical repo paths or local literature extracts. Do not invent thresholds or stale counts.",
    }


def build_research_packet(
    *,
    task_text: str,
    profile_id: str,
    root: Path = PROJECT_ROOT,
    db_path: Path = GOLD_DB_PATH,
) -> dict[str, Any]:
    profile = assert_openrouter_research_profile(profile_id)
    route = _resolve_route(task_text)
    system_brief = build_system_brief(
        root,
        task_text=task_text,
        briefing_level="read_only",
        context_name="codex-wsl",
        active_tool="ai_research_packet",
        active_mode="read-only",
        db_path=db_path,
    )

    routed_live_views = [view.id for view in route.live_views]
    generated_view_names = _dedupe(
        list(profile.context_views) + [view_id for view_id in routed_live_views if view_id in VIEW_BUILDERS]
    )
    context_views = {view_name: build_view(view_name, root, db_path) for view_name in generated_view_names}

    read_set = _dedupe(
        list(route.doctrine_files)
        + list(route.canonical_files)
        + list(system_brief.get("doctrine_chain", []))
        + list(system_brief.get("canonical_owners", []))
        + [LOCAL_MODEL_CONTEXT_PATH.as_posix(), CHATGPT_BUNDLE_INDEX_PATH.as_posix()]
    )

    literature_refs = _collect_literature_metadata(root, read_set)
    local_model_context = _read_repo_text(LOCAL_MODEL_CONTEXT_PATH, root)

    packet: dict[str, Any] = {
        "packet_kind": "ai_research_packet",
        "generated_at": datetime.now(UTC).isoformat(),
        "task": {
            "text": task_text,
            "route_id": route.manifest.id,
            "title": route.manifest.title,
            "verification_profile": route.verification.id,
            "verification_steps": [step.id for step in route.verification_steps],
        },
        "profile": profile.to_metadata(),
        "packet_contract": _build_packet_contract(profile),
        "system_prompt_seed": {
            "path": LOCAL_MODEL_CONTEXT_PATH.as_posix(),
            "content": local_model_context,
        },
        "system_brief": system_brief,
        "context_route": {
            "doctrine_chain": list(route.doctrine_files),
            "canonical_owners": list(route.canonical_files),
            "required_live_views": routed_live_views,
            "generated_context_views": generated_view_names,
            "expansion_triggers": list(route.expansion_triggers or route.briefing_contract.expansion_triggers),
        },
        "required_reads": read_set,
        "context_views": context_views,
        "grounding": {
            "corpus_inventory": _corpus_inventory(),
            "corpus_paths": get_corpus_file_paths(),
            "bundle_index_path": CHATGPT_BUNDLE_INDEX_PATH.as_posix(),
            "local_literature_refs": literature_refs,
            "repo_truth_protocol": {
                "canonical_active_work_truth": "docs/runtime/action-queue.yaml",
                "rendered_baton_only": "HANDOFF.md",
                "read_only_db_surface": "trading_app/mcp_server.py",
            },
        },
        "openrouter_request_defaults": get_openrouter_request_defaults(profile_id),
    }
    return packet


def format_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Research Packet",
        "",
        f"- Generated: `{packet['generated_at']}`",
        f"- Task: {packet['task']['text']}",
        f"- Route: `{packet['task']['route_id']}`",
        f"- Profile: `{packet['profile']['profile_id']}`",
        f"- Provider: `{packet['profile']['provider']}`",
        f"- Model configured: `{packet['profile']['model_configured']}`",
        "",
        "## Contract",
        "",
        f"- Mode: `{packet['packet_contract']['mode']}`",
        f"- Runtime class: `{packet['packet_contract']['runtime_class']}`",
        f"- Host tool loop enabled: `{packet['packet_contract']['host_tool_loop_enabled']}`",
        f"- Mutation allowed: `{packet['packet_contract']['mutation_allowed']}`",
        f"- Live control allowed: `{packet['packet_contract']['live_control_allowed']}`",
        f"- Allowed host tools: {', '.join(f'`{tool}`' for tool in packet['packet_contract']['allowed_host_tools']) or '`none`'}",
        "",
        "## Required Reads",
        "",
    ]
    lines.extend(f"- `{path}`" for path in packet["required_reads"])
    lines.extend(
        [
            "",
            "## Context Views",
            "",
        ]
    )
    for view_name in packet["context_views"]:
        lines.append(f"- `{view_name}`")
    lines.extend(
        [
            "",
            "## Local Literature",
            "",
        ]
    )
    for item in packet["grounding"]["local_literature_refs"]:
        lines.append(f"- `{item['path']}`")
    return "\n".join(lines)


def build_openrouter_request(packet: dict[str, Any]) -> dict[str, Any]:
    profile = packet["profile"]
    if profile["provider"] != "openrouter":
        raise ValueError("Packet is not configured for OpenRouter")
    if not profile["model"]:
        raise ValueError("OpenRouter packet has no configured model")

    user_payload = {
        "task": packet["task"],
        "packet_contract": packet["packet_contract"],
        "system_brief": packet["system_brief"],
        "context_route": packet["context_route"],
        "required_reads": packet["required_reads"],
        "context_views": packet["context_views"],
        "grounding": packet["grounding"],
    }

    request: dict[str, Any] = {
        "model": profile["model"],
        "messages": [
            {
                "role": "system",
                "content": packet["system_prompt_seed"]["content"],
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, indent=2, sort_keys=True),
            },
        ],
    }
    request.update(packet["openrouter_request_defaults"])
    return request
