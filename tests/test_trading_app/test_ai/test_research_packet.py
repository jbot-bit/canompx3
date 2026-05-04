"""Tests for trading_app.ai.research_packet."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from context.registry import resolve_task
from trading_app.ai.research_packet import (
    build_openrouter_request,
    build_research_packet,
    format_packet_markdown,
)


class TestResearchPacket:
    def test_build_research_packet_uses_canonical_contract(self) -> None:
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
                "trading_app.ai.research_packet.build_view",
                side_effect=lambda view_name, _root, _db_path: {"view": view_name},
            ),
            patch(
                "trading_app.ai.research_packet.build_system_brief",
                return_value={
                    "doctrine_chain": ["RESEARCH_RULES.md"],
                    "canonical_owners": ["trading_app/holdout_policy.py"],
                },
            ),
            patch(
                "trading_app.ai.research_packet._resolve_route",
                return_value=resolve_task("research_discovery"),
            ),
        ):
            packet = build_research_packet(
                task_text="Plan a grounded DeepSeek research workflow for this repo",
                profile_id="deepseek_planning",
                root=Path.cwd(),
            )

        assert packet["packet_kind"] == "ai_research_packet"
        assert packet["profile"]["provider"] == "openrouter"
        assert packet["packet_contract"]["mode"] == "research_planning_read_only"
        assert packet["packet_contract"]["host_tool_loop_enabled"] is True
        assert "get_context_view" in packet["packet_contract"]["allowed_host_tools"]
        assert "docs/ai-context/LOCAL_MODEL_CONTEXT.md" in packet["required_reads"]
        assert "chatgpt_bundle/00_INDEX.md" in packet["required_reads"]
        assert "corpus_inventory" in packet["grounding"]
        assert "openrouter_request_defaults" in packet

    def test_packet_harvests_local_literature_refs(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENROUTER_API_KEY": "sk-or-test",
                    "CANOMPX3_AI_DEEPSEEK_RESEARCH_LONG_CONTEXT_MODEL": "deepseek/deepseek-v4-pro",
                },
                clear=False,
            ),
            patch(
                "trading_app.ai.research_packet.build_view",
                side_effect=lambda view_name, _root, _db_path: {"view": view_name},
            ),
            patch(
                "trading_app.ai.research_packet.build_system_brief",
                return_value={
                    "doctrine_chain": ["RESEARCH_RULES.md"],
                    "canonical_owners": ["trading_app/holdout_policy.py"],
                },
            ),
            patch(
                "trading_app.ai.research_packet._resolve_route",
                return_value=resolve_task("research_discovery"),
            ),
        ):
            packet = build_research_packet(
                task_text="Audit the repo's research methodology against local literature",
                profile_id="deepseek_research_long_context",
                root=Path.cwd(),
            )
        refs = {item["path"] for item in packet["grounding"]["local_literature_refs"]}
        assert "docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md" in refs
        assert "docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md" in refs

    def test_markdown_summary_mentions_profile_and_contract(self) -> None:
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
                "trading_app.ai.research_packet.build_view",
                side_effect=lambda view_name, _root, _db_path: {"view": view_name},
            ),
            patch(
                "trading_app.ai.research_packet.build_system_brief",
                return_value={
                    "doctrine_chain": ["RESEARCH_RULES.md"],
                    "canonical_owners": ["trading_app/holdout_policy.py"],
                },
            ),
            patch(
                "trading_app.ai.research_packet._resolve_route",
                return_value=resolve_task("research_discovery"),
            ),
        ):
            packet = build_research_packet(
                task_text="Plan DeepSeek usage for repo research",
                profile_id="deepseek_planning",
                root=Path.cwd(),
            )
        markdown = format_packet_markdown(packet)
        assert "# AI Research Packet" in markdown
        assert "`deepseek_planning`" in markdown
        assert "`read_only_tool_loop`" in markdown

    def test_openrouter_request_uses_packet_seed_and_context(self) -> None:
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
                "trading_app.ai.research_packet.build_view",
                side_effect=lambda view_name, _root, _db_path: {"view": view_name},
            ),
            patch(
                "trading_app.ai.research_packet.build_system_brief",
                return_value={
                    "doctrine_chain": ["RESEARCH_RULES.md"],
                    "canonical_owners": ["trading_app/holdout_policy.py"],
                },
            ),
            patch(
                "trading_app.ai.research_packet._resolve_route",
                return_value=resolve_task("research_discovery"),
            ),
        ):
            packet = build_research_packet(
                task_text="Plan DeepSeek usage for repo research",
                profile_id="deepseek_planning",
                root=Path.cwd(),
            )
        request = build_openrouter_request(packet)
        assert request["model"] == "deepseek/deepseek-v4-pro"
        assert request["provider"]["allow_fallbacks"] is False
        assert request["reasoning"]["effort"] == "high"
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][1]["role"] == "user"

    def test_missing_openrouter_env_blocks_packet_build(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="missing required env"):
                build_research_packet(
                    task_text="Plan DeepSeek usage for repo research",
                    profile_id="deepseek_planning",
                    root=Path.cwd(),
                )
