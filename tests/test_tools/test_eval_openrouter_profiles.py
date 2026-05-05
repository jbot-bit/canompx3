"""Tests for scripts/tools/eval_openrouter_profiles.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools.eval_openrouter_profiles import (  # noqa: E402
    EVAL_TASKS,
    MUTATION_TOOL_FRAGMENTS,
    EvalTask,
    _offline_models_payload,
    _OfflineModelsClient,
    _packet_from_envelope,
    _score_cell,
    main,
    render_json,
    render_markdown,
    run_eval,
)

# --- helpers ----------------------------------------------------------


def _make_packet(
    *,
    profile_id: str = "deepseek_planning",
    host_tools: tuple[str, ...] = ("get_context_view", "get_canonical_context", "query_trading_db"),
    mutation_allowed: bool = False,
    live_control_allowed: bool = False,
    required_reads: list[str] | None = None,
    literature_refs: list[dict] | None = None,
) -> dict:
    return {
        "task": {"text": "x", "route_id": "research_discovery", "title": "T"},
        "profile": {
            "provider": "openrouter",
            "profile_id": profile_id,
            "model": "deepseek/deepseek-v4-pro",
            "host_tools": list(host_tools),
        },
        "packet_contract": {
            "mode": "research_planning_read_only",
            "mutation_allowed": mutation_allowed,
            "live_control_allowed": live_control_allowed,
            "allowed_host_tools": list(host_tools),
        },
        "system_brief": {},
        "context_route": {},
        "required_reads": (required_reads if required_reads is not None else ["RESEARCH_RULES.md", "TRADING_RULES.md"]),
        "context_views": {},
        "grounding": {
            "local_literature_refs": (literature_refs if literature_refs is not None else []),
        },
        "system_prompt_seed": {"content": "grounding"},
        "openrouter_request_defaults": {"provider": {"allow_fallbacks": False}},
    }


def _make_envelope(packet: dict, *, profile_id: str = "deepseek_planning") -> dict:
    """Build a dry-run envelope shape that mirrors run_openrouter_task."""
    request = {
        "model": packet["profile"]["model"],
        "messages": [
            {"role": "system", "content": packet["system_prompt_seed"]["content"]},
            {"role": "user", "content": json.dumps(packet)},
        ],
    }
    host_tools = packet["profile"].get("host_tools") or ()
    if host_tools:
        request["tools"] = [
            {"type": "function", "function": {"name": name, "description": "", "parameters": {}}} for name in host_tools
        ]
        request["tool_choice"] = "auto"
    return {
        "status": "dry_run",
        "task": packet["task"]["text"],
        "profile": profile_id,
        "model": packet["profile"]["model"],
        "runtime_class": "read_only_tool_loop",
        "evidence_refs": packet["required_reads"],
        "tool_history": [],
        "request": request,
        "capabilities": {
            "supported_parameters": ["reasoning", "tools"],
            "context_length": 1_048_576,
        },
    }


# --- offline fixture --------------------------------------------------


class TestOfflineFixture:
    def test_payload_covers_all_models(self) -> None:
        models = {"deepseek_planning": "x/y", "deepseek_research": "a/b"}
        payload = _offline_models_payload(models)
        ids = {entry["id"] for entry in payload["data"]}
        assert ids == {"x/y", "a/b"}
        for entry in payload["data"]:
            for required in ("reasoning", "tools", "response_format", "structured_outputs"):
                assert required in entry["supported_parameters"]

    def test_offline_client_blocks_post(self) -> None:
        client = _OfflineModelsClient(_offline_models_payload({"p": "m"}))
        with pytest.raises(RuntimeError, match="must not POST"):
            client.post("/x", json={})

    def test_offline_client_returns_payload(self) -> None:
        payload = _offline_models_payload({"p": "model-1"})
        client = _OfflineModelsClient(payload)
        resp = client.get("https://openrouter.ai/api/v1/models")
        resp.raise_for_status()
        assert resp.json()["data"][0]["id"] == "model-1"


# --- rubric -----------------------------------------------------------


class TestRubric:
    def test_clean_envelope_passes_all(self) -> None:
        packet = _make_packet(literature_refs=[{"path": "docs/institutional/literature/x.md"}])
        envelope = _make_envelope(packet)
        task = EvalTask(
            task_id="t",
            text="x",
            expects_research_view=True,
            expects_literature_ref=True,
        )
        checks = _score_cell("deepseek_planning", task, envelope)
        failed = [c for c in checks if not c.passed]
        assert failed == [], f"unexpected failures: {[c.name for c in failed]}"

    def test_status_not_dry_run_fails(self) -> None:
        packet = _make_packet()
        envelope = _make_envelope(packet)
        envelope["status"] = "completed"  # would mean live call fired
        checks = _score_cell("deepseek_planning", EVAL_TASKS[0], envelope)
        names = {c.name: c.passed for c in checks}
        assert names["status_dry_run"] is False

    def test_mutation_in_contract_fails_read_only(self) -> None:
        packet = _make_packet(mutation_allowed=True)
        envelope = _make_envelope(packet)
        checks = _score_cell("deepseek_planning", EVAL_TASKS[0], envelope)
        names = {c.name: c.passed for c in checks}
        assert names["read_only_contract"] is False

    def test_mutation_attempt_task_structural_check(self) -> None:
        # The mutation_attempt task is the third in EVAL_TASKS.
        mutation_task = next(t for t in EVAL_TASKS if t.is_mutation_attempt)
        packet = _make_packet()
        envelope = _make_envelope(packet)
        checks = _score_cell("deepseek_planning", mutation_task, envelope)
        names = {c.name: c.passed for c in checks}
        assert names["mutation_attempt_refusal_structural"] is True

    def test_mutation_shaped_tool_name_leaked_fails(self) -> None:
        packet = _make_packet(host_tools=("get_context_view", "write_lane_pause"))
        envelope = _make_envelope(packet)
        checks = _score_cell("deepseek_planning", EVAL_TASKS[0], envelope)
        names = {c.name: c.passed for c in checks}
        assert names["no_mutation_tools"] is False

    def test_mutation_fragments_are_lowercased_in_check(self) -> None:
        # Sanity: the canonical fragments include "write" and "pause".
        assert "write" in MUTATION_TOOL_FRAGMENTS
        assert "pause" in MUTATION_TOOL_FRAGMENTS

    def test_required_reads_missing_fails(self) -> None:
        packet = _make_packet(required_reads=[])
        envelope = _make_envelope(packet)
        checks = _score_cell("deepseek_planning", EVAL_TASKS[0], envelope)
        names = {c.name: c.passed for c in checks}
        assert names["required_reads_present"] is False

    def test_literature_ref_required_when_expected(self) -> None:
        packet = _make_packet(literature_refs=[])
        envelope = _make_envelope(packet)
        task = EvalTask(
            task_id="t",
            text="x",
            expects_literature_ref=True,
        )
        checks = _score_cell("deepseek_planning", task, envelope)
        names = {c.name: c.passed for c in checks}
        assert names["literature_ref_present"] is False

    def test_tool_spec_coherence_on_request_mismatch(self) -> None:
        packet = _make_packet(host_tools=("get_context_view", "query_trading_db"))
        envelope = _make_envelope(packet)
        # Sabotage: drop one tool from request.
        envelope["request"]["tools"] = [
            t for t in envelope["request"]["tools"] if t["function"]["name"] != "query_trading_db"
        ]
        checks = _score_cell("deepseek_planning", EVAL_TASKS[0], envelope)
        names = {c.name: c.passed for c in checks}
        assert names["tool_spec_coherence"] is False


# --- packet extraction ------------------------------------------------


class TestPacketExtraction:
    def test_pulls_user_message(self) -> None:
        packet = _make_packet()
        envelope = _make_envelope(packet)
        extracted = _packet_from_envelope(envelope)
        assert extracted["profile"]["profile_id"] == "deepseek_planning"

    def test_missing_user_message_returns_empty(self) -> None:
        envelope = {"request": {"messages": [{"role": "system", "content": "x"}]}}
        assert _packet_from_envelope(envelope) == {}


# --- driver -----------------------------------------------------------


class TestRunEval:
    def test_skips_profile_with_validation_errors(self) -> None:
        # No env vars -> deepseek_planning has missing OPENROUTER_API_KEY +
        # missing model -> validation_errors() is non-empty.
        with patch.dict("os.environ", {}, clear=True):
            results = run_eval(
                offline=True,
                profile_ids=["deepseek_planning"],
                tasks=(EVAL_TASKS[0],),
            )
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert results[0].profile_validation_errors  # non-empty

    def test_scored_cells_run_dry_run(self, tmp_path: Path) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-or-test",
            "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "trading_app.ai.openrouter_runtime.build_system_context",
                return_value=SimpleNamespace(git=SimpleNamespace(in_linked_worktree=True)),
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_research_packet",
                return_value=_make_packet(
                    literature_refs=[{"path": "docs/institutional/literature/x.md"}],
                ),
            ),
        ):
            results = run_eval(
                root=tmp_path,
                db_path=tmp_path / "fake.db",
                offline=True,
                profile_ids=["deepseek_planning"],
                tasks=EVAL_TASKS,
            )
        assert len(results) == 3
        for r in results:
            assert r.status == "scored", f"{r.profile_id}/{r.task_id}: {r.error}"
            assert r.fail_count == 0, f"{r.profile_id}/{r.task_id} fails: {[c.name for c in r.checks if not c.passed]}"


# --- rendering --------------------------------------------------------


class TestRendering:
    def test_markdown_lists_cells(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-or-test",
            "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "trading_app.ai.openrouter_runtime.build_system_context",
                return_value=SimpleNamespace(git=SimpleNamespace(in_linked_worktree=True)),
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_research_packet",
                return_value=_make_packet(
                    literature_refs=[{"path": "docs/institutional/literature/x.md"}],
                ),
            ),
        ):
            results = run_eval(
                offline=True,
                profile_ids=["deepseek_planning"],
                tasks=(EVAL_TASKS[0],),
            )
        md = render_markdown(results)
        assert "deepseek_planning" in md
        assert EVAL_TASKS[0].task_id in md
        assert "Failed Checks" in md

    def test_json_round_trips(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-or-test",
            "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "trading_app.ai.openrouter_runtime.build_system_context",
                return_value=SimpleNamespace(git=SimpleNamespace(in_linked_worktree=True)),
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_research_packet",
                return_value=_make_packet(),
            ),
        ):
            results = run_eval(
                offline=True,
                profile_ids=["deepseek_planning"],
                tasks=(EVAL_TASKS[0],),
            )
        payload = json.loads(render_json(results))
        assert payload[0]["profile_id"] == "deepseek_planning"
        assert payload[0]["status"] == "scored"


# --- main exit codes --------------------------------------------------


class TestExitCodes:
    def test_main_returns_zero_on_clean_pass(self, tmp_path: Path, capsys) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-or-test",
            "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "trading_app.ai.openrouter_runtime.build_system_context",
                return_value=SimpleNamespace(git=SimpleNamespace(in_linked_worktree=True)),
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_research_packet",
                return_value=_make_packet(
                    literature_refs=[{"path": "docs/institutional/literature/x.md"}],
                ),
            ),
        ):
            rc = main(["--offline", "--profile", "deepseek_planning"])
        assert rc == 0

    def test_main_returns_one_on_check_failure(self) -> None:
        # Force a failure by patching build_research_packet to return a
        # mutation-allowed contract.
        env = {
            "OPENROUTER_API_KEY": "sk-or-test",
            "CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL": "deepseek/deepseek-v4-pro",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "trading_app.ai.openrouter_runtime.build_system_context",
                return_value=SimpleNamespace(git=SimpleNamespace(in_linked_worktree=True)),
            ),
            patch(
                "trading_app.ai.openrouter_runtime.build_research_packet",
                return_value=_make_packet(mutation_allowed=True),
            ),
        ):
            rc = main(["--offline", "--profile", "deepseek_planning"])
        assert rc == 1
