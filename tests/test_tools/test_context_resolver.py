from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_context_resolver_matches_research_investigation_json() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/context_resolver.py",
            "--task",
            "Investigate why MES 5m ORB win rate dropped last week.",
            "--format",
            "json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["task"]["id"] == "research_investigation"
    assert payload["decision_protocol"]["id"] == "research_investigation_protocol"
    assert payload["answer_contract"]["id"] == "research_investigation_answer"
    assert payload["drilldown_playbook"]["id"] == "research_recent_performance_drilldown"
    assert any(pack["id"] == "coding_runtime_pack" for pack in payload["understanding_packs"])
    assert any(pack["id"] == "trading_runtime_pack" for pack in payload["understanding_packs"])
    assert any(variable["id"] == "orb_utc_window" for variable in payload["variables"])
    assert "RESEARCH_RULES.md" in payload["doctrine_files"]
    assert any(view["id"] == "gold_db_mcp" for view in payload["live_views"])
    assert any(view["id"] == "recent_performance_context" for view in payload["live_views"])
    assert any(step["id"] == "project_pulse_fast" for step in payload["verification_steps"])


def test_context_resolver_falls_back_for_unknown_task() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/context_resolver.py",
            "--task",
            "Compose a limerick about penguins.",
            "--format",
            "text",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Fallback read set" in result.stdout


def test_context_resolver_reports_reason_in_json_fallback() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/context_resolver.py",
            "--task",
            "Compose a limerick about penguins.",
            "--format",
            "json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["matched"] is False
    assert payload["reason"] == "No deterministic task match found."


def test_context_resolver_matches_completion_claim() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/context_resolver.py",
            "--task",
            "Is this done? verify done before we close it.",
            "--format",
            "json",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["task"]["id"] == "completion_claim"
    assert payload["verification"]["id"] == "done"
    assert payload["decision_protocol"]["id"] == "completion_protocol"
    assert any(step["id"] == "pytest_full" for step in payload["verification_steps"])
