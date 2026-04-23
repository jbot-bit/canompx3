from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.tools.prereg_front_door import build_route_decision, execute_route
from trading_app.hypothesis_loader import HypothesisLoaderError

PIPELINE_SURFACES = [
    "docs/institutional/research_pipeline_contract.md",
    "docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md",
    "docs/audit/hypotheses/README.md",
    "docs/institutional/hypothesis_registry_template.md",
    ".claude/skills/discover/SKILL.md",
    ".codex/skills/canompx3-research/SKILL.md",
    ".codex/WORKFLOWS.md",
    ".codex/COMMANDS.md",
]


def _write_yaml(path: Path, *, research_question_type: str, extra: str = "") -> None:
    role_block = ""
    if research_question_type == "conditional_role":
        role_block = """
    role:
      kind: "conditioner"
      parent: "Exact parent"
      comparator: "Signal vs complement"
      primary_metric: "policy_ev_per_opportunity_r"
      promotion_target: "shadow_only"
"""
    path.write_text(
        f"""
metadata:
  name: "test-prereg"
  date_locked: "2026-04-23T00:00:00+10:00"
  holdout_date: "2026-01-01"
  total_expected_trials: 1
  research_question_type: "{research_question_type}"
hypotheses:
  - id: 1
    name: "test"
    theory_citation: "docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md"
    economic_basis: "Test fixture."
    filter:
      type: "ORB_G5"
      column: "orb_TEST_size"
      thresholds: [5]
    scope:
      instruments: [MNQ]
      sessions: [US_DATA_1000]
      rr_targets: [1.5]
      entry_models: [E2]
      confirm_bars: [1]
      stop_multipliers: [1.0]
      orb_minutes: [15]
    expected_trial_count: 1
    kill_criteria:
      - "fixture"
{role_block}
{extra}
""".lstrip(),
        encoding="utf-8",
    )


def test_standalone_edge_routes_to_grid_discovery(tmp_path: Path) -> None:
    prereg = tmp_path / "standalone.yaml"
    _write_yaml(prereg, research_question_type="standalone_edge")

    route = build_route_decision(prereg)

    assert route.execution_mode == "grid_discovery"
    assert route.instrument == "MNQ"
    assert route.orb_minutes == 15
    assert route.writes_to == ["experimental_strategies"]
    assert route.route_kind == "standalone_discovery"
    assert route.research_destination == "experimental_strategies"
    assert route.validation_surface == "validated_setups"
    assert route.deployment_required is False
    assert route.execution_required is False
    assert "paper_trades are a validation prerequisite" in route.forbidden_claims
    assert route.next_surface.startswith("experimental_strategies -> strategy_validator -> validated_setups")
    assert "optional deployment" in route.next_surface
    assert "optional paper_trades" in route.next_surface


def test_conditional_role_routes_to_bounded_runner(tmp_path: Path) -> None:
    prereg = tmp_path / "conditional.yaml"
    _write_yaml(prereg, research_question_type="conditional_role")

    route = build_route_decision(prereg)

    assert route.execution_mode == "bounded_runner"
    assert "experimental_strategies" not in route.writes_to
    assert route.route_kind == "conditional_role"
    assert route.research_destination == "docs/audit/results (or bounded research artifact only)"
    assert route.validation_surface == "role-specific result doc / role contract"
    assert route.deployment_required is False
    assert route.execution_required is False
    assert route.next_surface.startswith("bounded result doc -> explicit role decision")
    assert "validated_setups" not in route.next_surface
    assert "paper_trades" not in route.next_surface
    assert "conditional-role results auto-promote to validated_setups" in route.forbidden_claims
    assert any("do not auto-write" in note for note in route.notes)


def test_unsupported_research_question_type_fails_closed(tmp_path: Path) -> None:
    prereg = tmp_path / "unsupported.yaml"
    _write_yaml(prereg, research_question_type="portfolio_magic")

    with pytest.raises(HypothesisLoaderError, match="research_question_type"):
        build_route_decision(prereg)


def test_multi_slice_standalone_requires_explicit_execution_slice(tmp_path: Path) -> None:
    prereg = tmp_path / "multi.yaml"
    _write_yaml(
        prereg,
        research_question_type="standalone_edge",
        extra="""
  - id: 2
    name: "test two"
    theory_citation: "docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md"
    economic_basis: "Test fixture."
    filter:
      type: "ORB_G5"
      column: "orb_TEST_size"
      thresholds: [5]
    scope:
      instruments: [MGC]
      sessions: [US_DATA_1000]
      rr_targets: [1.5]
      entry_models: [E2]
      confirm_bars: [1]
      stop_multipliers: [1.0]
      orb_minutes: [30]
    expected_trial_count: 1
    kill_criteria:
      - "fixture"
""",
    )

    route = build_route_decision(prereg)

    assert route.instrument is None
    assert route.orb_minutes is None
    assert any("Multi-slice prereg" in note for note in route.notes)


def test_conditional_execute_uses_bounded_runner_and_default_args(tmp_path: Path) -> None:
    prereg = tmp_path / "conditional_runner.yaml"
    _write_yaml(
        prereg,
        research_question_type="conditional_role",
        extra="""
execution:
  mode: "bounded_runner"
  entrypoint: "research/my_bounded_runner.py"
  default_args:
    - "--output"
    - "docs/audit/results/test.md"
""",
    )
    route = build_route_decision(prereg)

    with patch("scripts.tools.prereg_front_door.subprocess.run") as run:
        run.return_value.returncode = 0
        rc = execute_route(
            route,
            hypothesis_file=prereg,
            start=None,
            end=None,
            db=None,
            dry_run=False,
            runner_args=["--extra"],
        )

    assert rc == 0
    cmd = run.call_args.args[0]
    assert cmd[1].replace("\\", "/").endswith("research/my_bounded_runner.py")
    assert cmd[-3:] == ["--output", "docs/audit/results/test.md", "--extra"]


def test_cli_json_inspection_reports_pipeline_destination(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    prereg = tmp_path / "standalone.yaml"
    _write_yaml(prereg, research_question_type="standalone_edge")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/tools/prereg_front_door.py",
            "--hypothesis-file",
            str(prereg),
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
    expected_keys = {
        "route_kind",
        "status_path",
        "research_destination",
        "validation_surface",
        "deployment_required",
        "execution_required",
        "forbidden_claims",
    }
    assert expected_keys <= payload.keys()
    assert payload["execution_mode"] == "grid_discovery"
    assert payload["route_kind"] == "standalone_discovery"
    assert payload["writes_to"] == ["experimental_strategies"]
    assert payload["deployment_required"] is False
    assert payload["execution_required"] is False
    assert payload["next_surface"].startswith("experimental_strategies -> strategy_validator -> validated_setups")


def test_pipeline_surfaces_name_all_route_options() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    required_routes = [
        "standalone_discovery",
        "conditional_role",
        "confirmation",
        "deployment_readiness",
        "operations",
    ]

    for surface in PIPELINE_SURFACES:
        text = (repo_root / surface).read_text(encoding="utf-8")
        for route in required_routes:
            assert route in text, f"{surface} does not mention {route}"


def test_pipeline_surfaces_do_not_require_execution_for_validation() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    stale_phrases = [
        "profile/lane routing -> " + "paper_trades",
        "paper_trades are required " + "for validation",
        "paper_trades required " + "for validation",
        "live routing is required " + "for validation",
        "live routing required " + "for validation",
    ]

    for surface in PIPELINE_SURFACES + ["scripts/tools/prereg_front_door.py"]:
        text = (repo_root / surface).read_text(encoding="utf-8")
        for phrase in stale_phrases:
            assert phrase not in text, f"{surface} contains stale phrase: {phrase}"
