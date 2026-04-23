from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.tools.prereg_front_door import build_route_decision


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
    assert "strategy_validator" in route.next_surface


def test_conditional_role_routes_to_bounded_runner(tmp_path: Path) -> None:
    prereg = tmp_path / "conditional.yaml"
    _write_yaml(prereg, research_question_type="conditional_role")

    route = build_route_decision(prereg)

    assert route.execution_mode == "bounded_runner"
    assert "experimental_strategies" not in route.writes_to
    assert any("do not auto-write" in note for note in route.notes)


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
    assert payload["execution_mode"] == "grid_discovery"
    assert payload["writes_to"] == ["experimental_strategies"]
