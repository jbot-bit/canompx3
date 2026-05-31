from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from research import chordia_strict_unlock_v1 as strict_runner
from scripts.tools import chordia_evidence_factory as cef
from scripts.tools import chordia_replay_batch_bridge as bridge


def _bench_row(strategy_id: str, *, rank: int = 18) -> dict[str, str]:
    return {
        "rank": str(rank),
        "strategy_id": strategy_id,
        "state": "EXACT_LANE_READY_FOR_REPLAY",
        "primary_blocker": "MISSING_CHORDIA",
        "next_action": "RUN_STRICT_UNLOCK",
        "instrument": "MNQ",
        "session": "NYSE_OPEN",
        "orb_minutes": "15",
        "entry_model": "E2",
        "rr_target": "1.0",
        "filter_type": "NO_FILTER",
        "confirm_bars": "1",
        "trailing_expr": "0.1245",
        "trailing_n": "224",
        "annual_r_estimate": "25.7",
        "status": "DEPLOY",
        "status_reason": "Session HOT, ExpR=+0.1245, N=224",
        "chordia_verdict": "MISSING",
        "chordia_audit_age_days": "",
        "c8_oos_status": "",
        "family": "MNQ NYSE_OPEN baseline_orb",
        "family_role": "deployable_candidate",
        "family_priority": "0",
        "active_in_profile": "False",
    }


def _factory(tmp_path: Path) -> Path:
    rows = [
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15", rank=18),
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075", rank=19),
    ]
    cef.write_factory_artifacts(
        rows,
        output_dir=tmp_path,
        today="2026-05-31",
        limit=0,
        max_family_priority=0,
        batch_size=25,
    )
    return tmp_path


def test_plan_acceptance_uses_only_ready_rows(tmp_path: Path) -> None:
    factory_dir = _factory(tmp_path)

    plan = bridge.plan_acceptance(
        factory_dir=factory_dir,
        batch_id="batch_001",
        hypotheses_dir=tmp_path / "hypotheses",
    )

    assert [(row.strategy_id, row.bridge_status) for row in plan] == [
        ("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15", "READY_TO_ACTIVATE"),
        ("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075", "READY_TO_ACTIVATE"),
    ]
    assert plan[0].active_hypothesis_path.endswith(
        "2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-no-filter-o15-chordia-unlock-v1.yaml"
    )
    assert plan[1].active_hypothesis_path.endswith(
        "2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-no-filter-o15-s075-chordia-unlock-v1.yaml"
    )


def test_activate_draft_flips_execution_gate_without_theory_citation(tmp_path: Path) -> None:
    factory_dir = _factory(tmp_path)
    plan = bridge.plan_acceptance(
        factory_dir=factory_dir,
        batch_id="batch_001",
        hypotheses_dir=tmp_path / "hypotheses",
    )

    activated = bridge.activate_ready_drafts(plan, repo_root=Path.cwd())

    assert len(activated) == 2
    active_path = Path(activated[0].active_hypothesis_path)
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    source = yaml.safe_load(Path(activated[0].source_draft_path).read_text(encoding="utf-8"))
    assert source["execution_gate"]["allowed_now"] is False
    assert active["execution_gate"]["allowed_now"] is True
    assert "theory_citation" not in active["metadata"]
    assert active["bridge_activation"]["source_draft"].endswith(".draft.yaml")


def test_write_bridge_artifacts_outputs_plan_and_no_audit_mutation(tmp_path: Path) -> None:
    factory_dir = _factory(tmp_path / "factory")
    output_dir = tmp_path / "bridge"
    plan = bridge.plan_acceptance(
        factory_dir=factory_dir,
        batch_id="batch_001",
        hypotheses_dir=tmp_path / "hypotheses",
    )

    bridge.write_bridge_artifacts(plan, output_dir=output_dir, results=(), audit_date="2026-05-31")

    with (output_dir / "activation_plan.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    manifest = yaml.safe_load((output_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert rows[0]["bridge_status"] == "READY_TO_ACTIVATE"
    assert manifest["chordia_log_mutation"] is False
    assert manifest["live_mutation"] is False
    assert manifest["strict_replay_execution"] is False


def test_run_requires_active_hypothesis_file(tmp_path: Path) -> None:
    factory_dir = _factory(tmp_path)
    plan = bridge.plan_acceptance(
        factory_dir=factory_dir,
        batch_id="batch_001",
        hypotheses_dir=tmp_path / "hypotheses",
    )

    with patch("scripts.tools.chordia_replay_batch_bridge.subprocess.run") as run:
        try:
            bridge.run_strict_replays(plan, repo_root=Path.cwd(), max_runs=1)
        except FileNotFoundError as exc:
            assert "Active hypothesis file is missing" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError("run_strict_replays must fail before subprocess when the active file is absent")
        run.assert_not_called()


def test_run_uses_activated_hypothesis_path(tmp_path: Path) -> None:
    factory_dir = _factory(tmp_path)
    plan = bridge.plan_acceptance(
        factory_dir=factory_dir,
        batch_id="batch_001",
        hypotheses_dir=tmp_path / "hypotheses",
    )
    bridge.activate_ready_drafts(plan, repo_root=Path.cwd())
    proc = Mock(returncode=0, stdout="ok", stderr="")

    with patch("scripts.tools.chordia_replay_batch_bridge.subprocess.run", return_value=proc) as run:
        runs = bridge.run_strict_replays(plan, repo_root=Path.cwd(), max_runs=1)

    assert len(runs) == 1
    assert Path(run.call_args.args[0][-1]).is_absolute()
    assert runs[0].returncode == 0


def test_activated_factory_draft_loads_in_strict_runner(tmp_path: Path) -> None:
    factory_dir = _factory(tmp_path)
    plan = bridge.plan_acceptance(
        factory_dir=factory_dir,
        batch_id="batch_001",
        hypotheses_dir=tmp_path / "hypotheses",
    )
    activated = bridge.activate_ready_drafts(plan, repo_root=Path.cwd())

    cell = strict_runner._load_cell(Path(activated[0].active_hypothesis_path))

    assert cell.strategy_id == "MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15"
    assert cell.template_version == "chordia_strict_v1"


def test_strict_runner_script_entrypoint_imports_from_repo_root() -> None:
    proc = subprocess.run(
        [sys.executable, "research/chordia_strict_unlock_v1.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--hypothesis-file" in proc.stdout
