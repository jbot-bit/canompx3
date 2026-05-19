"""Tests for scripts/tools/fast_lane_status.py (Stage 2A.2 — status roll-up writer)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from scripts.tools.fast_lane_status import (
    SCHEMA_VERSION,
    STAGE_PRECEDENCE,
    StatusEntry,
    _classify_stage,
    build_status_entries,
    collect_active_preregs,
    collect_drafts,
    serialize_rollup,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _fake_chain(tmp_path: Path) -> dict[str, Path]:
    hyp = tmp_path / "docs" / "audit" / "hypotheses"
    drafts = hyp / "drafts"
    results = tmp_path / "docs" / "audit" / "results"
    runtime = tmp_path / "docs" / "runtime"
    for d in (hyp, drafts, results, runtime):
        d.mkdir(parents=True, exist_ok=True)
    return {"hyp": hyp, "drafts": drafts, "results": results, "runtime": runtime}


def test_collect_active_preregs_skips_template_placeholder(tmp_path: Path) -> None:
    paths = _fake_chain(tmp_path)
    _write_yaml(
        paths["hyp"] / "TEMPLATE-fast-lane-v5.1.yaml",
        {"scope": {"strategy_id": "<canonical_id>"}},
    )
    _write_yaml(
        paths["hyp"] / "real.yaml",
        {"scope": {"strategy_id": "MNQ_REAL_E1_RR1.0"}},
    )
    out = collect_active_preregs(paths["hyp"])
    assert "<canonical_id>" not in out
    assert "MNQ_REAL_E1_RR1.0" in out


def test_collect_drafts_pairs_draft_with_grounded_sibling(tmp_path: Path) -> None:
    paths = _fake_chain(tmp_path)
    draft = paths["drafts"] / "2026-05-19-foo.draft.yaml"
    grounded = paths["drafts"] / "2026-05-19-foo.grounded.yaml"
    _write_yaml(draft, {"scope": {"strategy_id": "MNQ_FOO"}})
    _write_yaml(grounded, {"scope": {"strategy_id": "MNQ_FOO"}})
    out = collect_drafts(paths["drafts"])
    assert out["MNQ_FOO"]["draft"] == draft
    assert out["MNQ_FOO"]["grounded"] == grounded


def test_classify_stage_terminal_short_circuits_downstream() -> None:
    """A REVOKED queue entry stays REVOKED even if a draft was authored later."""
    stage = _classify_stage(
        active_prereg=None,
        queue_entry={"status": "REVOKED"},
        journal_entry=None,
        is_ranked=True,
        drafts={"draft": Path("/tmp/foo.draft.yaml")},
        heavyweight_result=None,
    )
    assert stage == "REVOKED"


def test_classify_stage_downstream_wins_when_non_terminal() -> None:
    """ENRICHED beats HEAVYWEIGHT_COMPLETE when journal verdict is populated."""
    stage = _classify_stage(
        active_prereg=Path("/tmp/p.yaml"),
        queue_entry={"status": "ESCALATED"},
        journal_entry={"heavyweight_verdict": "PASS_CHORDIA"},
        is_ranked=True,
        drafts={"draft": Path("/tmp/d.yaml"), "grounded": Path("/tmp/g.yaml")},
        heavyweight_result=Path("/tmp/r.md"),
    )
    assert stage == "ENRICHED"


def test_classify_stage_active_prereg_only_path() -> None:
    stage = _classify_stage(
        active_prereg=Path("/tmp/p.yaml"),
        queue_entry=None,
        journal_entry=None,
        is_ranked=False,
        drafts=None,
        heavyweight_result=None,
    )
    assert stage == "ACTIVE_PREREG"


def test_classify_stage_returns_error_when_nothing_observed() -> None:
    stage = _classify_stage(
        active_prereg=None,
        queue_entry=None,
        journal_entry=None,
        is_ranked=False,
        drafts=None,
        heavyweight_result=None,
    )
    assert stage == "ERROR"


def test_stage_precedence_includes_all_terminal_and_pipeline_stages() -> None:
    """Locks the enum — adding/removing a stage requires updating the test."""
    assert set(STAGE_PRECEDENCE) == {
        "ACTIVE_PREREG",
        "FAST_LANE_RUN",
        "PROMOTE_QUEUED",
        "RANKED",
        "BRIDGED",
        "GROUNDED",
        "HEAVYWEIGHT_PENDING",
        "HEAVYWEIGHT_COMPLETE",
        "ENRICHED",
        "REVOKED",
        "PARKED",
        "REJECTED_OOS_UNPOWERED",
        "ERROR",
    }


def test_serialize_rollup_banner_fields_present() -> None:
    entries = [
        StatusEntry(
            strategy_id="MNQ_X",
            current_stage="ACTIVE_PREREG",
            age_days=0,
            next_action_token="run_fast_lane",
            upstream_artifact_path=None,
            downstream_artifact_path=None,
        )
    ]
    payload = yaml.safe_load(serialize_rollup(entries, today=date(2026, 5, 19)))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["do_not_hand_edit"] is True
    assert payload["source"] == "scripts/tools/fast_lane_status.py"
    assert payload["generated_at"] == "2026-05-19"
    assert "DERIVED STATE" in payload["warning"]
    assert payload["entries"][0]["strategy_id"] == "MNQ_X"


def test_build_status_entries_idempotent_on_fixed_inputs(tmp_path: Path) -> None:
    paths = _fake_chain(tmp_path)
    _write_yaml(
        paths["hyp"] / "p.yaml",
        {"scope": {"strategy_id": "MNQ_A"}},
    )
    first = build_status_entries(
        hypotheses_dir=paths["hyp"],
        drafts_dir=paths["drafts"],
        results_dir=paths["results"],
        runtime_dir=paths["runtime"],
        queue_cache=paths["runtime"] / "promote_queue.yaml",
        journal_path=paths["runtime"] / "cherry_pick_journal.yaml",
        today=date(2026, 5, 19),
    )
    second = build_status_entries(
        hypotheses_dir=paths["hyp"],
        drafts_dir=paths["drafts"],
        results_dir=paths["results"],
        runtime_dir=paths["runtime"],
        queue_cache=paths["runtime"] / "promote_queue.yaml",
        journal_path=paths["runtime"] / "cherry_pick_journal.yaml",
        today=date(2026, 5, 19),
    )
    # Same strategy_ids, same stages — idempotency invariant.
    assert [e.strategy_id for e in first] == [e.strategy_id for e in second]
    assert [e.current_stage for e in first] == [e.current_stage for e in second]


def test_build_status_entries_live_repo_smoke() -> None:
    """Stage acceptance criterion #4: the writer runs end-to-end on live repo."""
    entries = build_status_entries()
    # Non-empty (the live chain has dozens of active preregs).
    assert len(entries) > 0
    # Every entry has a valid stage from the canonical enum.
    for e in entries:
        assert e.current_stage in STAGE_PRECEDENCE, e
        assert isinstance(e.age_days, int) and e.age_days >= 0
