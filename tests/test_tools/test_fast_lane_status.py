"""Tests for scripts/tools/fast_lane_status.py (Stage 2A.2 — status roll-up writer)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from scripts.tools.fast_lane_status import (
    _NEXT_ACTION_BY_STAGE,
    _NEXT_ACTION_HEAVYWEIGHT_COMPLETE_NO_LINEAGE,
    SCHEMA_VERSION,
    STAGE_PRECEDENCE,
    StatusEntry,
    _classify_stage,
    _next_action_for,
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


# ---------------------------------------------------------------------------
# HEAVYWEIGHT_COMPLETE lineage-qualified next-action routing
# ---------------------------------------------------------------------------
#
# Background: heavyweight Chordia preregs authored directly (predating the
# 2026-05-19 cherry-pick loop) reach HEAVYWEIGHT_COMPLETE without ever being
# scored by the ranker. The cherry-pick journal has no row for them, and the
# enricher is structurally update-only — it cannot create entries. Emitting
# `run_cherry_pick_journal_enricher` for such entries silently no-ops on every
# walk. The lineage-qualified action token routes the operator to the
# heavyweight result MD (deployment decision) instead.


def test_next_action_heavyweight_complete_with_journal_lineage_uses_enricher() -> None:
    """A journal entry signals the strategy went through the ranker — enricher applies."""
    action = _next_action_for(
        "HEAVYWEIGHT_COMPLETE",
        journal_entry={"iter": 1, "strategy_id": "MNQ_X", "heavyweight_verdict": None},
    )
    assert action == "run_cherry_pick_journal_enricher"
    # Also verify the canonical mapping still carries the enricher token —
    # this guards against accidental rename of the lookup constant.
    assert _NEXT_ACTION_BY_STAGE["HEAVYWEIGHT_COMPLETE"] == "run_cherry_pick_journal_enricher"


def test_next_action_heavyweight_complete_without_journal_lineage_uses_deployment() -> None:
    """No journal entry → no fast-lane lineage → operator deployment decision, not enricher."""
    action = _next_action_for("HEAVYWEIGHT_COMPLETE", journal_entry=None)
    assert action == _NEXT_ACTION_HEAVYWEIGHT_COMPLETE_NO_LINEAGE
    assert action == "operator_deployment_decision"
    # Critical: never silently route to a stage script that will no-op.
    assert action != "run_cherry_pick_journal_enricher"


def test_next_action_other_stages_unchanged_by_lineage_qualifier() -> None:
    """Every non-HEAVYWEIGHT_COMPLETE stage routes identically with or without journal."""
    for stage in STAGE_PRECEDENCE:
        if stage == "HEAVYWEIGHT_COMPLETE":
            continue
        expected = _NEXT_ACTION_BY_STAGE[stage]
        assert _next_action_for(stage, journal_entry=None) == expected
        assert _next_action_for(stage, journal_entry={"iter": 1}) == expected


def test_next_action_for_unknown_stage_falls_back_to_error_token() -> None:
    """Defensive: an unrecognized stage string lands on the error token, not raise."""
    action = _next_action_for("WAT_NOT_A_STAGE", journal_entry=None)
    assert action == "operator_resolve_error"


def test_build_status_entries_pre_ranker_heavyweight_gets_deployment_action(tmp_path: Path) -> None:
    """Integration: a heavyweight result MD with no journal entry routes to deployment, not enricher.

    Mirrors the live-repo case for the 38 entries fixed by this stage.
    """
    paths = _fake_chain(tmp_path)
    # Heavyweight result MD with the canonical title format that
    # collect_heavyweight_results parses.
    md = paths["results"] / "2026-05-15-mnq-fake-chordia-unlock-v1.md"
    md.write_text(
        "# Chordia strict unlock audit — MNQ_FAKE_PRE_RANKER\n\n**MEASURED verdict:** `PASS_CHORDIA`\n",
        encoding="utf-8",
    )
    # NO journal entry — the journal YAML is absent / empty by design.
    entries = build_status_entries(
        hypotheses_dir=paths["hyp"],
        drafts_dir=paths["drafts"],
        results_dir=paths["results"],
        runtime_dir=paths["runtime"],
        queue_cache=paths["runtime"] / "promote_queue.yaml",
        journal_path=paths["runtime"] / "cherry_pick_journal.yaml",
        today=date(2026, 5, 20),
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.strategy_id == "MNQ_FAKE_PRE_RANKER"
    assert e.current_stage == "HEAVYWEIGHT_COMPLETE"
    assert e.next_action_token == "operator_deployment_decision"


def test_build_status_entries_post_ranker_heavyweight_gets_enricher_action(tmp_path: Path) -> None:
    """Integration: a heavyweight result MD WITH a journal entry routes to enricher.

    The cherry-pick loop authored a journal entry pre-heavyweight; once the
    heavyweight verdict lands, the enricher's job is to backfill it.
    """
    paths = _fake_chain(tmp_path)
    md = paths["results"] / "2026-05-19-mnq-real-chordia-heavyweight-v1.md"
    md.write_text(
        "# Chordia strict unlock audit — MNQ_REAL_POST_RANKER\n\n**MEASURED verdict:** `PASS_CHORDIA`\n",
        encoding="utf-8",
    )
    # Journal entry exists — pre-heavyweight (heavyweight_verdict=None).
    journal = paths["runtime"] / "cherry_pick_journal.yaml"
    _write_yaml(
        journal,
        {
            "schema_version": 1,
            "entries": [
                {
                    "iter": 1,
                    "strategy_id": "MNQ_REAL_POST_RANKER",
                    "heavyweight_verdict": None,
                }
            ],
        },
    )
    entries = build_status_entries(
        hypotheses_dir=paths["hyp"],
        drafts_dir=paths["drafts"],
        results_dir=paths["results"],
        runtime_dir=paths["runtime"],
        queue_cache=paths["runtime"] / "promote_queue.yaml",
        journal_path=journal,
        today=date(2026, 5, 20),
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.strategy_id == "MNQ_REAL_POST_RANKER"
    assert e.current_stage == "HEAVYWEIGHT_COMPLETE"
    assert e.next_action_token == "run_cherry_pick_journal_enricher"
