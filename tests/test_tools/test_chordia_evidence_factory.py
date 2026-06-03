from __future__ import annotations

from pathlib import Path

from scripts.tools import chordia_evidence_factory as cef
from trading_app.hypothesis_loader import check_mode_a_consistency, load_hypothesis_metadata


def _bench_row(
    strategy_id: str,
    *,
    rank: int = 18,
    state: str = "EXACT_LANE_READY_FOR_REPLAY",
    blocker: str = "MISSING_CHORDIA",
    family_priority: int = 0,
) -> dict[str, str]:
    return {
        "rank": str(rank),
        "strategy_id": strategy_id,
        "state": state,
        "primary_blocker": blocker,
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
        "family_priority": str(family_priority),
        "active_in_profile": "False",
    }


def test_select_replay_candidates_allows_supported_non_default_stop_by_default() -> None:
    rows = [
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15"),
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075", rank=19),
    ]

    plan = cef.plan_replay_work(rows, limit=5, max_family_priority=0, include_non_default_stop=False)

    assert [(row.strategy_id, row.factory_status) for row in plan] == [
        ("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15", "PREREG_DRAFT_READY"),
        ("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075", "PREREG_DRAFT_READY"),
    ]
    assert plan[1].draft_path is not None
    assert plan[1].stop_multiplier == 0.75


def test_build_prereg_draft_is_no_theory_and_runner_bounded() -> None:
    candidate = cef.FactoryCandidate.from_bench_row(_bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15"))

    draft = cef.build_prereg_draft(candidate, today="2026-05-31")

    assert draft["metadata"]["theory_grant"] is False
    assert "theory_citation" not in draft["metadata"]
    assert draft["execution"]["entrypoint"] == "research/chordia_strict_unlock_v1.py"
    assert draft["execution_gate"]["allowed_now"] is False
    assert draft["scope"]["strategy_id"] == "MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15"
    assert draft["scope"]["stop_multiplier"] == 1.0
    assert draft["hypotheses"][0]["scope"]["stop_multipliers"] == [1.0]
    assert draft["primary_schema"]["chordia_threshold_basis"].endswith("(t >= 3.79)")


def test_build_prereg_draft_preserves_supported_non_default_stop() -> None:
    candidate = cef.FactoryCandidate.from_bench_row(_bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075"))

    draft = cef.build_prereg_draft(candidate, today="2026-05-31")

    assert draft["scope"]["strategy_id"] == "MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075"
    assert draft["scope"]["stop_multiplier"] == 0.75
    assert draft["hypotheses"][0]["scope"]["stop_multipliers"] == [0.75]


def test_build_prereg_draft_loads_through_hypothesis_loader(tmp_path: Path) -> None:
    candidate = cef.FactoryCandidate.from_bench_row(_bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15"))
    draft = cef.build_prereg_draft(candidate, today="2026-05-31")
    path = tmp_path / "draft.yaml"

    cef.write_prereg_draft(draft, path)
    meta = load_hypothesis_metadata(path)
    check_mode_a_consistency(meta)

    assert meta["has_theory"] is False
    assert meta["research_question_type"] == "conditional_role"


def test_parse_result_md_builds_audit_log_proposal(tmp_path: Path) -> None:
    md = tmp_path / "result.md"
    md.write_text(
        "\n".join(
            [
                "# Chordia strict unlock audit - MNQ_TEST",
                "",
                "**MEASURED verdict:** `PASS_CHORDIA`",
                "",
                "IS clears strict threshold 3.79 with N=1230 and ExpR=0.1149; OOS sign matches at N_OOS=79.",
                "",
                "**MEASURED theory mode:** `NO_THEORY_GRANT`",
                "**MEASURED threshold applied:** `3.79`",
                "**MEASURED loader has_theory:** `False`",
                "",
                "| IS | 1547 | 1230 | 79.51% | 9 | 0 | 0.1149 | 0.0913 | 0.1217 | 4.269 | 0.00002 |",
            ]
        ),
        encoding="utf-8",
    )

    parsed = cef.parse_result_md(md)
    proposal = cef.build_audit_log_proposals([parsed], audit_date="2026-05-31")

    assert proposal[0].strategy_id == "MNQ_TEST"
    assert proposal[0].verdict == "PASS_CHORDIA"
    assert proposal[0].t_stat == 4.269
    assert proposal[0].sample_size == 1230
    assert proposal[0].has_theory is False


def test_audit_log_proposal_maps_runner_fail_strict_token_to_canonical_verdict(tmp_path: Path) -> None:
    md = tmp_path / "result.md"
    md.write_text(
        "\n".join(
            [
                "# Chordia strict unlock audit - MNQ_TEST",
                "",
                "**MEASURED verdict:** `FAIL_STRICT_CHORDIA`",
                "",
                "IS t=3.652 < 3.79.",
                "",
                "**MEASURED theory mode:** `UNSUPPORTED`",
                "**MEASURED threshold applied:** `3.79`",
                "**MEASURED loader has_theory:** `False`",
                "",
                "| IS | 1547 | 1547 | 100.00% | 144 | 0 | 0.0880 | 0.0880 | 0.0930 | 3.652 | 0.00026 |",
            ]
        ),
        encoding="utf-8",
    )

    parsed = cef.parse_result_md(md)
    proposal = cef.build_audit_log_proposals([parsed], audit_date="2026-05-31")

    assert parsed.verdict == "FAIL_STRICT_CHORDIA"
    assert proposal[0].verdict == "FAIL_CHORDIA"


def test_audit_log_proposal_maps_runner_fail_strict_below_three_to_fail_both(tmp_path: Path) -> None:
    result = tmp_path / "result.md"
    result.write_text(
        "\n".join(
            [
                "# Chordia strict unlock audit - TEST_FAIL_BOTH",
                "",
                "**MEASURED verdict:** `FAIL_STRICT_CHORDIA`",
                "**MEASURED threshold:** `3.79`",
                "**MEASURED has_theory:** `False`",
                "**IS t-stat:** `2.965`",
                "**IS sample size:** `1188`",
                "**IS ExpR:** `0.0500`",
            ]
        ),
        encoding="utf-8",
    )

    parsed = cef.parse_result_md(result)
    proposal = cef.build_audit_log_proposals([parsed], audit_date="2026-05-31")

    assert parsed.verdict == "FAIL_STRICT_CHORDIA"
    assert proposal[0].verdict == "FAIL_BOTH"


def test_audit_log_proposal_maps_oos_underpowered_pass_to_park(tmp_path: Path) -> None:
    result = tmp_path / "result.md"
    result.write_text(
        "\n".join(
            [
                "# Chordia strict unlock audit - TEST_UNDERPOWERED",
                "",
                "**MEASURED verdict:** `PASS_CHORDIA_OOS_UNDERPOWERED`",
                "**MEASURED threshold:** `3.79`",
                "**MEASURED has_theory:** `False`",
                "**IS t-stat:** `4.102`",
                "**IS sample size:** `706`",
                "**IS ExpR:** `0.1662`",
            ]
        ),
        encoding="utf-8",
    )

    parsed = cef.parse_result_md(result)
    proposal = cef.build_audit_log_proposals([parsed], audit_date="2026-05-31")

    assert parsed.verdict == "PASS_CHORDIA_OOS_UNDERPOWERED"
    assert proposal[0].verdict == "PARK"


def test_limit_zero_means_full_replay_queue() -> None:
    rows = [
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15", rank=18),
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075", rank=19),
        _bench_row("MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER", rank=20, family_priority=5),
    ]

    plan = cef.plan_replay_work(rows, limit=0, max_family_priority=5, include_non_default_stop=False)

    assert [row.strategy_id for row in plan] == [
        "MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15",
        "MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075",
        "MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER",
    ]


def test_batch_shards_partition_full_queue_and_keep_blocked_visible() -> None:
    rows = [
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15", rank=18),
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S050", rank=19),
        _bench_row("MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER", rank=20),
    ]
    plan = cef.plan_replay_work(rows, limit=0, max_family_priority=0, include_non_default_stop=False)

    shards = cef.plan_batch_shards(plan, batch_size=2)

    assert [(shard.batch_id, shard.item_count, shard.ready_count, shard.blocked_count) for shard in shards] == [
        ("batch_001", 2, 1, 1),
        ("batch_002", 1, 1, 0),
    ]
    assert shards[0].first_rank == 18
    assert shards[0].last_rank == 19


def test_write_factory_artifacts_emits_batch_files(tmp_path: Path) -> None:
    rows = [
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15", rank=18),
        _bench_row("MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075", rank=19),
        _bench_row("MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER", rank=20),
    ]

    cef.write_factory_artifacts(
        rows,
        output_dir=tmp_path,
        today="2026-05-31",
        limit=0,
        max_family_priority=0,
        batch_size=2,
    )

    assert (tmp_path / "batches" / "batch_001.csv").exists()
    assert (tmp_path / "batches" / "batch_001_commands.ps1").exists()
    commands = (tmp_path / "batches" / "batch_001_commands.ps1").read_text(encoding="utf-8")
    assert "MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15" in commands
    assert "MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15_S075" in commands
    assert '"batch_count": 2' in (tmp_path / "manifest.json").read_text(encoding="utf-8")
