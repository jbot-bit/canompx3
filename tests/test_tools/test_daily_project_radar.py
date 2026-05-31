from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from scripts.tools import daily_bug_scan, daily_project_radar


def _scan_packet(
    *,
    candidates: list[daily_bug_scan.CandidateCommit] | None = None,
    skipped: list[daily_bug_scan.SkippedCommit] | None = None,
    verification_mode: str = "full",
) -> daily_bug_scan.ScanPacket:
    return daily_bug_scan.ScanPacket(
        generated_at="2026-05-31T00:00:00+00:00",
        window=daily_bug_scan.ScanWindow("2026-05-30T00:00:00+00:00", "test", 24),
        git_context={"head": "abc1234", "branch": "main", "detached": False, "dirty": False, "base_ref": "origin/main"},
        verification=daily_bug_scan.VerificationStatus(verification_mode, "test"),
        scanned_commits=["abc1234"],
        skipped_commits=skipped or [],
        candidate_commits=candidates or [],
        review_next=[item.sha for item in (candidates or [])],
        total_candidate_count=len(candidates or []),
        omitted_candidate_count=0,
        risk_reason=[],
    )


def test_risk_lane_cannot_clear_without_false_negative_sample(tmp_path: Path) -> None:
    report = daily_project_radar.build_daily_radar(
        root=tmp_path,
        lane="risk",
        scan_packet=_scan_packet(),
    )

    risk = report["risk"]
    assert risk["verdict"] == "VERIFY_MORE"
    assert "false_negative_sample" in risk["audit_of_auditor"]
    assert risk["audit_of_auditor"]["unchecked_scope"]
    assert risk["audit_of_auditor"]["what_would_falsify_verdict"]


def test_capital_audit_runs_only_for_capital_paths(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_capital_audit(**_kwargs):
        calls.append("called")
        return {"verdict": "VERIFY_MORE", "blockers": [], "framing_defects": ["test"], "warnings": []}

    monkeypatch.setattr(daily_project_radar, "_build_capital_hard_audit", fake_capital_audit)
    non_capital = daily_bug_scan.CandidateCommit(
        sha="tool",
        committed_at=datetime.now(UTC).isoformat(),
        subject="tooling",
        touched_code_paths=["scripts/tools/daily_bug_scan.py"],
        touched_test_paths=[],
        diff_stats={"files": 1, "added": 1, "deleted": 0},
    )
    capital = daily_bug_scan.CandidateCommit(
        sha="live",
        committed_at=datetime.now(UTC).isoformat(),
        subject="live",
        touched_code_paths=["trading_app/live/session_orchestrator.py"],
        touched_test_paths=[],
        diff_stats={"files": 1, "added": 1, "deleted": 0},
    )

    daily_project_radar.build_daily_radar(
        root=tmp_path, lane="risk", scan_packet=_scan_packet(candidates=[non_capital])
    )
    assert calls == []

    daily_project_radar.build_daily_radar(root=tmp_path, lane="risk", scan_packet=_scan_packet(candidates=[capital]))
    assert calls == ["called"]


def test_targeted_sentinel_flags_hardcoded_windows_queue_command(tmp_path: Path) -> None:
    target = tmp_path / "scripts" / "tools" / "pulse.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "COMMAND = '.\\\\.venv\\\\Scripts\\\\python.exe scripts\\\\tools\\\\work_queue.py'\n", encoding="utf-8"
    )

    findings = daily_project_radar.targeted_behavioral_sentinels(tmp_path, ["scripts/tools/pulse.py"])

    assert any("hardcoded repo python" in finding["summary"].lower() for finding in findings)


def test_community_idea_cannot_be_promoted_to_research_truth() -> None:
    card = daily_project_radar.classify_idea(
        {
            "source_type": "community",
            "source": "reddit:/r/futurestrading",
            "claim": "Use opening imbalance as an ORB filter",
            "mechanism": "liquidity shock at session open",
        }
    )

    assert card["disposition"] == "PARK"
    assert "unofficial" in card["rationale"].lower()


def test_known_no_go_idea_rejects() -> None:
    card = daily_project_radar.classify_idea(
        {
            "source_type": "official",
            "source": "example",
            "claim": "Retest NR7 as a daily ORB filter",
            "mechanism": "range contraction",
        }
    )

    assert card["disposition"] == "REJECT"
    assert "no-go" in card["rationale"].lower()


def test_missing_mechanism_parks() -> None:
    card = daily_project_radar.classify_idea(
        {
            "source_type": "literature",
            "source": "arxiv",
            "claim": "Try a new futures classifier",
        }
    )

    assert card["disposition"] == "PARK"
    assert "mechanism" in card["rationale"].lower()


def test_all_lane_includes_ai_tooling_manifest_and_lane_audits(tmp_path: Path) -> None:
    skipped = daily_bug_scan.SkippedCommit(
        sha="doc",
        committed_at=datetime.now(UTC).isoformat(),
        subject="docs only",
        reason="doc-only/no production code",
        touched_paths=["docs/example.md"],
    )

    report = daily_project_radar.build_daily_radar(
        root=tmp_path,
        lane="all",
        scan_packet=_scan_packet(skipped=[skipped]),
        external_items=[],
        ai_tooling_items=[
            {
                "source_type": "official",
                "vendor": "openai",
                "source_url": "https://developers.openai.com/api/docs/changelog",
                "claim": "Responses API adds a compact endpoint for context reduction",
                "local_repo_touchpoints": ["scripts/tools"],
                "role": "context_hygiene",
            }
        ],
    )

    assert "lane_manifest" in report
    assert "ai_tooling" in report
    assert report["lane_manifest"]["lanes"]["ai_tooling"]["included"] is True
    assert report["risk"]["lane_audit"]["counter_framings"]
    assert report["opportunity"]["lane_audit"]["counter_framings"]
    assert report["ai_tooling"]["lane_audit"]["counter_framings"]


def test_all_lane_does_not_run_capital_audit_for_tooling_only_changes(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_capital_audit(**_kwargs):
        calls.append("called")
        return {"verdict": "BLOCK"}

    monkeypatch.setattr(daily_project_radar, "_build_capital_hard_audit", fake_capital_audit)
    tooling_candidate = daily_bug_scan.CandidateCommit(
        sha="tool",
        committed_at=datetime.now(UTC).isoformat(),
        subject="tooling",
        touched_code_paths=["scripts/tools/ai_tooling_leverage.py"],
        touched_test_paths=["tests/test_tools/test_ai_tooling_leverage.py"],
        diff_stats={"files": 2, "added": 10, "deleted": 0},
    )

    report = daily_project_radar.build_daily_radar(
        root=tmp_path,
        lane="all",
        scan_packet=_scan_packet(candidates=[tooling_candidate]),
        ai_tooling_items=[],
    )

    assert calls == []
    assert report["risk"]["capital_audit"]["status"] == "skipped"
