from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.tools import project_improvement_review as review


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _config(root: Path, **kwargs) -> review.ReviewConfig:
    return review.ReviewConfig(root=root, since_ref="HEAD~1", **kwargs)


def _clean_git(monkeypatch: pytest.MonkeyPatch, *, staged=(), dirty=(), untracked=(), changed=()) -> None:
    def fake_git(root: Path, args: list[str]) -> tuple[int, str, str]:
        if args == ["symbolic-ref", "--short", "HEAD"]:
            return 0, "main", ""
        if args == ["status", "--porcelain=v1"]:
            lines = [f"A  {item}" for item in staged]
            lines.extend(f" M {item}" for item in dirty)
            lines.extend(f"?? {item}" for item in untracked)
            return 0, "\n".join(lines), ""
        if args[:2] == ["diff", "--name-only"]:
            return 0, "\n".join(changed), ""
        if args == ["worktree", "list", "--porcelain"]:
            return 0, f"worktree {root}\nHEAD abc123\nbranch refs/heads/main\n", ""
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(review, "_run_git", fake_git)


def test_stdout_markdown_renders_fixed_categories_and_highest_action(tmp_path, monkeypatch, capsys) -> None:
    _write(tmp_path / "CLAUDE.md", "Project operating contract with threshold ≤ 1.")
    _clean_git(monkeypatch, changed=("CLAUDE.md",))
    monkeypatch.setattr(review, "PROJECT_ROOT", tmp_path)

    assert review.main([]) == 0

    out = capsys.readouterr().out
    assert "# Project Improvement Review" in out
    for category in review.CATEGORIES:
        assert f"## {category}" in out
    assert out.count("## Highest-EV next action") == 1


def test_out_writes_only_under_project_reviews(tmp_path) -> None:
    allowed = review._resolve_output_path(tmp_path, "docs/runtime/project_reviews/report.md")
    assert allowed == (tmp_path / "docs/runtime/project_reviews/report.md").resolve()

    with pytest.raises(argparse.ArgumentTypeError):
        review._resolve_output_path(tmp_path, "docs/runtime/project_reviews/../escape.md")

    with pytest.raises(argparse.ArgumentTypeError):
        review._resolve_output_path(tmp_path, "docs/runtime/project_reviews/report.txt")


def test_tool_does_not_import_or_open_duckdb_and_has_no_mutating_protected_calls() -> None:
    source = Path(review.__file__).read_text(encoding="utf-8")
    assert "import duckdb" not in source
    assert "duckdb.connect(" not in source
    assert ".upsert(" not in source
    assert ".delete(" not in source
    assert "apply_allocation" not in source
    assert "write_live_config" not in source


def test_flags_staged_handoff(tmp_path, monkeypatch) -> None:
    _write(tmp_path / "HANDOFF.md", "baton")
    _clean_git(monkeypatch, staged=("HANDOFF.md",), changed=("HANDOFF.md",))

    report = review.review(_config(tmp_path))

    assert any(item.evidence_path == "HANDOFF.md" and item.severity == "HIGH" for item in report.findings)


def test_detached_or_git_failure_reports_blocked_context(tmp_path, monkeypatch) -> None:
    def fake_git(root: Path, args: list[str]) -> tuple[int, str, str]:
        if args == ["symbolic-ref", "--short", "HEAD"]:
            return 1, "", "fatal: ref HEAD is not a symbolic ref"
        if args == ["status", "--porcelain=v1"]:
            return 0, "", ""
        if args[:2] == ["diff", "--name-only"]:
            return 1, "", "bad revision"
        if args == ["worktree", "list", "--porcelain"]:
            return 0, "", ""
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(review, "_run_git", fake_git)

    report = review.review(_config(tmp_path))

    assert report.git_state.context_status == "UNKNOWN/BLOCKED CONTEXT"
    assert any(item.evidence_path == "git" for item in report.findings)


def test_flags_non_read_only_db_pattern_as_advisory_drift_route(tmp_path, monkeypatch) -> None:
    _write(tmp_path / "scripts/tools/risky_reader.py", "import duckdb\ncon = duckdb.connect(str(path))\n")
    _clean_git(monkeypatch, changed=("scripts/tools/risky_reader.py",))

    report = review.review(_config(tmp_path))

    finding = next(item for item in report.findings if item.evidence_path == "scripts/tools/risky_reader.py")
    assert finding.category == "Code quality"
    assert "check_drift.py" in finding.rationale
    assert "check_drift.py --fast" in finding.suggested_test


def test_flags_e2_lookahead_fixture_without_policy(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "research/e2_scan.py",
        "entry_model='E2'\nfeature = row['rel_vol_NYSE_OPEN']\n",
    )
    _clean_git(monkeypatch, changed=("research/e2_scan.py",))

    report = review.review(_config(tmp_path))

    assert any(item.category == "Research integrity" and "E2" in item.evidence_snippet for item in report.findings)


def test_e2_policy_annotation_suppresses_lookahead_fixture(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "research/e2_scan.py",
        "# e2-lookahead-policy: late-fill-only\nentry_model='E2'\nfeature = row['rel_vol_NYSE_OPEN']\n",
    )
    _clean_git(monkeypatch, changed=("research/e2_scan.py",))

    report = review.review(_config(tmp_path))

    assert not any("break-bar-derived predictor" in item.evidence_snippet for item in report.findings)


def test_flags_oos_tuning_against_decision_surface(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "docs/audit/results/oos_tuning.md",
        "We will tune the threshold against OOS to rescue this strategy for promotion.",
    )
    _clean_git(monkeypatch, changed=("docs/audit/results/oos_tuning.md",))

    report = review.review(_config(tmp_path))

    assert any(
        item.evidence_path == "docs/audit/results/oos_tuning.md" and item.severity == "BLOCKER"
        for item in report.findings
    )


def test_oos_no_tuning_policy_language_is_not_a_blocker(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "docs/audit/hypotheses/drafts/mgc_cpcv.md",
        (
            "No 2026 holdout tuning: CPCV does NOT tune against holdout data. "
            "No post-hoc threshold changes. Thresholds are read-only constants. "
            "K=1992 is carried forward as the honest selection budget."
        ),
    )
    _clean_git(monkeypatch, changed=("docs/audit/hypotheses/drafts/mgc_cpcv.md",))

    report = review.review(_config(tmp_path))

    assert not any(
        item.evidence_path == "docs/audit/hypotheses/drafts/mgc_cpcv.md"
        and item.category == "Research integrity"
        and item.severity == "BLOCKER"
        for item in report.findings
    )


def test_powered_oos_resweep_title_is_not_tuning_by_itself(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "research/powered_oos_resweep.py",
        (
            "# Powered-OOS graveyard re-sweep\n"
            "OOS_FRACTION = 0.30\n"
            "print('does NOT rescue any candidate to deployable status')\n"
        ),
    )
    _clean_git(monkeypatch, changed=("research/powered_oos_resweep.py",))

    report = review.review(_config(tmp_path))

    assert not any(
        item.evidence_path == "research/powered_oos_resweep.py"
        and item.category == "Research integrity"
        and item.severity == "BLOCKER"
        for item in report.findings
    )


def test_generated_project_review_report_is_not_rescanned(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "docs/runtime/project_reviews/report.md",
        "Historical report text: tune the threshold against OOS to rescue this strategy for promotion.",
    )
    _clean_git(monkeypatch, changed=("docs/runtime/project_reviews/report.md",))

    report = review.review(_config(tmp_path))

    assert not any(item.evidence_path == "docs/runtime/project_reviews/report.md" for item in report.findings)


def test_live_safe_doc_without_evidence_is_flagged(tmp_path, monkeypatch) -> None:
    _write(tmp_path / "docs/runtime/live.md", "This lane is LIVE_SAFE and ready to deploy.")
    _clean_git(monkeypatch, changed=("docs/runtime/live.md",))

    report = review.review(_config(tmp_path))

    assert any(item.category == "Source-of-truth integrity" for item in report.findings)


def test_oversized_file_is_scan_silence(tmp_path, monkeypatch) -> None:
    _write(tmp_path / "docs/runtime/large.md", "x" * 20)
    _clean_git(monkeypatch, changed=("docs/runtime/large.md",))

    report = review.review(_config(tmp_path, max_file_bytes=5))

    assert any(
        item.evidence_path == "docs/runtime/large.md" and "exceeds" in item.evidence_snippet for item in report.findings
    )


def test_highest_ev_prefers_blocker_over_lower_severity(tmp_path, monkeypatch) -> None:
    _write(tmp_path / "trading_app/prop_profiles.py", "x = 1")
    _write(tmp_path / "docs/runtime/live.md", "This is LIVE_SAFE.")
    _clean_git(monkeypatch, dirty=("trading_app/prop_profiles.py",), changed=("docs/runtime/live.md",))

    report = review.review(_config(tmp_path))

    assert report.highest_ev.severity == "BLOCKER"
    assert report.highest_ev.evidence_path == "git"


def test_protected_surface_mentions_need_nearby_mutation_language(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "trading_app/live_reader.py",
        "from trading_app import prop_profiles\n\ndef explain():\n    return 'read-only report'\n",
    )
    _clean_git(monkeypatch, changed=("trading_app/live_reader.py",))

    report = review.review(_config(tmp_path))

    assert not any(
        item.evidence_path == "trading_app/live_reader.py" and item.category == "Git/worktree hygiene"
        for item in report.findings
    )


def test_protected_surface_mutation_is_flagged_in_code(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "scripts/tools/mutate_live.py",
        "from trading_app import prop_profiles\n\ndef run():\n    write_live_config(prop_profiles.ACCOUNT_PROFILES)\n",
    )
    _clean_git(monkeypatch, changed=("scripts/tools/mutate_live.py",))

    report = review.review(_config(tmp_path))

    assert any(
        item.evidence_path == "scripts/tools/mutate_live.py" and item.category == "Git/worktree hygiene"
        for item in report.findings
    )


def test_report_only_docs_and_live_readiness_are_not_protected_mutation_findings(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "docs/runtime/action-queue.yaml",
        "next_action: Do not mutate docs/runtime/lane_allocation.json; emit report-only proposal.\n",
    )
    _write(
        tmp_path / "scripts/tools/live_readiness_report.py",
        "from trading_app import prop_profiles\nout_path.write_text('profile report')\n",
    )
    _clean_git(
        monkeypatch,
        changed=("docs/runtime/action-queue.yaml", "scripts/tools/live_readiness_report.py"),
    )

    report = review.review(_config(tmp_path))

    assert not any(
        item.evidence_path in {"docs/runtime/action-queue.yaml", "scripts/tools/live_readiness_report.py"}
        and item.category == "Git/worktree hygiene"
        for item in report.findings
    )


def test_protected_mutation_scan_ignores_test_fixtures(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path / "tests/test_tools/test_live_readiness_report.py",
        "fixture = 'write docs/runtime/lane_allocation.json in a fake profile test'\n",
    )
    _clean_git(monkeypatch, changed=("tests/test_tools/test_live_readiness_report.py",))

    report = review.review(_config(tmp_path))

    assert not any(item.evidence_path == "tests/test_tools/test_live_readiness_report.py" for item in report.findings)


def test_known_canonical_db_writer_paths_are_not_reported_as_reader_risks(tmp_path, monkeypatch) -> None:
    _write(tmp_path / "pipeline/init_db.py", "import duckdb\ncon = duckdb.connect(str(path))\n")
    _write(tmp_path / "pipeline/build_daily_features.py", "import duckdb\ncon = duckdb.connect(str(path))\n")
    _clean_git(monkeypatch, changed=("pipeline/init_db.py", "pipeline/build_daily_features.py"))

    report = review.review(_config(tmp_path))

    flagged = {item.evidence_path for item in report.findings if item.category == "Code quality"}
    assert "pipeline/init_db.py" not in flagged
    assert "pipeline/build_daily_features.py" not in flagged


def test_feature_branch_default_diff_uses_origin_main_merge_base(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_git(root: Path, args: list[str]) -> tuple[int, str, str]:
        calls.append(args)
        if args == ["symbolic-ref", "--short", "HEAD"]:
            return 0, "codex/example", ""
        if args == ["status", "--porcelain=v1"]:
            return 0, "", ""
        if args == ["merge-base", "origin/main", "HEAD"]:
            return 0, "abc123", ""
        if args == ["diff", "--name-only", "abc123..HEAD"]:
            return 0, "scripts/tools/project_improvement_review.py", ""
        if args == ["worktree", "list", "--porcelain"]:
            return 0, f"worktree {root}\nHEAD abc123\nbranch refs/heads/codex/example\n", ""
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(review, "_run_git", fake_git)

    state = review.gather_git_state(tmp_path)

    assert state.changed_since_ref == ("scripts/tools/project_improvement_review.py",)
    assert ["diff", "--name-only", "abc123..HEAD"] in calls


def test_parse_status_preserves_first_line_unstaged_path() -> None:
    staged, dirty, untracked = review._parse_status_porcelain(
        " M docs/runtime/project_reviews/report.md\nA  scripts/tools/new_tool.py\n?? tests/test_tools/test_new_tool.py"
    )

    assert staged == ("scripts/tools/new_tool.py",)
    assert dirty == ("docs/runtime/project_reviews/report.md",)
    assert untracked == ("tests/test_tools/test_new_tool.py",)
