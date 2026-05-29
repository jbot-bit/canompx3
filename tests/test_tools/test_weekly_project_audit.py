from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.tools import weekly_project_audit


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], SimpleNamespace]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, cmd: list[str], *, cwd: Path, timeout: int = 30) -> SimpleNamespace:
        key = tuple(cmd)
        self.calls.append(key)
        if key not in self.responses:
            raise AssertionError(f"unexpected command: {cmd}")
        return self.responses[key]


def _base_responses(root: Path) -> dict[tuple[str, ...], SimpleNamespace]:
    pulse = {
        "items": [
            {
                "category": "broken",
                "severity": "high",
                "source": "criterion11",
                "summary": "BLOCKED: no Criterion 11 survival report",
                "action": "python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto",
            }
        ],
        "recommendation": "refresh control state",
        "counts": {"broken": 1},
        "survival_summary": {"gate_ok": False, "reason": "missing"},
        "sr_summary": None,
    }
    prs = [
        {
            "number": 314,
            "title": "validator writer retry",
            "url": "https://example.test/pr/314",
            "headRefName": "fix/validator-writer-retry-f1",
            "statusCheckRollup": [{"name": "checks", "conclusion": "FAILURE", "status": "COMPLETED"}],
        }
    ]
    pr_detail = {
        "number": 314,
        "title": "validator writer retry",
        "body": "Long PR body that should not be forwarded into the compact evidence packet.",
        "url": "https://example.test/pr/314",
        "mergeStateStatus": "UNKNOWN",
        "files": [{"path": "trading_app/validator.py"}],
        "commits": [{"oid": "abc123"}],
        "reviews": [{"state": "CHANGES_REQUESTED"}],
        "statusCheckRollup": [{"name": "checks", "conclusion": "FAILURE", "status": "COMPLETED"}],
    }
    issues: list[dict] = []
    return {
        ("git", "rev-parse", "--show-toplevel"): _result(stdout=f"{root}\n"),
        ("git", "remote", "get-url", "origin"): _result(stdout="https://github.com/jbot-bit/canompx3.git\n"),
        ("git", "status", "--short", "--branch"): _result(stdout="## HEAD (no branch)\n M foo.py\n"),
        ("git", "branch", "--show-current"): _result(stdout="\n"),
        ("git", "rev-parse", "HEAD"): _result(stdout="abc123\n"),
        ("git", "log", "--oneline", "-20"): _result(stdout="abc123 latest\n"),
        ("git", "worktree", "list", "--porcelain"): _result(stdout="worktree C:/repo\nHEAD abc123\n"),
        ("git", "stash", "list", "--date=local"): _result(stdout="stash@{0}: On main: carryover\n"),
        (
            "python",
            "scripts/tools/project_pulse.py",
            "--fast",
            "--format",
            "json",
        ): _result(stdout=json.dumps(pulse)),
        ("python", "scripts/tools/work_queue.py", "status"): _result(stdout="Open items: 1\n"),
        (
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "20",
            "--json",
            "number,title,headRefName,baseRefName,isDraft,updatedAt,statusCheckRollup,url",
        ): _result(stdout=json.dumps(prs)),
        (
            "gh",
            "pr",
            "view",
            "314",
            "--json",
            "number,title,url,files,commits,reviews,mergeStateStatus,statusCheckRollup",
        ): _result(stdout=json.dumps(pr_detail)),
        (
            "gh",
            "issue",
            "list",
            "--state",
            "open",
            "--limit",
            "20",
            "--json",
            "number,title,labels,updatedAt,url",
        ): _result(stdout=json.dumps(issues)),
        ("gh", "api", "repos/jbot-bit/canompx3/branches/main/protection"): _result(
            stdout=json.dumps(
                {"required_status_checks": {"contexts": ["checks"]}, "enforce_admins": {"enabled": False}}
            )
        ),
        (
            "gh",
            "api",
            "repos/jbot-bit/canompx3/code-scanning/alerts?state=open",
            "--jq",
            "length",
        ): _result(stdout="0\n"),
        (
            "gh",
            "api",
            "repos/jbot-bit/canompx3/dependabot/alerts?state=open",
            "--jq",
            "length",
        ): _result(stdout="0\n"),
        (
            "gh",
            "api",
            "repos/jbot-bit/canompx3/secret-scanning/alerts?state=open",
            "--jq",
            "length",
        ): _result(stdout="1\n"),
    }


def test_json_report_has_stable_top_level_schema(tmp_path: Path) -> None:
    runner = FakeRunner(_base_responses(tmp_path))

    report = weekly_project_audit.build_report(root=tmp_path, runner=runner)

    assert list(report) == [
        "generated_at",
        "repo",
        "git",
        "prs",
        "ci",
        "live_readiness",
        "security",
        "workflow",
        "carryovers",
        "recommended_attention_inputs",
    ]
    assert report["repo"]["slug"] == "jbot-bit/canompx3"
    assert report["git"]["dirty_count"] == 1
    assert report["ci"]["open_pr_failures"][0]["number"] == 314
    assert report["security"]["secret_scanning"]["open_alert_count"] == 1
    assert report["live_readiness"]["pulse_broken_count"] == 1


def test_markdown_report_contains_required_weekly_sections(tmp_path: Path) -> None:
    runner = FakeRunner(_base_responses(tmp_path))
    report = weekly_project_audit.build_report(root=tmp_path, runner=runner)

    markdown = weekly_project_audit.render_markdown(report)

    assert "# Weekly Project Improvement Audit Evidence" in markdown
    assert "## CI And PRs" in markdown
    assert "## Live Readiness" in markdown
    assert "## Security" in markdown
    assert "## Workflow And Carryovers" in markdown
    assert "## Recommended Attention Inputs" in markdown
    assert "PR #314" in markdown
    assert "workflow_security" in markdown


def test_open_pr_details_are_collected_for_review_context(tmp_path: Path) -> None:
    runner = FakeRunner(_base_responses(tmp_path))

    report = weekly_project_audit.build_report(root=tmp_path, runner=runner)

    assert report["prs"]["details"][0]["number"] == 314
    assert report["prs"]["details"][0]["status"] == "ok"
    assert report["prs"]["details"][0]["file_count"] == 1
    assert report["prs"]["details"][0]["file_paths"] == ["trading_app/validator.py"]
    assert "body" not in report["prs"]["details"][0]


def test_pr_details_are_bounded_for_token_efficiency(tmp_path: Path) -> None:
    responses = _base_responses(tmp_path)
    detail = json.loads(
        responses[
            (
                "gh",
                "pr",
                "view",
                "314",
                "--json",
                "number,title,url,files,commits,reviews,mergeStateStatus,statusCheckRollup",
            )
        ].stdout
    )
    detail["files"] = [{"path": f"file_{index}.py"} for index in range(25)]
    detail["commits"] = [{"oid": f"sha{index}"} for index in range(7)]
    responses[
        (
            "gh",
            "pr",
            "view",
            "314",
            "--json",
            "number,title,url,files,commits,reviews,mergeStateStatus,statusCheckRollup",
        )
    ] = _result(stdout=json.dumps(detail))
    runner = FakeRunner(responses)

    report = weekly_project_audit.build_report(root=tmp_path, runner=runner)
    pr_detail = report["prs"]["details"][0]

    assert pr_detail["file_count"] == 25
    assert len(pr_detail["file_paths"]) == 20
    assert pr_detail["file_paths_truncated"] is True
    assert pr_detail["commit_count"] == 7


def test_markdown_compacts_multiline_endpoint_errors(tmp_path: Path) -> None:
    responses = _base_responses(tmp_path)
    multiline_error = "gh: no analysis found (HTTP 404)\ngh: refresh with admin:repo_hook scope"
    responses[
        (
            "gh",
            "api",
            "repos/jbot-bit/canompx3/code-scanning/alerts?state=open",
            "--jq",
            "length",
        )
    ] = _result(returncode=1, stderr=multiline_error)
    runner = FakeRunner(responses)

    report = weekly_project_audit.build_report(root=tmp_path, runner=runner)
    markdown = weekly_project_audit.render_markdown(report)

    assert multiline_error in report["security"]["code_scanning"]["error"]
    assert "admin:repo_hook" not in markdown
    assert "gh: no analysis found (HTTP 404)" in markdown


def test_workflow_security_summary_flags_missing_guardrails(tmp_path: Path) -> None:
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ci.yml").write_text(
        """
name: CI
on: [pull_request]
jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: uv run pip-audit --desc on
        continue-on-error: true
""",
        encoding="utf-8",
    )
    runner = FakeRunner(_base_responses(tmp_path))

    report = weekly_project_audit.build_report(root=tmp_path, runner=runner)
    workflow_security = report["security"]["workflow_security"]

    assert workflow_security["workflow_count"] == 1
    assert workflow_security["has_top_level_permissions"] is False
    assert workflow_security["uses_dependency_review_action"] is False
    assert workflow_security["uses_codeql_action"] is False
    assert workflow_security["advisory_security_steps"][0]["tool"] == "pip-audit"
    assert workflow_security["advisory_security_steps"][0]["continue_on_error"] is True


def test_optional_github_endpoint_failure_is_reported_as_unknown(tmp_path: Path) -> None:
    responses = _base_responses(tmp_path)
    responses[
        (
            "gh",
            "api",
            "repos/jbot-bit/canompx3/code-scanning/alerts?state=open",
            "--jq",
            "length",
        )
    ] = _result(returncode=1, stderr="no analysis found")
    runner = FakeRunner(responses)

    report = weekly_project_audit.build_report(root=tmp_path, runner=runner)

    assert report["security"]["code_scanning"]["status"] == "unknown"
    assert report["security"]["code_scanning"]["endpoint"] == "code-scanning/alerts?state=open"
    assert "no analysis found" in report["security"]["code_scanning"]["error"]


def test_since_flag_changes_git_log_window(tmp_path: Path) -> None:
    responses = _base_responses(tmp_path)
    responses.pop(("git", "log", "--oneline", "-20"))
    responses[("git", "log", "--oneline", "--since=2026-05-01T00:00:00Z")] = _result(stdout="def456 scoped\n")
    runner = FakeRunner(responses)

    report = weekly_project_audit.build_report(
        root=tmp_path,
        runner=runner,
        since="2026-05-01T00:00:00Z",
    )

    assert report["git"]["recent_commits"] == ["def456 scoped"]


def test_default_collection_does_not_write_repo_files(tmp_path: Path) -> None:
    marker = tmp_path / "tracked.txt"
    marker.write_text("before", encoding="utf-8")
    runner = FakeRunner(_base_responses(tmp_path))

    weekly_project_audit.build_report(root=tmp_path, runner=runner)

    assert marker.read_text(encoding="utf-8") == "before"
    assert not (tmp_path / "weekly_project_audit.json").exists()


def test_main_writes_only_when_out_is_explicit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "packet.json"
    runner = FakeRunner(_base_responses(tmp_path))

    rc = weekly_project_audit.main(["--format", "json", "--out", str(out)], runner=runner, root=tmp_path)

    assert rc == 0
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["repo"]["slug"] == "jbot-bit/canompx3"
    assert str(out) in capsys.readouterr().out
