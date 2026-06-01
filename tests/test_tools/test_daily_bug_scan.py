from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.tools import daily_bug_scan


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(root: Path, rel_path: str, text: str, message: str) -> str:
    _write(root / rel_path, text)
    _git(root, "add", rel_path)
    env = {
        **dict(),
        **{
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    }
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _commit(root, "README.md", "# repo\n", "init docs")
    return root


def test_resolve_window_prefers_explicit_since() -> None:
    packet = daily_bug_scan.resolve_window(
        since="2026-05-31T00:00:00Z",
        hours=24,
        now=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
    )
    assert packet.source == "explicit-since"
    assert packet.since_iso == "2026-05-31T00:00:00+00:00"


def test_resolve_window_falls_back_to_hours() -> None:
    packet = daily_bug_scan.resolve_window(
        since=None,
        hours=24,
        now=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
    )
    assert packet.source == "hours-fallback"
    assert packet.since_iso == "2026-05-30T12:00:00+00:00"


def test_build_scan_skips_doc_only_and_keeps_code_commit(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc_sha = _commit(git_repo, "docs/note.md", "doc\n", "docs only")
    code_sha = _commit(git_repo, "scripts/tools/example.py", "print('x')\n", "code change")
    _git(git_repo, "update-ref", "refs/remotes/origin/main", code_sha)
    monkeypatch.setattr(
        daily_bug_scan, "_verification_status", lambda _root: daily_bug_scan.VerificationStatus("static_only", "test")
    )

    packet = daily_bug_scan.build_scan_packet(
        root=git_repo,
        since="2000-01-01T00:00:00Z",
        base_ref="origin/main",
    )

    assert packet.scanned_commits
    assert any(item.sha == doc_sha for item in packet.skipped_commits)
    assert any(item.sha == code_sha for item in packet.candidate_commits)


def test_merge_commit_with_production_paths_is_candidate(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_sha = _commit(git_repo, "docs/base.md", "base\n", "base")
    _git(git_repo, "checkout", "-b", "feature")
    _commit(git_repo, "trading_app/live/session_orchestrator.py", "VALUE = 1\n", "live path")
    _git(git_repo, "checkout", "main")
    _commit(git_repo, "docs/main.md", "main\n", "main docs")
    _git(git_repo, "merge", "--no-ff", "feature", "-m", "merge feature")
    merge_sha = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "update-ref", "refs/remotes/origin/main", merge_sha)
    monkeypatch.setattr(
        daily_bug_scan, "_verification_status", lambda _root: daily_bug_scan.VerificationStatus("static_only", "test")
    )

    packet = daily_bug_scan.build_scan_packet(
        root=git_repo,
        since="2000-01-01T00:00:00Z",
        base_ref="origin/main",
    )

    assert base_sha in packet.scanned_commits
    candidate = next(item for item in packet.candidate_commits if item.sha == merge_sha)
    assert "trading_app/live/session_orchestrator.py" in candidate.touched_code_paths


def test_working_tree_changes_are_local_candidates(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_sha = _commit(git_repo, "scripts/tools/base.py", "print('base')\n", "base code")
    _git(git_repo, "update-ref", "refs/remotes/origin/main", base_sha)
    _write(git_repo / "scripts/tools/base.py", "print('changed')\n")
    _write(git_repo / "scripts/tools/new_tool.py", "print('new')\n")
    monkeypatch.setattr(
        daily_bug_scan, "_verification_status", lambda _root: daily_bug_scan.VerificationStatus("static_only", "test")
    )

    packet = daily_bug_scan.build_scan_packet(
        root=git_repo,
        since="2000-01-01T00:00:00Z",
        base_ref="origin/main",
        include_working_tree=True,
    )

    local = [item for item in packet.candidate_commits if item.source == "working_tree"]
    assert len(local) == 1
    assert local[0].local_only is True
    assert set(local[0].touched_code_paths) >= {"scripts/tools/base.py", "scripts/tools/new_tool.py"}


def test_truncation_reports_omitted_candidates(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for idx in range(3):
        _commit(git_repo, f"scripts/tools/tool_{idx}.py", f"print({idx})\n", f"code {idx}")
    head = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "update-ref", "refs/remotes/origin/main", head)
    monkeypatch.setattr(
        daily_bug_scan, "_verification_status", lambda _root: daily_bug_scan.VerificationStatus("static_only", "test")
    )

    packet = daily_bug_scan.build_scan_packet(
        root=git_repo,
        since="2000-01-01T00:00:00Z",
        base_ref="origin/main",
        max_commits=1,
    )

    assert packet.total_candidate_count == 3
    assert packet.omitted_candidate_count == 2
    assert packet.risk_reason


def test_local_head_commit_included_only_when_requested(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_sha = _commit(git_repo, "scripts/tools/base.py", "print('base')\n", "base code")
    _git(git_repo, "update-ref", "refs/remotes/origin/main", base_sha)
    local_sha = _commit(git_repo, "scripts/tools/local_only.py", "print('local')\n", "local only code")
    monkeypatch.setattr(
        daily_bug_scan, "_verification_status", lambda _root: daily_bug_scan.VerificationStatus("static_only", "test")
    )

    packet_without = daily_bug_scan.build_scan_packet(
        root=git_repo,
        since="2000-01-01T00:00:00Z",
        base_ref="origin/main",
        include_local_head=False,
    )
    packet_with = daily_bug_scan.build_scan_packet(
        root=git_repo,
        since="2000-01-01T00:00:00Z",
        base_ref="origin/main",
        include_local_head=True,
    )

    assert all(item.sha != local_sha for item in packet_without.candidate_commits)
    local_entry = next(item for item in packet_with.candidate_commits if item.sha == local_sha)
    assert local_entry.local_only is True
    assert local_entry.source == "local_head"


def test_verification_mode_static_only_when_repo_python_missing(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-b", "main")
    status = daily_bug_scan._verification_status(tmp_path)
    assert status.mode in {"static_only", "blocked"}


def test_render_json_contains_stable_top_level_keys(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code_sha = _commit(git_repo, "scripts/tools/example.py", "print('x')\n", "code change")
    _git(git_repo, "update-ref", "refs/remotes/origin/main", code_sha)
    monkeypatch.setattr(
        daily_bug_scan, "_verification_status", lambda _root: daily_bug_scan.VerificationStatus("static_only", "test")
    )

    packet = daily_bug_scan.build_scan_packet(
        root=git_repo,
        since="2000-01-01T00:00:00Z",
        base_ref="origin/main",
    )
    payload = json.loads(daily_bug_scan.render_json(packet))

    assert set(payload) >= {
        "window",
        "git_context",
        "verification",
        "scanned_commits",
        "candidate_commits",
        "review_next",
        "total_candidate_count",
        "omitted_candidate_count",
        "risk_reason",
    }
