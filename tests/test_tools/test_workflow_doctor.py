from __future__ import annotations

import json
import os
import socket
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.tools import workflow_doctor as wd


def test_json_contract_has_required_keys(tmp_path: Path) -> None:
    fake_report = {
        "git": {"hooks_path": ".githooks", "hooks_ok": True, "dirty_count": 0, "detached": False},
        "worktrees": {"count": 0, "items": []},
        "lease": {"status": "OK"},
        "db": {"status": "OK"},
        "ports": {"status": "OK"},
        "dashboard": {"runtime": {}, "planned": {}},
        "stages": {"active_count": 0},
        "hooks": {"status": "OK"},
        "async_hooks": {"status": "OK"},
        "integrations": {"status": "OK"},
        "launchers": {"status": "OK"},
        "drift": {"recommended_probe": "python scripts/tools/profile_check_drift.py"},
    }
    fake_report["blocks"] = wd.build_blocks(fake_report)
    fake_report["next"] = wd.choose_next(fake_report, "status")
    encoded = json.loads(json.dumps({key: fake_report.get(key) for key in wd.JSON_KEYS}))
    assert set(wd.JSON_KEYS) <= set(encoded)


def test_parse_git_worktree_porcelain_handles_main_detached_and_prunable() -> None:
    text = "\n".join(
        [
            "worktree C:/repo",
            "HEAD abc123",
            "branch refs/heads/main",
            "",
            "worktree C:/repo/.worktrees/tasks/codex/foo",
            "HEAD def456",
            "detached",
            "prunable gitdir file points to non-existent location",
            "",
        ]
    )
    items = wd.parse_git_worktree_porcelain(text)
    assert items[0]["branch"] == "main"
    assert items[0]["detached"] is False
    assert items[1]["detached"] is True
    assert items[1]["prunable"]


def test_worktree_collection_marks_managed_and_unmanaged(tmp_path: Path) -> None:
    main = tmp_path / "repo"
    managed = main / ".worktrees" / "tasks" / "codex" / "foo"
    unmanaged = tmp_path / "scratch"
    main.mkdir()
    managed.mkdir(parents=True)
    unmanaged.mkdir()
    (managed / ".canompx3-worktree.json").write_text('{"tool":"codex","name":"foo"}', encoding="utf-8")
    porcelain = f"worktree {main}\nHEAD aaa\nbranch refs/heads/main\n\nworktree {managed}\nHEAD bbb\nbranch refs/heads/wt-codex-foo\n\nworktree {unmanaged}\nHEAD ccc\nbranch refs/heads/other\n\n"

    def fake_run(args: list[str], *, cwd: Path, timeout: float = wd.COMMAND_TIMEOUT_SECONDS) -> dict:
        return {"ok": True, "returncode": 0, "stdout": porcelain, "stderr": "", "timed_out": False, "command": args}

    with patch.object(wd, "run_command", side_effect=fake_run):
        result = wd.collect_worktrees(main)
    assert result["count"] == 3
    assert result["managed_count"] == 1
    assert [item["managed"] for item in result["items"]] == [False, True, False]


def test_worktree_collection_marks_metadata_outside_local_managed_root(tmp_path: Path) -> None:
    main = tmp_path / "repo"
    external_managed = tmp_path / "external-managed"
    main.mkdir()
    external_managed.mkdir()
    (external_managed / ".canompx3-worktree.json").write_text(
        '{"tool":"codex","name":"foo","branch":"wt-codex-foo"}',
        encoding="utf-8",
    )
    porcelain = (
        f"worktree {main}\nHEAD aaa\nbranch refs/heads/main\n\n"
        f"worktree {external_managed}\nHEAD bbb\nbranch refs/heads/wt-codex-foo\n\n"
    )

    def fake_run(args: list[str], *, cwd: Path, timeout: float = wd.COMMAND_TIMEOUT_SECONDS) -> dict:
        return {"ok": True, "returncode": 0, "stdout": porcelain, "stderr": "", "timed_out": False, "command": args}

    with patch.object(wd, "run_command", side_effect=fake_run):
        result = wd.collect_worktrees(main)
    assert result["managed_count"] == 1
    assert result["items"][1]["managed"] is True
    assert result["items"][1]["metadata"] == {"tool": "codex", "name": "foo", "branch": "wt-codex-foo"}


def test_worktree_collection_ignores_stale_metadata_branch_mismatch(tmp_path: Path) -> None:
    main = tmp_path / "repo"
    stale = tmp_path / "stale"
    main.mkdir()
    stale.mkdir()
    (stale / ".canompx3-worktree.json").write_text(
        '{"tool":"claude","name":"old","branch":"wt-claude-old"}',
        encoding="utf-8",
    )
    porcelain = (
        f"worktree {main}\nHEAD aaa\nbranch refs/heads/main\n\n"
        f"worktree {stale}\nHEAD bbb\nbranch refs/heads/codex/current\n\n"
    )

    def fake_run(args: list[str], *, cwd: Path, timeout: float = wd.COMMAND_TIMEOUT_SECONDS) -> dict:
        return {"ok": True, "returncode": 0, "stdout": porcelain, "stderr": "", "timed_out": False, "command": args}

    with patch.object(wd, "run_command", side_effect=fake_run):
        result = wd.collect_worktrees(main)
    assert result["managed_count"] == 0
    assert result["items"][1]["managed"] is False
    assert result["items"][1]["metadata"] is None
    assert result["items"][1]["metadata_status"] is None
    assert result["items"][1]["metadata_ignored"]


def test_worktree_collection_labels_stale_metadata_inside_managed_root(tmp_path: Path) -> None:
    main = tmp_path / "repo"
    managed = main / ".worktrees" / "tasks" / "codex" / "foo"
    main.mkdir()
    managed.mkdir(parents=True)
    (managed / ".canompx3-worktree.json").write_text(
        '{"tool":"codex","name":"old","branch":"wt-codex-old"}',
        encoding="utf-8",
    )
    porcelain = (
        f"worktree {main}\nHEAD aaa\nbranch refs/heads/main\n\n"
        f"worktree {managed}\nHEAD bbb\nbranch refs/heads/wt-codex-current\n\n"
    )

    def fake_run(args: list[str], *, cwd: Path, timeout: float = wd.COMMAND_TIMEOUT_SECONDS) -> dict:
        return {"ok": True, "returncode": 0, "stdout": porcelain, "stderr": "", "timed_out": False, "command": args}

    with patch.object(wd, "run_command", side_effect=fake_run):
        result = wd.collect_worktrees(main)
    assert result["managed_count"] == 1
    assert result["items"][1]["managed"] is True
    assert result["items"][1]["metadata"] is None
    assert result["items"][1]["metadata_status"] == "stale_branch_mismatch"


def test_lease_dead_ppid_with_fresh_peer_heartbeat_is_not_ok() -> None:
    snap = {
        "lease_present": True,
        "current_is_holder": False,
        "peer_live": True,
        "holder_ppid_alive": False,
        "fresh_peer_heartbeat": True,
    }
    assert wd.classify_lease(snap) == "BLOCK"


def test_lease_dead_ppid_without_live_peer_is_warn_not_ok() -> None:
    snap = {
        "lease_present": True,
        "current_is_holder": False,
        "peer_live": False,
        "holder_ppid_alive": False,
        "fresh_peer_heartbeat": False,
    }
    assert wd.classify_lease(snap) == "WARN"


def test_probe_port_in_use() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert wd.probe_port("127.0.0.1", port) is True


def test_db_readonly_path_and_deprecated_classification(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "gold.db"
    con = duckdb.connect(str(db_path))
    con.execute("create table t(x integer)")
    con.close()
    assert wd.check_db_readonly(db_path)["status"] == "OK"
    assert wd.is_deprecated_scratch_db("C:/db/gold.db") is True
    assert wd.is_deprecated_scratch_db("/mnt/c/db/gold.db") is True
    assert wd.is_deprecated_scratch_db(tmp_path / "gold.db") is False


def test_build_blocks_reports_hook_path_unset() -> None:
    report = {
        "git": {"dirty_count": 0, "detached": False, "hooks_ok": False},
        "lease": {"status": "OK"},
        "db": {"status": "OK"},
        "dashboard": {"runtime": {}, "port_listening": False},
        "stages": {"active_count": 0},
    }
    blocks = wd.build_blocks(report)
    assert any(block["code"] == "hook_path" and block["status"] == "BLOCK" for block in blocks)


def test_run_command_scrubs_git_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen_env = {}

    def fake_run(*args, **kwargs):
        seen_env.update(kwargs.get("env") or {})

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setenv("GIT_DIR", "wrong")
    monkeypatch.setenv("GIT_WORK_TREE", "wrong")
    monkeypatch.setenv("GIT_INDEX_FILE", "wrong")
    with patch.object(wd.subprocess, "run", side_effect=fake_run):
        result = wd.run_command(["git", "status"], cwd=tmp_path)
    assert result["ok"] is True
    assert "GIT_DIR" not in seen_env
    assert "GIT_WORK_TREE" not in seen_env
    assert "GIT_INDEX_FILE" not in seen_env


def test_collect_integrations_reports_mcp_and_plugins(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"gold-db": {}, "repo-state": {}}}),
        encoding="utf-8",
    )
    (tmp_path / ".claude" / "settings.json").write_text(
        json.dumps({"enabledPlugins": ["context7"], "disabledPlugins": ["firecrawl"]}),
        encoding="utf-8",
    )
    result = wd.collect_integrations(tmp_path)
    assert result["status"] == "WARN"
    assert result["mcp_server_count"] == 2
    assert result["mcp_servers"] == ["gold-db", "repo-state"]
    assert result["enabled_plugins"] == ["context7"]
    assert result["disabled_plugins"] == ["firecrawl"]


def test_collect_launchers_reports_presence(tmp_path: Path) -> None:
    (tmp_path / "START_BOT.bat").write_text("@echo off\n", encoding="utf-8")
    (tmp_path / "scripts" / "tools").mkdir(parents=True)
    (tmp_path / "scripts" / "tools" / "new_session.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    result = wd.collect_launchers(tmp_path)
    assert result["status"] == "WARN"
    assert result["items"]["START_BOT"]["exists"] is True
    assert result["items"]["START_WORKTREE"]["exists"] is False
    assert result["items"]["new_session"]["exists"] is True


def test_choose_next_prioritizes_peer_lease_over_dirty_tree() -> None:
    report = {
        "blocks": [
            {"status": "BLOCK", "code": "dirty_tree", "message": "dirty"},
            {"status": "BLOCK", "code": "peer_lease", "message": "peer"},
        ],
        "launchers": {
            "items": {
                "START_WORKTREE": {"exists": True},
                "new_session": {"exists": True},
            }
        },
        "drift": {"recommended_probe": "python scripts/tools/profile_check_drift.py"},
    }
    result = wd.choose_next(report, "status")
    assert result["command"] == "START_WORKTREE.bat <descriptor>"
    assert "force" not in result["reason"].lower()


def test_choose_next_peer_lease_uses_bash_worktree_fallback() -> None:
    report = {
        "blocks": [{"status": "BLOCK", "code": "peer_lease", "message": "peer"}],
        "launchers": {
            "items": {
                "START_WORKTREE": {"exists": False},
                "new_session": {"exists": True},
            }
        },
        "drift": {"recommended_probe": "python scripts/tools/profile_check_drift.py"},
    }
    result = wd.choose_next(report, "status")
    assert result["command"] == "scripts/tools/new_session.sh <descriptor> && cd ../canompx3-<descriptor>"


def test_choose_next_peer_lease_status_fallback_when_no_launcher_visible() -> None:
    report = {
        "blocks": [{"status": "BLOCK", "code": "peer_lease", "message": "peer"}],
        "launchers": {"items": {}},
        "drift": {"recommended_probe": "python scripts/tools/profile_check_drift.py"},
    }
    result = wd.choose_next(report, "status")
    assert result["command"] == "python scripts/tools/worktree_guard.py --status --json"


def test_json_mode_alias_emits_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    report = {key: {} for key in wd.JSON_KEYS}
    monkeypatch.setattr(wd, "collect_report", lambda *args, **kwargs: report)
    assert wd.main(["json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out) == report


def test_stage_fixture_reports_count_and_protected_scope(tmp_path: Path) -> None:
    stages = tmp_path / "docs" / "runtime" / "stages"
    stages.mkdir(parents=True)
    (stages / "active-live.md").write_text(
        "---\nmode: IMPLEMENTATION\nscope_lock:\n  - trading_app/live/session_orchestrator.py\n---\n",
        encoding="utf-8",
    )
    (stages / "active-doc.md").write_text("---\nmode: DESIGN\nscope_lock:\n  - docs/foo.md\n---\n", encoding="utf-8")
    (stages / "closed.md").write_text(
        "---\nmode: CLOSED\nscope_lock:\n  - pipeline/check_drift.py\n---\n", encoding="utf-8"
    )
    (stages / "ignored.md").write_text("mode: IMPLEMENTATION\n", encoding="utf-8")
    result = wd.collect_stages(tmp_path)
    assert result["active_count"] == 2
    assert result["total_files"] == 4
    assert result["closed_count"] == 1
    assert result["ignored_count"] == 1
    assert result["protected_scope_highlights"] == ["active-live.md"]


def test_dashboard_stale_signal_summary_does_not_dump_json(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    heartbeat = datetime.now(UTC) - timedelta(seconds=2444)
    (data / "bot_planned_launch.json").write_text(
        json.dumps({"mode": "SIGNAL", "profile_id": "topstep_50k_mnq_auto", "instruments": ["MNQ"], "copies": 1}),
        encoding="utf-8",
    )
    (data / "bot_state.json").write_text(
        json.dumps(
            {
                "mode": "SIGNAL",
                "signal_only": True,
                "profile_id": "topstep_50k_mnq_auto",
                "updated_at": heartbeat.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    with patch.object(wd, "probe_port", return_value=True):
        result = wd.collect_dashboard(tmp_path, 8080)
    assert result["status"] == "WARN"
    assert result["runtime"]["heartbeat_stale"] is True
    assert "signal_only" in result["runtime"]
    assert "bot_state" not in json.dumps(result)


def test_dashboard_reads_signal_only_from_broker_status(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    heartbeat = datetime.now(UTC).isoformat()
    (data / "bot_planned_launch.json").write_text(json.dumps({"mode": "SIGNAL"}), encoding="utf-8")
    (data / "bot_state.json").write_text(
        json.dumps(
            {"mode": "SIGNAL", "broker_status": {"signal_only": True, "demo": True}, "heartbeat_utc": heartbeat}
        ),
        encoding="utf-8",
    )
    with patch.object(wd, "probe_port", return_value=True):
        result = wd.collect_dashboard(tmp_path, 8080)
    assert result["runtime"]["signal_only"] is True
    assert result["runtime"]["demo"] is True


def test_ascii_safe_text_output() -> None:
    report = {
        "git": {
            "status": "OK",
            "root": "C:/repo",
            "context": "windows",
            "python": "python",
            "dirty_count": 0,
            "detached": False,
            "branch": "main",
            "head": "abcdef12",
            "ahead": 0,
            "behind": 0,
            "hooks_path": ".githooks",
        },
        "lease": {
            "status": "WARN",
            "lease_present": True,
            "holder_session_id": "peer",
            "peer_live": True,
            "holder_ppid_alive": False,
            "fresh_peer_heartbeat": True,
            "heartbeat_age_seconds": 12,
        },
        "db": {
            "status": "OK",
            "selected_path": "C:/repo/gold.db",
            "readonly_open": {"status": "OK"},
            "override_active": False,
            "deprecated_scratch": False,
            "hardlink_count": 1,
        },
        "ports": {"status": "OK", "port": 8080, "listening": False},
        "dashboard": {
            "planned": {"mode": "SIGNAL", "profile": "topstep_50k_mnq_auto"},
            "runtime": {
                "mode": "SIGNAL",
                "profile": "topstep_50k_mnq_auto",
                "signal_only": True,
                "heartbeat_age_seconds": 1,
            },
        },
        "stages": {"status": "OK", "active_count": 0, "protected_scope_highlights": []},
        "async_hooks": {"status": "OK", "configured_count": 0, "visible_state": []},
        "integrations": {"status": "OK", "mcp_server_count": 0, "enabled_plugin_count": 0, "disabled_plugin_count": 0},
        "launchers": {
            "status": "OK",
            "items": {
                "START_BOT": {"exists": True},
                "START_WORKTREE": {"exists": True},
                "new_session": {"exists": True},
            },
        },
        "drift": {
            "status": "WARN",
            "recommended_probe": "python scripts/tools/profile_check_drift.py",
            "recommended_fast": "python -u pipeline/check_drift.py --fast --quiet --skip-crg-advisory",
        },
        "blocks": [{"status": "WARN", "code": "stage_bloat", "message": "unicode test \u2192 still safe"}],
        "next": {"command": "python scripts/tools/stage_reaper_audit.py", "reason": "review stages"},
    }
    text = "\n".join(wd.format_status(report, mode="status"))
    text.encode("cp1252", errors="strict")
