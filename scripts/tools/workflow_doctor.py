#!/usr/bin/env python3
"""Fast, report-only workflow control-plane status for canompx3.

This tool is intentionally additive: it reads repo/runtime state, summarizes
the confusing bits, and never creates worktrees, moves stages, starts/stops live
processes, calls dashboard mutation endpoints, or opens DuckDB writable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HOOKS_PATH = ".githooks"
WORKTREE_META_FILENAME = ".canompx3-worktree.json"
DEFAULT_PORT = 8080
COMMAND_TIMEOUT_SECONDS = 5.0
GIT_TIMEOUT_SECONDS = 2.0
DASHBOARD_STALE_HEARTBEAT_SECONDS = 180
STAGE_BLOAT_WARN_COUNT = 10
JSON_KEYS = (
    "git",
    "worktrees",
    "lease",
    "db",
    "ports",
    "dashboard",
    "stages",
    "hooks",
    "async_hooks",
    "integrations",
    "launchers",
    "drift",
    "blocks",
    "next",
)


class WorkflowDoctorError(RuntimeError):
    """Raised for local, non-mutating collection failures."""


def _as_ascii(value: Any) -> str:
    text = str(value)
    return text.encode("ascii", errors="replace").decode("ascii")


def _print_ascii(line: str = "") -> None:
    print(_as_ascii(line))


def _path_str(path: Path | str | None) -> str | None:
    if path is None:
        return None
    return str(path)


def run_command(args: list[str], *, cwd: Path, timeout: float = COMMAND_TIMEOUT_SECONDS) -> dict[str, Any]:
    try:
        env = None
        if args and args[0] == "git":
            env = os.environ.copy()
            for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
                env.pop(name, None)
        result = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=env,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
            "command": args,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "command": args,
        }
    except OSError as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
            "command": args,
        }


def read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def parse_iso_age_seconds(value: Any, *, now: float | None = None) -> float | None:
    if not value:
        return None
    now = time.time() if now is None else now
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, now - parsed.timestamp())
    except (TypeError, ValueError, OSError):
        return None


def probe_port(host: str, port: int, *, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def parse_git_worktree_porcelain(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current is not None:
                items.append(current)
                current = None
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            if current is not None:
                items.append(current)
            current = {
                "path": value,
                "head": None,
                "branch": None,
                "detached": False,
                "bare": False,
                "prunable": None,
            }
        elif current is not None and key == "HEAD":
            current["head"] = value
        elif current is not None and key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif current is not None and key == "detached":
            current["detached"] = True
        elif current is not None and key == "bare":
            current["bare"] = True
        elif current is not None and key == "prunable":
            current["prunable"] = value or True
        elif current is not None and line.startswith("prunable "):
            current["prunable"] = line.removeprefix("prunable ").strip() or True
    if current is not None:
        items.append(current)
    return items


def parse_ahead_behind(branch_line: str) -> dict[str, int]:
    ahead = 0
    behind = 0
    if "[" in branch_line and "]" in branch_line:
        bracket = branch_line.split("[", 1)[1].split("]", 1)[0]
        for part in bracket.split(","):
            part = part.strip()
            if part.startswith("ahead "):
                ahead = int(part.removeprefix("ahead ").strip() or "0")
            elif part.startswith("behind "):
                behind = int(part.removeprefix("behind ").strip() or "0")
    return {"ahead": ahead, "behind": behind}


def is_deprecated_scratch_db(path: Path | str) -> bool:
    selected_text = str(path).replace("\\", "/").lower()
    return selected_text.endswith("/c/db/gold.db") or selected_text == "c:/db/gold.db"


def collect_git(root: Path) -> dict[str, Any]:
    status = run_command(["git", "status", "--short", "--branch"], cwd=root, timeout=GIT_TIMEOUT_SECONDS)
    head = run_command(["git", "rev-parse", "--short=8", "HEAD"], cwd=root, timeout=GIT_TIMEOUT_SECONDS)
    branch = run_command(["git", "branch", "--show-current"], cwd=root, timeout=GIT_TIMEOUT_SECONDS)
    hooks = run_command(["git", "config", "--get", "core.hooksPath"], cwd=root, timeout=GIT_TIMEOUT_SECONDS)
    root_cmd = run_command(["git", "rev-parse", "--show-toplevel"], cwd=root, timeout=GIT_TIMEOUT_SECONDS)

    branch_line = ""
    dirty_lines: list[str] = []
    if status["ok"]:
        lines = [line for line in status["stdout"].splitlines() if line.strip()]
        branch_line = lines[0] if lines else ""
        dirty_lines = lines[1:]

    branch_name = (branch["stdout"].strip() if branch["ok"] else "") or None
    detached = not bool(branch_name)
    hooks_path = hooks["stdout"].strip() if hooks["ok"] else ""
    ahead_behind = parse_ahead_behind(branch_line)
    selected_root = root_cmd["stdout"].strip() if root_cmd["ok"] else str(root)
    head_sha = head["stdout"].strip() if head["ok"] else None
    return {
        "status": "OK" if status["ok"] else "UNSUPPORTED",
        "root": selected_root,
        "cwd": str(root),
        "context": "windows" if os.name == "nt" else "wsl-or-posix",
        "platform": platform.system(),
        "python": sys.executable,
        "branch": branch_name,
        "head": head_sha,
        "dirty_count": len(dirty_lines),
        "dirty_files": dirty_lines[:25],
        "ahead": ahead_behind["ahead"],
        "behind": ahead_behind["behind"],
        "detached": detached,
        "hooks_path": hooks_path or None,
        "hooks_expected": EXPECTED_HOOKS_PATH,
        "hooks_ok": hooks_path == EXPECTED_HOOKS_PATH,
        "raw_branch_line": branch_line,
        "commands": {
            "status": _cmd_for_json(status),
            "hooks": _cmd_for_json(hooks),
        },
    }


def _cmd_for_json(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result.get("ok"),
        "returncode": result.get("returncode"),
        "timed_out": result.get("timed_out"),
        "command": result.get("command"),
    }


def collect_worktrees(root: Path, *, include_dirty: bool = False) -> dict[str, Any]:
    raw = run_command(["git", "worktree", "list", "--porcelain"], cwd=root, timeout=GIT_TIMEOUT_SECONDS)
    if not raw["ok"]:
        return {
            "status": "UNSUPPORTED",
            "count": 0,
            "detached_count": 0,
            "managed_count": 0,
            "items": [],
            "command": _cmd_for_json(raw),
        }
    items = parse_git_worktree_porcelain(raw["stdout"])
    managed_count = 0
    managed_root = (root / ".worktrees").resolve()
    for item in items:
        path = Path(str(item["path"]))
        meta = read_json_file(path / WORKTREE_META_FILENAME)
        try:
            path_is_managed = path.resolve().is_relative_to(managed_root)
        except OSError:
            path_is_managed = False
        metadata_matches = metadata_matches_worktree(meta, item)
        item_is_managed = path_is_managed or metadata_matches
        if item_is_managed:
            managed_count += 1
        item["managed"] = item_is_managed
        item["metadata"] = meta if metadata_matches else None
        if metadata_matches:
            item["metadata_status"] = "ok"
        elif meta is not None and item_is_managed:
            item["metadata_status"] = "stale_branch_mismatch"
            item["metadata_branch"] = meta.get("branch")
        elif item_is_managed:
            item["metadata_status"] = "missing"
        else:
            item["metadata_status"] = None
        if meta is not None and not item_is_managed:
            item["metadata_ignored"] = "metadata branch did not match worktree branch"
        if include_dirty:
            dirty = run_command(
                ["git", "-C", str(path), "status", "--short", "--untracked-files=no"],
                cwd=root,
                timeout=GIT_TIMEOUT_SECONDS,
            )
            item["dirty_count"] = (
                len([line for line in dirty["stdout"].splitlines() if line.strip()]) if dirty["ok"] else None
            )
    return {
        "status": "OK",
        "count": len(items),
        "detached_count": sum(1 for item in items if item.get("detached")),
        "managed_count": managed_count,
        "items": items,
        "command": _cmd_for_json(raw),
    }


def metadata_matches_worktree(meta: dict[str, Any] | None, item: dict[str, Any]) -> bool:
    if not meta:
        return False
    meta_branch = str(meta.get("branch") or "").removeprefix("refs/heads/")
    item_branch = str(item.get("branch") or "").removeprefix("refs/heads/")
    return bool(meta_branch and item_branch and meta_branch == item_branch)


def collect_lease(root: Path) -> dict[str, Any]:
    try:
        if str(PROJECT_ROOT / "scripts" / "tools") not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
        import worktree_guard  # type: ignore[import-not-found]

        data = worktree_guard.status(root)
    except Exception as exc:  # noqa: BLE001 - report-only status surface
        return {
            "status": "UNSUPPORTED",
            "lease_present": None,
            "peer_live": None,
            "holder_ppid_alive": None,
            "fresh_peer_heartbeat": None,
            "heartbeat_age_seconds": None,
            "current_is_holder": None,
            "error": str(exc),
        }
    data["status"] = classify_lease(data)
    return data


def classify_lease(snapshot: dict[str, Any]) -> str:
    if not snapshot.get("lease_present"):
        return "OK"
    if snapshot.get("current_is_holder"):
        return "OK"
    if snapshot.get("peer_live"):
        return "BLOCK"
    if snapshot.get("fresh_peer_heartbeat"):
        return "WARN"
    if snapshot.get("holder_ppid_alive") is False:
        return "WARN"
    return "WARN"


def check_db_readonly(path: Path) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore[import-not-found]
    except ImportError as exc:
        return {"ok": False, "status": "UNSUPPORTED", "reason": f"duckdb import failed: {exc}"}
    try:
        con = duckdb.connect(str(path), read_only=True)
        con.close()
        return {"ok": True, "status": "OK", "reason": None}
    except Exception as exc:  # noqa: BLE001 - probe result only
        return {"ok": False, "status": "BLOCK", "reason": str(exc)}


def collect_db(root: Path) -> dict[str, Any]:
    env_override = os.environ.get("DUCKDB_PATH")
    canonical = root / "gold.db"
    selected = canonical
    override_active = False
    deprecated = False
    import_error = None

    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from pipeline.paths import GOLD_DB_PATH  # type: ignore[import-not-found]

        selected = Path(GOLD_DB_PATH)
        canonical = root / "gold.db"
        override_active = bool(env_override) and Path(env_override).exists()
    except Exception as exc:  # noqa: BLE001 - surface as unsupported
        import_error = str(exc)

    deprecated = is_deprecated_scratch_db(selected)
    readonly = (
        check_db_readonly(selected)
        if selected.exists() and not deprecated
        else {
            "ok": False,
            "status": "BLOCK" if deprecated else "WARN",
            "reason": "selected DB missing" if not deprecated else "deprecated scratch DB path selected",
        }
    )
    try:
        stat = selected.stat()
        link_count = getattr(stat, "st_nlink", 1)
    except OSError:
        link_count = None
    status = "OK"
    if deprecated or readonly["status"] == "BLOCK":
        status = "BLOCK"
    elif import_error or readonly["status"] == "UNSUPPORTED":
        status = "UNSUPPORTED"
    elif override_active or link_count and link_count > 1:
        status = "WARN"
    return {
        "status": status,
        "selected_path": str(selected),
        "canonical_path": str(canonical),
        "override_env": env_override,
        "override_active": override_active,
        "deprecated_scratch": deprecated,
        "exists": selected.exists(),
        "is_symlink": selected.is_symlink(),
        "hardlink_count": link_count,
        "readonly_open": readonly,
        "import_error": import_error,
    }


def collect_ports(port: int = DEFAULT_PORT) -> dict[str, Any]:
    listening = probe_port("127.0.0.1", port)
    return {
        "status": "WARN" if listening else "OK",
        "host": "127.0.0.1",
        "port": port,
        "listening": listening,
        "note": "socket open; not proof of healthy runtime" if listening else "socket closed",
    }


def collect_dashboard(root: Path, port: int = DEFAULT_PORT, *, now: float | None = None) -> dict[str, Any]:
    planned = read_json_file(root / "data" / "bot_planned_launch.json") or {}
    state = read_json_file(root / "data" / "bot_state.json") or {}
    heartbeat_age = parse_iso_age_seconds(
        state.get("heartbeat_utc")
        or state.get("updated_at")
        or state.get("heartbeat_at")
        or state.get("last_heartbeat")
        or state.get("timestamp"),
        now=now,
    )
    listening = probe_port("127.0.0.1", port)
    stale = heartbeat_age is not None and heartbeat_age > DASHBOARD_STALE_HEARTBEAT_SECONDS
    status = "OK"
    if (listening and stale) or state.get("mode") in {"LIVE", "live"}:
        status = "WARN"
    return {
        "status": status,
        "port": port,
        "port_listening": listening,
        "planned": {
            "mode": planned.get("mode"),
            "profile": planned.get("profile_id") or planned.get("profile"),
            "instrument": _first_or_value(planned.get("instruments") or planned.get("instrument")),
            "copies": planned.get("copies"),
            "source": planned.get("source"),
        },
        "runtime": {
            "mode": state.get("mode"),
            "profile": state.get("profile_id") or state.get("profile"),
            "signal_only": _nested_state_value(state, "signal_only"),
            "demo": _nested_state_value(state, "demo"),
            "heartbeat_age_seconds": heartbeat_age,
            "heartbeat_stale": stale,
        },
        "note": "open port plus stale heartbeat is confusing/stale, not healthy" if listening and stale else None,
    }


def _first_or_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _nested_state_value(state: dict[str, Any], key: str) -> Any:
    if key in state:
        return state.get(key)
    for nested_key in ("control", "broker_status", "runtime", "account"):
        nested = state.get(nested_key)
        if isinstance(nested, dict) and key in nested:
            return nested.get(key)
    return None


def collect_stages(root: Path, *, use_system_context: bool = False) -> dict[str, Any]:
    if not use_system_context:
        return collect_stages_from_files(root)

    cmd = run_command(
        [
            sys.executable,
            "scripts/tools/system_context.py",
            "--context",
            "claude-windows" if os.name == "nt" else "codex-wsl",
            "--action",
            "orientation",
            "--format",
            "json",
        ],
        cwd=root,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if not cmd["ok"]:
        return {
            "status": "UNSUPPORTED",
            "active_count": None,
            "protected_scope_highlights": [],
            "command": _cmd_for_json(cmd),
            "next_command": "python scripts/tools/system_context.py --context claude-windows --action orientation --format json",
        }
    try:
        payload = json.loads(cmd["stdout"])
    except json.JSONDecodeError:
        return {
            "status": "UNSUPPORTED",
            "active_count": None,
            "protected_scope_highlights": [],
            "command": _cmd_for_json(cmd),
        }
    active = payload.get("snapshot", {}).get("active_stages", [])
    file_counts = scan_stage_file_counts(root)
    protected = []
    protected_terms = ("trading_app/live", "scripts/run_live_session.py", "gold.db", "pipeline/check_drift.py")
    for item in active:
        for scope in item.get("scope_lock") or []:
            if any(term in scope.replace("\\", "/") for term in protected_terms):
                protected.append(scope)
                break
    active_count = len(active)
    return {
        "status": "WARN" if active_count >= STAGE_BLOAT_WARN_COUNT else "OK",
        "active_count": active_count,
        "total_files": file_counts["total_files"],
        "closed_count": file_counts["closed_count"],
        "ignored_count": file_counts["ignored_count"],
        "protected_scope_highlights": protected[:10],
        "command": _cmd_for_json(cmd),
        "next_command": "python scripts/tools/stage_reaper_audit.py",
    }


def collect_stages_from_files(root: Path) -> dict[str, Any]:
    stages_dir = root / "docs" / "runtime" / "stages"
    try:
        files = sorted(stages_dir.glob("*.md"))
    except OSError:
        return {
            "status": "UNSUPPORTED",
            "active_count": None,
            "protected_scope_highlights": [],
            "next_command": "python scripts/tools/system_context.py --context claude-windows --action orientation --format json",
        }
    protected: list[str] = []
    protected_terms = ("trading_app/live", "scripts/run_live_session.py", "gold.db", "pipeline/check_drift.py")
    active_files: list[Path] = []
    closed_count = 0
    ignored_count = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            ignored_count += 1
            continue
        metadata = parse_stage_frontmatter(text)
        if metadata is None:
            ignored_count += 1
            continue
        if stage_text_is_closed(text, metadata):
            closed_count += 1
            continue
        active_files.append(path)
        scopes = metadata.get("scope_lock") or []
        if not isinstance(scopes, list):
            scopes = []
        for scope in [str(item).replace("\\", "/") for item in scopes]:
            if any(term in scope for term in protected_terms):
                protected.append(path.name)
                break
    active_count = len(active_files)
    return {
        "status": "WARN" if active_count >= STAGE_BLOAT_WARN_COUNT else "OK",
        "active_count": active_count,
        "total_files": len(files),
        "closed_count": closed_count,
        "ignored_count": ignored_count,
        "protected_scope_highlights": protected[:10],
        "source": "stage-files",
        "next_command": "python scripts/tools/stage_reaper_audit.py",
    }


def scan_stage_file_counts(root: Path) -> dict[str, int]:
    stage_dir = root / "docs" / "runtime" / "stages"
    try:
        files = sorted(stage_dir.glob("*.md"))
    except OSError:
        return {"total_files": 0, "closed_count": 0, "ignored_count": 0}
    closed_count = 0
    ignored_count = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            ignored_count += 1
            continue
        metadata = parse_stage_frontmatter(text)
        if metadata is None:
            ignored_count += 1
            continue
        if stage_text_is_closed(text, metadata):
            closed_count += 1
    return {"total_files": len(files), "closed_count": closed_count, "ignored_count": ignored_count}


def parse_stage_frontmatter(text: str) -> dict[Any, Any] | None:
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def stage_text_is_closed(text: str, metadata: dict[Any, Any]) -> bool:
    for field in ("status", "mode"):
        value = metadata.get(field)
        if value is None:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
        if normalized.split(" ", 1)[0] in {"closed", "complete", "completed", "done", "implemented"}:
            return True
    return re.search(r"(?m)^##\s+Execution Outcome\s*$", text) is not None


def collect_hooks(git: dict[str, Any]) -> dict[str, Any]:
    ok = git.get("hooks_path") == EXPECTED_HOOKS_PATH
    return {
        "status": "OK" if ok else "BLOCK",
        "core_hooks_path": git.get("hooks_path"),
        "expected": EXPECTED_HOOKS_PATH,
        "next_command": "git config core.hooksPath .githooks",
    }


def collect_integrations(root: Path) -> dict[str, Any]:
    mcp = read_json_file(root / ".mcp.json") or {}
    settings = read_json_file(root / ".claude" / "settings.json") or {}
    mcp_servers = mcp.get("mcpServers") if isinstance(mcp, dict) else None
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    plugins = settings.get("enabledPlugins") if isinstance(settings, dict) else None
    disabled_plugins = settings.get("disabledPlugins") if isinstance(settings, dict) else None
    if not isinstance(plugins, list):
        plugins = []
    if not isinstance(disabled_plugins, list):
        disabled_plugins = []
    return {
        "status": "WARN" if mcp_servers or plugins else "OK",
        "mcp_server_count": len(mcp_servers),
        "mcp_servers": sorted(str(key) for key in mcp_servers)[:25],
        "enabled_plugin_count": len(plugins),
        "enabled_plugins": sorted(str(item) for item in plugins)[:25],
        "disabled_plugin_count": len(disabled_plugins),
        "disabled_plugins": sorted(str(item) for item in disabled_plugins)[:25],
        "note": "enabled integrations are report-only here; workflow_doctor does not start MCP/plugins",
    }


def collect_launchers(root: Path) -> dict[str, Any]:
    candidates = {
        "START_BOT": root / "START_BOT.bat",
        "START_WORKTREE": root / "START_WORKTREE.bat",
        "new_session": root / "scripts" / "tools" / "new_session.sh",
    }
    items: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, path in candidates.items():
        exists = path.exists()
        if not exists:
            missing.append(name)
        items[name] = {"exists": exists, "path": str(path)}
    return {
        "status": "WARN" if missing else "OK",
        "items": items,
        "missing": missing,
        "note": "visibility only; workflow_doctor does not run launchers",
    }


def collect_async_hooks(root: Path) -> dict[str, Any]:
    settings = read_json_file(root / ".claude" / "settings.json") or {}
    hooks = settings.get("hooks") or {}
    async_items: list[dict[str, Any]] = []
    for event, entries in hooks.items():
        for entry in entries or []:
            matcher = entry.get("matcher") if isinstance(entry, dict) else None
            for hook in entry.get("hooks", []) if isinstance(entry, dict) else []:
                command = hook.get("command") if isinstance(hook, dict) else None
                if command and hook.get("async") is True:
                    async_items.append(
                        {
                            "event": event,
                            "matcher": matcher,
                            "command": command,
                            "timeout": hook.get("timeout"),
                        }
                    )
    visible_state: list[dict[str, Any]] = []
    for folder in (root / ".claude" / "hooks", root / ".claude" / "logs"):
        if not folder.exists():
            continue
        for candidate in folder.glob("*"):
            if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl", ".log", ".txt"}:
                try:
                    age = time.time() - candidate.stat().st_mtime
                except OSError:
                    age = None
                visible_state.append({"path": str(candidate), "age_seconds": age})
    return {
        "status": "WARN" if async_items and not visible_state else "OK",
        "configured_count": len(async_items),
        "configured": async_items,
        "visible_state": visible_state[:10],
        "note": "configured async hooks have no visible state/log file" if async_items and not visible_state else None,
    }


def collect_drift(root: Path) -> dict[str, Any]:
    profile_script = root / "scripts" / "tools" / "profile_check_drift.py"
    profile_text = ""
    try:
        profile_text = profile_script.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return {
        "status": "WARN",
        "runs_by_default": False,
        "recommended_probe": "python scripts/tools/profile_check_drift.py",
        "recommended_fast": "python -u pipeline/check_drift.py --fast --quiet --skip-crg-advisory",
        "full_gate": "python -u pipeline/check_drift.py --quiet",
        "profile_has_json": "--json" in profile_text,
        "profile_has_timeout": "timeout" in profile_text.lower(),
        "note": "drift is surfaced but not run by workflow_doctor status",
    }


def collect_preflight(root: Path) -> dict[str, Any]:
    cmd = run_command(
        [sys.executable, "scripts/tools/session_preflight.py", "--quiet"], cwd=root, timeout=COMMAND_TIMEOUT_SECONDS
    )
    lines = [line for line in (cmd["stdout"] + cmd["stderr"]).splitlines() if line.strip()]
    return {
        "status": "OK" if cmd["ok"] else "BLOCK",
        "returncode": cmd["returncode"],
        "timed_out": cmd["timed_out"],
        "summary": lines[:20],
        "command": _cmd_for_json(cmd),
    }


def build_blocks(report: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    git = report.get("git", {})
    if git.get("dirty_count", 0):
        blocks.append({"status": "BLOCK", "code": "dirty_tree", "message": "current tree has uncommitted changes"})
    if git.get("detached"):
        blocks.append({"status": "BLOCK", "code": "detached_head", "message": "current checkout is detached"})
    if not git.get("hooks_ok", False):
        blocks.append({"status": "BLOCK", "code": "hook_path", "message": "core.hooksPath is not .githooks"})
    lease = report.get("lease", {})
    if lease.get("status") == "BLOCK":
        blocks.append({"status": "BLOCK", "code": "peer_lease", "message": "fresh peer lease owns this worktree"})
    db = report.get("db", {})
    if db.get("status") == "BLOCK":
        blocks.append({"status": "BLOCK", "code": "db", "message": "DB path/read-only probe is unsafe"})
    dashboard = report.get("dashboard", {})
    runtime = dashboard.get("runtime", {})
    if runtime.get("heartbeat_stale") and dashboard.get("port_listening"):
        blocks.append(
            {"status": "WARN", "code": "dashboard_stale", "message": "dashboard port open but bot heartbeat is stale"}
        )
    stages = report.get("stages", {})
    if (stages.get("active_count") or 0) >= STAGE_BLOAT_WARN_COUNT:
        blocks.append({"status": "WARN", "code": "stage_bloat", "message": "many active stage files are present"})
    return blocks


def _peer_lease_escape_command(report: dict[str, Any]) -> str:
    """Return the safest operator escape hatch for a live peer lease block.

    A fresh peer lease means the current harness is anchored to a hot worktree.
    The correct recovery is to open a new isolated session, not to inspect the
    holder and then be tempted to force-release it. Prefer the Windows Claude
    launcher when present, fall back to the repo's bash worktree spawner, and
    include the read-only status command only when no launcher is visible.
    """
    launchers = report.get("launchers", {})
    items = launchers.get("items") if isinstance(launchers, dict) else None
    if isinstance(items, dict):
        start_worktree = items.get("START_WORKTREE")
        if isinstance(start_worktree, dict) and start_worktree.get("exists"):
            return "START_WORKTREE.bat <descriptor>"
        new_session = items.get("new_session")
        if isinstance(new_session, dict) and new_session.get("exists"):
            return "scripts/tools/new_session.sh <descriptor> && cd ../canompx3-<descriptor>"
    return "python scripts/tools/worktree_guard.py --status --json"


def choose_next(report: dict[str, Any], mode: str) -> dict[str, str]:
    blocks_by_code = {block.get("code"): block for block in report.get("blocks", [])}
    for code in ("hook_path", "peer_lease", "db", "detached_head", "dirty_tree", "dashboard_stale", "stage_bloat"):
        if code not in blocks_by_code:
            continue
        if code == "hook_path":
            return {
                "status": "NEXT",
                "command": "git config core.hooksPath .githooks",
                "reason": "activate commit guardrail",
            }
        if code == "peer_lease":
            return {
                "status": "NEXT",
                "command": _peer_lease_escape_command(report),
                "reason": "open an isolated worktree while the live peer lease remains intact",
            }
        if code == "db":
            return {
                "status": "NEXT",
                "command": "python scripts/tools/workflow_doctor.py db",
                "reason": "inspect DB selection",
            }
        if code == "detached_head":
            return {"status": "NEXT", "command": "git branch --show-current", "reason": "confirm current branch"}
        if code == "dirty_tree":
            return {"status": "NEXT", "command": "git status --short", "reason": "inspect uncommitted work"}
        if code == "dashboard_stale":
            return {
                "status": "NEXT",
                "command": "python scripts/tools/workflow_doctor.py dashboard",
                "reason": "separate open port from healthy runtime",
            }
        if code == "stage_bloat":
            return {
                "status": "NEXT",
                "command": "python scripts/tools/stage_reaper_audit.py",
                "reason": "review stale stages manually",
            }
    if mode == "drift":
        return {
            "status": "NEXT",
            "command": report["drift"]["recommended_probe"],
            "reason": "profile drift before running full gate",
        }
    return {
        "status": "NEXT",
        "command": "python -m pytest tests/test_tools/test_workflow_doctor.py -q",
        "reason": "prove workflow doctor wiring",
    }


def collect_report(root: Path, *, mode: str, port: int = DEFAULT_PORT) -> dict[str, Any]:
    root = root.resolve()
    git = collect_git(root)
    include_dirty_worktrees = mode == "worktrees"
    report: dict[str, Any] = {
        "git": git,
        "worktrees": collect_worktrees(root, include_dirty=include_dirty_worktrees),
        "lease": collect_lease(root),
        "db": collect_db(root),
        "ports": collect_ports(port),
        "dashboard": collect_dashboard(root, port),
        "stages": collect_stages(root, use_system_context=mode == "stages"),
        "hooks": collect_hooks(git),
        "async_hooks": collect_async_hooks(root),
        "integrations": collect_integrations(root),
        "launchers": collect_launchers(root),
        "drift": collect_drift(root),
        "blocks": [],
        "next": {},
    }
    if mode == "preflight":
        report["preflight"] = collect_preflight(root)
    report["blocks"] = build_blocks(report)
    report["next"] = choose_next(report, mode)
    return {key: report.get(key) for key in JSON_KEYS} | (
        {"preflight": report["preflight"]} if "preflight" in report else {}
    )


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def format_status(report: dict[str, Any], *, mode: str) -> list[str]:
    git = report["git"]
    lease = report["lease"]
    db = report["db"]
    ports = report["ports"]
    dashboard = report["dashboard"]
    stages = report["stages"]
    async_hooks = report["async_hooks"]
    integrations = report["integrations"]
    launchers = report["launchers"]
    drift = report["drift"]
    lines = [
        "WORKFLOW DOCTOR",
        f"{git['status']} where root={git.get('root')} context={git.get('context')} python={git.get('python')}",
        (
            f"{'OK' if git.get('dirty_count') == 0 and not git.get('detached') else 'BLOCK'} git "
            f"branch={git.get('branch') or 'DETACHED'} head={git.get('head')} dirty={git.get('dirty_count')} "
            f"ahead={git.get('ahead')} behind={git.get('behind')} hooks={git.get('hooks_path') or 'unset'}"
        ),
        (
            f"{lease.get('status', 'UNSUPPORTED')} lease present={_fmt_bool(lease.get('lease_present'))} "
            f"holder={lease.get('holder_session_id') or lease.get('session_id') or 'unknown'} "
            f"peer_live={_fmt_bool(lease.get('peer_live'))} ppid_alive={_fmt_bool(lease.get('holder_ppid_alive'))} "
            f"fresh_peer_beat={_fmt_bool(lease.get('fresh_peer_heartbeat'))} age={_seconds(lease.get('heartbeat_age_seconds'))}"
        ),
        (
            f"{db.get('status')} db selected={db.get('selected_path')} readonly={db.get('readonly_open', {}).get('status')} "
            f"override={_fmt_bool(db.get('override_active'))} deprecated={_fmt_bool(db.get('deprecated_scratch'))} "
            f"links={db.get('hardlink_count')}"
        ),
        (
            f"{ports.get('status')} port {ports.get('port')} listening={_fmt_bool(ports.get('listening'))}; "
            f"dashboard planned={dashboard.get('planned', {}).get('mode')}/{dashboard.get('planned', {}).get('profile')} "
            f"runtime={dashboard.get('runtime', {}).get('mode')}/{dashboard.get('runtime', {}).get('profile')} "
            f"signal_only={_fmt_bool(dashboard.get('runtime', {}).get('signal_only'))} "
            f"heartbeat_age={_seconds(dashboard.get('runtime', {}).get('heartbeat_age_seconds'))}"
        ),
        (
            f"{stages.get('status')} stages active={stages.get('active_count')} "
            f"total={stages.get('total_files')} closed={stages.get('closed_count')} "
            f"ignored={stages.get('ignored_count')} "
            f"protected_highlights={len(stages.get('protected_scope_highlights') or [])}"
        ),
        (
            f"{async_hooks.get('status')} async hooks={async_hooks.get('configured_count')} "
            f"visible_state={len(async_hooks.get('visible_state') or [])}"
        ),
        (
            f"{integrations.get('status')} integrations mcp={integrations.get('mcp_server_count')} "
            f"plugins_on={integrations.get('enabled_plugin_count')} plugins_off={integrations.get('disabled_plugin_count')}"
        ),
        (
            f"{launchers.get('status')} launchers START_BOT={_fmt_launcher(launchers, 'START_BOT')} "
            f"START_WORKTREE={_fmt_launcher(launchers, 'START_WORKTREE')} "
            f"new_session={_fmt_launcher(launchers, 'new_session')}"
        ),
        (
            f'{drift.get("status")} drift default_run=no probe="{drift.get("recommended_probe")}" '
            f'fast="{drift.get("recommended_fast")}"'
        ),
    ]
    if mode == "worktrees":
        wt = report["worktrees"]
        lines.append(
            f"{wt.get('status')} worktrees count={wt.get('count')} detached={wt.get('detached_count')} managed={wt.get('managed_count')}"
        )
        for item in wt.get("items", [])[:15]:
            lines.append(
                f"{'WARN' if item.get('detached') else 'OK'} wt path={item.get('path')} "
                f"branch={item.get('branch') or 'DETACHED'} managed={_fmt_bool(item.get('managed'))} "
                f"meta={item.get('metadata_status') or 'none'} dirty={item.get('dirty_count', 'n/a')}"
            )
    if mode == "preflight" and "preflight" in report:
        preflight = report["preflight"]
        lines.append(
            f"{preflight.get('status')} preflight rc={preflight.get('returncode')} timed_out={_fmt_bool(preflight.get('timed_out'))}"
        )
        for line in preflight.get("summary", [])[:8]:
            lines.append(f"WARN preflight {line}")
    if mode == "db":
        lines.append(f"{db.get('status')} canonical={db.get('canonical_path')}")
        if db.get("readonly_open", {}).get("reason"):
            lines.append(
                f"{db.get('readonly_open', {}).get('status')} db_reason={db.get('readonly_open', {}).get('reason')}"
            )
    if mode == "dashboard":
        note = dashboard.get("note")
        if note:
            lines.append(f"WARN dashboard_note={note}")
    if mode == "async":
        for item in async_hooks.get("configured", [])[:8]:
            lines.append(
                f"WARN async event={item.get('event')} matcher={item.get('matcher')} command={item.get('command')}"
            )
    if mode == "stages":
        for item in stages.get("protected_scope_highlights", [])[:8]:
            lines.append(f"WARN stage_scope={item}")
    for block in report.get("blocks", [])[:8]:
        lines.append(f"{block.get('status')} {block.get('code')}: {block.get('message')}")
    next_item = report.get("next", {})
    lines.append(f"NEXT {next_item.get('command')} # {next_item.get('reason')}")
    return [_as_ascii(line) for line in lines]


def _seconds(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.0f}s"
    except (TypeError, ValueError):
        return "unknown"


def _fmt_launcher(launchers: dict[str, Any], name: str) -> str:
    item = (launchers.get("items") or {}).get(name) or {}
    return "yes" if item.get("exists") else "no"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report-only workflow status for canompx3")
    parser.add_argument(
        "mode",
        nargs="?",
        default="status",
        choices=("status", "preflight", "ports", "db", "stages", "worktrees", "dashboard", "async", "drift", "json"),
    )
    parser.add_argument("--json", action="store_true", help="Emit stable JSON contract")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Repo root")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Dashboard port to socket-probe")
    return parser


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("CANOMPX3_WORKFLOW_DOCTOR_DISABLE") == "1":
        _print_ascii("OK workflow_doctor disabled by CANOMPX3_WORKFLOW_DOCTOR_DISABLE=1")
        return 0
    args = build_parser().parse_args(argv)
    if args.mode == "json":
        args.json = True
        args.mode = "status"
    report = collect_report(Path(args.root), mode=args.mode, port=args.port)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        for line in format_status(report, mode=args.mode):
            _print_ascii(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
