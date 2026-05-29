#!/usr/bin/env python3
"""Weekly project-improvement audit evidence collector.

This tool is intentionally not the auditor. It gathers deterministic repo,
CI, live-readiness, workflow, and security evidence for a recurring PASS 1
review. By default it writes nothing and emits the packet to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_CMD = "python"
MAX_MARKDOWN_ITEMS = 5
MAX_PR_FILE_PATHS = 20
MAX_ERROR_CHARS = 160

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _run(cmd: list[str], *, cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _call(
    runner: Callable[..., Any],
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int = 30,
    optional: bool = False,
) -> dict[str, Any]:
    try:
        result = runner(cmd, cwd=cwd, timeout=timeout)
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        return {
            "status": "unknown" if optional else "error",
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": str(exc),
        }
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    returncode = int(getattr(result, "returncode", 1))
    status = "ok" if returncode == 0 else ("unknown" if optional else "error")
    return {
        "status": status,
        "command": cmd,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": (stderr.strip() or stdout.strip()) if status != "ok" else None,
    }


def _lines(text: str, *, limit: int | None = None) -> list[str]:
    rows = [line.rstrip() for line in text.splitlines() if line.strip()]
    return rows[:limit] if limit is not None else rows


def _compact_text(value: Any, *, max_chars: int = MAX_ERROR_CHARS) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    return first_line[: max_chars - 3].rstrip() + "..." if len(first_line) > max_chars else first_line


def _json_or_unknown(call: dict[str, Any], fallback: Any) -> Any:
    try:
        return json.loads(call["stdout"] or "null")
    except json.JSONDecodeError:
        return fallback


def _count_or_unknown(call: dict[str, Any], *, endpoint: str) -> dict[str, Any]:
    if call["status"] != "ok":
        return {
            "status": "unknown",
            "endpoint": endpoint,
            "open_alert_count": None,
            "error": call.get("error") or "request failed",
        }
    try:
        count = int((call["stdout"] or "0").strip())
    except ValueError:
        return {
            "status": "unknown",
            "endpoint": endpoint,
            "open_alert_count": None,
            "error": f"non-integer response: {(call['stdout'] or '').strip()}",
        }
    return {"status": "ok", "endpoint": endpoint, "open_alert_count": count, "error": None}


def _repo_slug(remote_url: str) -> str | None:
    raw = remote_url.strip()
    if not raw:
        return None
    if raw.startswith("git@github.com:"):
        raw = raw.split(":", 1)[1]
    elif "github.com/" in raw:
        raw = raw.split("github.com/", 1)[1]
    raw = raw.removesuffix(".git").strip("/")
    return raw or None


def _active_stage_summary(root: Path) -> dict[str, Any]:
    stage_dir = root / "docs" / "runtime" / "stages"
    entries: list[dict[str, str]] = []
    if not stage_dir.exists():
        return {"count": 0, "active_count": 0, "entries": []}
    for path in sorted(stage_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            entries.append({"path": path.as_posix(), "mode": "UNKNOWN", "task": f"read failed: {exc}"})
            continue
        mode = "UNKNOWN"
        task = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("mode:"):
                mode = stripped.split(":", 1)[1].strip() or "UNKNOWN"
            if stripped.startswith("task:") and not task:
                task = stripped.split(":", 1)[1].strip()
        entries.append({"path": path.relative_to(root).as_posix(), "mode": mode, "task": task})
    active = [entry for entry in entries if entry["mode"] not in {"CLOSED", "DONE"}]
    return {"count": len(entries), "active_count": len(active), "entries": entries[:20]}


def _file_context(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"path": str(p), "exists": True, "error": str(exc)}
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": p.stat().st_size,
        "last_lines": _lines(text)[-5:],
    }


def _summarize_pr_failures(prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for pr in prs:
        failed = [
            {
                "name": check.get("name"),
                "workflowName": check.get("workflowName"),
                "conclusion": check.get("conclusion"),
                "status": check.get("status"),
            }
            for check in pr.get("statusCheckRollup", [])
            if check.get("conclusion") == "FAILURE"
        ]
        if failed:
            failures.append(
                {
                    "number": pr.get("number"),
                    "title": pr.get("title"),
                    "url": pr.get("url"),
                    "headRefName": pr.get("headRefName"),
                    "failed_checks": failed,
                }
            )
    return failures


def _collect_pr_details(
    prs: list[dict[str, Any]],
    *,
    runner: Callable[..., Any],
    root: Path,
    limit: int = 20,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for pr in prs[:limit]:
        number = pr.get("number")
        if number is None:
            continue
        call = _call(
            runner,
            [
                "gh",
                "pr",
                "view",
                str(number),
                "--json",
                "number,title,url,files,commits,reviews,mergeStateStatus,statusCheckRollup",
            ],
            cwd=root,
            optional=True,
        )
        payload = _json_or_unknown(call, {})
        if not isinstance(payload, dict):
            details.append(
                {
                    "number": number,
                    "status": call["status"],
                    "error": call.get("error"),
                    "raw_type": type(payload).__name__,
                }
            )
            continue

        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        file_paths = [str(item.get("path") or item.get("filename") or "") for item in files if isinstance(item, dict)]
        file_paths = [path for path in file_paths if path]
        commits = payload.get("commits") if isinstance(payload.get("commits"), list) else []
        reviews = payload.get("reviews") if isinstance(payload.get("reviews"), list) else []
        status_checks = payload.get("statusCheckRollup") if isinstance(payload.get("statusCheckRollup"), list) else []
        details.append(
            {
                "number": payload.get("number", number),
                "title": payload.get("title"),
                "url": payload.get("url"),
                "status": call["status"],
                "error": call.get("error"),
                "mergeStateStatus": payload.get("mergeStateStatus"),
                "file_count": len(file_paths),
                "file_paths": file_paths[:MAX_PR_FILE_PATHS],
                "file_paths_truncated": len(file_paths) > MAX_PR_FILE_PATHS,
                "commit_count": len(commits),
                "review_states": sorted(
                    {str(review.get("state")) for review in reviews if isinstance(review, dict) and review.get("state")}
                ),
                "failed_checks": [
                    {
                        "name": check.get("name"),
                        "workflowName": check.get("workflowName"),
                        "conclusion": check.get("conclusion"),
                        "status": check.get("status"),
                    }
                    for check in status_checks
                    if isinstance(check, dict) and check.get("conclusion") == "FAILURE"
                ],
            }
        )
    return details


def _nearby_continue_on_error(lines: list[str], index: int) -> bool:
    window = lines[index : min(len(lines), index + 5)]
    return any(re.match(r"^\s*continue-on-error\s*:\s*true\s*$", line, re.IGNORECASE) for line in window)


def _workflow_security_summary(root: Path) -> dict[str, Any]:
    workflow_dir = root / ".github" / "workflows"
    files = sorted(path for path in workflow_dir.glob("*.y*ml") if path.is_file()) if workflow_dir.exists() else []
    entries: list[dict[str, Any]] = []
    advisory_steps: list[dict[str, Any]] = []
    broad_permissions: list[dict[str, Any]] = []
    has_top_level_permissions = False
    has_any_permissions = False
    uses_dependency_review_action = False
    uses_codeql_action = False
    uses_pull_request_target = False

    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            entries.append({"path": rel, "status": "unknown", "error": str(exc)})
            continue

        lines = text.splitlines()
        top_level_permissions = any(re.match(r"^permissions\s*:", line) for line in lines)
        any_permissions = any(re.match(r"^\s*permissions\s*:", line) for line in lines)
        lower = text.lower()
        dep_review = "actions/dependency-review-action" in lower
        codeql = "github/codeql-action" in lower
        pull_request_target = re.search(r"^\s*pull_request_target\s*:", text, re.MULTILINE) is not None

        has_top_level_permissions = has_top_level_permissions or top_level_permissions
        has_any_permissions = has_any_permissions or any_permissions
        uses_dependency_review_action = uses_dependency_review_action or dep_review
        uses_codeql_action = uses_codeql_action or codeql
        uses_pull_request_target = uses_pull_request_target or pull_request_target

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip().lower()
            if "pip-audit" in stripped:
                advisory_steps.append(
                    {
                        "path": rel,
                        "line": line_number,
                        "tool": "pip-audit",
                        "continue_on_error": _nearby_continue_on_error(lines, line_number - 1),
                    }
                )
            if re.match(r"^\s*permissions\s*:\s*(write-all|read-all)\s*$", line, re.IGNORECASE):
                broad_permissions.append(
                    {"path": rel, "line": line_number, "value": line.strip().split(":", 1)[1].strip()}
                )

        entries.append(
            {
                "path": rel,
                "status": "ok",
                "top_level_permissions": top_level_permissions,
                "any_permissions": any_permissions,
                "uses_dependency_review_action": dep_review,
                "uses_codeql_action": codeql,
                "uses_pull_request_target": pull_request_target,
            }
        )

    return {
        "workflow_count": len(files),
        "has_top_level_permissions": has_top_level_permissions,
        "has_any_permissions": has_any_permissions,
        "uses_dependency_review_action": uses_dependency_review_action,
        "uses_codeql_action": uses_codeql_action,
        "uses_pull_request_target": uses_pull_request_target,
        "advisory_security_steps": advisory_steps,
        "broad_permissions": broad_permissions,
        "entries": entries,
    }


def build_report(
    *,
    root: Path = PROJECT_ROOT,
    runner: Callable[..., Any] = _run,
    since: str | None = None,
    last_run_file: str | None = None,
    automation_memory: str | None = None,
) -> OrderedDict[str, Any]:
    root = Path(root)
    remote = _call(runner, ["git", "remote", "get-url", "origin"], cwd=root)
    repo_slug = _repo_slug(remote["stdout"] if remote["status"] == "ok" else "")
    status = _call(runner, ["git", "status", "--short", "--branch"], cwd=root)
    branch = _call(runner, ["git", "branch", "--show-current"], cwd=root)
    head = _call(runner, ["git", "rev-parse", "HEAD"], cwd=root)
    log_cmd = ["git", "log", "--oneline", f"--since={since}"] if since else ["git", "log", "--oneline", "-20"]
    recent = _call(runner, log_cmd, cwd=root)
    worktrees = _call(runner, ["git", "worktree", "list", "--porcelain"], cwd=root)
    stashes = _call(runner, ["git", "stash", "list", "--date=local"], cwd=root)

    pulse_call = _call(
        runner,
        [PYTHON_CMD, "scripts/tools/project_pulse.py", "--fast", "--format", "json"],
        cwd=root,
        timeout=120,
        optional=True,
    )
    pulse = _json_or_unknown(pulse_call, {})
    queue = _call(runner, [PYTHON_CMD, "scripts/tools/work_queue.py", "status"], cwd=root, optional=True)

    pr_call = _call(
        runner,
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "20",
            "--json",
            "number,title,headRefName,baseRefName,isDraft,updatedAt,statusCheckRollup,url",
        ],
        cwd=root,
        optional=True,
    )
    prs = _json_or_unknown(pr_call, [])
    pr_list = prs if isinstance(prs, list) else []
    pr_details = _collect_pr_details(pr_list, runner=runner, root=root)
    issue_call = _call(
        runner,
        ["gh", "issue", "list", "--state", "open", "--limit", "20", "--json", "number,title,labels,updatedAt,url"],
        cwd=root,
        optional=True,
    )
    issues = _json_or_unknown(issue_call, [])

    protection: dict[str, Any] = {"status": "unknown", "required_status_checks": None, "enforce_admins": None}
    security: dict[str, Any]
    if repo_slug:
        protection_call = _call(
            runner, ["gh", "api", f"repos/{repo_slug}/branches/main/protection"], cwd=root, optional=True
        )
        protection_payload = _json_or_unknown(protection_call, {})
        protection = {
            "status": protection_call["status"],
            "required_status_checks": (protection_payload.get("required_status_checks") or {}).get("contexts"),
            "enforce_admins": (protection_payload.get("enforce_admins") or {}).get("enabled"),
            "error": protection_call.get("error"),
        }
        security = {
            "code_scanning": _count_or_unknown(
                _call(
                    runner,
                    ["gh", "api", f"repos/{repo_slug}/code-scanning/alerts?state=open", "--jq", "length"],
                    cwd=root,
                    optional=True,
                ),
                endpoint="code-scanning/alerts?state=open",
            ),
            "dependabot": _count_or_unknown(
                _call(
                    runner,
                    ["gh", "api", f"repos/{repo_slug}/dependabot/alerts?state=open", "--jq", "length"],
                    cwd=root,
                    optional=True,
                ),
                endpoint="dependabot/alerts?state=open",
            ),
            "secret_scanning": _count_or_unknown(
                _call(
                    runner,
                    ["gh", "api", f"repos/{repo_slug}/secret-scanning/alerts?state=open", "--jq", "length"],
                    cwd=root,
                    optional=True,
                ),
                endpoint="secret-scanning/alerts?state=open",
            ),
        }
    else:
        security = {
            "code_scanning": {
                "status": "unknown",
                "endpoint": "code-scanning/alerts?state=open",
                "error": "repo slug unknown",
            },
            "dependabot": {
                "status": "unknown",
                "endpoint": "dependabot/alerts?state=open",
                "error": "repo slug unknown",
            },
            "secret_scanning": {
                "status": "unknown",
                "endpoint": "secret-scanning/alerts?state=open",
                "error": "repo slug unknown",
            },
        }

    status_lines = _lines(status["stdout"])
    dirty_lines = [line for line in status_lines[1:] if line.strip()]
    pulse_items = pulse.get("items") if isinstance(pulse, dict) else []
    if not isinstance(pulse_items, list):
        pulse_items = []

    report: OrderedDict[str, Any] = OrderedDict()
    report["generated_at"] = datetime.now(UTC).isoformat()
    report["repo"] = {
        "root": str(root),
        "slug": repo_slug,
        "branch_protection": protection,
    }
    report["git"] = {
        "branch": (branch["stdout"].strip() or "detached") if branch["status"] == "ok" else "unknown",
        "head": head["stdout"].strip() if head["status"] == "ok" else "unknown",
        "status": status_lines,
        "dirty_count": len(dirty_lines),
        "recent_commits": _lines(recent["stdout"], limit=20),
        "worktrees": _lines(worktrees["stdout"], limit=80),
        "stashes": _lines(stashes["stdout"], limit=20),
        "since": since,
    }
    report["prs"] = {
        "status": pr_call["status"],
        "open": pr_list,
        "details": pr_details,
        "error": pr_call.get("error"),
    }
    report["ci"] = {
        "open_pr_failures": _summarize_pr_failures(prs if isinstance(prs, list) else []),
        "required_checks": protection.get("required_status_checks"),
    }
    report["live_readiness"] = {
        "pulse_status": pulse_call["status"],
        "pulse_broken_count": (pulse.get("counts") or {}).get("broken") if isinstance(pulse, dict) else None,
        "survival_summary": pulse.get("survival_summary") if isinstance(pulse, dict) else None,
        "sr_summary": pulse.get("sr_summary") if isinstance(pulse, dict) else None,
        "high_severity_items": [
            item for item in pulse_items if item.get("severity") == "high" or item.get("category") == "broken"
        ][:10],
        "pulse_error": pulse_call.get("error"),
    }
    security["workflow_security"] = _workflow_security_summary(root)
    report["security"] = security
    report["workflow"] = {
        "work_queue_status": queue["stdout"].strip() if queue["status"] == "ok" else None,
        "work_queue_error": queue.get("error") if queue["status"] != "ok" else None,
        "active_stages": _active_stage_summary(root),
        "issues": issues if isinstance(issues, list) else [],
        "issues_status": issue_call["status"],
        "issues_error": issue_call.get("error"),
    }
    report["carryovers"] = {
        "last_run_file": _file_context(last_run_file),
        "automation_memory": _file_context(automation_memory),
        "stashes": report["git"]["stashes"],
        "dirty_files": dirty_lines,
    }
    report["recommended_attention_inputs"] = [
        "Required CI failures on open PRs",
        "Live-readiness broken/high pulse items",
        "Unknown or nonzero security scanners",
        "Workflow security guardrail gaps",
        "Dirty worktrees, active stages, and stale queue items",
        "Carryovers repeated from the previous weekly run",
    ]
    return report


def render_markdown(report: dict[str, Any]) -> str:
    failures = report["ci"].get("open_pr_failures") or []
    live_items = report["live_readiness"].get("high_severity_items") or []
    security = report["security"]
    active_stages = report["workflow"]["active_stages"]
    lines = [
        "# Weekly Project Improvement Audit Evidence",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Repo: `{report['repo'].get('slug') or 'unknown'}`",
        f"- Branch: `{report['git'].get('branch')}`",
        f"- Dirty files: `{report['git'].get('dirty_count')}`",
        "",
        "## CI And PRs",
    ]
    if failures:
        for item in failures[:MAX_MARKDOWN_ITEMS]:
            checks = ", ".join(check.get("name") or "unknown" for check in item.get("failed_checks", []))
            lines.append(
                f"- PR #{item.get('number')} checks={checks or 'unknown'} "
                f"head={item.get('headRefName') or 'unknown'} title={_compact_text(item.get('title'))}"
            )
        if len(failures) > MAX_MARKDOWN_ITEMS:
            lines.append(f"- ... {len(failures) - MAX_MARKDOWN_ITEMS} more failing PR(s) in JSON")
    else:
        lines.append("- No failing open-PR checks found in the evidence packet.")
    lines.extend(["", "## Live Readiness"])
    if live_items:
        for item in live_items[:MAX_MARKDOWN_ITEMS]:
            lines.append(
                f"- {item.get('source', 'pulse')} severity={item.get('severity', 'unknown')} "
                f"category={item.get('category', 'unknown')} summary={_compact_text(item.get('summary'))}"
            )
        if len(live_items) > MAX_MARKDOWN_ITEMS:
            lines.append(f"- ... {len(live_items) - MAX_MARKDOWN_ITEMS} more live item(s) in JSON")
    else:
        lines.append("- No broken/high live-readiness pulse items found.")
    lines.extend(["", "## Security"])
    for key in ("code_scanning", "dependabot", "secret_scanning"):
        item = security.get(key, {})
        if item.get("status") == "ok":
            lines.append(f"- {key}: {item.get('open_alert_count')} open alert(s)")
        else:
            lines.append(
                f"- {key}: unknown endpoint={item.get('endpoint', 'unknown')} error={_compact_text(item.get('error'))}"
            )
    workflow_security = security.get("workflow_security", {})
    lines.append(
        "- workflow_security: "
        f"{workflow_security.get('workflow_count', 0)} workflow(s), "
        f"top-level permissions={workflow_security.get('has_top_level_permissions')}, "
        f"dependency-review={workflow_security.get('uses_dependency_review_action')}, "
        f"codeql={workflow_security.get('uses_codeql_action')}"
    )
    lines.extend(["", "## Workflow And Carryovers"])
    lines.append(f"- Active stage files: {active_stages.get('active_count')} / {active_stages.get('count')}")
    lines.append(f"- Stashes: {len(report['carryovers'].get('stashes') or [])}")
    lines.append(f"- Open issues: {len(report['workflow'].get('issues') or [])}")
    lines.extend(["", "## Recommended Attention Inputs"])
    for item in report.get("recommended_attention_inputs", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--since", help="Limit git recent-commit evidence to this git --since value.")
    parser.add_argument("--last-run-file", help="Optional prior weekly run file to summarize.")
    parser.add_argument("--automation-memory", help="Optional Codex automation memory file to summarize.")
    parser.add_argument("--out", help="Optional output path. Without this, writes to stdout only.")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Any] = _run,
    root: Path = PROJECT_ROOT,
) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        root=root,
        runner=runner,
        since=args.since,
        last_run_file=args.last_run_file,
        automation_memory=args.automation_memory,
    )
    if args.format == "json":
        output = json.dumps(report, indent=2, sort_keys=False) + "\n"
    else:
        output = render_markdown(report)
    if args.out:
        path = Path(args.out)
        path.write_text(output, encoding="utf-8")
        print(str(path))
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
