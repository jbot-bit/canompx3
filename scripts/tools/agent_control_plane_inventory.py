#!/usr/bin/env python3
"""Read-only agent-control-plane inventory for canompx3 workstreams.

This exports managed worktree state in a shape that an external organizer such
as Paperclip can ingest without learning repo internals or scanning mutable
paths itself. It works from either the main checkout or any linked worktree by
resolving the shared git common directory first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKTREE_META = ".canompx3-worktree.json"


@dataclass(frozen=True)
class ActiveWorktree:
    path: Path
    head: str | None
    branch: str | None


@dataclass(frozen=True)
class ControlPlaneWorkstream:
    tool: str
    name: str
    path: str
    branch: str
    head: str | None
    state: str
    purpose: str | None
    dirty: bool
    last_opened_at: str | None
    handoff_note: str | None
    recommended_action: str
    write_policy: Literal["isolated_worktree_only"]
    truth_policy: Literal["repo_canonical_layers_only"]


def _git_output(args: list[str], *, cwd: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_common_root(cwd: Path = PROJECT_ROOT) -> Path:
    common_dir = _git_output(["rev-parse", "--git-common-dir"], cwd=cwd)
    if not common_dir:
        return cwd
    path = Path(common_dir)
    if not path.is_absolute():
        path = (cwd / path).resolve()
    if path.name == ".git":
        return path.parent
    return path


def _parse_worktree_list(output: str) -> list[ActiveWorktree]:
    items: list[ActiveWorktree] = []
    current: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                items.append(_active_from_raw(current))
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    if current:
        items.append(_active_from_raw(current))
    return items


def _active_from_raw(raw: dict[str, str]) -> ActiveWorktree:
    return ActiveWorktree(
        path=Path(raw["worktree"]).resolve(),
        head=raw.get("HEAD"),
        branch=(raw.get("branch") or "").removeprefix("refs/heads/") or None,
    )


def _active_worktrees(control_root: Path) -> dict[Path, ActiveWorktree]:
    output = _git_output(["worktree", "list", "--porcelain"], cwd=control_root)
    return {item.path: item for item in _parse_worktree_list(output)}


def _read_metadata(path: Path) -> dict[str, Any] | None:
    meta_path = path / WORKTREE_META
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _managed_metadata(control_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    root = control_root / ".worktrees"
    if not root.exists():
        return []
    items: list[tuple[Path, dict[str, Any]]] = []
    for meta_path in sorted(root.rglob(WORKTREE_META)):
        path = meta_path.parent.resolve()
        meta = _read_metadata(path)
        if meta is not None:
            items.append((path, meta))
    return items


def _worktree_status(path: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return [result.stderr.strip() or "unable to read status"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def _recommended_action(item: Any) -> str:
    if item.dirty:
        return "inspect_dirty_worktree_before_assignment"
    if item.state == "handoff":
        return "claim_or_close_handoff"
    if item.branch and item.branch.startswith("wt-"):
        return "review_for_pr_or_close"
    return "monitor"


def build_inventory() -> dict[str, object]:
    control_root = _git_common_root()
    active = _active_worktrees(control_root)
    workstreams: list[ControlPlaneWorkstream] = []
    for path, meta in _managed_metadata(control_root):
        active_info = active.get(path)
        if active_info is None:
            continue
        dirty = bool(_worktree_status(path))
        branch = str(meta.get("branch") or active_info.branch or "")
        state = str(meta.get("state") or "active")
        item = SimpleNamespace(dirty=dirty, state=state, branch=branch)
        workstreams.append(
            ControlPlaneWorkstream(
                tool=str(meta.get("tool") or "unknown"),
                name=str(meta.get("name") or path.name),
                path=str(path),
                branch=branch,
                head=active_info.head,
                state=state,
                purpose=meta.get("purpose"),
                dirty=dirty,
                last_opened_at=meta.get("last_opened_at"),
                handoff_note=meta.get("handoff_note"),
                recommended_action=_recommended_action(item),
                write_policy="isolated_worktree_only",
                truth_policy="repo_canonical_layers_only",
            )
        )

    return {
        "schema": "canompx3.agent_control_plane_inventory.v1",
        "repo": str(control_root),
        "main_branch": _git_output(["branch", "--show-current"], cwd=control_root),
        "origin_main_head": _git_output(["rev-parse", "origin/main"], cwd=control_root),
        "workstreams": [asdict(item) for item in workstreams],
        "global_policies": {
            "no_llm_trade_scoring": True,
            "no_ai_promotion_writes": True,
            "control_plane_role": "orchestration_only",
            "lona_role": "advisory_external_sandbox",
        },
    }


def render_markdown(payload: dict[str, object]) -> str:
    workstreams = payload["workstreams"]
    assert isinstance(workstreams, list)
    lines = [
        "# Agent Control Plane Inventory",
        "",
        f"- Repo: `{payload['repo']}`",
        f"- Main branch: `{payload['main_branch']}`",
        f"- Origin main: `{payload['origin_main_head']}`",
        "",
        "| Tool | Name | Branch | State | Dirty | Recommended action |",
        "|---|---|---|---|---:|---|",
    ]
    for raw in workstreams:
        assert isinstance(raw, dict)
        lines.append(
            "| {tool} | {name} | `{branch}` | {state} | {dirty} | {action} |".format(
                tool=raw["tool"],
                name=raw["name"],
                branch=raw["branch"],
                state=raw["state"],
                dirty=str(raw["dirty"]).lower(),
                action=raw["recommended_action"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export canompx3 workstream inventory for external control planes.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()

    payload = build_inventory()
    if args.format == "markdown":
        print(render_markdown(payload))
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
