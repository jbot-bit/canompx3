#!/usr/bin/env python3
"""Emit a stable bootstrap-health proof artifact for repo session startup."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from pipeline.system_context import list_claims  # noqa: E402


def _run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _expected_interpreter(root: Path, context: str) -> Path:
    if context == "codex-wsl":
        return root / ".venv-wsl" / "bin" / "python"
    if context in {"claude-windows", "windows"} or os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _git_state(root: Path) -> dict[str, Any]:
    status = _run_git(root, "status", "--short")
    return {
        "branch": _run_git(root, "branch", "--show-current") or "unknown",
        "head": _run_git(root, "rev-parse", "--short", "HEAD") or "unknown",
        "dirty_files": [line for line in (status or "").splitlines() if line.strip()],
    }


def _database_state(db_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "size_bytes": None,
        "mtime_utc": None,
    }
    if not db_path.exists():
        return state
    stat = db_path.stat()
    state["size_bytes"] = stat.st_size
    state["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    return state


def _same_path(left: str | Path, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _active_mutating_claims(root: Path) -> list[Any]:
    claims = []
    for claim in list_claims(fresh_only=True):
        if getattr(claim, "mode", "") != "mutating":
            continue
        claim_root = getattr(claim, "root", "")
        if claim_root and not _same_path(claim_root, root):
            continue
        claims.append(claim)
    return claims


def _claim_payload(claim: Any) -> dict[str, Any]:
    return {
        "tool": getattr(claim, "tool", None),
        "branch": getattr(claim, "branch", None),
        "head_sha": getattr(claim, "head_sha", None),
        "pid": getattr(claim, "pid", None),
        "mode": getattr(claim, "mode", None),
        "root": getattr(claim, "root", None),
        "runtime": getattr(claim, "runtime", None),
        "fresh": getattr(claim, "fresh", None),
    }


def _pulse_state(pulse_payload: dict[str, Any] | None) -> dict[str, Any]:
    if pulse_payload is None:
        return {
            "available": False,
            "broken_count": None,
            "startup_blockers": [],
            "source": "not collected",
        }
    items = pulse_payload.get("items", [])
    startup_blockers = [
        item
        for item in items
        if item.get("category") == "broken"
        and any(token in str(item.get("summary", "")).lower() for token in ("startup", "preflight", "bootstrap"))
    ]
    return {
        "available": True,
        "broken_count": int((pulse_payload.get("counts") or {}).get("broken") or 0),
        "startup_blockers": startup_blockers,
        "source": "project_pulse --fast --format json",
    }


def _collect_project_pulse(root: Path) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [sys.executable, "scripts/tools/project_pulse.py", "--fast", "--format", "json", "--root", str(root)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _blocker(code: str, message: str, detail: str | None = None) -> dict[str, str]:
    payload = {"code": code, "message": message}
    if detail:
        payload["detail"] = detail
    return payload


def _next_command(blockers: list[dict[str, str]], expected_python: Path, db_path: Path) -> str:
    codes = {blocker["code"] for blocker in blockers}
    if "missing_expected_interpreter" in codes:
        return f"uv sync --frozen --python {expected_python}"
    if "missing_canonical_db" in codes:
        return f"python scripts/tools/project_pulse.py --fast --format json --root {db_path.parent}"
    if "mutating_claim_collision" in codes:
        return "python scripts/tools/session_preflight.py --context codex-wsl --mode read-only"
    if "dirty_tree" in codes:
        return "git status --short"
    if "pulse_broken" in codes:
        return "python scripts/tools/project_pulse.py --fast --format json"
    return "python scripts/tools/session_preflight.py --context codex-wsl"


def build_bootstrap_health_proof(
    *,
    root: Path = PROJECT_ROOT,
    context: str = "codex-wsl",
    db_path: Path = GOLD_DB_PATH,
    pulse_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    expected_python = _expected_interpreter(root, context)
    git = _git_state(root)
    database = _database_state(db_path)
    claims = [_claim_payload(claim) for claim in _active_mutating_claims(root)]
    pulse = _pulse_state(pulse_payload)

    blockers: list[dict[str, str]] = []
    if not expected_python.exists():
        blockers.append(
            _blocker("missing_expected_interpreter", "Expected interpreter is missing", str(expected_python))
        )
    if not database["exists"]:
        blockers.append(_blocker("missing_canonical_db", "Canonical DB is missing", str(db_path)))
    if git["dirty_files"]:
        blockers.append(_blocker("dirty_tree", "Git tree has uncommitted changes", str(len(git["dirty_files"]))))
    if claims:
        blockers.append(_blocker("mutating_claim_collision", "Fresh mutating session claim exists", str(len(claims))))
    if pulse["broken_count"]:
        blockers.append(_blocker("pulse_broken", "Project pulse reports broken items", str(pulse["broken_count"])))

    proof = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(root),
        "context": context,
        "expected_interpreter": str(expected_python),
        "expected_interpreter_exists": expected_python.exists(),
        "actual_interpreter": str(Path(sys.executable).resolve()),
        "git": {
            **git,
            "dirty": bool(git["dirty_files"]),
        },
        "database": database,
        "mutating_session_claims": {
            "active": bool(claims),
            "claims": claims,
        },
        "pulse": pulse,
        "blockers": blockers,
        "warnings": [],
        "next_command": "",
    }
    proof["next_command"] = _next_command(blockers, expected_python, db_path)
    return proof


def render_markdown(proof: dict[str, Any]) -> str:
    lines = [
        "# Bootstrap Health Proof",
        "",
        f"- Generated: `{proof['generated_at']}`",
        f"- Repo root: `{proof['repo_root']}`",
        f"- Context: `{proof['context']}`",
        f"- Expected interpreter: `{proof['expected_interpreter']}` exists=`{proof['expected_interpreter_exists']}`",
        f"- Actual interpreter: `{proof['actual_interpreter']}`",
        f"- Git: `{proof['git']['branch']}` `{proof['git']['head']}` dirty=`{proof['git']['dirty']}`",
        f"- Canonical DB: `{proof['database']['path']}` exists=`{proof['database']['exists']}`",
        f"- Mutating claim active: `{proof['mutating_session_claims']['active']}`",
        f"- Pulse broken count: `{proof['pulse']['broken_count']}`",
        f"- Next command: `{proof['next_command']}`",
        "",
        "## Blockers",
        "",
    ]
    if proof["blockers"]:
        for blocker in proof["blockers"]:
            lines.append(f"- `{blocker['code']}`: {blocker['message']}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_bootstrap_health_artifacts(proof: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bootstrap_health_proof.json"
    markdown_path = output_dir / "bootstrap_health_proof.md"
    json_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(proof), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit bootstrap health proof artifacts.")
    parser.add_argument("--context", default="codex-wsl")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH))
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-pulse", action="store_true")
    parser.add_argument("--fail-on-blocker", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(args.root).resolve()
    pulse = None if args.skip_pulse else _collect_project_pulse(root)
    proof = build_bootstrap_health_proof(
        root=root,
        context=args.context,
        db_path=Path(args.db_path),
        pulse_payload=pulse,
    )
    if args.output_dir:
        write_bootstrap_health_artifacts(proof, Path(args.output_dir))
    if args.format == "markdown":
        print(render_markdown(proof), end="")
    else:
        print(json.dumps(proof, indent=2, sort_keys=True))
    if args.fail_on_blocker and proof["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
