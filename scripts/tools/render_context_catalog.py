#!/usr/bin/env python3
"""Render generated docs for deterministic task-context routing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _writable_project_root() -> Path:
    """Resolve repo root without tripping WSL's /mnt/c/Users read-only alias."""
    resolved = Path(__file__).resolve().parents[2]
    text = str(resolved)
    if text.startswith("/mnt/c/Users/"):
        candidate = Path(text.replace("/mnt/c/Users/", "/mnt/c/users/", 1))
        if candidate.exists():
            return candidate
    return resolved


PROJECT_ROOT = _writable_project_root()


def _preferred_repo_python() -> Path | None:
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv-wsl" / "bin" / "python"
    return candidate if candidate.exists() else None


def _preferred_repo_prefix(expected_python: Path) -> Path:
    return expected_python.parent.parent.resolve()


def _ensure_repo_python() -> None:
    expected_python = _preferred_repo_python()
    if expected_python is None:
        return
    current_prefix = Path(sys.prefix).resolve()
    expected_prefix = _preferred_repo_prefix(expected_python)
    if current_prefix == expected_prefix or os.environ.get("CANOMPX3_BOOTSTRAP_DONE") == "1":
        return

    env = os.environ.copy()
    env["CANOMPX3_BOOTSTRAP_DONE"] = "1"
    env.setdefault("CANOMPX3_BOOTSTRAPPED_FROM", str(Path(sys.executable).resolve()))
    raise SystemExit(
        subprocess.call(
            [str(expected_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    )


_ensure_repo_python()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from context.registry import (  # noqa: E402
    render_institutional_markdown,
    render_readme_markdown,
    render_source_catalog_markdown,
    render_task_routes_markdown,
)


def main() -> None:
    out_dir = PROJECT_ROOT / "docs" / "context"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        out_dir / "README.md": render_readme_markdown(),
        out_dir / "source-catalog.md": render_source_catalog_markdown(),
        out_dir / "task-routes.md": render_task_routes_markdown(),
        out_dir / "institutional-contracts.md": render_institutional_markdown(),
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
