#!/usr/bin/env python3
"""Render the canonical system authority map from the code-backed registry."""

from __future__ import annotations

from pathlib import Path

from pipeline.system_authority import SYSTEM_AUTHORITY_MAP_RELATIVE_PATH, render_system_authority_map


def _writable_project_root() -> Path:
    """Resolve repo root without tripping WSL's /mnt/c/Users read-only alias."""
    resolved = Path(__file__).resolve().parents[2]
    text = str(resolved)
    if text.startswith("/mnt/c/Users/"):
        candidate = Path(text.replace("/mnt/c/Users/", "/mnt/c/users/", 1))
        if candidate.exists():
            return candidate
    return resolved


def main() -> None:
    project_root = _writable_project_root()
    out_path = project_root / SYSTEM_AUTHORITY_MAP_RELATIVE_PATH
    out_path.write_text(render_system_authority_map(), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
