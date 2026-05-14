"""Tests for session-start.py legacy stage-line fallback."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HOOK_PATH = Path(__file__).resolve().parents[1] / "session-start.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("session_start", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["session_start"] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_legacy_startup_lines_only_include_active_stage_files(tmp_path: Path, monkeypatch) -> None:
    hook = _load_hook()
    stages_dir = tmp_path / "docs" / "runtime" / "stages"
    _write(stages_dir / "active.md", "---\ntask: Active work\nmode: IMPLEMENTATION\n---\n")
    _write(stages_dir / "closed.md", "---\ntask: Closed work\nmode: IMPLEMENTATION\nstatus: closed\n---\n")
    _write(stages_dir / "loose.md", "task: Loose template\nmode: IMPLEMENTATION\n")
    monkeypatch.setattr(hook, "PROJECT_ROOT", tmp_path)

    lines = hook._legacy_startup_lines()
    text = "\n".join(lines)

    assert "Active stage [active]: task: Active work" in text
    assert "Closed work" not in text
    assert "Loose template" not in text
