"""Tests for stage-awareness.py active-stage filtering."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

HOOK_PATH = Path(__file__).resolve().parents[1] / "stage-awareness.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("stage_awareness", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage_awareness"] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _StubStdin:
    def __init__(self, payload: str):
        self._payload = payload

    def read(self, *_args, **_kwargs) -> str:
        return self._payload


def test_stage_awareness_ignores_closed_and_loose_stage_files(tmp_path: Path, capsys, monkeypatch) -> None:
    hook = _load_hook()
    stages_dir = tmp_path / "docs" / "runtime" / "stages"
    _write(
        stages_dir / "active.md",
        "\n".join(
            [
                "---",
                "task: Active work",
                "mode: IMPLEMENTATION",
                "scope_lock:",
                "  - pipeline/system_context.py",
                "---",
                "## Blast Radius",
                "Touches one startup reporting path and its focused tests.",
            ]
        ),
    )
    _write(
        stages_dir / "closed.md",
        "---\ntask: Closed work\nmode: IMPLEMENTATION\nstatus: closed\n---\n",
    )
    _write(stages_dir / "loose.md", "task: Loose template\nmode: IMPLEMENTATION\n")

    monkeypatch.setattr(hook, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hook, "STAGE_STATE", tmp_path / "docs" / "runtime" / "STAGE_STATE.md")
    monkeypatch.setattr(hook, "STAGES_DIR", stages_dir)

    event = {"hook_event_name": "UserPromptSubmit", "prompt": "continue"}
    with pytest.raises(SystemExit) as exc:
        with patch.object(hook.sys, "stdin", _StubStdin(json.dumps(event))):
            hook.main()

    assert exc.value.code == 0
    stderr = capsys.readouterr().err
    assert "active:IMPLEMENTATION Active work" in stderr
    assert "Closed work" not in stderr
    assert "Loose template" not in stderr
