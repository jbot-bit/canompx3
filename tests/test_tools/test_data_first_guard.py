from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stderr
from datetime import UTC, datetime
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / ".claude" / "hooks" / "data-first-guard.py"
    spec = importlib.util.spec_from_file_location("data_first_guard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules.pop("data_first_guard", None)
    spec.loader.exec_module(module)
    return module


def test_repeated_prompt_directives_are_suppressed_within_cooldown(tmp_path: Path) -> None:
    module = _load_module()
    module.STATE_FILE = tmp_path / ".data-first-state.json"
    state = {
        "investigation_mode": False,
        "consecutive_reads": 0,
        "last_updated": datetime.now(UTC).isoformat(),
        "last_prompt_directive_key": module._directive_key([module.ORIENT_DIRECTIVE]),
        "last_prompt_directive_at": datetime.now(UTC).isoformat(),
    }
    module.save_state(state)

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            module.handle_user_prompt({"prompt": "what's the status"})
    except SystemExit as exc:
        assert exc.code == 0

    assert stderr.getvalue() == ""


def test_implementation_prompt_clears_stale_investigation_mode(tmp_path: Path) -> None:
    module = _load_module()
    module.STATE_FILE = tmp_path / ".data-first-state.json"
    module.save_state(
        {
            "investigation_mode": True,
            "consecutive_reads": 5,
            "last_updated": datetime.now(UTC).isoformat(),
            "last_prompt_directive_key": None,
            "last_prompt_directive_at": None,
        }
    )

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            module.handle_user_prompt({"prompt": "build it now"})
    except SystemExit as exc:
        assert exc.code == 0

    state = module.load_state()
    assert state["investigation_mode"] is False
    assert state["consecutive_reads"] == 0
    assert "IMPLEMENT MODE" in stderr.getvalue()


def test_investigation_prompt_sets_mode_and_emits_once(tmp_path: Path) -> None:
    module = _load_module()
    module.STATE_FILE = tmp_path / ".data-first-state.json"

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            module.handle_user_prompt({"prompt": "investigate why the numbers are off"})
    except SystemExit as exc:
        assert exc.code == 0

    state = module.load_state()
    assert state["investigation_mode"] is True
    assert state["consecutive_reads"] == 0
    assert "DATA FIRST" in stderr.getvalue()
