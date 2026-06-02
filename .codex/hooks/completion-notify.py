#!/usr/bin/env python3
"""Completion notification hook for Codex.

Fires on Notification and Stop events. On Stop, it emits a top-level
`systemMessage` cue when production paths are dirty, an implementation stage is
active, and the cue cooldown has elapsed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUE_STATE_FILE = PROJECT_ROOT / ".codex" / "hooks" / "state" / "completion-cue.json"
CUE_COOLDOWN_S = 1800
PROD_PATH_PREFIXES = ("pipeline/", "trading_app/")
CUE_TEXT = (
    "Pre-commit cue -> run /verify done and /code-review "
    "(production paths edited, stage=IMPLEMENTATION)"
)


def _play_sound() -> None:
    try:
        import winsound

        winsound.Beep(800, 200)
        winsound.Beep(1000, 200)
    except (ImportError, RuntimeError):
        print("\a", end="", file=sys.stderr)


def _git_diff_names() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _has_prod_edits() -> bool:
    for name in _git_diff_names():
        normalized = name.replace("\\", "/")
        for prefix in PROD_PATH_PREFIXES:
            if normalized.startswith(prefix):
                return True
    return False


def _has_implementation_stage() -> bool:
    try:
        candidates: list[Path] = []
        stages_dir = PROJECT_ROOT / "docs" / "runtime" / "stages"
        if stages_dir.is_dir():
            candidates.extend(p for p in stages_dir.glob("*.md") if p.name != ".gitkeep")
        legacy = PROJECT_ROOT / "docs" / "runtime" / "STAGE_STATE.md"
        if legacy.exists():
            candidates.append(legacy)
        for path in candidates:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line in text.splitlines():
                stripped = line.strip().lower()
                if stripped.startswith("mode:"):
                    if "implementation" in stripped:
                        return True
                    break
        return False
    except BaseException:
        return False


def _load_cue_state() -> dict:
    try:
        return json.loads(CUE_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return {}


def _save_cue_state(state: dict) -> None:
    try:
        CUE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CUE_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def _is_cue_cooling(state: dict) -> bool:
    last = state.get("last_at")
    if not last:
        return False
    try:
        age = (datetime.now(UTC) - datetime.fromisoformat(last)).total_seconds()
    except (TypeError, ValueError):
        return False
    return age < CUE_COOLDOWN_S


def _maybe_emit_cue() -> None:
    """Emit Stop-hook cue text using the real runtime contract."""
    state = _load_cue_state()
    if _is_cue_cooling(state):
        return
    if not _has_prod_edits():
        return
    if not _has_implementation_stage():
        return
    print(json.dumps({"systemMessage": CUE_TEXT}))
    state["last_at"] = datetime.now(UTC).isoformat()
    _save_cue_state(state)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        raise SystemExit(0) from None

    hook_event = event.get("hook_event_name", "")
    if hook_event not in ("Notification", "Stop"):
        raise SystemExit(0)

    _play_sound()
    if hook_event == "Stop":
        try:
            _maybe_emit_cue()
        except BaseException:
            pass

    raise SystemExit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        raise SystemExit(0) from None
