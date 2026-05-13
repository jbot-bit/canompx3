#!/usr/bin/env python3
"""Completion notification hook: plays a sound when Claude finishes a task or needs input.

Fires on: Notification, Stop events.
Windows: uses winsound beep. Cross-platform fallback: terminal bell.

On Stop, additionally injects a pre-commit cue via `additionalContext` IFF all
three conditions hold:
  1. Git working tree has uncommitted edits under `pipeline/` or `trading_app/`.
  2. Any active stage in `docs/runtime/stages/` declares `mode: IMPLEMENTATION`.
  3. Cooldown for the cue (30 min) has elapsed.

The cue tells the operator to run `/verify done` and `/code-review` before
committing. Fail-open at every step: any error → silent exit 0.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUE_STATE_FILE = PROJECT_ROOT / ".claude" / "hooks" / "state" / "completion-cue.json"
CUE_COOLDOWN_S = 1800  # 30 min
PROD_PATH_PREFIXES = ("pipeline/", "trading_app/")
CUE_TEXT = (
    "Pre-commit cue → run /verify done and /code-review "
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
    """Return uncommitted-edit file paths (staged + unstaged). Empty on failure."""
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if r.returncode != 0:
            return []
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def _has_prod_edits() -> bool:
    for name in _git_diff_names():
        norm = name.replace("\\", "/")
        for prefix in PROD_PATH_PREFIXES:
            if norm.startswith(prefix):
                return True
    return False


def _has_implementation_stage() -> bool:
    """Return True if any docs/runtime/stages/*.md or STAGE_STATE.md declares
    `mode: IMPLEMENTATION`."""
    try:
        candidates: list[Path] = []
        stages_dir = PROJECT_ROOT / "docs" / "runtime" / "stages"
        if stages_dir.is_dir():
            candidates.extend(p for p in stages_dir.glob("*.md") if p.name != ".gitkeep")
        legacy = PROJECT_ROOT / "docs" / "runtime" / "STAGE_STATE.md"
        if legacy.exists():
            candidates.append(legacy)
        for f in candidates:
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line in text.splitlines():
                if line.strip().lower().startswith("mode:"):
                    if "IMPLEMENTATION" in line.upper():
                        return True
                    break  # first mode line per file is authoritative
        return False
    except BaseException:  # pragma: no cover - fail-open
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
    """Emit the pre-commit cue as Stop-hook JSON when all conditions hold.

    Stop hooks emit additionalContext via the same envelope shape as
    UserPromptSubmit (per Claude Code 2026 hook spec).
    """
    state = _load_cue_state()
    if _is_cue_cooling(state):
        return
    if not _has_prod_edits():
        return
    if not _has_implementation_stage():
        return
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": CUE_TEXT,
        }
    }
    print(json.dumps(payload))
    state["last_at"] = datetime.now(UTC).isoformat()
    _save_cue_state(state)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    hook_event = event.get("hook_event_name", "")
    if hook_event not in ("Notification", "Stop"):
        sys.exit(0)

    _play_sound()

    if hook_event == "Stop":
        try:
            _maybe_emit_cue()
        except BaseException:  # pragma: no cover - fail-open
            pass

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # pragma: no cover - fail-open
        sys.exit(0)
